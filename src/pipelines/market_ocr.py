"""Market OCR ingestion pipeline with team bundling and confidence scoring.

Responsibilities:
- Accept screenshot(s) or images
- Run OCR via `src.ocr.ocr` helpers
- Group parsed lines into team bundles (home/away with ordered markets: ML, spread, total)
- Score confidence per line using heuristics (odds validity, keyword matching, line position)
- Flag gaps (missing market types) for manual review
- Write staging rows via `src.data.betting_repository`
- Emit bundled JSON when `--json-output` is set
- Perform fuzzy team matching via `src.utils.identity`

JSON Schema (when json_output is set):
[
  {
    "image": str,
    "captured_at": str,
    "book": str,
    "team_home_raw": str,
    "team_away_raw": str,
    "markets": [
      {
        "market_type": "ML"|"spread"|"total",
        "selection": str,
        "odds": int,
        "line": float|null,
        "raw": str,
        "font_row_index": int,  # position in OCR output (0-based)
        "odds_confidence": float,  # 0.0-1.0 based on odds validity + keyword match
        "keywords": [str, ...],  # evidence strings (e.g., ["Lakers"], ["over", "under"])
        "gap_flags": [str, ...]  # at bundle level: missing market types if <3 markets
      }
    ],
    "gap_flags": [str, ...]  # market types missing from this team bundle
  }
]

Gap flags are populated when a team bundle has fewer than 3 ordered markets (ML, spread, total).
Confidence scoring uses font row position, keyword heuristics, and field validity.

Timestamping: when the `captured_at` parameter is omitted, the pipeline records
a per-image ISO timestamp at the moment each image is read. This timestamp is
used for JSON output and when persisting staging rows to the DB.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import re
from pathlib import Path
import json

import src.ocr.ocr as ocr_module
from src.data.betting_repository import (
    _clean_odds_field,
    _clean_line_field,
    add_staging_row,
    resolve_staging_to_game,
    _utcnow_iso,
)


TEAM_SEP_RE = re.compile(r"(?:\b(?:vs\.?|v\.|v)\b|@)", flags=re.IGNORECASE)
# Odds regex: match +/- followed by 2-4 digits (typical odds like +110, -110, +200, etc.)
# Use negative lookahead to avoid matching single digit odds followed by a decimal point (which would be a spread)
ODDS_RE = re.compile(r"(?P<odds>[+-](?:1\d{2}|[2-9]\d{1,2})(?!\.\d))")
LINE_RE = re.compile(r"(?P<line>-?\d+(?:\.\d+)?)")

def _normalize_text(s: str) -> str:
    # Normalize dash characters and trim, but avoid aggressive letter->digit substitutions
    return s.replace("−", "-").replace("–", "-").strip()


def _compute_line_confidence(parsed: dict, raw_text: str, font_row_index: int) -> dict:
    """Compute confidence scores and keyword evidence for a parsed market line.
    
    Returns dict with keys: odds_confidence, market_type_confidence, keywords, font_row_index
    """
    odds = parsed.get("odds")
    market_type = parsed.get("market_type", "ML")
    raw = raw_text.lower()
    keywords = []
    
    # Odds confidence: penalty for missing or out-of-range odds
    odds_confidence = 0.0
    if odds is not None and -1000 <= odds <= 1000 and odds != 0:
        odds_confidence = 0.8  # base score for valid odds
        # Boost if odds look reasonable (typical range -200 to +200)
        if -200 <= odds <= 200:
            odds_confidence = 1.0
    else:
        odds_confidence = 0.3  # fallback if odds missing or invalid
    
    # Market-type-specific confidence
    market_confidence = 0.7  # base for ML (default)
    if market_type == "total":
        if "over" in raw or "under" in raw or "o/u" in raw or "total" in raw:
            market_confidence = 1.0
            keywords.extend(["over", "under"] if "over" in raw or "under" in raw else ["o/u"])
        else:
            market_confidence = 0.5
    elif market_type == "spread":
        if "spread" in raw or "line" in raw or "+" in raw or "-" in raw:
            market_confidence = 1.0
            keywords.append("spread")
        else:
            market_confidence = 0.7
        if parsed.get("line") is not None:
            market_confidence = 1.0
            keywords.append(f"line:{parsed.get('line')}")
    
    # Add selection as keyword evidence
    selection = parsed.get("selection", "")
    if selection:
        keywords.append(selection)
    
    # Font row index penalty: lines near the top (headers) get lower weight
    row_penalty = 0.0
    if font_row_index < 2:
        row_penalty = -0.15  # likely header lines
    
    combined_confidence = min(1.0, max(0.0, (odds_confidence + market_confidence) / 2.0 + row_penalty))
    
    return {
        "odds_confidence": round(odds_confidence, 2),
        "market_confidence": round(market_confidence, 2),
        "combined_confidence": round(combined_confidence, 2),
        "keywords": list(set(keywords)),  # deduplicate
        "font_row_index": font_row_index,
    }


def _bundle_team_markets(
    parsed_lines: List[Tuple[dict, int]],  # list of (parsed_dict, row_index) tuples
    team_home: str,
    team_away: str,
) -> dict:
    """Group parsed lines for a team matchup into an ordered bundle.
    
    Returns dict with structure:
    {
      "team_home_raw": str,
      "team_away_raw": str,
      "markets": [market1, market2, market3],  # ordered by type: ML, spread, total
      "gap_flags": ["spread"] | []  # missing market types
    }
    """
    markets_by_type = {}
    
    for parsed, row_idx in parsed_lines:
        market_type = parsed.get("market_type", "ML")
        if market_type not in markets_by_type:
            # Compute confidence for this line
            confidence_data = _compute_line_confidence(parsed, parsed.get("raw", ""), row_idx)
            markets_by_type[market_type] = {
                "market_type": market_type,
                "selection": parsed.get("selection"),
                "odds": parsed.get("odds"),
                "line": parsed.get("line"),
                "raw": parsed.get("raw"),
                "font_row_index": confidence_data["font_row_index"],
                "odds_confidence": confidence_data["odds_confidence"],
                "keywords": confidence_data["keywords"],
            }
    
    # Order markets: ML, spread, total
    market_order = ["ML", "spread", "total"]
    markets = []
    for mt in market_order:
        if mt in markets_by_type:
            markets.append(markets_by_type[mt])
    
    # Flag missing market types
    gap_flags = []
    for mt in market_order:
        if mt not in markets_by_type:
            gap_flags.append(mt)
    
    return {
        "team_home_raw": team_home,
        "team_away_raw": team_away,
        "markets": markets,
        "gap_flags": gap_flags,
    }



def parse_market_line(line: str) -> dict | None:
    """Attempt to parse a single OCR text line into market fields.

    Returns dict with keys: selection, odds, line, market_type (ML/spread/total), raw
    or None if parsing failed.
    """
    raw = line.strip()
    if not raw:
        return None
    
    # Check for total keywords BEFORE normalizing (which replaces 'o' with '0')
    raw_lower = raw.lower()
    has_over_under = "over" in raw_lower or "under" in raw_lower or re.search(r"o/u|o\.|u\.", raw, flags=re.I)
    
    s = _normalize_text(raw)

    # Try to find an odds token
    odds_m = ODDS_RE.search(s)
    odds = _clean_odds_field(odds_m.group("odds")) if odds_m else None

    # Try to find a line (spread/total). We'll pick the first signed float that isn't the odds token
    # But skip if we already detected a total keyword (to avoid matching "O" in "Over/Under")
    line_val = None
    if not has_over_under:
        # remove the odds token to avoid conflict
        s_no_odds = s
        if odds_m:
            s_no_odds = s.replace(odds_m.group(0), " ")
        # search for a float
        line_m = LINE_RE.search(s_no_odds)
        if line_m:
            try:
                line_val = _clean_line_field(line_m.group("line"))
            except Exception:
                line_val = None
    else:
        # For totals, still search for the line value using raw (not normalized) text
        raw_no_odds = raw
        if odds_m:
            raw_no_odds = raw.replace(odds_m.group(0), " ")
        line_m = LINE_RE.search(raw_no_odds)
        if line_m:
            try:
                line_val = _clean_line_field(line_m.group("line"))
            except Exception:
                line_val = None

    # Infer market type: use pre-normalized keyword check for total
    market_type = "ML"
    if has_over_under:
        market_type = "total"
    elif line_val is not None:
        # Only classify as spread if no total keywords and a line value exists
        market_type = "spread"

    # Selection heuristics: team name before odds or line
    # e.g., "Lakers +110" or "L.A. Clippers -3.5 -110"
    # split on odds if present
    selection = None
    if odds_m:
        parts = s.split(odds_m.group(0))
        selection = parts[0].strip()
    elif line_val is not None:
        parts = s.split(str(line_m.group("line")))
        selection = parts[0].strip()
    else:
        selection = s

    # Clean selection: remove trailing separators
    selection = selection.strip().strip(".,:;-")

    return {
        "raw": raw,
        "selection": selection or None,
        "odds": odds,
        "line": line_val,
        "market_type": market_type,
    }


def _gather_images(paths: Iterable[str]) -> List[str]:
    result: List[str] = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            for child in pth.iterdir():
                if child.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
                    result.append(str(child))
        else:
            result.append(str(pth))
    return result


def ingest_screenshots(
    screenshot_paths: Iterable[str], *, db_path: str | Path, sport: str, season: str, source: str = "screenshot", book: Optional[str] = None, captured_at: Optional[str] = None, json_output: Optional[str] = None
) -> int:
    """Process screenshots into staging rows using OCR, team bundling, and confidence scoring.

    When json_output is set, emits bundled JSON with team-grouped markets and gap flags.
    When json_output is None, writes individual staging rows to DB (backward compatible).

    Returns number of staging rows created (or bundle records in JSON mode).
    """
    images = _gather_images(screenshot_paths)
    created = 0
    json_results: list[dict] = []
    for img in images:
        try:
            # Resolve the OCR function from the module at call-time so tests can patch
            # `src.ocr.ocr.ocr_image` and have this pipeline pick up the mock.
            text = ocr_module.ocr_image(img)
        except Exception:
            # record a staging row with error
            if not json_output:
                add_staging_row(
                    db_path,
                    source=source,
                    captured_at=captured_at or _utcnow_iso(),
                    image_path=img,
                    raw_text=None,
                    book=book,
                    market_type=None,
                    selection=None,
                    line=None,
                    odds=None,
                    team_home_raw=None,
                    team_away_raw=None,
                    game_date=None,
                    match_status="unmatched",
                    match_confidence=0.0,
                    game_id=None,
                )
            created += 1
            continue

        # Per-image captured timestamp: use provided captured_at or record now
        image_captured_at = captured_at or _utcnow_iso()

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        
        # Detect team names from the OCR text top lines
        home_raw = None
        away_raw = None
        for ln in lines[:4]:
            if TEAM_SEP_RE.search(ln):
                parts = re.split(TEAM_SEP_RE, ln, maxsplit=1)
                if len(parts) >= 2:
                    home_raw = parts[0].strip()
                    away_raw = parts[1].strip()
                    break
        
        # Parse all lines and collect with row indices
        parsed_with_indices: List[Tuple[dict, int]] = []
        for row_idx, ln in enumerate(lines):
            parsed = parse_market_line(ln)
            if parsed:
                parsed_with_indices.append((parsed, row_idx))
        
        # JSON mode: bundle markets and emit JSON (even if empty -> gap_flags will signal missing data)
        if json_output:
            # collect debug info for this image so we can inspect parsed lines
            # parsed_with_indices is a list of (parsed_dict, row_idx)
            # make it serializable for debugging output
            debug_parsed = [
                {"row_index": idx, **parsed}
                for parsed, idx in parsed_with_indices
            ]
            bundle = _bundle_team_markets(parsed_with_indices, home_raw or "Unknown", away_raw or "Unknown")
            result_record = {
                "image": img,
                "captured_at": image_captured_at,
                "book": book,
                "team_home_raw": bundle["team_home_raw"],
                "team_away_raw": bundle["team_away_raw"],
                "markets": bundle["markets"],
                "gap_flags": bundle["gap_flags"],
            }
            json_results.append(result_record)
            created += 1
            continue
        
        # DB mode (backward compatible): write individual staging rows
        # In DB mode, write one staging row per market type (ML, spread, total).
        # This avoids duplicate staging rows when OCR yields multiple lines for the same market.
        seen_market_types = set()
        for parsed, row_idx in parsed_with_indices:
            mt = parsed.get("market_type")
            if mt in seen_market_types:
                continue
            seen_market_types.add(mt)
            odds = parsed.get("odds")
            line_val = parsed.get("line")
            
            # Apply validation: odds must be in range and non-zero
            if odds is None or odds == 0 or odds < -1000 or odds > 1000:
                match_status = "unmatched"
                match_confidence = 0.0
            else:
                # attempt to resolve to a game using the per-image captured timestamp
                res = resolve_staging_to_game(
                    db_path,
                    sport=sport,
                    season=season,
                    team_home_raw=home_raw,
                    team_away_raw=away_raw,
                    game_date=image_captured_at,
                )
                match_status = res.get("match_status")
                match_confidence = res.get("match_confidence")

            add_staging_row(
                db_path,
                source=source,
                captured_at=image_captured_at,
                image_path=img,
                raw_text=parsed.get("raw"),
                book=book,
                market_type=parsed.get("market_type"),
                selection=parsed.get("selection"),
                line=line_val,
                odds=odds,
                team_home_raw=home_raw,
                team_away_raw=away_raw,
                game_date=image_captured_at,
                match_status=match_status,
                match_confidence=match_confidence,
                game_id=None if match_status != "matched" else res.get("game_id") if 'res' in locals() else None,
            )
            created += 1
    
    # Write bundled JSON if requested
    if json_output:
        out_path = Path(json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf8") as fh:
            json.dump(json_results, fh, ensure_ascii=False, indent=2)

    return created
