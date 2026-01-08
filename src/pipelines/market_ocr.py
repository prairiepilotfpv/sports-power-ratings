"""Market OCR ingestion pipeline (skeleton).

Responsibilities:
- Accept screenshot(s) or images
- Run OCR via `src.ocr.ocr` helpers
- Write staging rows via `src.data.betting_repository`
- Perform fuzzy team matching via `src.utils.identity`

This module provides a pipeline entrypoint but implementations will be
added as we iterate on matching and validation rules.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import re
from pathlib import Path
import json

from src.ocr.ocr import ocr_image
from src.data.betting_repository import (
    _clean_odds_field,
    _clean_line_field,
    add_staging_row,
    resolve_staging_to_game,
    _utcnow_iso,
)


TEAM_SEP_RE = re.compile(r"\b(?:vs\.?|v\.|v|@)\b", flags=re.IGNORECASE)
ODDS_RE = re.compile(r"(?P<odds>[+-][0-9OIl]{1,4})")
LINE_RE = re.compile(r"(?P<line>-?\d+(?:\.\d+)?)")

def _normalize_text(s: str) -> str:
    return s.replace("−", "-").replace("–", "-").replace("O", "0").replace("o", "0").strip()


def parse_market_line(line: str) -> dict | None:
    """Attempt to parse a single OCR text line into market fields.

    Returns dict with keys: selection, odds, line, market_type (ML/spread/total), raw
    or None if parsing failed.
    """
    raw = line.strip()
    if not raw:
        return None
    s = _normalize_text(raw)

    # Try to find an odds token
    odds_m = ODDS_RE.search(s)
    odds = _clean_odds_field(odds_m.group("odds")) if odds_m else None

    # Try to find a line (spread/total). We'll pick the first signed float that isn't the odds token
    line_val = None
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

    # Infer market type
    market_type = "ML"
    if "over" in s_no_odds.lower() or "under" in s_no_odds.lower() or re.search(r"o/u|o\.|u\.", s_no_odds, flags=re.I):
        market_type = "total"
    elif line_val is not None:
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
    """Process screenshots into staging rows using OCR and heuristics.

    Returns number of staging rows created.
    """
    images = _gather_images(screenshot_paths)
    created = 0
    json_results: list[dict] = []
    for img in images:
        try:
            text = ocr_image(img)
        except Exception:
            # record a staging row with error
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

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # attempt to detect team names from the OCR text top lines
        home_raw = None
        away_raw = None
        for ln in lines[:4]:
            if TEAM_SEP_RE.search(ln):
                parts = re.split(TEAM_SEP_RE, ln)
                if len(parts) >= 3:
                    # parts like [home, sep, away]
                    home_raw = parts[0].strip()
                    away_raw = parts[2].strip()
                    break
        # parse each line for markets
        for ln in lines:
            parsed = parse_market_line(ln)
            if not parsed:
                continue
            # set team fields from detected header if available
            team_home_raw = home_raw
            team_away_raw = away_raw
            game_date = captured_at

            # Build a minimal record for JSON mode
            rec = {
                "image": img,
                "captured_at": captured_at,
                "book": book,
                "raw": parsed.get("raw"),
                "market_type": parsed.get("market_type"),
                "selection": parsed.get("selection"),
                "line": parsed.get("line"),
                "odds": parsed.get("odds"),
                "team_home_raw": team_home_raw,
                "team_away_raw": team_away_raw,
            }

            if json_output:
                json_results.append(rec)
                created += 1
                continue

            # Apply validation: odds must be in range and non-zero
            odds = parsed.get("odds")
            line_val = parsed.get("line")
            if odds is None or odds == 0 or odds < -1000 or odds > 1000:
                match_status = "unmatched"
                match_confidence = 0.0
            else:
                # attempt to resolve to a game
                res = resolve_staging_to_game(db_path, sport=sport, season=season, team_home_raw=team_home_raw, team_away_raw=team_away_raw, game_date=game_date)
                match_status = res.get("match_status")
                match_confidence = res.get("match_confidence")

            add_staging_row(
                db_path,
                source=source,
                captured_at=captured_at or _utcnow_iso(),
                image_path=img,
                raw_text=ln,
                book=book,
                market_type=parsed.get("market_type"),
                selection=parsed.get("selection"),
                line=line_val,
                odds=odds,
                team_home_raw=team_home_raw,
                team_away_raw=team_away_raw,
                game_date=game_date,
                match_status=match_status,
                match_confidence=match_confidence,
                game_id=None if match_status != "matched" else res.get("game_id") if 'res' in locals() else None,
            )
            created += 1
    # If JSON mode, write results to file if requested
    if json_output:
        out_path = Path(json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf8") as fh:
            json.dump(json_results, fh, ensure_ascii=False, indent=2)

    return created
