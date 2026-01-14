"""Simple paste-parser for betting markets.

Parses plain-text/HTML-copy pasted market blocks and extracts market rows
compatible with the project's betting staging schema.

Functions:
- parse_paste(text: str) -> list[dict]
- parse_market_line(line: str) -> dict | None
- match_team_name(name: str, candidates: list[str]) -> str | None

This module is intentionally dependency-free and uses only stdlib.
"""
from __future__ import annotations

import re
from typing import List, Dict, Optional, Tuple
from difflib import get_close_matches
from math import isfinite

TEAM_NAME_RE = re.compile(r"^[A-Za-z0-9 .&'\-()]{2,40}$")
TEAM_SEP_RE = re.compile(r"\b(?:vs\.?|v\.?|@)\b", flags=re.IGNORECASE)
ODDS_RE = re.compile(r"(?P<odds>[+-]\d{2,3})")
LINE_RE = re.compile(r"(?P<line>-?\d+(?:\.\d+)?)")

_AD_FILTER_KEYWORDS = (
    "logo",
    "promo",
    "promotion",
    "get a",
    "bet and",
    "terms",
    "new users",
    "call",
    "claim",
    "offers",
    "final",
)

# Words that should not be interpreted as team names
_TEAM_STOPWORDS = {"consensus", "best odds", "offers", "promotion", "promo", "n/a", "final"}


def _normalize_text(s: str) -> str:
    return s.replace("−", "-").replace("–", "-").strip()


def parse_market_line(line: str) -> Optional[Dict]:
    """Parse a single line for selection, odds, line, and market_type.

    Returns a dict with keys: raw, selection, odds (int|None), line (float|None), market_type
    or None when the line contains no useful tokens.
    """
    raw = line.strip()
    if not raw:
        return None
    s = _normalize_text(raw)

    # filter obvious non-content
    low = s.lower()
    if any(k in low for k in _AD_FILTER_KEYWORDS) or "www." in low or "http" in low:
        return None

    # find odds
    odds_m = ODDS_RE.search(s)
    odds = None
    if odds_m:
        try:
            odds = int(odds_m.group("odds"))
        except Exception:
            odds = None

    # detect total keywords
    has_total = bool(re.search(r"\bover\b|\bunder\b|o/u|total", s, flags=re.IGNORECASE))

    # find a numeric line value (avoid capturing the odds token)
    line_val = None
    s_no_odds = s
    if odds_m:
        s_no_odds = s.replace(odds_m.group(0), " ")
    line_m = LINE_RE.search(s_no_odds)
    if line_m:
        try:
            line_val = float(line_m.group("line"))
        except Exception:
            line_val = None

    # infer market type
    if has_total:
        market_type = "total"
    elif line_val is not None:
        market_type = "spread"
    elif odds is not None:
        market_type = "ML"
    else:
        # no market present
        return None

    # selection heuristics: text before odds or before the numeric line
    selection = None
    if odds_m:
        parts = s.split(odds_m.group(0), 1)
        selection = parts[0].strip()
    elif line_m:
        parts = s_no_odds.split(line_m.group("line"), 1)
        selection = parts[0].strip()
    else:
        selection = s

    # normalize selection for totals
    if market_type == "total":
        sel_low = selection.lower() if selection else ""
        if "over" in sel_low:
            selection = "Over"
        elif "under" in sel_low:
            selection = "Under"
        else:
            # sometimes the line precedes the over/under token; try to infer from raw
            if selection and (selection.strip().upper().startswith("O/") or selection.strip().upper() == "O"):
                selection = "Over"

    selection = selection.strip(" .,:;-") if selection else None

    return {
        "raw": raw,
        "selection": selection,
        "odds": odds,
        "line": line_val,
        "market_type": market_type,
    }

def _american_odds_to_prob(odds: int) -> float:
    """Convert American odds to implied probability (0..1).

    Returns 0.0 for invalid inputs.
    """
    try:
        if odds is None:
            return 0.0
        odds = int(odds)
        if odds < 0:
            return -odds / (-odds + 100)
        return 100.0 / (odds + 100.0)
    except Exception:
        return 0.0


def match_team_name(name: str, candidates: List[str], *, cutoff: float = 0.8) -> Optional[str]:
    """Return best match from candidates using exact then fuzzy matching.

    Returns None when no suitable match found.
    """
    if not name or not candidates:
        return None
    name_clean = name.strip()
    # exact match (case-insensitive)
    for c in candidates:
        if c.lower() == name_clean.lower():
            return c
    # substring/token match: accept when the short name appears in candidate (e.g., 'Bucks' in 'Milwaukee Bucks')
    for c in candidates:
        if name_clean.lower() in c.lower() or c.lower() in name_clean.lower():
            return c
    # fuzzy
    matches = get_close_matches(name_clean, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def parse_paste(text: str, *, team_list: Optional[List[str]] = None, sport: Optional[str] = None) -> List[Dict]:
    """Parse pasted market text into market rows.

    This function detects matchups heuristically (pairing nearby team-name lines)
    and extracts markets within a window after those lines. If `sport` is
    provided it will attempt to canonicalize team names using the project's
    identity alias map.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]

    # filter out ad/logo lines early
    filtered_lines: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(k in low for k in _AD_FILTER_KEYWORDS) or "logo" in low:
            continue
        filtered_lines.append((i, ln))

    # find candidate team-name lines
    team_indices: List[Tuple[int, str]] = []
    for idx, ln in filtered_lines:
        low = ln.lower()
        if "team icon" in low:
            continue
        if TEAM_SEP_RE.search(ln):
            parts = re.split(TEAM_SEP_RE, ln, maxsplit=1)
            if len(parts) >= 2:
                left = parts[0].strip()
                right = parts[1].strip()
                if TEAM_NAME_RE.match(left) and left.lower() not in _TEAM_STOPWORDS:
                    team_indices.append((idx, left))
                if TEAM_NAME_RE.match(right) and right.lower() not in _TEAM_STOPWORDS:
                    team_indices.append((idx, right))
            continue
        # single-line name heuristic
        if TEAM_NAME_RE.match(ln) and not any(ch.isdigit() for ch in ln):
            if len(ln) <= 40 and low not in _TEAM_STOPWORDS:
                team_indices.append((idx, ln))

    # pair sequential team names into matchups
    matchups: List[Tuple[int, str, str]] = []
    team_indices.sort()
    i = 0
    while i + 1 < len(team_indices):
        left_idx, left_name = team_indices[i]
        right_idx, right_name = team_indices[i + 1]
        if right_idx - left_idx <= 12:
            matchups.append((left_idx, left_name, right_name))
            i += 2
        else:
            i += 1

    # parse every filtered line into market tokens
    parsed_with_idx: List[Tuple[int, Dict]] = []
    for idx, ln in filtered_lines:
        parsed = parse_market_line(ln)
        if parsed:
            parsed_with_idx.append((idx, parsed))

    # load alias candidates when sport provided
    alias_map = {}
    candidates: List[str] = []
    if sport:
        try:
            alias_map = idu.load_alias_map(sport.lower())
            candidates = list(alias_map.keys())
        except Exception:
            alias_map = {}
            candidates = []

    results: List[Dict] = []

    # for each matchup, collect parsed markets within a window following the matchup line
    for idx_m, home, away in matchups:
        # determine window end (start of next matchup or +40 lines)
        next_starts = [t for t, _, _ in matchups if t > idx_m]
        window_end = next_starts[0] if next_starts else idx_m + 40
        window_start = idx_m

        markets_for_pair = [p for idx, p in parsed_with_idx if window_start <= idx < window_end]

        # group by market_type so we can infer selections when missing
        by_type: dict = {}
        for p in markets_for_pair:
            mt = p.get("market_type") or "ML"
            by_type.setdefault(mt, []).append(p)

        # Attempt to infer favorite from ML if present
        # Attempt to infer favorite from ML if present using implied probabilities
        favorite_ml: Optional[str] = None
        ml_items = by_type.get("ML", [])
        if ml_items:
            best_prob = -1.0
            for m in ml_items:
                sel = m.get("selection")
                odds = m.get("odds")
                if sel and odds is not None:
                    prob = _american_odds_to_prob(odds)
                    if prob > best_prob:
                        best_prob = prob
                        favorite_ml = sel

        # Infer favorite from spread polarity when available
        favorite_spread: Optional[str] = None
        spread_items = by_type.get("spread", [])
        for m in spread_items:
            sel = m.get("selection")
            ln = m.get("line")
            if sel and ln is not None:
                # negative line indicates the selection is the favorite
                if isinstance(ln, (int, float)) and ln < 0:
                    favorite_spread = sel
                    break

        # Decide reconciled favorite and confidence baseline
        reconciled_fav: Optional[str] = None
        fav_confidence = 0.0
        if favorite_ml and favorite_spread:
            if favorite_ml == favorite_spread:
                reconciled_fav = favorite_ml
                fav_confidence = 0.9
            else:
                # prefer ML as primary signal but mark lower confidence when they disagree
                reconciled_fav = favorite_ml
                fav_confidence = 0.6
        elif favorite_ml:
            reconciled_fav = favorite_ml
            fav_confidence = 0.8
        elif favorite_spread:
            reconciled_fav = favorite_spread
            fav_confidence = 0.7

        # For spreads and ML, use combination of ML favorite and line sign to assign selections
        for mt, items in by_type.items():
            if mt == "total":
                # Totals: if both present but selection missing, assign Over/Under by ordinal
                if len(items) >= 2:
                    for i, p in enumerate(items):
                        sel = p.get("selection")
                        if not sel or sel.strip() == "":
                            p["selection"] = "Over" if i == 0 else "Under"
                continue

            # For ML/spread: when multiple entries exist, prefer ML-based favorite mapping
            if len(items) >= 2:
                # If favorite known, map items by checking odds or line sign
                for p in items:
                    sel = p.get("selection")
                    line_val = p.get("line")
                    odds = p.get("odds")
                    assigned = False

                    if sel and isinstance(sel, str) and len(sel.strip()) > 2:
                        assigned = True

                    # If reconciled favorite known and this entry has negative odds or negative line, map to favorite
                    if not assigned and reconciled_fav:
                        if odds is not None and odds < 0:
                            p["selection"] = reconciled_fav
                            assigned = True
                        elif line_val is not None and line_val < 0:
                            p["selection"] = reconciled_fav
                            assigned = True

                    # If still unassigned, use line polarity: negative -> favorite (if favorite known else home), positive -> other
                    if not assigned and line_val is not None:
                        if line_val < 0:
                            p["selection"] = reconciled_fav if reconciled_fav else home
                            assigned = True
                        elif line_val > 0:
                            p["selection"] = away
                            assigned = True

                # Fallback: ensure first -> home, second -> away
                for i, p in enumerate(items):
                    if not p.get("selection") or len(str(p.get("selection"))) <= 2:
                        p["selection"] = home if i == 0 else away
            else:
                # single item: if selection missing, prefer favorite/home
                p = items[0]
                if not p.get("selection") or len(str(p.get("selection"))) <= 2:
                    if p.get("market_type") == "spread" and p.get("line") is not None:
                        # if line negative -> favorite/home, else away
                        if p.get("line") < 0:
                            p["selection"] = reconciled_fav if reconciled_fav else home
                        else:
                            p["selection"] = away
                    else:
                        p["selection"] = reconciled_fav if reconciled_fav else home

        for p in markets_for_pair:
            out: Dict = {
                "source": "paste",
                "captured_at": None,
                "raw_text": p.get("raw"),
                "book": None,
                "team_home_raw": home,
                "team_away_raw": away,
                "game_date": None,
                "market_type": p.get("market_type"),
                "selection": p.get("selection"),
                "line": p.get("line"),
                "odds": p.get("odds"),
                "match_status": "unmatched",
                "match_confidence": 0.0,
                "game_id": None,
                "created_at": None,
            }

            # canonicalize team names using alias_map/candidates if available
            if candidates:
                hcanon = idu.resolve_team_alias(home, alias_map) if alias_map else None
                if hcanon:
                    out["team_home_raw"] = hcanon
                else:
                    hmatch, hscore = idu.fuzzy_match_team(home, candidates)
                    if hmatch and hscore >= 0.6:
                        out["team_home_raw"] = hmatch

                acanon = idu.resolve_team_alias(away, alias_map) if alias_map else None
                if acanon:
                    out["team_away_raw"] = acanon
                else:
                    amatch, ascore = idu.fuzzy_match_team(away, candidates)
                    if amatch and ascore >= 0.6:
                        out["team_away_raw"] = amatch

                # selection resolution
                sel = out.get("selection")
                if sel and isinstance(sel, str):
                    scanon = idu.resolve_team_alias(sel, alias_map) if alias_map else None
                    if scanon:
                        out["selection"] = scanon
                    else:
                        smatch, sscore = idu.fuzzy_match_team(sel, candidates)
                        if smatch and sscore >= 0.6:
                            out["selection"] = smatch
                # set match confidence based on reconciled favorite and selection alignment
                base_conf = fav_confidence if fav_confidence and fav_confidence > 0 else 0.5
                sel_norm = out.get("selection")
                if sel_norm and sel_norm not in (out.get("team_home_raw"), out.get("team_away_raw")) and sel_norm not in ("Over", "Under"):
                    base_conf = base_conf * 0.6
                out["match_confidence"] = round(float(base_conf), 2)
            elif team_list:
                home_match = match_team_name(home, team_list)
                away_match = match_team_name(away, team_list)
                if home_match:
                    out["team_home_raw"] = home_match
                if away_match:
                    out["team_away_raw"] = away_match
                sel = out.get("selection")
                if sel and isinstance(sel, str):
                    sel_match = match_team_name(sel, team_list)
                    if sel_match:
                        out["selection"] = sel_match
                # set match confidence based on reconciled favorite and selection alignment
                base_conf = fav_confidence if fav_confidence and fav_confidence > 0 else 0.5
                sel_norm = out.get("selection")
                if sel_norm and sel_norm not in (out.get("team_home_raw"), out.get("team_away_raw")) and sel_norm not in ("Over", "Under"):
                    base_conf = base_conf * 0.6
                out["match_confidence"] = round(float(base_conf), 2)

            results.append(out)

    # fallback: if no matchups, return standalone parsed rows
    if not results:
        for _, p in parsed_with_idx:
            out = {
                "source": "paste",
                "captured_at": None,
                "raw_text": p.get("raw"),
                "book": None,
                "team_home_raw": None,
                "team_away_raw": None,
                "game_date": None,
                "market_type": p.get("market_type"),
                "selection": p.get("selection"),
                "line": p.get("line"),
                "odds": p.get("odds"),
                "match_status": "unmatched",
                "match_confidence": 0.5,
                "game_id": None,
                "created_at": None,
            }
            results.append(out)

    return results
