"""Identity utilities for team normalization and fuzzy matching.

This module will provide helpers for canonicalizing team names, resolving
aliases, and computing a confidence score for matches between OCR text and
`games` team names.
"""

from __future__ import annotations

from typing import Tuple, Dict, Iterable
import json
import re
from difflib import SequenceMatcher
import os
from pathlib import Path

CONFIG_TEAM_ALIASES = Path("data/config/team_aliases.json")
CONFIG_IDENTITY = Path("data/config/identity.json")


def _alias_config_path() -> Path:
    env_path = os.environ.get("TEAM_ALIAS_FILE")
    if env_path:
        return Path(env_path)
    return CONFIG_TEAM_ALIASES


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def normalize_team_name(raw: str) -> str:
    """Return a normalized team name suitable for fuzzy matching.

    Normalization lowercases, strips punctuation, and collapses whitespace.
    """
    if raw is None:
        return ""
    s = raw.strip().lower()
    # remove punctuation
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s


def resolve_team_alias(name: str, alias_map: dict) -> str | None:
    """Return canonical team name if `name` matches an alias in `alias_map`.

    alias_map is expected as {canonical: [aliases...]}
    Matching is case-insensitive using normalized forms.
    """
    if not name:
        return None
    needle = normalize_team_name(name)
    # direct canonical
    for canon, aliases in alias_map.items():
        if normalize_team_name(canon) == needle:
            return canon
    # check aliases
    for canon, aliases in alias_map.items():
        for alias in aliases:
            if normalize_team_name(alias) == needle:
                return canon
    return None


def fuzzy_match_team(raw: str, candidates: Iterable[str]) -> Tuple[str | None, float]:
    """Return (best_match_or_None, score) using difflib SequenceMatcher.

    Score is in 0.0-1.0 range.
    """
    if not raw:
        return None, 0.0
    nr = normalize_team_name(raw)
    best = None
    best_score = 0.0
    for c in candidates:
        nc = normalize_team_name(c)
        if nc == nr:
            return c, 0.95  # normalized exact match
        ratio = SequenceMatcher(None, nr, nc).ratio()
        # Scale difflib ratio gently into 0.6-0.9 band: keep raw ratio
        if ratio > best_score:
            best_score = ratio
            best = c
    # if best_score is close to 1 but not exact normalized, map to [0.6,0.9]
    if best is not None:
        # Keep the SequenceMatcher ratio as returned (0-1). Caller will interpret
        return best, best_score
    return None, 0.0


def load_alias_map(sport: str) -> dict:
    data = _load_json(_alias_config_path())
    return data.get(sport, {})


def save_team_alias(sport: str, canonical: str, alias: str) -> None:
    if not canonical or not alias or canonical.strip() == "" or alias.strip() == "":
        return
    path = _alias_config_path()
    data = _load_json(path)
    sport_map = data.setdefault(sport, {})
    aliases = sport_map.setdefault(canonical, [])
    if alias not in aliases:
        aliases.append(alias)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))


def load_identity_config() -> dict:
    data = _load_json(CONFIG_IDENTITY)
    return data.get("match_thresholds", {"auto": 0.9, "needs_review": 0.75})
