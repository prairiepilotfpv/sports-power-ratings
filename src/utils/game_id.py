from __future__ import annotations

from hashlib import sha1
from datetime import date as _date

from .identity import normalize_team_name


def make_game_id(sport: str, season: str, date_value, away_team: str, home_team: str) -> str:
    """Deterministic, stable game id for a game identity.

    Returns a readable id like: "{sport}:{season}:{YYYY-MM-DD}:{hash12}" where
    the hash is SHA1 over a canonical key built from normalized inputs.
    """
    if not sport or not season or not date_value or not away_team or not home_team:
        raise ValueError("sport, season, date, away_team and home_team are required to build game_id")

    # normalize date into ISO date string
    if isinstance(date_value, _date):
        date_iso = date_value.isoformat()
    else:
        date_iso = str(date_value).strip()
        # accept full datetime strings but only keep date portion
        if "T" in date_iso:
            date_iso = date_iso.split("T", 1)[0]
        if " " in date_iso:
            date_iso = date_iso.split(" ", 1)[0]

    sport_key = str(sport).strip().lower()
    season_key = str(season).strip().lower()
    away_norm = normalize_team_name(away_team)
    home_norm = normalize_team_name(home_team)

    key = f"{sport_key}|{season_key}|{date_iso}|{away_norm}|{home_norm}"
    digest = sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{sport_key}:{season_key}:{date_iso}:{digest}"
