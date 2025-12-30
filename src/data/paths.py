from __future__ import annotations

"""Path helpers for data storage."""

from pathlib import Path


def db_path_for(sport: str, season: str) -> Path:
    """Return the default SQLite DB path for a sport/season pair."""
    return Path("data/db") / sport / f"{season}.db"
