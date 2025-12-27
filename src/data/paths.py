from __future__ import annotations

from pathlib import Path


def db_path_for(sport: str, season: str) -> Path:
    return Path("data/db") / sport / f"{season}.db"
