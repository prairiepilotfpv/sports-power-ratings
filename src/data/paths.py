"""Path helpers for data storage."""

from __future__ import annotations

from pathlib import Path


_DATA_DIR = Path("data")


def db_path_for(sport: str, season: str) -> Path:
    """Return the default SQLite DB path for a sport/season pair."""
    return db_dir() / sport / f"{season}.db"


def db_dir() -> Path:
    """Return the base directory for SQLite databases."""
    return _DATA_DIR / "db"


def raw_data_dir() -> Path:
    """Return the base directory for raw ingest files."""
    return _DATA_DIR / "raw"


def processed_dir(*, sport: str | None = None, season: str | None = None) -> Path:
    """Return the base directory for processed outputs, optionally scoped by sport/season."""
    base = _DATA_DIR / "processed"
    if sport and season:
        return base / sport / season
    return base


def processed_path_for(sport: str, season: str, filename: str) -> Path:
    """Return a processed output path for a sport/season filename."""
    return processed_dir(sport=sport, season=season) / filename
