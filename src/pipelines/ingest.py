"""Pipeline helpers for ingesting game results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.paths import raw_data_dir
from data.validation import validate_dataset
from ingest.base import IngestSource
from ingest.normalize import normalize_games


def resolve_input_path(value: str | Path) -> Path:
    """Resolve an input path, falling back to the raw data directory."""
    candidate = Path(value)
    if candidate.exists():
        return candidate
    raw_candidate = raw_data_dir() / str(value)
    if raw_candidate.exists():
        return raw_candidate
    raise FileNotFoundError(
        f"Input not found: '{value}' (also tried '{raw_candidate}')"
    )


def ingest_games(
    source: IngestSource,
    *,
    input_path: Path | None,
    input_text: str | None,
    sport: str,
    season: str,
    format_hint: str | None = None,
) -> list:
    """Load, normalize, and validate games from an ingest source."""
    if input_text:
        games = source.load_text(input_text, sport=sport, season=season)
    elif input_path is not None:
        games = source.load_path(
            input_path,
            sport=sport,
            season=season,
            format_hint=format_hint,
        )
    else:
        raise ValueError("Provide input_path or input_text for ingestion.")

    normalized = normalize_games(games, sport=sport, season=season)
    validate_dataset(pd.DataFrame([game.model_dump() for game in normalized]))
    return normalized
