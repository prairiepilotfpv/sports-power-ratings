"""Pipeline helpers for ingesting game results."""

from __future__ import annotations

from pathlib import Path
import inspect

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
    division: str | None = None,
    conference: str | None = None,
    format_hint: str | None = None,
) -> list:
    """Load, normalize, and validate games from an ingest source."""
    def _supports_param(func, name: str) -> bool:
        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return False
        return name in params or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )

    if input_text:
        load_kwargs = {"sport": sport, "season": season}
        if _supports_param(source.load_text, "division"):
            load_kwargs["division"] = division
        if _supports_param(source.load_text, "conference"):
            load_kwargs["conference"] = conference
        games = source.load_text(input_text, **load_kwargs)
    elif input_path is not None:
        load_kwargs = {"sport": sport, "season": season, "format_hint": format_hint}
        if _supports_param(source.load_path, "division"):
            load_kwargs["division"] = division
        if _supports_param(source.load_path, "conference"):
            load_kwargs["conference"] = conference
        games = source.load_path(input_path, **load_kwargs)
    else:
        raise ValueError("Provide input_path or input_text for ingestion.")

    normalized = normalize_games(
        games,
        sport=sport,
        season=season,
        division=division,
        conference=conference,
    )
    validate_dataset(pd.DataFrame([game.model_dump() for game in normalized]))
    return normalized
