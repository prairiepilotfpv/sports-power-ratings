"""Shared helpers for pipeline data normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from models.registry import get_model_abbreviation


def normalize_games(rows: Iterable[Any]) -> pd.DataFrame:
    """Convert rows or Pydantic models into a normalized DataFrame."""
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "model_dump"):
            normalized_rows.append(row.model_dump())
        else:
            normalized_rows.append(dict(row))
    df = pd.DataFrame(normalized_rows)
    if df.empty:
        return df
    for score_col in ("home_score", "away_score"):
        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if "start_time" in df.columns:
            start_dt = pd.to_datetime(df["start_time"], errors="coerce")
            if start_dt.notna().any():
                dt = start_dt.where(start_dt.notna(), dt)
        if dt.notna().any():
            df = (
                df.assign(_dt=dt)
                .sort_values(["_dt", "game_id"])
                .drop(columns=["_dt"], errors="ignore")
            )
    return df


def resolve_output_path(
    output_path: str | Path | None,
    *,
    default_path: Path,
    model: str | None = None,
    add_prefix: bool = False,
) -> Path:
    """Resolve an output path, allowing directory inputs and model prefixes."""
    resolved = Path(output_path) if output_path is not None else default_path
    if resolved.is_dir() or resolved.suffix == "":
        resolved = resolved / default_path.name
    if add_prefix and model:
        abbrev = get_model_abbreviation(model)
        resolved = resolved.with_name(f"{abbrev}_{resolved.name}")
    return resolved
