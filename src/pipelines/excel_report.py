from __future__ import annotations

"""Excel report pipeline for ranking summaries."""

from pathlib import Path
from typing import Iterable, List

try:  # Allow execution from repository root or nested directories
    from bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - fallback when bootstrap isn't on sys.path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from bootstrap import ensure_src_on_path

ensure_src_on_path()

import pandas as pd

from data.repository import load_games
from pipelines.common import normalize_games
from models.registry import (
    get_model_abbreviation,
    list_models,
    normalize_model_name,
)
from pipelines.run_rankings import build_rankings


def _resolve_output_path(
    output_path: str | Path | None,
    *,
    sport: str,
    season: str,
    model: str,
    add_prefix: bool,
) -> Path:
    """Resolve output location, allowing directory paths."""
    if output_path is None:
        resolved = Path("data/processed") / sport / season / "report.xlsx"
    else:
        resolved = Path(output_path)
        if resolved.is_dir() or resolved.suffix == "":
            resolved = resolved / "report.xlsx"
    if add_prefix:
        abbrev = get_model_abbreviation(model)
        resolved = resolved.with_name(f"{abbrev}_{resolved.name}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def build_excel_report(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    models: Iterable[str] | None = None,
    output_path: str | Path | None = None,
) -> Path | list[Path]:
    """Write rankings to an Excel workbook (one sheet per model)."""
    requested_models: List[str]
    if models is None:
        requested_models = list_models()
    else:
        requested_models = [normalize_model_name(model) for model in models]
    rows = load_games(db_path, sport=sport, season=season)
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    multiple = len(requested_models) > 1
    report_paths: list[Path] = []
    for model in requested_models:
        report_path = _resolve_output_path(
            output_path,
            sport=sport,
            season=season,
            model=model,
            add_prefix=multiple,
        )
        with pd.ExcelWriter(report_path) as writer:
            try:
                rankings = build_rankings(df, model=model)
            except ValueError as exc:
                if "No completed games" in str(exc):
                    raise ValueError(
                        f"No completed games found for sport={sport!r}, season={season!r}"
                    ) from exc
                raise
            rankings = rankings.loc[:, ["team", "rating", "points", "games"]]
            rankings.to_excel(writer, sheet_name=model, index=False)

            summary = pd.DataFrame()
            summary.to_excel(writer, sheet_name="Summary", index=False)
        report_paths.append(report_path)

    return report_paths[0] if len(report_paths) == 1 else report_paths
