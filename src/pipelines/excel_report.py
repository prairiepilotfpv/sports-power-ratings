"""Excel report pipeline for ranking summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from data.paths import processed_path_for
from data.repository import load_games
from pipelines.common import normalize_games, resolve_output_path
from models.registry import list_models, normalize_model_name
from pipelines.model_params import resolve_model_params
from pipelines.run_rankings import build_rankings


def build_excel_report(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    division: str | None = None,
    conference: str | None = None,
    models: Iterable[str] | None = None,
    output_path: str | Path | None = None,
    model_params: dict[str, float] | None = None,
    model_params_file: str | Path | None = None,
) -> Path | list[Path]:
    """Write rankings to an Excel workbook (one sheet per model)."""
    requested_models: List[str]
    if models is None:
        requested_models = list_models()
    else:
        requested_models = [normalize_model_name(model) for model in models]
    rows = load_games(
        db_path,
        sport=sport,
        season=season,
        division=division,
        conference=conference,
    )
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    multiple = len(requested_models) > 1
    report_paths: list[Path] = []
    for model in requested_models:
        default_path = processed_path_for(sport, season, "report.xlsx")
        report_path = resolve_output_path(
            output_path,
            default_path=default_path,
            model=model,
            add_prefix=multiple,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(report_path) as writer:
            try:
                resolved_params = resolve_model_params(
                    model, params=model_params, params_file=model_params_file
                )
                rankings = build_rankings(
                    df.copy(deep=True), model=model, model_params=resolved_params
                )
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
