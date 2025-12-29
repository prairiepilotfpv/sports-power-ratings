from __future__ import annotations

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
from pipelines.run_rankings import build_rankings


def _resolve_output_path(
    output_path: str | Path | None,
    *,
    sport: str,
    season: str,
) -> Path:
    if output_path is None:
        resolved = Path("data/processed") / sport / season / "report.xlsx"
    else:
        resolved = Path(output_path)
        if resolved.is_dir() or resolved.suffix == "":
            resolved = resolved / "report.xlsx"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def build_excel_report(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    models: Iterable[str] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    requested_models: List[str] = list(models) if models is not None else ["bradley-terry"]
    rows = load_games(db_path, sport=sport, season=season)
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    report_path = _resolve_output_path(output_path, sport=sport, season=season)

    with pd.ExcelWriter(report_path) as writer:
        for model in requested_models:
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

    return report_path
