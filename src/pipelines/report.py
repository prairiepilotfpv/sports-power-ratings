from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.paths import processed_path_for
from data.repository import load_games
from pipelines.common import normalize_games, resolve_output_path
from models.registry import list_models, normalize_model_name
from pipelines.run_rankings import build_rankings
from pipelines.schedule import build_schedule_with_projections


def _today_spreads(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return schedule_df
    today = pd.Timestamp.today().date()
    df = schedule_df.assign(
        _date=pd.to_datetime(schedule_df["date"], errors="coerce").dt.date
    )
    df = df[(df["status"] == "scheduled") & (df["_date"] == today)]
    return df[
        [
            "date",
            "away_team",
            "home_team",
            "projected_spread",
            "projected_home_spread",
            "projected_total",
            "projected_winner",
            "home_rating",
            "away_rating",
        ]
    ]


def _write_section(
    writer: pd.ExcelWriter,
    sheet_name: str,
    title: str,
    df: pd.DataFrame,
    start_row: int,
) -> int:
    if sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
    else:
        ws = writer.book.create_sheet(sheet_name)
        writer.sheets[sheet_name] = ws

    ws.cell(row=start_row + 1, column=1, value=title)
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row + 1)
    return start_row + len(df) + 3


def _resolve_models(model: str | None) -> list[str]:
    if model is None:
        return list_models()
    return [normalize_model_name(model)]


def _build_single_report(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    output_path: str | Path | None,
    add_prefix: bool,
) -> Path:
    rows = load_games(db_path, sport=sport, season=season)
    df = normalize_games(rows)
    if df.empty:
        raise ValueError(f"No games found for sport={sport!r}, season={season!r}")

    rankings = build_rankings(df, model=model, require_scores=False)
    schedule_path = build_schedule_with_projections(
        db_path,
        sport=sport,
        season=season,
        model=model,
    )
    schedule_df = pd.read_csv(schedule_path)
    spreads = _today_spreads(schedule_df)

    default_path = processed_path_for(sport, season, "daily_report.xlsx")
    output_path = resolve_output_path(
        output_path,
        default_path=default_path,
        model=model,
        add_prefix=add_prefix,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if "Sheet" in writer.book.sheetnames:
            writer.book.remove(writer.book["Sheet"])
        start_row = 0
        start_row = _write_section(
            writer,
            "Summary",
            f"Aggregated ({model})",
            rankings,
            start_row,
        )
        _write_section(
            writer,
            "Summary",
            "Spreads",
            spreads,
            start_row,
        )

    return output_path


def build_excel_report(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str | None = None,
    output_path: str | Path | None = None,
) -> Path | list[Path]:
    models = _resolve_models(model)
    multiple = len(models) > 1
    results = [
        _build_single_report(
            db_path,
            sport=sport,
            season=season,
            model=model_name,
            output_path=output_path,
            add_prefix=multiple,
        )
        for model_name in models
    ]
    return results[0] if len(results) == 1 else results
