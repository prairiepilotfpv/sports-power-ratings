from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from bootstrap import ensure_src_on_path

ensure_src_on_path()

from data.repository import load_games
from pipelines.common import normalize_games
from pipelines.run_rankings import build_rankings
from pipelines.schedule import build_schedule_with_projections


def _today_spreads(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return schedule_df
    today = pd.Timestamp.today().date()
    df = schedule_df.assign(_date=pd.to_datetime(schedule_df["date"], errors="coerce").dt.date)
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


def build_excel_report(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str = "bradley-terry",
    output_path: str | Path | None = None,
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

    if output_path is None:
        output_path = Path("data/processed") / sport / season / "daily_report.xlsx"
    else:
        output_path = Path(output_path)
        if output_path.is_dir() or output_path.suffix == "":
            output_path = output_path / "daily_report.xlsx"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if "Sheet" in writer.book.sheetnames:
            writer.book.remove(writer.book["Sheet"])
        start_row = 0
        start_row = _write_section(
            writer,
            "Summary",
            "Aggregated (Bradley-Terry only)",
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
