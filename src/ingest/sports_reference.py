from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd

from ingest.schema import GameResult


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _as_int(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(str(value).strip()))
        except Exception:
            return None


def _resolve_pts_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    pts_cols = [c for c in df.columns if c == "pts" or c.startswith("pts.")]
    if len(pts_cols) >= 2:
        return pts_cols[0], pts_cols[1]

    away_pts = _find_column(df, "visitor pts", "visitor_pts", "away pts", "away_pts")
    home_pts = _find_column(df, "home pts", "home_pts")
    return away_pts, home_pts


def _parse_sr_dataframe(
    df: pd.DataFrame,
    sport: str | None = None,
    season: str | None = None,
) -> List[GameResult]:
    df = _normalize_columns(df)

    date_col = _find_column(df, "date")
    away_col = _find_column(df, "visitor/neutral", "visitor", "away", "away/neutral")
    home_col = _find_column(df, "home/neutral", "home")
    ot_col = _find_column(df, "ot")
    box_col = _find_column(df, "box score", "boxscore", "box")
    notes_col = _find_column(df, "notes")
    away_pts_col, home_pts_col = _resolve_pts_columns(df)

    if not date_col or not away_col or not home_col:
        missing = [
            name
            for name, col in {
                "date": date_col,
                "away": away_col,
                "home": home_col,
            }.items()
            if col is None
        ]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    games: List[GameResult] = []
    for _, row in df.iterrows():
        if pd.isna(row.get(date_col)) or pd.isna(row.get(away_col)) or pd.isna(row.get(home_col)):
            continue

        away_team = str(row[away_col]).strip()
        home_team = str(row[home_col]).strip()

        away_score = _as_int(row.get(away_pts_col))
        home_score = _as_int(row.get(home_pts_col))

        ot_raw = ""
        if ot_col and pd.notna(row.get(ot_col)):
            ot_raw = str(row.get(ot_col)).strip()
        overtime = bool(ot_raw)

        game_id = None
        if box_col and pd.notna(row.get(box_col)):
            game_id = str(row.get(box_col)).strip()
        if not game_id:
            game_id = f"{pd.to_datetime(row[date_col]).date()}|{away_team}|{home_team}"

        notes = None
        if notes_col and pd.notna(row.get(notes_col)):
            notes = str(row.get(notes_col)).strip()

        games.append(
            GameResult(
                date=pd.to_datetime(row[date_col]).date(),
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                overtime=overtime,
                game_id=game_id,
                sport=sport,
                season=season,
                notes=notes,
            )
        )

    return games


def parse_sr_csv(path: str | Path, sport: str | None = None, season: str | None = None) -> List[GameResult]:
    df = pd.read_csv(Path(path))
    return _parse_sr_dataframe(df, sport=sport, season=season)


def parse_sr_html(path: str | Path, sport: str | None = None, season: str | None = None) -> List[GameResult]:
    tables = pd.read_html(Path(path))
    last_error: ValueError | None = None
    for table in tables:
        try:
            return _parse_sr_dataframe(table, sport=sport, season=season)
        except ValueError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError("No tables found in HTML input")


def parse_sr_csv_text(text: str, sport: str | None = None, season: str | None = None) -> List[GameResult]:
    df = pd.read_csv(StringIO(text))
    return _parse_sr_dataframe(df, sport=sport, season=season)
