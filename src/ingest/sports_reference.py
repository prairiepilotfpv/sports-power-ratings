from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

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
    pts_cols = [c for c in df.columns if c == "pts"]
    if len(pts_cols) >= 2:
        return pts_cols[0], pts_cols[1]

    visitor_pts = _find_column(df, "visitor pts", "visitor_pts")
    home_pts = _find_column(df, "home pts", "home_pts")
    return visitor_pts, home_pts


def parse_sr_csv(path: str | Path, sport: str | None = None, season: str | None = None) -> List[GameResult]:
    df = pd.read_csv(Path(path))
    df = _normalize_columns(df)

    date_col = _find_column(df, "date")
    visitor_col = _find_column(df, "visitor/neutral", "visitor")
    home_col = _find_column(df, "home/neutral", "home")
    ot_col = _find_column(df, "ot")
    box_col = _find_column(df, "box score", "boxscore", "box")
    visitor_pts_col, home_pts_col = _resolve_pts_columns(df)

    if not date_col or not visitor_col or not home_col:
        missing = [name for name, col in {
            "date": date_col,
            "visitor": visitor_col,
            "home": home_col,
        }.items() if col is None]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    games: List[GameResult] = []
    for _, row in df.iterrows():
        if pd.isna(row.get(date_col)) or pd.isna(row.get(visitor_col)) or pd.isna(row.get(home_col)):
            continue

        visitor_team = str(row[visitor_col]).strip()
        home_team = str(row[home_col]).strip()

        visitor_pts = _as_int(row.get(visitor_pts_col))
        home_pts = _as_int(row.get(home_pts_col))

        ot_raw = ""
        if ot_col and pd.notna(row.get(ot_col)):
            ot_raw = str(row.get(ot_col)).strip()
        ot = bool(ot_raw)

        game_id = None
        if box_col and pd.notna(row.get(box_col)):
            game_id = str(row.get(box_col)).strip()
        if not game_id:
            game_id = f"{pd.to_datetime(row[date_col]).date()}|{visitor_team}|{home_team}"

        games.append(
            GameResult(
                date=pd.to_datetime(row[date_col]).date(),
                visitor_team=visitor_team,
                visitor_pts=visitor_pts,
                home_team=home_team,
                home_pts=home_pts,
                ot=ot,
                game_id=game_id,
                sport=sport,
                season=season,
            )
        )

    return games
