"""Parsers for Sports-Reference schedule/results exports."""

from __future__ import annotations

import csv
import re
from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd

from ingest.schema import GameResult


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim and lowercase column headers for more forgiving matching."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _rename_duplicate_pts(df: pd.DataFrame) -> pd.DataFrame:
    """Disambiguate duplicate PTS columns (Visitor/Home)."""
    cols = list(df.columns)
    pts_indices = [i for i, c in enumerate(cols) if c == "pts"]
    if len(pts_indices) >= 2:
        cols[pts_indices[0]] = "pts_away"
        cols[pts_indices[1]] = "pts_home"
        df = df.copy()
        df.columns = cols
    return df


def _find_column(df: pd.DataFrame, *names: str) -> str | None:
    """Return the first matching column name from a list of aliases."""
    for name in names:
        if name in df.columns:
            return name
    return None


def _as_int(value) -> int | None:
    """Convert numeric-like fields to integers, returning None on failure."""
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
    """Detect the away/home points columns regardless of naming conventions."""
    if "pts_away" in df.columns and "pts_home" in df.columns:
        return "pts_away", "pts_home"
    pts_cols = [c for c in df.columns if c == "pts" or c.startswith("pts.")]
    if len(pts_cols) >= 2:
        return pts_cols[0], pts_cols[1]

    away_pts = _find_column(
        df,
        "visitor pts",
        "visitor_pts",
        "away pts",
        "away_pts",
        "pts_visitor",
        "ptsaway",
        "pts away",
    )
    home_pts = _find_column(
        df, "home pts", "home_pts", "pts_home", "ptshome", "pts home"
    )
    return away_pts, home_pts


def _parse_sr_dataframe(
    df: pd.DataFrame,
    sport: str | None = None,
    season: str | None = None,
) -> List[GameResult]:
    """Convert a Sports-Reference dataframe into structured GameResult rows."""
    df = _normalize_columns(df)
    df = _rename_duplicate_pts(df)

    date_col = _find_column(df, "date", "game date")
    away_col = _find_column(
        df, "visitor/neutral", "visitor", "away", "away/neutral", "road", "road team"
    )
    home_col = _find_column(df, "home/neutral", "home", "home team")
    ot_col = _find_column(df, "ot", "overtime")
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
        # Validate required fields and parse any available scoring data.
        raw_date = row.get(date_col)
        if (
            pd.isna(raw_date)
            or pd.isna(row.get(away_col))
            or pd.isna(row.get(home_col))
        ):
            continue

        parsed_date = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(parsed_date):
            continue

        away_team = str(row[away_col]).strip()
        home_team = str(row[home_col]).strip()

        away_score = _as_int(row.get(away_pts_col)) if away_pts_col else None
        home_score = _as_int(row.get(home_pts_col)) if home_pts_col else None

        ot_raw = ""
        if ot_col and pd.notna(row.get(ot_col)):
            ot_raw = str(row.get(ot_col)).strip()
        overtime = bool(ot_raw)

        game_id = None
        if box_col and pd.notna(row.get(box_col)):
            raw_game_id = str(row.get(box_col)).strip()
            if not looks_like_tip_time(raw_game_id):
                game_id = raw_game_id
        if not game_id:
            game_id = f"{parsed_date.date()}|{away_team}|{home_team}"

        notes = None
        if notes_col and pd.notna(row.get(notes_col)):
            notes = str(row.get(notes_col)).strip()

        games.append(
            GameResult(
                date=parsed_date.date(),
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


def looks_like_tip_time(value: str) -> bool:
    if not value:
        return False
    time_re = re.compile(r"^\s*\d{1,2}:\d{2}\s*[ap]m?\s*$", re.IGNORECASE)
    return bool(time_re.match(value))


def load_sr_csv_lenient(text: str) -> pd.DataFrame:
    """
    Read Sports-Reference CSV exports that may include an unlabeled start-time column.
    If a row has one more column than the header and the second field looks like a tip time (e.g., '7:00p'),
    drop that field so Visitor/Home/PTS stay aligned. Rows with missing columns are padded.
    """
    reader = csv.reader(StringIO(text))
    header: List[str] = []
    rows: List[List[str]] = []
    time_re = re.compile(r"\d{1,2}:\d{2}[ap]", re.IGNORECASE)

    try:
        header = next(reader)
    except StopIteration:
        return pd.DataFrame()

    for row in reader:
        # Skip empty rows
        if not row or all(cell.strip() == "" for cell in row):
            continue

        if len(row) > len(header):
            if (
                len(row) == len(header) + 1
                and len(row) >= 2
                and time_re.search(row[1] or "")
            ):
                # Drop unlabeled start time to realign columns
                row = row[:1] + row[2:]
            else:
                row = row[: len(header)]
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))

        rows.append(row)

    return pd.DataFrame(rows, columns=header)


def parse_sr_csv(
    path: str | Path, sport: str | None = None, season: str | None = None
) -> List[GameResult]:
    """Parse a Sports-Reference CSV file from disk."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    df = load_sr_csv_lenient(text)
    return _parse_sr_dataframe(df, sport=sport, season=season)


def parse_sr_html(
    path: str | Path, sport: str | None = None, season: str | None = None
) -> List[GameResult]:
    """Parse a Sports-Reference HTML schedule/results table."""
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    tables = pd.read_html(StringIO(html), flavor="bs4")
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


def parse_sr_csv_text(
    text: str, sport: str | None = None, season: str | None = None
) -> List[GameResult]:
    """Parse pasted Sports-Reference CSV text."""
    df = load_sr_csv_lenient(text)
    return _parse_sr_dataframe(df, sport=sport, season=season)
