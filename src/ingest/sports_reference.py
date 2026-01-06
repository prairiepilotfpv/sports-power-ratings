"""Parsers for Sports-Reference schedule/results exports."""

from __future__ import annotations

import csv
import re
from io import StringIO
from pathlib import Path
from collections import defaultdict
from typing import Iterable, List

import pandas as pd

from ingest.schema import GameResult

_NHL_TEAM_ALIASES = {
    "anaheim ducks": "ANA",
    "arizona coyotes": "ARI",
    "boston bruins": "BOS",
    "buffalo sabres": "BUF",
    "calgary flames": "CGY",
    "carolina hurricanes": "CAR",
    "chicago blackhawks": "CHI",
    "colorado avalanche": "COL",
    "columbus blue jackets": "CBJ",
    "dallas stars": "DAL",
    "detroit red wings": "DET",
    "edmonton oilers": "EDM",
    "florida panthers": "FLA",
    "los angeles kings": "LAK",
    "la kings": "LAK",
    "minnesota wild": "MIN",
    "montreal canadiens": "MTL",
    "nashville predators": "NSH",
    "new jersey devils": "NJD",
    "new york islanders": "NYI",
    "new york rangers": "NYR",
    "ottawa senators": "OTT",
    "philadelphia flyers": "PHI",
    "pittsburgh penguins": "PIT",
    "san jose sharks": "SJS",
    "seattle kraken": "SEA",
    "st louis blues": "STL",
    "st. louis blues": "STL",
    "tampa bay lightning": "TBL",
    "toronto maple leafs": "TOR",
    "utah mammoth": "UTA",
    "utah hockey club": "UTA",
    "vancouver canucks": "VAN",
    "vegas golden knights": "VGK",
    "washington capitals": "WSH",
    "winnipeg jets": "WPG",
}

_NHL_ABBREVIATIONS = {
    "ANA",
    "ARI",
    "BOS",
    "BUF",
    "CGY",
    "CAR",
    "CHI",
    "COL",
    "CBJ",
    "DAL",
    "DET",
    "EDM",
    "FLA",
    "LAK",
    "MIN",
    "MTL",
    "NSH",
    "NJD",
    "NYI",
    "NYR",
    "OTT",
    "PHI",
    "PIT",
    "SJS",
    "SEA",
    "STL",
    "TBL",
    "TOR",
    "UTA",
    "VAN",
    "VGK",
    "WSH",
    "WPG",
}


def _normalize_team_key(name: str) -> str:
    cleaned = re.sub(r"[.']", "", name).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def _normalize_nhl_team(name: str, unknown: set[str]) -> str:
    if not name:
        unknown.add(name)
        return name
    raw = name.strip()
    upper = raw.upper()
    if upper in _NHL_ABBREVIATIONS:
        return upper
    key = _normalize_team_key(raw)
    mapped = _NHL_TEAM_ALIASES.get(key)
    if mapped is None:
        unknown.add(raw)
        return raw
    return mapped


def _format_unknown_names(names: Iterable[str]) -> str:
    return ", ".join(sorted({name for name in names if name}))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim and lowercase column headers for more forgiving matching."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _rename_duplicate_stat(
    df: pd.DataFrame, label: str, away_name: str, home_name: str
) -> pd.DataFrame:
    """Disambiguate duplicate stat columns (Visitor/Home)."""
    cols = list(df.columns)
    stat_indices = [i for i, c in enumerate(cols) if c == label]
    if len(stat_indices) >= 2:
        cols[stat_indices[0]] = away_name
        cols[stat_indices[1]] = home_name
        df = df.copy()
        df.columns = cols
    return df


def _find_column(df: pd.DataFrame, *names: str) -> str | None:
    """Return the first matching column name from a list of aliases."""
    for name in names:
        if name in df.columns:
            return name
    return None


def _normalize_decision_type(value: object) -> str | None:
    """Normalize overtime/shootout markers like OT, 2OT, or SO."""
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip().upper()
    if not raw:
        return None
    if raw == "SO":
        return "SO"
    if raw == "OT":
        return "OT"
    if raw.endswith("OT") and raw[:-2].isdigit():
        return raw
    return None


def _find_unlabeled_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that look unlabeled in CSV exports."""
    unlabeled = []
    for col in df.columns:
        label = str(col).strip().lower()
        if not label or label.startswith("unnamed"):
            unlabeled.append(col)
    return unlabeled


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
    """Detect the away/home points/goals columns regardless of naming conventions."""
    if "pts_away" in df.columns and "pts_home" in df.columns:
        return "pts_away", "pts_home"
    if "g_away" in df.columns and "g_home" in df.columns:
        return "g_away", "g_home"
    pts_cols = [c for c in df.columns if c == "pts" or c.startswith("pts.")]
    if len(pts_cols) >= 2:
        return pts_cols[0], pts_cols[1]
    g_cols = [c for c in df.columns if c == "g" or c.startswith("g.")]
    if len(g_cols) >= 2:
        return g_cols[0], g_cols[1]

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
    if away_pts is None and home_pts is None:
        away_pts = _find_column(df, "visitor g", "visitor_g", "away g", "away_g", "g_away")
        home_pts = _find_column(df, "home g", "home_g", "g_home", "g home")
    return away_pts, home_pts


def _parse_sr_dataframe(
    df: pd.DataFrame,
    sport: str | None = None,
    season: str | None = None,
    division: str | None = None,
    conference: str | None = None,
) -> List[GameResult]:
    """Convert a Sports-Reference dataframe into structured GameResult rows."""
    df = _normalize_columns(df)
    df = _rename_duplicate_stat(df, "pts", "pts_away", "pts_home")
    df = _rename_duplicate_stat(df, "g", "g_away", "g_home")

    sport_key = (sport or "").lower()
    date_col = _find_column(df, "date", "game date")
    away_col = _find_column(
        df, "visitor/neutral", "visitor", "away", "away/neutral", "road", "road team"
    )
    home_col = _find_column(df, "home/neutral", "home", "home team")
    ot_col = _find_column(df, "ot", "overtime")
    box_col = _find_column(df, "box score", "boxscore", "box")
    notes_col = _find_column(df, "notes")
    away_pts_col, home_pts_col = _resolve_pts_columns(df)
    unlabeled_cols = _find_unlabeled_columns(df)

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
    unknown_teams: set[str] = set()
    game_id_counts: dict[str, int] = defaultdict(int)
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
        if sport_key == "nhl":
            away_team = _normalize_nhl_team(away_team, unknown_teams)
            home_team = _normalize_nhl_team(home_team, unknown_teams)

        away_score = _as_int(row.get(away_pts_col)) if away_pts_col else None
        home_score = _as_int(row.get(home_pts_col)) if home_pts_col else None

        ot_raw = ""
        if ot_col and pd.notna(row.get(ot_col)):
            ot_raw = str(row.get(ot_col)).strip()
        decision_type = _normalize_decision_type(ot_raw)
        if decision_type is None and (sport or "").lower() == "nhl":
            for col in unlabeled_cols:
                candidate = _normalize_decision_type(row.get(col))
                if candidate:
                    decision_type = candidate
                    break
        overtime = decision_type is not None or bool(ot_raw)

        game_id = None
        if sport_key != "nhl" and box_col and pd.notna(row.get(box_col)):
            raw_game_id = str(row.get(box_col)).strip()
            if not looks_like_tip_time(raw_game_id):
                game_id = raw_game_id
        if not game_id:
            if sport_key == "nhl":
                base_id = f"nhl|{parsed_date.date()}|{away_team}|{home_team}"
                game_id_counts[base_id] += 1
                suffix = game_id_counts[base_id]
                game_id = base_id if suffix == 1 else f"{base_id}|{suffix}"
            else:
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
                decision_type=decision_type,
                game_id=game_id,
                sport=sport,
                season=season,
                division=division,
                conference=conference,
                notes=notes,
            )
        )

    if unknown_teams and sport_key == "nhl":
        unknown_list = _format_unknown_names(unknown_teams)
        raise ValueError(f"Unknown NHL team names found: {unknown_list}")

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
        elif (
            len(row) == len(header)
            and len(row) >= 3
            and time_re.fullmatch((row[1] or "").strip())
            # Ensure we are not stripping a real team name; the trailing cell should be a team.
            and not time_re.search(row[2] or "")
            and not time_re.search(row[-1] or "")
        ):
            # Some exports inline an unlabeled tip time as the Visitor value, shifting teams/scores right.
            # Drop the time field and pad so Visitor/Home stay aligned.
            row = row[:1] + row[2:]
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))

        rows.append(row)

    return pd.DataFrame(rows, columns=header)


def parse_sr_csv(
    path: str | Path,
    sport: str | None = None,
    season: str | None = None,
    division: str | None = None,
    conference: str | None = None,
) -> List[GameResult]:
    """Parse a Sports-Reference CSV file from disk."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    df = load_sr_csv_lenient(text)
    return _parse_sr_dataframe(
        df,
        sport=sport,
        season=season,
        division=division,
        conference=conference,
    )


def parse_sr_html(
    path: str | Path,
    sport: str | None = None,
    season: str | None = None,
    division: str | None = None,
    conference: str | None = None,
) -> List[GameResult]:
    """Parse a Sports-Reference HTML schedule/results table."""
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    tables = pd.read_html(StringIO(html), flavor="bs4")
    last_error: ValueError | None = None
    for table in tables:
        try:
            return _parse_sr_dataframe(
                table,
                sport=sport,
                season=season,
                division=division,
                conference=conference,
            )
        except ValueError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError("No tables found in HTML input")


def parse_sr_csv_text(
    text: str,
    sport: str | None = None,
    season: str | None = None,
    division: str | None = None,
    conference: str | None = None,
) -> List[GameResult]:
    """Parse pasted Sports-Reference CSV text."""
    df = load_sr_csv_lenient(text)
    return _parse_sr_dataframe(
        df,
        sport=sport,
        season=season,
        division=division,
        conference=conference,
    )
