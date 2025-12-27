from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from ingest.schema import GameResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    visitor_team TEXT NOT NULL,
    visitor_pts INTEGER,
    home_team TEXT NOT NULL,
    home_pts INTEGER,
    ot INTEGER NOT NULL DEFAULT 0,
    game_id TEXT,
    sport TEXT,
    season TEXT,
    UNIQUE(game_id, sport, season)
);
"""


def init_db(db_path: str | Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def save_games(db_path: str | Path, games: Iterable[GameResult]) -> int:
    init_db(db_path)
    rows = [
        (
            g.date.isoformat(),
            g.visitor_team,
            g.visitor_pts,
            g.home_team,
            g.home_pts,
            1 if g.ot else 0,
            g.game_id,
            g.sport,
            g.season,
        )
        for g in games
    ]

    with sqlite3.connect(Path(db_path)) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO games (
                date,
                visitor_team,
                visitor_pts,
                home_team,
                home_pts,
                ot,
                game_id,
                sport,
                season
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return conn.total_changes


def load_games(
    db_path: str | Path,
    sport: str | None = None,
    season: str | None = None,
) -> List[GameResult]:
    filters = []
    params: list[str] = []

    if sport is not None:
        filters.append("sport = ?")
        params.append(sport)
    if season is not None:
        filters.append("season = ?")
        params.append(season)

    where_clause = ""
    if filters:
        where_clause = f"WHERE {' AND '.join(filters)}"

    query = f"""
        SELECT date,
               visitor_team,
               visitor_pts,
               home_team,
               home_pts,
               ot,
               game_id,
               sport,
               season
        FROM games
        {where_clause}
        ORDER BY date, visitor_team, home_team
    """

    with sqlite3.connect(Path(db_path)) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        GameResult(
            date=date.fromisoformat(row[0]),
            visitor_team=row[1],
            visitor_pts=row[2],
            home_team=row[3],
            home_pts=row[4],
            ot=bool(row[5]),
            game_id=row[6],
            sport=row[7],
            season=row[8],
        )
        for row in rows
    ]


def list_sports(db_path: str | Path) -> List[str]:
    with sqlite3.connect(Path(db_path)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT sport FROM games WHERE sport IS NOT NULL ORDER BY sport"
        ).fetchall()
    return [row[0] for row in rows]


def list_seasons(db_path: str | Path, sport: str) -> List[str]:
    with sqlite3.connect(Path(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT season
            FROM games
            WHERE sport = ? AND season IS NOT NULL
            ORDER BY season
            """,
            (sport,),
        ).fetchall()
    return [row[0] for row in rows]
