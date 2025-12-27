from __future__ import annotations

import sqlite3
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
    *,
    sport: str | None = None,
    season: str | None = None,
) -> list[dict[str, Any]]:
    init_db(db_path)
    query = (
        "SELECT date, visitor_team, visitor_pts, home_team, home_pts, ot, game_id, sport, season "
        "FROM games"
    )
    params: list[Any] = []
    clauses: list[str] = []
    if sport is not None:
        clauses.append("sport = ?")
        params.append(sport)
    if season is not None:
        clauses.append("season = ?")
        params.append(season)
    if clauses:
        query = f"{query} WHERE {' AND '.join(clauses)}"
    query = f"{query} ORDER BY date, game_id"

    with sqlite3.connect(Path(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]
