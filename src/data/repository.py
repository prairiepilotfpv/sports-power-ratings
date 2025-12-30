from __future__ import annotations

"""SQLite persistence layer for games and model calibration metrics."""

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Iterable, List

from config import DEFAULT_WIN_PROB_K
from ingest.schema import GameResult


# Schema is small and append-only; migrations are handled via helper checks.
SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    neutral INTEGER NOT NULL DEFAULT 0,
    overtime INTEGER NOT NULL DEFAULT 0,
    game_id TEXT,
    sport TEXT,
    season TEXT,
    notes TEXT,
    UNIQUE(game_id, sport, season)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    model TEXT NOT NULL,
    home_advantage REAL NOT NULL,
    model_error REAL NOT NULL,
    win_prob_k REAL NOT NULL,
    base_total REAL NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(sport, season, model)
);
"""


def init_db(db_path: str | Path) -> None:
    """Create the SQLite database (and tables) if they do not already exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        _ensure_model_metrics_columns(conn)
        conn.commit()


def _ensure_model_metrics_columns(conn: sqlite3.Connection) -> None:
    """Backfill columns for older databases that predate new metrics."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(model_metrics)").fetchall()}
    if "win_prob_k" not in existing:
        conn.execute(
            f"ALTER TABLE model_metrics ADD COLUMN win_prob_k REAL NOT NULL DEFAULT {DEFAULT_WIN_PROB_K}"
        )
    if "base_total" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN base_total REAL NOT NULL DEFAULT 0.0")


def save_games(db_path: str | Path, games: Iterable[GameResult]) -> int:
    """Upsert game rows into the SQLite database and return change count."""
    init_db(db_path)
    rows = [
        (
            g.date.isoformat(),
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            1 if g.neutral else 0,
            1 if g.overtime else 0,
            g.game_id,
            g.sport,
            g.season,
            g.notes,
        )
        for g in games
    ]

    with sqlite3.connect(Path(db_path)) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO games (
                date,
                home_team,
                away_team,
                home_score,
                away_score,
                neutral,
                overtime,
                game_id,
                sport,
                season,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    """Load games from SQLite, filtered by optional sport/season."""
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
               home_team,
               away_team,
               home_score,
               away_score,
               neutral,
               overtime,
               game_id,
               sport,
               season,
               notes
        FROM games
        {where_clause}
        ORDER BY date, away_team, home_team
    """

    with sqlite3.connect(Path(db_path)) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        GameResult(
            date=date.fromisoformat(row[0]),
            home_team=row[1],
            away_team=row[2],
            home_score=row[3],
            away_score=row[4],
            neutral=bool(row[5]),
            overtime=bool(row[6]),
            game_id=row[7],
            sport=row[8],
            season=row[9],
            notes=row[10],
        )
        for row in rows
    ]


def list_sports(db_path: str | Path) -> List[str]:
    """Return distinct sports stored in the games table."""
    with sqlite3.connect(Path(db_path)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT sport FROM games WHERE sport IS NOT NULL ORDER BY sport"
        ).fetchall()
    return [row[0] for row in rows]


def list_seasons(db_path: str | Path, sport: str) -> List[str]:
    """Return distinct seasons for a given sport."""
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


def save_model_metrics(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    home_advantage: float,
    model_error: float,
    win_prob_k: float,
    base_total: float,
) -> None:
    """Persist calibration metrics produced by the ranking step."""
    init_db(db_path)
    with sqlite3.connect(Path(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO model_metrics (
                sport,
                season,
                model,
                home_advantage,
                model_error,
                win_prob_k,
                base_total,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (sport, season, model, home_advantage, model_error, win_prob_k, base_total),
        )
        conn.commit()


def load_model_metrics(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
) -> dict[str, float] | None:
    """Load calibration metrics for a sport/season/model combination."""
    init_db(db_path)
    with sqlite3.connect(Path(db_path)) as conn:
        row = conn.execute(
            """
            SELECT home_advantage, model_error, win_prob_k, base_total
            FROM model_metrics
            WHERE sport = ? AND season = ? AND model = ?
            """,
            (sport, season, model),
        ).fetchone()
    if row is None:
        return None
    return {
        "home_advantage": float(row[0]),
        "model_error": float(row[1]),
        "win_prob_k": float(row[2]),
        "base_total": float(row[3]),
    }
