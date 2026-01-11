from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from src.data import migrations
from src.data import repository as repo


def _create_legacy_core_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE games (
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

        CREATE TABLE model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sport TEXT NOT NULL,
            season TEXT NOT NULL,
            model TEXT NOT NULL,
            home_advantage REAL NOT NULL,
            model_error REAL NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(sport, season, model)
        );
        """
    )


def test_init_db_upgrades_legacy_core_schema() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "legacy.db"
        with closing(sqlite3.connect(db_path)) as conn:
            _create_legacy_core_schema(conn)
            conn.commit()

        repo.init_db(db_path)

        with closing(sqlite3.connect(db_path)) as conn:
            game_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(games)")
            }
            metric_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(model_metrics)")
            }
            schema_version = conn.execute(
                "SELECT schema_version FROM schema_meta WHERE id = 1"
            ).fetchone()

        assert {"division", "conference", "decision_type"}.issubset(game_cols)
        assert {"win_prob_k", "base_total", "tuned_params_json"}.issubset(metric_cols)
        assert schema_version is not None
        assert int(schema_version[0]) == migrations.LATEST_SCHEMA_VERSION
