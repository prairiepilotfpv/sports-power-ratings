from __future__ import annotations

import sqlite3
from pathlib import Path

from data.repository import init_db, load_model_metrics, save_model_metrics


def test_save_and_load_model_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "games.db"
    save_model_metrics(
        db_path,
        sport="nba",
        season="2024-25",
        model="bradley-terry",
        home_advantage=2.5,
        model_error=8.0,
        win_prob_k=12.0,
        base_total=210.0,
    )

    metrics = load_model_metrics(
        db_path,
        sport="nba",
        season="2024-25",
        model="bradley-terry",
    )

    assert metrics == {
        "home_advantage": 2.5,
        "model_error": 8.0,
        "win_prob_k": 12.0,
        "base_total": 210.0,
    }


def test_init_db_adds_missing_model_metrics_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_metrics (
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
        conn.commit()

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(model_metrics)").fetchall()}

    assert {"win_prob_k", "base_total"}.issubset(cols)
