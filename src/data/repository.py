"""SQLite persistence layer for games and model calibration metrics."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Iterable, List

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
    decision_type TEXT,
    game_id TEXT,
    sport TEXT,
    season TEXT,
    division TEXT,
    conference TEXT,
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
    margin_std REAL,
    total_std REAL,
    conditional_sd_intercept REAL,
    conditional_sd_slope REAL,
    margin_mean REAL,
    total_mean REAL,
    backtest_log_loss REAL,
    backtest_brier_score REAL,
    backtest_mae_margin REAL,
    backtest_win_prob_k REAL,
    backtest_run_id TEXT,
    backtest_updated_at TEXT,
    tuned_params_json TEXT,
    tuned_params_metric TEXT,
    tuned_params_updated_at TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(sport, season, model)
);

CREATE TABLE IF NOT EXISTS model_tuned_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    model TEXT NOT NULL,
    metric TEXT NOT NULL,
    run_id TEXT NOT NULL,
    params_json TEXT NOT NULL,
    best_score REAL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(sport, season, model, metric)
);
"""


def init_db(db_path: str | Path) -> None:
    """Create the SQLite database (and tables) if they do not already exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA)
        _ensure_games_columns(conn)
        _ensure_model_metrics_columns(conn)
        conn.commit()


def _ensure_games_columns(conn: sqlite3.Connection) -> None:
    """Backfill columns for older databases that predate new game metadata."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
    if "division" not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN division TEXT")
    if "conference" not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN conference TEXT")
    if "decision_type" not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN decision_type TEXT")


def _ensure_model_metrics_columns(conn: sqlite3.Connection) -> None:
    """Backfill columns for older databases that predate new metrics."""
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(model_metrics)").fetchall()
    }
    if "win_prob_k" not in existing:
        conn.execute(
            f"ALTER TABLE model_metrics ADD COLUMN win_prob_k REAL NOT NULL DEFAULT {DEFAULT_WIN_PROB_K}"
        )
    if "base_total" not in existing:
        conn.execute(
            "ALTER TABLE model_metrics ADD COLUMN base_total REAL NOT NULL DEFAULT 0.0"
        )
    if "backtest_log_loss" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_log_loss REAL")
    if "backtest_brier_score" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_brier_score REAL")
    if "backtest_mae_margin" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_mae_margin REAL")
    if "backtest_win_prob_k" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_win_prob_k REAL")
    if "margin_std" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN margin_std REAL")
    if "total_std" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN total_std REAL")
    if "conditional_sd_intercept" not in existing:
        conn.execute(
            "ALTER TABLE model_metrics ADD COLUMN conditional_sd_intercept REAL"
        )
    if "conditional_sd_slope" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN conditional_sd_slope REAL")
    if "margin_mean" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN margin_mean REAL")
    if "total_mean" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN total_mean REAL")
    if "backtest_run_id" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_run_id TEXT")
    if "backtest_updated_at" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_updated_at TEXT")
    if "tuned_params_json" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN tuned_params_json TEXT")
    if "tuned_params_metric" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN tuned_params_metric TEXT")
    if "tuned_params_updated_at" not in existing:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN tuned_params_updated_at TEXT")


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
            g.decision_type,
            g.game_id,
            g.sport,
            g.season,
            g.division,
            g.conference,
            g.notes,
        )
        for g in games
    ]

    with closing(sqlite3.connect(Path(db_path))) as conn:
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
                decision_type,
                game_id,
                sport,
                season,
                division,
                conference,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return conn.total_changes


def load_games(
    db_path: str | Path,
    sport: str | None = None,
    season: str | None = None,
    division: str | None = None,
    conference: str | None = None,
) -> List[GameResult]:
    """Load games from SQLite, filtered by optional sport/season/division/conference."""
    filters = []
    params: list[str] = []

    if sport is not None:
        filters.append("sport = ?")
        params.append(sport)
    if season is not None:
        filters.append("season = ?")
        params.append(season)
    if division is not None:
        filters.append("division = ?")
        params.append(division)
    if conference is not None:
        filters.append("conference = ?")
        params.append(conference)

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
               decision_type,
               game_id,
               sport,
               season,
               division,
               conference,
               notes
        FROM games
        {where_clause}
        ORDER BY date, away_team, home_team
    """

    with closing(sqlite3.connect(Path(db_path))) as conn:
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
            decision_type=row[7],
            game_id=row[8],
            sport=row[9],
            season=row[10],
            division=row[11],
            conference=row[12],
            notes=row[13],
        )
        for row in rows
    ]


def list_sports(db_path: str | Path) -> List[str]:
    """Return distinct sports stored in the games table."""
    with closing(sqlite3.connect(Path(db_path))) as conn:
        rows = conn.execute(
            "SELECT DISTINCT sport FROM games WHERE sport IS NOT NULL ORDER BY sport"
        ).fetchall()
    return [row[0] for row in rows]


def list_seasons(db_path: str | Path, sport: str) -> List[str]:
    """Return distinct seasons for a given sport."""
    with closing(sqlite3.connect(Path(db_path))) as conn:
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
    margin_std: float | None = None,
    total_std: float | None = None,
    conditional_sd_intercept: float | None = None,
    conditional_sd_slope: float | None = None,
    margin_mean: float | None = None,
    total_mean: float | None = None,
) -> None:
    """Persist calibration metrics produced by the ranking step."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO model_metrics (
                sport,
                season,
                model,
                home_advantage,
                model_error,
                win_prob_k,
                base_total,
                margin_std,
                total_std,
                conditional_sd_intercept,
                conditional_sd_slope,
                margin_mean,
                total_mean,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(sport, season, model) DO UPDATE SET
                home_advantage = excluded.home_advantage,
                model_error = excluded.model_error,
                win_prob_k = excluded.win_prob_k,
                base_total = excluded.base_total,
                margin_std = excluded.margin_std,
                total_std = excluded.total_std,
                conditional_sd_intercept = excluded.conditional_sd_intercept,
                conditional_sd_slope = excluded.conditional_sd_slope,
                margin_mean = excluded.margin_mean,
                total_mean = excluded.total_mean,
                updated_at = datetime('now')
            """,
            (
                sport,
                season,
                model,
                home_advantage,
                model_error,
                win_prob_k,
                base_total,
                margin_std,
                total_std,
                conditional_sd_intercept,
                conditional_sd_slope,
                margin_mean,
                total_mean,
            ),
        )
        conn.commit()


def save_backtest_metrics(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    log_loss: float | None,
    brier_score: float | None,
    mae_margin: float | None,
    win_prob_k: float | None,
    run_id: str,
) -> None:
    """Persist backtest calibration metrics for a sport/season/model combination."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT home_advantage,
                   model_error,
                   win_prob_k,
                   base_total,
                   margin_std,
                   total_std,
                   conditional_sd_intercept,
                   conditional_sd_slope,
                   margin_mean,
                   total_mean,
                   backtest_log_loss,
                   backtest_brier_score,
                   backtest_mae_margin,
                   backtest_win_prob_k
            FROM model_metrics
            WHERE sport = ? AND season = ? AND model = ?
            """,
            (sport, season, model),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO model_metrics (
                    sport,
                    season,
                    model,
                    home_advantage,
                    model_error,
                    win_prob_k,
                    base_total,
                    backtest_log_loss,
                    backtest_brier_score,
                    backtest_mae_margin,
                    backtest_win_prob_k,
                    backtest_run_id,
                    backtest_updated_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                """,
                (
                    sport,
                    season,
                    model,
                    0.0,
                    0.0,
                    DEFAULT_WIN_PROB_K,
                    0.0,
                    log_loss,
                    brier_score,
                    mae_margin,
                    win_prob_k,
                    run_id,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE model_metrics
                SET backtest_log_loss = ?,
                    backtest_brier_score = ?,
                    backtest_mae_margin = ?,
                    backtest_win_prob_k = ?,
                    backtest_run_id = ?,
                    backtest_updated_at = datetime('now')
                WHERE sport = ? AND season = ? AND model = ?
                """,
                (
                    log_loss,
                    brier_score,
                    mae_margin,
                    win_prob_k,
                    run_id,
                    sport,
                    season,
                    model,
                ),
            )
        conn.commit()


def load_model_metrics(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
) -> dict[str, float | None] | None:
    """Load calibration metrics for a sport/season/model combination."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT home_advantage,
                   model_error,
                   win_prob_k,
                   base_total,
                   margin_std,
                   total_std,
                   conditional_sd_intercept,
                   conditional_sd_slope,
                   margin_mean,
                   total_mean,
                   backtest_log_loss,
                   backtest_brier_score,
                   backtest_mae_margin,
                   backtest_win_prob_k
            FROM model_metrics
            WHERE sport = ? AND season = ? AND model = ?
            """,
            (sport, season, model),
        ).fetchone()
    if row is None:
        return None
    metrics: dict[str, float | None] = {
        "home_advantage": float(row[0]),
        "model_error": float(row[1]),
        "win_prob_k": float(row[2]),
        "base_total": float(row[3]),
    }
    extra_values = {
        "margin_std": row[4],
        "total_std": row[5],
        "conditional_sd_intercept": row[6],
        "conditional_sd_slope": row[7],
        "margin_mean": row[8],
        "total_mean": row[9],
    }
    if any(value is not None for value in extra_values.values()):
        metrics.update(extra_values)
    backtest_values = {
        "backtest_log_loss": row[10],
        "backtest_brier_score": row[11],
        "backtest_mae_margin": row[12],
        "backtest_win_prob_k": row[13],
    }
    if any(value is not None for value in backtest_values.values()):
        metrics.update(backtest_values)
    return metrics


def save_tuned_params(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    metric: str,
    run_id: str,
    params_json: str,
    best_score: float | None,
) -> None:
    """Persist the best tuned parameters for a model/metric combination."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO model_tuned_params (
                sport,
                season,
                model,
                metric,
                run_id,
                params_json,
                best_score,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(sport, season, model, metric) DO UPDATE SET
                run_id = excluded.run_id,
                params_json = excluded.params_json,
                best_score = excluded.best_score,
                updated_at = datetime('now')
            """,
            (sport, season, model, metric, run_id, params_json, best_score),
        )
        conn.commit()


def set_active_tuned_params(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    metric: str,
) -> None:
    """Promote tuned params for a metric into the active model_metrics row."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT params_json
            FROM model_tuned_params
            WHERE sport = ? AND season = ? AND model = ? AND metric = ?
            """,
            (sport, season, model, metric),
        ).fetchone()
        if row is None or row[0] is None:
            raise ValueError(
                "No tuned params found for "
                f"sport={sport}, season={season}, model={model}, metric={metric}"
            )
        params_json = row[0]
        existing = conn.execute(
            """
            SELECT 1
            FROM model_metrics
            WHERE sport = ? AND season = ? AND model = ?
            """,
            (sport, season, model),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO model_metrics (
                    sport,
                    season,
                    model,
                    home_advantage,
                    model_error,
                    win_prob_k,
                    base_total,
                    tuned_params_json,
                    tuned_params_metric,
                    tuned_params_updated_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                """,
                (
                    sport,
                    season,
                    model,
                    0.0,
                    0.0,
                    DEFAULT_WIN_PROB_K,
                    0.0,
                    params_json,
                    metric,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE model_metrics
                SET tuned_params_json = ?,
                    tuned_params_metric = ?,
                    tuned_params_updated_at = datetime('now')
                WHERE sport = ? AND season = ? AND model = ?
                """,
                (params_json, metric, sport, season, model),
            )
        conn.commit()


def load_active_tuned_params(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
) -> dict | None:
    """Load the currently active tuned params for a model."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT tuned_params_json
            FROM model_metrics
            WHERE sport = ? AND season = ? AND model = ?
            """,
            (sport, season, model),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    data = json.loads(row[0])
    if not isinstance(data, dict):
        raise ValueError("Stored tuned params must be a JSON object.")
    return data


def load_tuned_params_for_metric(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    metric: str,
) -> dict | None:
    """Load tuned params for a specific metric."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT params_json
            FROM model_tuned_params
            WHERE sport = ? AND season = ? AND model = ? AND metric = ?
            """,
            (sport, season, model, metric),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    data = json.loads(row[0])
    if not isinstance(data, dict):
        raise ValueError("Stored tuned params must be a JSON object.")
    return data


def load_active_tuned_metric(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
) -> str | None:
    """Return the metric name for the active tuned params."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT tuned_params_metric
            FROM model_metrics
            WHERE sport = ? AND season = ? AND model = ?
            """,
            (sport, season, model),
        ).fetchone()
    return row[0] if row and row[0] is not None else None
