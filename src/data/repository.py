"""SQLite persistence layer for games and model calibration metrics."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List

from config import DEFAULT_WIN_PROB_K
from ingest.schema import GameResult
from . import teams as team_repo
from .migrations import apply_migrations


# Schema is small and append-only; migrations are handled via helper checks.
SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    start_time TEXT,
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

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    UNIQUE(sport, season, canonical_name)
);

CREATE TABLE IF NOT EXISTS team_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    alias_text TEXT NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY(team_id) REFERENCES teams(id),
    UNIQUE(sport, season, alias_text)
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
    backtest_mae_total REAL,
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

CREATE TABLE IF NOT EXISTS model_market_tuning_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    model TEXT NOT NULL,
    market TEXT NOT NULL,
    metric_optimized TEXT NOT NULL,
    run_id TEXT NOT NULL,
    best_score REAL,
    best_params_json TEXT,
    summary_metrics_json TEXT,
    started_at TEXT,
    finished_at TEXT,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_market_tuning_runs
    ON model_market_tuning_runs(sport, season, model, market, metric_optimized);

CREATE TABLE IF NOT EXISTS model_market_active_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    model TEXT NOT NULL,
    market TEXT NOT NULL,
    params_json TEXT NOT NULL,
    source_run_id TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(sport, season, model, market)
);

CREATE TABLE IF NOT EXISTS ensemble_market_tuning_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    market TEXT NOT NULL,
    ensemble_id TEXT NOT NULL,
    metric_optimized TEXT NOT NULL,
    run_id TEXT NOT NULL,
    best_score REAL,
    weights_json TEXT,
    models_json TEXT,
    summary_metrics_json TEXT,
    started_at TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_ensemble_market_tuning_runs
    ON ensemble_market_tuning_runs(sport, season, market, ensemble_id);

CREATE TABLE IF NOT EXISTS ensemble_market_active_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    market TEXT NOT NULL,
    ensemble_id TEXT NOT NULL,
    weights_json TEXT NOT NULL,
    models_json TEXT NOT NULL,
    source_run_id TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(sport, season, market, ensemble_id)
);

CREATE TABLE IF NOT EXISTS market_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    game_id TEXT NOT NULL,
    game_date TEXT NOT NULL,
    market_type TEXT NOT NULL,
    selection_team_id INTEGER,
    selection TEXT,
    line REAL,
    odds INTEGER NOT NULL,
    book TEXT,
    imported_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_market_lines_key
    ON market_lines(
        sport,
        season,
        game_id,
        market_type,
        COALESCE(selection_team_id, -1),
        COALESCE(selection, ''),
        COALESCE(line, -9999),
        COALESCE(book, '')
    );

CREATE TABLE IF NOT EXISTS market_line_import_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    row_data TEXT NOT NULL,
    failure_reason TEXT NOT NULL,
    failure_details TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db(db_path: str | Path) -> None:
    """Create the SQLite database (and tables) if they do not already exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA)
        apply_migrations(conn)
        conn.commit()


def save_games(db_path: str | Path, games: Iterable[GameResult]) -> int:
    """Upsert game rows into the SQLite database and return change count."""
    init_db(db_path)
    rows = [
        (
            g.date.isoformat(),
            g.start_time.isoformat() if g.start_time else None,
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
        team_repo.ensure_teams_for_games(conn, games)
        conn.executemany(
            """
            INSERT OR REPLACE INTO games (
                date,
                start_time,
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
               start_time,
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
        ORDER BY COALESCE(start_time, date), away_team, home_team
    """

    with closing(sqlite3.connect(Path(db_path))) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        GameResult(
            date=date.fromisoformat(row[0]),
            start_time=datetime.fromisoformat(row[1]) if row[1] else None,
            home_team=row[2],
            away_team=row[3],
            home_score=row[4],
            away_score=row[5],
            neutral=bool(row[6]),
            overtime=bool(row[7]),
            decision_type=row[8],
            game_id=row[9],
            sport=row[10],
            season=row[11],
            division=row[12],
            conference=row[13],
            notes=row[14],
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
    # Coerce required numeric fields to sane defaults to avoid NOT NULL DB errors
    if home_advantage is None or not math.isfinite(float(home_advantage)):
        home_advantage = 0.0
    if model_error is None or not math.isfinite(float(model_error)):
        # Use a conservative default error when unknown
        model_error = 1.0
    if win_prob_k is None or not math.isfinite(float(win_prob_k)):
        win_prob_k = DEFAULT_WIN_PROB_K
    if base_total is None or not math.isfinite(float(base_total)):
        base_total = 0.0
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
    mae_total: float | None,
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
                   backtest_mae_total,
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
                    backtest_mae_total,
                    backtest_win_prob_k,
                    backtest_run_id,
                    backtest_updated_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
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
                    mae_total,
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
                    backtest_mae_total = ?,
                    backtest_win_prob_k = ?,
                    backtest_run_id = ?,
                    backtest_updated_at = datetime('now')
                WHERE sport = ? AND season = ? AND model = ?
                """,
                (
                    log_loss,
                    brier_score,
                    mae_margin,
                    mae_total,
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
                   backtest_mae_total,
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
        "backtest_mae_total": row[13],
        "backtest_win_prob_k": row[14],
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


def save_model_market_tuning_run(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    market: str,
    metric_optimized: str,
    run_id: str,
    best_score: float | None,
    best_params_json: str | None,
    summary_metrics_json: str | None,
    started_at: str | None,
    finished_at: str | None,
    notes: str | None = None,
) -> None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO model_market_tuning_runs (
                sport,
                season,
                model,
                market,
                metric_optimized,
                run_id,
                best_score,
                best_params_json,
                summary_metrics_json,
                started_at,
                finished_at,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sport,
                season,
                model,
                market,
                metric_optimized,
                run_id,
                best_score,
                best_params_json,
                summary_metrics_json,
                started_at,
                finished_at,
                notes,
            ),
        )
        conn.commit()


def set_active_model_market_params(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    market: str,
    params: dict,
    source_run_id: str | None,
) -> None:
    if not isinstance(params, dict):
        raise ValueError("Active market params must be a JSON object (dict).")
    init_db(db_path)
    params_json = json.dumps(params, sort_keys=True)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO model_market_active_params (
                sport,
                season,
                model,
                market,
                params_json,
                source_run_id,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(sport, season, model, market) DO UPDATE SET
                params_json = excluded.params_json,
                source_run_id = excluded.source_run_id,
                updated_at = datetime('now')
            """,
            (sport, season, model, market, params_json, source_run_id),
        )
        conn.commit()


def get_active_model_market_params(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    market: str,
) -> dict | None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT params_json
            FROM model_market_active_params
            WHERE sport = ? AND season = ? AND model = ? AND market = ?
            """,
            (sport, season, model, market),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    data = json.loads(row[0])
    if not isinstance(data, dict):
        raise ValueError("Stored market params must be a JSON object.")
    return data


def get_active_model_market_params_source(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    market: str,
) -> str | None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT source_run_id
            FROM model_market_active_params
            WHERE sport = ? AND season = ? AND model = ? AND market = ?
            """,
            (sport, season, model, market),
        ).fetchone()
    return row[0] if row and row[0] is not None else None


def save_ensemble_market_tuning_run(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
    metric_optimized: str,
    run_id: str,
    best_score: float | None,
    weights_json: str | None,
    models_json: str | None,
    summary_metrics_json: str | None,
    started_at: str | None,
    finished_at: str | None,
) -> None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO ensemble_market_tuning_runs (
                sport,
                season,
                market,
                ensemble_id,
                metric_optimized,
                run_id,
                best_score,
                weights_json,
                models_json,
                summary_metrics_json,
                started_at,
                finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sport,
                season,
                market,
                ensemble_id,
                metric_optimized,
                run_id,
                best_score,
                weights_json,
                models_json,
                summary_metrics_json,
                started_at,
                finished_at,
            ),
        )
        conn.commit()


def set_active_ensemble_market_weights(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
    weights_json: str,
    models_json: str,
    source_run_id: str | None,
) -> None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO ensemble_market_active_weights (
                sport,
                season,
                market,
                ensemble_id,
                weights_json,
                models_json,
                source_run_id,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(sport, season, market, ensemble_id) DO UPDATE SET
                weights_json = excluded.weights_json,
                models_json = excluded.models_json,
                source_run_id = excluded.source_run_id,
                updated_at = datetime('now')
            """,
            (sport, season, market, ensemble_id, weights_json, models_json, source_run_id),
        )
        conn.commit()


def get_active_ensemble_market_weights(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
) -> dict | None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT weights_json
            FROM ensemble_market_active_weights
            WHERE sport = ? AND season = ? AND market = ? AND ensemble_id = ?
            """,
            (sport, season, market, ensemble_id),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    data = json.loads(row[0])
    if not isinstance(data, dict):
        raise ValueError("Stored ensemble weights must be a JSON object.")
    return data


def get_active_ensemble_market_weights_source(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
) -> str | None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT source_run_id
            FROM ensemble_market_active_weights
            WHERE sport = ? AND season = ? AND market = ? AND ensemble_id = ?
            """,
            (sport, season, market, ensemble_id),
        ).fetchone()
    return row[0] if row and row[0] is not None else None


def load_best_model_market_tuning_params_by_optimized_metric(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    market: str,
    metric_optimized: str,
) -> tuple[dict | None, str | None]:
    """Load the best tuned params (best_score) for a model+market by optimized metric.

    Returns (params_dict, run_id) or (None, None) when not found.
    """
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT best_params_json, run_id
            FROM model_market_tuning_runs
            WHERE sport = ? AND season = ? AND model = ? AND market = ? AND metric_optimized = ?
            ORDER BY best_score ASC
            LIMIT 1
            """,
            (sport, season, model, market, metric_optimized),
        ).fetchone()
    if row is None or row[0] is None:
        return None, None
    data = json.loads(row[0])
    if not isinstance(data, dict):
        raise ValueError("Stored model market tuned params must be a JSON object.")
    return data, row[1]


def load_model_market_tuning_run_by_run_id(
    db_path: str | Path, *, run_id: str
) -> tuple[dict | None, str | None]:
    """Load a model_market_tuning_runs entry by run_id. Returns (params_dict, run_id) or (None,None)."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT best_params_json, run_id
            FROM model_market_tuning_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if row is None or row[0] is None:
        return None, None
    data = json.loads(row[0])
    if not isinstance(data, dict):
        raise ValueError("Stored model market tuned params must be a JSON object.")
    return data, row[1]


def load_best_ensemble_market_tuning_weights_by_optimized_metric(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
    metric_optimized: str,
) -> tuple[dict | None, str | None]:
    """Load best ensemble weights for a market+ensemble by optimized metric.

    Returns (weights_dict, run_id) or (None, None).
    """
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT weights_json, run_id
            FROM ensemble_market_tuning_runs
            WHERE sport = ? AND season = ? AND market = ? AND ensemble_id = ? AND metric_optimized = ?
            ORDER BY best_score ASC
            LIMIT 1
            """,
            (sport, season, market, ensemble_id, metric_optimized),
        ).fetchone()
    if row is None or row[0] is None:
        return None, None
    data = json.loads(row[0])
    if not isinstance(data, dict):
        raise ValueError("Stored ensemble weights must be a JSON object.")
    return data, row[1]


def load_ensemble_market_tuning_run_by_run_id(
    db_path: str | Path, *, run_id: str
) -> tuple[dict | None, str | None]:
    """Load an ensemble_market_tuning_runs entry by run_id. Returns (weights_dict, run_id) or (None,None)."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT weights_json, run_id
            FROM ensemble_market_tuning_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if row is None or row[0] is None:
        return None, None
    data = json.loads(row[0])
    if not isinstance(data, dict):
        raise ValueError("Stored ensemble weights must be a JSON object.")
    return data, row[1]
