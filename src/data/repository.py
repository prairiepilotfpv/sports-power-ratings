"""SQLite persistence layer for games and model calibration metrics."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, List

from config import DEFAULT_WIN_PROB_K
from ingest.schema import GameResult
from . import teams as team_repo
from .migrations import apply_migrations
from src.utils.game_id import make_game_id


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
    params_source TEXT,
    metric_optimized TEXT,
    best_score REAL,
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
    baseline_score REAL,
    weights_json TEXT,
    models_json TEXT,
    selection_run_id TEXT,
    selection_models_json TEXT,
    summary_metrics_json TEXT,
    data_source TEXT,
    db_path TEXT,
    csv_path TEXT,
    asof TEXT,
    window_start TEXT,
    window_end TEXT,
    notes TEXT,
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

CREATE TABLE IF NOT EXISTS ensemble_market_selection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    market TEXT NOT NULL,
    ensemble_id TEXT NOT NULL,
    run_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    window_start TEXT,
    window_end TEXT,
    asof TEXT,
    data_source TEXT NOT NULL,
    db_path TEXT,
    csv_path TEXT,
    scorable_games INTEGER NOT NULL,
    date_min TEXT,
    date_max TEXT,
    candidates_json TEXT NOT NULL,
    selected_json TEXT NOT NULL,
    objective_metric TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    baseline_score REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS ensemble_market_active_selection (
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    market TEXT NOT NULL,
    ensemble_id TEXT NOT NULL,
    active_run_id TEXT NOT NULL,
    activated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(sport, season, market, ensemble_id)
);

CREATE TABLE IF NOT EXISTS ensemble_market_active_tuning (
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    market TEXT NOT NULL,
    ensemble_id TEXT NOT NULL,
    active_run_id TEXT NOT NULL,
    activated_at TEXT NOT NULL DEFAULT (datetime('now')),
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

CREATE TABLE IF NOT EXISTS validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    config_json TEXT,
    summary_json TEXT,
    issues_json TEXT,
    workbook_path TEXT,
    report_path TEXT,
    summary_path TEXT,
    artifacts_dir TEXT
);

CREATE INDEX IF NOT EXISTS idx_validation_runs_sport_season
    ON validation_runs(sport, season, created_at);

CREATE TABLE IF NOT EXISTS bets_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    prediction_date TEXT NOT NULL,
    home_win_prob REAL NOT NULL,
    model_prob REAL,
    edge REAL,
    ev REAL,
    market_type TEXT,
    selection TEXT,
    line REAL,
    market_forecast_source TEXT,
    ml_ensemble_components_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(game_id, sport, season, prediction_date),
    FOREIGN KEY(game_id) REFERENCES games(game_id)
);
"""


def init_db(db_path: str | Path) -> None:
    """Create the SQLite database (and tables) if they do not already exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA)
        apply_migrations(conn)
        _ensure_model_market_active_params_columns(conn)
        _ensure_ensemble_market_tuning_runs_columns(conn)
        conn.commit()


def _ensure_model_market_active_params_columns(conn: sqlite3.Connection) -> None:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(model_market_active_params)")]
    if not cols:
        return
    if "params_source" not in cols:
        conn.execute("ALTER TABLE model_market_active_params ADD COLUMN params_source TEXT")
    if "metric_optimized" not in cols:
        conn.execute("ALTER TABLE model_market_active_params ADD COLUMN metric_optimized TEXT")
    if "best_score" not in cols:
        conn.execute("ALTER TABLE model_market_active_params ADD COLUMN best_score REAL")


def _ensure_ensemble_market_tuning_runs_columns(conn: sqlite3.Connection) -> None:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(ensemble_market_tuning_runs)")]
    if not cols:
        return
    additional_columns = [
        ("baseline_score", "REAL"),
        ("selection_run_id", "TEXT"),
        ("selection_models_json", "TEXT"),
        ("data_source", "TEXT"),
        ("db_path", "TEXT"),
        ("csv_path", "TEXT"),
        ("asof", "TEXT"),
        ("window_start", "TEXT"),
        ("window_end", "TEXT"),
        ("notes", "TEXT"),
    ]
    for name, col_type in additional_columns:
        if name not in cols:
            conn.execute(f"ALTER TABLE ensemble_market_tuning_runs ADD COLUMN {name} {col_type}")


def save_games(db_path: str | Path, games: Iterable[GameResult]) -> int:
    """Upsert game rows into the SQLite database and return change count."""
    init_db(db_path)
    # validate that every game has a non-empty game_id before persisting
    missing = []
    rows = []
    for g in games:
        # attempt to compute missing game_id when possible
        if not g.game_id or str(g.game_id).strip() == "":
            try:
                if g.sport and g.season and g.date and g.away_team and g.home_team:
                    computed = make_game_id(g.sport, g.season, g.date, g.away_team, g.home_team)
                    g = GameResult(
                        date=g.date,
                        start_time=g.start_time,
                        home_team=g.home_team,
                        away_team=g.away_team,
                        home_score=g.home_score,
                        away_score=g.away_score,
                        neutral=g.neutral,
                        overtime=g.overtime,
                        decision_type=g.decision_type,
                        game_id=computed,
                        sport=g.sport,
                        season=g.season,
                        division=g.division,
                        conference=g.conference,
                        notes=g.notes,
                    )
                else:
                    missing.append({
                        "sport": g.sport,
                        "season": g.season,
                        "date": getattr(g.date, "isoformat", lambda: str(g.date))(),
                        "away_team": g.away_team,
                        "home_team": g.home_team,
                    })
            except Exception:
                missing.append({
                    "sport": g.sport,
                    "season": g.season,
                    "date": getattr(g.date, "isoformat", lambda: str(g.date))(),
                    "away_team": g.away_team,
                    "home_team": g.home_team,
                })
        rows.append(
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
        )

    if missing:
        lines = [
            f"sport={m.get('sport')} season={m.get('season')} date={m.get('date')} away={m.get('away_team')} home={m.get('home_team')}"
            for m in missing
        ]
        raise ValueError("Refusing to save games with missing game_id. Rows:\n" + "\n".join(lines))

    with closing(sqlite3.connect(Path(db_path))) as conn:
        team_repo.ensure_teams_for_games(conn, games)
        conn.executemany(
            """
            INSERT INTO games (
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
            ON CONFLICT(game_id, sport, season) DO UPDATE SET
                date = excluded.date,
                start_time = COALESCE(excluded.start_time, games.start_time),
                home_team = excluded.home_team,
                away_team = excluded.away_team,
                home_score = COALESCE(excluded.home_score, games.home_score),
                away_score = COALESCE(excluded.away_score, games.away_score),
                neutral = excluded.neutral,
                overtime = excluded.overtime,
                decision_type = COALESCE(excluded.decision_type, games.decision_type),
                division = COALESCE(excluded.division, games.division),
                conference = COALESCE(excluded.conference, games.conference),
                notes = COALESCE(excluded.notes, games.notes)
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
    # Ensure DB and tables exist so callers don't fail when the DB file
    # exists but hasn't been initialized yet.
    init_db(db_path)
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
    params_source: str | None = None,
    metric_optimized: str | None = None,
    best_score: float | None = None,
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
                params_source,
                metric_optimized,
                best_score,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(sport, season, model, market) DO UPDATE SET
                params_json = excluded.params_json,
                source_run_id = excluded.source_run_id,
                params_source = excluded.params_source,
                metric_optimized = excluded.metric_optimized,
                best_score = excluded.best_score,
                updated_at = datetime('now')
            """,
            (
                sport,
                season,
                model,
                market,
                params_json,
                source_run_id,
                params_source,
                metric_optimized,
                best_score,
            ),
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


def load_model_market_active_params(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    market: str,
) -> dict | None:
    """Return the full active params record for a model/market, if present."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT params_json, source_run_id, params_source, metric_optimized, best_score, updated_at
            FROM model_market_active_params
            WHERE sport = ? AND season = ? AND model = ? AND market = ?
            """,
            (sport, season, model, market),
        ).fetchone()
    if row is None:
        return None
    params_json = row[0]
    params = json.loads(params_json) if params_json is not None else None
    if params is not None and not isinstance(params, dict):
        raise ValueError("Stored market params must be a JSON object.")
    return {
        "params": params,
        "source_run_id": row[1],
        "params_source": row[2],
        "metric_optimized": row[3],
        "best_score": row[4],
        "updated_at": row[5],
    }


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


def has_model_market_active_params(
    db_path: str | Path, *, sport: str, season: str
) -> bool:
    """Return True when any market-active params exist for the sport/season."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM model_market_active_params
            WHERE sport = ? AND season = ?
            LIMIT 1
            """,
            (sport, season),
        ).fetchone()
    return bool(row)


def list_model_market_active_params(
    db_path: str | Path, *, sport: str, season: str
) -> list[dict[str, object]]:
    """Return all active params rows for a sport/season."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        rows = conn.execute(
            """
            SELECT model, market, params_json, source_run_id, params_source, metric_optimized, best_score, updated_at
            FROM model_market_active_params
            WHERE sport = ? AND season = ?
            ORDER BY model, market
            """,
            (sport, season),
        ).fetchall()
    results: list[dict[str, object]] = []
    for row in rows:
        params_json = row[2]
        params = json.loads(params_json) if params_json is not None else None
        if params is not None and not isinstance(params, dict):
            raise ValueError("Stored market params must be a JSON object.")
        results.append(
            {
                "model": row[0],
                "market": row[1],
                "params": params,
                "source_run_id": row[3],
                "params_source": row[4],
                "metric_optimized": row[5],
                "best_score": row[6],
                "updated_at": row[7],
            }
        )
    return results


def has_model_market_tuning_runs(
    db_path: str | Path, *, sport: str, season: str
) -> bool:
    """Return True when any market tuning runs exist for the sport/season."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM model_market_tuning_runs
            WHERE sport = ? AND season = ?
            LIMIT 1
            """,
            (sport, season),
        ).fetchone()
    return bool(row)


def model_market_tuning_run_exists(
    db_path: str | Path, *, sport: str, season: str, model: str, market: str
) -> bool:
    """Return True when at least one tuning run exists for the model+market."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM model_market_tuning_runs
            WHERE sport = ? AND season = ? AND model = ? AND market = ?
            LIMIT 1
            """,
            (sport, season, model, market),
        ).fetchone()
    return bool(row)


def has_nonempty_model_market_tuning_params(
    db_path: str | Path, *, sport: str, season: str, model: str, market: str
) -> bool:
    """Return True when at least one tuning run with non-empty params exists."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM model_market_tuning_runs
            WHERE sport = ? AND season = ? AND model = ? AND market = ?
              AND best_params_json IS NOT NULL
              AND best_params_json != ''
              AND best_params_json != '{}'
            LIMIT 1
            """,
            (sport, season, model, market),
        ).fetchone()
    return bool(row)


def legacy_tuned_params_exist(
    db_path: str | Path, *, sport: str, season: str
) -> bool:
    """Return True when legacy tuned params exist for the sport/season."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM model_tuned_params
            WHERE sport = ? AND season = ?
            LIMIT 1
            """,
            (sport, season),
        ).fetchone()
    return bool(row)


def upsert_model_tuned_params(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    metric: str,
    run_id: str,
    params: dict,
    best_score: float | None,
) -> None:
    """Persist legacy tuned params table (backward compatibility)."""
    if not isinstance(params, dict):
        raise ValueError("Legacy tuned params must be a JSON object (dict).")
    init_db(db_path)
    params_json = json.dumps(params, sort_keys=True)
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


def load_model_tuned_params(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    metric: str,
) -> tuple[dict | None, float | None, str | None]:
    """Load legacy tuned params row by metric."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT params_json, best_score, run_id
            FROM model_tuned_params
            WHERE sport = ? AND season = ? AND model = ? AND metric = ?
            LIMIT 1
            """,
            (sport, season, model, metric),
        ).fetchone()
    if row is None or row[0] is None:
        return None, None, None
    params = json.loads(row[0])
    if params is not None and not isinstance(params, dict):
        raise ValueError("Stored legacy tuned params must be a JSON object.")
    return params, row[1], row[2]


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
    baseline_score: float | None = None,
    weights_json: str | None = None,
    models_json: str | None = None,
    selection_run_id: str | None = None,
    selection_models_json: str | None = None,
    summary_metrics_json: str | None = None,
    data_source: str | None = None,
    dataset_db_path: str | None = None,
    csv_path: str | None = None,
    asof: str | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
    notes: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
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
                baseline_score,
                weights_json,
                models_json,
                selection_run_id,
                selection_models_json,
                summary_metrics_json,
                data_source,
                db_path,
                csv_path,
                asof,
                window_start,
                window_end,
                notes,
                started_at,
                finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sport,
                season,
                market,
                ensemble_id,
                metric_optimized,
                run_id,
                best_score,
                baseline_score,
                weights_json,
                models_json,
                selection_run_id,
                selection_models_json,
                summary_metrics_json,
                data_source,
                dataset_db_path,
                csv_path,
                asof,
                window_start,
                window_end,
                notes,
                started_at,
                finished_at,
            ),
        )
        conn.commit()


def _json_from_value(value: object | None) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def save_ensemble_market_selection_run(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
    run_id: str,
    window_start: str | None,
    window_end: str | None,
    asof: str | None,
    data_source: str,
    dataset_db_path: str | None,
    csv_path: str | None,
    scorable_games: int,
    date_min: str | None,
    date_max: str | None,
    candidates: list[str],
    selected: list[str],
    objective_metric: str,
    summary: dict[str, object],
    baseline_score: float | None = None,
    notes: str | None = None,
) -> None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO ensemble_market_selection_runs (
                sport,
                season,
                market,
                ensemble_id,
                run_id,
                window_start,
                window_end,
                asof,
                data_source,
                db_path,
                csv_path,
                scorable_games,
                date_min,
                date_max,
                candidates_json,
                selected_json,
                objective_metric,
                summary_json,
                baseline_score,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sport,
                season,
                market,
                ensemble_id,
                run_id,
                window_start,
                window_end,
                asof,
                data_source,
                dataset_db_path,
                csv_path,
                scorable_games,
                date_min,
                date_max,
                _json_from_value(candidates),
                _json_from_value(selected),
                objective_metric,
                _json_from_value(summary),
                baseline_score,
                notes,
            ),
        )
        conn.commit()


def set_active_ensemble_market_selection(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
    active_run_id: str,
) -> None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO ensemble_market_active_selection (
                sport,
                season,
                market,
                ensemble_id,
                active_run_id,
                activated_at
            ) VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(sport, season, market, ensemble_id) DO UPDATE SET
                active_run_id = excluded.active_run_id,
                activated_at = datetime('now')
            """,
            (sport, season, market, ensemble_id, active_run_id),
        )
        conn.commit()


def get_active_ensemble_market_selection(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
) -> dict[str, object] | None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT active_run_id, activated_at
            FROM ensemble_market_active_selection
            WHERE sport = ? AND season = ? AND market = ? AND ensemble_id = ?
            """,
            (sport, season, market, ensemble_id),
        ).fetchone()
    if row is None:
        return None
    return {"active_run_id": row[0], "activated_at": row[1]}


def set_active_ensemble_market_tuning_run(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
    active_run_id: str,
) -> None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO ensemble_market_active_tuning (
                sport,
                season,
                market,
                ensemble_id,
                active_run_id,
                activated_at
            ) VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(sport, season, market, ensemble_id) DO UPDATE SET
                active_run_id = excluded.active_run_id,
                activated_at = datetime('now')
            """,
            (sport, season, market, ensemble_id, active_run_id),
        )
        conn.commit()


def get_active_ensemble_market_tuning_run(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
) -> dict[str, object] | None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT active_run_id, activated_at
            FROM ensemble_market_active_tuning
            WHERE sport = ? AND season = ? AND market = ? AND ensemble_id = ?
            """,
            (sport, season, market, ensemble_id),
        ).fetchone()
    if row is None:
        return None
    return {"active_run_id": row[0], "activated_at": row[1]}


def load_selection_run(
    db_path: str | Path,
    *,
    run_id: str,
) -> dict[str, object] | None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT
                sport,
                season,
                market,
                ensemble_id,
                run_id,
                window_start,
                window_end,
                asof,
                data_source,
                db_path,
                csv_path,
                scorable_games,
                date_min,
                date_max,
                candidates_json,
                selected_json,
                objective_metric,
                summary_json,
                baseline_score,
                notes
            FROM ensemble_market_selection_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "sport": row[0],
        "season": row[1],
        "market": row[2],
        "ensemble_id": row[3],
        "run_id": row[4],
        "window_start": row[5],
        "window_end": row[6],
        "asof": row[7],
        "data_source": row[8],
        "db_path": row[9],
        "csv_path": row[10],
        "scorable_games": row[11],
        "date_min": row[12],
        "date_max": row[13],
        "candidates": json.loads(row[14]) if row[14] else [],
        "selected": json.loads(row[15]) if row[15] else [],
        "objective_metric": row[16],
        "summary": json.loads(row[17]) if row[17] else {},
        "baseline_score": row[18],
        "notes": row[19],
    }


def load_tuning_run(
    db_path: str | Path,
    *,
    run_id: str,
) -> dict[str, object] | None:
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT
                sport,
                season,
                market,
                ensemble_id,
                run_id,
                metric_optimized,
                best_score,
                baseline_score,
                weights_json,
                models_json,
                selection_run_id,
                selection_models_json,
                summary_metrics_json,
                data_source,
                db_path,
                csv_path,
                asof,
                window_start,
                window_end,
                notes
            FROM ensemble_market_tuning_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "sport": row[0],
        "season": row[1],
        "market": row[2],
        "ensemble_id": row[3],
        "run_id": row[4],
        "metric_optimized": row[5],
        "best_score": row[6],
        "baseline_score": row[7],
        "weights": json.loads(row[8]) if row[8] else {},
        "models": json.loads(row[9]) if row[9] else [],
        "selection_run_id": row[10],
        "selection_models": json.loads(row[11]) if row[11] else [],
        "summary": json.loads(row[12]) if row[12] else {},
        "data_source": row[13],
        "db_path": row[14],
        "csv_path": row[15],
        "asof": row[16],
        "window_start": row[17],
        "window_end": row[18],
        "notes": row[19],
    }


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


def get_active_ensemble_market_weights_and_models(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
) -> tuple[dict | None, list[str] | None]:
    """Return (weights, models_list) for an active ensemble, or (None, None)."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT weights_json, models_json
            FROM ensemble_market_active_weights
            WHERE sport = ? AND season = ? AND market = ? AND ensemble_id = ?
            """,
            (sport, season, market, ensemble_id),
        ).fetchone()
    if row is None or row[0] is None:
        return None, None
    weights = json.loads(row[0])
    models = json.loads(row[1]) if row[1] else None
    if not isinstance(weights, dict):
        raise ValueError("Stored ensemble weights must be a JSON object.")
    return weights, models


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
) -> tuple[dict | None, str | None, str | None, float | None]:
    """Load a model_market_tuning_runs entry by run_id. Returns (params_dict, run_id, metric_optimized, best_score) or (None, None, None, None)."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT best_params_json, run_id, metric_optimized, best_score
            FROM model_market_tuning_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if row is None or row[0] is None:
        return None, None, None, None
    data = json.loads(row[0])
    if not isinstance(data, dict):
        raise ValueError("Stored model market tuned params must be a JSON object.")
    return data, row[1], row[2], row[3]


def load_best_model_market_tuning_run(
    db_path: str | Path,
    *,
    sport: str,
    season: str,
    model: str,
    market: str,
) -> dict[str, Any] | None:
    """Return the best tuning run for the given model and market, if present."""
    init_db(db_path)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        row = conn.execute(
            """
            SELECT run_id, metric_optimized, best_params_json, best_score
            FROM model_market_tuning_runs
            WHERE sport = ? AND season = ? AND model = ? AND market = ?
            AND best_score IS NOT NULL
            ORDER BY best_score ASC
            LIMIT 1
            """,
            (sport, season, model, market),
        ).fetchone()
    if row is None or row[2] is None:
        return None
    params = json.loads(row[2])
    if params is not None and not isinstance(params, dict):
        raise ValueError("Stored model market tuned params must be a JSON object.")
    return {
        "run_id": row[0],
        "metric_optimized": row[1],
        "params": params,
        "best_score": row[3],
    }


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
            AND best_score IS NOT NULL
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


def save_validation_run(
    db_path: str | Path,
    *,
    run_id: str,
    sport: str,
    season: str,
    config: dict[str, Any],
    summary: dict[str, Any] | None,
    issues: list[str] | None,
    workbook_path: str | None,
    report_path: str | None,
    summary_path: str | None,
    artifacts_dir: str | None,
) -> None:
    """Persist a validation run summary to the database."""
    init_db(db_path)
    payload_config = json.dumps(config or {}, sort_keys=True)
    payload_summary = json.dumps(summary or {}, sort_keys=True)
    payload_issues = json.dumps(issues or [], sort_keys=True)
    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO validation_runs (
                run_id,
                sport,
                season,
                config_json,
                summary_json,
                issues_json,
                workbook_path,
                report_path,
                summary_path,
                artifacts_dir
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                sport,
                season,
                payload_config,
                payload_summary,
                payload_issues,
                workbook_path,
                report_path,
                summary_path,
                artifacts_dir,
            ),
        )
        conn.commit()
