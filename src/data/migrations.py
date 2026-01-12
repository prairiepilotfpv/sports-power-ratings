"""Migration helpers for SQLite schemas."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Callable, Iterable

from config import DEFAULT_WIN_PROB_K


MigrationFn = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    fn: MigrationFn


def _ensure_schema_meta(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            schema_version INTEGER NOT NULL,
            updated_at TEXT
        );
        """
    )
    existing = conn.execute("SELECT schema_version FROM schema_meta WHERE id = 1").fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO schema_meta (id, schema_version, updated_at) VALUES (1, 0, datetime('now'))"
        )


def get_schema_version(conn: sqlite3.Connection) -> int:
    _ensure_schema_meta(conn)
    row = conn.execute("SELECT schema_version FROM schema_meta WHERE id = 1").fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    _ensure_schema_meta(conn)
    conn.execute(
        "UPDATE schema_meta SET schema_version = ?, updated_at = datetime('now') WHERE id = 1",
        (int(version),),
    )


def _add_hold_reason_to_staging(conn: sqlite3.Connection) -> None:
    staging_cols = [row[1] for row in conn.execute("PRAGMA table_info(market_snapshot_staging)")]
    if not staging_cols:
        return
    if "hold_reason" not in staging_cols:
        conn.execute("ALTER TABLE market_snapshot_staging ADD COLUMN hold_reason TEXT")


def _add_clv_snapshots_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clv_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            market_type TEXT,
            selection TEXT,
            close_line REAL,
            close_odds INTEGER,
            captured_at TEXT,
            created_at TEXT
        );
        """
    )


def _add_prediction_exclusions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_exclusions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_run_id TEXT,
            game_id TEXT,
            model TEXT,
            excluded_reason TEXT,
            created_at TEXT,
            UNIQUE(review_run_id, game_id, model, excluded_reason)
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_exclusions_review_run ON prediction_exclusions(review_run_id);"
    )


def _add_games_metadata_columns(conn: sqlite3.Connection) -> None:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(games)")]
    if not cols:
        return
    if "division" not in cols:
        conn.execute("ALTER TABLE games ADD COLUMN division TEXT")
    if "conference" not in cols:
        conn.execute("ALTER TABLE games ADD COLUMN conference TEXT")
    if "decision_type" not in cols:
        conn.execute("ALTER TABLE games ADD COLUMN decision_type TEXT")


def _add_model_metrics_columns(conn: sqlite3.Connection) -> None:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(model_metrics)")]
    if not cols:
        return
    if "win_prob_k" not in cols:
        conn.execute(
            f"ALTER TABLE model_metrics ADD COLUMN win_prob_k REAL NOT NULL DEFAULT {DEFAULT_WIN_PROB_K}"
        )
    if "base_total" not in cols:
        conn.execute(
            "ALTER TABLE model_metrics ADD COLUMN base_total REAL NOT NULL DEFAULT 0.0"
        )
    if "backtest_log_loss" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_log_loss REAL")
    if "backtest_brier_score" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_brier_score REAL")
    if "backtest_mae_margin" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_mae_margin REAL")
    if "backtest_mae_total" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_mae_total REAL")
    if "backtest_win_prob_k" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_win_prob_k REAL")
    if "margin_std" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN margin_std REAL")
    if "total_std" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN total_std REAL")
    if "conditional_sd_intercept" not in cols:
        conn.execute(
            "ALTER TABLE model_metrics ADD COLUMN conditional_sd_intercept REAL"
        )
    if "conditional_sd_slope" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN conditional_sd_slope REAL")
    if "margin_mean" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN margin_mean REAL")
    if "total_mean" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN total_mean REAL")
    if "backtest_run_id" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_run_id TEXT")
    if "backtest_updated_at" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_updated_at TEXT")
    if "tuned_params_json" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN tuned_params_json TEXT")
    if "tuned_params_metric" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN tuned_params_metric TEXT")
    if "tuned_params_updated_at" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN tuned_params_updated_at TEXT")


def _add_market_tuning_tables(conn: sqlite3.Connection) -> None:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(model_metrics)")]
    if cols and "backtest_mae_total" not in cols:
        conn.execute("ALTER TABLE model_metrics ADD COLUMN backtest_mae_total REAL")
    conn.execute(
        """
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
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_model_market_tuning_runs
        ON model_market_tuning_runs(sport, season, model, market, metric_optimized);
        """
    )
    conn.execute(
        """
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
        """
    )
    conn.execute(
        """
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
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ensemble_market_tuning_runs
        ON ensemble_market_tuning_runs(sport, season, market, ensemble_id);
        """
    )
    conn.execute(
        """
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
        """
    )


MIGRATIONS: list[Migration] = [
    Migration(1, "add_hold_reason_to_staging", _add_hold_reason_to_staging),
    Migration(2, "add_clv_snapshots_table", _add_clv_snapshots_table),
    Migration(3, "add_prediction_exclusions_table", _add_prediction_exclusions_table),
    Migration(4, "add_games_metadata_columns", _add_games_metadata_columns),
    Migration(5, "add_model_metrics_columns", _add_model_metrics_columns),
    Migration(6, "add_market_tuning_tables", _add_market_tuning_tables),
]

LATEST_SCHEMA_VERSION = max((m.version for m in MIGRATIONS), default=0)


def apply_migrations(
    conn: sqlite3.Connection, migrations: Iterable[Migration] = MIGRATIONS
) -> int:
    """Apply migrations in order and return the resulting schema version."""
    _ensure_schema_meta(conn)
    current = get_schema_version(conn)
    for migration in sorted(migrations, key=lambda m: m.version):
        if migration.version <= current:
            continue
        migration.fn(conn)
        set_schema_version(conn, migration.version)
        current = migration.version
    return current
