"""Migration helpers for SQLite schemas."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Callable, Iterable

from config import DEFAULT_WIN_PROB_K
from src.utils.game_id import make_game_id
from src.utils.identity import normalize_team_name


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
    if "start_time" not in cols:
        conn.execute("ALTER TABLE games ADD COLUMN start_time TEXT")
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


def _add_games_start_time(conn: sqlite3.Connection) -> None:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(games)")]
    if not cols:
        return
    if "start_time" not in cols:
        conn.execute("ALTER TABLE games ADD COLUMN start_time TEXT")


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
            params_source TEXT,
            metric_optimized TEXT,
            best_score REAL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(sport, season, model, market)
        );
        """
    )
    cols = [row[1] for row in conn.execute("PRAGMA table_info(model_market_active_params)")]
    if "params_source" not in cols:
        conn.execute("ALTER TABLE model_market_active_params ADD COLUMN params_source TEXT")
    if "metric_optimized" not in cols:
        conn.execute("ALTER TABLE model_market_active_params ADD COLUMN metric_optimized TEXT")
    if "best_score" not in cols:
        conn.execute("ALTER TABLE model_market_active_params ADD COLUMN best_score REAL")
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


def _add_team_identity_and_market_lines(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sport TEXT NOT NULL,
            season TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            UNIQUE(sport, season, canonical_name)
        );
        """
    )
    conn.execute(
        """
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
        """
    )
    conn.execute(
        """
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
        """
    )
    conn.execute(
        """
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
        """
    )
    conn.execute(
        """
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
    )


def _add_validation_runs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
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
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_validation_runs_sport_season
        ON validation_runs(sport, season, created_at);
        """
    )


def _add_validation_runs_workbook_path(conn: sqlite3.Connection) -> None:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(validation_runs)")]
    if not cols:
        return
    if "workbook_path" not in cols:
        conn.execute("ALTER TABLE validation_runs ADD COLUMN workbook_path TEXT")


def _add_bets_predictions_components_json(conn: sqlite3.Connection) -> None:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(bets_predictions)")]
    if not cols:
        return
    if "ml_ensemble_components_json" not in cols:
        conn.execute(
            "ALTER TABLE bets_predictions ADD COLUMN ml_ensemble_components_json TEXT"
        )


def _backfill_game_ids(conn: sqlite3.Connection) -> None:
    """Compute deterministic game_id for games missing one and update rows.

    If a computed game_id would collide with an existing row, raise with details.
    """
    cols = [row[1] for row in conn.execute("PRAGMA table_info(games)")]
    if not cols:
        return
    rows = conn.execute(
        "SELECT id, sport, season, date, away_team, home_team FROM games WHERE game_id IS NULL OR game_id = ''"
    ).fetchall()
    if not rows:
        return
    duplicates = []
    updates = []
    for r in rows:
        gid = None
        try:
            gid = make_game_id(r[1], r[2], r[3], r[4], r[5])
        except Exception:
            continue
        # check for existing rows with same game_id
        existing = conn.execute(
            "SELECT COUNT(*) FROM games WHERE sport = ? AND season = ? AND game_id = ?",
            (r[1], r[2], gid),
        ).fetchone()
        if existing and existing[0] > 0:
            duplicates.append({
                "sport": r[1],
                "season": r[2],
                "date": r[3],
                "away": r[4],
                "home": r[5],
                "would_be": gid,
            })
        else:
            updates.append((gid, r[0]))

    if duplicates:
        details = "\n".join(
            [f"sport={d['sport']} season={d['season']} date={d['date']} away={d['away']} home={d['home']} -> {d['would_be']}" for d in duplicates]
        )
        raise RuntimeError(f"Backfill would create duplicate game_id(s):\n{details}")

    for gid, rowid in updates:
        conn.execute("UPDATE games SET game_id = ? WHERE id = ?", (gid, rowid))


def _migrate_game_ids_and_update_references(conn: sqlite3.Connection) -> None:
    """Migrate games to deterministic game_id, preserve legacy_game_id,
    and update dependent tables to reference the new ids.
    """
    # add legacy column if missing
    cols = [row[1] for row in conn.execute("PRAGMA table_info(games)")]
    if not cols:
        return
    if "legacy_game_id" not in cols:
        conn.execute("ALTER TABLE games ADD COLUMN legacy_game_id TEXT")

    # load all games
    rows = conn.execute(
        "SELECT id, sport, season, date, home_team, away_team, game_id, legacy_game_id FROM games"
    ).fetchall()

    old_to_new: dict[str, str] = {}
    new_to_rows: dict[str, list] = {}

    for r in rows:
        gid_id, sport, season, date_str, home, away, old_gid, legacy_gid = r
        try:
            new_gid = make_game_id(sport or "", season or "", date_str, away or "", home or "")
        except Exception:
            continue

        # preserve legacy_game_id when empty
        if (not legacy_gid or str(legacy_gid).strip() == "") and old_gid:
            conn.execute("UPDATE games SET legacy_game_id = ? WHERE id = ?", (old_gid, gid_id))

        new_to_rows.setdefault(new_gid, []).append((gid_id, sport, season, date_str, away, home, old_gid))

        # replace game_id with computed value
        conn.execute("UPDATE games SET game_id = ? WHERE id = ?", (new_gid, gid_id))

        if old_gid and old_gid != new_gid:
            old_to_new[old_gid] = new_gid

    # detect collisions where same new_gid assigned to multiple distinct rows
    collisions = {k: v for k, v in new_to_rows.items() if len(v) > 1}
    if collisions:
        details = []
        for new_gid, items in collisions.items():
            for it in items:
                details.append(f"new_gid={new_gid} id={it[0]} sport={it[1]} season={it[2]} date={it[3]} away={it[4]} home={it[5]} old={it[6]}")
        raise RuntimeError("Game ID collision detected during migration:\n" + "\n".join(details))

    # update dependent tables using mapping old->new
    dependent_tables = [
        "market_lines",
        "market_snapshot_staging",
        "market_snapshots",
        "forecast_snapshots",
        "opportunities",
        "bets",
        "clv_snapshots",
        "prediction_exclusions",
    ]

    for old, new in old_to_new.items():
        for tbl in dependent_tables:
            try:
                conn.execute(f"UPDATE {tbl} SET game_id = ? WHERE game_id = ?", (new, old))
            except Exception:
                # table may not exist in this DB
                pass

    # attempt to resolve staging rows that remain unmatched by exact normalized name/date
    try:
        staging_rows = conn.execute(
            "SELECT id, team_home_raw, team_away_raw, game_date FROM market_snapshot_staging WHERE game_id IS NULL OR game_id = ''"
        ).fetchall()
    except Exception:
        staging_rows = []

    for sid, home_raw, away_raw, gdate in staging_rows:
        if not gdate or not (home_raw or away_raw):
            continue
        candidates = conn.execute(
            "SELECT game_id, home_team, away_team FROM games WHERE date = ?",
            (gdate,),
        ).fetchall()
        matches = []
        for cand in candidates:
            cand_gid, home_c, away_c = cand
            if normalize_team_name(home_raw) == normalize_team_name(home_c) and normalize_team_name(away_raw) == normalize_team_name(away_c):
                matches.append(cand_gid)
        if len(matches) == 1:
            conn.execute("UPDATE market_snapshot_staging SET game_id = ? WHERE id = ?", (matches[0], sid))

    # cascade resolved staging -> market_snapshots -> opportunities -> bets where source ids exist
    try:
        conn.execute(
            "UPDATE market_snapshots SET game_id = (SELECT game_id FROM market_snapshot_staging WHERE market_snapshot_staging.id = market_snapshots.source_staging_id) WHERE (game_id IS NULL OR game_id = '') AND source_staging_id IS NOT NULL"
        )
    except Exception:
        pass
    try:
        conn.execute(
            "UPDATE opportunities SET game_id = (SELECT game_id FROM market_snapshots WHERE market_snapshots.id = opportunities.source_market_snapshot_id) WHERE (game_id IS NULL OR game_id = '') AND source_market_snapshot_id IS NOT NULL"
        )
    except Exception:
        pass
    try:
        conn.execute(
            "UPDATE bets SET game_id = (SELECT game_id FROM opportunities WHERE opportunities.id = bets.source_opportunity_id) WHERE (game_id IS NULL OR game_id = '') AND source_opportunity_id IS NOT NULL"
        )
    except Exception:
        pass

    # Do not commit here; caller will commit. This allows callers to run
    # the migration inside a transaction for dry-run analysis.


def analyze_migration(conn: sqlite3.Connection) -> dict:
    """Analyze the deterministic game_id migration and return a report
    without modifying the database.

    Returns a dict containing:
      - mappings: dict of legacy_id -> new_id
      - collisions: dict of new_id -> list of rows that would collide
      - sample_updates: list of (rowid, old_gid, new_gid)
      - counts: summary counts
    """
    cols = [row[1] for row in conn.execute("PRAGMA table_info(games)")]
    if not cols:
        return {"mappings": {}, "collisions": {}, "sample_updates": [], "counts": {}}

    rows = conn.execute(
        "SELECT id, sport, season, date, home_team, away_team, game_id FROM games"
    ).fetchall()

    old_to_new: dict[str, str] = {}
    new_to_rows: dict[str, list] = {}
    sample_updates: list[tuple] = []

    for r in rows:
        gid_id, sport, season, date_str, home, away, old_gid = r
        try:
            new_gid = make_game_id(sport or "", season or "", date_str, away or "", home or "")
        except Exception:
            continue
        new_to_rows.setdefault(new_gid, []).append((gid_id, sport, season, date_str, away, home, old_gid))
        if old_gid and old_gid != new_gid:
            old_to_new[old_gid] = new_gid
            sample_updates.append((gid_id, old_gid, new_gid))

    collisions = {k: v for k, v in new_to_rows.items() if len(v) > 1}

    return {
        "mappings": old_to_new,
        "collisions": {k: [(it[0], it[4], it[5], it[6]) for it in v] for k, v in collisions.items()},
        "sample_updates": sample_updates[:20],
        "counts": {
            "games_total": len(rows),
            "would_update": len(sample_updates),
            "collisions": len(collisions),
        },
    }


MIGRATIONS: list[Migration] = [
    Migration(1, "add_hold_reason_to_staging", _add_hold_reason_to_staging),
    Migration(2, "add_clv_snapshots_table", _add_clv_snapshots_table),
    Migration(3, "add_prediction_exclusions_table", _add_prediction_exclusions_table),
    Migration(4, "add_games_metadata_columns", _add_games_metadata_columns),
    Migration(5, "add_model_metrics_columns", _add_model_metrics_columns),
    Migration(6, "add_market_tuning_tables", _add_market_tuning_tables),
    Migration(7, "add_games_start_time", _add_games_start_time),
    Migration(8, "add_team_identity_and_market_lines", _add_team_identity_and_market_lines),
    Migration(9, "backfill_game_ids", _backfill_game_ids),
    Migration(10, "migrate_to_deterministic_game_id_and_update_refs", _migrate_game_ids_and_update_references),
    Migration(11, "add_validation_runs_table", _add_validation_runs_table),
    Migration(12, "add_validation_runs_workbook_path", _add_validation_runs_workbook_path),
    Migration(13, "add_bets_predictions_components_json", _add_bets_predictions_components_json),
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
