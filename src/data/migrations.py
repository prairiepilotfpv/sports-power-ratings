"""Migration helpers for SQLite schemas."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Callable, Iterable


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


MIGRATIONS: list[Migration] = [
    Migration(1, "add_hold_reason_to_staging", _add_hold_reason_to_staging),
    Migration(2, "add_clv_snapshots_table", _add_clv_snapshots_table),
    Migration(3, "add_prediction_exclusions_table", _add_prediction_exclusions_table),
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
