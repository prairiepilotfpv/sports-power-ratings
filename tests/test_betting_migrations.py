import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from src.data import betting_repository as br
from src.data import migrations


def _create_old_betting_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE market_snapshot_staging (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            captured_at TEXT,
            image_path TEXT,
            raw_text TEXT,
            book TEXT,
            market_type TEXT,
            selection TEXT,
            line REAL,
            odds INTEGER,
            team_home_raw TEXT,
            team_away_raw TEXT,
            game_date TEXT,
            match_status TEXT,
            match_confidence REAL,
            game_id TEXT,
            created_at TEXT
        );
        """
    )


def test_init_db_applies_betting_migrations():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "legacy.db"
        with closing(sqlite3.connect(db_path)) as conn:
            _create_old_betting_schema(conn)
            conn.commit()

        br.init_db(db_path)

        with closing(sqlite3.connect(db_path)) as conn:
            staging_cols = [
                row[1] for row in conn.execute("PRAGMA table_info(market_snapshot_staging)")
            ]
            assert "hold_reason" in staging_cols

            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "clv_snapshots" in tables
            assert "schema_meta" in tables

            version_row = conn.execute(
                "SELECT schema_version FROM schema_meta WHERE id = 1"
            ).fetchone()
            assert version_row is not None
            assert int(version_row[0]) == migrations.LATEST_SCHEMA_VERSION
