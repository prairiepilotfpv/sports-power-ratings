import sqlite3
from pathlib import Path
import tempfile

from src.data import betting_repository as br


EXPECTED_TABLES = {
    "review_runs",
    "market_snapshot_staging",
    "market_snapshots",
    "forecast_snapshots",
    "opportunities",
    "bets",
    "clv_snapshots",
    "prediction_exclusions",
    "schema_meta",
}


def test_init_creates_tables():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        br.init_db(db_path)
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            names = {r[0] for r in rows}
            assert EXPECTED_TABLES.issubset(names)
        finally:
            conn.close()
