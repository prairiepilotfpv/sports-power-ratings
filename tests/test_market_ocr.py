import tempfile
from pathlib import Path
from unittest import mock

from src.pipelines.market_ocr import ingest_screenshots
from src.data import betting_repository as br
from src.data import repository as repo


def fake_ocr_image(path: str) -> str:
    # header with teams, and market lines
    return "Los Angeles Lakers vs LA Clippers\nLakers +110\nClippers +120\nTotal Over 215.5 -110"


def test_ingest_screenshots_parses_and_stages(monkeypatch):
    monkeypatch.setattr("src.ocr.ocr.ocr_image", fake_ocr_image)
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        # create a dummy image file
        img = Path(td) / "img1.png"
        img.write_text("dummy")
        count = ingest_screenshots([str(img)], db_path=db_path, sport="nba", season="2025-26", source="screenshot", book="BookA", captured_at="2025-11-10")
        assert count >= 1
        # assert staging rows exist
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            c = conn.execute("SELECT COUNT(*) FROM market_snapshot_staging").fetchone()[0]
            rows = conn.execute("SELECT selection, odds, market_type, match_status, match_confidence FROM market_snapshot_staging").fetchall()
            # ensure at least one staging row and that match_status/confidence present
            assert c >= 1
            assert all(isinstance(r[3], str) for r in rows)
            assert all(isinstance(r[4], float) or isinstance(r[4], int) for r in rows)
        finally:
            conn.close()
