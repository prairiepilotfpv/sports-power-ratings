from pathlib import Path
import tempfile
from datetime import date
import sqlite3

from src.data import betting_repository as br
from src.data import repository as repo


def test_create_review_run():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        rid = br.create_review_run(db_path, sport="nba", season="2025-26", model="elo", notes="test")
        conn = sqlite3.connect(db_path)
        try:
            r = conn.execute("SELECT id, sport, season, model, notes, created_at FROM review_runs WHERE id = ?", (rid,)).fetchone()
            assert r is not None
            assert r[1] == "nba"
            assert r[2] == "2025-26"
            assert r[3] == "elo"
            assert r[4] == "test"
            assert r[5] is not None
        finally:
            conn.close()


def test_commit_market_snapshot_and_update_staging():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        # seed games
        repo.init_db(db_path)
        repo.save_games(
            db_path,
            [
                repo.GameResult(
                    date=date(2025, 11, 10),
                    home_team="Los Angeles Lakers",
                    away_team="LA Clippers",
                    home_score=None,
                    away_score=None,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="2025-11-10-lakers-clippers",
                    sport="nba",
                    season="2025-26",
                    division=None,
                    conference=None,
                    notes=None,
                )
            ],
        )
        br.init_db(db_path)
        sid = br.add_staging_row(
            db_path,
            source="action-screenshot",
            captured_at="2025-11-10T10:00:00",
            image_path="/tmp/foo.png",
            raw_text="Lakers vs Clippers +110",
            book="BookA",
            market_type="ML",
            selection="Los Angeles Lakers",
            line=0.0,
            odds=110,
            team_home_raw="LA Lakers",
            team_away_raw="Clippers",
            game_date="2025-11-10",
            match_status="matched",
            match_confidence=0.95,
            game_id="2025-11-10-lakers-clippers",
        )

        committed = br.commit_market_snapshots(db_path, snapshot_run_id="run1", staging_ids=[sid])
        assert committed == 1

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT match_status FROM market_snapshot_staging WHERE id = ?", (sid,)).fetchone()
            assert row[0] == "committed"
            ms = conn.execute("SELECT snapshot_run_id, game_id, market_type, selection, odds FROM market_snapshots WHERE snapshot_run_id = ?", ("run1",)).fetchone()
            assert ms is not None
            assert ms[1] == "2025-11-10-lakers-clippers"
        finally:
            conn.close()


def test_commit_refuses_needs_review():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        sid = br.add_staging_row(db_path, source="action-screenshot", match_status="needs_review")
        try:
            br.commit_market_snapshots(db_path, snapshot_run_id="run2", staging_ids=[sid])
            assert False, "Expected ValueError due to needs_review"
        except ValueError:
            pass
        # But force should work
        committed = br.commit_market_snapshots(db_path, snapshot_run_id="run2", staging_ids=[sid], force=True)
        assert committed == 1


def test_export_needs_review():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        br.init_db(db_path)
        br.add_staging_row(db_path, source="action-screenshot", match_status="needs_review")
        rows = br.export_needs_review(db_path)
        assert len(rows) >= 1
