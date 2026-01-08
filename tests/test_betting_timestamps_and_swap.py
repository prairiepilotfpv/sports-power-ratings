from pathlib import Path
import tempfile
from datetime import date
import sqlite3

from src.data import betting_repository as br
from src.data import repository as repo


def test_create_review_run_has_created_at():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        rid = br.create_review_run(db_path, sport="nba", season="2025-26", model="elo", notes="test")
        conn = sqlite3.connect(db_path)
        try:
            r = conn.execute("SELECT created_at FROM review_runs WHERE id = ?", (rid,)).fetchone()
            assert r is not None
            assert r[0] is not None
        finally:
            conn.close()


def test_add_staging_row_sets_created_at():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        br.init_db(db_path)
        sid = br.add_staging_row(db_path, source="s", match_status="unmatched")
        conn = sqlite3.connect(db_path)
        try:
            r = conn.execute("SELECT created_at FROM market_snapshot_staging WHERE id = ?", (sid,)).fetchone()
            assert r is not None
            assert r[0] is not None
        finally:
            conn.close()


def test_resolve_handles_swapped_home_away():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
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
        result = br.resolve_staging_to_game(
            db_path,
            sport="nba",
            season="2025-26",
            team_home_raw="Clippers",
            team_away_raw="LA Lakers",
            game_date="2025-11-10",
        )
        assert result["game_id"] == "2025-11-10-lakers-clippers"
        assert result["match_confidence"] > 0.4
