from pathlib import Path
import tempfile

from src.data import betting_repository as br
from src.data import repository as repo
from src.pipelines import market_review


def test_accept_and_reject_match_updates_staging():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)

        staging_id = br.add_staging_row(
            db_path,
            source="ocr",
            captured_at="2025-11-09T12:00:00Z",
            image_path="img.png",
            raw_text="Team A +110",
            book="BookA",
            market_type="ML",
            selection="Team A",
            line=0.0,
            odds=110,
            team_home_raw="Team A",
            team_away_raw="Team B",
            game_date="2025-11-10",
            match_status="needs_review",
            match_confidence=0.6,
            game_id=None,
        )

        accepted = market_review.accept_match(
            db_path,
            staging_id=staging_id,
            game_id="game-1",
            match_confidence=0.9,
        )
        assert accepted["match_status"] == "matched"
        assert accepted["game_id"] == "game-1"
        assert accepted["match_confidence"] == 0.9

        rejected = market_review.reject_match(db_path, staging_id=staging_id)
        assert rejected["match_status"] == "unmatched"
        assert rejected["game_id"] is None
        assert rejected["match_confidence"] == 0.0


def test_list_staging_rows_respects_status_filter():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)

        br.add_staging_row(
            db_path,
            source="ocr",
            captured_at="2025-11-09T12:00:00Z",
            image_path="img1.png",
            raw_text="Team A +110",
            book="BookA",
            market_type="ML",
            selection="Team A",
            line=0.0,
            odds=110,
            team_home_raw="Team A",
            team_away_raw="Team B",
            game_date="2025-11-10",
            match_status="matched",
            match_confidence=1.0,
            game_id="game-1",
        )
        br.add_staging_row(
            db_path,
            source="ocr",
            captured_at="2025-11-09T12:00:01Z",
            image_path="img2.png",
            raw_text="Team B +120",
            book="BookA",
            market_type="ML",
            selection="Team B",
            line=0.0,
            odds=120,
            team_home_raw="Team A",
            team_away_raw="Team B",
            game_date="2025-11-10",
            match_status="needs_review",
            match_confidence=0.7,
            game_id=None,
        )

        matched_rows = market_review.list_staging_rows(db_path, match_statuses=["matched"], limit=None)
        assert len(matched_rows) == 1
        assert matched_rows[0]["match_status"] == "matched"

        all_rows = market_review.list_staging_rows(db_path, match_statuses=None, limit=None)
        assert len(all_rows) == 2
