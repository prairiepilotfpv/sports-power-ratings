from pathlib import Path
import sqlite3
import tempfile

from datetime import date

from src.data import betting_repository as br
from src.data import repository as repo
from src.pipelines import staging_bets


def _seed_game(db_path: Path) -> None:
    repo.save_games(
        db_path,
        [
            repo.GameResult(
                date=date.fromisoformat("2025-11-10"),
                home_team="Team A",
                away_team="Team B",
                home_score=None,
                away_score=None,
                neutral=False,
                overtime=False,
                decision_type=None,
                game_id="game-1",
                sport="nba",
                season="2025-26",
                division=None,
                conference=None,
                notes=None,
            )
        ],
    )


def test_pivot_staging_to_bets_tags_duplicates_and_inserts():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        _seed_game(db_path)

        first_id = br.add_staging_row(
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
            match_confidence=0.95,
            game_id="game-1",
        )
        duplicate_id = br.add_staging_row(
            db_path,
            source="ocr",
            captured_at="2025-11-09T12:00:01Z",
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
            match_confidence=0.94,
            game_id="game-1",
        )
        br.add_staging_row(
            db_path,
            source="ocr",
            captured_at="2025-11-09T12:00:02Z",
            image_path="img2.png",
            raw_text="Team B -3.5 -110",
            book="BookB",
            market_type="spread",
            selection="Team B",
            line=-3.5,
            odds=-110,
            team_home_raw="Team A",
            team_away_raw="Team B",
            game_date="2025-11-10",
            match_status="matched",
            match_confidence=0.9,
            game_id="game-1",
        )

        result = staging_bets.pivot_staging_to_bets(
            db_path,
            review_run_id="rr-1",
            match_statuses=["matched"],
            stake_preset="double",
            unit_stake=2.0,
            default_book=None,
        )

        assert result["inserted"] == 2
        assert result["held"] == 1
        assert result["skipped"] == 0
        assert result["review_run_id"] == "rr-1"
        assert result["stake"] == 4.0

        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT game_id, market_type, selection, stake, book FROM bets ORDER BY id").fetchall()
            assert len(rows) == 2
            # first ML row kept, second ML held
            assert any(r[0] == "game-1" and r[1] == "ML" and r[2] == "Team A" and abs(r[3] - 4.0) < 1e-6 for r in rows)
            assert any(r[1] == "spread" for r in rows)
        finally:
            conn.close()

        # held row should have hold_reason set
        staging_rows = br.list_staging_rows(db_path, match_statuses=["matched"])
        held = [r for r in staging_rows if r.get("id") == duplicate_id]
        assert held and held[0].get("hold_reason") == "duplicate_in_image"
        kept = [r for r in staging_rows if r.get("id") == first_id]
        assert kept and kept[0].get("hold_reason") is None


def test_pivot_staging_to_bets_dry_run_skips_writes():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        _seed_game(db_path)

        br.add_staging_row(
            db_path,
            source="ocr",
            captured_at="2025-11-09T12:00:00Z",
            image_path="img3.png",
            raw_text="Team A +110",
            book=None,
            market_type="ML",
            selection="Team A",
            line=0.0,
            odds=110,
            team_home_raw="Team A",
            team_away_raw="Team B",
            game_date="2025-11-10",
            match_status="matched",
            match_confidence=0.95,
            game_id="game-1",
        )

        result = staging_bets.pivot_staging_to_bets(
            db_path,
            review_run_id=None,
            match_statuses=["matched"],
            stake_preset="half",
            unit_stake=1.0,
            default_book="Fallback",
            dry_run=True,
        )

        assert result["inserted"] == 1
        assert result["held"] == 0
        assert result["skipped"] == 0
        # ensure no bets were written
        conn = sqlite3.connect(db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
            assert count == 0
        finally:
            conn.close()
