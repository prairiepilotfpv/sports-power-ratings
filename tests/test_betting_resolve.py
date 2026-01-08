from pathlib import Path
import tempfile
from datetime import date

from src.data import betting_repository as br
from src.data import repository as repo


def test_resolve_staging_to_game():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        # create base db and games
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
            team_home_raw="LA Lakers",
            team_away_raw="Clippers",
            game_date="2025-11-10",
        )

        assert result["match_status"] in ("matched", "needs_review")
        assert result["game_id"] == "2025-11-10-lakers-clippers"
        assert result["match_confidence"] > 0.7
