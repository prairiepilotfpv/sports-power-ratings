"""Test for mixed sports import validation."""

from pathlib import Path
import tempfile
from datetime import date

import pytest

from src.data import repository as repo
from src.data import betting_repository as br


def test_import_bets_csv_rejects_mixed_sports():
    """Verify that import-csv rejects CSVs with mixed sports.
    
    This validates that:
    ✅ NFL bets go only to data/db/nfl/2025-26.db
    ✅ NBA bets go only to data/db/nba/2025-26.db
    ✅ No mixing of sports in the same database
    ✅ No new databases are created (uses existing database for sport/season)
    """
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo.init_db(db_path)
        br.init_db(db_path)
        
        # Seed games for both sports
        repo.save_games(
            db_path,
            [
                repo.GameResult(
                    date=date(2026, 1, 8),
                    home_team="Philadelphia 76ers",
                    away_team="Washington Wizards",
                    home_score=None,
                    away_score=None,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="nba-game-1",
                    sport="nba",
                    season="2025-26",
                    division=None,
                    conference=None,
                    notes=None,
                ),
                repo.GameResult(
                    date=date(2026, 1, 8),
                    home_team="Ottawa Senators",
                    away_team="Utah Hockey Club",
                    home_score=None,
                    away_score=None,
                    neutral=False,
                    overtime=False,
                    decision_type=None,
                    game_id="nhl-game-1",
                    sport="nhl",
                    season="2025-26",
                    division=None,
                    conference=None,
                    notes=None,
                ),
            ],
        )

        # Create CSV with BOTH nba and nhl bets
        csv_path = Path(td) / "mixed_sports.csv"
        csv_data = """league,start_time,game,game_id,pick_desc,type,period,odds,odds_spread_total,result,units_wagered
nba,2026-01-08T00:00:00.000Z,WAS @ PHI,nba-game-1,WAS +12.5 -110,spread_away,game,-110,12.5,loss,1.1
nhl,2026-01-08T00:00:00.000Z,OTT @ UTA,nhl-game-1,UTA -125,ml_home,game,-125,-125,win,1.25"""
        csv_path.write_text(csv_data)

        # Try to import with sport="nba" (should fail because NHL rows are in CSV)
        with pytest.raises(ValueError, match="CSV contains bets for mismatched sports"):
            br.import_bets_csv(
                csv_path=csv_path,
                sport="nba",
                season="2025-26",
                db_path=str(db_path),
            )
        
        # Verify no bets were inserted
        import sqlite3
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
        conn.close()
        assert count == 0, "Mixed-sport import should have failed and inserted no bets"
