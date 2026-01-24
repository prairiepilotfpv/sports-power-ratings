"""Test that BETS sheet properly orders games with/without market lines."""

from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd

from src.pipelines.schedule import _build_bets_dataframe


def test_bets_sheet_games_with_lines_appear_before_without():
    """Verify games with market lines appear before games without in BETS sheet."""
    as_of = date(2025, 11, 10)
    
    # Create two games: one WITH market lines, one WITHOUT
    schedule_df = pd.DataFrame([
        # Game 1: HAS market lines (game_id='g1')
        {
            "date": as_of.isoformat(),
            "status": "scheduled",
            "game_id": "g1",
            "away_team": "A",
            "home_team": "H",
            "home_win_prob": 0.6,
            "away_win_prob": 0.4,
            "win_prob_source": "model_x",
            "margin_mean": 3.0,
            "margin_sd": 1.2,
            "total": 210.5,
            "total_sd": 5.0,
        },
        # Game 2: NO market lines (game_id='g2')
        {
            "date": as_of.isoformat(),
            "status": "scheduled",
            "game_id": "g2",
            "away_team": "B",
            "home_team": "I",
            "home_win_prob": 0.55,
            "away_win_prob": 0.45,
            "win_prob_source": "model_x",
            "margin_mean": 2.5,
            "margin_sd": 1.1,
            "total": 205.0,
            "total_sd": 4.5,
        },
    ])

    def mock_get_latest_market_line(db_path, sport, season, game_id, market_type, selection=None, selection_team_id=None):
        """Return market lines only for g1."""
        if game_id == "g1":
            return {"line": "-110", "odds": -110, "id": f"{game_id}_{market_type}_{selection or selection_team_id}"}
        return None  # No lines for g2
    
    with patch("src.data.betting_repository.get_latest_market_line", side_effect=mock_get_latest_market_line):
        bets_df = _build_bets_dataframe(
            schedule_df,
            model_name="m",
            as_of_date=as_of,
            review_run_id="rr1",
            db_path="/tmp/test.db",
            sport="nba",
            season="2025-26",
        )
        
        # Check that all rows are present
        assert len(bets_df) > 0, "BETS dataframe should not be empty"
        
        # Check game_ids are present
        game_ids = bets_df["game_id"].unique()
        assert "g1" in game_ids, "Game g1 should be in BETS sheet"
        assert "g2" in game_ids, "Game g2 should be in BETS sheet"
        
        # Get indices of first occurrence of each game
        g1_first_idx = bets_df[bets_df["game_id"] == "g1"].index[0]
        g2_first_idx = bets_df[bets_df["game_id"] == "g2"].index[0]
        
        # Verify g1 (with lines) appears before g2 (without lines)
        assert g1_first_idx < g2_first_idx, (
            f"Games with market lines (g1 at {g1_first_idx}) should appear before "
            f"games without lines (g2 at {g2_first_idx}) in BETS sheet"
        )
        
        # Verify g1 rows have filled lines while g2 rows have empty lines
        g1_rows = bets_df[bets_df["game_id"] == "g1"]
        g2_rows = bets_df[bets_df["game_id"] == "g2"]
        
        # At least some g1 rows should have non-empty lines
        assert (g1_rows["line"] != "").any(), "Game g1 should have some rows with market lines"
        
        # All g2 rows should have empty lines
        assert (g2_rows["line"] == "").all(), "Game g2 should have all rows without market lines"
        
        # Verify market order within game: ML before spread before total
        ml_rows = g1_rows[g1_rows["market_type"] == "ML"]
        spread_rows = g1_rows[g1_rows["market_type"] == "spread"]
        total_rows = g1_rows[g1_rows["market_type"] == "total"]
        
        if not ml_rows.empty and not spread_rows.empty:
            assert ml_rows.index[0] < spread_rows.index[0], "ML rows should appear before spread rows"
        if not spread_rows.empty and not total_rows.empty:
            assert spread_rows.index[0] < total_rows.index[0], "Spread rows should appear before total rows"

