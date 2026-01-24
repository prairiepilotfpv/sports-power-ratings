"""Test that BETS sheet enforces the canonical 6-rows-per-game invariant."""

from datetime import date

import pandas as pd

from src.pipelines.schedule import _build_bets_dataframe


def test_bets_invariant_exactly_six_rows_per_game():
    """Verify the invariant: each game_id appears exactly 6 times (2×ML, 2×spread, 2×total)."""
    as_of = date(2025, 11, 10)
    
    # Create 3 games with different combinations of market data
    schedule_df = pd.DataFrame([
        # Game 1: Fully populated
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
        # Game 2: Minimal data
        {
            "date": as_of.isoformat(),
            "status": "scheduled",
            "game_id": "g2",
            "away_team": "B",
            "home_team": "I",
            "home_win_prob": 0.55,
            "away_win_prob": 0.45,
            "win_prob_source": "model_y",
        },
        # Game 3: Partially populated
        {
            "date": as_of.isoformat(),
            "status": "scheduled",
            "game_id": "g3",
            "away_team": "C",
            "home_team": "J",
            "home_win_prob": 0.50,
            "away_win_prob": 0.50,
            "margin_mean": 0.0,
            "margin_sd": 0.5,
        },
    ])

    bets_df = _build_bets_dataframe(
        schedule_df,
        model_name="m",
        as_of_date=as_of,
        review_run_id="rr1",
    )

    # Verify invariant
    assert not bets_df.empty, "BETS dataframe should not be empty"
    
    game_counts = bets_df["game_id"].value_counts()
    
    # All games should have exactly 6 rows
    assert (game_counts == 6).all(), (
        f"All games should have exactly 6 rows. Got: {game_counts.to_dict()}"
    )
    
    # Expected games are present
    assert set(game_counts.index) == {"g1", "g2", "g3"}, "All three games should be present"
    
    # BETS row count equals 6 × number of games
    expected_total_rows = 6 * 3
    assert len(bets_df) == expected_total_rows, (
        f"Expected {expected_total_rows} rows ({6} per game × {3} games), "
        f"got {len(bets_df)}"
    )
    
    # Verify each game has the correct market type distribution
    for game_id in ["g1", "g2", "g3"]:
        game_rows = bets_df[bets_df["game_id"] == game_id]
        ml_count = (game_rows["market_type"] == "ML").sum()
        spread_count = (game_rows["market_type"] == "spread").sum()
        total_count = (game_rows["market_type"] == "total").sum()
        
        assert ml_count == 2, f"Game {game_id} should have 2 ML rows, got {ml_count}"
        assert spread_count == 2, f"Game {game_id} should have 2 spread rows, got {spread_count}"
        assert total_count == 2, f"Game {game_id} should have 2 total rows, got {total_count}"
        
        # Verify selection order within each market type
        ml_selections = game_rows[game_rows["market_type"] == "ML"]["selection"].tolist()
        away_team = game_rows.iloc[0]["away_team"]
        home_team = game_rows.iloc[0]["home_team"]
        assert ml_selections == [away_team, home_team], (
            f"Game {game_id} ML rows should be ordered [away, home], got {ml_selections}"
        )


def test_bets_invariant_no_rows_appended_after_canonical():
    """Verify that canonical 6 rows per game are created once, with no additional appends."""
    as_of = date(2025, 11, 10)
    
    schedule_df = pd.DataFrame([
        {
            "date": as_of.isoformat(),
            "status": "scheduled",
            "game_id": "g1",
            "away_team": "A",
            "home_team": "H",
            "home_win_prob": 0.6,
            "away_win_prob": 0.4,
            "win_prob_source": "model_x",
        }
    ])

    bets_df = _build_bets_dataframe(
        schedule_df,
        model_name="m",
        as_of_date=as_of,
        review_run_id="rr1",
    )

    # Should have exactly 6 rows
    assert len(bets_df) == 6, f"Expected 6 rows, got {len(bets_df)}"
    
    # Verify rows are in expected order: 2 ML, 2 spread, 2 total
    market_types = bets_df["market_type"].tolist()
    assert market_types == ["ML", "ML", "spread", "spread", "total", "total"], (
        f"Market types should be in order [ML, ML, spread, spread, total, total], got {market_types}"
    )


def test_bets_deduplicates_input_game_ids():
    """Verify that duplicate game_ids in input schedule are deduplicated before row creation."""
    as_of = date(2025, 11, 10)
    
    # Create input with the SAME game_id appearing twice (simulating multi-market or multi-model input)
    schedule_df = pd.DataFrame([
        {
            "date": as_of.isoformat(),
            "status": "scheduled",
            "game_id": "g1",
            "away_team": "A",
            "home_team": "H",
            "home_win_prob": 0.6,
            "away_win_prob": 0.4,
            "win_prob_source": "model_x",
        },
        # DUPLICATE: same game_id, possibly from a different market run
        {
            "date": as_of.isoformat(),
            "status": "scheduled",
            "game_id": "g1",  # Same game_id
            "away_team": "A",
            "home_team": "H",
            "home_win_prob": 0.55,  # Different probabilities
            "away_win_prob": 0.45,
            "win_prob_source": "model_y",
        },
    ])

    bets_df = _build_bets_dataframe(
        schedule_df,
        model_name="m",
        as_of_date=as_of,
        review_run_id="rr1",
    )

    # Should have exactly 6 rows (not 12), proving deduplication worked
    assert len(bets_df) == 6, (
        f"Expected 6 rows after deduplication, got {len(bets_df)}. "
        f"Duplicate game_ids in input should be handled."
    )
    
    # Verify game_id only appears once (6 times total, all for the same game)
    game_counts = bets_df["game_id"].value_counts()
    assert len(game_counts) == 1, f"Expected 1 unique game_id, got {len(game_counts)}"
    assert game_counts["g1"] == 6, f"Expected game g1 to have 6 rows, got {game_counts.get('g1')}"

