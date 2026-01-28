"""Test BETS sheet: always creates 6 rows per game with ensemble-sourced SPREAD/TOTAL."""

from datetime import date

import pandas as pd
import pytest

from src.pipelines.schedule import _build_bets_dataframe


class TestBetsEnsembleGating:
    """Test suite for BETS sheet builder: always 6 rows per game, ensemble sources for SPREAD/TOTAL."""

    def _create_sample_schedule(self, num_games: int = 2) -> pd.DataFrame:
        """Create a sample schedule dataframe with N games."""
        games = []
        for i in range(num_games):
            games.append({
                "date": date(2025, 11, 10).isoformat(),
                "status": "scheduled",
                "game_id": f"game_{i}",
                "away_team": f"Team_A{i}",
                "home_team": f"Team_H{i}",
                "home_win_prob": 0.60 + (0.01 * i),
                "away_win_prob": 0.40 - (0.01 * i),
                "win_prob_source": "model_x",
                "ml_ensemble_components_json": '{"models": ["model_x"], "prob": 0.60}',
                "margin_mean": 3.0 + i,
                "margin_sd": 1.2,
                "spread_source": "ensemble_spread_v1",
                "spread_ensemble_components_json": '{"models": ["model_x"], "margin_mean": 3.0}',
                "total": 210.5 + (5 * i),
                "total_sd": 5.0,
                "total_source": "ensemble_total_v1",
                "total_ensemble_components_json": '{"models": ["model_x"], "total": 210.5}',
            })
        return pd.DataFrame(games)

    def test_always_6_rows_per_game_regardless_of_flags(self):
        """
        Core invariant: BETS always contains exactly 6 rows per game (2 ML, 2 SPREAD, 2 TOTAL).
        Ensemble flags are deprecated and should not affect row count anymore.
        """
        schedule_df = self._create_sample_schedule(num_games=2)
        
        # Test all 4 combinations of deprecated flags
        for spread_flag in [True, False]:
            for total_flag in [True, False]:
                bets_df = _build_bets_dataframe(
                    schedule_df,
                    model_name="test_model",
                    as_of_date=date(2025, 11, 10),
                    review_run_id="review_1",
                    spread_ensemble_applied=spread_flag,
                    total_ensemble_applied=total_flag,
                )
                
                # Always 6 rows per game (invariant)
                assert len(bets_df) == 12, (
                    f"Expected 12 rows (6 per 2 games), got {len(bets_df)} "
                    f"(spread_flag={spread_flag}, total_flag={total_flag})"
                )
                
                game_counts = bets_df["game_id"].value_counts()
                assert (game_counts == 6).all(), (
                    f"All games should have exactly 6 rows, got: {game_counts.to_dict()} "
                    f"(spread_flag={spread_flag}, total_flag={total_flag})"
                )

    def test_market_distribution_always_2_2_2(self):
        """Verify market type distribution: always 2 ML, 2 SPREAD, 2 TOTAL per game."""
        schedule_df = self._create_sample_schedule(num_games=3)
        
        bets_df = _build_bets_dataframe(
            schedule_df,
            model_name="test_model",
            as_of_date=date(2025, 11, 10),
            review_run_id="review_1",
            spread_ensemble_applied=False,
            total_ensemble_applied=False,
        )
        
        for game_id in bets_df["game_id"].unique():
            game_rows = bets_df[bets_df["game_id"] == game_id]
            ml_count = (game_rows["market_type"] == "ML").sum()
            spread_count = (game_rows["market_type"] == "spread").sum()
            total_count = (game_rows["market_type"] == "total").sum()
            
            assert ml_count == 2, f"Game {game_id}: expected 2 ML rows, got {ml_count}"
            assert spread_count == 2, f"Game {game_id}: expected 2 SPREAD rows, got {spread_count}"
            assert total_count == 2, f"Game {game_id}: expected 2 TOTAL rows, got {total_count}"

    def test_spread_rows_always_ensemble_sourced(self):
        """SPREAD rows always have source_id='ensemble_spread_v1' (no direct variants)."""
        schedule_df = self._create_sample_schedule(num_games=2)
        
        bets_df = _build_bets_dataframe(
            schedule_df,
            model_name="test_model",
            as_of_date=date(2025, 11, 10),
            review_run_id="review_1",
        )
        
        spread_rows = bets_df[bets_df["market_type"] == "spread"]
        
        # All SPREAD rows should have spread_source='ensemble_spread_v1'
        assert len(spread_rows) == 4, f"Expected 4 SPREAD rows (2 games × 2), got {len(spread_rows)}"
        assert (spread_rows["spread_source"] == "ensemble_spread_v1").all(), (
            "All SPREAD rows should have spread_source='ensemble_spread_v1'. "
            f"Got: {spread_rows['spread_source'].unique()}"
        )
        
        # Verify no duplicate SPREAD sources per game (only ensemble-sourced)
        for game_id in spread_rows["game_id"].unique():
            game_spread_rows = spread_rows[spread_rows["game_id"] == game_id]
            unique_sources = game_spread_rows["spread_source"].unique()
            assert len(unique_sources) == 1, (
                f"Game {game_id}: should have exactly 1 SPREAD source, got {len(unique_sources)}: {unique_sources}"
            )

    def test_total_rows_always_ensemble_sourced(self):
        """TOTAL rows always have source_id='ensemble_total_v1' (no direct variants)."""
        schedule_df = self._create_sample_schedule(num_games=2)
        
        bets_df = _build_bets_dataframe(
            schedule_df,
            model_name="test_model",
            as_of_date=date(2025, 11, 10),
            review_run_id="review_1",
        )
        
        total_rows = bets_df[bets_df["market_type"] == "total"]
        
        # All TOTAL rows should have total_source='ensemble_total_v1'
        assert len(total_rows) == 4, f"Expected 4 TOTAL rows (2 games × 2), got {len(total_rows)}"
        assert (total_rows["total_source"] == "ensemble_total_v1").all(), (
            "All TOTAL rows should have total_source='ensemble_total_v1'. "
            f"Got: {total_rows['total_source'].unique()}"
        )
        
        # Verify no duplicate TOTAL sources per game (only ensemble-sourced)
        for game_id in total_rows["game_id"].unique():
            game_total_rows = total_rows[total_rows["game_id"] == game_id]
            unique_sources = game_total_rows["total_source"].unique()
            assert len(unique_sources) == 1, (
                f"Game {game_id}: should have exactly 1 TOTAL source, got {len(unique_sources)}: {unique_sources}"
            )

    def test_ml_rows_unchanged_regardless_of_ensemble_flags(self):
        """
        ML rows should always be present and unchanged regardless of deprecated ensemble flags.
        """
        schedule_df = self._create_sample_schedule(num_games=1)
        base_home_prob = schedule_df.iloc[0]["home_win_prob"]
        base_away_prob = schedule_df.iloc[0]["away_win_prob"]
        
        # Test all 4 combinations of deprecated flags
        for spread_flag in [True, False]:
            for total_flag in [True, False]:
                bets_df = _build_bets_dataframe(
                    schedule_df,
                    model_name="test_model",
                    as_of_date=date(2025, 11, 10),
                    review_run_id="review_1",
                    spread_ensemble_applied=spread_flag,
                    total_ensemble_applied=total_flag,
                )
                
                # ML rows should always be 2 per game
                ml_rows = bets_df[bets_df["market_type"] == "ML"]
                assert len(ml_rows) == 2, (
                    f"Expected 2 ML rows (spread={spread_flag}, total={total_flag}), got {len(ml_rows)}"
                )
                
                # ML probabilities should match input (home_win_prob and away_win_prob)
                home_row = ml_rows[ml_rows["selection"] == schedule_df.iloc[0]["home_team"]]
                away_row = ml_rows[ml_rows["selection"] == schedule_df.iloc[0]["away_team"]]
                
                assert len(home_row) == 1, "Expected 1 ML row for home team"
                assert len(away_row) == 1, "Expected 1 ML row for away team"
                
                # ML rows should have win_prob_source from input
                assert home_row.iloc[0]["win_prob_source"] == "model_x", (
                    f"ML win_prob_source should be 'model_x', got {home_row.iloc[0]['win_prob_source']}"
                )

    def test_no_duplicate_direct_ensemble_variants(self):
        """
        Verify no duplicate direct+ensemble variants exist.
        Each market type per game should have only one source_id.
        """
        schedule_df = self._create_sample_schedule(num_games=2)
        
        bets_df = _build_bets_dataframe(
            schedule_df,
            model_name="test_model",
            as_of_date=date(2025, 11, 10),
            review_run_id="review_1",
        )
        
        # Check SPREAD: should all be ensemble_spread_v1 (no direct duplicates)
        spread_rows = bets_df[bets_df["market_type"] == "spread"]
        spread_sources = spread_rows.groupby("game_id")["spread_source"].nunique()
        assert (spread_sources == 1).all(), (
            f"Each game should have exactly 1 SPREAD source (no duplicates). Got: {spread_sources.to_dict()}"
        )
        
        # Check TOTAL: should all be ensemble_total_v1 (no direct duplicates)
        total_rows = bets_df[bets_df["market_type"] == "total"]
        total_sources = total_rows.groupby("game_id")["total_source"].nunique()
        assert (total_sources == 1).all(), (
            f"Each game should have exactly 1 TOTAL source (no duplicates). Got: {total_sources.to_dict()}"
        )

    def test_empty_schedule_returns_empty_bets(self):
        """Empty schedule should return empty BETS dataframe."""
        empty_df = pd.DataFrame({
            "date": [],
            "status": [],
            "game_id": [],
            "away_team": [],
            "home_team": [],
        })
        
        bets_df = _build_bets_dataframe(
            empty_df,
            model_name="test_model",
            as_of_date=date(2025, 11, 10),
            review_run_id="review_1",
        )
        
        assert bets_df.empty, "Empty schedule should produce empty BETS"

    def test_large_game_count_always_6_rows_per_game(self):
        """
        Verify that large game counts always produce 6 rows per game.
        """
        schedule_df = self._create_sample_schedule(num_games=50)
        
        bets_df = _build_bets_dataframe(
            schedule_df,
            model_name="test_model",
            as_of_date=date(2025, 11, 10),
            review_run_id="review_1",
        )
        
        # Always 6 rows per game regardless of input size
        assert len(bets_df) == 300, f"Expected 300 rows (50 games × 6), got {len(bets_df)}"
        
        game_counts = bets_df["game_id"].value_counts()
        assert (game_counts == 6).all(), (
            f"All 50 games should have exactly 6 rows each. Got: {game_counts.value_counts().to_dict()}"
        )

    def test_selection_ordering_away_home_over_under(self):
        """
        Verify consistent selection ordering within each market type per game.
        ML: away, home | SPREAD: away, home | TOTAL: Over, Under
        """
        schedule_df = self._create_sample_schedule(num_games=1)
        away_team = schedule_df.iloc[0]["away_team"]
        home_team = schedule_df.iloc[0]["home_team"]
        
        bets_df = _build_bets_dataframe(
            schedule_df,
            model_name="test_model",
            as_of_date=date(2025, 11, 10),
            review_run_id="review_1",
        )
        
        game_rows = bets_df[bets_df["game_id"] == schedule_df.iloc[0]["game_id"]]
        
        # Extract selections in order
        ml_selections = game_rows[game_rows["market_type"] == "ML"]["selection"].tolist()
        spread_selections = game_rows[game_rows["market_type"] == "spread"]["selection"].tolist()
        total_selections = game_rows[game_rows["market_type"] == "total"]["selection"].tolist()
        
        assert ml_selections == [away_team, home_team], (
            f"ML selections should be [away, home], got {ml_selections}"
        )
        assert spread_selections == [away_team, home_team], (
            f"SPREAD selections should be [away, home], got {spread_selections}"
        )
        assert total_selections == ["Over", "Under"], (
            f"TOTAL selections should be [Over, Under], got {total_selections}"
        )
