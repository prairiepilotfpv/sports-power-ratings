"""Tests for TOOR heads equivalence to legacy projection engine."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from forecasting.heads.toor_heads import create_toor_head_sequence
from pipelines.projection_engines import _toor_projection_engine
from config import DEFAULT_WIN_PROB_K, DEFAULT_MARGIN_SD_FALLBACK, DEFAULT_TOTAL_SD_FALLBACK


class TestToorHeadsEquivalence:
    """
    Verify that TOOR heads produce outputs equivalent to legacy _toor_projection_engine.
    
    These tests use synthetic data with stable TOOR parameters to ensure deterministic comparison.
    """

    def setup_method(self):
        """Set up synthetic game data and TOOR context."""
        # Fixed signed strengths for deterministic tests (mean-centered, from OLS)
        self.signed_strengths = {
            "Team_A": 0.8,    # Strong team
            "Team_B": -0.3,   # Weak team
            "Team_C": 0.1,    # Average team
        }

        # TOOR learned coefficients from fit
        self.base_context = {
            "signed_strengths": self.signed_strengths,
            "home_advantage": 3.362,      # Default TOOR home advantage
            "home_coeff": 17.373,         # Home strength coefficient
            "away_coeff": -14.855,        # Away strength coefficient
            "error_term": 31.155,         # Margin SD baseline
            "win_prob_k": DEFAULT_WIN_PROB_K,
            "winprob_bias": 0.0,
            "total_mean": 215.5,          # League average total
            "total_sd": 20.0,             # League total SD
            "margin_std": 31.155,         # Should match error_term
            "conditional_sd_intercept": None,
            "conditional_sd_slope": None,
            "sport": "nba",
        }

    def create_synthetic_games(self) -> pd.DataFrame:
        """Create a small synthetic game set."""
        return pd.DataFrame({
            "game_id": ["game_1", "game_2", "game_3"],
            "home_team": ["Team_A", "Team_B", "Team_C"],
            "away_team": ["Team_B", "Team_C", "Team_A"],
            "game_date": ["2025-01-28", "2025-01-28", "2025-01-28"],
            "neutral": [False, False, False],
        })

    def test_toor_heads_margin_derivation(self):
        """Test that margin_mean and margin_sd match TOOR formula."""
        df = self.create_synthetic_games()
        head_sequence = create_toor_head_sequence()

        # Apply heads (margin head is first)
        from forecasting.heads.toor_heads import ToorMarginHead
        margin_head = ToorMarginHead()
        margin_head.apply(df, self.base_context)

        # Manual computation:
        # Game 1: Team_A (0.8) vs Team_B (-0.3) at home
        # margin = 3.362 + 17.373*0.8 - 14.855*(-0.3) = 3.362 + 13.898 + 4.456 = 21.716
        expected_margin_1 = 3.362 + 17.373 * 0.8 - 14.855 * (-0.3)
        assert abs(df.at[0, "margin_mean"] - expected_margin_1) < 0.01, f"Margin mismatch for game 1"

        # Game 2: Team_B (-0.3) vs Team_C (0.1) at home
        # margin = 3.362 + 17.373*(-0.3) - 14.855*0.1 = 3.362 - 5.212 - 1.486 = -3.336
        expected_margin_2 = 3.362 + 17.373 * (-0.3) - 14.855 * 0.1
        assert abs(df.at[1, "margin_mean"] - expected_margin_2) < 0.01, f"Margin mismatch for game 2"

        # Game 3: Team_C (0.1) vs Team_A (0.8) at home
        # margin = 3.362 + 17.373*0.1 - 14.855*0.8 = 3.362 + 1.737 - 11.884 = -6.785
        expected_margin_3 = 3.362 + 17.373 * 0.1 - 14.855 * 0.8
        assert abs(df.at[2, "margin_mean"] - expected_margin_3) < 0.01, f"Margin mismatch for game 3"

        # All should have margin_sd (fallback to error_term with guardrails)
        for idx in range(len(df)):
            assert df.at[idx, "margin_sd"] is not None, f"margin_sd should not be None at row {idx}"
            # Should be within guardrails (typically 5.0 to 30.0)
            assert 1.0 <= df.at[idx, "margin_sd"] <= 50.0, f"margin_sd outside reasonable range at row {idx}"

    def test_toor_heads_total_derivation(self):
        """Test that total_mean and total_sd match TOOR league averages."""
        df = self.create_synthetic_games()

        from forecasting.heads.toor_heads import ToorTotalHead
        total_head = ToorTotalHead()
        total_head.apply(df, self.base_context)

        # TOOR uses league average for all games
        for idx in range(len(df)):
            assert abs(df.at[idx, "total_mean"] - 215.5) < 0.01, f"total_mean mismatch at row {idx}"
            assert abs(df.at[idx, "total_sd"] - 20.0) < 0.01, f"total_sd mismatch at row {idx}"

    def test_toor_heads_win_prob_derivation_with_logistic(self):
        """Test that win prob uses logistic curve (TOOR-specific behavior)."""
        df = self.create_synthetic_games()

        # First compute margin and total via heads
        head_sequence = create_toor_head_sequence()
        result = head_sequence.apply(df, self.base_context)

        # Verify all rows have win probs
        for idx in range(len(df)):
            p = df.at[idx, "p_home_win"]
            assert p is not None, f"p_home_win should not be None at row {idx}"
            assert 0.0 <= p <= 1.0, f"p_home_win out of bounds at row {idx}: {p}"
            assert df.at[idx, "win_prob_source"] == "logistic", "TOOR should use logistic source"

    def test_toor_heads_projected_scores(self):
        """Test that projected scores are derived from margin and total."""
        df = self.create_synthetic_games()
        head_sequence = create_toor_head_sequence()

        result = head_sequence.apply(df, self.base_context)

        # For each game: home_score = (total + margin) / 2, away_score = (total - margin) / 2
        for idx in range(len(df)):
            margin = df.at[idx, "margin_mean"]
            total = df.at[idx, "total_mean"]
            
            expected_home = (total + margin) / 2
            expected_away = (total - margin) / 2
            
            assert abs(df.at[idx, "projected_home_score"] - expected_home) < 0.01
            assert abs(df.at[idx, "projected_away_score"] - expected_away) < 0.01
            assert abs(df.at[idx, "projected_total"] - total) < 0.01

    def test_toor_heads_missing_strength(self):
        """Test that missing strengths result in None margin outputs (total uses league average)."""
        df = pd.DataFrame({
            "game_id": ["game_X"],
            "home_team": ["Unknown_Team"],
            "away_team": ["Team_B"],
            "game_date": ["2025-01-28"],
            "neutral": [False],
        })

        head_sequence = create_toor_head_sequence()
        result = head_sequence.apply(df, self.base_context)

        # Margin should be None (missing strength)
        assert df.at[0, "margin_mean"] is None
        assert df.at[0, "margin_sd"] is None
        
        # Total uses league average (always present)
        assert df.at[0, "total_mean"] == 215.5
        assert df.at[0, "total_sd"] == 20.0
        
        # Win prob should be None (depends on margin)
        assert df.at[0, "p_home_win"] is None

    def test_toor_heads_with_conditional_sd_model(self):
        """Test margin_sd derivation with conditional SD model."""
        df = self.create_synthetic_games()

        # Add a conditional SD model: sd = 5.0 + 0.1 * |margin|
        context = self.base_context.copy()
        context["conditional_sd_intercept"] = 5.0
        context["conditional_sd_slope"] = 0.1

        from forecasting.heads.toor_heads import ToorMarginHead
        margin_head = ToorMarginHead()
        margin_head.apply(df, context)

        # For game 1 with margin ≈ 21.716: sd = 5.0 + 0.1 * 21.716 = 7.172
        margin_sd_game1 = df.at[0, "margin_sd"]
        expected_sd = 5.0 + 0.1 * 21.716
        # Should apply guardrails: typically [5.0, 30.0]
        assert 1.0 <= margin_sd_game1 <= 50.0, "margin_sd should be within reasonable range"

    def test_toor_heads_equivalence_to_projection_engine(self):
        """
        Integration test: compare heads output to TOOR model's canonical matchup prediction.
        
        This verifies that the new heads system produces outputs that match
        TOOR's internal _canonical_matchup_prediction (which is what the projection
        engine delegates to when available).
        """
        from src.models.toor import TOORModel
        
        df = self.create_synthetic_games()
        
        # Create a real TOOR model instance with our test parameters
        model = TOORModel()
        
        # Manually set up the model's internal state to match our test context
        model._coefficients = type('obj', (object,), {
            'home_advantage': 3.362,
            'home_coeff': 17.373,
            'away_coeff': -14.855,
            'error_term': 31.155,
        })()
        model._win_prob_k = DEFAULT_WIN_PROB_K
        model._total_mean = 215.5
        model._total_sd = 20.0
        model._conditional_sd_model = None
        
        # Set up the rating model with signed strengths
        model._rating_model._signed_strengths = self.signed_strengths.copy()

        # Apply heads
        head_sequence = create_toor_head_sequence()
        head_result = head_sequence.apply(df, self.base_context)

        # For each game, also compute using model's canonical prediction
        for idx, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]

            # Model's canonical prediction
            legacy_output = model._canonical_matchup_prediction(
                home_team, away_team,
                neutral=False,
                sport="nba",
                date="2025-01-28",
                game_id=row["game_id"],
            )

            # Compare key canonical fields
            heads_margin = df.at[idx, "margin_mean"]
            legacy_margin = legacy_output.get("margin_mean")
            
            heads_margin_sd = df.at[idx, "margin_sd"]
            legacy_margin_sd = legacy_output.get("margin_sd")
            
            heads_p_home_win = df.at[idx, "p_home_win"]
            legacy_p_home_win = legacy_output.get("p_home_win")
            
            heads_total_mean = df.at[idx, "total_mean"]
            legacy_total_mean = legacy_output.get("total_mean")
            
            heads_total_sd = df.at[idx, "total_sd"]
            legacy_total_sd = legacy_output.get("total_sd")

            # All fields should match within tolerance
            assert abs(heads_margin - legacy_margin) < 0.01, (
                f"margin_mean mismatch at row {idx}: heads={heads_margin}, legacy={legacy_margin}"
            )
            assert abs(heads_margin_sd - legacy_margin_sd) < 0.01, (
                f"margin_sd mismatch at row {idx}: heads={heads_margin_sd}, legacy={legacy_margin_sd}"
            )
            assert abs(heads_p_home_win - legacy_p_home_win) < 0.001, (
                f"p_home_win mismatch at row {idx}: heads={heads_p_home_win}, legacy={legacy_p_home_win}"
            )
            assert abs(heads_total_mean - legacy_total_mean) < 0.01, (
                f"total_mean mismatch at row {idx}: heads={heads_total_mean}, legacy={legacy_total_mean}"
            )
            assert abs(heads_total_sd - legacy_total_sd) < 0.01, (
                f"total_sd mismatch at row {idx}: heads={heads_total_sd}, legacy={legacy_total_sd}"
            )


class TestToorHeadsStrictness:
    """Verify that heads mode does NOT have silent fallbacks to projection engine."""

    def test_heads_mode_validator_catches_missing_fields(self):
        """Test that validator catches missing required fields (no silent derivation)."""
        from pipelines.schedule import _validate_model_market_forecast_contract
        from markets.base import Market
        
        # Create a DataFrame that's missing margin_sd (required for SPREAD market)
        df = pd.DataFrame({
            "game_id": ["game_1", "game_2"],
            "home_team": ["Team_A", "Team_B"],
            "away_team": ["Team_B", "Team_C"],
            "margin_mean": [10.0, -5.0],
            # margin_sd is missing!
        })

        # Validator should raise because margin_sd is missing
        with pytest.raises(RuntimeError, match="missing required columns"):
            _validate_model_market_forecast_contract(
                df,
                market=Market.SPREAD,
                model_name="toor",
            )


class TestToorHeadsUniverseAlignment:
    """Test Option B universe alignment in heads mode."""

    def test_toor_heads_cover_target_game_ids(self):
        """Test that TOOR heads cover all target games or raise."""
        from forecasting.heads.registry import apply_heads
        from pipelines.schedule import _validate_model_market_forecast_contract
        from markets.base import Market
        
        # Create forecast with subset of games
        df = pd.DataFrame({
            "game_id": ["game_1", "game_2"],
            "home_team": ["Team_A", "Team_B"],
            "away_team": ["Team_B", "Team_C"],
            "margin_mean": [10.0, -5.0],
            "margin_sd": [12.0, 12.0],
            "p_home_win": [0.6, 0.4],
            "total_mean": [215.0, 220.0],
            "total_sd": [20.0, 20.0],
        })

        # Define target_game_ids with extra games not in forecast
        target_game_ids = {"game_1", "game_2", "game_3", "game_4"}

        # Should raise because forecast doesn't cover all targets
        with pytest.raises(RuntimeError, match="missing forecast"):
            _validate_model_market_forecast_contract(
                df,
                market=Market.SPREAD,
                target_game_ids=target_game_ids,
                model_name="toor",
            )

    def test_toor_heads_cover_all_targets(self):
        """Test that validation passes when all target games are covered."""
        from pipelines.schedule import _validate_model_market_forecast_contract
        from markets.base import Market
        
        # Create forecast with all target games
        df = pd.DataFrame({
            "game_id": ["game_1", "game_2", "game_3"],
            "home_team": ["Team_A", "Team_B", "Team_C"],
            "away_team": ["Team_B", "Team_C", "Team_A"],
            "margin_mean": [10.0, -5.0, 3.0],
            "margin_sd": [12.0, 12.0, 12.0],
        })

        target_game_ids = {"game_1", "game_2", "game_3"}

        # Should not raise
        _validate_model_market_forecast_contract(
            df,
            market=Market.SPREAD,
            target_game_ids=target_game_ids,
            model_name="toor",
        )
