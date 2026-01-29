"""Tests for GSSD heads equivalence to legacy projection engine."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from forecasting.heads.gssd_heads import create_gssd_head_sequence
from models.gssd import GSSDModel, GSSDPowerRating, GSSDCalibration
from config import DEFAULT_WIN_PROB_K, DEFAULT_MARGIN_SD_FALLBACK, DEFAULT_TOTAL_SD_FALLBACK


class TestGssdHeadsEquivalence:
    """
    Verify that GSSD heads produce outputs equivalent to legacy GSSD.project_matchup.
    
    These tests use synthetic team stats and GSSD calibration to ensure deterministic comparison.
    """

    def setup_method(self):
        """Set up synthetic GSSD data."""
        # Synthetic team scoring stats (pfh, pah, pfa, paa)
        self.team_stats = {
            "Team_A": {
                "pfh": 112.5,  # Points for at home
                "pah": 108.2,  # Points against at home
                "pfa": 105.8,  # Points for away
                "paa": 110.3,  # Points against away
            },
            "Team_B": {
                "pfh": 98.5,   # Weaker team
                "pah": 102.1,
                "pfa": 96.2,
                "paa": 103.4,
            },
            "Team_C": {
                "pfh": 104.0,  # Average team
                "pah": 104.0,
                "pfa": 104.0,
                "paa": 104.0,
            },
        }

        # GSSD calibration coefficients (from fit)
        self.base_context = {
            "team_stats": self.team_stats,
            "intercept": 0.5,
            "beta_pfh": 0.15,
            "beta_pah": -0.12,
            "beta_pfa": -0.14,
            "beta_paa": 0.11,
            "home_advantage_points": 2.8,
            "error_term": 12.5,
            "conditional_sd_intercept": None,
            "conditional_sd_slope": None,
            "total_mean": 206.5,       # League average total
            "total_sd": 15.0,          # League total SD
            "win_prob_k": DEFAULT_WIN_PROB_K,
            "winprob_bias": 0.0,
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

    def test_gssd_heads_margin_derivation(self):
        """Test that margin_mean and margin_sd match GSSD formula."""
        df = self.create_synthetic_games()
        head_sequence = create_gssd_head_sequence()

        # Apply heads (margin head is first)
        from forecasting.heads.gssd_heads import GssdMarginHead
        margin_head = GssdMarginHead()
        margin_head.apply(df, self.base_context)

        # Manual computation:
        # margin = intercept + beta_pfh*pfh + beta_pah*pah + beta_pfa*pfa + beta_paa*paa + home_advantage*(0 if neutral else 1)
        
        # Game 1: Team_A (home) vs Team_B (away)
        a_stats = self.team_stats["Team_A"]
        b_stats = self.team_stats["Team_B"]
        expected_margin_1 = (
            0.5 
            + 0.15 * a_stats["pfh"] 
            + (-0.12) * a_stats["pah"] 
            + (-0.14) * b_stats["pfa"] 
            + 0.11 * b_stats["paa"]
            + 2.8  # home advantage
        )
        assert abs(df.at[0, "margin_mean"] - expected_margin_1) < 0.1, f"Margin mismatch for game 1"

        # Game 2: Team_B (home) vs Team_C (away)
        c_stats = self.team_stats["Team_C"]
        expected_margin_2 = (
            0.5 
            + 0.15 * b_stats["pfh"] 
            + (-0.12) * b_stats["pah"] 
            + (-0.14) * c_stats["pfa"] 
            + 0.11 * c_stats["paa"]
            + 2.8
        )
        assert abs(df.at[1, "margin_mean"] - expected_margin_2) < 0.1, f"Margin mismatch for game 2"

        # All should have margin_sd
        for idx in range(len(df)):
            assert df.at[idx, "margin_sd"] is not None, f"margin_sd should not be None at row {idx}"
            assert df.at[idx, "margin_sd"] > 0, f"margin_sd should be positive at row {idx}"
            assert 1.0 <= df.at[idx, "margin_sd"] <= 50.0, f"margin_sd outside reasonable range at row {idx}"

    def test_gssd_heads_total_derivation(self):
        """Test that total_mean is computed from team stats when available."""
        df = self.create_synthetic_games()

        from forecasting.heads.gssd_heads import GssdTotalHead
        total_head = GssdTotalHead()
        total_head.apply(df, self.base_context)

        # GSSD computes total from team stats: (pfh + paa + pfa + pah) / 2
        # Game 1: Team_A vs Team_B
        a_stats = self.team_stats["Team_A"]
        b_stats = self.team_stats["Team_B"]
        expected_total_1 = (a_stats["pfh"] + b_stats["paa"] + b_stats["pfa"] + a_stats["pah"]) / 2.0
        assert abs(df.at[0, "total_mean"] - expected_total_1) < 0.1, f"total_mean mismatch for game 1"

        # All should have total_sd (league average)
        for idx in range(len(df)):
            assert abs(df.at[idx, "total_sd"] - 15.0) < 0.01, f"total_sd mismatch at row {idx}"

    def test_gssd_heads_win_prob_derivation_with_logistic(self):
        """Test that win prob uses logistic curve (GSSD-specific behavior)."""
        df = self.create_synthetic_games()

        # Apply full head sequence
        head_sequence = create_gssd_head_sequence()
        result = head_sequence.apply(df, self.base_context)

        # Verify all rows have win probs
        for idx in range(len(df)):
            p = df.at[idx, "p_home_win"]
            assert p is not None, f"p_home_win should not be None at row {idx}"
            assert np.isfinite(p), f"p_home_win should be finite at row {idx}: {p}"
            assert 0.0 <= p <= 1.0, f"p_home_win out of bounds at row {idx}: {p}"
            assert df.at[idx, "win_prob_source"] == "logistic", "GSSD should use logistic source"

    def test_gssd_heads_projected_scores(self):
        """Test that projected scores are derived from margin and total."""
        df = self.create_synthetic_games()
        head_sequence = create_gssd_head_sequence()

        result = head_sequence.apply(df, self.base_context)

        # For each game: home_score = (total + margin) / 2, away_score = (total - margin) / 2
        for idx in range(len(df)):
            margin = df.at[idx, "margin_mean"]
            total = df.at[idx, "total_mean"]
            
            expected_home = (total + margin) / 2
            expected_away = (total - margin) / 2
            
            assert abs(df.at[idx, "projected_home_score"] - expected_home) < 0.1
            assert abs(df.at[idx, "projected_away_score"] - expected_away) < 0.1
            assert abs(df.at[idx, "projected_total"] - total) < 0.01

    def test_gssd_heads_missing_team_stats(self):
        """Test that missing team stats result in None margin outputs."""
        df = pd.DataFrame({
            "game_id": ["game_X"],
            "home_team": ["Unknown_Team"],
            "away_team": ["Team_B"],
            "game_date": ["2025-01-28"],
            "neutral": [False],
        })

        head_sequence = create_gssd_head_sequence()
        result = head_sequence.apply(df, self.base_context)

        # Margin should be None (missing team stats)
        assert df.at[0, "margin_mean"] is None
        assert df.at[0, "margin_sd"] is None

    def test_gssd_heads_nan_prevention_in_margin_sd(self):
        """Test that margin_sd guardrails ensure valid values (not NaN)."""
        df = self.create_synthetic_games()

        # Context with negative error_term will still produce valid margin_sd via fallback
        bad_context = self.base_context.copy()
        bad_context["error_term"] = -1.0  # Invalid: negative SD
        bad_context["conditional_sd_intercept"] = None
        bad_context["conditional_sd_slope"] = None

        from forecasting.heads.gssd_heads import GssdMarginHead
        margin_head = GssdMarginHead()

        # Should NOT raise (guardrail_margin_sd handles fallback)
        margin_head.apply(df, bad_context)
        
        # Verify all margin_sd values are valid (not NaN, positive)
        for idx in range(len(df)):
            margin_sd = df.at[idx, "margin_sd"]
            assert margin_sd is not None, f"margin_sd should not be None at row {idx}"
            assert np.isfinite(margin_sd), f"margin_sd should be finite at row {idx}"
            assert margin_sd > 0, f"margin_sd should be positive at row {idx}"


    def test_gssd_heads_nan_prevention_in_win_prob(self):
        """Test that invalid p_home_win raises RuntimeError (no silent NaN)."""
        df = self.create_synthetic_games()

        # Set up a DataFrame with valid margin/sd but will trigger nan in logistic
        # Use very large positive/negative margins (but not so extreme as to overflow math)
        df["margin_mean"] = [500.0, -500.0, 0.0]  # Extreme but realistic bounds
        df["margin_sd"] = [12.0, 12.0, 12.0]

        from forecasting.heads.gssd_heads import GssdWinProbHead
        win_prob_head = GssdWinProbHead()

        # Should handle extreme margins gracefully (logistic should still return valid [0,1])
        # Or raise if it's genuinely invalid
        try:
            win_prob_head.apply(df, self.base_context)
            # If no exception, verify all are valid
            for idx in range(len(df)):
                p = df.at[idx, "p_home_win"]
                if p is not None:
                    assert np.isfinite(p), f"p_home_win should be finite at row {idx}"
                    assert 0.0 <= p <= 1.0, f"p_home_win should be in [0,1] at row {idx}"
        except RuntimeError as e:
            # Expected if logistic computation fails
            assert "p_home_win produced invalid value" in str(e)


    def test_gssd_heads_neutral_site_no_home_advantage(self):
        """Test that neutral sites apply 0 home advantage."""
        df = pd.DataFrame({
            "game_id": ["neutral_game"],
            "home_team": ["Team_A"],
            "away_team": ["Team_B"],
            "game_date": ["2025-01-28"],
            "neutral": [True],  # Neutral site
        })

        from forecasting.heads.gssd_heads import GssdMarginHead
        margin_head = GssdMarginHead()
        margin_head.apply(df, self.base_context)

        # Manual computation with 0 home advantage (neutral = True)
        a_stats = self.team_stats["Team_A"]
        b_stats = self.team_stats["Team_B"]
        expected_margin = (
            0.5 
            + 0.15 * a_stats["pfh"] 
            + (-0.12) * a_stats["pah"] 
            + (-0.14) * b_stats["pfa"] 
            + 0.11 * b_stats["paa"]
            + 0.0  # No home advantage at neutral site
        )
        assert abs(df.at[0, "margin_mean"] - expected_margin) < 0.1


class TestGssdHeadsContractValidation:
    """Test GSSD heads strict contract validation."""

    def test_gssd_margin_sd_must_be_positive(self):
        """Test that margin_sd <= 0 is caught and raises."""
        from pipelines.schedule import _validate_model_market_forecast_contract
        from markets.base import Market

        df = pd.DataFrame({
            "game_id": ["game_1", "game_2"],
            "home_team": ["Team_A", "Team_B"],
            "away_team": ["Team_B", "Team_C"],
            "margin_mean": [10.0, 5.0],
            "margin_sd": [12.0, 0.0],  # Second row has invalid SD
        })

        with pytest.raises(RuntimeError, match="non-positive.*margin_sd"):
            _validate_model_market_forecast_contract(
                df,
                market=Market.SPREAD,
                model_name="gssd",
            )

    def test_gssd_p_home_win_must_be_finite(self):
        """Test that non-finite p_home_win would be caught by validation."""
        from pipelines.schedule import _validate_model_market_forecast_contract
        from markets.base import Market

        df = pd.DataFrame({
            "game_id": ["game_1", "game_2"],
            "home_team": ["Team_A", "Team_B"],
            "away_team": ["Team_B", "Team_C"],
            "p_home_win": [0.6, np.inf],  # Second row has infinite prob
        })

        with pytest.raises(RuntimeError, match="non-finite"):
            _validate_model_market_forecast_contract(
                df,
                market=Market.ML,
                model_name="gssd",
            )


class TestGssdHeadsOptionB:
    """Test GSSD heads Option B universe alignment."""

    def test_gssd_heads_cover_all_target_games(self):
        """Test that validation passes when all target games are covered."""
        from pipelines.schedule import _validate_model_market_forecast_contract
        from markets.base import Market

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
            model_name="gssd",
        )

    def test_gssd_heads_missing_target_games_raises(self):
        """Test that missing target games raises."""
        from pipelines.schedule import _validate_model_market_forecast_contract
        from markets.base import Market

        df = pd.DataFrame({
            "game_id": ["game_1", "game_2"],
            "home_team": ["Team_A", "Team_B"],
            "away_team": ["Team_B", "Team_C"],
            "margin_mean": [10.0, -5.0],
            "margin_sd": [12.0, 12.0],
        })

        target_game_ids = {"game_1", "game_2", "game_3", "game_4"}

        # Should raise because games 3 and 4 are missing
        with pytest.raises(RuntimeError, match="missing forecast"):
            _validate_model_market_forecast_contract(
                df,
                market=Market.SPREAD,
                target_game_ids=target_game_ids,
                model_name="gssd",
            )
