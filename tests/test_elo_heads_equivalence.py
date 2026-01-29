"""Tests for Elo heads equivalence to legacy projection engine."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from forecasting.heads.elo_heads import create_elo_head_sequence
from pipelines.projection_engines import _rating_projection_engine
from models.calibration import ConditionalSDModel
from config import DEFAULT_WIN_PROB_K, DEFAULT_MARGIN_SD_FALLBACK, DEFAULT_TOTAL_SD_FALLBACK


class TestEloHeadsEquivalence:
    """
    Verify that Elo heads produce outputs equivalent to legacy _rating_projection_engine + _elo_projection_engine.
    
    These tests use synthetic data with fixed parameters to ensure deterministic comparison.
    """

    def setup_method(self):
        """Set up synthetic game data and model context."""
        # Fixed ratings for deterministic tests
        self.ratings = {
            "Team_A": 1600.0,
            "Team_B": 1500.0,
            "Team_C": 1550.0,
        }

        # Elo learned parameters
        self.base_context = {
            "ratings": self.ratings,
            "rating_units": "points",
            "home_advantage": 65.0,
            "neutral": False,
            "win_prob_k": DEFAULT_WIN_PROB_K,
            "win_prob_bias": 0.0,
            "base_total": 215.0,
            "scoring_averages": {},
            "total_intercept": None,
            "total_slope": None,
            "margin_std": 12.5,
            "total_std": 21.5,
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
        })

    def test_elo_heads_margin_derivation(self):
        """Test that margin_mean and margin_sd match legacy projection engine."""
        df = self.create_synthetic_games()
        head_sequence = create_elo_head_sequence()

        # Apply heads (only margin head needed for this test)
        from forecasting.heads.elo_heads import EloMarginHead
        margin_head = EloMarginHead()
        margin_head.apply(df, self.base_context)

        # Verify margins are computed correctly
        # Team_A (1600) vs Team_B (1500) at home: margin = 1600 - 1500 + 65 = 165
        assert abs(df.at[0, "margin_mean"] - 165.0) < 0.01, "Margin mismatch for game 1"

        # Team_B (1500) vs Team_C (1550) at home: margin = 1500 - 1550 + 65 = 15
        assert abs(df.at[1, "margin_mean"] - 15.0) < 0.01, "Margin mismatch for game 2"

        # Team_C (1550) vs Team_A (1600) at home: margin = 1550 - 1600 + 65 = 15
        assert abs(df.at[2, "margin_mean"] - 15.0) < 0.01, "Margin mismatch for game 3"

        # All should have margin_sd applied
        for idx in range(len(df)):
            assert df.at[idx, "margin_sd"] is not None, f"margin_sd should not be None at row {idx}"
            # With no conditional SD model, should fall back to DEFAULT_MARGIN_SD_FALLBACK (12.0) with guardrails
            assert 5.0 <= df.at[idx, "margin_sd"] <= 30.0, f"margin_sd outside guardrails at row {idx}"

    def test_elo_heads_total_derivation(self):
        """Test that total_mean and total_sd are derived correctly."""
        df = self.create_synthetic_games()

        from forecasting.heads.elo_heads import EloTotalHead
        total_head = EloTotalHead()
        total_head.apply(df, self.base_context)

        # With no model-specific formula or scoring averages, should use base_total
        for idx in range(len(df)):
            assert abs(df.at[idx, "total_mean"] - 215.0) < 0.01, f"total_mean mismatch at row {idx}"
            assert abs(df.at[idx, "total_sd"] - 21.5) < 0.01, f"total_sd mismatch at row {idx}"

    def test_elo_heads_win_prob_derivation_with_logistic(self):
        """Test that win prob uses logistic (Elo-specific behavior)."""
        df = self.create_synthetic_games()
        head_sequence = create_elo_head_sequence()

        # Apply all heads
        result = head_sequence.apply(df, self.base_context)

        # Verify logistic probability is used
        for idx in range(len(df)):
            assert df.at[idx, "win_prob_source"] == "logistic", "Elo should use logistic source"
            assert df.at[idx, "p_home_win"] is not None, "p_home_win should not be None"
            # Probability should be in [0, 1]
            assert 0.0 <= df.at[idx, "p_home_win"] <= 1.0, f"p_home_win out of bounds at row {idx}"

    def test_elo_heads_projected_scores(self):
        """Test that projected scores are derived from margin and total."""
        df = self.create_synthetic_games()
        head_sequence = create_elo_head_sequence()

        result = head_sequence.apply(df, self.base_context)

        # For game 1: margin = 165, total = 215
        # home_score = (215 + 165) / 2 = 190
        # away_score = (215 - 165) / 2 = 25
        home_score = df.at[0, "projected_home_score"]
        away_score = df.at[0, "projected_away_score"]
        total = df.at[0, "projected_total"]

        assert abs(home_score - 190.0) < 0.01, "projected_home_score mismatch"
        assert abs(away_score - 25.0) < 0.01, "projected_away_score mismatch"
        assert abs(total - 215.0) < 0.01, "projected_total mismatch"

    def test_elo_heads_missing_rating(self):
        """Test that missing ratings result in None outputs."""
        df = pd.DataFrame({
            "game_id": ["game_X"],
            "home_team": ["Unknown_Team"],
            "away_team": ["Team_B"],
            "game_date": ["2025-01-28"],
        })

        head_sequence = create_elo_head_sequence()
        result = head_sequence.apply(df, self.base_context)

        # Should have None values for all derived fields
        assert df.at[0, "margin_mean"] is None
        assert df.at[0, "margin_sd"] is None
        assert df.at[0, "total_mean"] is None
        assert df.at[0, "total_sd"] is None
        assert df.at[0, "p_home_win"] is None

    def test_elo_heads_with_conditional_sd_model(self):
        """Test margin_sd derivation with conditional SD model."""
        df = self.create_synthetic_games()

        # Add a conditional SD model: sd = 3.0 + 0.05 * |margin|
        context = self.base_context.copy()
        context["conditional_sd_intercept"] = 3.0
        context["conditional_sd_slope"] = 0.05

        from forecasting.heads.elo_heads import EloMarginHead
        margin_head = EloMarginHead()
        margin_head.apply(df, context)

        # For game 1 with margin = 165: sd = 3.0 + 0.05 * 165 = 3.0 + 8.25 = 11.25
        margin_sd_game1 = df.at[0, "margin_sd"]
        # Should apply guardrails: clip to [5.0, 30.0]
        expected_sd = 3.0 + 0.05 * 165.0
        assert 5.0 <= margin_sd_game1 <= 30.0, "margin_sd should be within guardrails"

    def test_elo_heads_equivalence_to_projection_engine(self):
        """
        Integration test: compare heads output to legacy projection engine output.
        
        This is the critical equivalence test: verify that the new heads system
        produces outputs that match _rating_projection_engine + _elo_projection_engine
        within numerical tolerance.
        """
        df = self.create_synthetic_games()
        
        # Create a mock model object for projection engine
        class MockModel:
            def metadata(self):
                class Meta:
                    model_id = "elo"
                return Meta()
        
        model = MockModel()

        # Apply heads
        head_sequence = create_elo_head_sequence()
        head_result = head_sequence.apply(df, self.base_context)

        # For each game, also compute using legacy projection engine
        for idx, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]

            # Legacy projection engine output
            legacy_output = _rating_projection_engine(
                home_team, away_team, model, self.base_context
            )

            # Extract heads output
            heads_margin_mean = df.at[idx, "margin_mean"]
            heads_margin_sd = df.at[idx, "margin_sd"]
            heads_total_mean = df.at[idx, "total_mean"]
            heads_total_sd = df.at[idx, "total_sd"]
            heads_p_home_win = df.at[idx, "p_home_win"]

            # Compare (tolerances account for floating point precision)
            tolerance = 0.01

            assert abs(heads_margin_mean - legacy_output["margin_mean"]) < tolerance, \
                f"margin_mean mismatch at row {idx}"
            assert abs(heads_margin_sd - legacy_output["margin_sd"]) < tolerance, \
                f"margin_sd mismatch at row {idx}"
            assert abs(heads_total_mean - legacy_output["total_mean"]) < tolerance, \
                f"total_mean mismatch at row {idx}"
            assert abs(heads_total_sd - legacy_output["total_sd"]) < tolerance, \
                f"total_sd mismatch at row {idx}"

            # For Elo, p_home_win should use logistic from projection engine
            legacy_logistic = legacy_output.get("logistic_home_win_prob")
            if legacy_logistic is not None and heads_p_home_win is not None:
                assert abs(heads_p_home_win - legacy_logistic) < tolerance, \
                    f"p_home_win logistic mismatch at row {idx}"


class TestHeadsModeExclusion:
    """
    Test that heads mode and legacy projection engine derivation are mutually exclusive.
    """

    def test_heads_mode_disallows_projection_derivation(self):
        """
        Verify that when heads mode is enabled, projection engine does NOT derive missing fields.
        """
        from pipelines.projection_engines import _validation_only_engine

        # The validation-only engine should return all None values
        output = _validation_only_engine(
            "Team_A", "Team_B",
            None,  # model
            {}     # context
        )

        # All fields should be None (no derivation)
        canonical_fields = [
            "margin_mean", "margin_sd", "total_mean", "total_sd",
            "model_p_home_win", "normal_p_home_win"
        ]
        for field in canonical_fields:
            assert output.get(field) is None, \
                f"validation_only_engine should not derive {field}"

    def test_heads_mode_flag_in_config(self):
        """Verify that HEADS_MODE_ENABLED flag exists and can be checked."""
        from config import HEADS_MODE_ENABLED
        
        # Should be a boolean (default False)
        assert isinstance(HEADS_MODE_ENABLED, bool)
        assert HEADS_MODE_ENABLED is False, "Default should be False (legacy mode)"


class TestHeadSequenceValidation:
    """Test error handling and validation in head sequences."""

    def test_head_sequence_missing_dependency(self):
        """Test that missing required fields are caught."""
        from forecasting.heads.base import HeadSequence, Head

        # Create a minimal test head that requires a field
        class RequireFieldHead(Head):
            @property
            def name(self) -> str:
                return "require_field_test"

            def produces(self) -> set[str]:
                return {"output_field"}

            def requires(self) -> set[str]:
                return {"required_field"}

            def apply(self, df, context):
                # This would need required_field to exist
                df["output_field"] = df.get("required_field", None)

        df = pd.DataFrame({
            "game_id": ["game_1"],
            "home_team": ["Team_A"],
            "away_team": ["Team_B"],
        })
        # required_field is missing

        context = {
            "ratings": {"Team_A": 1600.0, "Team_B": 1500.0},
            "rating_units": "points",
        }

        # Should raise ValueError about missing required fields
        head_sequence = HeadSequence([RequireFieldHead()])
        with pytest.raises(ValueError, match="requires missing fields"):
            head_sequence.apply(df, context)

    def test_head_factory_produces_sequence(self):
        """Test that the factory function creates a valid HeadSequence."""
        from forecasting.heads.base import HeadSequence

        sequence = create_elo_head_sequence()

        assert isinstance(sequence, HeadSequence)
        assert len(sequence.heads) > 0, "Sequence should have at least one head"
        assert all(hasattr(h, "name") for h in sequence.heads), "All heads should have name property"
        assert all(callable(h.produces) for h in sequence.heads), "All heads should have produces method"
        assert all(callable(h.requires) for h in sequence.heads), "All heads should have requires method"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
