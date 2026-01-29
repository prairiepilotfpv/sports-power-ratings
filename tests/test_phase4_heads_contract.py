"""Phase 4 tests: heads mode contract enforcement and model support matrix.

Tests cover:
1. Projection engine derivation lockout in heads mode
2. Model support matrix filtering
3. Ensemble config alignment validation
4. Producer label normalization
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from forecasting.model_support import (
    ModelSupport,
    get_model_support,
    get_supported_markets,
    filter_models_for_market,
)
from pipelines.projection_engines import (
    _assert_derivation_locked,
    _CANONICAL_DERIVABLE_FIELDS,
)


class TestModelSupportMatrix:
    """Test the model support matrix module."""

    def test_get_model_support_elo(self) -> None:
        """Test Elo model support lookup."""
        support = get_model_support("elo")
        assert support is not None
        assert support.supports_ml
        assert support.supports_spread
        assert support.supports_total
        assert "model_p_home_win" in support.native_fields
        assert "margin_mean" in support.native_fields

    def test_get_model_support_bradley_terry(self) -> None:
        """Test Bradley-Terry model support lookup."""
        support = get_model_support("bradley-terry")
        assert support is not None
        assert support.supports_ml
        assert support.supports_spread
        assert support.supports_total
        assert "p_home_win" in support.native_fields

    def test_get_model_support_poisson(self) -> None:
        """Test Poisson model support lookup."""
        support = get_model_support("poisson")
        assert support is not None
        assert support.supports_ml
        assert support.supports_spread
        assert support.supports_total

    def test_get_model_support_unknown_model(self) -> None:
        """Test unknown model returns None."""
        support = get_model_support("unknown_model")
        assert support is None

    def test_get_supported_markets_elo(self) -> None:
        """Test getting supported markets for Elo."""
        markets = get_supported_markets("elo")
        assert "ML" in markets
        assert "SPREAD" in markets
        assert "TOTAL" in markets

    def test_get_supported_markets_unknown(self) -> None:
        """Test unknown model returns empty list."""
        markets = get_supported_markets("unknown_model")
        assert markets == []

    def test_filter_models_for_market_ml(self) -> None:
        """Test filtering models for ML market."""
        models = ["elo", "bradley-terry", "toor", "poisson"]
        supported, unsupported = filter_models_for_market(models, "ML")
        assert "elo" in supported
        assert "bradley-terry" in supported
        assert len(unsupported) == 0

    def test_filter_models_for_market_with_unknown(self) -> None:
        """Test filtering with unknown model drops to unsupported."""
        models = ["elo", "unknown_model", "poisson"]
        supported, unsupported = filter_models_for_market(models, "ML")
        assert "elo" in supported
        assert "poisson" in supported
        assert "unknown_model" in unsupported

    def test_model_support_market_case_insensitive(self) -> None:
        """Test market checking is case-insensitive."""
        support = get_model_support("elo")
        assert support is not None
        assert support.supports_market("ml")
        assert support.supports_market("ML")
        assert support.supports_market("Ml")
        assert support.supports_market("spread")
        assert support.supports_market("SPREAD")

    def test_model_support_all_supported_markets(self) -> None:
        """Test all_supported_markets returns list in canonical order."""
        support = get_model_support("elo")
        assert support is not None
        markets = support.all_supported_markets()
        assert markets == ["ML", "SPREAD", "TOTAL"]

    def test_model_support_derived_fields(self) -> None:
        """Test derived fields are marked correctly."""
        support = get_model_support("elo")
        assert support is not None
        assert "projected_home_score" in support.derived_fields
        assert "projected_away_score" in support.derived_fields


class TestProjectionEngineDerivationLockout:
    """Test derivation lockout enforcement in heads mode."""

    def test_assert_derivation_locked_heads_disabled(self) -> None:
        """Test no error when heads mode is disabled."""
        # Should not raise
        _assert_derivation_locked("margin_mean", False)
        _assert_derivation_locked("margin_sd", False)
        _assert_derivation_locked("total_mean", False)

    def test_assert_derivation_locked_heads_enabled(self) -> None:
        """Test error raised when heads mode enabled and canonical field."""
        with pytest.raises(RuntimeError, match="heads-mode derivation lockout"):
            _assert_derivation_locked("margin_mean", True)

        with pytest.raises(RuntimeError, match="heads-mode derivation lockout"):
            _assert_derivation_locked("margin_sd", True)

        with pytest.raises(RuntimeError, match="heads-mode derivation lockout"):
            _assert_derivation_locked("total_mean", True)

        with pytest.raises(RuntimeError, match="heads-mode derivation lockout"):
            _assert_derivation_locked("projected_home_score", True)

    def test_assert_derivation_locked_with_context(self) -> None:
        """Test error includes game context when provided."""
        context = {
            "game_id": "game_123",
            "home_team": "TeamA",
            "away_team": "TeamB",
        }
        with pytest.raises(RuntimeError) as exc_info:
            _assert_derivation_locked("margin_mean", True, context)
        
        error_msg = str(exc_info.value)
        assert "margin_mean" in error_msg
        assert "game_123" in error_msg or "TeamA" in error_msg

    def test_canonical_derivable_fields_complete(self) -> None:
        """Test canonical fields list is complete."""
        expected_fields = {
            "p_home_win",
            "model_p_home_win",
            "margin_mean",
            "margin_sd",
            "total_mean",
            "total_sd",
            "projected_home_score",
            "projected_away_score",
            "projected_total",
        }
        assert _CANONICAL_DERIVABLE_FIELDS == expected_fields


class TestModelSupportIntegration:
    """Integration tests for model support with filtering."""

    def test_filter_all_models_support_ml(self) -> None:
        """Test all registered models support ML."""
        for model_id in ["bradley-terry", "elo", "toor", "gssd", "poisson"]:
            support = get_model_support(model_id)
            assert support is not None, f"{model_id} not found in registry"
            assert support.supports_ml, f"{model_id} does not support ML"

    def test_filter_all_models_support_spread(self) -> None:
        """Test all registered models support SPREAD."""
        for model_id in ["bradley-terry", "elo", "toor", "gssd", "poisson"]:
            support = get_model_support(model_id)
            assert support is not None
            assert support.supports_spread, f"{model_id} does not support SPREAD"

    def test_filter_all_models_support_total(self) -> None:
        """Test all registered models support TOTAL."""
        for model_id in ["bradley-terry", "elo", "toor", "gssd", "poisson"]:
            support = get_model_support(model_id)
            assert support is not None
            assert support.supports_total, f"{model_id} does not support TOTAL"

    def test_filter_models_with_all_supported(self) -> None:
        """Test filtering when all models are supported."""
        all_models = ["bradley-terry", "elo", "toor"]
        for market in ["ML", "SPREAD", "TOTAL"]:
            supported, unsupported = filter_models_for_market(all_models, market)
            assert len(supported) == 3, f"All models should support {market}"
            assert len(unsupported) == 0

    def test_filter_handles_duplicates(self) -> None:
        """Test filtering handles duplicate models."""
        models = ["elo", "elo", "bradley-terry"]
        supported, unsupported = filter_models_for_market(models, "ML")
        assert "elo" in supported
        assert "bradley-terry" in supported

    def test_support_matrix_immutability(self) -> None:
        """Test model support is frozen (immutable)."""
        support = get_model_support("elo")
        assert support is not None
        
        # Attempting to modify should raise (frozen dataclass)
        with pytest.raises((AttributeError, TypeError)):
            support.supports_ml = False  # type: ignore


class TestCanonicalFieldEnforcement:
    """Test enforcement of canonical field contract."""

    def test_all_canonical_fields_protected(self) -> None:
        """Test all expected canonical fields are in the lockout set."""
        expected = {
            "p_home_win",
            "model_p_home_win",
            "margin_mean",
            "margin_sd",
            "total_mean",
            "total_sd",
            "projected_home_score",
            "projected_away_score",
            "projected_total",
        }
        assert _CANONICAL_DERIVABLE_FIELDS == expected

    def test_non_canonical_fields_allowed_in_heads_mode(self) -> None:
        """Test non-canonical fields don't trigger lockout."""
        # These should not raise
        _assert_derivation_locked("win_prob_source", True)
        _assert_derivation_locked("margin_dist_assumption", True)
        _assert_derivation_locked("logistic_home_win_prob", True)
        _assert_derivation_locked("normal_p_home_win", True)
