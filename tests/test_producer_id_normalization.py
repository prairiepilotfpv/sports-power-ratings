"""Tests for producer ID and provenance normalization."""

from __future__ import annotations

import pytest

from forecasting.producer_id import (
    normalize_win_prob_source,
    get_ensemble_producer_id,
    is_ensemble_producer,
    is_valid_producer_in_market,
    validate_producer_in_heads_mode,
    ENSEMBLE_ML_V1,
    ENSEMBLE_SPREAD_V1,
    ENSEMBLE_TOTAL_V1,
)


class TestProducerIDNormalization:
    """Test producer ID normalization functions."""

    def test_normalize_win_prob_source_none(self) -> None:
        """Test normalizing None returns None."""
        result = normalize_win_prob_source(None, "ML")
        assert result is None

    def test_normalize_ensemble_ml_v1(self) -> None:
        """Test normalizing ensemble_ml_v1 returns canonical form."""
        result = normalize_win_prob_source("ensemble_ml", "ML")
        assert result == ENSEMBLE_ML_V1

    def test_normalize_ensemble_spread_v1(self) -> None:
        """Test normalizing ensemble_spread returns canonical form."""
        result = normalize_win_prob_source("ensemble_spread", "SPREAD")
        assert result == ENSEMBLE_SPREAD_V1

    def test_normalize_ensemble_total_v1(self) -> None:
        """Test normalizing ensemble_total returns canonical form."""
        result = normalize_win_prob_source("ensemble_total", "TOTAL")
        assert result == ENSEMBLE_TOTAL_V1

    def test_normalize_model_name(self) -> None:
        """Test normalizing model names passes through."""
        result = normalize_win_prob_source("elo", "ML")
        assert result == "elo"

        result = normalize_win_prob_source("bradley-terry", "ML")
        assert result == "bradley-terry"

    def test_normalize_with_calibration_tags(self) -> None:
        """Test calibration tags are preserved."""
        result = normalize_win_prob_source("elo+calibrated_ml", "ML")
        assert result == "elo+calibrated_ml"

        result = normalize_win_prob_source("ensemble_ml+calibrated_ml+calibrated_spread", "ML")
        assert result == ENSEMBLE_ML_V1 + "+calibrated_ml+calibrated_spread"

    def test_normalize_direct_in_heads_mode(self) -> None:
        """Test 'direct' is flagged in heads mode."""
        # With heads_mode=False, "direct" should remain
        result = normalize_win_prob_source("direct", "ML", heads_mode=False)
        assert result == "direct"

        # With heads_mode=True, "direct" should be replaced
        result = normalize_win_prob_source("direct", "ML", heads_mode=True)
        assert result == "model_direct"  # Fallback marker

    def test_normalize_whitespace_handling(self) -> None:
        """Test whitespace is handled correctly."""
        result = normalize_win_prob_source("  elo  ", "ML")
        assert result == "elo"

        result = normalize_win_prob_source("elo + calibrated_ml", "ML")
        assert "elo" in result
        assert "calibrated_ml" in result


class TestGetEnsembleProducerID:
    """Test ensemble producer ID getter."""

    def test_get_ensemble_ml(self) -> None:
        """Test getting ML ensemble producer ID."""
        result = get_ensemble_producer_id("ML")
        assert result == ENSEMBLE_ML_V1

    def test_get_ensemble_spread(self) -> None:
        """Test getting SPREAD ensemble producer ID."""
        result = get_ensemble_producer_id("SPREAD")
        assert result == ENSEMBLE_SPREAD_V1

    def test_get_ensemble_total(self) -> None:
        """Test getting TOTAL ensemble producer ID."""
        result = get_ensemble_producer_id("TOTAL")
        assert result == ENSEMBLE_TOTAL_V1

    def test_get_ensemble_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert get_ensemble_producer_id("ml") == ENSEMBLE_ML_V1
        assert get_ensemble_producer_id("Ml") == ENSEMBLE_ML_V1
        assert get_ensemble_producer_id("spread") == ENSEMBLE_SPREAD_V1


class TestIsEnsembleProducer:
    """Test ensemble producer checking."""

    def test_is_ensemble_ml_v1(self) -> None:
        """Test detecting ensemble_ml_v1."""
        assert is_ensemble_producer(ENSEMBLE_ML_V1)
        assert is_ensemble_producer(ENSEMBLE_ML_V1 + "+calibrated_ml")

    def test_is_ensemble_spread_v1(self) -> None:
        """Test detecting ensemble_spread_v1."""
        assert is_ensemble_producer(ENSEMBLE_SPREAD_V1)

    def test_is_ensemble_total_v1(self) -> None:
        """Test detecting ensemble_total_v1."""
        assert is_ensemble_producer(ENSEMBLE_TOTAL_V1)

    def test_is_not_ensemble_model(self) -> None:
        """Test model names are not ensemble."""
        assert not is_ensemble_producer("elo")
        assert not is_ensemble_producer("bradley-terry")
        assert not is_ensemble_producer("poisson")

    def test_is_not_ensemble_none(self) -> None:
        """Test None is not ensemble."""
        assert not is_ensemble_producer(None)
        assert not is_ensemble_producer("")
        assert not is_ensemble_producer("  ")


class TestIsValidProducerInMarket:
    """Test market-specific producer validation."""

    def test_valid_ensemble_ml(self) -> None:
        """Test ensemble_ml_v1 is valid for ML."""
        assert is_valid_producer_in_market(ENSEMBLE_ML_V1, "ML")

    def test_valid_model_for_ml(self) -> None:
        """Test model names are valid for ML."""
        assert is_valid_producer_in_market("elo", "ML")
        assert is_valid_producer_in_market("bradley-terry", "ML")
        assert is_valid_producer_in_market("poisson", "ML")

    def test_invalid_model_for_spread(self) -> None:
        """Test model names are invalid for SPREAD."""
        assert not is_valid_producer_in_market("elo", "SPREAD")
        assert not is_valid_producer_in_market("bradley-terry", "SPREAD")

    def test_valid_ensemble_spread(self) -> None:
        """Test ensemble_spread_v1 is valid for SPREAD."""
        assert is_valid_producer_in_market(ENSEMBLE_SPREAD_V1, "SPREAD")

    def test_invalid_ensemble_ml_for_spread(self) -> None:
        """Test ensemble_ml_v1 is invalid for SPREAD."""
        assert not is_valid_producer_in_market(ENSEMBLE_ML_V1, "SPREAD")

    def test_valid_ensemble_total(self) -> None:
        """Test ensemble_total_v1 is valid for TOTAL."""
        assert is_valid_producer_in_market(ENSEMBLE_TOTAL_V1, "TOTAL")

    def test_with_calibration_tags(self) -> None:
        """Test validation works with calibration tags."""
        assert is_valid_producer_in_market(ENSEMBLE_ML_V1 + "+calibrated_ml", "ML")
        assert is_valid_producer_in_market(ENSEMBLE_SPREAD_V1 + "+calibrated_spread", "SPREAD")
        assert is_valid_producer_in_market("elo+calibrated_ml", "ML")


class TestValidateProducerInHeadsMode:
    """Test heads mode compliance validation."""

    def test_valid_ensemble_ml_heads_mode(self) -> None:
        """Test ensemble_ml_v1 is valid in heads mode."""
        is_valid, error = validate_producer_in_heads_mode(ENSEMBLE_ML_V1, "ML")
        assert is_valid
        assert error is None

    def test_valid_model_ml_heads_mode(self) -> None:
        """Test model names are valid for ML in heads mode."""
        is_valid, error = validate_producer_in_heads_mode("elo", "ML")
        assert is_valid
        assert error is None

    def test_invalid_direct_heads_mode(self) -> None:
        """Test 'direct' is rejected in heads mode."""
        is_valid, error = validate_producer_in_heads_mode("direct", "ML")
        assert not is_valid
        assert error is not None
        assert "direct" in error.lower()
        assert "forbidden" in error.lower()

    def test_invalid_ensemble_ml_for_spread_heads_mode(self) -> None:
        """Test ensemble_ml_v1 is invalid for SPREAD even in heads mode."""
        is_valid, error = validate_producer_in_heads_mode(ENSEMBLE_ML_V1, "SPREAD")
        assert not is_valid
        assert error is not None

    def test_invalid_model_for_spread_heads_mode(self) -> None:
        """Test model names are invalid for SPREAD in heads mode."""
        is_valid, error = validate_producer_in_heads_mode("elo", "SPREAD")
        assert not is_valid
        assert error is not None

    def test_valid_ensemble_spread_heads_mode(self) -> None:
        """Test ensemble_spread_v1 is valid for SPREAD in heads mode."""
        is_valid, error = validate_producer_in_heads_mode(ENSEMBLE_SPREAD_V1, "SPREAD")
        assert is_valid
        assert error is None

    def test_valid_with_calibration_tags(self) -> None:
        """Test calibration tags don't break validation."""
        is_valid, error = validate_producer_in_heads_mode(
            ENSEMBLE_ML_V1 + "+calibrated_ml+calibrated_spread", "ML"
        )
        assert is_valid
        assert error is None

    def test_none_producer(self) -> None:
        """Test None producer is invalid."""
        is_valid, error = validate_producer_in_heads_mode(None, "ML")
        assert not is_valid
        assert error is not None

    def test_error_messages_are_informative(self) -> None:
        """Test error messages guide users toward correct format."""
        is_valid, error = validate_producer_in_heads_mode("elo", "SPREAD")
        assert not is_valid
        assert error is not None
        assert "elo" in error
        assert "SPREAD" in error
        assert "ensemble_spread_v1" in error
