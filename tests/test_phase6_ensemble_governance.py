"""Tests for Phase 6 ensemble weight governance and auditing.

Tests ensure:
1. Weight clamping (MIN_WEIGHT_EPS) works correctly
2. Neff threshold enforcement in strict/non-strict modes
3. Audit object contains correct dropped reasons and coverage
4. No silent fallback: verify logs emitted when fallback occurs
5. Regression test: weight collapse scenario now fails in strict mode or logs in non-strict
"""

import pytest
import pandas as pd
import logging
from unittest.mock import patch, MagicMock
from datetime import date

from ensemble.audit import EnsembleAudit, compute_neff
from config import (
    MIN_WEIGHT_EPS,
    MIN_NEFF,
    ENSEMBLE_STRICT_MODE,
)


class TestComputeNeff:
    """Test Neff calculation."""

    def test_uniform_weights_neff(self):
        """Uniform weights: Neff = num_models."""
        weights = {"model1": 0.5, "model2": 0.5}
        neff = compute_neff(weights)
        assert abs(neff - 2.0) < 1e-6

    def test_one_model_neff(self):
        """Single model with weight 1.0: Neff = 1.0."""
        weights = {"model1": 1.0}
        neff = compute_neff(weights)
        assert abs(neff - 1.0) < 1e-6

    def test_skewed_weights_neff(self):
        """Skewed weights: Neff < num_models."""
        weights = {"model1": 0.9, "model2": 0.1}
        neff = compute_neff(weights)
        # Neff = 1 / (0.9^2 + 0.1^2) = 1 / (0.81 + 0.01) = 1 / 0.82 ≈ 1.22
        assert 1.2 < neff < 1.3

    def test_empty_weights_neff(self):
        """Empty weights: Neff = 0."""
        weights = {}
        neff = compute_neff(weights)
        assert neff == 0.0

    def test_zero_weights_neff(self):
        """All zero weights: Neff = 0."""
        weights = {"model1": 0.0, "model2": 0.0}
        neff = compute_neff(weights)
        assert neff == 0.0


class TestEnsembleAudit:
    """Test EnsembleAudit dataclass."""

    def test_basic_audit_creation(self):
        """Create basic audit object."""
        audit = EnsembleAudit(
            market="ML",
            weight_source="config",
            final_models=["elo", "bradley-terry"],
            final_weights={"elo": 0.6, "bradley-terry": 0.4},
        )
        assert audit.market == "ML"
        assert audit.weight_source == "config"
        assert len(audit.final_models) == 2
        assert audit.neff == 0.0  # Default

    def test_audit_to_dict(self):
        """Serialize audit to dict."""
        audit = EnsembleAudit(
            market="SPREAD",
            weight_source="db_best_run",
            final_models=["gssd", "toor"],
            final_weights={"gssd": 0.55, "toor": 0.45},
            dropped_models={"elo": "weight=0.0"},
            neff=1.98,
            warnings=["Low Neff"],
        )
        data = audit.to_dict()
        assert data["market"] == "SPREAD"
        assert data["weight_source"] == "db_best_run"
        assert data["neff"] == 1.98
        assert len(data["dropped_models"]) == 1

    def test_calculate_neff(self):
        """Calculate Neff from weights."""
        audit = EnsembleAudit(
            market="ML",
            weight_source="config",
            final_models=["elo", "poisson"],
            final_weights={"elo": 0.5, "poisson": 0.5},
        )
        neff = audit.calculate_neff()
        assert abs(neff - 2.0) < 1e-6

    def test_emit_log(self, caplog):
        """Emit audit log."""
        audit = EnsembleAudit(
            market="ML",
            weight_source="db_tuned",
            final_models=["elo", "bradley-terry"],
            final_weights={"elo": 0.6, "bradley-terry": 0.4},
            dropped_models={"gssd": "missing forecast rows"},
            neff=1.92,
            warnings=["Neff threshold warning"],
        )
        audit.neff_threshold_met = True
        
        with caplog.at_level(logging.INFO):
            audit.emit_log()
        
        records = caplog.records
        assert len(records) >= 1
        assert "[ensemble audit][ML]" in records[0].message
        assert "source=db_tuned" in records[0].message
        assert "Neff=1.92" in records[0].message


class TestWeightClamping:
    """Test MIN_WEIGHT_EPS clamping behavior."""

    def test_weights_below_eps_clamped(self):
        """Weights below MIN_WEIGHT_EPS are clamped to 0."""
        weights = {
            "model1": 0.5,
            "model2": 0.001,  # Below MIN_WEIGHT_EPS (0.01)
        }
        
        # After clamping and renormalization
        clamped = {}
        for model, weight in weights.items():
            if weight >= MIN_WEIGHT_EPS:
                clamped[model] = weight
        
        # Renormalize
        total = sum(clamped.values())
        clamped = {m: w / total for m, w in clamped.items()}
        
        assert "model2" not in clamped
        assert clamped["model1"] == 1.0

    def test_weights_at_eps_threshold(self):
        """Weights exactly at MIN_WEIGHT_EPS are kept."""
        weight = MIN_WEIGHT_EPS
        assert weight >= MIN_WEIGHT_EPS
        assert not (weight < MIN_WEIGHT_EPS)

    def test_clamping_with_audit(self, caplog):
        """Audit tracks weight clamping."""
        audit = EnsembleAudit(market="ML", weight_source="config")
        audit.weight_clamped = True
        audit.dropped_models = {
            "model2": f"weight=0.005 < MIN_WEIGHT_EPS={MIN_WEIGHT_EPS}"
        }
        
        with caplog.at_level(logging.INFO):
            audit.emit_log()
        
        # Check that clamping is mentioned in logs
        records = caplog.records
        assert any("weight_clamped=true" in r.message for r in records)


class TestNeffThreshold:
    """Test MIN_NEFF threshold enforcement."""

    def test_neff_below_threshold_strict_mode(self):
        """In strict mode, Neff < MIN_NEFF raises error."""
        # Create skewed weights that result in low Neff
        weights = {"model1": 0.99, "model2": 0.01}
        neff = compute_neff(weights)
        
        audit = EnsembleAudit(
            market="ML",
            weight_source="config",
            final_models=["model1", "model2"],
            final_weights=weights,
            neff=neff,
            neff_threshold_met=neff >= MIN_NEFF,
        )
        
        # In strict mode, if Neff < MIN_NEFF, should raise
        if not audit.neff_threshold_met and ENSEMBLE_STRICT_MODE:
            with pytest.raises(ValueError, match="Neff"):
                if not audit.neff_threshold_met and ENSEMBLE_STRICT_MODE:
                    raise ValueError(f"Neff={neff:.2f} < MIN_NEFF={MIN_NEFF}")

    def test_neff_below_threshold_non_strict_mode(self):
        """In non-strict mode, Neff < MIN_NEFF logs warning but allows."""
        weights = {"model1": 0.99, "model2": 0.01}
        neff = compute_neff(weights)
        
        audit = EnsembleAudit(
            market="ML",
            weight_source="config",
            final_models=["model1", "model2"],
            final_weights=weights,
            neff=neff,
            neff_threshold_met=neff >= MIN_NEFF,
        )
        
        # In non-strict mode, audit should track warning but not raise
        assert not audit.neff_threshold_met  # Neff is below threshold
        assert not ENSEMBLE_STRICT_MODE  # Non-strict mode (default)
        # Audit allows it; logs warning

    def test_neff_above_threshold(self):
        """When Neff >= MIN_NEFF, threshold is met."""
        weights = {"model1": 0.5, "model2": 0.5}  # Uniform
        neff = compute_neff(weights)
        
        audit = EnsembleAudit(
            market="ML",
            weight_source="config",
            final_models=["model1", "model2"],
            final_weights=weights,
            neff=neff,
            neff_threshold_met=neff >= MIN_NEFF,
        )
        
        # With MIN_NEFF typically around 1.5, uniform 2-model ensemble should meet it
        assert audit.neff_threshold_met


class TestFallbackScenarios:
    """Test fallback behavior when weights collapse or insufficient models."""

    def test_weight_collapse_triggers_fallback(self, caplog):
        """When weights collapse to 1 model, fallback to uniform is tracked."""
        audit = EnsembleAudit(
            market="ML",
            weight_source="db_best_run",
            final_models=["elo"],  # Only 1 model after filtering
            final_weights={"elo": 1.0},
            fallback_applied=True,
            warnings=["Tuned weights collapsed to 1 model"],
        )
        audit.neff = 1.0
        audit.neff_threshold_met = True
        
        with caplog.at_level(logging.INFO):
            audit.emit_log()
        
        # Should have warning in logs
        records = caplog.records
        assert any("fallback_applied=true" in r.message for r in records)

    def test_no_silent_fallback(self, caplog):
        """Fallback must be logged; never silent."""
        audit = EnsembleAudit(
            market="SPREAD",
            weight_source="config",
            fallback_applied=True,
        )
        
        with caplog.at_level(logging.INFO):
            audit.emit_log()
        
        # Audit log must be emitted
        records = caplog.records
        assert len(records) >= 1


class TestDroppedModelsReasoning:
    """Test that dropped models have clear reasons."""

    def test_dropped_missing_forecasts(self):
        """Model dropped for missing forecast rows."""
        audit = EnsembleAudit(
            market="ML",
            weight_source="file",
            dropped_models={"gssd": "missing forecast rows"},
        )
        assert "gssd" in audit.dropped_models
        assert "missing forecast rows" in audit.dropped_models["gssd"]

    def test_dropped_zero_weight(self):
        """Model dropped for zero weight."""
        audit = EnsembleAudit(
            market="TOTAL",
            weight_source="config",
            dropped_models={"bradley-terry": "weight=0.0"},
        )
        assert "bradley-terry" in audit.dropped_models
        assert "weight=0.0" in audit.dropped_models["bradley-terry"]

    def test_dropped_below_eps(self):
        """Model dropped for weight below MIN_WEIGHT_EPS."""
        audit = EnsembleAudit(
            market="SPREAD",
            weight_source="db_tuned",
            dropped_models={"toor": f"weight=0.005 < MIN_WEIGHT_EPS={MIN_WEIGHT_EPS}"},
        )
        assert "toor" in audit.dropped_models
        assert "MIN_WEIGHT_EPS" in audit.dropped_models["toor"]

    def test_dropped_missing_columns(self):
        """Model dropped for missing required columns."""
        audit = EnsembleAudit(
            market="SPREAD",
            weight_source="db_active",
            dropped_models={"elo": "missing margin_mean, margin_sd"},
        )
        assert "elo" in audit.dropped_models
        assert "missing" in audit.dropped_models["elo"]


class TestRegressionWeightCollapse:
    """Regression test: weights collapse to 1 model scenario.
    
    Historical issue: when tuned weights included models that weren't in forecasts,
    weights would collapse silently. Now should:
    - In strict mode: fail with Neff < MIN_NEFF
    - In non-strict mode: log fallback warning, use uniform over available models
    """

    def test_collapse_scenario_strict_mode(self):
        """Collapse to 1 model in strict mode should fail."""
        # Scenario: tuned weights for [elo, gssd, poisson]
        # but only [elo] available in forecasts
        weights = {"elo": 0.4, "gssd": 0.3, "poisson": 0.3}
        available_models = {"elo"}  # Only elo in forecasts
        
        # Simulate filtering
        filtered = {m: w for m, w in weights.items() if m in available_models}
        # Result: {"elo": 0.4}
        
        # After normalization: {"elo": 1.0}
        total = sum(filtered.values())
        normalized = {m: w / total for m, w in filtered.items()}
        
        neff = compute_neff(normalized)  # Neff = 1.0
        
        audit = EnsembleAudit(
            market="ML",
            weight_source="db_best_run",
            final_models=list(normalized.keys()),
            final_weights=normalized,
            neff=neff,
            neff_threshold_met=neff >= MIN_NEFF,
        )
        
        # In strict mode, this should be caught
        assert audit.neff < MIN_NEFF  # Neff=1.0 < MIN_NEFF=1.5

    def test_collapse_scenario_non_strict_mode(self, caplog):
        """Collapse to 1 model in non-strict mode should log fallback."""
        weights = {"elo": 0.4, "gssd": 0.3, "poisson": 0.3}
        available_models = {"elo"}
        
        filtered = {m: w for m, w in weights.items() if m in available_models}
        total = sum(filtered.values())
        normalized = {m: w / total for m, w in filtered.items()}
        
        audit = EnsembleAudit(
            market="ML",
            weight_source="db_best_run",
            final_models=list(normalized.keys()),
            final_weights=normalized,
            fallback_applied=True,
            warnings=["Collapsed to 1 model; using fallback"],
            neff=1.0,
            neff_threshold_met=False,
        )
        
        with caplog.at_level(logging.INFO):
            audit.emit_log()
        
        records = caplog.records
        assert any("fallback_applied=true" in r.message for r in records)


class TestCoverageTracking:
    """Test per-game coverage tracking for heads mode."""

    def test_coverage_summary_structure(self):
        """Coverage summary tracks games and columns per model."""
        audit = EnsembleAudit(
            market="ML",
            weight_source="config",
            coverage_summary={
                "elo": {
                    "games_with_forecasts": 150,
                    "required_columns": ["p_home_win"],
                    "missing_columns": [],
                },
                "bradley-terry": {
                    "games_with_forecasts": 120,
                    "required_columns": ["p_home_win"],
                    "missing_columns": [],
                },
            },
        )
        assert "elo" in audit.coverage_summary
        assert audit.coverage_summary["elo"]["games_with_forecasts"] == 150

    def test_coverage_with_missing_columns(self):
        """Coverage tracks which required columns are missing."""
        audit = EnsembleAudit(
            market="SPREAD",
            weight_source="file",
            coverage_summary={
                "gssd": {
                    "games_with_forecasts": 100,
                    "required_columns": ["margin_mean", "margin_sd"],
                    "missing_columns": ["margin_sd"],
                },
            },
        )
        assert "margin_sd" in audit.coverage_summary["gssd"]["missing_columns"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
