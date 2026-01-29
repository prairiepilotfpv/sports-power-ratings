"""Tests for Phase 8 calibration: mean and variance separation, diagnostics, and validation.

This test suite ensures:
1. Mean calibration only modifies mean, not SD
2. Variance calibration only modifies SD, not mean
3. Data source validation catches NaNs and invalid inputs
4. Variance audit metrics compute correctly
5. Calibration window logging works
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from calibration.mean_calibrator import MeanCalibrator
from calibration.distribution import VarianceCalibrator
from calibration.variance_audit import compute_variance_audit, VarianceAuditResult
from calibration.validation import (
    validate_calibration_data_source,
    validate_no_nans_in_columns,
    check_outcome_range,
)


class TestMeanCalibrator:
    """Tests for MeanCalibrator separation of responsibilities."""

    def test_mean_calibrator_shifts_mean_only(self):
        """Mean calibrator must NOT modify SD."""
        # Create synthetic data with known bias
        np.random.seed(42)
        pred_mean = np.array([100.0, 105.0, 95.0, 110.0, 90.0])
        pred_sd = np.array([8.0, 8.0, 8.0, 8.0, 8.0])
        actual_value = pred_mean + 5.0  # +5 point bias

        df = pd.DataFrame({
            "pred_mean": pred_mean,
            "pred_sd": pred_sd,
            "actual_value": actual_value,
        })

        calibrator = MeanCalibrator(delta_min=-30, delta_max=30)
        calibrator.fit(df)

        # Transform data
        calib_result = calibrator.transform(df[["pred_mean", "pred_sd"]])

        # Check: mean shifted, SD unchanged
        expected_mean = pred_mean + calibrator.delta
        assert np.allclose(calib_result["calibrated_mean"].values, expected_mean, atol=0.1)
        assert np.allclose(calib_result["calibrated_sd"].values, pred_sd, atol=1e-6)

    def test_mean_calibrator_metadata_stores_stats(self):
        """Mean calibrator must store pre/post RMSE and delta."""
        df = pd.DataFrame({
            "pred_mean": np.array([100.0, 105.0, 95.0]),
            "pred_sd": np.array([8.0, 8.0, 8.0]),
            "actual_value": np.array([105.0, 110.0, 100.0]),
        })

        calibrator = MeanCalibrator()
        calibrator.fit(df)

        assert calibrator.metadata["method"] == "mean_shift"
        assert "delta" in calibrator.metadata
        assert "rmse_before" in calibrator.metadata
        assert "rmse_after" in calibrator.metadata
        assert calibrator.metadata["samples"] == 3

    def test_mean_calibrator_rejects_nans(self):
        """Mean calibrator must fail on NaN inputs."""
        df = pd.DataFrame({
            "pred_mean": np.array([100.0, np.nan, 95.0]),
            "pred_sd": np.array([8.0, 8.0, 8.0]),
            "actual_value": np.array([105.0, 110.0, 100.0]),
        })

        calibrator = MeanCalibrator()
        with pytest.raises(ValueError, match="Data validation failed"):
            calibrator.fit(df)

    def test_mean_calibrator_rejects_negative_sd(self):
        """Mean calibrator must fail on SD <= 0."""
        df = pd.DataFrame({
            "pred_mean": np.array([100.0, 105.0, 95.0]),
            "pred_sd": np.array([8.0, -1.0, 8.0]),
            "actual_value": np.array([105.0, 110.0, 100.0]),
        })

        calibrator = MeanCalibrator()
        with pytest.raises(ValueError, match="pred_sd must be positive"):
            calibrator.fit(df)

    def test_mean_calibrator_respects_delta_bounds(self):
        """Mean calibrator must respect delta_min/max guardrails."""
        df = pd.DataFrame({
            "pred_mean": np.array([100.0, 100.0]),
            "pred_sd": np.array([8.0, 8.0]),
            "actual_value": np.array([200.0, 200.0]),  # Very large bias
        })

        calibrator = MeanCalibrator(delta_min=-10, delta_max=10)
        calibrator.fit(df)

        # Delta should be clipped to [−10, 10]
        assert -10 <= calibrator.delta <= 10


class TestVarianceCalibrator:
    """Tests for VarianceCalibrator separation of responsibilities."""

    def test_variance_calibrator_preserves_mean(self):
        """Variance calibrator must NOT modify mean."""
        np.random.seed(42)
        residuals = np.random.normal(0, 3.0, 100)
        pred_mean = np.array([100.0] * 100)
        pred_sd = np.array([5.0] * 100)
        actual_value = pred_mean + residuals

        df = pd.DataFrame({
            "pred_mean": pred_mean,
            "pred_sd": pred_sd,
            "actual_value": actual_value,
        })

        calibrator = VarianceCalibrator()
        calibrator.fit(df)

        # Transform data
        calib_result = calibrator.transform(df[["pred_mean", "pred_sd"]])

        # Check: mean unchanged, SD may be scaled
        assert np.allclose(calib_result["calibrated_mean"].values, pred_mean, atol=1e-6)
        # SD should be different (since we fit it)
        assert not np.allclose(calib_result["calibrated_sd"].values, pred_sd, atol=0.1)

    def test_variance_calibrator_metadata(self):
        """Variance calibrator must record c, tau, clip_rate."""
        df = pd.DataFrame({
            "pred_mean": np.array([100.0, 105.0, 95.0]),
            "pred_sd": np.array([8.0, 8.0, 8.0]),
            "actual_value": np.array([100.5, 105.5, 94.5]),
        })

        calibrator = VarianceCalibrator()
        calibrator.fit(df)

        assert "variance" in calibrator.metadata.get("method", "")
        assert "c" in calibrator.metadata
        assert "tau" in calibrator.metadata
        assert "clip_rate" in calibrator.metadata

    def test_variance_calibrator_rejects_zero_sd(self):
        """Variance calibrator must fail on SD <= 0."""
        df = pd.DataFrame({
            "pred_mean": np.array([100.0, 105.0]),
            "pred_sd": np.array([8.0, 0.0]),
            "actual_value": np.array([100.5, 105.5]),
        })

        calibrator = VarianceCalibrator()
        with pytest.raises(ValueError):
            calibrator.fit(df)


class TestCalibrationSeparation:
    """Tests that mean and variance calibration stages work independently."""

    def test_mean_then_variance_order(self):
        """Test applying mean calibration then variance calibration."""
        np.random.seed(42)
        residuals = np.random.normal(0, 2.0, 50)
        pred_mean = np.array([100.0] * 50)
        pred_sd = np.array([5.0] * 50)
        actual_value = pred_mean + 3.0 + residuals  # +3 bias

        df = pd.DataFrame({
            "pred_mean": pred_mean,
            "pred_sd": pred_sd,
            "actual_value": actual_value,
        })

        # Stage 1: Mean calibration
        mean_cal = MeanCalibrator()
        mean_cal.fit(df)
        mean_result = mean_cal.transform(df[["pred_mean", "pred_sd"]])

        # Stage 2: Variance calibration on output of mean calibrator
        df_after_mean = pd.DataFrame({
            "pred_mean": mean_result["calibrated_mean"],
            "pred_sd": mean_result["calibrated_sd"],
            "actual_value": actual_value,
        })

        var_cal = VarianceCalibrator()
        var_cal.fit(df_after_mean)
        var_result = var_cal.transform(df_after_mean[["pred_mean", "pred_sd"]])

        # Verify: mean from mean_cal should match mean from var_cal
        assert np.allclose(
            var_result["calibrated_mean"].values,
            mean_result["calibrated_mean"].values,
            atol=1e-6,
        )


class TestVarianceAudit:
    """Tests for variance audit diagnostics."""

    def test_variance_audit_metrics_computed(self):
        """Variance audit must compute coverage and MAE metrics."""
        np.random.seed(42)
        residuals = np.random.normal(0, 5.0, 100)
        pred_mean = np.array([100.0] * 100)
        pred_sd = np.array([5.0] * 100)
        actual_value = pred_mean + residuals

        df = pd.DataFrame({
            "pred_mean": pred_mean,
            "pred_sd": pred_sd,
            "actual_value": actual_value,
        })

        audit = compute_variance_audit(df)

        # Check all metrics exist and are sensible
        assert audit.sd_before > 0
        assert audit.empirical_mae > 0
        assert 0 <= audit.coverage_1sd <= 1
        assert 0 <= audit.coverage_2sd <= 1
        assert 0 <= audit.outside_95ci <= 1
        assert audit.coverage_2sd >= audit.coverage_1sd

    def test_variance_audit_high_coverage_with_small_sd(self):
        """When residuals are small relative to SD, coverage should be high."""
        np.random.seed(42)
        residuals = np.random.normal(0, 0.5, 100)  # Small residuals
        pred_mean = np.array([100.0] * 100)
        pred_sd = np.array([5.0] * 100)  # Large SD relative to residuals
        actual_value = pred_mean + residuals

        df = pd.DataFrame({
            "pred_mean": pred_mean,
            "pred_sd": pred_sd,
            "actual_value": actual_value,
        })

        audit = compute_variance_audit(df)
        assert audit.coverage_1sd > 0.95  # Should have very high coverage


class TestValidation:
    """Tests for calibration data source validation."""

    def test_validate_no_nans_passes_clean_data(self):
        """Validation should pass on clean data."""
        df = pd.DataFrame({
            "pred_mean": [100.0, 105.0, 95.0],
            "actual_value": [101.0, 106.0, 96.0],
        })

        result = validate_no_nans_in_columns(df, ["pred_mean", "actual_value"], fail=True)
        assert result is True

    def test_validate_no_nans_fails_with_nans(self):
        """Validation must fail when NaNs present."""
        df = pd.DataFrame({
            "pred_mean": [100.0, np.nan, 95.0],
            "actual_value": [101.0, 106.0, 96.0],
        })

        with pytest.raises(ValueError):
            validate_no_nans_in_columns(df, ["pred_mean"], fail=True)

    def test_validate_calibration_data_source_checks_required_cols(self):
        """Validation must check for required columns by market."""
        df = pd.DataFrame({"some_col": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError):
            validate_calibration_data_source(
                df,
                market="TOTAL",
                sport="nba",
                season="2025-26",
                fail_on_errors=True,
            )

    def test_check_outcome_range_passes_valid_data(self):
        """Outcome range check should pass on valid data."""
        df = pd.DataFrame({"total": [100.0, 105.0, 95.0]})
        result = check_outcome_range(df, "total", min_val=0, max_val=250, fail=True)
        assert result is True

    def test_check_outcome_range_fails_out_of_bounds(self):
        """Outcome range check must fail on out-of-bounds data."""
        df = pd.DataFrame({"total": [100.0, 500.0, 95.0]})  # 500 is out of bounds

        with pytest.raises(ValueError):
            check_outcome_range(df, "total", min_val=0, max_val=250, fail=True)


class TestMeanCalibratorTransform:
    """Tests for MeanCalibrator.transform method."""

    def test_transform_fails_if_not_fitted(self):
        """Transform must fail if calibrator not fitted."""
        calibrator = MeanCalibrator()
        df = pd.DataFrame({"pred_mean": [100.0], "pred_sd": [8.0]})

        with pytest.raises(RuntimeError):
            calibrator.transform(df)

    def test_transform_requires_pred_mean(self):
        """Transform must require pred_mean column."""
        df = pd.DataFrame({"pred_mean": [100.0, 101.0], "actual_value": [101.0, 102.0], "pred_sd": [8.0, 8.0]})
        calibrator = MeanCalibrator()
        calibrator.fit(df)

        df_no_mean = pd.DataFrame({"pred_sd": [8.0]})
        with pytest.raises(KeyError):
            calibrator.transform(df_no_mean)

    def test_transform_handles_missing_sd(self):
        """Transform should handle DataFrames without SD column."""
        df = pd.DataFrame({
            "pred_mean": [100.0, 105.0],
            "actual_value": [101.0, 106.0],
        })
        calibrator = MeanCalibrator()
        calibrator.fit(df)

        result = calibrator.transform(df[["pred_mean"]])
        assert "calibrated_mean" in result.columns
        assert "calibrated_sd" in result.columns
        assert result["calibrated_sd"].isna().all()
