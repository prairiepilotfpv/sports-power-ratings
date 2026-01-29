"""Tests for Phase 11: calibration A/B evaluation and policy selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipelines.calibration_ab import (
    ArmMetrics,
    BucketArmMetrics,
    CalibrationABReport,
    PolicyDecision,
    _apply_bucketed_arm,
    _apply_global_arm,
    _compute_arm_metrics,
    _compute_bucket_metrics,
    _evaluate_policy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_total_dataset(
    n: int = 200, seed: int = 42, bias: float = 0.0
) -> pd.DataFrame:
    """Generate synthetic TOTAL calibration dataset.

    Columns: pred_mean, pred_sd, actual_value
    """
    rng = np.random.RandomState(seed)
    pred_mean = rng.uniform(190, 240, n)
    pred_sd = rng.uniform(10, 25, n)
    noise = rng.normal(0, pred_sd)
    actual_value = pred_mean + bias + noise
    return pd.DataFrame(
        {
            "pred_mean": pred_mean,
            "pred_sd": pred_sd,
            "actual_value": actual_value,
        }
    )


def _make_arm_metrics(
    mae: float = 5.0,
    rmse: float = 7.0,
    bias: float = 0.5,
    coverage_1sd: float = 0.68,
    coverage_2sd: float = 0.93,
    tail_miss_rate: float = 0.07,
    sample_count: int = 200,
) -> ArmMetrics:
    """Build ArmMetrics with supplied or default values."""
    return ArmMetrics(
        mae=mae,
        rmse=rmse,
        bias=bias,
        coverage_1sd=coverage_1sd,
        coverage_2sd=coverage_2sd,
        tail_miss_rate=tail_miss_rate,
        sample_count=sample_count,
    )


# ---------------------------------------------------------------------------
# TestArmMetrics
# ---------------------------------------------------------------------------


class TestArmMetrics:
    """Metric computation on synthetic data via Phase 9 functions."""

    def test_perfect_calibration(self):
        """Perfect predictions yield MAE~0, full coverage, tail_miss~0."""
        n = 500
        rng = np.random.RandomState(42)
        pred_mean = rng.uniform(200, 230, n)
        # actual == pred_mean => residual is 0 => any sd covers
        df = pd.DataFrame(
            {
                "pred_mean": pred_mean,
                "pred_sd": np.full(n, 15.0),
                "actual_value": pred_mean,
            }
        )
        m = _compute_arm_metrics(df)
        assert m.mae == pytest.approx(0.0, abs=1e-9)
        assert m.rmse == pytest.approx(0.0, abs=1e-9)
        assert m.bias == pytest.approx(0.0, abs=1e-9)
        assert m.coverage_1sd == pytest.approx(1.0)
        assert m.coverage_2sd == pytest.approx(1.0)
        assert m.tail_miss_rate == pytest.approx(0.0)
        assert m.sample_count == n

    def test_biased_predictions(self):
        """Biased predictions yield nonzero bias and MAE."""
        n = 300
        rng = np.random.RandomState(42)
        pred_mean = rng.uniform(200, 230, n)
        actual = pred_mean + 5.0  # systematic +5 bias
        df = pd.DataFrame(
            {
                "pred_mean": pred_mean,
                "pred_sd": np.full(n, 15.0),
                "actual_value": actual,
            }
        )
        m = _compute_arm_metrics(df)
        # Bias = mean(pred - actual) = -5
        assert m.bias == pytest.approx(-5.0, abs=0.01)
        assert m.mae == pytest.approx(5.0, abs=0.01)
        assert m.rmse == pytest.approx(5.0, abs=0.01)

    def test_serialization(self):
        """ArmMetrics.to_dict() produces JSON-safe values."""
        m = _make_arm_metrics()
        d = m.to_dict()
        # Verify serializable
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        # Round-trip
        restored = json.loads(serialized)
        assert restored["mae"] == 5.0
        assert restored["sample_count"] == 200


# ---------------------------------------------------------------------------
# TestPolicyDecision
# ---------------------------------------------------------------------------


class TestPolicyDecision:
    """Policy gate logic."""

    def test_recommends_when_all_pass(self):
        """All four checks pass -> recommended."""
        baseline = _make_arm_metrics(
            mae=5.0, rmse=7.0, tail_miss_rate=0.10, coverage_2sd=0.90
        )
        treatment = _make_arm_metrics(
            mae=5.1, rmse=7.2, tail_miss_rate=0.07, coverage_2sd=0.93
        )
        policy = _evaluate_policy(baseline, treatment, tolerance_pct=0.05)
        assert policy.recommendation == "recommended"
        assert policy.tail_miss_improved is True
        assert policy.coverage_2sd_improved is True
        assert policy.mae_within_tolerance is True
        assert policy.rmse_within_tolerance is True
        assert len(policy.reasoning) == 4

    def test_rejects_worse_tail_miss(self):
        """Bucketed worsens tail_miss -> not_recommended."""
        baseline = _make_arm_metrics(tail_miss_rate=0.07)
        treatment = _make_arm_metrics(tail_miss_rate=0.10)
        policy = _evaluate_policy(baseline, treatment)
        assert policy.recommendation == "not_recommended"
        assert policy.tail_miss_improved is False

    def test_rejects_worse_coverage(self):
        """Bucketed worsens coverage_2sd -> not_recommended."""
        baseline = _make_arm_metrics(coverage_2sd=0.95)
        treatment = _make_arm_metrics(
            coverage_2sd=0.90, tail_miss_rate=0.04
        )
        policy = _evaluate_policy(baseline, treatment)
        assert policy.recommendation == "not_recommended"
        assert policy.coverage_2sd_improved is False

    def test_rejects_mae_beyond_tolerance(self):
        """MAE exceeds tolerance -> not_recommended."""
        baseline = _make_arm_metrics(mae=5.0, tail_miss_rate=0.10, coverage_2sd=0.90)
        treatment = _make_arm_metrics(
            mae=5.30, tail_miss_rate=0.07, coverage_2sd=0.93
        )
        # tolerance 5% => limit 5.25, treatment has 5.30
        policy = _evaluate_policy(baseline, treatment, tolerance_pct=0.05)
        assert policy.recommendation == "not_recommended"
        assert policy.mae_within_tolerance is False

    def test_rejects_rmse_beyond_tolerance(self):
        """RMSE exceeds tolerance -> not_recommended."""
        baseline = _make_arm_metrics(rmse=7.0, tail_miss_rate=0.10, coverage_2sd=0.90)
        treatment = _make_arm_metrics(
            rmse=7.40, tail_miss_rate=0.07, coverage_2sd=0.93
        )
        # tolerance 5% => limit 7.35, treatment has 7.40
        policy = _evaluate_policy(baseline, treatment, tolerance_pct=0.05)
        assert policy.recommendation == "not_recommended"
        assert policy.rmse_within_tolerance is False

    def test_serialization(self):
        """PolicyDecision.to_dict() is JSON-serializable."""
        baseline = _make_arm_metrics()
        treatment = _make_arm_metrics()
        policy = _evaluate_policy(baseline, treatment)
        d = policy.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# TestApplyArms
# ---------------------------------------------------------------------------


class TestApplyArms:
    """Non-mutation and correct application of calibrators."""

    def test_global_arm_does_not_mutate_input(self):
        """Input DataFrame is not modified by global arm."""
        df = _make_synthetic_total_dataset(50)
        original_pred_mean = df["pred_mean"].copy()
        original_pred_sd = df["pred_sd"].copy()

        # No calibrators => output should equal input
        result = _apply_global_arm(df, mean_cal=None, var_cal=None)

        # Original unchanged
        pd.testing.assert_series_equal(df["pred_mean"], original_pred_mean)
        pd.testing.assert_series_equal(df["pred_sd"], original_pred_sd)
        # Result is a copy
        assert result is not df

    def test_bucketed_arm_does_not_mutate_input(self):
        """Input DataFrame is not modified by bucketed arm."""
        df = _make_synthetic_total_dataset(50)
        original_pred_mean = df["pred_mean"].copy()
        original_pred_sd = df["pred_sd"].copy()

        result = _apply_bucketed_arm(
            df,
            global_mean_cal=None,
            global_var_cal=None,
            bucket_cals={},
            thresholds=(210.0, 225.0),
        )

        pd.testing.assert_series_equal(df["pred_mean"], original_pred_mean)
        pd.testing.assert_series_equal(df["pred_sd"], original_pred_sd)
        assert result is not df

    def test_global_arm_shifts_mean(self):
        """Fitted MeanCalibrator delta is applied by global arm."""
        from calibration.mean_calibrator import MeanCalibrator

        # Build a dataset with systematic +3 bias (actual = pred + 3)
        n = 100
        rng = np.random.RandomState(42)
        pred_mean = rng.uniform(200, 230, n)
        df = pd.DataFrame(
            {
                "pred_mean": pred_mean,
                "pred_sd": np.full(n, 15.0),
                "actual_value": pred_mean + 3.0,
            }
        )

        mean_cal = MeanCalibrator()
        mean_cal.fit(df)
        # delta should be approximately +3
        assert abs(mean_cal.delta - 3.0) < 1.0

        result = _apply_global_arm(df, mean_cal=mean_cal, var_cal=None)
        # pred_mean should be shifted by delta
        expected = pred_mean + mean_cal.delta
        np.testing.assert_allclose(
            result["pred_mean"].values, expected, atol=0.01
        )


# ---------------------------------------------------------------------------
# TestComputeBucketMetrics
# ---------------------------------------------------------------------------


class TestComputeBucketMetrics:
    """Per-bucket metric breakdown."""

    def test_returns_metrics_per_bucket(self):
        """Metrics are computed for each non-empty bucket."""
        rng = np.random.RandomState(42)
        # Spread values across low (<210), mid (210-225), high (>225)
        pred_mean = np.concatenate([
            rng.uniform(190, 209, 50),  # low
            rng.uniform(211, 224, 50),  # mid
            rng.uniform(226, 245, 50),  # high
        ])
        df = pd.DataFrame(
            {
                "pred_mean": pred_mean,
                "pred_sd": np.full(150, 15.0),
                "actual_value": pred_mean + rng.normal(0, 5, 150),
            }
        )

        buckets = _compute_bucket_metrics(df, thresholds=(210.0, 225.0))
        assert len(buckets) == 3
        bucket_names = {b.bucket for b in buckets}
        assert bucket_names == {"low", "mid", "high"}

        for b in buckets:
            assert b.sample_count > 0
            assert np.isfinite(b.metrics.mae)
            assert np.isfinite(b.metrics.rmse)


# ---------------------------------------------------------------------------
# TestCalibrationABReport
# ---------------------------------------------------------------------------


class TestCalibrationABReport:
    """Report serialization."""

    def _make_report(self) -> CalibrationABReport:
        baseline = _make_arm_metrics()
        treatment = _make_arm_metrics(
            mae=4.8, tail_miss_rate=0.05, coverage_2sd=0.95
        )
        policy = _evaluate_policy(baseline, treatment, tolerance_pct=0.05)
        return CalibrationABReport(
            sport="nba",
            season="2025-26",
            market="total",
            source="ensemble_total_v1",
            fit_window={"start": "2025-10-01", "end": "2025-12-31"},
            eval_window={"start": "2026-01-01", "end": "2026-01-28"},
            baseline=baseline,
            treatment=treatment,
            deltas={
                "mae": treatment.mae - baseline.mae,
                "rmse": treatment.rmse - baseline.rmse,
                "bias": treatment.bias - baseline.bias,
                "coverage_1sd": treatment.coverage_1sd - baseline.coverage_1sd,
                "coverage_2sd": treatment.coverage_2sd - baseline.coverage_2sd,
                "tail_miss_rate": treatment.tail_miss_rate - baseline.tail_miss_rate,
            },
            baseline_buckets=[],
            treatment_buckets=[],
            policy=policy,
            bucket_thresholds=(210.0, 225.0),
            min_samples_per_bucket=200,
            timestamp="2026-01-28T12:00:00",
        )

    def test_to_dict_json_serializable(self):
        """CalibrationABReport.to_dict() produces JSON-safe values."""
        report = self._make_report()
        d = report.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        restored = json.loads(serialized)
        assert restored["sport"] == "nba"
        assert restored["market"] == "total"
        assert "baseline" in restored
        assert "treatment" in restored
        assert "policy" in restored
        assert "deltas" in restored

    def test_to_json_creates_file(self, tmp_path):
        """CalibrationABReport.to_json() writes valid JSON to disk."""
        report = self._make_report()
        out_path = tmp_path / "report.json"
        report.to_json(out_path)

        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert data["sport"] == "nba"
        assert data["policy"]["recommendation"] in (
            "recommended",
            "not_recommended",
        )


# ---------------------------------------------------------------------------
# TestEndToEndAB
# ---------------------------------------------------------------------------


class TestEndToEndAB:
    """Integration test with synthetic data (no DB required)."""

    def test_deterministic_ab_pipeline(self):
        """Full A/B pipeline on synthetic data yields complete report."""
        from calibration.mean_calibrator import MeanCalibrator
        from calibration.distribution import VarianceCalibrator

        # Build fit and eval datasets
        fit_df = _make_synthetic_total_dataset(300, seed=42)
        eval_df = _make_synthetic_total_dataset(100, seed=99)

        # Fit global arm
        global_mean_cal = MeanCalibrator()
        global_mean_cal.fit(fit_df)

        global_var_cal = VarianceCalibrator(
            guardrail_min=8.0, guardrail_max=35.0
        )
        global_var_cal.fit(fit_df)

        # Apply global arm
        global_eval = _apply_global_arm(eval_df, global_mean_cal, global_var_cal)

        # Apply bucketed arm (using global cals as fallback, no bucket-specific)
        bucketed_eval = _apply_bucketed_arm(
            eval_df,
            global_mean_cal=global_mean_cal,
            global_var_cal=global_var_cal,
            bucket_cals={},
            thresholds=(210.0, 225.0),
        )

        # Compute metrics
        baseline_metrics = _compute_arm_metrics(global_eval)
        treatment_metrics = _compute_arm_metrics(bucketed_eval)

        # Verify metrics are finite
        for m in [baseline_metrics, treatment_metrics]:
            assert np.isfinite(m.mae)
            assert np.isfinite(m.rmse)
            assert np.isfinite(m.bias)
            assert np.isfinite(m.coverage_1sd)
            assert np.isfinite(m.coverage_2sd)
            assert np.isfinite(m.tail_miss_rate)

        # Compute deltas
        deltas = {
            "mae": treatment_metrics.mae - baseline_metrics.mae,
            "rmse": treatment_metrics.rmse - baseline_metrics.rmse,
            "bias": treatment_metrics.bias - baseline_metrics.bias,
            "coverage_1sd": treatment_metrics.coverage_1sd - baseline_metrics.coverage_1sd,
            "coverage_2sd": treatment_metrics.coverage_2sd - baseline_metrics.coverage_2sd,
            "tail_miss_rate": treatment_metrics.tail_miss_rate - baseline_metrics.tail_miss_rate,
        }
        assert len(deltas) == 6

        # Policy gate
        policy = _evaluate_policy(baseline_metrics, treatment_metrics, tolerance_pct=0.05)
        assert policy.recommendation in ("recommended", "not_recommended")
        assert len(policy.reasoning) == 4

        # Build report
        report = CalibrationABReport(
            sport="nba",
            season="2025-26",
            market="total",
            source="test_source",
            fit_window={"start": "2025-10-01", "end": "2025-12-31"},
            eval_window={"start": "2026-01-01", "end": "2026-01-28"},
            baseline=baseline_metrics,
            treatment=treatment_metrics,
            deltas=deltas,
            baseline_buckets=_compute_bucket_metrics(global_eval, (210.0, 225.0)),
            treatment_buckets=_compute_bucket_metrics(bucketed_eval, (210.0, 225.0)),
            policy=policy,
            bucket_thresholds=(210.0, 225.0),
            min_samples_per_bucket=200,
            timestamp="2026-01-28T12:00:00",
        )

        # Verify report is complete and serializable
        d = report.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        restored = json.loads(serialized)
        assert len(restored["deltas"]) == 6
        assert restored["policy"]["recommendation"] in (
            "recommended",
            "not_recommended",
        )

    def test_stable_across_runs(self):
        """Same seed produces identical metrics (deterministic)."""
        df = _make_synthetic_total_dataset(200, seed=42)
        m1 = _compute_arm_metrics(df)
        m2 = _compute_arm_metrics(df)
        assert m1.mae == m2.mae
        assert m1.rmse == m2.rmse
        assert m1.bias == m2.bias
        assert m1.coverage_1sd == m2.coverage_1sd
        assert m1.coverage_2sd == m2.coverage_2sd
        assert m1.tail_miss_rate == m2.tail_miss_rate
