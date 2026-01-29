"""Test calibration evaluation metrics, slicing, and drift diagnostics.

These tests verify that evaluation metrics are computed correctly on synthetic
data without modifying any predictions or calibration logic.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime, timedelta

from src.pipelines.calibration_evaluation import (
    # ML Metrics
    brier_score,
    log_loss,
    calibration_error,
    reliability_curve,
    
    # SPREAD/TOTAL Metrics
    mean_absolute_error,
    root_mean_squared_error,
    mean_squared_error,
    mean_signed_error,
    empirical_coverage,
    tail_miss_rate,
    
    # Slicing
    SlicerConfig,
    RegimeType,
    FavoriteVsUnderdogSlicer,
    SpreadBucketSlicer,
    TotalBucketSlicer,
    HomeAwaySlicer,
    SeasonSegmentSlicer,
    MarketDistanceSlicer,
    build_slicer,
    
    # Drift & Stability
    DriftMetrics,
    compute_drift_metrics,
    rolling_window_metrics,
    
    # Reporting
    MarketEvaluationReport,
    summarize_evaluation,
)


# ============================================================================
# ML METRICS TESTS
# ============================================================================


class TestMLMetrics:
    """Test ML calibration metrics."""
    
    def test_brier_score_perfect_predictions(self):
        """Perfect predictions should give Brier score of 0."""
        probs = np.array([0.0, 1.0, 0.5])
        outcomes = np.array([0.0, 1.0, 0.5])
        assert brier_score(probs, outcomes) == 0.0
    
    def test_brier_score_worst_predictions(self):
        """Worst predictions should give Brier score of 1."""
        probs = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        assert brier_score(probs, outcomes) == 1.0
    
    def test_brier_score_clipping(self):
        """Probabilities outside [0, 1] should be clipped."""
        probs = np.array([-0.5, 1.5, 0.5])
        outcomes = np.array([0.0, 1.0, 0.5])
        result = brier_score(probs, outcomes)
        assert 0 <= result <= 1
    
    def test_brier_score_empty(self):
        """Empty arrays should return NaN."""
        assert np.isnan(brier_score(np.array([]), np.array([])))
    
    def test_brier_score_pandas_series(self):
        """Should work with pandas Series."""
        probs = pd.Series([0.7, 0.3])
        outcomes = pd.Series([1, 0])
        result = brier_score(probs, outcomes)
        expected = ((0.7 - 1)**2 + (0.3 - 0)**2) / 2
        assert np.isclose(result, expected)
    
    def test_log_loss_perfect_predictions(self):
        """Well-calibrated predictions should give low log loss."""
        probs = np.array([0.1, 0.9])
        outcomes = np.array([0, 1])
        result = log_loss(probs, outcomes)
        # With 0.1 for outcome=0 and 0.9 for outcome=1, loss is ~0.105
        # This is better than uniform 0.5 (loss ~0.693)
        assert result < 0.2
    
    def test_log_loss_uniform(self):
        """Uniform predictions (0.5) on balanced data should give log(2)."""
        probs = np.array([0.5] * 100)
        outcomes = np.array([0] * 50 + [1] * 50)
        result = log_loss(probs, outcomes)
        assert np.isclose(result, np.log(2))
    
    def test_log_loss_clipping(self):
        """Extreme probabilities should be clipped to avoid log(0)."""
        probs = np.array([0.0, 1.0, 0.5])
        outcomes = np.array([0, 1, 0])
        result = log_loss(probs, outcomes)
        assert np.isfinite(result)
    
    def test_log_loss_empty(self):
        """Empty arrays should return NaN."""
        assert np.isnan(log_loss(np.array([]), np.array([])))
    
    def test_calibration_error_perfect(self):
        """Perfect predictions should give ECE of 0."""
        # Predictions: 0.3, 0.7
        # Outcomes: 0, 1
        # Bin: [0, 0.1) contains 0, [0.6, 0.7) contains 1
        probs = np.array([0.3, 0.7])
        outcomes = np.array([0.3, 0.7])
        result = calibration_error(probs, outcomes)
        assert result < 0.01
    
    def test_calibration_error_biased(self):
        """Biased predictions should have high ECE."""
        # All predictions 0.9 but outcomes are 50/50
        probs = np.array([0.9] * 100)
        outcomes = np.array([0] * 50 + [1] * 50)
        result = calibration_error(probs, outcomes)
        assert result > 0.3  # Large error
    
    def test_reliability_curve_shape(self):
        """Reliability curve should have correct structure."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 0, 0, 1, 1])
        result = reliability_curve(probs, outcomes, n_bins=5, min_count=1)
        
        assert isinstance(result, pd.DataFrame)
        assert "bin_lower" in result.columns
        assert "bin_upper" in result.columns
        assert "count" in result.columns
        assert "avg_pred" in result.columns
        assert "avg_actual" in result.columns
    
    def test_reliability_curve_empty(self):
        """Empty DataFrame should return empty."""
        result = reliability_curve(np.array([]), np.array([]))
        assert len(result) == 0


# ============================================================================
# SPREAD / TOTAL METRICS TESTS
# ============================================================================


class TestSpreadTotalMetrics:
    """Test SPREAD/TOTAL calibration metrics."""
    
    def test_mean_absolute_error_perfect(self):
        """Perfect predictions should give MAE of 0."""
        assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0.0
    
    def test_mean_absolute_error_off_by_one(self):
        """Predictions off by 1 should give MAE of 1."""
        result = mean_absolute_error([1, 2, 3], [2, 3, 4])
        assert np.isclose(result, 1.0)
    
    def test_mean_absolute_error_empty(self):
        """Empty arrays should return NaN."""
        assert np.isnan(mean_absolute_error([], []))
    
    def test_root_mean_squared_error_perfect(self):
        """Perfect predictions should give RMSE of 0."""
        assert root_mean_squared_error([1, 2, 3], [1, 2, 3]) == 0.0
    
    def test_root_mean_squared_error_magnitude(self):
        """RMSE should penalize larger errors more."""
        mae = mean_absolute_error([1, 2], [2, 4])  # Errors: 1, 2
        rmse = root_mean_squared_error([1, 2], [2, 4])
        assert rmse > mae  # RMSE penalizes more
    
    def test_mean_squared_error_perfect(self):
        """Perfect predictions should give MSE of 0."""
        assert mean_squared_error([1, 2, 3], [1, 2, 3]) == 0.0
    
    def test_mean_signed_error_positive_bias(self):
        """Overestimates should give positive MSE."""
        result = mean_signed_error([5, 6, 7], [3, 4, 5])
        assert result > 0
    
    def test_mean_signed_error_negative_bias(self):
        """Underestimates should give negative MSE."""
        result = mean_signed_error([3, 4, 5], [5, 6, 7])
        assert result < 0
    
    def test_empirical_coverage_perfect_1sigma(self):
        """Perfect 1σ calibration should have ~68% coverage."""
        # Generate samples from normal distribution
        np.random.seed(42)
        n = 10000
        pred_mean = np.zeros(n)
        pred_sd = np.ones(n)
        actual = np.random.normal(0, 1, n)
        
        coverage = empirical_coverage(pred_mean, pred_sd, actual, sigma=1.0)
        assert 0.6 < coverage < 0.75  # Should be near 68%
    
    def test_empirical_coverage_perfect_2sigma(self):
        """Perfect 2σ calibration should have ~95% coverage."""
        np.random.seed(42)
        n = 10000
        pred_mean = np.zeros(n)
        pred_sd = np.ones(n)
        actual = np.random.normal(0, 1, n)
        
        coverage = empirical_coverage(pred_mean, pred_sd, actual, sigma=2.0)
        assert 0.93 < coverage < 0.98  # Should be near 95%
    
    def test_empirical_coverage_overconfident(self):
        """Overconfident predictions should have low coverage."""
        pred_mean = np.array([0, 0, 0])
        pred_sd = np.array([0.1, 0.1, 0.1])  # Very tight
        actual = np.array([3, 3, 3])  # Far away
        
        coverage = empirical_coverage(pred_mean, pred_sd, actual, sigma=1.0)
        assert coverage < 0.1
    
    def test_tail_miss_rate_perfect_2sigma(self):
        """Perfect 2σ calibration should have ~5% tail miss rate."""
        np.random.seed(42)
        n = 10000
        pred_mean = np.zeros(n)
        pred_sd = np.ones(n)
        actual = np.random.normal(0, 1, n)
        
        miss_rate = tail_miss_rate(pred_mean, pred_sd, actual, sigma=2.0)
        assert 0.02 < miss_rate < 0.08  # Should be near 5%
    
    def test_tail_miss_rate_empty(self):
        """Empty arrays should return NaN."""
        assert np.isnan(tail_miss_rate([], [], []))


# ============================================================================
# SLICING TESTS
# ============================================================================


class TestSlicing:
    """Test regime slicing utilities."""
    
    @pytest.fixture
    def sample_schedule(self) -> pd.DataFrame:
        """Create sample schedule DataFrame."""
        return pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=100),
            "game_id": [f"g{i}" for i in range(100)],
            "home_team": ["TeamA"] * 50 + ["TeamB"] * 50,
            "away_team": ["TeamB"] * 50 + ["TeamA"] * 50,
            "is_home": [True] * 50 + [False] * 50,
            "opening_spread": np.concatenate([
                np.random.uniform(-10, 10, 50),
                np.random.uniform(-10, 10, 50),
            ]),
            "opening_total": np.concatenate([
                np.random.uniform(180, 240, 50),
                np.random.uniform(180, 240, 50),
            ]),
            "proj_margin_mean": np.concatenate([
                np.random.uniform(-5, 5, 50),
                np.random.uniform(-5, 5, 50),
            ]),
        })
    
    def test_favorite_vs_underdog_slicer(self, sample_schedule):
        """Favorite/underdog slicer should separate by line sign."""
        slicer = FavoriteVsUnderdogSlicer()
        slices = slicer.slice(sample_schedule)
        
        assert "favorites" in slices
        assert "underdogs" in slices
        
        # Each slice should have positive counts
        assert len(slices["favorites"]) > 0
        assert len(slices["underdogs"]) > 0
    
    def test_favorite_vs_underdog_slicer_missing_column(self):
        """Should return empty dict if spread column missing."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        slicer = FavoriteVsUnderdogSlicer()
        slices = slicer.slice(df)
        assert slices == {}
    
    def test_spread_bucket_slicer(self, sample_schedule):
        """Spread bucket slicer should partition by magnitude."""
        slicer = SpreadBucketSlicer()
        slices = slicer.slice(sample_schedule)
        
        # Should have multiple buckets
        assert len(slices) > 1
        
        # All rows should be assigned to some bucket
        total_rows = sum(len(df) for df in slices.values())
        assert total_rows == len(sample_schedule)
    
    def test_spread_bucket_slicer_custom_buckets(self, sample_schedule):
        """Should respect custom bucket boundaries."""
        slicer = SpreadBucketSlicer(buckets=[0, 2, 5, 10])
        slices = slicer.slice(sample_schedule)
        
        # Should have buckets for each boundary pair
        assert "0-2" in slices or "10+" in slices
    
    def test_total_bucket_slicer(self, sample_schedule):
        """Total bucket slicer should partition by total."""
        slicer = TotalBucketSlicer()
        slices = slicer.slice(sample_schedule)
        
        assert len(slices) > 1
        total_rows = sum(len(df) for df in slices.values())
        assert total_rows == len(sample_schedule)
    
    def test_home_away_slicer(self, sample_schedule):
        """Home/away slicer should split by is_home column."""
        slicer = HomeAwaySlicer()
        slices = slicer.slice(sample_schedule)
        
        assert "home" in slices
        assert "away" in slices
        assert len(slices["home"]) == 50
        assert len(slices["away"]) == 50
    
    def test_home_away_slicer_missing_column(self):
        """Should return empty dfs if is_home missing."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        slicer = HomeAwaySlicer()
        slices = slicer.slice(df)
        
        assert len(slices["home"]) == 0
        assert len(slices["away"]) == 0
    
    def test_season_segment_slicer(self, sample_schedule):
        """Season segment slicer should partition by month ranges."""
        slicer = SeasonSegmentSlicer()
        slices = slicer.slice(sample_schedule)
        
        # Should have segments for each boundary pair
        assert len(slices) > 0
        
        # All rows should be assigned
        total_rows = sum(len(df) for df in slices.values())
        assert total_rows == len(sample_schedule)
    
    def test_market_distance_slicer(self, sample_schedule):
        """Market distance slicer should partition by |line - mean|."""
        slicer = MarketDistanceSlicer()
        slices = slicer.slice(sample_schedule)
        
        assert len(slices) > 0
        total_rows = sum(len(df) for df in slices.values())
        assert total_rows == len(sample_schedule)
    
    def test_build_slicer_factory(self):
        """Factory should create correct slicer instances."""
        config = SlicerConfig(
            regime_type=RegimeType.FAVORITE_VS_UNDERDOG,
            name="fav_vs_ud",
        )
        slicer = build_slicer(config)
        assert isinstance(slicer, FavoriteVsUnderdogSlicer)
    
    def test_build_slicer_with_kwargs(self):
        """Factory should pass kwargs to slicer constructor."""
        config = SlicerConfig(
            regime_type=RegimeType.SPREAD_BUCKET,
            name="spread_buckets",
            kwargs={"buckets": [0, 5, 10]},
        )
        slicer = build_slicer(config)
        assert isinstance(slicer, SpreadBucketSlicer)
        assert slicer.buckets == [0, 5, 10]
    
    def test_build_slicer_invalid_regime(self):
        """Should raise on unknown regime type."""
        with pytest.raises(ValueError):
            config = SlicerConfig(
                regime_type="invalid",
                name="bad",
            )
            build_slicer(config)


# ============================================================================
# DRIFT & STABILITY TESTS
# ============================================================================


class TestDriftAndStability:
    """Test drift and stability diagnostics."""
    
    def test_drift_metrics_no_degradation(self):
        """Metrics within tolerance should not be flagged as degraded."""
        drift = DriftMetrics(
            metric_name="test_metric",
            fit_window_value=1.0,
            eval_window_value=1.05,  # 5% change
            absolute_change=0.05,
            percent_change=0.05,
            tolerance=0.15,
        )
        assert not drift.is_degraded
    
    def test_drift_metrics_degradation(self):
        """Metrics beyond tolerance should be flagged as degraded."""
        drift = DriftMetrics(
            metric_name="test_metric",
            fit_window_value=1.0,
            eval_window_value=1.20,  # 20% change
            absolute_change=0.20,
            percent_change=0.20,
            tolerance=0.15,
        )
        assert drift.is_degraded
    
    def test_drift_metrics_nan_handling(self):
        """NaN values should not trigger degradation flag."""
        drift = DriftMetrics(
            metric_name="test_metric",
            fit_window_value=np.nan,
            eval_window_value=1.0,
            absolute_change=np.nan,
            percent_change=np.nan,
        )
        assert not drift.is_degraded
    
    def test_compute_drift_metrics(self):
        """Should compute drift metrics correctly."""
        fit_df = pd.DataFrame({
            "value": [1.0, 2.0, 3.0],
            "outcome": [1, 1, 1],
        })
        eval_df = pd.DataFrame({
            "value": [1.1, 2.1, 3.1],
            "outcome": [1, 1, 1],
        })
        
        def mean_error(df):
            return df["value"].mean() - df["outcome"].mean()
        
        drift = compute_drift_metrics(
            fit_df,
            eval_df,
            metric_fn=mean_error,
            metric_name="mean_error",
        )
        
        assert drift.metric_name == "mean_error"
        assert np.isclose(drift.fit_window_value, 1.0)  # mean([1, 2, 3]) - mean([1, 1, 1]) = 2 - 1 = 1
        assert np.isclose(drift.eval_window_value, 1.1)  # mean([1.1, 2.1, 3.1]) - mean([1, 1, 1]) = 2.1 - 1 = 1.1
    
    def test_rolling_window_metrics(self):
        """Should compute metrics over rolling windows."""
        dates = pd.date_range("2025-01-01", periods=30)
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(30),
        })
        
        def count_metric(x):
            return float(len(x))
        
        result = rolling_window_metrics(
            df,
            metric_fn=count_metric,
            window_days=7,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert "window_start" in result.columns
        assert "window_end" in result.columns
        assert "metric_value" in result.columns
        assert len(result) > 0
    
    def test_rolling_window_metrics_missing_date_column(self):
        """Should return empty if date column missing."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = rolling_window_metrics(df, metric_fn=lambda x: float(len(x)))
        assert len(result) == 0


# ============================================================================
# REPORTING TESTS
# ============================================================================


class TestReporting:
    """Test reporting utilities."""
    
    def test_market_evaluation_report_creation(self):
        """Should create evaluation report with metadata."""
        report = MarketEvaluationReport(
            market="ML",
            num_samples=100,
            num_regimes=5,
        )
        
        assert report.market == "ML"
        assert report.num_samples == 100
        assert report.num_regimes == 5
        assert len(report.timestamp) > 0
    
    def test_market_evaluation_report_to_dict(self):
        """Should convert to JSON-serializable dict."""
        report = MarketEvaluationReport(
            market="spread",
            num_samples=50,
            num_regimes=3,
            metrics={"mae": 2.5, "rmse": 3.1},
            regime_metrics={"fav": {"mae": 2.0}, "ud": {"mae": 3.0}},
        )
        
        result = report.to_dict()
        assert result["market"] == "spread"
        assert result["num_samples"] == 50
        assert result["metrics"]["mae"] == 2.5
    
    def test_market_evaluation_report_json_write(self, tmp_path):
        """Should write report to JSON file."""
        report = MarketEvaluationReport(
            market="total",
            num_samples=75,
            num_regimes=4,
            metrics={"rmse": 1.5},
        )
        
        json_path = tmp_path / "report.json"
        report.to_json(json_path)
        
        assert json_path.exists()
        
        # Verify content
        import json
        with open(json_path) as f:
            data = json.load(f)
        assert data["market"] == "total"
    
    def test_summarize_evaluation_single_market(self):
        """Should produce readable summary."""
        report = MarketEvaluationReport(
            market="ML",
            num_samples=100,
            num_regimes=5,
            metrics={"brier": 0.15, "log_loss": 0.4},
        )
        
        summary = summarize_evaluation({"ML": report})
        
        assert "ML" in summary
        assert "brier" in summary
        assert "log_loss" in summary
    
    def test_summarize_evaluation_multiple_markets(self):
        """Should handle multiple markets in summary."""
        reports = {
            "ML": MarketEvaluationReport(
                market="ML",
                num_samples=100,
                num_regimes=3,
                metrics={"brier": 0.15},
            ),
            "SPREAD": MarketEvaluationReport(
                market="SPREAD",
                num_samples=100,
                num_regimes=4,
                metrics={"mae": 2.5},
            ),
        }
        
        summary = summarize_evaluation(reports)
        
        assert "ML" in summary
        assert "SPREAD" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
