"""Calibration quality evaluation: metrics, regime slicing, and drift diagnostics.

This module provides evaluation utilities to assess calibration quality without
modifying predictions or calibration logic. Key components:

1. **Quality Metrics**:
   - ML: Brier score, log loss, reliability curve
   - SPREAD/TOTAL: MAE, RMSE, empirical coverage at 1σ/2σ, tail miss rate, bias

2. **Regime Slicing**:
   - Composable, declarative slicing by favorite/underdog, spread/total bucket, 
     home/away, season segment, market distance, etc.

3. **Drift & Stability**:
   - Rolling-window metric tracking
   - Fit vs eval window comparison
   - Degradation warnings

4. **Reporting**:
   - JSON reports with per-market diagnostics
   - Optional CSV for plotting
   - Summary console output
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

_LOG = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class Market(Enum):
    """Market types."""
    ML = "ML"
    SPREAD = "spread"
    TOTAL = "total"


class RegimeType(Enum):
    """Regime slicing dimension types."""
    FAVORITE_VS_UNDERDOG = "favorite_vs_underdog"
    SPREAD_BUCKET = "spread_bucket"
    TOTAL_BUCKET = "total_bucket"
    HOME_VS_AWAY = "home_vs_away"
    SEASON_SEGMENT = "season_segment"
    MARKET_DISTANCE = "market_distance"


# ============================================================================
# ML METRICS
# ============================================================================


def brier_score(probs: pd.Series | np.ndarray, outcomes: pd.Series | np.ndarray) -> float:
    """Compute Brier score: mean((p - y)^2).
    
    Args:
        probs: Predicted probabilities [0, 1]
        outcomes: Binary outcomes {0, 1}
    
    Returns:
        Brier score (lower is better)
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return np.nan
    return float(np.mean((p - y) ** 2))


def log_loss(probs: pd.Series | np.ndarray, outcomes: pd.Series | np.ndarray) -> float:
    """Compute log loss: -mean(y*log(p) + (1-y)*log(1-p)).
    
    Args:
        probs: Predicted probabilities [0, 1]
        outcomes: Binary outcomes {0, 1}
    
    Returns:
        Log loss (lower is better)
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return np.nan
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_error(probs: pd.Series | np.ndarray, outcomes: pd.Series | np.ndarray) -> float:
    """Compute Expected Calibration Error (ECE): average |avg_pred - avg_actual| per bin.
    
    Args:
        probs: Predicted probabilities [0, 1]
        outcomes: Binary outcomes {0, 1}
    
    Returns:
        ECE (lower is better)
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return np.nan
    
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    n_samples = 0
    for i in range(len(bins) - 1):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() > 0:
            avg_pred = p[mask].mean()
            avg_actual = y[mask].mean()
            ece += abs(avg_pred - avg_actual) * mask.sum()
            n_samples += mask.sum()
    
    return float(ece / n_samples) if n_samples > 0 else np.nan


def reliability_curve(
    probs: pd.Series | np.ndarray,
    outcomes: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
    min_count: int = 5,
) -> pd.DataFrame:
    """Compute reliability curve (predicted vs actual in bins).
    
    Args:
        probs: Predicted probabilities [0, 1]
        outcomes: Binary outcomes {0, 1}
        n_bins: Number of bins for discretization
        min_count: Minimum samples per bin (skip smaller bins)
    
    Returns:
        DataFrame with columns: bin_lower, bin_upper, count, avg_pred, avg_actual, calibration_gap
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y = np.asarray(outcomes, dtype=float)
    
    if len(p) == 0:
        return pd.DataFrame()
    
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    
    for i in range(len(bins) - 1):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        count = mask.sum()
        
        if count < min_count:
            continue
        
        avg_pred = float(p[mask].mean())
        avg_actual = float(y[mask].mean())
        gap = avg_pred - avg_actual
        
        rows.append({
            "bin_lower": float(bins[i]),
            "bin_upper": float(bins[i + 1]),
            "count": int(count),
            "avg_pred": avg_pred,
            "avg_actual": avg_actual,
            "calibration_gap": gap,
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# SPREAD / TOTAL METRICS
# ============================================================================


def mean_absolute_error(
    pred_means: pd.Series | np.ndarray, actual: pd.Series | np.ndarray
) -> float:
    """MAE: mean(|pred - actual|)."""
    pred = np.asarray(pred_means, dtype=float)
    act = np.asarray(actual, dtype=float)
    if len(pred) == 0:
        return np.nan
    return float(np.mean(np.abs(pred - act)))


def root_mean_squared_error(
    pred_means: pd.Series | np.ndarray, actual: pd.Series | np.ndarray
) -> float:
    """RMSE: sqrt(mean((pred - actual)^2))."""
    pred = np.asarray(pred_means, dtype=float)
    act = np.asarray(actual, dtype=float)
    if len(pred) == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred - act) ** 2)))


def mean_squared_error(
    pred_means: pd.Series | np.ndarray, actual: pd.Series | np.ndarray
) -> float:
    """MSE: mean((pred - actual)^2)."""
    pred = np.asarray(pred_means, dtype=float)
    act = np.asarray(actual, dtype=float)
    if len(pred) == 0:
        return np.nan
    return float(np.mean((pred - act) ** 2))


def mean_signed_error(
    pred_means: pd.Series | np.ndarray, actual: pd.Series | np.ndarray
) -> float:
    """MSE (bias): mean(pred - actual). Positive = overestimate."""
    pred = np.asarray(pred_means, dtype=float)
    act = np.asarray(actual, dtype=float)
    if len(pred) == 0:
        return np.nan
    return float(np.mean(pred - act))


def empirical_coverage(
    pred_means: pd.Series | np.ndarray,
    pred_sds: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    *,
    sigma: float = 1.0,
) -> float:
    """Empirical coverage: fraction of actuals within pred_mean ± sigma*pred_sd.
    
    Args:
        pred_means: Predicted mean values
        pred_sds: Predicted standard deviations
        actual: Actual realized values
        sigma: Number of standard deviations (default 1.0)
    
    Returns:
        Coverage fraction [0, 1]
    """
    pred_mu = np.asarray(pred_means, dtype=float)
    pred_sigma = np.asarray(pred_sds, dtype=float)
    act = np.asarray(actual, dtype=float)
    
    if len(pred_mu) == 0:
        return np.nan
    
    lower = pred_mu - sigma * pred_sigma
    upper = pred_mu + sigma * pred_sigma
    covered = np.logical_and(act >= lower, act <= upper)
    
    return float(covered.mean())


def tail_miss_rate(
    pred_means: pd.Series | np.ndarray,
    pred_sds: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    *,
    sigma: float = 2.0,
) -> float:
    """Tail miss rate: fraction of actuals beyond sigma*pred_sd.
    
    Args:
        pred_means: Predicted mean values
        pred_sds: Predicted standard deviations
        actual: Actual realized values
        sigma: Number of standard deviations (default 2.0)
    
    Returns:
        Tail miss fraction [0, 1]
    """
    pred_mu = np.asarray(pred_means, dtype=float)
    pred_sigma = np.asarray(pred_sds, dtype=float)
    act = np.asarray(actual, dtype=float)
    
    if len(pred_mu) == 0:
        return np.nan
    
    lower = pred_mu - sigma * pred_sigma
    upper = pred_mu + sigma * pred_sigma
    misses = np.logical_or(act < lower, act > upper)
    
    return float(misses.mean())


# ============================================================================
# REGIME SLICING
# ============================================================================


@dataclass(frozen=True)
class SlicerConfig:
    """Configuration for a single slicer dimension."""
    regime_type: RegimeType
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


class Slicer:
    """Base class for regime slicers."""
    
    def slice(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Slice DataFrame by regime.
        
        Args:
            df: Schedule DataFrame with required columns
        
        Returns:
            Dict mapping slice name -> slice DataFrame
        """
        raise NotImplementedError


class FavoriteVsUnderdogSlicer(Slicer):
    """Slice by favorite vs underdog (based on opening line)."""
    
    def __init__(self, spread_col: str = "opening_spread"):
        self.spread_col = spread_col
    
    def slice(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if self.spread_col not in df.columns:
            return {}
        
        spread = pd.to_numeric(df[self.spread_col], errors="coerce")
        
        favorites = df[spread < 0].copy()
        underdogs = df[spread > 0].copy()
        
        return {
            "favorites": favorites,
            "underdogs": underdogs,
        }


class SpreadBucketSlicer(Slicer):
    """Slice by spread magnitude bucket."""
    
    def __init__(self, spread_col: str = "opening_spread", buckets: List[float] | None = None):
        self.spread_col = spread_col
        self.buckets = buckets or [0, 3, 6, 10, 20]  # Default: PK-3, 3-6, 6-10, 10-20, 20+
    
    def slice(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if self.spread_col not in df.columns:
            return {}
        
        abs_spread = pd.to_numeric(df[self.spread_col], errors="coerce").abs()
        slices = {}
        
        for i in range(len(self.buckets) - 1):
            lower = self.buckets[i]
            upper = self.buckets[i + 1]
            mask = (abs_spread >= lower) & (abs_spread < upper)
            label = f"{lower}-{upper}"
            slices[label] = df[mask].copy()
        
        # Handle tail bucket
        if len(self.buckets) > 0:
            mask = abs_spread >= self.buckets[-1]
            slices[f"{self.buckets[-1]}+"] = df[mask].copy()
        
        return slices


class TotalBucketSlicer(Slicer):
    """Slice by total bucket."""
    
    def __init__(self, total_col: str = "opening_total", buckets: List[float] | None = None):
        self.total_col = total_col
        self.buckets = buckets or [0, 180, 210, 240, 500]  # Default for sports scoring
    
    def slice(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if self.total_col not in df.columns:
            return {}
        
        total = pd.to_numeric(df[self.total_col], errors="coerce")
        slices = {}
        
        for i in range(len(self.buckets) - 1):
            lower = self.buckets[i]
            upper = self.buckets[i + 1]
            mask = (total >= lower) & (total < upper)
            label = f"{lower}-{upper}"
            slices[label] = df[mask].copy()
        
        if len(self.buckets) > 0:
            mask = total >= self.buckets[-1]
            slices[f"{self.buckets[-1]}+"] = df[mask].copy()
        
        return slices


class HomeAwaySlicer(Slicer):
    """Slice by home vs away."""
    
    def slice(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {
            "home": df[df["is_home"] == True].copy() if "is_home" in df.columns else pd.DataFrame(),
            "away": df[df["is_home"] == False].copy() if "is_home" in df.columns else pd.DataFrame(),
        }


class SeasonSegmentSlicer(Slicer):
    """Slice by season segment (early/mid/late)."""
    
    def __init__(self, date_col: str = "date", boundaries: List[Tuple[int, int]] | None = None):
        self.date_col = date_col
        # Default: (month_start, month_end) pairs
        self.boundaries = boundaries or [(1, 3), (4, 6), (7, 9), (10, 12)]
    
    def slice(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if self.date_col not in df.columns:
            return {}
        
        dates = pd.to_datetime(df[self.date_col], errors="coerce")
        slices = {}
        
        for start_month, end_month in self.boundaries:
            mask = (dates.dt.month >= start_month) & (dates.dt.month <= end_month)
            label = f"{start_month:02d}-{end_month:02d}"
            slices[label] = df[mask].copy()
        
        return slices


class MarketDistanceSlicer(Slicer):
    """Slice by market distance (|line - pred_mean|)."""
    
    def __init__(
        self,
        line_col: str = "opening_spread",
        mean_col: str = "proj_margin_mean",
        buckets: List[float] | None = None,
    ):
        self.line_col = line_col
        self.mean_col = mean_col
        self.buckets = buckets or [0, 1, 2, 4, 100]  # Points distance
    
    def slice(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if self.line_col not in df.columns or self.mean_col not in df.columns:
            return {}
        
        line = pd.to_numeric(df[self.line_col], errors="coerce")
        mean = pd.to_numeric(df[self.mean_col], errors="coerce")
        distance = (line - mean).abs()
        
        slices = {}
        for i in range(len(self.buckets) - 1):
            lower = self.buckets[i]
            upper = self.buckets[i + 1]
            mask = (distance >= lower) & (distance < upper)
            label = f"{lower}-{upper}"
            slices[label] = df[mask].copy()
        
        if len(self.buckets) > 0:
            mask = distance >= self.buckets[-1]
            slices[f"{self.buckets[-1]}+"] = df[mask].copy()
        
        return slices


def build_slicer(config: SlicerConfig) -> Slicer:
    """Factory function to build a slicer from config."""
    slicer_map = {
        RegimeType.FAVORITE_VS_UNDERDOG: FavoriteVsUnderdogSlicer,
        RegimeType.SPREAD_BUCKET: SpreadBucketSlicer,
        RegimeType.TOTAL_BUCKET: TotalBucketSlicer,
        RegimeType.HOME_VS_AWAY: HomeAwaySlicer,
        RegimeType.SEASON_SEGMENT: SeasonSegmentSlicer,
        RegimeType.MARKET_DISTANCE: MarketDistanceSlicer,
    }
    
    cls = slicer_map.get(config.regime_type)
    if cls is None:
        raise ValueError(f"Unknown regime type: {config.regime_type}")
    
    return cls(**config.kwargs)


# ============================================================================
# DRIFT & STABILITY DIAGNOSTICS
# ============================================================================


@dataclass
class DriftMetrics:
    """Drift diagnostics comparing fit window vs eval window."""
    metric_name: str
    fit_window_value: float
    eval_window_value: float
    absolute_change: float
    percent_change: float
    tolerance: float = 0.15  # 15% default tolerance
    is_degraded: bool = field(init=False)
    
    def __post_init__(self):
        """Determine if metric degraded beyond tolerance."""
        if np.isnan(self.fit_window_value) or np.isnan(self.eval_window_value):
            self.is_degraded = False
        else:
            if self.fit_window_value == 0:
                self.is_degraded = self.percent_change != 0
            else:
                self.is_degraded = abs(self.percent_change) > self.tolerance


def compute_drift_metrics(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    metric_fn: Callable[[pd.DataFrame], float],
    metric_name: str,
    tolerance: float = 0.15,
) -> DriftMetrics:
    """Compute drift for a single metric across fit vs eval windows.
    
    Args:
        fit_df: DataFrame from fit window
        eval_df: DataFrame from eval window
        metric_fn: Function that takes DataFrame and returns float
        metric_name: Name of metric for logging
        tolerance: Percent change tolerance [0, 1]
    
    Returns:
        DriftMetrics object with absolute and relative changes
    """
    fit_value = metric_fn(fit_df)
    eval_value = metric_fn(eval_df)
    
    abs_change = eval_value - fit_value
    
    if fit_value == 0:
        pct_change = 0.0 if eval_value == 0 else float("inf")
    else:
        pct_change = abs_change / fit_value
    
    return DriftMetrics(
        metric_name=metric_name,
        fit_window_value=fit_value,
        eval_window_value=eval_value,
        absolute_change=abs_change,
        percent_change=pct_change,
        tolerance=tolerance,
    )


def rolling_window_metrics(
    df: pd.DataFrame,
    *,
    metric_fn: Callable[[pd.DataFrame], float],
    date_col: str = "date",
    window_days: int = 7,
) -> pd.DataFrame:
    """Compute metric over rolling time windows.
    
    Args:
        df: Schedule DataFrame with date column
        metric_fn: Function that takes DataFrame and returns float
        date_col: Name of date column
        window_days: Rolling window size in days
    
    Returns:
        DataFrame with columns: window_start, window_end, metric_value
    """
    if date_col not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    
    if len(df) == 0:
        return pd.DataFrame()
    
    df = df.sort_values(date_col)
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    rows = []
    current = min_date
    
    while current <= max_date:
        window_end = current + timedelta(days=window_days)
        mask = (df[date_col] >= current) & (df[date_col] < window_end)
        
        window_df = df[mask]
        if len(window_df) > 0:
            value = metric_fn(window_df)
            rows.append({
                "window_start": current.date(),
                "window_end": window_end.date(),
                "metric_value": value,
                "sample_count": len(window_df),
            })
        
        current = window_end
    
    return pd.DataFrame(rows)


# ============================================================================
# REPORTING & OUTPUTS
# ============================================================================


@dataclass
class MarketEvaluationReport:
    """Calibration evaluation report for a single market."""
    market: str
    num_samples: int
    num_regimes: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Top-level metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Per-regime metrics
    regime_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Drift diagnostics
    drift_metrics: Dict[str, DriftMetrics] = field(default_factory=dict)
    
    # Rolling window results
    rolling_window_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "market": self.market,
            "num_samples": self.num_samples,
            "num_regimes": self.num_regimes,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "regime_metrics": self.regime_metrics,
            "drift_metrics": {
                k: asdict(v) for k, v in self.drift_metrics.items()
            },
        }
        
        if not self.rolling_window_results.empty:
            result["rolling_window_results"] = self.rolling_window_results.to_dict(orient="records")
        
        return result
    
    def to_json(self, path: Path | str) -> None:
        """Write report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        _LOG.info(f"[MarketEvaluationReport] Wrote report to {path}")


def summarize_evaluation(
    reports: Dict[str, MarketEvaluationReport],
    *,
    max_regimes: int = 5,
) -> str:
    """Generate concise summary of evaluation results.
    
    Args:
        reports: Dict mapping market name -> report
        max_regimes: Max regimes to show in summary
    
    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 70,
        "CALIBRATION EVALUATION SUMMARY",
        "=" * 70,
    ]
    
    for market_name, report in reports.items():
        lines.append(f"\n{market_name.upper()}:")
        lines.append(f"  Samples: {report.num_samples}")
        lines.append(f"  Regimes evaluated: {report.num_regimes}")
        
        if report.metrics:
            lines.append("  Top-level metrics:")
            for metric_name, value in report.metrics.items():
                if np.isnan(value):
                    lines.append(f"    {metric_name}: N/A")
                else:
                    lines.append(f"    {metric_name}: {value:.4f}")
        
        if report.drift_metrics:
            lines.append("  Drift diagnostics:")
            for drift_name, drift in report.drift_metrics.items():
                status = "[DEGRADED]" if drift.is_degraded else "[OK]"
                lines.append(f"    {drift_name}: {drift.eval_window_value:.4f} (Change {drift.percent_change:+.2%}) {status}")
        
        worst_regimes = sorted(
            report.regime_metrics.items(),
            key=lambda x: max((v for v in x[1].values() if not np.isnan(v)), default=np.nan),
            reverse=True,
        )[:max_regimes]
        
        if worst_regimes:
            lines.append(f"  Worst-performing regimes (by first metric):")
            for regime_name, metrics in worst_regimes:
                first_metric = next(iter(metrics.values()), np.nan)
                if not np.isnan(first_metric):
                    lines.append(f"    {regime_name}: {first_metric:.4f}")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
