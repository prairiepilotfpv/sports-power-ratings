"""Variance audit diagnostics for distribution calibrators.

This module provides tools to assess calibration quality by comparing
predicted uncertainty (SD) to empirical coverage of residuals.

Metrics computed:
- MAE of predicted SD vs empirical absolute residuals
- Coverage at ±1σ and ±2σ
- Percentage of outcomes outside 95% CI
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


class VarianceAuditResult(NamedTuple):
    """Results from variance audit."""
    sd_before: float
    sd_after: float
    empirical_mae: float
    coverage_1sd: float  # fraction within ±1σ
    coverage_2sd: float  # fraction within ±2σ
    outside_95ci: float  # fraction outside 95% CI


def compute_variance_audit(
    predictions: pd.DataFrame,
    *,
    pred_mean_col: str = "pred_mean",
    pred_sd_col: str = "pred_sd",
    actual_col: str = "actual_value",
) -> VarianceAuditResult:
    """Compute variance audit metrics.

    Args:
        predictions: DataFrame with predictions and actuals
        pred_mean_col: Name of predicted mean column
        pred_sd_col: Name of predicted SD column
        actual_col: Name of actual value column

    Returns:
        VarianceAuditResult with audit metrics

    Raises:
        ValueError: if inputs are invalid
    """
    # Validate inputs
    for col in [pred_mean_col, pred_sd_col, actual_col]:
        if col not in predictions.columns:
            raise ValueError(f"Missing required column: {col}")

    # Extract valid rows
    valid_mask = (
        (predictions[pred_mean_col].notna())
        & (predictions[pred_sd_col].notna())
        & (predictions[pred_sd_col] > 0)
        & (predictions[actual_col].notna())
    )

    if not valid_mask.any():
        raise ValueError("No valid samples for variance audit")

    means = predictions.loc[valid_mask, pred_mean_col].astype(float).values
    sds = predictions.loc[valid_mask, pred_sd_col].astype(float).values
    actuals = predictions.loc[valid_mask, actual_col].astype(float).values

    # Compute residuals
    residuals = actuals - means
    abs_residuals = np.abs(residuals)

    # Metrics
    sd_before = float(np.median(sds))
    empirical_mae = float(np.mean(abs_residuals))

    # Coverage within ±1σ and ±2σ
    within_1sd = np.abs(residuals) <= sds
    within_2sd = np.abs(residuals) <= 2 * sds
    coverage_1sd = float(np.mean(within_1sd))
    coverage_2sd = float(np.mean(within_2sd))

    # Fraction outside 95% CI (±1.96σ)
    outside_95ci = np.abs(residuals) > 1.96 * sds
    outside_95ci_pct = float(np.mean(outside_95ci))

    return VarianceAuditResult(
        sd_before=sd_before,
        sd_after=sd_before,  # Will be updated if calibrated
        empirical_mae=empirical_mae,
        coverage_1sd=coverage_1sd,
        coverage_2sd=coverage_2sd,
        outside_95ci=outside_95ci_pct,
    )


def log_variance_audit(audit: VarianceAuditResult, market: str, *, level: int = logging.INFO) -> None:
    """Log variance audit results in a concise format.

    Args:
        audit: VarianceAuditResult from compute_variance_audit
        market: Market name (e.g., "TOTAL", "SPREAD")
        level: Logging level (default INFO)
    """
    _LOG.log(
        level,
        "[VARIANCE AUDIT %s] sd_before=%.3f, sd_after=%.3f, "
        "empirical_mae=%.3f, coverage_1sd=%.1f%%, coverage_2sd=%.1f%%, "
        "outside_95ci=%.1f%%",
        market,
        audit.sd_before,
        audit.sd_after,
        audit.empirical_mae,
        100.0 * audit.coverage_1sd,
        100.0 * audit.coverage_2sd,
        100.0 * audit.outside_95ci,
    )
