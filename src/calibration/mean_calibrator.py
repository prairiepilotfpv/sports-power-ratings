"""Mean calibration for SPREAD and TOTAL predictions.

This module provides a dedicated calibrator for shifting the mean (location)
of a distribution prediction without modifying variance (uncertainty).

The MeanCalibrator fits an additive bias correction:
  predicted_mean' = predicted_mean + delta

where delta is chosen to minimize squared error between predictions and actuals.
This approach is distinct from variance calibration, which adjusts uncertainty.

Design principles:
- Modifies mean ONLY (SD unchanged)
- Fails on invalid inputs (NaN, SD <= 0, non-numeric)
- Logs pre/post statistics for transparency
- Guardrails bound the mean shift to prevent extreme corrections
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseCalibrator
from .validation import validate_no_nans_in_columns

_LOG = logging.getLogger(__name__)


def _ensure_float_series(series: pd.Series) -> pd.Series:
    """Convert series to float, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce").astype(float)


class MeanCalibrator(BaseCalibrator):
    """Calibrate the mean (location) of a distribution prediction.

    Fits an additive bias δ such that:
      predicted_value' = predicted_value + δ

    This preserves variance and only shifts the center of the prediction.
    """

    def __init__(
        self,
        *,
        delta_min: float = -30.0,
        delta_max: float = 30.0,
        regularization_strength: float = 0.1,
    ) -> None:
        """Initialize MeanCalibrator.

        Args:
            delta_min: Minimum allowed bias shift (guardrail)
            delta_max: Maximum allowed bias shift (guardrail)
            regularization_strength: L2 regularization weight (pushes toward 0)
        """
        super().__init__()
        if delta_min >= delta_max:
            raise ValueError("delta_min must be < delta_max")
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.regularization_strength = regularization_strength
        self.delta = 0.0
        self.mean_before: Optional[float] = None
        self.mean_after: Optional[float] = None
        self.rmse_before: Optional[float] = None
        self.rmse_after: Optional[float] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit mean bias correction.

        Expects DataFrame with:
          - pred_mean: predicted mean values
          - actual_value: observed outcomes
          - pred_sd: (optional) predicted standard deviations (validated but not modified)

        Raises:
            ValueError: if inputs are invalid
            RuntimeError: if fitting fails
        """
        # Validate required columns
        if "pred_mean" not in df.columns:
            raise ValueError("DataFrame must contain 'pred_mean' column")
        if "actual_value" not in df.columns:
            raise ValueError("DataFrame must contain 'actual_value' column")

        # Validate no NaNs in critical columns (strict check)
        try:
            validate_no_nans_in_columns(df, ["pred_mean", "actual_value"], fail=True)
        except Exception as e:
            raise ValueError(f"Data validation failed: {e}")

        # Extract and validate data
        pred_mean = _ensure_float_series(df["pred_mean"])
        actual_value = _ensure_float_series(df["actual_value"])

        # If pred_sd is present, validate it (but don't modify it)
        if "pred_sd" in df.columns:
            pred_sd = _ensure_float_series(df["pred_sd"])
            # Check SD > 0 where present
            if (pred_sd[pred_sd.notna()] <= 0).any():
                raise ValueError("pred_sd must be positive where non-null")
        else:
            pred_sd = None

        # Filter to valid rows
        valid_mask = (pred_mean.notna()) & (actual_value.notna())
        if pred_sd is not None:
            valid_mask = valid_mask & (pred_sd.notna()) & (pred_sd > 0)

        if not valid_mask.any():
            raise ValueError("No valid samples for mean calibration fitting")

        valid_pred_mean = pred_mean[valid_mask].to_numpy()
        valid_actual = actual_value[valid_mask].to_numpy()
        valid_count = len(valid_pred_mean)

        if valid_count < 2:
            raise ValueError(
                f"Need at least 2 valid samples for mean calibration, got {valid_count}"
            )

        # Compute residuals
        residuals = valid_actual - valid_pred_mean

        # Fit delta to minimize MSE + regularization
        # Objective: min_delta [MSE(delta) + lambda * delta^2]
        # = min_delta [mean((actual - (pred_mean + delta))^2) + lambda * delta^2]
        # = min_delta [mean((residuals - delta)^2) + lambda * delta^2]
        # Derivative: -2 * mean(residuals - delta) + 2 * lambda * delta = 0
        # => delta = mean(residuals) / (1 + lambda)
        mean_residual = float(np.mean(residuals))
        denominator = 1.0 + self.regularization_strength
        self.delta = mean_residual / denominator

        # Apply guardrails
        self.delta = np.clip(self.delta, self.delta_min, self.delta_max)

        # Store statistics
        self.mean_before = float(np.mean(valid_pred_mean))
        self.mean_after = self.mean_before + self.delta
        self.rmse_before = float(np.sqrt(np.mean(residuals ** 2)))
        self.rmse_after = float(
            np.sqrt(np.mean((residuals - self.delta) ** 2))
        )

        # Store metadata
        self.metadata = {
            "method": "mean_shift",
            "samples": valid_count,
            "delta": self.delta,
            "delta_bounds": (self.delta_min, self.delta_max),
            "regularization": self.regularization_strength,
            "mean_before": self.mean_before,
            "mean_after": self.mean_after,
            "rmse_before": self.rmse_before,
            "rmse_after": self.rmse_after,
        }

        self._fitted = True

        # Log results
        pct_improvement = (
            100.0 * (self.rmse_before - self.rmse_after) / self.rmse_before
            if self.rmse_before > 0
            else 0.0
        )
        _LOG.debug(
            "[MeanCalibrator] Fitted: delta=%+.4f, "
            "mean_before=%.4f, mean_after=%.4f, "
            "rmse_before=%.4f, rmse_after=%.4f, improvement=%+.1f%%",
            self.delta,
            self.mean_before,
            self.mean_after,
            self.rmse_before,
            self.rmse_after,
            pct_improvement,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply mean bias correction.

        Args:
            df: DataFrame with 'pred_mean' (and optionally 'pred_sd')

        Returns:
            DataFrame with 'calibrated_mean' and 'calibrated_sd' columns.
            SD is unchanged; mean is shifted by delta.

        Raises:
            RuntimeError: if not fitted
            KeyError: if required columns missing
        """
        if not self._fitted:
            raise RuntimeError("MeanCalibrator must be fitted before calling transform.")

        if "pred_mean" not in df.columns:
            raise KeyError("DataFrame must contain 'pred_mean' column")

        # Validate SD if present
        if "pred_sd" in df.columns:
            pred_sd = _ensure_float_series(df["pred_sd"])
            if (pred_sd[pred_sd.notna()] <= 0).any():
                raise ValueError("pred_sd must be positive where non-null")
        else:
            pred_sd = None

        pred_mean = _ensure_float_series(df["pred_mean"])

        result = pd.DataFrame(
            {
                "calibrated_mean": pred_mean + self.delta,
                "calibrated_sd": pred_sd if pred_sd is not None else None,
            },
            index=df.index,
        )

        return result
