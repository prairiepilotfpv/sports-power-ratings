"""Variance-based calibrators for SPREAD and TOTAL markets.

Modern SPREAD/TOTAL pipelines model predictions as Normal distributions
with a mean and standard deviation. This calibrator adjusts the variance
term directly while keeping the location term untouched:
σ' = sqrt((c · σ)^2 + τ^2).

The variances are fit by minimizing the Gaussian negative log likelihood
with simple quadratic regularization on c/τ and hard bounds to prevent
collapse. Health checks guard against over-clipping and extreme parameters
so downstream betting pipelines stay conservative.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import (
    VARIANCE_CALIBRATION_C_MIN,
    VARIANCE_CALIBRATION_C_MAX,
    VARIANCE_CALIBRATION_TAU_MAX,
    VARIANCE_CALIBRATION_LAMBDA_C,
    VARIANCE_CALIBRATION_LAMBDA_TAU,
    VARIANCE_CALIBRATION_CLIP_RATE_THRESHOLD,
)
from .base import BaseCalibrator

_LOG = logging.getLogger(__name__)


def _find_first_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _ensure_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


class VarianceCalibrator:
    """Calibrate the SD of a distribution prediction via variance scaling."""

    def __init__(
        self,
        *,
        guardrail_min: float | None = None,
        guardrail_max: float | None = None,
        c_min: float = VARIANCE_CALIBRATION_C_MIN,
        c_max: float = VARIANCE_CALIBRATION_C_MAX,
        tau_max: float = VARIANCE_CALIBRATION_TAU_MAX,
        lambda_c: float = VARIANCE_CALIBRATION_LAMBDA_C,
        lambda_tau: float = VARIANCE_CALIBRATION_LAMBDA_TAU,
        clip_rate_threshold: float = VARIANCE_CALIBRATION_CLIP_RATE_THRESHOLD,
    ) -> None:
        if c_min <= 0 or c_max <= 0 or c_min >= c_max:
            raise ValueError("c_min/c_max must be positive with c_min < c_max")
        if tau_max < 0:
            raise ValueError("tau_max must be non-negative")
        self.guardrail_min = guardrail_min
        self.guardrail_max = guardrail_max
        self.c_min = c_min
        self.c_max = c_max
        self.tau_max = tau_max
        self.lambda_c = lambda_c
        self.lambda_tau = lambda_tau
        self.clip_rate_threshold = clip_rate_threshold
        self.c = 1.0
        self.tau = 0.0
        self.clip_rate = 0.0
        self.median_raw_sd: float | None = None
        self.median_calibrated_sd: float | None = None
        self.healthy = True
        self.metadata: dict[str, object] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit c/τ to minimize the Gaussian NLL on (pred_mean, pred_sd)."""
        valid_df = df[
            (df["pred_mean"].notna())
            & (df["pred_sd"].notna())
            & (df["pred_sd"] > 0)
            & (df["actual_value"].notna())
        ].copy()

        if len(valid_df) < 2:
            raise ValueError(
                f"Not enough valid samples to fit calibrator (need 2+, got {len(valid_df)})"
            )

        pred_sd = _ensure_float_series(valid_df["pred_sd"]).to_numpy()
        residuals = _ensure_float_series(valid_df["actual_value"] - valid_df["pred_mean"]).to_numpy()
        samples = len(pred_sd)

        def objective(params: np.ndarray) -> float:
            c_val, tau_val = params
            variance = (c_val * pred_sd) ** 2 + tau_val ** 2
            nll = 0.5 * np.sum(np.log(variance) + (residuals ** 2) / variance)
            penalty = self.lambda_c * (c_val - 1.0) ** 2 + self.lambda_tau * (tau_val ** 2)
            return float(nll + penalty)

        bounds = [(self.c_min, self.c_max), (0.0, self.tau_max)]
        initial_tau = min(self.tau_max / 2.0 if self.tau_max > 0 else 0.0, 1.0)
        initial = np.array([1.0, initial_tau], dtype=float)

        try:
            result = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                bounds=bounds,
            )
        except Exception as exc:
            _LOG.warning(
                "[VarianceCalibrator] Optimization failed (exception): %s. Falling back to identity.",
                exc,
            )
            self._apply_identity(samples, reason="optimization_exception")
            return

        if not result.success or len(result.x) != 2:
            _LOG.warning(
                "[VarianceCalibrator] Optimization failed (%s). Falling back to identity.",
                result.message if result is not None else "unknown",
            )
            self._apply_identity(samples, reason="optimization_failure")
            return

        self.c = float(np.clip(result.x[0], self.c_min, self.c_max))
        self.tau = float(min(max(result.x[1], 0.0), self.tau_max))
        self._fitted = True
        self._finalize_stats(pred_sd, samples)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the variance calibration to predictions (means remain untouched)."""
        if not self._fitted:
            raise RuntimeError("VarianceCalibrator must be fitted before calling transform.")

        mean_col = _find_first_column(df, ["pred_mean", "margin_mean", "total_mean"])
        sd_col = _find_first_column(df, ["pred_sd", "margin_sd", "total_sd"])
        if mean_col is None or sd_col is None:
            missing = [name for name in ("pred_mean", "pred_sd") if name not in df.columns]
            raise KeyError(
                f"DataFrame must contain distribution columns; missing {missing}"
            )

        preds = df[[mean_col, sd_col]].copy()
        preds["pred_mean"] = _ensure_float_series(preds[mean_col])
        preds["pred_sd"] = _ensure_float_series(preds[sd_col])
        sd_values = preds["pred_sd"].to_numpy()
        calibrated_raw = np.sqrt((self.c * sd_values) ** 2 + self.tau ** 2)
        calibrated_used = self._apply_guardrail(calibrated_raw)

        result = pd.DataFrame(
            {
                "calibrated_mean": preds["pred_mean"],
                "calibrated_sd": calibrated_used,
            },
            index=df.index,
        )
        return result

    def _apply_guardrail(self, values: np.ndarray) -> np.ndarray:
        clipped = np.array(values, copy=True)
        if self.guardrail_min is not None:
            clipped = np.maximum(clipped, self.guardrail_min)
        if self.guardrail_max is not None:
            clipped = np.minimum(clipped, self.guardrail_max)
        return clipped

    def _finalize_stats(self, raw_sd: np.ndarray, samples: int) -> None:
        pre_guardrail = np.sqrt((self.c * raw_sd) ** 2 + self.tau ** 2)
        used_sd = self._apply_guardrail(pre_guardrail)
        self.median_raw_sd = float(np.nanmedian(raw_sd))
        self.median_calibrated_sd = float(np.nanmedian(used_sd))
        mask_clip = (
            (pre_guardrail < self.guardrail_min)
            if (self.guardrail_min is not None)
            else np.zeros_like(pre_guardrail, dtype=bool)
        )
        self.clip_rate = float(np.count_nonzero(mask_clip)) / samples if samples else 0.0
        boundary_hit = (
            abs(self.c - self.c_min) < 1e-3
            or abs(self.c - self.c_max) < 1e-3
            or (self.tau_max > 0 and abs(self.tau - self.tau_max) < 1e-3)
        )
        self.healthy = (self.clip_rate <= self.clip_rate_threshold) and not boundary_hit
        self.metadata = {
            "method": "variance_distribution",
            "samples": samples,
            "c": self.c,
            "tau": self.tau,
            "clip_rate": self.clip_rate,
            "healthy": self.healthy,
            "guardrail_min": self.guardrail_min,
            "guardrail_max": self.guardrail_max,
            "lambda_c": self.lambda_c,
            "lambda_tau": self.lambda_tau,
            "c_bounds": (self.c_min, self.c_max),
            "tau_max": self.tau_max,
        }
        if not self.healthy:
            _LOG.warning(
                "[VarianceCalibrator] Marked unhealthy (clip_rate=%.3f, guardrail=%s, c=%.3f, tau=%.3f)",
                self.clip_rate,
                self.guardrail_min,
                self.c,
                self.tau,
            )

    def _apply_identity(self, samples: int, reason: str) -> None:
        self.c = 1.0
        self.tau = 0.0
        self.clip_rate = 0.0
        self.median_raw_sd = None
        self.median_calibrated_sd = None
        self.healthy = False
        self._fitted = True
        self.metadata = {
            "method": "variance_identity",
            "samples": samples,
            "healthy": False,
            "reason": reason,
        }


class MarginalDistributionCalibrator(VarianceCalibrator):
    """Legacy alias remaining for backwards compatibility."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MarginalDistributionCalibrator is deprecated; use VarianceCalibrator instead.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class ProbabilityFromDistributionCalibrator(BaseCalibrator):
    """Convert distribution parameters to well-calibrated probabilities.

    For SPREAD/TOTAL markets, we predict distributions but need probabilities
    for evaluation/comparison. This calibrator:

    1. Given predicted distribution (mean, SD) and a line (threshold)
    2. Compute probability that actual > line using normal CDF
    3. Calibrate those probabilities using Platt/Isotonic methods

    This is different from direct probability calibration because we're also
    accounting for uncertainty in the predicted distribution itself.
    """

    def __init__(self, method: str = "platt") -> None:
        super().__init__()
        self.method = method  # "platt" or "isotonic"
        self.calibrator: Optional[BaseCalibrator] = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit calibrator from DataFrame with distribution + outcome info."""
        if df.empty:
            raise ValueError("Cannot fit calibrator on empty DataFrame")

        required = {"pred_mean", "pred_sd", "outcome"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"DataFrame must contain columns: {required}")

        valid_df = df[
            (df["pred_mean"].notna())
            & (df["pred_sd"].notna())
            & (df["pred_sd"] > 0)
            & (df["outcome"].notna())
        ].copy()

        if len(valid_df) < 2:
            raise ValueError(
                f"Not enough valid samples to fit calibrator (need 2+, got {len(valid_df)})"
            )

        def _prob_over_line(row: pd.Series) -> float:
            sd = float(row["pred_sd"])
            if sd <= 0:
                return 0.5
            mean = float(row["pred_mean"])
            line = float(row.get("line", mean))
            z = (line - mean) / sd
            from scipy.stats import norm
            return float(norm.sf(z))

        valid_df["p_outcome"] = valid_df.apply(_prob_over_line, axis=1)

        if self.method == "isotonic" or len(valid_df) >= 500:
            from .isotonic import IsotonicCalibrator
            self.calibrator = IsotonicCalibrator()
        else:
            from .platt import PlattScalingCalibrator
            self.calibrator = PlattScalingCalibrator()

        calib_df = pd.DataFrame({
            "p_home_win": valid_df["p_outcome"],
            "home_win": valid_df["outcome"].astype(int),
        })

        self.calibrator.fit(calib_df)
        self.metadata = {
            "method": f"prob_from_distribution_{self.method}",
            "n": int(len(valid_df)),
            "inner_method": getattr(self.calibrator, "metadata", {}).get("method", "unknown"),
        }

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.calibrator is None:
            raise RuntimeError("Calibrator not fitted")

        def _prob_over_line(row: pd.Series) -> float:
            mean = float(row.get("pred_mean", 0))
            sd = float(row.get("pred_sd", 1))
            if sd <= 0:
                return 0.5
            line = float(row.get("line", mean))
            z = (line - mean) / sd
            from scipy.stats import norm
            return float(norm.sf(z))

        derived_probs = df.apply(_prob_over_line, axis=1)
        return self.calibrator.transform(derived_probs)
