"""Distribution-based calibrators for SPREAD and TOTAL markets.

These calibrators work with continuous distribution parameters (mean, std dev)
rather than binary probabilities. They're used for markets where predictions
are given as normal distributions (e.g., "margin will be 5.2 ± 3.1 points").

Key insight: For SPREAD/TOTAL, we don't calibrate a single probability,
we calibrate the distribution parameters themselves so that predicted
distributions better match empirical outcomes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize

from .base import BaseCalibrator


class MarginalDistributionCalibrator(BaseCalibrator):
    """Calibrate margin/total distribution parameters to empirical outcomes.

    Instead of calibrating a binary probability, this class fits the predicted
    distribution (mean and SD) to better match actual outcomes.

    For a margin prediction with mean μ and SD σ, we want P(actual_margin) to be
    well-calibrated. This is achieved by adjusting μ and σ.

    Approach:
    - Store pairs of (predicted_mean, predicted_sd, actual_value)
    - Fit a transformation that improves alignment
    - Simple linear adjustment: actual ≈ α * predicted_mean + β
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean_scale: float = 1.0
        self.mean_shift: float = 0.0
        self.sd_scale: float = 1.0
        self.sd_clip_min: float = 0.1  # Prevent SD from collapsing to 0

    def fit(self, df: pd.DataFrame) -> None:
        """Fit calibrator from DataFrame with columns: pred_mean, pred_sd, actual_value.

        Args:
            df: DataFrame with:
                - pred_mean: predicted distribution mean
                - pred_sd: predicted distribution standard deviation
                - actual_value: actual realized value (margin or total)
        """
        if df.empty:
            raise ValueError("Cannot fit calibrator on empty DataFrame")

        if not all(col in df.columns for col in ["pred_mean", "pred_sd", "actual_value"]):
            raise ValueError(
                "DataFrame must contain columns: pred_mean, pred_sd, actual_value"
            )

        # Remove rows with invalid predictions
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

        # Fit mean calibration: predict actual_value from pred_mean
        X_mean = np.array(valid_df["pred_mean"]).reshape(-1, 1)
        y_actual = np.array(valid_df["actual_value"])

        # Simple linear regression: actual = α * pred_mean + β
        # Add regularization to prevent extreme values
        from sklearn.linear_model import Ridge

        mean_model = Ridge(alpha=0.1)
        mean_model.fit(X_mean, y_actual)

        self.mean_scale = float(mean_model.coef_[0])
        self.mean_shift = float(mean_model.intercept_)

        # Fit SD calibration: predict residual SD from predicted SD
        predicted_actual = self.mean_scale * valid_df["pred_mean"] + self.mean_shift
        residuals = np.abs(y_actual - predicted_actual)

        # Empirical SD of residuals vs predicted SD
        X_sd = np.array(valid_df["pred_sd"]).reshape(-1, 1)
        sd_model = Ridge(alpha=0.1)
        sd_model.fit(X_sd, residuals)

        self.sd_scale = float(sd_model.coef_[0])
        if self.sd_scale < 0:
            # If fitted scale is negative, use empirical SD ratio
            self.sd_scale = np.mean(residuals) / np.mean(valid_df["pred_sd"])

        self.metadata = {
            "method": "marginal_distribution",
            "n": int(len(valid_df)),
            "mean_scale": self.mean_scale,
            "mean_shift": self.mean_shift,
            "sd_scale": self.sd_scale,
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply calibration to predicted distributions.

        Args:
            df: DataFrame with columns pred_mean and pred_sd

        Returns:
            DataFrame with calibrated_mean and calibrated_sd
        """
        result = df[["pred_mean", "pred_sd"]].copy()

        result["calibrated_mean"] = (
            self.mean_scale * df["pred_mean"] + self.mean_shift
        )
        result["calibrated_sd"] = np.maximum(
            self.sd_clip_min, self.sd_scale * df["pred_sd"]
        )

        return result


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
        """Fit calibrator from DataFrame with distribution + outcome info.

        Args:
            df: DataFrame with columns:
                - pred_mean: predicted mean
                - pred_sd: predicted std dev
                - line: threshold for computing probability (can be pred_mean if not given)
                - outcome: actual binary outcome (0 or 1)
        """
        if df.empty:
            raise ValueError("Cannot fit calibrator on empty DataFrame")

        required = {"pred_mean", "pred_sd", "outcome"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"DataFrame must contain columns: {required}")

        # Compute implied probabilities from distributions
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

        # Compute probability that actual > line using normal CDF
        def _prob_over_line(row):
            mean = row["pred_mean"]
            sd = row["pred_sd"]
            line = row.get("line", mean)  # Default to mean if line not provided
            if sd <= 0:
                return 0.5
            z = (line - mean) / sd
            # Normal CDF: P(X > line) where X ~ N(mean, sd)
            from scipy.stats import norm
            return float(norm.sf(z))  # survival function = 1 - CDF

        valid_df["p_outcome"] = valid_df.apply(_prob_over_line, axis=1)

        # Now fit a probability calibrator on these derived probabilities
        # Use appropriate calibrator based on sample size
        if self.method == "isotonic" or len(valid_df) >= 500:
            from .isotonic import IsotonicCalibrator
            self.calibrator = IsotonicCalibrator()
        else:
            from .platt import PlattScalingCalibrator
            self.calibrator = PlattScalingCalibrator()

        # Fit on (implied_probability, actual_outcome)
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
        """Calibrate probabilities derived from distributions.

        Args:
            df: DataFrame with pred_mean, pred_sd, and optional line

        Returns:
            Series of calibrated probabilities
        """
        if self.calibrator is None:
            raise RuntimeError("Calibrator not fitted")

        # Recompute probabilities from distributions
        def _prob_over_line(row):
            mean = row.get("pred_mean", 0)
            sd = row.get("pred_sd", 1)
            line = row.get("line", mean)
            if sd <= 0:
                return 0.5
            z = (line - mean) / sd
            from scipy.stats import norm
            return float(norm.sf(z))

        derived_probs = df.apply(_prob_over_line, axis=1)

        # Now calibrate these probabilities
        return self.calibrator.transform(derived_probs)
