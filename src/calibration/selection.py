"""Helpers for selecting a calibrator implementation."""

from __future__ import annotations

from calibration.isotonic import IsotonicCalibrator
from calibration.platt import PlattScalingCalibrator


def select_calibrator(method: str | None, n_samples: int):
    """Pick the right calibrator based on method hint and sample count."""
    normalized = (method or "").strip().lower()
    if normalized in {"platt", "logistic"}:
        return PlattScalingCalibrator()
    if normalized == "isotonic":
        return IsotonicCalibrator()
    if normalized and normalized != "auto":
        raise ValueError("calibrator method must be auto, platt, or isotonic.")
    return IsotonicCalibrator() if n_samples >= 500 else PlattScalingCalibrator()
