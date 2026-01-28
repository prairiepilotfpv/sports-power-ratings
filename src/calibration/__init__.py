"""Calibration utilities for post-fit probability calibration.

This package provides a small interface and implementations for mapping raw
model win-probabilities to calibrated probabilities suitable for downstream
EV and betting calculations. Implementations included:

- `PlattScalingCalibrator` - logistic (sigmoid) scaling using sklearn.
- `IsotonicCalibrator` - monotonic isotonic regression mapping.
- `VarianceCalibrator` - variance scaling for SPREAD/TOTAL distributions.
- `MarginalDistributionCalibrator` - legacy alias for variance calibrator.

Calibrators are intentionally simple and focus only on model outputs vs
observed outcomes; they must not access sportsbook data or change model
parameters.
"""

__all__ = [
    # Existing calibrator classes
    "BaseCalibrator",
    "PlattScalingCalibrator",
    "IsotonicCalibrator",
    "CalibratorRegistry",
    # Distribution calibrators
    "VarianceCalibrator",
    "MarginalDistributionCalibrator",
]

# Register sensible defaults for common sport/model combinations.
from .platt import PlattScalingCalibrator
from .isotonic import IsotonicCalibrator
from .registry import register_calibrator
from .distribution import MarginalDistributionCalibrator, VarianceCalibrator

# Use isotonic calibration for high-data professional leagues by default.
try:
    register_calibrator("nba", "elo", "ML", IsotonicCalibrator)
    register_calibrator("nba", "gssd", "ML", IsotonicCalibrator)
    register_calibrator("nhl", "gssd", "ML", IsotonicCalibrator)
    register_calibrator("nhl", "elo", "ML", IsotonicCalibrator)
except Exception:
    # Best-effort registration; avoid import-time failures in environments
    # where calibration package isn't used.
    pass

# Use Platt scaling for smaller-sample contexts (college sports).
try:
    register_calibrator("cbb", "elo", "ML", PlattScalingCalibrator)
except Exception:
    pass
