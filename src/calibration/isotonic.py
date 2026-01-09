"""Isotonic regression calibrator (monotonic mapping)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .base import BaseCalibrator


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression mapping for probability calibration.

    This ensures monotonicity and is non-parametric. It can overfit on
    small samples; callers should choose isotonic only when sufficient data
    is available.
    """

    def __init__(self, out_of_bounds: str = "clip") -> None:
        super().__init__()
        self.out_of_bounds = out_of_bounds
        self.model: Optional[IsotonicRegression] = None

    def fit(self, df: pd.DataFrame) -> None:
        xs = np.clip(df["p_home_win"].astype(float).fillna(0.5).values, 1e-6, 1 - 1e-6)
        ys = df["home_win"].astype(float).values
        self.model = IsotonicRegression(increasing=True, out_of_bounds=self.out_of_bounds)
        self.model.fit(xs, ys)
        self.metadata = {"method": "isotonic", "n": int(len(xs))}

    def transform(self, probs: pd.Series) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Calibrator not fitted")
        xs = np.clip(probs.astype(float).fillna(0.5).values, 1e-6, 1 - 1e-6)
        preds = self.model.transform(xs)
        # ensure valid range
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=probs.index)
