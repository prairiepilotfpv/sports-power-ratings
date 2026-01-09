"""Platt scaling calibrator (logistic regression on raw probabilities)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import BaseCalibrator


class PlattScalingCalibrator(BaseCalibrator):
    """Simple Platt scaling using sklearn LogisticRegression.

    Uses raw probability as a single feature. Logistic regression ensures a
    smooth, monotonic mapping from raw probabilities to calibrated ones.
    """

    def __init__(self, C: float = 1.0) -> None:
        super().__init__()
        self.C = float(C)
        self.model: Optional[LogisticRegression] = None

    def fit(self, df: pd.DataFrame) -> None:
        xs = np.clip(df["p_home_win"].astype(float).fillna(0.5).values.reshape(-1, 1), 1e-6, 1 - 1e-6)
        ys = df["home_win"].astype(int).values
        self.model = LogisticRegression(C=self.C, solver="lbfgs")
        self.model.fit(xs, ys)
        self.metadata = {"method": "platt", "n": int(len(xs)), "C": self.C}

    def transform(self, probs: pd.Series) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Calibrator not fitted")
        xs = np.clip(probs.astype(float).fillna(0.5).values.reshape(-1, 1), 1e-6, 1 - 1e-6)
        preds = self.model.predict_proba(xs)[:, 1]
        return pd.Series(preds, index=probs.index)
