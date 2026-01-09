"""Evaluation helpers for calibration: brier, logloss, reliability table."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def brier_score(probs: pd.Series, outcomes: pd.Series) -> float:
    p = np.clip(probs.astype(float), 0.0, 1.0)
    y = outcomes.astype(float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs: pd.Series, outcomes: pd.Series) -> float:
    p = np.clip(probs.astype(float), 1e-12, 1 - 1e-12)
    y = outcomes.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(probs: pd.Series, outcomes: pd.Series, *, bin_width: float = 0.05, min_count: int = 20) -> pd.DataFrame:
    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    bins = np.digitize(np.clip(probs.astype(float), 0.0, 1.0), edges, right=True) - 1
    rows = []
    total = len(probs)
    for i in range(len(edges) - 1):
        mask = bins == i
        count = int(mask.sum())
        if count < min_count:
            continue
        avg_pred = float(np.mean(probs[mask]))
        avg_actual = float(np.mean(outcomes[mask]))
        rows.append({"bucket": f"{edges[i]:.2f}-{edges[i+1]:.2f}", "count": count, "avg_pred": avg_pred, "avg_actual": avg_actual})
    return pd.DataFrame(rows)
