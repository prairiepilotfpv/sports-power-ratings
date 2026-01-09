"""Base calibrator interface and helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import joblib


class BaseCalibrator(ABC):
    """Abstract base calibrator.

    Expected fit input: DataFrame with columns `p_home_win` (raw probs) and
    `home_win` (0/1 outcome). Implementations must be deterministic.
    """

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Fit calibrator from a DataFrame with `p_home_win` and `home_win`."""

    @abstractmethod
    def transform(self, probs: pd.Series) -> pd.Series:
        """Transform raw probabilities into calibrated probabilities."""

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(p))

    @classmethod
    def load(cls, path: str | Path) -> "BaseCalibrator":
        obj = joblib.load(str(path))
        if not isinstance(obj, BaseCalibrator):
            raise TypeError("Loaded object is not a BaseCalibrator")
        return obj
