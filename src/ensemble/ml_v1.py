"""ML ensemble implementation: weighted-average of model probabilities."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .io import load_ml_weights
from .base import BaseEnsemble, Market


class MLWeightedAverageEnsemble:
    """Combine multiple model ML probabilities using weighted average.

    Weights are loaded from `outputs/ensembles/<sport>/<season>/ensemble_ml_v1.json`.
    If missing, equal weights are used and re-normalized for the available models.
    """

    def __init__(self, sport: str, season: str, ensemble_id: str = "ensemble_ml_v1") -> None:
        self.sport = sport
        self.season = season
        self._ensemble_id = ensemble_id
        self._weights = load_ml_weights(sport, season, ensemble_id) or {}

    @property
    def ensemble_id(self) -> str:
        return self._ensemble_id

    @property
    def market(self) -> Market:
        return Market.ML

    def combine(self, forecast_df: pd.DataFrame) -> tuple[float | None, str]:
        """Combine forecasts for a single game.

        `forecast_df` must contain columns: `model_name`, `p_home_win`.
        Returns (p_home_win_raw, components_json).
        """
        if forecast_df is None or forecast_df.empty:
            return None, "[]"

        rows = list(forecast_df.itertuples(index=False))
        models = []
        probs = []
        for r in rows:
            # Support both attribute and dict-style access
            try:
                model = getattr(r, "model_name")
                prob = getattr(r, "p_home_win")
            except Exception:
                model = r[0]
                prob = r[1]
            models.append(str(model))
            try:
                probs.append(float(prob))
            except Exception:
                probs.append(float("nan"))

        # Build weights (default 1.0 when not specified)
        raw_weights: list[float] = [float(self._weights.get(m, 1.0)) for m in models]
        total = sum(raw_weights)
        if total <= 0 or any(pd.isna(w) for w in raw_weights):
            norm_weights = [1.0 / len(raw_weights)] * len(raw_weights)
        else:
            norm_weights = [w / total for w in raw_weights]

        components: list[dict[str, Any]] = []
        combined = 0.0
        valid_any = False
        for m, p, w in zip(models, probs, norm_weights):
            comp = {"model": m, "prob": None if pd.isna(p) else float(p), "weight": float(w)}
            components.append(comp)
            if comp["prob"] is not None:
                combined += comp["prob"] * w
                valid_any = True

        if not valid_any:
            return None, json.dumps(components, sort_keys=True)

        return float(combined), json.dumps(components, sort_keys=True)
