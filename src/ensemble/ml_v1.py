"""ML ensemble implementation: weighted-average of model probabilities."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .io import load_ml_weights
from .base import BaseEnsemble
from markets.base import Market


class MLWeightedAverageEnsemble:
    """Combine multiple model ML probabilities using weighted average.

    Weights are loaded from `outputs/ensembles/<sport>/<season>/ML/ensemble_ml_v1.json`
    (with legacy fallback to the pre-market path).
    If missing, equal weights are used and re-normalized for the available models.
    """

    def __init__(
        self,
        sport: str,
        season: str,
        ensemble_id: str = "ensemble_ml_v1",
        weights: dict[str, float] | None = None,
    ) -> None:
        self.sport = sport
        self.season = season
        self._ensemble_id = ensemble_id
        self._weights = (
            weights
            if weights is not None
            else (load_ml_weights(sport, season, ensemble_id) or {})
        )

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

        # Build raw weights (default 1.0 when not specified)
        raw_weights: list[float] = [float(self._weights.get(m, 1.0)) for m in models]

        # Identify which model probs are valid and renormalize weights over them.
        is_valid_prob = [not pd.isna(p) for p in probs]
        # Sum weights only for models with valid probs
        total_valid = sum(w for w, v in zip(raw_weights, is_valid_prob) if v)

        components: list[dict[str, Any]] = []
        combined = 0.0
        valid_any = False

        if total_valid <= 0:
            # No valid probabilities or no positive weight to distribute.
            # Return components JSON (weights set to 0 for invalid/unused) and None for combined.
            for m, p, w in zip(models, probs, raw_weights):
                comp = {"model": m, "prob": None if pd.isna(p) else float(p), "weight": 0.0}
                components.append(comp)
            return None, json.dumps(components, sort_keys=True)

        # Compute adjusted weights for valid models; invalid models get weight 0.
        for m, p, w in zip(models, probs, raw_weights):
            if pd.isna(p):
                adj_w = 0.0
            else:
                adj_w = float(w) / float(total_valid)
            comp = {"model": m, "prob": None if pd.isna(p) else float(p), "weight": float(adj_w)}
            components.append(comp)
            if comp["prob"] is not None:
                combined += comp["prob"] * comp["weight"]
                valid_any = True

        if not valid_any:
            return None, json.dumps(components, sort_keys=True)

        return float(combined), json.dumps(components, sort_keys=True)
