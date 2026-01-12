"""Total ensemble implementation: weighted-average of model total forecasts."""

from __future__ import annotations

import json
from math import sqrt
from typing import Any

import pandas as pd

from .io import load_market_weights
from markets.base import Market


class TotalWeightedAverageEnsemble:
    """Combine multiple model total forecasts using weighted average."""

    def __init__(self, sport: str, season: str, ensemble_id: str = "ensemble_total_v1") -> None:
        self.sport = sport
        self.season = season
        self._ensemble_id = ensemble_id
        self._weights = load_market_weights(sport, season, Market.TOTAL.name, ensemble_id) or {}

    @property
    def ensemble_id(self) -> str:
        return self._ensemble_id

    @property
    def market(self) -> Market:
        return Market.TOTAL

    def combine(
        self, game_rows: pd.DataFrame
    ) -> tuple[float | None, float | None, str]:
        """Combine forecasts for a single game.

        `game_rows` must contain columns: `model_name`, `total_mean`/`total`, `total_sd`.
        Returns (total_mean_raw, total_sd_raw, components_json).
        """
        if game_rows is None or game_rows.empty:
            return None, None, "[]"

        rows = list(game_rows.itertuples(index=False))
        models: list[str] = []
        totals: list[float] = []
        sds: list[float] = []
        for r in rows:
            try:
                model = getattr(r, "model_name")
                total = getattr(r, "total_mean")
                if total is None:
                    total = getattr(r, "total")
                sd = getattr(r, "total_sd")
            except Exception:
                model = r[0]
                total = r[1] if len(r) > 1 else None
                sd = r[2] if len(r) > 2 else None
            models.append(str(model))
            try:
                totals.append(float(total))
            except Exception:
                totals.append(float("nan"))
            try:
                sds.append(float(sd))
            except Exception:
                sds.append(float("nan"))

        raw_weights: list[float] = [float(self._weights.get(m, 1.0)) for m in models]
        is_valid_total = [not pd.isna(t) for t in totals]
        total_valid = sum(w for w, v in zip(raw_weights, is_valid_total) if v)

        components: list[dict[str, Any]] = []
        if total_valid <= 0:
            for m, mean, sd in zip(models, totals, sds):
                comp = {
                    "model": m,
                    "total_mean": None if pd.isna(mean) else float(mean),
                    "total_sd": None if pd.isna(sd) else float(sd),
                    "w": 0.0,
                }
                components.append(comp)
            return None, None, json.dumps(components, sort_keys=True)

        combined_mean = 0.0
        for m, mean, sd, w in zip(models, totals, sds, raw_weights):
            if pd.isna(mean):
                adj_w = 0.0
            else:
                adj_w = float(w) / float(total_valid)
            comp = {
                "model": m,
                "total_mean": None if pd.isna(mean) else float(mean),
                "total_sd": None if pd.isna(sd) else float(sd),
                "w": float(adj_w),
            }
            components.append(comp)
            if comp["total_mean"] is not None:
                combined_mean += comp["total_mean"] * comp["w"]

        sd_weights = []
        sd_values = []
        for comp in components:
            if comp["total_sd"] is None:
                continue
            sd_weights.append(comp["w"])
            sd_values.append(comp["total_sd"])

        combined_sd = None
        if sd_weights:
            total_sd_weight = sum(sd_weights)
            if total_sd_weight > 0:
                normalized = [w / total_sd_weight for w in sd_weights]
                combined_sd = sqrt(sum(w * (sd ** 2) for w, sd in zip(normalized, sd_values)))

        return float(combined_mean), combined_sd, json.dumps(components, sort_keys=True)
