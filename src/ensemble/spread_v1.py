"""Spread ensemble implementation: weighted-average of model margin forecasts."""

from __future__ import annotations

import json
import logging
import atexit
from math import sqrt
from typing import Any

import pandas as pd

from .io import load_market_weights
from markets.base import Market


logger = logging.getLogger(__name__)
_LOG_REGISTERED = False


def _register_log_hook() -> None:
    global _LOG_REGISTERED
    if _LOG_REGISTERED:
        return
    atexit.register(SpreadWeightedAverageEnsemble.log_between_var_usage)
    _LOG_REGISTERED = True


class SpreadWeightedAverageEnsemble:
    """Combine multiple model margin forecasts using weighted average."""

    _combine_calls: int = 0
    _between_var_applied_calls: int = 0

    def __init__(
        self,
        sport: str,
        season: str,
        ensemble_id: str = "ensemble_spread_v1",
        weights: dict[str, float] | None = None,
        include_between_model_variance: bool = True,
    ) -> None:
        self.sport = sport
        self.season = season
        self._ensemble_id = ensemble_id
        self._include_between_model_variance = include_between_model_variance
        self._weights = (
            weights
            if weights is not None
            else (load_market_weights(sport, season, Market.SPREAD.name, ensemble_id) or {})
        )
        _register_log_hook()

    @property
    def ensemble_id(self) -> str:
        return self._ensemble_id

    @property
    def market(self) -> Market:
        return Market.SPREAD

    def combine(
        self, game_rows: pd.DataFrame
    ) -> tuple[float | None, float | None, str]:
        """Combine forecasts for a single game.

        `game_rows` must contain columns: `model_name`, `margin_mean`, `margin_sd`.
        Returns (margin_mean_raw, margin_sd_raw, components_json).
        """
        if game_rows is None or game_rows.empty:
            return None, None, "[]"

        cls = type(self)
        cls._combine_calls += 1
        applied_between_var = False

        rows = list(game_rows.itertuples(index=False))
        models: list[str] = []
        margins: list[float] = []
        sds: list[float] = []
        for r in rows:
            try:
                model = getattr(r, "model_name")
                margin = getattr(r, "margin_mean")
                sd = getattr(r, "margin_sd")
            except Exception:
                model = r[0]
                margin = r[1]
                sd = r[2] if len(r) > 2 else None
            models.append(str(model))
            try:
                margins.append(float(margin))
            except Exception:
                margins.append(float("nan"))
            try:
                sds.append(float(sd))
            except Exception:
                sds.append(float("nan"))

        raw_weights: list[float] = [float(self._weights.get(m, 1.0)) for m in models]
        is_valid_margin = [not pd.isna(m) for m in margins]
        total_valid = sum(w for w, v in zip(raw_weights, is_valid_margin) if v)

        components: list[dict[str, Any]] = []
        if total_valid <= 0:
            for m, mean, sd in zip(models, margins, sds):
                comp = {
                    "model": m,
                    "margin_mean": None if pd.isna(mean) else float(mean),
                    "margin_sd": None if pd.isna(sd) else float(sd),
                    "w": 0.0,
                }
                components.append(comp)
            return None, None, json.dumps(components, sort_keys=True)

        combined_mean = 0.0
        for m, mean, sd, w in zip(models, margins, sds, raw_weights):
            if pd.isna(mean):
                adj_w = 0.0
            else:
                adj_w = float(w) / float(total_valid)
            comp = {
                "model": m,
                "margin_mean": None if pd.isna(mean) else float(mean),
                "margin_sd": None if pd.isna(sd) else float(sd),
                "w": float(adj_w),
            }
            components.append(comp)
            if comp["margin_mean"] is not None:
                combined_mean += comp["margin_mean"] * comp["w"]

        # Compute weighted RMS SD over components that reported sd values.
        sd_weights = []
        sd_values = []
        for comp in components:
            if comp["margin_sd"] is None:
                continue
            sd_weights.append(comp["w"])
            sd_values.append(comp["margin_sd"])

        combined_sd = None
        within_var = None
        if sd_weights:
            total_sd_weight = sum(sd_weights)
            if total_sd_weight > 0:
                normalized = [w / total_sd_weight for w in sd_weights]
                within_var = sum(w * (sd ** 2) for w, sd in zip(normalized, sd_values))
                combined_sd = sqrt(within_var)

        if within_var is not None and self._include_between_model_variance:
            var_set = [
                comp
                for comp in components
                if comp.get("margin_mean") is not None
                and comp.get("margin_sd") is not None
                and comp.get("w", 0.0) > 0.0
            ]
            if len(var_set) >= 2:
                weight_sum = sum(comp["w"] for comp in var_set)
                if weight_sum > 0:
                    normalized_weights = [comp["w"] / weight_sum for comp in var_set]
                    mean_for_var_set = sum(
                        w * comp["margin_mean"] for w, comp in zip(normalized_weights, var_set)
                    )
                    between_var = sum(
                        w * ((comp["margin_mean"] - mean_for_var_set) ** 2)
                        for w, comp in zip(normalized_weights, var_set)
                    )
                    combined_sd = sqrt(within_var + between_var)
                    applied_between_var = True

        if applied_between_var:
            cls._between_var_applied_calls += 1

        return float(combined_mean), combined_sd, json.dumps(components, sort_keys=True)

    @classmethod
    def log_between_var_usage(cls) -> None:
        """Log aggregated usage of between-model variance application."""
        if cls._combine_calls <= 0:
            return
        logger.info(
            "Spread ensemble between_var applied for %s/%s games (>=2 mean+sd components).",
            cls._between_var_applied_calls,
            cls._combine_calls,
        )
