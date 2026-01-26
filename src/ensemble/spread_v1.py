"""Spread ensemble implementation: weighted-average of model margin forecasts."""

from __future__ import annotations

import json
import logging
from math import sqrt
from typing import Any

import pandas as pd

from .io import load_market_weights
from .schema import create_component, sort_components
from markets.base import Market
from models.registry import normalize_model_name


logger = logging.getLogger(__name__)


class SpreadWeightedAverageEnsemble:
    """Combine multiple model margin forecasts using weighted average."""

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
            # Normalize model name for consistent lookups (Issue #10 fix)
            models.append(normalize_model_name(str(model)))
            try:
                margins.append(float(margin))
            except Exception:
                margins.append(float("nan"))
            try:
                sds.append(float(sd))
            except Exception:
                sds.append(float("nan"))

        # Build raw weights (default 0.0 when not specified - Issue #2 fix)
        raw_weights: list[float] = [float(self._weights.get(m, 0.0)) for m in models]

        # Warn about models with zero weight
        zero_weight_models = [m for m, w in zip(models, raw_weights) if w == 0.0]
        if zero_weight_models:
            logger.warning(
                "SPREAD ensemble: Models excluded (zero weight): %s. "
                "To include, add weights to config for sport=%s, season=%s",
                zero_weight_models, self.sport, self.season
            )

        is_valid_margin = [not pd.isna(m) for m in margins]
        total_valid = sum(w for w, v in zip(raw_weights, is_valid_margin) if v)

        components: list[dict[str, Any]] = []
        if total_valid <= 0:
            for m, mean, sd in zip(models, margins, sds):
                comp = create_component(
                    model=m,
                    weight=0.0,
                    value=None if pd.isna(mean) else float(mean),
                    uncertainty=None if pd.isna(sd) else float(sd)
                )
                components.append(comp)
            # Sort for deterministic ordering (Issue #11 fix)
            components = sort_components(components)
            return None, None, json.dumps(components, sort_keys=True)

        combined_mean = 0.0
        for m, mean, sd, w in zip(models, margins, sds, raw_weights):
            if pd.isna(mean):
                adj_w = 0.0
            else:
                adj_w = float(w) / float(total_valid)
            comp = create_component(
                model=m,
                weight=float(adj_w),
                value=None if pd.isna(mean) else float(mean),
                uncertainty=None if pd.isna(sd) else float(sd)
            )
            components.append(comp)
            if comp["value"] is not None:
                combined_mean += comp["value"] * comp["weight"]

        # Compute weighted RMS SD over components that reported sd values.
        sd_weights = []
        sd_values = []
        for comp in components:
            if comp["uncertainty"] is None:
                continue
            sd_weights.append(comp["weight"])
            sd_values.append(comp["uncertainty"])

        combined_sd = None
        within_var = None
        if sd_weights:
            total_sd_weight = sum(sd_weights)
            if total_sd_weight > 0:
                normalized = [w / total_sd_weight for w in sd_weights]
                within_var = sum(w * (sd ** 2) for w, sd in zip(normalized, sd_values))
                combined_sd = sqrt(within_var)

        # Between-model variance calculation (Issue #1 fix)
        if within_var is not None and self._include_between_model_variance:
            var_set = [
                comp
                for comp in components
                if comp.get("value") is not None
                and comp.get("uncertainty") is not None
                and comp.get("weight", 0.0) > 0.0
            ]
            if len(var_set) >= 2:
                # Issue #1 FIX: Use the ALREADY-NORMALIZED weights from components
                # Don't re-normalize! The weights in components already sum to 1.0
                weights_for_var = [comp["weight"] for comp in var_set]

                # Use the already-computed weighted mean (no need to recompute)
                mean_for_var_set = combined_mean

                # Compute between-model variance using the SAME weights as the mean
                between_var = sum(
                    w * ((comp["value"] - mean_for_var_set) ** 2)
                    for w, comp in zip(weights_for_var, var_set)
                )
                combined_sd = sqrt(within_var + between_var)

                logger.debug(
                    "SPREAD ensemble: Applied between-model variance (within=%.2f, between=%.2f, "
                    "total_sd=%.2f) for %d models with uncertainty",
                    within_var, between_var, combined_sd, len(var_set)
                )
            else:
                # Issue #8 fix: Warn when variance is skipped
                models_with_sd = [c["model"] for c in components if c.get("uncertainty") is not None and c.get("weight", 0.0) > 0.0]
                if len(models_with_sd) == 1:
                    logger.debug(
                        "SPREAD ensemble: Between-model variance skipped (only 1 model with SD: %s). "
                        "Need ≥2 models with both value and uncertainty.",
                        models_with_sd[0]
                    )

        # Sort for deterministic ordering (Issue #11 fix)
        components = sort_components(components)
        return float(combined_mean), combined_sd, json.dumps(components, sort_keys=True)

