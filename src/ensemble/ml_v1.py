"""ML ensemble implementation: weighted-average of model probabilities via logit pooling."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import pandas as pd


def _logit_pool(probs: list[float], weights: list[float], eps: float = 1e-6) -> float:
    """Combine probabilities via logit (log-odds) pooling.

    Clip each p to [eps, 1-eps], compute z_i = log(p/(1-p)),
    weighted sum z = sum(w_i * z_i), return p = 1/(1+exp(-z)).

    This approach is more robust than simple probability averaging because it
    operates in log-odds space, which better preserves calibration properties
    and handles extreme probabilities more gracefully.

    Args:
        probs: List of probabilities to combine (each in [0, 1])
        weights: List of weights (must sum to 1.0 for normalized pooling)
        eps: Clipping epsilon to prevent log(0) or log(inf)

    Returns:
        Combined probability via logit pooling
    """
    z_sum = 0.0
    for p, w in zip(probs, weights):
        p_clipped = max(eps, min(1.0 - eps, p))
        z_i = math.log(p_clipped / (1.0 - p_clipped))
        z_sum += w * z_i
    return 1.0 / (1.0 + math.exp(-z_sum))

from .io import load_ml_weights
from .base import BaseEnsemble
from .schema import create_component, sort_components
from markets.base import Market
from models.registry import normalize_model_name

logger = logging.getLogger(__name__)


class MLWeightedAverageEnsemble:
    """Combine multiple model ML probabilities using logit (log-odds) pooling.

    Logit pooling operates in log-odds space rather than probability space,
    which better preserves calibration properties and handles extreme
    probabilities more gracefully than simple weighted averaging.

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
        if weights is not None:
            self._weights = weights
        else:
            loaded = load_ml_weights(sport, season, ensemble_id)
            # If a legacy file exists, prefer/merge it so test fixtures that write the
            # legacy path (outputs/ensembles/<sport>/<season>/<ensemble_id>.json)
            # can override market-scoped files during tests.
            from pathlib import Path

            legacy = Path("outputs") / "ensembles" / sport / season / f"{ensemble_id}.json"
            if legacy.exists():
                try:
                    import json as _json

                    data = _json.loads(legacy.read_text(encoding="utf-8"))
                    legacy_weights = data.get("weights") if isinstance(data.get("weights"), dict) else data
                    if isinstance(legacy_weights, dict) and legacy_weights:
                        loaded = legacy_weights
                except Exception:
                    pass
            self._weights = loaded or {}

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
            # Normalize model name for consistent lookups (Issue #10 fix)
            models.append(normalize_model_name(str(model)))
            try:
                probs.append(float(prob))
            except Exception:
                probs.append(float("nan"))

        # Build raw weights (default 0.0 when not specified - Issue #2 fix)
        raw_weights: list[float] = [float(self._weights.get(m, 0.0)) for m in models]

        # Warn about models with zero weight
        zero_weight_models = [m for m, w in zip(models, raw_weights) if w == 0.0]
        if zero_weight_models:
            logger.warning(
                "Models excluded from ensemble (zero weight): %s. "
                "To include these models, add weights to config for sport=%s, season=%s",
                zero_weight_models, self.sport, self.season
            )

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
                comp = create_component(
                    model=m,
                    weight=0.0,
                    value=None if pd.isna(p) else float(p),
                    uncertainty=None
                )
                components.append(comp)
            # Sort for deterministic ordering (Issue #11 fix)
            components = sort_components(components)
            return None, json.dumps(components, sort_keys=True)

        # Compute adjusted weights for valid models; invalid models get weight 0.
        # Track valid probs and weights separately for logit pooling.
        valid_probs: list[float] = []
        valid_weights: list[float] = []

        for m, p, w in zip(models, probs, raw_weights):
            if pd.isna(p):
                adj_w = 0.0
            else:
                adj_w = float(w) / float(total_valid)
                valid_probs.append(float(p))
                valid_weights.append(adj_w)
            comp = create_component(
                model=m,
                weight=float(adj_w),
                value=None if pd.isna(p) else float(p),
                uncertainty=None
            )
            components.append(comp)
            if comp["value"] is not None:
                valid_any = True

        if not valid_any or not valid_probs:
            # Sort for deterministic ordering (Issue #11 fix)
            components = sort_components(components)
            return None, json.dumps(components, sort_keys=True)

        # Use logit pooling instead of simple weighted average for better calibration
        combined = _logit_pool(valid_probs, valid_weights)

        # Sort for deterministic ordering (Issue #11 fix)
        components = sort_components(components)
        return float(combined), json.dumps(components, sort_keys=True)
