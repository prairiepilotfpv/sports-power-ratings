"""Minimal IO helpers for ensemble configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from markets.registry import get_market_spec


def load_ml_weights(
    sport: str, season: str, ensemble_id: str = "ensemble_ml_v1", market: str | None = "ML"
) -> dict[str, float] | None:
    """Load ML ensemble weights from the MarketSpec ensemble path.

    File format: {"model_name": weight, ...}
    If missing, return None to signal equal-weight fallback.
    """
    legacy_path = Path("outputs") / "ensembles" / sport / season / f"{ensemble_id}.json"
    try:
        spec = get_market_spec(market or "ML")
        path = spec.ensemble_weights_path(sport, season, ensemble_id)
    except Exception:
        path = legacy_path

    if not path.exists():
        if legacy_path.exists():
            path = legacy_path
        else:
            return None
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        raw_weights = data.get("weights") if isinstance(data.get("weights"), dict) else data
        weights: dict[str, float] = {}
        for k, v in raw_weights.items():
            try:
                weights[str(k)] = float(v)
            except Exception:
                continue
        if not weights:
            return None
        return weights
    except Exception:
        return None
