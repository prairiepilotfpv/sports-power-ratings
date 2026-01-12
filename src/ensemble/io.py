"""Minimal IO helpers for ensemble configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from markets.registry import get_market_spec


def _market_weight_candidates(
    sport: str, season: str, ensemble_id: str, market: str | None
) -> Iterable[Path]:
    legacy_path = Path("outputs") / "ensembles" / sport / season / f"{ensemble_id}.json"
    if market is None:
        return [legacy_path]
    try:
        spec = get_market_spec(market)
        spec_path = spec.ensemble_weights_path(sport, season, ensemble_id)
    except Exception:
        spec_path = None

    candidates = []
    if spec_path is not None:
        candidates.append(spec_path)
    # Also consider the raw market name to preserve legacy casing (ex: "SPREAD").
    market_token = str(market).strip()
    if market_token:
        candidates.append(
            Path("outputs") / "ensembles" / sport / season / market_token / f"{ensemble_id}.json"
        )
    # Always allow the legacy no-market path as a final fallback.
    candidates.append(legacy_path)
    # De-dupe while preserving order.
    seen = set()
    ordered: list[Path] = []
    for path in candidates:
        if path is None:
            continue
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def load_market_weights(
    sport: str, season: str, market: str | None, ensemble_id: str
) -> dict[str, float] | None:
    """Load ensemble weights from the market-scoped path with legacy fallback.

    File format: {"model_name": weight, ...}
    If missing, return None to signal equal-weight fallback.
    """
    path = None
    for candidate in _market_weight_candidates(sport, season, ensemble_id, market):
        if candidate.exists():
            path = candidate
            break
    if path is None or not path.exists():
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


def load_ml_weights(
    sport: str, season: str, ensemble_id: str = "ensemble_ml_v1", market: str | None = "ML"
) -> dict[str, float] | None:
    """Load ML ensemble weights from the MarketSpec ensemble path."""
    return load_market_weights(sport, season, market, ensemble_id)
