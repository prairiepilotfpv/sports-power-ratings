from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List

from ensemble.config import (
    DEFAULT_ENSEMBLE_IDS,
    DEFAULT_MARKET_METRICS,
    DEFAULT_MARKET_MODELS,
)


@dataclass
class InitResult:
    market: str
    path: Path
    created: bool
    skipped: bool


def _equal_weights(models: Iterable[str]) -> dict[str, float]:
    models = list(models)
    if not models:
        return {}
    weight = 1.0 / len(models)
    return {m: weight for m in models}


def init_default_ensemble_configs(
    *,
    sport: str,
    season: str,
    overwrite: bool = False,
) -> List[InitResult]:
    results: List[InitResult] = []
    for market, models in DEFAULT_MARKET_MODELS.items():
        weights = _equal_weights(models)
        cfg = {
            "sport": sport,
            "season": season,
            "market": market,
            "ensemble_id": DEFAULT_ENSEMBLE_IDS.get(market),
            "metric_slot": DEFAULT_MARKET_METRICS.get(market),
            "models": models,
            "weights": weights,
        }
        path = Path("outputs") / "ensembles" / sport / season / market / "default.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            results.append(InitResult(market=market, path=path, created=False, skipped=True))
            continue
        path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
        results.append(InitResult(market=market, path=path, created=True, skipped=False))
    return results
