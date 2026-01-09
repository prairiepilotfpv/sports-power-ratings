"""Registry for calibrators keyed by (sport, model_name, market)."""
from __future__ import annotations

from typing import Dict, Tuple

_REGISTRY: Dict[Tuple[str, str, str], object] = {}


def register_calibrator(sport: str, model_name: str, market: str, cls_or_factory) -> None:
    key = (sport, model_name, market)
    _REGISTRY[key] = cls_or_factory


def get_calibrator(sport: str, model_name: str, market: str = "ML"):
    key = (sport, model_name, market)
    return _REGISTRY.get(key)


def list_calibrators() -> Dict[Tuple[str, str, str], object]:
    return dict(_REGISTRY)
