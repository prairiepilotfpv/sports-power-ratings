"""Minimal IO helpers for ensemble configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_ml_weights(sport: str, season: str, ensemble_id: str = "ensemble_ml_v1") -> dict[str, float] | None:
    """Load ML ensemble weights from outputs/ensembles/<sport>/<season>/<ensemble_id>.json.

    File format: {"model_name": weight, ...}
    If missing, return None to signal equal-weight fallback.
    """
    path = Path("outputs") / "ensembles" / sport / season / f"{ensemble_id}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        weights: dict[str, float] = {}
        for k, v in data.items():
            try:
                weights[str(k)] = float(v)
            except Exception:
                continue
        if not weights:
            return None
        return weights
    except Exception:
        return None
