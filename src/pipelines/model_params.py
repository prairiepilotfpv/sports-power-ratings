"""Helpers for resolving model parameter overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.registry import normalize_model_name


def resolve_model_params(
    model: str,
    *,
    params: dict[str, Any] | None = None,
    params_file: str | Path | None = None,
) -> dict[str, Any] | None:
    """Resolve model parameters from a JSON blob or file."""
    if params and params_file:
        raise ValueError("Provide either params or params_file, not both.")
    if params is not None:
        return params
    if params_file is None:
        return None
    payload = _load_params_file(params_file)
    if payload is None:
        return None
    model_name = normalize_model_name(model)
    if isinstance(payload, dict) and model_name in payload:
        scoped = payload.get(model_name)
        if isinstance(scoped, dict):
            return scoped
    if isinstance(payload, dict):
        return payload
    raise ValueError("Model params file must contain a JSON object.")


def _load_params_file(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model params file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("Model params file must contain a JSON object.")
    return data
