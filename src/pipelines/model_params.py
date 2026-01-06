"""Helpers for resolving model parameter overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.registry import normalize_model_name
from data.repository import (
    load_active_tuned_metric,
    load_active_tuned_params,
    load_tuned_params_for_metric,
)


def resolve_model_params(
    model: str,
    *,
    params: dict[str, Any] | None = None,
    params_file: str | Path | None = None,
    db_path: str | Path | None = None,
    sport: str | None = None,
    season: str | None = None,
    tuned_metric: str | None = None,
) -> dict[str, Any] | None:
    """Resolve model parameters from a JSON blob or file."""
    if params and params_file:
        raise ValueError("Provide either params or params_file, not both.")
    if params is not None:
        return params
    if params_file is None:
        resolved = _load_tuned_params(
            model,
            db_path=db_path,
            sport=sport,
            season=season,
            tuned_metric=tuned_metric,
        )
        return resolved
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


def _load_tuned_params(
    model: str,
    *,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    tuned_metric: str | None,
) -> dict[str, Any] | None:
    if db_path is None or sport is None or season is None:
        return None
    model_name = normalize_model_name(model)
    if tuned_metric:
        tuned_metric = tuned_metric.strip().lower()
        params = load_tuned_params_for_metric(
            db_path,
            sport=sport,
            season=season,
            model=model_name,
            metric=tuned_metric,
        )
        if params is not None:
            print(
                "Using tuned params from DB "
                f"(metric={tuned_metric}) for model={model_name}"
            )
        return params
    params = load_active_tuned_params(
        db_path,
        sport=sport,
        season=season,
        model=model_name,
    )
    if params is None:
        return None
    metric = load_active_tuned_metric(
        db_path,
        sport=sport,
        season=season,
        model=model_name,
    )
    metric_label = metric or "unknown"
    print(
        "Using tuned params from DB "
        f"(metric={metric_label}) for model={model_name}"
    )
    return params


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
