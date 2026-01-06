"""Helpers for resolving model parameter overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.repository import (
    load_active_tuned_metric,
    load_active_tuned_params,
    load_tuned_params_for_metric,
)
from models.registry import normalize_model_name


@dataclass(frozen=True)
class ModelParamsResolution:
    """Resolved model parameters and their provenance."""

    params: dict[str, Any] | None
    params_source: str
    tuned_metric_used: str | None


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
    resolution = resolve_model_params_with_metadata(
        model,
        params=params,
        params_file=params_file,
        db_path=db_path,
        sport=sport,
        season=season,
        tuned_metric=tuned_metric,
    )
    return resolution.params


def resolve_model_params_with_metadata(
    model: str,
    *,
    params: dict[str, Any] | None = None,
    params_file: str | Path | None = None,
    db_path: str | Path | None = None,
    sport: str | None = None,
    season: str | None = None,
    tuned_metric: str | None = None,
) -> ModelParamsResolution:
    """Resolve model parameters and return provenance metadata."""
    if params and params_file:
        raise ValueError("Provide either params or params_file, not both.")
    if params is not None:
        return ModelParamsResolution(
            params=params,
            params_source="cli",
            tuned_metric_used=None,
        )
    if params_file is None:
        resolved = _load_tuned_params_with_metadata(
            model,
            db_path=db_path,
            sport=sport,
            season=season,
            tuned_metric=tuned_metric,
        )
        return resolved
    payload = _load_params_file(params_file)
    if payload is None:
        return ModelParamsResolution(
            params=None,
            params_source="file",
            tuned_metric_used=None,
        )
    model_name = normalize_model_name(model)
    if isinstance(payload, dict) and model_name in payload:
        scoped = payload.get(model_name)
        if isinstance(scoped, dict):
            return ModelParamsResolution(
                params=scoped,
                params_source="file",
                tuned_metric_used=None,
            )
    if isinstance(payload, dict):
        return ModelParamsResolution(
            params=payload,
            params_source="file",
            tuned_metric_used=None,
        )
    raise ValueError("Model params file must contain a JSON object.")


def _load_tuned_params_with_metadata(
    model: str,
    *,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    tuned_metric: str | None,
) -> ModelParamsResolution:
    if db_path is None or sport is None or season is None:
        return ModelParamsResolution(
            params=None,
            params_source="default",
            tuned_metric_used=None,
        )
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
            return ModelParamsResolution(
                params=params,
                params_source="db_metric",
                tuned_metric_used=tuned_metric,
            )
        return ModelParamsResolution(
            params=None,
            params_source="default",
            tuned_metric_used=None,
        )
    params = load_active_tuned_params(
        db_path,
        sport=sport,
        season=season,
        model=model_name,
    )
    if params is None:
        return ModelParamsResolution(
            params=None,
            params_source="default",
            tuned_metric_used=None,
        )
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
    return ModelParamsResolution(
        params=params,
        params_source="db_active",
        tuned_metric_used=metric,
    )


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
