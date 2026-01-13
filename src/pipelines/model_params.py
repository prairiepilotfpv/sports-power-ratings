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
from data.repository import (
    get_active_model_market_params,
    get_active_model_market_params_source,
    get_active_ensemble_market_weights,
    get_active_ensemble_market_weights_source,
    load_best_model_market_tuning_params_by_optimized_metric,
    load_model_market_tuning_run_by_run_id,
    load_best_ensemble_market_tuning_weights_by_optimized_metric,
    load_ensemble_market_tuning_run_by_run_id,
    set_active_model_market_params,
)
from pipelines.market_tuning import _resolve_market_metric
from markets.base import Market


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


@dataclass(frozen=True)
class ActiveParamsResolution:
    params: dict[str, Any] | None
    params_source: str
    tuned_metric_used: str | None
    source_run_id: str | None


def resolve_active_model_market_params(
    *,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    model: str,
    market: str,
) -> ActiveParamsResolution:
    """Resolve active params for a specific (model, market, sport, season).

    Resolution order:
      1) model_market_active_params table
      2) best model_market_tuning_runs for the market's optimized metric
      3) defaults (None)
    """
    if db_path is None or sport is None or season is None:
        return ActiveParamsResolution(params=None, params_source="default", tuned_metric_used=None, source_run_id=None)
    model_name = normalize_model_name(model)
    # 1) market-specific active params
    active = get_active_model_market_params(
        db_path, sport=sport, season=season, model=model_name, market=market
    )
    if active is not None:
        source = get_active_model_market_params_source(
            db_path, sport=sport, season=season, model=model_name, market=market
        )
        return ActiveParamsResolution(params=active, params_source="db_market_active", tuned_metric_used=None, source_run_id=source)

    # 2) attempt to auto-select the best tuned run for this market
    metric_name, metric_optimized = _resolve_market_metric(market, None)
    params, run_id = load_best_model_market_tuning_params_by_optimized_metric(
        db_path, sport=sport, season=season, model=model_name, market=market, metric_optimized=metric_optimized
    )
    if params is not None:
        print(f"Auto-selected tuned params from best run (metric={metric_name}) for model={model_name} market={market} run_id={run_id}")
        return ActiveParamsResolution(params=params, params_source="db_market_best_run", tuned_metric_used=metric_name, source_run_id=run_id)

    return ActiveParamsResolution(params=None, params_source="default", tuned_metric_used=None, source_run_id=None)


def resolve_active_ensemble_weights(
    *,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    market: str,
    ensemble_id: str,
) -> ActiveParamsResolution:
    """Resolve active ensemble weights for a market/ensemble.

    Resolution order:
      1) ensemble_market_active_weights
      2) best ensemble_market_tuning_runs for the market's optimized metric
      3) defaults (None)
    """
    if db_path is None or sport is None or season is None:
        return ActiveParamsResolution(params=None, params_source="default", tuned_metric_used=None, source_run_id=None)
    weights = get_active_ensemble_market_weights(
        db_path, sport=sport, season=season, market=market, ensemble_id=ensemble_id
    )
    if weights is not None:
        src = get_active_ensemble_market_weights_source(
            db_path, sport=sport, season=season, market=market, ensemble_id=ensemble_id
        )
        return ActiveParamsResolution(params=weights, params_source="db_ensemble_active", tuned_metric_used=None, source_run_id=src)

    metric_name, metric_optimized = _resolve_market_metric(market, None)
    weights, run_id = load_best_ensemble_market_tuning_weights_by_optimized_metric(
        db_path, sport=sport, season=season, market=market, ensemble_id=ensemble_id, metric_optimized=metric_optimized
    )
    if weights is not None:
        print(f"Auto-selected ensemble weights from best run (metric={metric_name}) for ensemble={ensemble_id} market={market} run_id={run_id}")
        return ActiveParamsResolution(params=weights, params_source="db_ensemble_best_run", tuned_metric_used=metric_name, source_run_id=run_id)

    return ActiveParamsResolution(params=None, params_source="default", tuned_metric_used=None, source_run_id=None)


def bootstrap_market_active_params(
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    models: list[str],
    include_ml: bool = True,
) -> dict:
    summary: dict[str, list[tuple[str, str, str | None]]] = {
        "created_from_best_run": [],
        "created_from_model_metric": [],
        "created_default": [],
        "skipped_existing": [],
    }

    markets = [Market.SPREAD, Market.TOTAL]
    if include_ml:
        markets.insert(0, Market.ML)

    for model in models:
        model_name = normalize_model_name(model)
        for market in markets:
            market_name = market.name
            existing = get_active_model_market_params(
                db_path,
                sport=sport,
                season=season,
                model=model_name,
                market=market_name,
            )
            if existing is not None:
                summary["skipped_existing"].append((model_name, market_name, None))
                continue

            _, metric_optimized = _resolve_market_metric(market_name, None)
            params, run_id = load_best_model_market_tuning_params_by_optimized_metric(
                db_path,
                sport=sport,
                season=season,
                model=model_name,
                market=market_name,
                metric_optimized=metric_optimized,
            )
            if params is not None:
                set_active_model_market_params(
                    db_path,
                    sport=sport,
                    season=season,
                    model=model_name,
                    market=market_name,
                    params=params,
                    source_run_id=run_id,
                )
                summary["created_from_best_run"].append((model_name, market_name, run_id))
                continue

            fallback_metric = None
            if market_name == "SPREAD":
                fallback_metric = "mae_margin"
            elif market_name == "TOTAL":
                fallback_metric = "mae_total"

            if fallback_metric:
                tuned = load_tuned_params_for_metric(
                    db_path,
                    sport=sport,
                    season=season,
                    model=model_name,
                    metric=fallback_metric,
                )
                if tuned is not None:
                    source = f"model_tuned_params:{fallback_metric}"
                    set_active_model_market_params(
                        db_path,
                        sport=sport,
                        season=season,
                        model=model_name,
                        market=market_name,
                        params=tuned,
                        source_run_id=source,
                    )
                    summary["created_from_model_metric"].append((model_name, market_name, source))
                    continue

            set_active_model_market_params(
                db_path,
                sport=sport,
                season=season,
                model=model_name,
                market=market_name,
                params={},
                source_run_id="default",
            )
            summary["created_default"].append((model_name, market_name, "default"))

    summary["counts"] = {key: len(value) for key, value in summary.items() if key != "counts"}
    return summary
