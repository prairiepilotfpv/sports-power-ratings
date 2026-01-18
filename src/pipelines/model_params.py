"""Helpers for resolving model parameter overrides."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.repository import (
    get_active_ensemble_market_weights,
    get_active_ensemble_market_weights_source,
    get_active_model_market_params,
    get_active_model_market_params_source,
    has_model_market_active_params,
    has_model_market_tuning_runs,
    legacy_tuned_params_exist,
    load_best_ensemble_market_tuning_weights_by_optimized_metric,
    load_best_model_market_tuning_params_by_optimized_metric,
    load_ensemble_market_tuning_run_by_run_id,
    load_model_market_tuning_run_by_run_id,
    set_active_model_market_params,
)
from markets.base import Market
from models.registry import normalize_model_name
from pipelines.market_tuning import _resolve_market_metric

logger = logging.getLogger(__name__)
LEGACY_WARNING_EMITTED: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class ModelParamsResolution:
    """Resolved model parameters along with their provenance metadata."""

    params: dict[str, Any] | None
    params_source: str
    tuned_metric_used: str | None
    source_run_id: str | None = None
    market: str = Market.ML.name


_MARKET_NAMES = {market.name for market in Market}


def _normalize_market(market: str | Market | None) -> str:
    if isinstance(market, Market):
        return market.name
    if market is None:
        return Market.ML.name
    normalized = str(market).strip().upper()
    if not normalized:
        return Market.ML.name
    if normalized not in _MARKET_NAMES:
        raise ValueError(f"Unsupported market: {market}")
    return normalized


def _metric_display_from_optimized(metric_optimized: str | None) -> str | None:
    if metric_optimized is None:
        return None
    if metric_optimized.startswith("backtest_"):
        return metric_optimized.replace("backtest_", "", 1)
    return metric_optimized


def _log_db_params_usage(
    model_name: str,
    market_name: str,
    params_source: str,
    tuned_metric_used: str | None,
) -> None:
    metric_label = tuned_metric_used or "unknown"
    logger.info(
        "Using tuned params from DB (market=%s metric=%s source=%s) for model=%s",
        market_name,
        metric_label,
        params_source,
        model_name,
    )


def _warn_if_only_legacy_data(
    db_path: str | Path,
    sport: str,
    season: str,
) -> None:
    key = (sport, season)
    if key in LEGACY_WARNING_EMITTED:
        return
    if has_model_market_active_params(db_path, sport=sport, season=season):
        return
    if has_model_market_tuning_runs(db_path, sport=sport, season=season):
        return
    if not legacy_tuned_params_exist(db_path, sport=sport, season=season):
        return
    logger.warning(
        "No per-market tuning rows found; using defaults. Re-run tune to populate market params."
    )
    LEGACY_WARNING_EMITTED.add(key)


def resolve_model_params(
    model: str,
    *,
    params: dict[str, Any] | None = None,
    params_file: str | Path | None = None,
    db_path: str | Path | None = None,
    sport: str | None = None,
    season: str | None = None,
    tuned_metric: str | None = None,
    market: str | Market | None = None,
) -> dict[str, Any] | None:
    """Resolve model parameters from CLI inputs or persisted tuning rows."""
    return resolve_model_market_params_with_metadata(
        model,
        params=params,
        params_file=params_file,
        db_path=db_path,
        sport=sport,
        season=season,
        tuned_metric=tuned_metric,
        market=market,
    ).params


def resolve_model_market_params_with_metadata(
    model: str,
    *,
    params: dict[str, Any] | None = None,
    params_file: str | Path | None = None,
    db_path: str | Path | None = None,
    sport: str | None = None,
    season: str | None = None,
    tuned_metric: str | None = None,
    market: str | Market | None = None,
) -> ModelParamsResolution:
    """Resolve parameters scoped to a single market and return richer metadata."""
    if params and params_file:
        raise ValueError("Provide either params or params_file, not both.")
    market_name = _normalize_market(market)
    if params is not None:
        return ModelParamsResolution(
            params=params,
            params_source="cli",
            tuned_metric_used=None,
            source_run_id=None,
            market=market_name,
        )
    if params_file is not None:
        payload = _load_params_file(params_file)
        if payload is None:
            return ModelParamsResolution(
                params=None,
                params_source="file",
                tuned_metric_used=None,
                source_run_id=None,
                market=market_name,
            )
        model_name = normalize_model_name(model)
        if isinstance(payload, dict) and model_name in payload:
            scoped = payload.get(model_name)
            if isinstance(scoped, dict):
                return ModelParamsResolution(
                    params=scoped,
                    params_source="file",
                    tuned_metric_used=None,
                    source_run_id=None,
                    market=market_name,
                )
        if isinstance(payload, dict):
            return ModelParamsResolution(
                params=payload,
                params_source="file",
                tuned_metric_used=None,
                source_run_id=None,
                market=market_name,
            )
        raise ValueError("Model params file must contain a JSON object.")
    if tuned_metric is not None:
        logger.debug("Ignoring tuned_metric=%s for market=%s resolution", tuned_metric, market_name)
    return _resolve_model_params_from_db(
        model=model,
        db_path=db_path,
        sport=sport,
        season=season,
        market=market_name,
    )


def resolve_model_params_with_metadata(
    model: str,
    *,
    params: dict[str, Any] | None = None,
    params_file: str | Path | None = None,
    db_path: str | Path | None = None,
    sport: str | None = None,
    season: str | None = None,
    tuned_metric: str | None = None,
    market: str | Market | None = None,
) -> ModelParamsResolution:
    return resolve_model_market_params_with_metadata(
        model,
        params=params,
        params_file=params_file,
        db_path=db_path,
        sport=sport,
        season=season,
        tuned_metric=tuned_metric,
        market=market,
    )


def _resolve_model_params_from_db(
    *,
    model: str,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    market: str,
) -> ModelParamsResolution:
    market_name = _normalize_market(market)
    if db_path is None or sport is None or season is None:
        return ModelParamsResolution(
            params=None,
            params_source="default",
            tuned_metric_used=None,
            source_run_id=None,
            market=market_name,
        )
    model_name = normalize_model_name(model)
    resolved = resolve_active_model_market_params(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model_name,
        market=market_name,
    )
    return ModelParamsResolution(
        params=resolved.params,
        params_source=resolved.params_source,
        tuned_metric_used=resolved.tuned_metric_used,
        source_run_id=resolved.source_run_id,
        market=market_name,
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
    if db_path is None or sport is None or season is None:
        return ActiveParamsResolution(params=None, params_source="default", tuned_metric_used=None, source_run_id=None)
    model_name = normalize_model_name(model)
    market_name = _normalize_market(market)

    active = get_active_model_market_params(
        db_path, sport=sport, season=season, model=model_name, market=market_name
    )
    if active is not None:
        source_run_id = get_active_model_market_params_source(
            db_path, sport=sport, season=season, model=model_name, market=market_name
        )
        tuned_metric = None
        if source_run_id:
            _, _, metric_optimized = load_model_market_tuning_run_by_run_id(
                db_path, run_id=source_run_id
            )
            tuned_metric = _metric_display_from_optimized(metric_optimized)
        _log_db_params_usage(model_name, market_name, "db_market_active", tuned_metric)
        return ActiveParamsResolution(
            params=active,
            params_source="db_market_active",
            tuned_metric_used=tuned_metric,
            source_run_id=source_run_id,
        )

    metric_name, metric_optimized = _resolve_market_metric(market_name, None)
    params, run_id = load_best_model_market_tuning_params_by_optimized_metric(
        db_path,
        sport=sport,
        season=season,
        model=model_name,
        market=market_name,
        metric_optimized=metric_optimized,
    )
    if params is not None:
        _log_db_params_usage(model_name, market_name, "db_market_best_run", metric_name)
        return ActiveParamsResolution(
            params=params,
            params_source="db_market_best_run",
            tuned_metric_used=metric_name,
            source_run_id=run_id,
        )

    _warn_if_only_legacy_data(db_path, sport, season)
    return ActiveParamsResolution(params=None, params_source="default", tuned_metric_used=None, source_run_id=None)


def resolve_active_ensemble_weights(
    *,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    market: str,
    ensemble_id: str,
) -> ActiveParamsResolution:
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
        db_path,
        sport=sport,
        season=season,
        market=market,
        ensemble_id=ensemble_id,
        metric_optimized=metric_optimized,
    )
    if weights is not None:
        logger.info(
            "Using tuned ensemble weights from best run (market=%s metric=%s ensemble=%s run_id=%s)",
            market,
            metric_name,
            ensemble_id,
            run_id,
        )
        return ActiveParamsResolution(
            params=weights,
            params_source="db_ensemble_best_run",
            tuned_metric_used=metric_name,
            source_run_id=run_id,
        )

    return ActiveParamsResolution(params=None, params_source="default", tuned_metric_used=None, source_run_id=None)


def bootstrap_market_active_params(
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    models: list[str],
    include_ml: bool = True,
) -> dict[str, list[tuple[str, str, str | None]] | dict[str, int]]:
    summary: dict[str, list[tuple[str, str, str | None]]] = {
        "created_from_best_run": [],
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

