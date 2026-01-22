"""Helpers for resolving model parameter overrides."""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.repository import (
    get_active_ensemble_market_weights,
    get_active_ensemble_market_weights_source,
    get_active_model_market_params,
    get_active_model_market_params_source,
    load_model_market_active_params,
    has_model_market_active_params,
    has_model_market_tuning_runs,
    has_nonempty_model_market_tuning_params,
    legacy_tuned_params_exist,
    load_best_ensemble_market_tuning_weights_by_optimized_metric,
    load_best_model_market_tuning_params_by_optimized_metric,
    load_ensemble_market_tuning_run_by_run_id,
    load_model_market_tuning_run_by_run_id,
    load_model_tuned_params,
    load_best_model_market_tuning_run,
    model_market_tuning_run_exists,
    set_active_model_market_params,
    upsert_model_tuned_params,
)
from markets.base import Market
from models.registry import normalize_model_name
from pipelines.market_utils import _resolve_market_metric

logger = logging.getLogger(__name__)
LEGACY_WARNING_EMITTED: set[tuple[str, str]] = set()
MISSING_ACTIVE_WARNING_EMITTED: set[tuple[str, str, str, str]] = set()


@dataclass(frozen=True)
class ModelParamsResolution:
    """Resolved model parameters along with their provenance metadata."""

    params: dict[str, Any] | None
    params_source: str
    tuned_metric_used: str | None
    source_run_id: str | None = None
    market: str = Market.ML.name
    params_source_label: str | None = None
    metric_optimized: str | None = None
    best_score: float | None = None
    params_fingerprint: str | None = None
    params_nonempty: bool | None = None


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


def _fingerprint_params(params: dict[str, Any] | None) -> str:
    payload = params if isinstance(params, dict) else {}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _coerce_best_score(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        if numeric != numeric:  # NaN
            return None
        return numeric
    except Exception:
        return None


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


def activate_best_params(
    *,
    db_path: str | Path,
    sport: str,
    season: str,
    model: str,
    market: str,
    run_id: str,
    best_params: dict[str, Any],
    best_score: float | None,
    metric_optimized: str | None,
) -> None:
    """Persist the best tuned params as active (and legacy mirror)."""
    if not isinstance(best_params, dict):
        raise ValueError("activate_best_params expects best_params as a dict.")
    model_name = normalize_model_name(model)
    market_name = _normalize_market(market)
    set_active_model_market_params(
        db_path,
        sport=sport,
        season=season,
        model=model_name,
        market=market_name,
        params=best_params,
        source_run_id=run_id,
        params_source="tuned",
        metric_optimized=metric_optimized,
        best_score=best_score,
    )

    metric_display = _metric_display_from_optimized(metric_optimized)
    if metric_display:
        try:
            upsert_model_tuned_params(
                db_path,
                sport=sport,
                season=season,
                model=model_name,
                metric=metric_display,
                run_id=run_id,
                params=best_params,
                best_score=best_score,
            )
        except Exception:
            logger.exception("Failed to persist legacy tuned params mirror; continuing.")


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
        fingerprint = _fingerprint_params(params)
        return ModelParamsResolution(
            params=params,
            params_source="cli",
            tuned_metric_used=None,
            source_run_id=None,
            market=market_name,
            params_source_label="cli",
            metric_optimized=None,
            best_score=None,
            params_fingerprint=fingerprint,
            params_nonempty=bool(params),
        )
    if params_file is not None:
        payload = _load_params_file(params_file)
        if payload is None:
            fingerprint = _fingerprint_params(None)
            return ModelParamsResolution(
                params=None,
                params_source="file",
                tuned_metric_used=None,
                source_run_id=None,
                market=market_name,
                params_source_label="file",
                metric_optimized=None,
                best_score=None,
                params_fingerprint=fingerprint,
                params_nonempty=False,
            )
        model_name = normalize_model_name(model)
        if isinstance(payload, dict) and model_name in payload:
            scoped = payload.get(model_name)
            if isinstance(scoped, dict):
                fingerprint = _fingerprint_params(scoped)
                return ModelParamsResolution(
                    params=scoped,
                    params_source="file",
                    tuned_metric_used=None,
                    source_run_id=None,
                    market=market_name,
                    params_source_label="file",
                    metric_optimized=None,
                    best_score=None,
                    params_fingerprint=fingerprint,
                    params_nonempty=bool(scoped),
                )
        if isinstance(payload, dict):
            fingerprint = _fingerprint_params(payload)
            return ModelParamsResolution(
                params=payload,
                params_source="file",
                tuned_metric_used=None,
                source_run_id=None,
                market=market_name,
                params_source_label="file",
                metric_optimized=None,
                best_score=None,
                params_fingerprint=fingerprint,
                params_nonempty=bool(payload),
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
    effective = resolve_effective_params(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        market=market_name,
    )
    tuned_metric_used = _metric_display_from_optimized(effective.metric_optimized)
    return ModelParamsResolution(
        params=effective.params,
        params_source=effective.params_source_label,
        tuned_metric_used=tuned_metric_used,
        source_run_id=effective.source_run_id,
        market=market_name,
        params_source_label=effective.params_source_label,
        metric_optimized=effective.metric_optimized,
        best_score=effective.best_score,
        params_fingerprint=effective.params_fingerprint,
        params_nonempty=effective.params_nonempty,
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


@dataclass(frozen=True)
class EffectiveParamsResolution:
    params: dict[str, Any] | None
    params_source_label: str
    source_run_id: str | None
    metric_optimized: str | None
    best_score: float | None
    params_fingerprint: str
    params_nonempty: bool


def resolve_active_model_market_params(
    *,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    model: str,
    market: str,
) -> ActiveParamsResolution:
    effective = resolve_effective_params(
        db_path=db_path,
        sport=sport,
        season=season,
        model=model,
        market=market,
    )
    tuned_metric = _metric_display_from_optimized(effective.metric_optimized)
    return ActiveParamsResolution(
        params=effective.params,
        params_source=effective.params_source_label,
        tuned_metric_used=tuned_metric,
        source_run_id=effective.source_run_id,
    )


def resolve_effective_params(
    *,
    db_path: str | Path | None,
    sport: str | None,
    season: str | None,
    model: str,
    market: str,
) -> EffectiveParamsResolution:
    model_name = normalize_model_name(model)
    market_name = _normalize_market(market)
    default_metric_name, default_metric_optimized = _resolve_market_metric(market_name, None)
    fingerprint = _fingerprint_params(None)
    if db_path is None or sport is None or season is None:
        return EffectiveParamsResolution(
            params=None,
            params_source_label="missing_active",
            source_run_id=None,
            metric_optimized=default_metric_optimized,
            best_score=None,
            params_fingerprint=fingerprint,
            params_nonempty=False,
        )

    active = load_model_market_active_params(
        db_path,
        sport=sport,
        season=season,
        model=model_name,
        market=market_name,
    )

    params: dict[str, Any] | None = None
    source_run_id: str | None = None
    metric_optimized: str | None = default_metric_optimized
    best_score: float | None = None
    params_source_label = "missing_active"

    if active is not None:
        params = active.get("params") if isinstance(active, dict) else None
        source_run_id = active.get("source_run_id") if isinstance(active, dict) else None
        metric_optimized = active.get("metric_optimized") or metric_optimized
        best_score = _coerce_best_score(active.get("best_score") if isinstance(active, dict) else None)
        params_source = (active.get("params_source") or "") if isinstance(active, dict) else ""
        params_source_label = "legacy_active"

        if (source_run_id == "default" or params_source == "default") and not params:
            params_source_label = "default_active"
        elif source_run_id and source_run_id.startswith("model_tuned_params"):
            legacy_metric = None
            if ":" in source_run_id:
                legacy_metric = source_run_id.split(":", 1)[1]
            legacy_params, legacy_best_score, legacy_run_id = load_model_tuned_params(
                db_path,
                sport=sport,
                season=season,
                model=model_name,
                metric=legacy_metric or default_metric_name,
            )
            if legacy_params is not None and not params:
                params = legacy_params
            if legacy_best_score is not None and best_score is None:
                best_score = _coerce_best_score(legacy_best_score)
            if legacy_run_id and not source_run_id:
                source_run_id = legacy_run_id
            metric_optimized = metric_optimized or (f"backtest_{legacy_metric}" if legacy_metric else None)
            params_source_label = "legacy_active"
        else:
            _params_from_run, run_id, run_metric_opt, run_best_score = load_model_market_tuning_run_by_run_id(
                db_path, run_id=source_run_id
            )
            metric_optimized = run_metric_opt or metric_optimized
            run_best = _coerce_best_score(run_best_score)
            if run_id and run_best is not None:
                params_source_label = (
                    "tuned_active" if params_source == "tuned" else "db_market_active"
                )
                best_score = best_score if best_score is not None else run_best
            elif run_id:
                params_source_label = "legacy_active"
            elif params is not None:
                params_source_label = "legacy_active"
            else:
                params_source_label = "default_active"

    params_nonempty = bool(params)
    fingerprint = _fingerprint_params(params)

    if params_source_label in {"missing_active", "default_active"}:
        has_runs = model_market_tuning_run_exists(
            db_path,
            sport=sport,
            season=season,
            model=model_name,
            market=market_name,
        )
        has_nonempty_runs = has_nonempty_model_market_tuning_params(
            db_path,
            sport=sport,
            season=season,
            model=model_name,
            market=market_name,
        )
        warning_key = (sport, season, model_name, market_name)
        if has_nonempty_runs and warning_key not in MISSING_ACTIVE_WARNING_EMITTED:
            logger.warning(
                "Tuned params exist but are not active — run activation or check DB (model=%s market=%s)",
                model_name,
                market_name,
            )
            MISSING_ACTIVE_WARNING_EMITTED.add(warning_key)
        elif has_runs and not has_nonempty_runs and warning_key not in MISSING_ACTIVE_WARNING_EMITTED:
            logger.info(
                "Tuning ran but produced no improved params; defaults remain active (model=%s market=%s)",
                model_name,
                market_name,
            )
            MISSING_ACTIVE_WARNING_EMITTED.add(warning_key)
        if params_source_label == "missing_active" and has_runs:
            best_run = load_best_model_market_tuning_run(
                db_path,
                sport=sport,
                season=season,
                model=model_name,
                market=market_name,
            )
            if best_run and best_run.get("params"):
                params = best_run["params"]
                source_run_id = source_run_id or best_run.get("run_id")
                metric_optimized = best_run.get("metric_optimized") or metric_optimized
                best_score = best_score if best_score is not None else _coerce_best_score(
                    best_run.get("best_score")
                )
                params_source_label = "db_market_best_run"

    if params_source_label == "missing_active":
        _warn_if_only_legacy_data(db_path, sport, season)

    return EffectiveParamsResolution(
        params=params,
        params_source_label=params_source_label,
        source_run_id=source_run_id,
        metric_optimized=metric_optimized,
        best_score=best_score,
        params_fingerprint=fingerprint,
        params_nonempty=params_nonempty,
    )


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
                    params_source="tuned_bootstrap",
                    metric_optimized=metric_optimized,
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
                params_source="default",
                metric_optimized=metric_optimized,
            )
            summary["created_default"].append((model_name, market_name, "default"))

    summary["counts"] = {key: len(value) for key, value in summary.items() if key != "counts"}
    return summary

