"""Market-aware tuning wrappers for models and ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

from backtest.runner import load_games_df_from_csv, run_backtest
from data.repository import (
    save_ensemble_market_tuning_run,
    save_model_market_tuning_run,
    set_active_ensemble_market_weights,
)
from markets.base import Market
from models.registry import get_backtest_model, normalize_model_name
from pipelines.ensemble_tuning import tune_market_ensemble
from pipelines.market_utils import _metric_optimized_label, _resolve_market_metric
from pipelines.model_params import activate_best_params
from pipelines.tuning import run_tuning_pipeline

SUPPORTED_MARKETS = ("ML", "SPREAD", "TOTAL")


@dataclass(frozen=True)
class ModelMarketTuningResult:
    model: str
    market: str
    metric_optimized: str
    run_id: str
    best_score: float | None
    best_params: dict[str, Any]
    summary_metrics: dict[str, Any]
    output_dir: Path
    params_source: str
    activated: bool


@dataclass(frozen=True)
class ModelMarketTuningOutcome:
    market: str
    result: ModelMarketTuningResult | None
    error: str | None


def run_model_markets_tuning(
    *,
    sport: str,
    season: str,
    model: str,
    markets: Iterable[str] | None,
    start_date: str,
    end_date: str,
    window: str,
    rolling_days: int | None,
    rolling_games: int | None,
    csv_path: str | Path,
    output_dir: str | Path | None,
    grid_override: dict[str, Any] | None,
    db_path: str | Path,
    metric_overrides: dict[str, str] | None = None,
    allow_worse: bool = False,
    jobs: int = 1,
    activate_best: bool = False,
) -> list[ModelMarketTuningOutcome]:
    normalized_markets: list[str] = []
    seen: set[str] = set()
    raw_markets = markets if markets is not None else SUPPORTED_MARKETS
    for market in raw_markets:
        normalized = _normalize_market(market)
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_markets.append(normalized)
    if not normalized_markets:
        raise ValueError("No valid markets provided for tuning.")

    overrides: dict[str, str] = {}
    if metric_overrides:
        for market_name, metric in metric_overrides.items():
            overrides[_normalize_market(market_name)] = metric

    outcomes: list[ModelMarketTuningOutcome] = []
    for market in normalized_markets:
        metric_override = overrides.get(market)
        try:
            result = run_model_market_tuning(
                sport=sport,
                season=season,
                model=model,
                market=market,
                start_date=start_date,
                end_date=end_date,
                window=window,
                rolling_days=rolling_days,
                rolling_games=rolling_games,
                csv_path=csv_path,
                output_dir=output_dir,
                grid_override=grid_override,
                db_path=db_path,
                metric_override=metric_override,
                allow_worse=allow_worse,
                jobs=jobs,
                activate_best=activate_best,
            )
            outcomes.append(ModelMarketTuningOutcome(market=market, result=result, error=None))
        except Exception as exc:  # pragma: no cover - errors bubble to CLI
            outcomes.append(
                ModelMarketTuningOutcome(
                    market=market,
                    result=None,
                    error=str(exc),
                )
            )
    return outcomes


def run_model_market_tuning(
    *,
    sport: str,
    season: str,
    model: str,
    market: str,
    start_date: str,
    end_date: str,
    window: str,
    rolling_days: int | None,
    rolling_games: int | None,
    csv_path: str | Path,
    output_dir: str | Path | None,
    grid_override: dict[str, Any] | None,
    db_path: str | Path,
    metric_override: str | None = None,
    allow_worse: bool = False,
    jobs: int = 1,
    activate_best: bool = False,
) -> ModelMarketTuningResult:
    normalized_market = _normalize_market(market)
    tuning_metric, metric_optimized = _resolve_market_metric(
        normalized_market, metric_override
    )
    started_at = _utc_now()
    tuning_outputs = run_tuning_pipeline(
        csv_path=csv_path,
        model=model,
        start_date=start_date,
        end_date=end_date,
        window=window,
        rolling_days=rolling_days,
        rolling_games=rolling_games,
        metric=tuning_metric,
        output_dir=output_dir,
        grid_override=grid_override,
        apply_best=False,
        require_improvement=not allow_worse,
        db_path=db_path,
        jobs=jobs,
        sport=sport,
        season=season,
    )
    run_id = _extract_run_id(tuning_outputs)
    model_name = normalize_model_name(tuning_outputs.model)
    best_params = tuning_outputs.best_params or {}

    games_df = load_games_df_from_csv(csv_path, sport=sport, season=season)
    model_cls = get_backtest_model(model_name)
    best_dir = Path(tuning_outputs.output_dir) / f"{run_id}__best_market"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_outputs = run_backtest(
        model_factory=_build_model_factory(model_cls, best_params),
        games_df=games_df,
        start_date=start_date,
        end_date=end_date,
        window=window,
        rolling_days=rolling_days,
        rolling_games=rolling_games,
        output_dir=best_dir,
        model_name=model_name,
    )
    summary_metrics = (
        best_outputs.metrics_overall.iloc[0].to_dict()
        if not best_outputs.metrics_overall.empty
        else {}
    )
    finished_at = _utc_now()

    best_score_numeric: float | None
    try:
        best_score_numeric = float(tuning_outputs.best_score)
    except Exception:
        best_score_numeric = None
    is_best_score_finite = best_score_numeric is not None and math.isfinite(best_score_numeric)

    if grid_override is not None:
        params_source = "file"
    elif best_params:
        params_source = "tuned"
    else:
        params_source = "default"

    notes: str | None = None
    if not best_params:
        notes = "skip: no_improvement"
    elif not is_best_score_finite:
        notes = "skip: missing_best_score"

    save_model_market_tuning_run(
        db_path,
        sport=sport,
        season=season,
        model=model_name,
        market=normalized_market,
        metric_optimized=metric_optimized,
        run_id=run_id,
        best_score=best_score_numeric,
        best_params_json=json.dumps(best_params, sort_keys=True),
        summary_metrics_json=_json_dumps(summary_metrics),
        started_at=started_at,
        finished_at=finished_at,
        notes=notes,
    )
    activated = False
    if best_params and is_best_score_finite and activate_best:
        activate_best_params(
            db_path=db_path,
            sport=sport,
            season=season,
            model=model_name,
            market=normalized_market,
            run_id=run_id,
            best_params=best_params,
            best_score=best_score_numeric,
            metric_optimized=metric_optimized,
        )
        activated = True
    return ModelMarketTuningResult(
        model=model_name,
        market=normalized_market,
        metric_optimized=metric_optimized,
        run_id=run_id,
        best_score=best_score_numeric,
        best_params=best_params,
        summary_metrics=summary_metrics,
        output_dir=Path(tuning_outputs.output_dir),
        params_source=params_source,
        activated=activated,
    )


@dataclass(frozen=True)
class EnsembleMarketTuningResult:
    market: str
    ensemble_id: str
    metric_optimized: str
    run_id: str
    games: int
    best_score: float | None
    weights: dict[str, float]
    models: list[str]
    summary_metrics: dict[str, Any] | None
    artifact_path: Path


def run_ensemble_market_tuning(
    *,
    sport: str,
    season: str,
    market: str,
    ensemble_id: str,
    start_date: str,
    end_date: str,
    models: Iterable[str] | None,
    csv_path: str | Path | None,
    db_path: str | Path,
) -> EnsembleMarketTuningResult:
    normalized_market = _normalize_market(market)
    metric_optimized = _metric_optimized_label(normalized_market)
    started_at = _utc_now()
    result = tune_market_ensemble(
        sport=sport,
        season=season,
        market=Market[normalized_market],
        start_date=start_date,
        end_date=end_date,
        ensemble_id=ensemble_id,
        models=models,
        csv_path=csv_path,
        db_path=db_path,
    )
    run_id = _build_ensemble_run_id(start_date, end_date)
    finished_at = _utc_now()

    save_ensemble_market_tuning_run(
        db_path,
        sport=sport,
        season=season,
        market=normalized_market,
        ensemble_id=ensemble_id,
        metric_optimized=metric_optimized,
        run_id=run_id,
        best_score=result.best_score,
        weights_json=json.dumps(result.weights, sort_keys=True),
        models_json=json.dumps(result.models, sort_keys=True),
        summary_metrics_json=(
            _json_dumps(result.summary_metrics)
            if result.summary_metrics is not None
            else None
        ),
        started_at=started_at,
        finished_at=finished_at,
    )
    set_active_ensemble_market_weights(
        db_path,
        sport=sport,
        season=season,
        market=normalized_market,
        ensemble_id=ensemble_id,
        weights_json=json.dumps(result.weights, sort_keys=True),
        models_json=json.dumps(result.models, sort_keys=True),
        source_run_id=run_id,
    )
    return EnsembleMarketTuningResult(
        market=normalized_market,
        ensemble_id=ensemble_id,
        metric_optimized=metric_optimized,
        run_id=run_id,
        games=int(result.games),
        best_score=result.best_score,
        weights=result.weights,
        models=result.models,
        summary_metrics=result.summary_metrics,
        artifact_path=result.artifact_path,
    )


def _normalize_market(market: str) -> str:
    normalized = market.strip().upper()
    if normalized not in {"ML", "SPREAD", "TOTAL"}:
        raise ValueError(f"Unsupported market: {market}")
    return normalized



def _build_ensemble_run_id(start_date: str, end_date: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{start_date}_to_{end_date}_ensemble_{timestamp}"


def _extract_run_id(outputs) -> str:
    if outputs.results is not None and not outputs.results.empty:
        run_id = outputs.results["run_id"].iloc[0]
        if isinstance(run_id, str) and run_id:
            return run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"tune_{timestamp}"


def _build_model_factory(model_cls, params: dict[str, Any]):
    if not params:
        return lambda cls=model_cls: cls()
    return lambda cls=model_cls, params=params: cls(**params)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(payload: dict[str, Any]) -> str:
    safe = {k: _coerce_json_value(v) for k, v in payload.items()}
    return json.dumps(safe, sort_keys=True)


def _coerce_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        return float(value)
    except Exception:
        return str(value)
