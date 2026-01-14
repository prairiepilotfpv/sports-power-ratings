"""Batch tuning helper that runs tuning across multiple models/metrics."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

from ensemble.config import DEFAULT_MARKET_MODELS, load_ensemble_config
from models.registry import (
    is_experimental_model,
    list_backtest_models,
    normalize_model_name,
)
from pipelines.tuning import run_tuning_pipeline
from utils.parallel import parallel_map, parallel_map_process, resolve_jobs


DEFAULT_BATCH_METRICS = ["log_loss", "mae_margin", "mae_total"]


@dataclass(frozen=True)
class _TuneBatchTask:
    model: str
    metric: str
    csv_path: str
    db_path: str
    start_date: str
    end_date: str
    window: str
    rolling_days: int | None
    rolling_games: int | None
    sport: str
    season: str
    per_run_jobs: int


def _execute_tune_task(task: _TuneBatchTask) -> dict[str, object]:
    try:
        outputs = run_tuning_pipeline(
            csv_path=task.csv_path,
            model=task.model,
            start_date=task.start_date,
            end_date=task.end_date,
            window=task.window,
            rolling_days=task.rolling_days,
            rolling_games=task.rolling_games,
            metric=task.metric,
            output_dir=None,
            grid_override=None,
            apply_best=True,
            require_improvement=True,
            db_path=task.db_path,
            jobs=task.per_run_jobs,
            sport=task.sport,
            season=task.season,
        )
        best_run_id = None
        if not outputs.results.empty and "run_id" in outputs.results.columns:
            best_run_id = outputs.results.iloc[0].get("run_id")
        result = {
            "model": task.model,
            "metric": task.metric,
            "best_score": outputs.best_score,
            "run_id": best_run_id,
            "output_dir": str(outputs.output_dir),
            "applied": outputs.applied,
        }
        print(
            "TUNE-BATCH "
            f"model={task.model} metric={task.metric} run_id={best_run_id} "
            f"best_score={outputs.best_score} applied={outputs.applied}"
        )
        return result
    except Exception as exc:  # pragma: no cover - defensive
        print(f"TUNE-BATCH ERROR model={task.model} metric={task.metric}: {exc}")
        return {
            "model": task.model,
            "metric": task.metric,
            "error": str(exc),
        }


def _default_allowlist() -> list[str]:
    union: list[str] = []
    for market_models in DEFAULT_MARKET_MODELS.values():
        for name in market_models:
            norm = normalize_model_name(name)
            if norm not in union:
                union.append(norm)
    return union


def _models_from_config_union(sport: str, season: str) -> list[str] | None:
    try:
        config = load_ensemble_config(
            sport,
            season,
            available_models=list_backtest_models(),
        )
    except TypeError:
        # Backward compatibility for tests that monkeypatch load_ensemble_config without kwargs support.
        config = load_ensemble_config(sport, season)
    markets = config.get("markets", {}) or {}
    collected: list[str] = []
    for entry in markets.values():
        models = entry.get("models") if isinstance(entry, dict) else None
        if not models:
            continue
        for m in models:
            norm = normalize_model_name(m)
            if norm not in collected:
                collected.append(norm)
    return collected or None


def run_tune_batch(
    *,
    sport: str,
    season: str,
    start_date: str,
    end_date: str,
    csv_path: str | Path,
    db_path: str | Path,
    models: Iterable[str] | None = None,
    metrics: Iterable[str] | None = None,
    window: str = "expanding",
    rolling_days: int | None = None,
    rolling_games: int | None = None,
    include_all_models: bool = False,
    include_experimental: bool = False,
    jobs: int | None = None,
) -> list[dict[str, object]]:
    """Run tuning across many models/metrics and apply best params.

    Returns a list of result dicts for progress/leaderboard reporting.
    """

    metric_list = [str(m).strip().lower() for m in (metrics or DEFAULT_BATCH_METRICS)]

    if models is not None:
        model_list = [normalize_model_name(m) for m in models]
    elif include_all_models:
        model_list = list_backtest_models()
        include_experimental = True
    else:
        model_list = _models_from_config_union(sport, season)
        if model_list is None:
            model_list = _default_allowlist()

    normalized_models: list[str] = []
    for name in model_list:
        norm = normalize_model_name(name)
        if norm not in normalized_models:
            normalized_models.append(norm)

    if not include_experimental and models is None and not include_all_models:
        filtered = [m for m in normalized_models if not is_experimental_model(m)]
        if not filtered:
            filtered = [m for m in _default_allowlist() if not is_experimental_model(m)]
        normalized_models = filtered

    available_set = set(list_backtest_models())
    filtered_models = [m for m in normalized_models if m in available_set]
    if filtered_models:
        normalized_models = filtered_models

    # Defensive exclusion: do not include explicit HFA backtest variants in
    # batch tuning unless the caller explicitly provided a models list that
    # includes them. This prevents tuning of HFA-specific backtests when a
    # user requests "all models" but doesn't actually use those variants in
    # any ensemble/config.
    hfa_exclusions = {"bradley_terry_hfa", "bradley_terry_calibrated_hfa"}
    if models is None:
        normalized_models = [m for m in normalized_models if m not in hfa_exclusions]

    print(f"TUNE-BATCH selected models: {normalized_models}")
    print(f"TUNE-BATCH metrics: {metric_list}")

    if not normalized_models:
        print("TUNE-BATCH no models selected after filtering; nothing to tune.")
        return []

    total_workers = resolve_jobs(jobs)
    if normalized_models:
        batch_workers = min(total_workers, len(metric_list) * len(normalized_models))
    else:
        batch_workers = 1
    per_run_jobs = max(1, total_workers // batch_workers)

    tasks = [
        _TuneBatchTask(
            model=model,
            metric=metric,
            csv_path=str(csv_path),
            db_path=str(db_path),
            start_date=start_date,
            end_date=end_date,
            window=window,
            rolling_days=rolling_days,
            rolling_games=rolling_games,
            sport=sport,
            season=season,
            per_run_jobs=per_run_jobs,
        )
        for metric in metric_list
        for model in normalized_models
    ]
    batch_workers = min(batch_workers, len(tasks)) if tasks else 1

    if batch_workers <= 1:
        return [_execute_tune_task(task) for task in tasks]

    backend = os.getenv("SPR_TUNE_BATCH_BACKEND", "process").lower()
    if backend == "thread":
        return parallel_map(_execute_tune_task, tasks, max_workers=batch_workers)
    return parallel_map_process(_execute_tune_task, tasks, max_workers=batch_workers)


def summarize_tune_batch(results: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    """Build a leaderboard per metric from batch tuning results."""
    leaderboard: dict[str, list[dict[str, object]]] = {}
    for row in results:
        metric = str(row.get("metric"))
        if not metric:
            continue
        leaderboard.setdefault(metric, []).append(row)
    for metric, rows in leaderboard.items():
        leaderboard[metric] = sorted(
            [r for r in rows if "best_score" in r], key=lambda r: r.get("best_score", float("inf"))
        )
    return leaderboard
