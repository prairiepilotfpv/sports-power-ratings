"""Hyperparameter tuning pipeline for backtest models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backtest.runner import load_games_df_from_csv, run_backtest
from models.registry import get_backtest_model, normalize_model_name

_METRICS = {"log_loss", "brier_score", "mae_margin"}


@dataclass(frozen=True)
class TuningOutputs:
    """Top-level outputs from a tuning run."""

    model: str
    metric: str
    best_params: dict[str, Any]
    best_score: float
    baseline_score: float | None
    improved: bool
    applied: bool
    results: pd.DataFrame
    output_dir: Path


def run_tuning_pipeline(
    *,
    csv_path: str | Path,
    model: str,
    start_date: str,
    end_date: str,
    window: str = "expanding",
    rolling_days: int | None = None,
    rolling_games: int | None = None,
    metric: str = "log_loss",
    output_dir: str | Path | None = None,
    grid_override: dict[str, Any] | None = None,
    apply_best: bool = False,
    require_improvement: bool = True,
    db_path: str | Path | None = None,
    sport: str | None = None,
    season: str | None = None,
) -> TuningOutputs:
    """Tune model hyperparameters by backtesting each candidate grid point."""
    metric = metric.strip().lower()
    if metric not in _METRICS:
        raise ValueError(f"Unsupported metric: {metric}. Use one of {_METRICS}.")
    if apply_best and (db_path is None or sport is None or season is None):
        raise ValueError("apply_best requires db_path, sport, and season.")
    if not require_improvement and apply_best:
        raise ValueError("apply_best requires require_improvement to be enabled.")

    model_name = normalize_model_name(model)
    model_cls = get_backtest_model(model_name)
    games_df = load_games_df_from_csv(csv_path, sport=sport, season=season)

    grid = _resolve_param_grid(model_name, grid_override)
    candidates = list(_iter_param_grid(grid))
    if not candidates:
        candidates = [{}]

    base_dir = Path(output_dir) if output_dir else Path("outputs/tuning") / model_name
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = _build_run_id(start_date, end_date, window, rolling_days, rolling_games)

    baseline_score = None
    if require_improvement:
        baseline_dir = base_dir / f"{run_id}__baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_outputs = run_backtest(
            lambda: model_cls(),
            games_df,
            start_date=start_date,
            end_date=end_date,
            window=window,
            rolling_days=rolling_days,
            rolling_games=rolling_games,
            output_dir=baseline_dir,
            model_name=model_name,
        )
        baseline_metrics = (
            baseline_outputs.metrics_overall.iloc[0].to_dict()
            if not baseline_outputs.metrics_overall.empty
            else {}
        )
        baseline_score = baseline_metrics.get(metric)

    results_rows: list[dict[str, Any]] = []
    for params in candidates:
        params_label = _format_params(params)
        candidate_dir = base_dir / f"{run_id}__{params_label}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        outputs = run_backtest(
            lambda params=params: model_cls(**params),
            games_df,
            start_date=start_date,
            end_date=end_date,
            window=window,
            rolling_days=rolling_days,
            rolling_games=rolling_games,
            output_dir=candidate_dir,
            model_name=model_name,
        )
        metrics = (
            outputs.metrics_overall.iloc[0].to_dict()
            if not outputs.metrics_overall.empty
            else {}
        )
        metric_value = metrics.get(metric)
        results_rows.append(
            {
                "run_id": run_id,
                "params": params,
                "params_json": json.dumps(params, sort_keys=True),
                "log_loss": metrics.get("log_loss"),
                "brier_score": metrics.get("brier_score"),
                "mae_margin": metrics.get("mae_margin"),
                "metric": metric,
                "metric_value": metric_value,
                "output_dir": str(candidate_dir),
            }
        )

    results = pd.DataFrame(results_rows)
    if results.empty:
        raise ValueError("No tuning results produced.")

    best_idx = _select_best_index(results, metric)
    best_row = results.loc[best_idx]
    candidate_params = (
        best_row["params"] if isinstance(best_row["params"], dict) else {}
    )
    candidate_score = float(best_row["metric_value"])

    improved = True
    if require_improvement and baseline_score is not None:
        improved = candidate_score < float(baseline_score)

    best_params = candidate_params if improved else {}
    best_score = candidate_score if improved else float(baseline_score)

    applied = False
    if apply_best and improved and db_path and sport and season:
        best_dir = base_dir / f"{run_id}__best"
        best_dir.mkdir(parents=True, exist_ok=True)
        run_backtest(
            lambda: model_cls(**best_params),
            games_df,
            start_date=start_date,
            end_date=end_date,
            window=window,
            rolling_days=rolling_days,
            rolling_games=rolling_games,
            output_dir=best_dir,
            model_name=model_name,
            db_path=db_path,
            sport=sport,
            season=season,
        )
        applied = True

    results.to_csv(base_dir / f"tuning_results_{run_id}.csv", index=False)
    with (base_dir / f"best_params_{run_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model_name,
                "metric": metric,
                "best_score": best_score,
                "baseline_score": baseline_score,
                "improved": improved,
                "applied": applied,
                "best_params": best_params,
                "run_id": run_id,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    return TuningOutputs(
        model=model_name,
        metric=metric,
        best_params=best_params,
        best_score=best_score,
        baseline_score=baseline_score,
        improved=improved,
        applied=applied,
        results=results,
        output_dir=base_dir,
    )


def _resolve_param_grid(
    model: str, grid_override: dict[str, Any] | None
) -> dict[str, Iterable[Any]]:
    if grid_override:
        override = _normalize_grid_override(model, grid_override)
        if override:
            return override
    return _default_param_grid(model)


def _normalize_grid_override(
    model: str, grid_override: dict[str, Any]
) -> dict[str, Iterable[Any]]:
    if not grid_override:
        return {}
    if all(isinstance(value, list) for value in grid_override.values()):
        return grid_override
    model_grid = grid_override.get(model)
    if isinstance(model_grid, dict):
        return model_grid
    return {}


def _default_param_grid(model: str) -> dict[str, Iterable[Any]]:
    if model == "elo":
        return {
            "k_factor": [10.0, 20.0, 40.0],
            "home_advantage": [0.0, 50.0, 100.0],
            "initial_rating": [1500.0],
            "min_rating": [1.0],
        }
    if model in {"bradley_terry_hfa", "bradley_terry_calibrated_hfa"}:
        return {
            "max_iter": [200, 500, 800],
            "tol": [1e-6, 1e-8],
        }
    if model == "toor":
        return {
            "max_iter": [200, 500, 800],
            "tol": [1e-6, 1e-8],
        }
    if model == "gssd":
        return {}
    if model == "poisson":
        return {
            "learning_rate": [0.02, 0.05],
            "reg_strength": [0.05, 0.2],
            "max_iter": [1500],
        }
    return {}


def _iter_param_grid(grid: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    values = [list(grid[key]) for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=False))


def _select_best_index(results: pd.DataFrame, metric: str) -> int:
    metrics = pd.to_numeric(results["metric_value"], errors="coerce")
    if metrics.isna().all():
        raise ValueError("No numeric metric values found in tuning results.")
    return int(metrics.idxmin())


def _format_params(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    parts = []
    for key in sorted(params.keys()):
        value = params[key]
        safe_value = str(value).replace(" ", "").replace("/", "-")
        parts.append(f"{key}={safe_value}")
    return "_".join(parts)


def _build_run_id(
    start_date: str,
    end_date: str,
    window: str,
    rolling_days: int | None,
    rolling_games: int | None,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if window == "expanding":
        return f"{start_date}_to_{end_date}_expanding_{timestamp}"
    details = []
    if rolling_days is not None:
        details.append(f"{rolling_days}d")
    if rolling_games is not None:
        details.append(f"{rolling_games}g")
    detail_label = "_".join(details) if details else "rolling"
    return f"{start_date}_to_{end_date}_rolling_{detail_label}_{timestamp}"
