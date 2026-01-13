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
from data.repository import save_tuned_params, set_active_tuned_params
from models.registry import get_backtest_model, normalize_model_name
from joblib import Parallel, delayed
import os

_METRICS = {"log_loss", "brier_score", "mae_margin", "mae_total"}


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
    jobs: int = 1,
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
    rows_count = _validate_tuning_games(games_df, start_date, end_date)
    print(
        "TUNING "
        f"model={model_name} metric={metric} rows={rows_count} "
        f"sport={sport} season={season}"
    )

    grid = _resolve_param_grid(model_name, grid_override)
    candidates = list(_iter_param_grid(grid))
    if not candidates:
        candidates = [{}]

    base_dir = _resolve_tuning_output_dir(
        output_dir=output_dir,
        model=model_name,
        metric=metric,
        sport=sport,
        season=season,
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = _build_run_id(start_date, end_date, window, rolling_days, rolling_games)
    print(
        "TUNING "
        f"model={model_name} metric={metric} run_id={run_id} "
        f"rows={rows_count} out={base_dir}"
    )

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

    # Prepare execution context for workers. Use strings for paths to ensure
    # picklability on Windows. When jobs==1, fall back to the original serial
    # loop to guarantee identical behavior.
    context = {
        "model_name": model_name,
        "start_date": start_date,
        "end_date": end_date,
        "window": window,
        "rolling_days": rolling_days,
        "rolling_games": rolling_games,
        "metric": metric,
        "base_dir": str(base_dir),
        "run_id": run_id,
    }

    if jobs == 1:
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
                    "mae_total": metrics.get("mae_total"),
                    "metric": metric,
                    "metric_value": metric_value,
                    "output_dir": str(candidate_dir),
                }
            )
    else:
        # Limit BLAS thread usage in child processes to avoid oversubscription.
        prev_env = {
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        }
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        try:
            # Use joblib with loky backend for process-based parallelism.
            tasks = [delayed(_eval_candidate)(i, params, context, games_df) for i, params in enumerate(candidates)]
            raw_results = Parallel(n_jobs=jobs, backend="loky")(tasks)
        finally:
            # restore previous environment
            for k, v in prev_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # Ensure deterministic ordering by sorting on the candidate index.
        raw_results_sorted = sorted(raw_results, key=lambda r: int(r.get("index", 0)))
        for r in raw_results_sorted:
            results_rows.append(
                {
                    "run_id": run_id,
                    "params": r.get("params", {}),
                    "params_json": json.dumps(r.get("params", {}), sort_keys=True),
                    "log_loss": r.get("log_loss"),
                    "brier_score": r.get("brier_score"),
                    "mae_margin": r.get("mae_margin"),
                    "mae_total": r.get("mae_total"),
                    "metric": metric,
                    "metric_value": r.get("metric_value"),
                    "output_dir": r.get("output_dir"),
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

    if db_path and sport and season:
        save_tuned_params(
            db_path,
            sport=sport,
            season=season,
            model=model_name,
            metric=metric,
            run_id=run_id,
            params_json=json.dumps(candidate_params, sort_keys=True),
            best_score=candidate_score,
        )
        if apply_best and improved:
            set_active_tuned_params(
                db_path, sport=sport, season=season, model=model_name, metric=metric
            )
            print(f"Applied ACTIVE metric={metric} for model={model_name}")

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
    if model == "bradley-terry":
        return {
            "max_iter": [200, 500, 800],
            "tol": [1e-6, 1e-8],
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
        return {
            "recency_lambda": [None, 0.005, 0.01, 0.02],
            "learn_home_advantage": [False, True],
            "conditional_sd": [False, True],
            "learn_winprob_bias": [False, True],
        }
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


def _validate_tuning_games(
    games_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> int:
    if "date" not in games_df.columns:
        raise ValueError("Tuning data must include a date column.")
    games = games_df.copy()
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.normalize()
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()
    evaluation = games[(games["date"] >= start_dt) & (games["date"] <= end_dt)]
    if evaluation.empty:
        raise ValueError(
            "No tuning rows found within the start/end date range "
            f"({start_date} to {end_date})."
        )
    if "home_score" not in evaluation.columns or "away_score" not in evaluation.columns:
        raise ValueError("Tuning data must include home_score and away_score columns.")
    has_scores = evaluation["home_score"].notna() & evaluation["away_score"].notna()
    if not has_scores.any():
        raise ValueError(
            "Tuning data must include at least some rows with final scores."
        )
    return int(len(evaluation))


def _resolve_tuning_output_dir(
    *,
    output_dir: str | Path | None,
    model: str,
    metric: str,
    sport: str | None,
    season: str | None,
) -> Path:
    if output_dir is not None:
        base = Path(output_dir)
        if base.name != metric:
            base = base / metric
        return base
    if sport and season:
        return Path("outputs/tuning") / sport / season / model / metric
    return Path("outputs/tuning") / model / metric


def list_metrics() -> list[str]:
    return sorted(_METRICS)


def _eval_candidate(index: int, params: dict[str, Any], context: dict[str, Any], games_df: pd.DataFrame) -> dict[str, Any]:
    """Worker function to evaluate a single candidate.

    Returns a dict containing the candidate index and metrics. This function
    is top-level so it can be pickled by joblib on Windows.
    """
    try:
        from pathlib import Path
        # Resolve model class in child process to avoid pickling issues.
        model_name = context.get("model_name")
        model_cls = get_backtest_model(model_name)

        params_label = _format_params(params)
        candidate_dir = Path(context.get("base_dir")) / f"{context.get('run_id')}__{params_label}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        outputs = run_backtest(
            lambda params=params: model_cls(**params),
            games_df,
            start_date=context.get("start_date"),
            end_date=context.get("end_date"),
            window=context.get("window"),
            rolling_days=context.get("rolling_days"),
            rolling_games=context.get("rolling_games"),
            output_dir=candidate_dir,
            model_name=model_name,
        )
        metrics = (
            outputs.metrics_overall.iloc[0].to_dict()
            if not outputs.metrics_overall.empty
            else {}
        )
        metric_value = metrics.get(context.get("metric"))
        return {
            "index": int(index),
            "params": params,
            "metric_value": metric_value,
            "log_loss": metrics.get("log_loss"),
            "brier_score": metrics.get("brier_score"),
            "mae_margin": metrics.get("mae_margin"),
            "mae_total": metrics.get("mae_total"),
            "output_dir": str(candidate_dir),
        }
    except Exception as exc:  # pragma: no cover - bubble up with candidate index
        raise RuntimeError(f"Candidate {index} failed: {exc}") from exc
