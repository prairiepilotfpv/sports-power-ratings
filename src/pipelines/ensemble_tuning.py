"""ML ensemble tuning and calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import json
import logging
import math
import numpy as np
import pandas as pd

from backtest.runner import load_games_df_from_csv, load_games_df_from_db, run_backtest
from pipelines.calibration_utils import select_calibrator
from data.paths import db_path_for
from data.repository import (
    get_active_ensemble_market_selection,
    get_active_model_market_params,
    load_selection_run,
    save_ensemble_market_selection_run,
    set_active_ensemble_market_selection,
)
from ensemble.config import DEFAULT_MARKET_MODELS
from ensemble.io import load_ml_weights
from markets.base import Market
from markets.registry import get_market_spec
from models.registry import (
    get_backtest_model,
    list_backtest_models,
    list_models,
    normalize_model_name,
)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class GamesDatasetMetadata:
    scorable_games: int
    date_min: str | None
    date_max: str | None
    asof: str
    data_source: str
    db_path: str | None
    csv_path: str | None


@dataclass(frozen=True)
class EnsembleSelectionResult:
    run_id: str
    selected_models: list[str]
    objective_metric: str
    score: float
    games: int
    metadata: GamesDatasetMetadata
    candidates: list[str]
    steps: list[dict[str, object]]
    coverage: dict[str, float]
    summary: dict[str, object]


def resolve_games_df(
    *,
    sport: str,
    season: str,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
    csv_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> tuple[pd.DataFrame, GamesDatasetMetadata]:
    """Load the completed-game dataset for ensemble work with strict window controls.

    The dataset is filtered to only include games with final home and away scores,
    bounded by the optional `start_date`/`end_date`, and capped at `as_of_date`
    (defaults to today or the provided `end_date`). The helper always returns
    metadata describing the scorable game count and source to keep downstream
    logic auditable.
    """
    source = "csv" if csv_path else "db"
    if csv_path:
        games_df = load_games_df_from_csv(csv_path, sport=sport, season=season)
        resolved_db_path = None
    else:
        resolved_db_path = str(db_path or db_path_for(sport, season))
        games_df = load_games_df_from_db(resolved_db_path, sport=sport, season=season)

    df = games_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    if df.empty:
        raise ValueError("No completed games found for the requested dataset.")

    def _normalize_cutoff(value: str | None) -> pd.Timestamp | None:
        if value is None:
            return None
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Invalid ISO date: {value}")
        return ts.normalize()

    start_ts = _normalize_cutoff(start_date)
    end_ts = _normalize_cutoff(end_date)
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start_date must not be after end_date.")
    asof_ts = _normalize_cutoff(as_of_date) or end_ts
    if asof_ts is None:
        asof_ts = pd.Timestamp.utcnow().normalize()

    date_col = df["date"]
    if start_ts is not None:
        df = df[date_col >= start_ts]
    if end_ts is not None:
        df = df[date_col <= end_ts]
    df = df[df["date"] <= asof_ts]
    if df.empty:
        raise ValueError("No scored games remain after applying the requested window.")

    if "game_key" not in df.columns:
        df["game_key"] = _build_game_key(df)

    game_id_series = df["game_id"].dropna().astype(str)
    unique_games = (
        game_id_series.unique().tolist()
        if not game_id_series.empty
        else df["game_key"].dropna().str.lower().unique().tolist()
    )
    scorable_games = len(unique_games)
    if scorable_games == 0:
        raise ValueError("Dataset contains no uniquely identifiable games.")

    date_min = str(df["date"].min().date())
    date_max = str(df["date"].max().date())
    metadata = GamesDatasetMetadata(
        scorable_games=scorable_games,
        date_min=date_min,
        date_max=date_max,
        asof=str(asof_ts.date()),
        data_source=source,
        db_path=resolved_db_path,
        csv_path=str(csv_path) if csv_path else None,
    )
    return df, metadata


@dataclass(frozen=True)
class EnsembleTuningResult:
    weights: dict[str, float]
    artifact_path: Path
    games: int
    models: list[str]
    best_score: float | None = None
    metric_optimized: str | None = None
    summary_metrics: dict[str, float] | None = None
    metadata: GamesDatasetMetadata | None = None
    dataset: EnsembleDataset | None = None


def tune_ml_ensemble(
    *,
    sport: str,
    season: str,
    start_date: str,
    end_date: str,
    as_of_date: str | None = None,
    ensemble_id: str,
    models: Iterable[str] | None = None,
    csv_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> EnsembleTuningResult:
    return tune_market_ensemble(
        sport=sport,
        season=season,
        market=Market.ML,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
        ensemble_id=ensemble_id,
        models=models,
        csv_path=csv_path,
        db_path=db_path,
    )


def tune_market_ensemble(
    *,
    sport: str,
    season: str,
    market: Market | str,
    start_date: str,
    end_date: str,
    as_of_date: str | None = None,
    ensemble_id: str,
    models: Iterable[str] | None = None,
    csv_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> EnsembleTuningResult:
    spec = get_market_spec(market)
    bundle = _collect_market_predictions(
        sport=sport,
        season=season,
        market=spec.market,
        models=models,
        csv_path=csv_path,
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
    )
    dataset = _ensemble_dataset_from_predictions(bundle.predictions, bundle.models, spec.market)
    weights = optimize_ensemble_weights(
        dataset.pred_matrix, dataset.targets, dataset.models, market=spec.market
    )
    metrics = _ensemble_metrics(
        dataset.pred_matrix,
        dataset.targets,
        weights,
        market=spec.market,
        models=dataset.models,
    )
    artifact_path = save_ensemble_weights(
        sport=sport,
        season=season,
        ensemble_id=ensemble_id,
        market=spec.market.name,
        start_date=start_date,
        end_date=end_date,
        models=dataset.models,
        weights=weights,
        objective=_metric_name_for_market(spec.market),
    )
    return EnsembleTuningResult(
        weights=weights,
        artifact_path=artifact_path,
        games=int(len(dataset.targets)),
        models=list(dataset.models),
        best_score=metrics.get(_metric_name_for_market(spec.market)),
        metric_optimized=_metric_name_for_market(spec.market),
        summary_metrics=metrics,
        metadata=bundle.metadata,
        dataset=dataset,
    )


@dataclass(frozen=True)
class EnsembleDataset:
    pred_matrix: np.ndarray
    targets: np.ndarray
    models: list[str]
    game_keys: list[str]


@dataclass(frozen=True)
class MarketPredictionBundle:
    models: list[str]
    predictions: dict[str, pd.DataFrame]
    metadata: GamesDatasetMetadata


def _collect_market_predictions(
    *,
    sport: str,
    season: str,
    market: Market,
    models: Iterable[str] | None,
    csv_path: str | Path | None,
    db_path: str | Path | None,
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None,
) -> MarketPredictionBundle:
    """Run each candidate model over the canonical completed-game dataset."""
    model_list = _resolve_ml_models(models)
    games_df, metadata = resolve_games_df(
        sport=sport,
        season=season,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
        csv_path=csv_path,
        db_path=db_path,
    )
    preds_by_model: dict[str, pd.DataFrame] = {}
    for model_name in model_list:
        model_cls = get_backtest_model(model_name)
        params = _load_market_model_params(
            db_path=db_path,
            sport=sport,
            season=season,
            model=model_name,
            market=market,
        )
        outputs = run_backtest(
            model_factory=_build_model_factory(model_cls, params),
            games_df=games_df,
            start_date=start_date,
            end_date=end_date,
            model_name=model_name,
        )
        preds = _prepare_predictions_for_market(
            outputs.predictions, model_name=model_name, market=market
        )
        if preds is not None and not preds.empty:
            preds_by_model[model_name] = preds

    non_empty_models = [m for m in model_list if m in preds_by_model]
    if not non_empty_models:
        raise ValueError("No models produced predictions for the requested market/window.")

    filtered_preds = {m: preds_by_model[m] for m in non_empty_models}
    return MarketPredictionBundle(
        models=non_empty_models,
        predictions=filtered_preds,
        metadata=metadata,
    )


def _ensemble_dataset_from_predictions(
    predictions: dict[str, pd.DataFrame],
    models: Iterable[str],
    market: Market,
) -> EnsembleDataset:
    """Align a subset of predictions so the same games are evaluated for each model."""
    model_list = [m for m in models if m in predictions]
    non_empty_preds = {m: predictions[m] for m in model_list if not predictions[m].empty}
    if not non_empty_preds:
        raise ValueError("No overlapping predictions found for the requested models.")

    aligned = _align_market_predictions(non_empty_preds)
    pred_matrix = aligned.pivot_table(
        index="game_key", columns="model_name", values="pred_value", aggfunc="mean"
    )
    pred_matrix = pred_matrix[model_list]
    targets = (
        aligned.drop_duplicates("game_key")[["game_key", "target_value"]]
        .set_index("game_key")
    )

    pred_values = pred_matrix.to_numpy(dtype=float)
    target_values = targets.loc[pred_matrix.index, "target_value"].to_numpy(dtype=float)

    if pred_values.size == 0:
        raise ValueError("No overlapping predictions found across models.")

    return EnsembleDataset(
        pred_matrix=pred_values,
        targets=target_values,
        models=model_list,
        game_keys=[str(key) for key in pred_matrix.index.tolist()],
    )


def _score_model_subset(
    predictions: dict[str, pd.DataFrame],
    models: Iterable[str],
    market: Market,
) -> tuple[float, EnsembleDataset]:
    dataset = _ensemble_dataset_from_predictions(predictions, models, market)
    metric_key = _metric_name_for_market(market)
    metrics = _ensemble_metrics(
        dataset.pred_matrix,
        dataset.targets,
        {m: 1.0 for m in dataset.models},
        market=market,
        models=dataset.models,
    )
    score = metrics.get(metric_key)
    if score is None or (isinstance(score, float) and math.isnan(score)):
        score = float("inf")
    return float(score), dataset


def _build_selection_run_id(
    *,
    sport: str,
    season: str,
    market: Market,
    ensemble_id: str,
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    start_label = start_date or "start"
    end_label = end_date or "end"
    asof_label = as_of_date or "asof"
    return (
        f"selection-{sport}-{season}-{market.name}-{ensemble_id}-"
        f"{start_label}-{end_label}-{asof_label}-{timestamp}"
    )


def select_market_ensemble(
    *,
    sport: str,
    season: str,
    market: Market | str,
    ensemble_id: str,
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None,
    candidates: Iterable[str] | None,
    min_coverage: float,
    max_members: int,
    epsilon: float,
    csv_path: str | Path | None,
    db_path: str | Path | None,
) -> EnsembleSelectionResult:
    spec = get_market_spec(market)
    bundle = _collect_market_predictions(
        sport=sport,
        season=season,
        market=spec.market,
        models=candidates,
        csv_path=csv_path,
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
    )
    quality_metric = _metric_name_for_market(spec.market)
    coverage = {}
    # Count unique games per model so coverage represents the ratio of scored rows each model produced.
    for model_name, df in bundle.predictions.items():
        unique_games = df["game_key"].nunique()
        coverage[model_name] = unique_games / bundle.metadata.scorable_games
    eligible_models = [
        model for model in bundle.models if coverage.get(model, 0.0) >= min_coverage
    ]
    if not eligible_models:
        raise ValueError(
            "No candidate model met the minimum coverage requirement "
            f"({min_coverage:.2f})."
        )

    standalone_scores: dict[str, float] = {}
    for model_name in eligible_models:
        score, _ = _score_model_subset(bundle.predictions, [model_name], spec.market)
        standalone_scores[model_name] = score

    best_model = min(eligible_models, key=lambda name: standalone_scores[name])
    steps: list[dict[str, object]] = []
    selected_models = [best_model]
    current_score = standalone_scores[best_model]
    steps.append(
        {
            "stage": 0,
            "picked": best_model,
            "score": current_score,
        }
    )
    remaining = [model for model in eligible_models if model != best_model]

    # Forward-selection loop: add the candidate that provides the best metric drop, stopping when no
    # meaningful improvement remains.
    while remaining and len(selected_models) < max_members:
        best_candidate = None
        best_candidate_score = current_score
        for candidate in remaining:
            score, _ = _score_model_subset(
                bundle.predictions, selected_models + [candidate], spec.market
            )
            if best_candidate is None or score < best_candidate_score:
                best_candidate = candidate
                best_candidate_score = score
        improvement = current_score - best_candidate_score
        if improvement < epsilon:
            break
        selected_models.append(best_candidate)
        remaining.remove(best_candidate)
        steps.append(
            {
                "stage": len(steps),
                "picked": best_candidate,
                "score": best_candidate_score,
                "delta": improvement,
            }
        )
        current_score = best_candidate_score

    final_score = current_score
    summary = {
        "objective_metric": quality_metric,
        "standalone_scores": standalone_scores,
        "coverage": coverage,
        "steps": steps,
        "final_score": final_score,
        "selected_models": selected_models,
        "scorable_games": bundle.metadata.scorable_games,
        "date_min": bundle.metadata.date_min,
        "date_max": bundle.metadata.date_max,
        "as_of": bundle.metadata.asof,
    }
    run_id = _build_selection_run_id(
        sport=sport,
        season=season,
        market=spec.market,
        ensemble_id=ensemble_id,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date or bundle.metadata.asof,
    )
    return EnsembleSelectionResult(
        run_id=run_id,
        selected_models=selected_models,
        objective_metric=quality_metric,
        score=final_score,
        games=bundle.metadata.scorable_games,
        metadata=bundle.metadata,
        candidates=eligible_models,
        steps=steps,
        coverage=coverage,
        summary=summary,
    )


def run_market_ensemble_selection(
    *,
    sport: str,
    season: str,
    market: Market | str,
    ensemble_id: str,
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None,
    candidates: Iterable[str] | None,
    min_coverage: float,
    max_members: int,
    epsilon: float,
    activate: bool,
    notes: str | None,
    csv_path: str | Path | None,
    db_path: str | Path,
) -> tuple[EnsembleSelectionResult, bool]:
    """Execute a forward-selection workflow and persist the selection result."""
    spec = get_market_spec(market)
    selection = select_market_ensemble(
        sport=sport,
        season=season,
        market=spec.market,
        ensemble_id=ensemble_id,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
        candidates=candidates,
        min_coverage=min_coverage,
        max_members=max_members,
        epsilon=epsilon,
        csv_path=csv_path,
        db_path=db_path,
    )
    baseline_score = min(selection.summary["standalone_scores"].values())
    save_ensemble_market_selection_run(
        db_path=db_path,
        sport=sport,
        season=season,
        market=spec.market.name,
        ensemble_id=ensemble_id,
        run_id=selection.run_id,
        window_start=start_date,
        window_end=end_date,
        asof=selection.metadata.asof,
        data_source=selection.metadata.data_source,
        dataset_db_path=selection.metadata.db_path,
        csv_path=selection.metadata.csv_path,
        scorable_games=selection.metadata.scorable_games,
        date_min=selection.metadata.date_min,
        date_max=selection.metadata.date_max,
        candidates=selection.candidates,
        selected=selection.selected_models,
        objective_metric=selection.objective_metric,
        summary=selection.summary,
        baseline_score=float(baseline_score),
        notes=notes,
    )

    active_selection = get_active_ensemble_market_selection(
        db_path=db_path,
        sport=sport,
        season=season,
        market=spec.market.name,
        ensemble_id=ensemble_id,
    )
    active_score: float | None = None
    if active_selection:
        loaded = load_selection_run(db_path, run_id=active_selection["active_run_id"])
        if loaded:
            existing_final = loaded.get("summary", {}).get("final_score")
            active_score = float(existing_final) if existing_final is not None else None
    should_activate = False
    if activate:
        current_score = selection.summary["final_score"]
        previous_score = float(active_score) if active_score is not None else float("inf")
        if current_score + epsilon < previous_score:
            set_active_ensemble_market_selection(
                db_path=db_path,
                sport=sport,
                season=season,
                market=spec.market.name,
                ensemble_id=ensemble_id,
                active_run_id=selection.run_id,
            )
            should_activate = True
        else:
            _LOG.info(
                "Selection run did not improve (current=%.6f active=%.6f); skipping activation.",
                current_score,
                previous_score,
            )
    return selection, should_activate


def build_ml_ensemble_dataset(
    *,
    sport: str,
    season: str,
    start_date: str,
    end_date: str,
    as_of_date: str | None = None,
    models: Iterable[str] | None = None,
    csv_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> EnsembleDataset:
    return build_market_ensemble_dataset(
        sport=sport,
        season=season,
        market=Market.ML,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
        models=models,
        csv_path=csv_path,
        db_path=db_path,
    )


def build_market_ensemble_dataset(
    *,
    sport: str,
    season: str,
    market: Market,
    start_date: str,
    end_date: str,
    as_of_date: str | None = None,
    models: Iterable[str] | None = None,
    csv_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> EnsembleDataset:
    bundle = _collect_market_predictions(
        sport=sport,
        season=season,
        market=market,
        models=models,
        csv_path=csv_path,
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
    )
    return _ensemble_dataset_from_predictions(bundle.predictions, bundle.models, market)


def optimize_ensemble_weights(
    pred_matrix: np.ndarray,
    targets: np.ndarray,
    models: Iterable[str],
    *,
    market: Market = Market.ML,
) -> dict[str, float]:
    model_list = list(models)
    n_models = len(model_list)
    if n_models == 0:
        raise ValueError("No models provided for ensemble optimization.")
    if n_models == 1:
        return {model_list[0]: 1.0}

    if market == Market.ML:
        weights = _optimize_simplex_log_loss(
            pred_matrix=pred_matrix,
            targets=targets,
            max_iter=5000,
            learning_rate=0.2,
            tol=1e-8,
        )
    else:
        weights = _optimize_simplex_mae(
            pred_matrix=pred_matrix,
            targets=targets,
            max_iter=4000,
            learning_rate=0.1,
            tol=1e-8,
        )
    return {name: float(w) for name, w in zip(model_list, weights)}


def save_ensemble_weights(
    *,
    sport: str,
    season: str,
    ensemble_id: str,
    market: str,
    start_date: str,
    end_date: str,
    models: Iterable[str],
    weights: dict[str, float],
    objective: str = "log_loss",
) -> Path:
    spec = get_market_spec(market)
    out_path = spec.ensemble_weights_path(sport, season, ensemble_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ensemble_id": ensemble_id,
        "market": str(market),
        "objective": objective,
        "train_window": {"start": start_date, "end": end_date},
        "models": list(models),
        "weights": {k: float(v) for k, v in weights.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(
        json_dumps(payload),
        encoding="utf-8",
    )
    return out_path


def calibrate_ml_ensemble(
    *,
    sport: str,
    season: str,
    start_date: str,
    end_date: str,
    ensemble_id: str,
    models: Iterable[str] | None = None,
    csv_path: str | Path | None = None,
    db_path: str | Path | None = None,
    method: str = "auto",
) -> Path:
    dataset = build_ml_ensemble_dataset(
        sport=sport,
        season=season,
        start_date=start_date,
        end_date=end_date,
        models=models,
        csv_path=csv_path,
        db_path=db_path,
    )
    weights = load_ml_weights(sport, season, ensemble_id, market="ML") or {}
    if weights:
        weight_vec = _weights_vector(dataset.models, weights)
    else:
        weight_vec = np.full(len(dataset.models), 1.0 / len(dataset.models))

    probs = np.clip(dataset.pred_matrix @ weight_vec, 1e-6, 1 - 1e-6)
    calib_df = pd.DataFrame(
        {
            "p_home_win": probs,
            "home_win": dataset.targets,
        }
    )
    calibrator = select_calibrator(method, len(calib_df))
    calibrator.fit(calib_df)

    spec = get_market_spec("ML")
    calib_dir = spec.calibrator_dir(sport, season, ensemble_id)
    calib_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = calib_dir / f"{ensemble_id}_{timestamp}.joblib"
    calibrator.save(out_path)
    return out_path


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)



def _load_games_df(
    sport: str,
    season: str,
    *,
    csv_path: str | Path | None,
    db_path: str | Path | None,
) -> pd.DataFrame:
    df, _ = resolve_games_df(
        sport=sport,
        season=season,
        start_date=None,
        end_date=None,
        as_of_date=None,
        csv_path=csv_path,
        db_path=db_path,
    )
    return df


def _resolve_ml_models(models: Iterable[str] | None) -> list[str]:
    if models:
        model_list = [normalize_model_name(m) for m in models if m.strip()]
    else:
        available = set(list_models())
        default_models = DEFAULT_MARKET_MODELS.get("ML", [])
        model_list = [m for m in default_models if m in available]
        if not model_list:
            model_list = list_models()

    backtest_models = set(list_backtest_models())
    missing = [m for m in model_list if m not in backtest_models]
    if missing:
        raise ValueError(
            "Unsupported backtest models: "
            f"{', '.join(missing)}. "
            "Use models with backtest implementations."
        )
    return model_list


def _load_market_model_params(
    *,
    db_path: str | Path | None,
    sport: str,
    season: str,
    model: str,
    market: Market,
) -> dict | None:
    if db_path is None:
        return None
    try:
        return get_active_model_market_params(
            db_path,
            sport=sport,
            season=season,
            model=model,
            market=market.name,
        )
    except Exception:
        return None


def _build_model_factory(model_cls, params: dict | None):
    if not params:
        return lambda cls=model_cls: cls()
    return lambda cls=model_cls, params=params: cls(**params)


def _prepare_predictions(pred_df: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    df = pred_df.copy()
    df["model_name"] = model_name
    df["game_key"] = _build_game_key(df)
    df = df.dropna(subset=["p_home_win", "home_win", "game_key"])
    return df[["game_key", "game_id", "model_name", "p_home_win", "home_win"]]


def _prepare_predictions_for_market(
    pred_df: pd.DataFrame,
    *,
    model_name: str,
    market: Market,
) -> pd.DataFrame:
    df = pred_df.copy()
    df["model_name"] = model_name
    df["game_key"] = _build_game_key(df)
    if market == Market.ML:
        df = df.dropna(subset=["p_home_win", "home_win", "game_key"])
        df = df.rename(columns={"p_home_win": "pred_value", "home_win": "target_value"})
        return df[["game_key", "game_id", "model_name", "pred_value", "target_value"]]
    if market == Market.SPREAD:
        df = df.dropna(subset=["pred_margin", "actual_margin", "game_key"])
        df = df.rename(
            columns={"pred_margin": "pred_value", "actual_margin": "target_value"}
        )
        return df[["game_key", "game_id", "model_name", "pred_value", "target_value"]]
    df = df.dropna(subset=["pred_total", "home_score", "away_score", "game_key"])
    df["target_value"] = df["home_score"] + df["away_score"]
    df = df.rename(columns={"pred_total": "pred_value"})
    return df[["game_key", "game_id", "model_name", "pred_value", "target_value"]]


def _align_predictions(preds_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_keys: set[str] | None = None
    cleaned_frames: list[pd.DataFrame] = []
    for model_name, df in preds_by_model.items():
        grouped = df.groupby("game_key", as_index=False).mean(numeric_only=True)
        grouped["model_name"] = model_name
        keys = set(grouped["game_key"].astype(str).tolist())
        common_keys = keys if common_keys is None else common_keys & keys
        cleaned_frames.append(grouped)

    if not common_keys:
        raise ValueError("No overlapping games across model predictions.")

    aligned_frames = [
        frame[frame["game_key"].astype(str).isin(common_keys)] for frame in cleaned_frames
    ]
    aligned = pd.concat(aligned_frames, ignore_index=True)
    return aligned


def _align_market_predictions(preds_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_keys: set[str] | None = None
    cleaned_frames: list[pd.DataFrame] = []
    for model_name, df in preds_by_model.items():
        grouped = df.groupby("game_key", as_index=False).mean(numeric_only=True)
        grouped["model_name"] = model_name
        keys = set(grouped["game_key"].astype(str).tolist())
        common_keys = keys if common_keys is None else common_keys & keys
        cleaned_frames.append(grouped)

    if not common_keys:
        raise ValueError("No overlapping games across model predictions.")

    aligned_frames = [
        frame[frame["game_key"].astype(str).isin(common_keys)] for frame in cleaned_frames
    ]
    aligned = pd.concat(aligned_frames, ignore_index=True)
    return aligned


def _build_game_key(df: pd.DataFrame) -> pd.Series:
    has_game_id = "game_id" in df.columns
    if has_game_id:
        game_id = df["game_id"].astype(str).str.strip()
        valid = game_id != ""
        if valid.any():
            return game_id.where(valid, _fallback_game_key(df))
    return _fallback_game_key(df)


def _fallback_game_key(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    home = df["home_team"].astype(str).str.strip().str.lower()
    away = df["away_team"].astype(str).str.strip().str.lower()
    return date + "-" + home + "-" + away


def _weights_vector(models: Iterable[str], weights: dict[str, float]) -> np.ndarray:
    raw = np.array([float(weights.get(m, 0.0)) for m in models], dtype=float)
    if raw.sum() <= 0:
        return np.full(len(raw), 1.0 / len(raw))
    return raw / raw.sum()


def _metric_name_for_market(market: Market) -> str:
    if market == Market.ML:
        return "log_loss"
    if market == Market.SPREAD:
        return "mae_margin"
    return "mae_total"


def _ensemble_metrics(
    pred_matrix: np.ndarray,
    targets: np.ndarray,
    weights: dict[str, float],
    *,
    market: Market,
    models: Iterable[str],
) -> dict[str, float]:
    weight_vec = _weights_vector(models, weights)
    preds = pred_matrix @ weight_vec
    metrics: dict[str, float] = {}
    if market == Market.ML:
        probs = np.clip(preds.astype(float), 1e-6, 1 - 1e-6)
        targets_arr = targets.astype(float)
        metrics["log_loss"] = float(
            -np.mean(targets_arr * np.log(probs) + (1 - targets_arr) * np.log(1 - probs))
        )
        metrics["brier_score"] = float(np.mean((probs - targets_arr) ** 2))
        return metrics
    metrics[_metric_name_for_market(market)] = _mae(preds, targets)
    return metrics


def _optimize_simplex_log_loss(
    *,
    pred_matrix: np.ndarray,
    targets: np.ndarray,
    max_iter: int,
    learning_rate: float,
    tol: float,
) -> np.ndarray:
    n_models = pred_matrix.shape[1]
    weights = np.full(n_models, 1.0 / n_models, dtype=float)
    best_weights = weights.copy()
    best_loss = _log_loss(pred_matrix @ weights, targets)
    lr = float(learning_rate)

    for _ in range(max_iter):
        probs = np.clip(pred_matrix @ weights, 1e-6, 1 - 1e-6)
        grad = _log_loss_grad(pred_matrix, targets, probs)
        candidate = _project_to_simplex(weights - lr * grad)
        loss = _log_loss(pred_matrix @ candidate, targets)
        if loss + tol < best_loss:
            best_loss = loss
            best_weights = candidate
            weights = candidate
            continue
        lr *= 0.5
        if lr < 1e-6:
            break
        weights = candidate
    return best_weights


def _optimize_simplex_mae(
    *,
    pred_matrix: np.ndarray,
    targets: np.ndarray,
    max_iter: int,
    learning_rate: float,
    tol: float,
) -> np.ndarray:
    n_models = pred_matrix.shape[1]
    weights = np.full(n_models, 1.0 / n_models, dtype=float)
    best_weights = weights.copy()
    best_loss = _mae(pred_matrix @ weights, targets)
    lr = float(learning_rate)

    for _ in range(max_iter):
        preds = pred_matrix @ weights
        grad = _mae_grad(pred_matrix, targets, preds)
        candidate = _project_to_simplex(weights - lr * grad)
        loss = _mae(pred_matrix @ candidate, targets)
        if loss + tol < best_loss:
            best_loss = loss
            best_weights = candidate
            weights = candidate
            continue
        lr *= 0.5
        if lr < 1e-6:
            break
        weights = candidate
    return best_weights


def _log_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    probs = np.clip(probs.astype(float), 1e-6, 1 - 1e-6)
    targets = targets.astype(float)
    return float(-np.mean(targets * np.log(probs) + (1 - targets) * np.log(1 - probs)))


def _log_loss_grad(
    prob_matrix: np.ndarray, targets: np.ndarray, probs: np.ndarray
) -> np.ndarray:
    denom = probs * (1 - probs)
    denom = np.clip(denom, 1e-6, None)
    residual = (probs - targets) / denom
    grad = prob_matrix.T @ residual
    return grad / len(targets)


def _mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds.astype(float) - targets.astype(float))))


def _mae_grad(
    pred_matrix: np.ndarray, targets: np.ndarray, preds: np.ndarray
) -> np.ndarray:
    residual = preds.astype(float) - targets.astype(float)
    grad = pred_matrix.T @ np.sign(residual)
    return grad / len(targets)


def _project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Project vector onto simplex: non-negative entries that sum to 1."""
    if vector.ndim != 1:
        raise ValueError("Simplex projection expects a 1D vector.")
    n = vector.size
    if n == 1:
        return np.array([1.0])
    sorted_vec = np.sort(vector)[::-1]
    cumsum = np.cumsum(sorted_vec)
    rho = np.where(sorted_vec - (cumsum - 1) / np.arange(1, n + 1) > 0)[0]
    if rho.size == 0:
        return np.full(n, 1.0 / n)
    rho_idx = rho[-1]
    theta = (cumsum[rho_idx] - 1) / float(rho_idx + 1)
    projected = np.maximum(vector - theta, 0.0)
    if projected.sum() <= 0:
        return np.full(n, 1.0 / n)
    return projected / projected.sum()
