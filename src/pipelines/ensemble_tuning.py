"""ML ensemble tuning and calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from backtest.runner import load_games_df_from_csv, load_games_df_from_db, run_backtest
from calibration.isotonic import IsotonicCalibrator
from calibration.platt import PlattScalingCalibrator
from data.repository import get_active_model_market_params
from ensemble.io import load_ml_weights
from markets.base import Market
from markets.registry import get_market_spec
from models.registry import (
    get_backtest_model,
    list_backtest_models,
    list_models,
    normalize_model_name,
)


DEFAULT_ML_MODELS = ("bradley-terry", "elo", "gssd", "poisson", "toor")


@dataclass(frozen=True)
class EnsembleTuningResult:
    weights: dict[str, float]
    artifact_path: Path
    games: int
    models: list[str]
    best_score: float | None = None
    metric_optimized: str | None = None
    summary_metrics: dict[str, float] | None = None


def tune_ml_ensemble(
    *,
    sport: str,
    season: str,
    start_date: str,
    end_date: str,
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
    ensemble_id: str,
    models: Iterable[str] | None = None,
    csv_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> EnsembleTuningResult:
    spec = get_market_spec(market)
    dataset = build_market_ensemble_dataset(
        sport=sport,
        season=season,
        market=spec.market,
        start_date=start_date,
        end_date=end_date,
        models=models,
        csv_path=csv_path,
        db_path=db_path,
    )
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
    )


@dataclass(frozen=True)
class EnsembleDataset:
    pred_matrix: np.ndarray
    targets: np.ndarray
    models: list[str]
    game_keys: list[str]


def build_ml_ensemble_dataset(
    *,
    sport: str,
    season: str,
    start_date: str,
    end_date: str,
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
    models: Iterable[str] | None = None,
    csv_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> EnsembleDataset:
    model_list = _resolve_ml_models(models)
    games_df = _load_games_df(sport, season, csv_path=csv_path, db_path=db_path)

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
        preds_by_model[model_name] = preds

    aligned = _align_market_predictions(preds_by_model)
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
    calibrator = _select_calibrator(method, len(calib_df))
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


def _select_calibrator(method: str, n_samples: int):
    normalized = (method or "").strip().lower()
    if normalized in {"platt", "logistic"}:
        return PlattScalingCalibrator()
    if normalized == "isotonic":
        return IsotonicCalibrator()
    if normalized != "auto":
        raise ValueError("calibrator method must be auto, platt, or isotonic.")
    return IsotonicCalibrator() if n_samples >= 500 else PlattScalingCalibrator()


def _load_games_df(
    sport: str,
    season: str,
    *,
    csv_path: str | Path | None,
    db_path: str | Path | None,
) -> pd.DataFrame:
    if csv_path:
        return load_games_df_from_csv(csv_path, sport=sport, season=season)
    if db_path is None:
        from data.paths import db_path_for

        db_path = db_path_for(sport, season)
    return load_games_df_from_db(db_path, sport=sport, season=season)


def _resolve_ml_models(models: Iterable[str] | None) -> list[str]:
    if models:
        model_list = [normalize_model_name(m) for m in models if m.strip()]
    else:
        available = set(list_models())
        model_list = [m for m in DEFAULT_ML_MODELS if m in available]
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
