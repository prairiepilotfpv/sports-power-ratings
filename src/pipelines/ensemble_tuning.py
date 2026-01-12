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
from ensemble.io import load_ml_weights
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
    dataset = build_ml_ensemble_dataset(
        sport=sport,
        season=season,
        start_date=start_date,
        end_date=end_date,
        models=models,
        csv_path=csv_path,
        db_path=db_path,
    )
    weights = optimize_ensemble_weights(dataset.prob_matrix, dataset.targets, dataset.models)
    artifact_path = save_ensemble_weights(
        sport=sport,
        season=season,
        ensemble_id=ensemble_id,
        market="ML",
        start_date=start_date,
        end_date=end_date,
        models=dataset.models,
        weights=weights,
    )
    return EnsembleTuningResult(
        weights=weights,
        artifact_path=artifact_path,
        games=int(len(dataset.targets)),
        models=list(dataset.models),
    )


@dataclass(frozen=True)
class EnsembleDataset:
    prob_matrix: np.ndarray
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
    model_list = _resolve_ml_models(models)
    games_df = _load_games_df(sport, season, csv_path=csv_path, db_path=db_path)

    preds_by_model: dict[str, pd.DataFrame] = {}
    for model_name in model_list:
        model_cls = get_backtest_model(model_name)
        outputs = run_backtest(
            model_factory=lambda cls=model_cls: cls(),
            games_df=games_df,
            start_date=start_date,
            end_date=end_date,
            model_name=model_name,
        )
        preds = _prepare_predictions(outputs.predictions, model_name=model_name)
        preds_by_model[model_name] = preds

    aligned = _align_predictions(preds_by_model)
    prob_matrix = aligned.pivot_table(
        index="game_key", columns="model_name", values="p_home_win", aggfunc="mean"
    )
    prob_matrix = prob_matrix[model_list]
    targets = aligned.drop_duplicates("game_key")[["game_key", "home_win"]].set_index("game_key")

    prob_values = prob_matrix.to_numpy(dtype=float)
    target_values = targets.loc[prob_matrix.index, "home_win"].to_numpy(dtype=float)

    if prob_values.size == 0:
        raise ValueError("No overlapping predictions found across models.")

    return EnsembleDataset(
        prob_matrix=prob_values,
        targets=target_values,
        models=model_list,
        game_keys=[str(key) for key in prob_matrix.index.tolist()],
    )


def optimize_ensemble_weights(
    prob_matrix: np.ndarray, targets: np.ndarray, models: Iterable[str]
) -> dict[str, float]:
    model_list = list(models)
    n_models = len(model_list)
    if n_models == 0:
        raise ValueError("No models provided for ensemble optimization.")
    if n_models == 1:
        return {model_list[0]: 1.0}

    weights = _optimize_simplex(
        prob_matrix=prob_matrix,
        targets=targets,
        max_iter=5000,
        learning_rate=0.2,
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
) -> Path:
    spec = get_market_spec(market)
    out_path = spec.ensemble_weights_path(sport, season, ensemble_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ensemble_id": ensemble_id,
        "market": str(market),
        "objective": "log_loss",
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

    probs = np.clip(dataset.prob_matrix @ weight_vec, 1e-6, 1 - 1e-6)
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


def _prepare_predictions(pred_df: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    df = pred_df.copy()
    df["model_name"] = model_name
    df["game_key"] = _build_game_key(df)
    df = df.dropna(subset=["p_home_win", "home_win", "game_key"])
    return df[["game_key", "game_id", "model_name", "p_home_win", "home_win"]]


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


def _optimize_simplex(
    *,
    prob_matrix: np.ndarray,
    targets: np.ndarray,
    max_iter: int,
    learning_rate: float,
    tol: float,
) -> np.ndarray:
    n_models = prob_matrix.shape[1]
    weights = np.full(n_models, 1.0 / n_models, dtype=float)
    best_weights = weights.copy()
    best_loss = _log_loss(prob_matrix @ weights, targets)
    lr = float(learning_rate)

    for _ in range(max_iter):
        probs = np.clip(prob_matrix @ weights, 1e-6, 1 - 1e-6)
        grad = _log_loss_grad(prob_matrix, targets, probs)
        candidate = _project_to_simplex(weights - lr * grad)
        loss = _log_loss(prob_matrix @ candidate, targets)
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
