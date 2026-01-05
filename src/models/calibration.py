"""Calibration helpers for weighted regression and variance modeling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

ABS_RESID_TO_SD = math.sqrt(math.pi / 2.0)
MIN_SD = 1e-3


def resolve_fit_end_date(games_df: pd.DataFrame) -> pd.Timestamp | None:
    """Resolve the end date used for recency weighting."""
    if "date" not in games_df.columns:
        return None
    dates = pd.to_datetime(games_df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max()


def resolve_fit_end_date_from_games(
    games: Iterable[Mapping[str, object]],
) -> pd.Timestamp | None:
    """Resolve the end date from an iterable of game mappings."""
    dates = pd.to_datetime(
        [game.get("date") for game in games],
        errors="coerce",
    ).dropna()
    if dates.empty:
        return None
    return dates.max()


def recency_weight(
    game_date: object,
    fit_end_date: pd.Timestamp | None,
    recency_lambda: float | None,
) -> float:
    """Compute a recency weight from a game date."""
    if recency_lambda is None or recency_lambda <= 0 or fit_end_date is None:
        return 1.0
    dt = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(dt):
        return 1.0
    days_ago = max(0.0, (fit_end_date - dt).days)
    return math.exp(-recency_lambda * days_ago)


def weighted_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve weighted least squares with a stable solver."""
    if weights is None:
        coeffs, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        return coeffs

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size != target.size:
        raise ValueError("weights must be a 1D array aligned with target.")
    xtwx = matrix.T @ (weights[:, None] * matrix)
    xtwy = matrix.T @ (weights * target)
    try:
        return np.linalg.solve(xtwx, xtwy)
    except np.linalg.LinAlgError:
        sqrt_w = np.sqrt(weights)
        matrix_w = matrix * sqrt_w[:, None]
        target_w = target * sqrt_w
        coeffs, *_ = np.linalg.lstsq(matrix_w, target_w, rcond=None)
        return coeffs


def weighted_rmse(residuals: np.ndarray, weights: np.ndarray | None) -> float:
    """Compute weighted RMSE for residuals."""
    if residuals.size == 0:
        return 0.0
    if weights is None:
        return float(np.sqrt(np.mean(residuals**2)))
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float(np.sqrt(np.mean(residuals**2)))
    return float(np.sqrt(np.sum(weights * residuals**2) / weight_sum))


@dataclass(frozen=True)
class ConditionalSDModel:
    intercept: float
    slope: float
    scale: float = ABS_RESID_TO_SD
    min_sd: float = MIN_SD

    def predict(self, pred_margin: float) -> float:
        abs_margin = abs(float(pred_margin))
        abs_resid = max(0.0, self.intercept + self.slope * abs_margin)
        sd = abs_resid * self.scale
        return max(self.min_sd, sd)


def fit_conditional_sd(
    predicted: np.ndarray,
    residuals: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> ConditionalSDModel | None:
    """Fit a conditional SD model from predicted margins and residuals."""
    if predicted.size < 2 or residuals.size != predicted.size:
        return None
    design = np.column_stack([np.ones_like(predicted), np.abs(predicted)])
    target = np.abs(residuals)
    coeffs = weighted_least_squares(design, target, weights=weights)
    return ConditionalSDModel(intercept=float(coeffs[0]), slope=float(coeffs[1]))


def fit_win_prob_bias(
    spreads: Iterable[float],
    outcomes: Iterable[int],
    *,
    win_prob_k: float,
    weights: Iterable[float] | None = None,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> float:
    """Fit a win-prob bias term with fixed logistic scale."""
    if win_prob_k <= 0:
        return 0.0
    spreads_arr = np.asarray(list(spreads), dtype=float)
    outcomes_arr = np.asarray(list(outcomes), dtype=float)
    if spreads_arr.size == 0 or outcomes_arr.size == 0:
        return 0.0
    if weights is None:
        weights_arr = np.ones_like(outcomes_arr)
    else:
        weights_arr = np.asarray(list(weights), dtype=float)
        if weights_arr.size != outcomes_arr.size:
            raise ValueError("weights must align with outcomes.")
    alpha = 0.0
    for _ in range(max_iter):
        z = spreads_arr / win_prob_k - alpha
        p = 1.0 / (1.0 + np.exp(z))
        grad = float(np.sum(weights_arr * (outcomes_arr - p)))
        hess = float(np.sum(weights_arr * p * (1.0 - p)))
        if hess <= 0:
            break
        step = grad / hess
        alpha += step
        if abs(step) < tol:
            break
    return float(alpha * win_prob_k)


def align_spread_with_margin(pred_margin: float, spread: float) -> float:
    """Ensure spread sign does not contradict the predicted margin sign."""
    if pred_margin > 0 and spread > 0:
        return 0.0
    if pred_margin < 0 and spread < 0:
        return 0.0
    return spread
