"""Calibration helpers for weighted regression and variance modeling."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from config import (
    DEFAULT_MARGIN_SD_FALLBACK,
    LEAGUE_MARGIN_SD_DEFAULT,
    MARGIN_SD_GUARDRAIL_MAX,
    MARGIN_SD_GUARDRAIL_MIN,
)

ABS_RESID_TO_SD = math.sqrt(math.pi / 2.0)
# Floor for predicted standard deviation to avoid unrealistically tiny
# values (e.g. 0.001) which can cause downstream logic and warnings
# to behave incorrectly. We align this with the global guardrail to
# avoid silently emitting implausible variances.
MIN_SD = MARGIN_SD_GUARDRAIL_MIN

logger = logging.getLogger(__name__)


def guardrail_margin_sd(
    raw_sd: float | None,
    *,
    fallback_sd: float = LEAGUE_MARGIN_SD_DEFAULT,
    guardrail_min: float = MARGIN_SD_GUARDRAIL_MIN,
    guardrail_max: float | None = MARGIN_SD_GUARDRAIL_MAX,
) -> tuple[float, str | None]:
    """Clamp or replace a margin SD with sane bounds.

    Returns a tuple of (applied_sd, reason) where reason is None when no
    guardrail was applied.
    """
    reason: str | None = None
    safe_fallback = max(fallback_sd, guardrail_min)
    if guardrail_max is not None:
        safe_fallback = min(safe_fallback, guardrail_max)

    if raw_sd is None or not math.isfinite(float(raw_sd)) or float(raw_sd) <= 0:
        return safe_fallback, "sd_nonfinite_or_missing"

    applied = float(raw_sd)
    if applied < guardrail_min:
        applied = safe_fallback
        reason = "sd_below_guardrail"
    if guardrail_max is not None and applied > guardrail_max:
        applied = guardrail_max
        reason = reason or "sd_above_guardrail"
    return applied, reason


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
        try:
            coeffs, *_ = np.linalg.lstsq(matrix, target, rcond=None)
            return coeffs
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse if SVD does not converge
            try:
                return np.linalg.pinv(matrix) @ target
            except Exception:
                # As a last resort, use tiny ridge regularization
                lam = 1e-8
                mtm = matrix.T @ matrix
                n = mtm.shape[0]
                return np.linalg.solve(mtm + lam * np.eye(n), matrix.T @ target)

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
        try:
            coeffs, *_ = np.linalg.lstsq(matrix_w, target_w, rcond=None)
            return coeffs
        except np.linalg.LinAlgError:
            try:
                return np.linalg.pinv(matrix_w) @ target_w
            except Exception:
                lam = 1e-8
                mtm = matrix_w.T @ matrix_w
                n = mtm.shape[0]
                return np.linalg.solve(mtm + lam * np.eye(n), matrix_w.T @ target_w)


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
    fallback_sd: float = LEAGUE_MARGIN_SD_DEFAULT
    n_samples: int = 0
    residual_min: float | None = None
    residual_max: float | None = None

    def predict(
        self,
        pred_margin: float,
        *,
        guardrail_min: float | None = None,
        guardrail_max: float | None = None,
        fallback_sd: float | None = None,
        logger_override: logging.Logger | None = None,
        log_context: dict[str, Any] | None = None,
        debug_assert: bool = False,
    ) -> float:
        abs_margin = abs(float(pred_margin))
        abs_resid = max(0.0, self.intercept + self.slope * abs_margin)
        sd = abs_resid * self.scale
        raw_sd = float(sd)
        # If the caller did not request any guardrails (all three args None),
        # return the raw SD so callers that expect the unconstrained model
        # behavior (e.g., unit tests, internal calibration) receive it.
        if guardrail_min is None and guardrail_max is None and fallback_sd is None:
            return raw_sd

        applied, reason = guardrail_margin_sd(
            raw_sd,
            fallback_sd=fallback_sd or self.fallback_sd,
            guardrail_min=guardrail_min or self.min_sd,
            guardrail_max=guardrail_max,
        )

        if reason and (logger_override or logger).isEnabledFor(logging.DEBUG):
            active_logger = logger_override or logger
            ctx = {
                "reason": reason,
                "raw_margin_sd": raw_sd,
                "applied_margin_sd": applied,
                "sd_sample_size": self.n_samples,
                "residual_min": self.residual_min,
                "residual_max": self.residual_max,
                "pred_margin": pred_margin,
            }
            if log_context:
                ctx.update(log_context)
            parts = [
                f"reason={ctx.get('reason')}",
                f"raw_sd={ctx.get('raw_margin_sd')}",
                f"applied_sd={ctx.get('applied_margin_sd')}",
                f"date={ctx.get('date')}",
                f"game_id={ctx.get('game_id')}",
                f"home={ctx.get('home_team')}",
                f"away={ctx.get('away_team')}",
                f"n={ctx.get('sd_sample_size')}",
                f"resid_min={ctx.get('residual_min')}",
                f"resid_max={ctx.get('residual_max')}",
            ]
            message = "margin_sd guardrail applied; " + ", ".join(
                str(part) for part in parts if part is not None
            )
            active_logger.debug(
                message,
                extra=ctx,
            )
        if debug_assert and applied < (guardrail_min or self.min_sd):
            raise AssertionError(
                f"margin_sd guardrail violated: applied={applied} raw={raw_sd}"
            )
        return applied


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
    return ConditionalSDModel(
        intercept=float(coeffs[0]),
        slope=float(coeffs[1]),
        min_sd=MARGIN_SD_GUARDRAIL_MIN,
        fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
        n_samples=int(predicted.size),
        residual_min=float(np.min(residuals)) if residuals.size else None,
        residual_max=float(np.max(residuals)) if residuals.size else None,
    )


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
