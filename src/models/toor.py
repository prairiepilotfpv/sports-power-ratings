"""TOOR (Team OLS Optimized Rating) model."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from math import log
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import (
    DEFAULT_MARGIN_SD_FALLBACK,
    DEFAULT_WIN_PROB_K,
    DEFAULT_TOTAL_SD_FALLBACK,
    DEFAULT_TOTAL_MEAN_FALLBACK,
    LEAGUE_MARGIN_SD_DEFAULT,
    MARGIN_SD_GUARDRAIL_MIN,
    MARGIN_SD_GUARDRAIL_MAX,
)
from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns
from models.calibration import (
    ConditionalSDModel,
    align_spread_with_margin,
    guardrail_margin_sd,
    fit_conditional_sd,
    fit_win_prob_bias,
    recency_weight,
    resolve_fit_end_date,
    resolve_fit_end_date_from_games,
    weighted_least_squares,
    weighted_rmse,
)
from eval.validation import get_validation_config
from pipelines.projections import (
    fit_win_prob_scale,
    logistic_win_prob,
    win_prob_distribution,
)


@dataclass
class ToorCoefficients:
    home_advantage: float
    home_coeff: float
    away_coeff: float
    error_term: float


DEFAULT_COEFFICIENTS = ToorCoefficients(
    home_advantage=3.362,
    home_coeff=17.373,
    away_coeff=-14.855,
    error_term=31.155,
)

_LOG = logging.getLogger(__name__)


class TOORPowerRating:
    """Power ratings fit via OLS on game margins."""

    def __init__(
        self,
        *,
        max_iter: int = 500,
        tol: float = 1e-8,
        recency_lambda: float | None = None,
    ) -> None:
        """Initialize the TOOR ratings.

        recency_lambda: None disables recency weighting (default = current behavior).
        """
        self.model_id = "toor"
        self.model_version = "1.0"
        self.params = {
            "max_iter": max_iter,
            "tol": tol,
            "recency_lambda": recency_lambda,
        }
        self._ratings: dict[str, float] = {}
        self._recency_lambda = recency_lambda

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_version=self.model_version,
            params=self.params,
            supports_margin=True,
            supports_total=False,
            supports_win_prob=True,
            role="primary",
            ensemble_weight=1.0,
        )

    def fit(self, games: Iterable[Mapping[str, Any]], *, fit_end_date: pd.Timestamp | None = None) -> None:
        teams: list[str] = []
        seen: set[str] = set()
        samples: list[tuple[str, str, float, float]] = []
        weights: list[float] = []
        # Use explicit fit_end_date when provided; otherwise fall back to resolving from the games iterable
        fit_end_date = fit_end_date if fit_end_date is not None else resolve_fit_end_date_from_games(games)

        for game in games:
            home = str(game.get("home_team", "")).strip()
            away = str(game.get("away_team", "")).strip()
            if not home or not away:
                continue
            try:
                margin = float(game.get("home_score")) - float(game.get("away_score"))
            except Exception:
                continue

            if home not in seen:
                seen.add(home)
                teams.append(home)
            if away not in seen:
                seen.add(away)
                teams.append(away)

            neutral_raw = game.get("neutral", False)
            neutral = (
                False
                if isinstance(neutral_raw, float) and np.isnan(neutral_raw)
                else bool(neutral_raw)
            )
            home_advantage = 0.0 if neutral else 1.0

            samples.append((home, away, margin, home_advantage))
            weights.append(
                recency_weight(game.get("date"), fit_end_date, self._recency_lambda)
            )

        if len(teams) < 2 or not samples:
            self._ratings = {}
            return

        index = {team: idx for idx, team in enumerate(teams)}
        feature_count = len(teams) + 1
        design_matrix: list[list[float]] = []
        margins: list[float] = []
        design_weights: list[float] = []
        for (home, away, margin, home_advantage), weight in zip(
            samples, weights, strict=False
        ):
            row = [0.0] * feature_count
            row[index[home]] = 1.0
            row[index[away]] = -1.0
            row[-1] = home_advantage
            design_matrix.append(row)
            margins.append(margin)
            design_weights.append(weight)

        if len(design_matrix) < 2:
            self._ratings = {}
            return

        matrix = np.asarray(design_matrix, dtype=float)
        target = np.asarray(margins, dtype=float)
        # Apply recency weighting once in the core OLS step
        weight_arr = np.asarray(design_weights, dtype=float) if design_weights else None
        coeffs = weighted_least_squares(matrix, target, weights=weight_arr)
        team_coeffs = coeffs[: len(teams)]
        mean_coeff = float(np.mean(team_coeffs)) if team_coeffs.size else 0.0
        centered = team_coeffs - mean_coeff
        # Stabilize exponentiation to avoid overflow for large coefficient magnitudes.
        # Subtract the maximum centered value so the largest exponent is 0, then
        # exponentiate and normalize to keep ratings on a reasonable scale.
        if centered.size:
            max_center = float(np.max(centered))
            stabilized = centered - max_center
            # clip to a safe range for exp to avoid any numerical issues
            stabilized = np.clip(stabilized, -700.0, 700.0)
            exp_vals = np.exp(stabilized)
            # normalize so ratings have mean 1.0 (keeps scale comparable across fits)
            mean_exp = float(np.mean(exp_vals)) if exp_vals.size else 1.0
            normalized = exp_vals / mean_exp if mean_exp != 0.0 else exp_vals
            # positive normalized ratings (for display/rankings)
            self._ratings = {team: float(normalized[index[team]]) for team in teams}
            # preserve signed, mean-centered strengths for MOV regression
            # (these retain sign and are mean-centered; may be negative)
            self._signed_strengths = {team: float(centered[index[team]]) for team in teams}
        else:
            self._ratings = {}
            self._signed_strengths = {}

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(self._ratings.items(), key=lambda item: item[1], reverse=True)

    def signed_strengths(self) -> dict[str, float]:
        """Return signed, mean-centered team strengths suitable for MOV regression.

        These are the centered coefficients from the team-indicator OLS (can be
        negative). If unavailable, returns an empty dict.
        """
        return getattr(self, "_signed_strengths", {}).copy()


class TOORModel(BaseModel):
    """TOOR backtest model: OLS-mapped team strengths -> MOV, with logistic win-prob and TOTAL head.

    Self-contained (no cross-model coupling): fits signed, mean-centered team strengths
    via team-indicator OLS and then fits an MOV regression (home_adv, home/away strengths).
    """

    def __init__(
        self,
        *,
        max_iter: int = 500,
        tol: float = 1e-8,
        recency_lambda: float | None = None,
        learn_home_advantage: bool = True,
        conditional_sd: bool = False,
        winprob_bias: float = 0.0,
        learn_winprob_bias: bool = False,
        optimizer: str = "scipy",
        initial_home_adv: float | None = None,
        initial_home_coeff: float | None = None,
        initial_away_coeff: float | None = None,
    ) -> None:
        """Initialize the backtest model.

        recency_lambda: None disables recency weighting (default = current behavior).
        conditional_sd: when False, uses the constant error_term for margin SDs.
        learn_winprob_bias: when False, uses the provided winprob_bias as-is.
        optimizer: "scipy" for iterative optimization (tunable), "ols" for closed-form (legacy).
        initial_home_adv: Initial guess for home advantage (default: 3.362).
        initial_home_coeff: Initial guess for home coefficient (default: 17.373).
        initial_away_coeff: Initial guess for away coefficient (default: -14.855).
        """
        self._max_iter = max_iter
        self._tol = tol
        self._optimizer = optimizer
        self._optimizer = optimizer
        # Store initial guesses for scipy optimization
        self._initial_home_adv = initial_home_adv if initial_home_adv is not None else DEFAULT_COEFFICIENTS.home_advantage
        self._initial_home_coeff = initial_home_coeff if initial_home_coeff is not None else DEFAULT_COEFFICIENTS.home_coeff
        self._initial_away_coeff = initial_away_coeff if initial_away_coeff is not None else DEFAULT_COEFFICIENTS.away_coeff
        # self-contained OLS-based team strengths (no cross-model coupling)
        self._rating_model = TOORPowerRating(
            max_iter=max_iter, recency_lambda=recency_lambda
        )
        self._coefficients = DEFAULT_COEFFICIENTS
        self._win_prob_k = DEFAULT_WIN_PROB_K
        self._recency_lambda = recency_lambda
        self._learn_home_advantage = learn_home_advantage
        self._conditional_sd = conditional_sd
        self._conditional_sd_model: ConditionalSDModel | None = None
        self._win_prob_bias = float(winprob_bias)
        self._learn_winprob_bias = learn_winprob_bias
        # league-level total statistics (computed during fit)
        self._total_mean: float | None = None
        self._total_sd: float | None = None
        self._sd_fit_stats: dict[str, Any] = {
            "n": 0,
            "residual_min": None,
            "residual_max": None,
        }
        debug_flag = os.getenv("TOOR_MARGIN_SD_ASSERT", "").lower()
        self._debug_assert = debug_flag in {"1", "true", "yes", "on"}

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="toor",
            model_version="1.1",
            params={
                "max_iter": self._max_iter,
                "tol": self._tol,
                "recency_lambda": self._recency_lambda,
                "learn_home_advantage": self._learn_home_advantage,
                "conditional_sd": self._conditional_sd,
                "winprob_bias": self._win_prob_bias,
                "learn_winprob_bias": self._learn_winprob_bias,
                "optimizer": self._optimizer,
                "initial_home_adv": self._initial_home_adv,
                "initial_home_coeff": self._initial_home_coeff,
                "initial_away_coeff": self._initial_away_coeff,
            },
            supports_margin=True,
            supports_total=True,
            supports_win_prob=True,
            role="primary",
            ensemble_weight=1.0,
        )

    def _capture_sd_stats(self, residuals: np.ndarray) -> None:
        if residuals.size:
            self._sd_fit_stats = {
                "n": int(residuals.size),
                "residual_min": float(np.min(residuals)),
                "residual_max": float(np.max(residuals)),
            }
        else:
            self._sd_fit_stats = {
                "n": 0,
                "residual_min": None,
                "residual_max": None,
            }

    def fit(self, games_df: Any, *, fit_end_date: pd.Timestamp | None = None) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        games = games_df.to_dict(orient="records")
        # Fit internal OLS team strengths (signed, mean-centered)
        # Use explicit fit_end_date when provided; otherwise fall back within rating model
        self._rating_model.fit(games, fit_end_date=fit_end_date)

        design_matrix: list[list[float]] = []
        margins: list[float] = []
        totals: list[float] = []
        # Recency weighting is applied in TOORPowerRating.fit; do not apply again here.
        win_prob_samples: list[tuple[float, int]] = []
        win_prob_spreads: list[float] = []
        win_prob_outcomes: list[int] = []
        residuals_arr = np.array([], dtype=float)

        # Use explicit fit_end_date when provided; otherwise fall back to resolving from the DataFrame
        fit_end_date = fit_end_date if fit_end_date is not None else resolve_fit_end_date(games_df)
        strengths = self._rating_model.signed_strengths()

        for game in games:
            home = str(game.get("home_team", "")).strip()
            away = str(game.get("away_team", "")).strip()
            if not home or not away:
                continue
            try:
                home_score = float(game.get("home_score"))
                away_score = float(game.get("away_score"))
                margin = home_score - away_score
            except Exception:
                continue

            neutral_raw = game.get("neutral", False)
            neutral = (
                False
                if isinstance(neutral_raw, float) and np.isnan(neutral_raw)
                else bool(neutral_raw)
            )
            home_advantage = 0.0 if neutral else 1.0

            home_strength = float(strengths.get(home, 0.0))
            away_strength = float(strengths.get(away, 0.0))

            design_matrix.append([home_advantage, home_strength, away_strength])
            margins.append(margin)
            totals.append(home_score + away_score)
            # No additional recency weighting here.

        # Compute unweighted league total mean/sd for TOTAL head
        if totals:
            tot_arr = np.asarray(totals, dtype=float)
            self._total_mean = float(np.mean(tot_arr))
            self._total_sd = float(np.std(tot_arr, ddof=0))

        if design_matrix:
            matrix = np.asarray(design_matrix, dtype=float)
            target = np.asarray(margins, dtype=float)
            # Unweighted calibration: keep a single recency application (in TOORPowerRating.fit)
            weight_arr = None

            # Store matrix and target for scipy optimization
            self._fit_matrix = matrix
            self._fit_target = target
            self._fit_weights = weight_arr

            # Choose optimization method
            if self._optimizer == "scipy":
                self._coefficients = self._fit_coefficients_scipy(
                    matrix, target, weight_arr
                )
            else:  # "ols" or legacy
                self._coefficients = self._fit_coefficients_ols(
                    matrix, target, weight_arr
                )

            # Compute predictions and residuals
            predictions = (
                self._coefficients.home_advantage * matrix[:, 0]
                + self._coefficients.home_coeff * matrix[:, 1]
                + self._coefficients.away_coeff * matrix[:, 2]
            )
            residuals = target - predictions
            residuals_arr = np.asarray(residuals, dtype=float)

            # Fit conditional SD model if enabled
            if self._conditional_sd:
                self._conditional_sd_model = fit_conditional_sd(
                    predictions, residuals_arr, weights=weight_arr
                )
            else:
                self._conditional_sd_model = None

            # populate win-prob calibration samples
            for predicted_margin, actual_margin in zip(predictions, target):
                if actual_margin == 0:
                    continue
                projected_spread = -float(predicted_margin)
                win_prob_samples.append((projected_spread, 1 if actual_margin > 0 else 0))
                win_prob_spreads.append(projected_spread)
                win_prob_outcomes.append(1 if actual_margin > 0 else 0)

        self._capture_sd_stats(residuals_arr)

        self._win_prob_k = fit_win_prob_scale(
            win_prob_samples, default_k=DEFAULT_WIN_PROB_K
        )
        if self._learn_winprob_bias and win_prob_spreads:
            self._win_prob_bias = fit_win_prob_bias(
                win_prob_spreads,
                win_prob_outcomes,
                win_prob_k=self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K,
            )

    def _fit_coefficients_scipy(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray | None,
    ) -> ToorCoefficients:
        """Fit TOOR coefficients using scipy.optimize.minimize.
        
        This provides iterative optimization that responds to hyperparameter tuning,
        unlike the closed-form OLS solution.
        """
        
        def objective(params: np.ndarray) -> float:
            """Sum of squared errors for optimization."""
            predictions = (
                params[0] * matrix[:, 0]  # home_advantage * neutral_flag
                + params[1] * matrix[:, 1]  # home_coeff * home_strength
                + params[2] * matrix[:, 2]  # away_coeff * away_strength
            )
            residuals = target - predictions
            if weights is not None:
                return float(np.sum((residuals ** 2) * weights))
            return float(np.sum(residuals ** 2))
        
        # Initial guess using configured or default values
        if self._learn_home_advantage:
            initial_guess = np.array([
                self._initial_home_adv,
                self._initial_home_coeff,
                self._initial_away_coeff,
            ])
        else:
            # If not learning HFA, fix it and only optimize strength coefficients
            initial_guess = np.array([
                DEFAULT_COEFFICIENTS.home_advantage,  # fixed
                self._initial_home_coeff,
                self._initial_away_coeff,
            ])
        
        # Try multiple optimization methods with fallback
        methods = ["L-BFGS-B", "SLSQP"]
        result = None
        
        for method in methods:
            try:
                if not self._learn_home_advantage:
                    # Fix home advantage by using bounds that lock it
                    bounds = [
                        (DEFAULT_COEFFICIENTS.home_advantage, DEFAULT_COEFFICIENTS.home_advantage),
                        (None, None),
                        (None, None),
                    ]
                    result = minimize(
                        objective,
                        initial_guess,
                        method=method,
                        bounds=bounds,
                        options={"ftol": self._tol, "maxiter": self._max_iter},
                    )
                else:
                    result = minimize(
                        objective,
                        initial_guess,
                        method=method,
                        options={"ftol": self._tol, "maxiter": self._max_iter},
                    )
                
                if result.success:
                    _LOG.debug(
                        f"TOOR scipy optimization converged with {method}: "
                        f"nit={result.get('nit', 'N/A')}, fun={result.fun:.4f}"
                    )
                    break
            except Exception as e:
                _LOG.warning(f"TOOR scipy optimization failed with {method}: {e}")
                continue
        
        # Fall back to OLS if scipy optimization fails
        if result is None or not result.success:
            _LOG.warning(
                f"TOOR scipy optimization failed (tried {methods}), falling back to OLS"
            )
            return self._fit_coefficients_ols(matrix, target, weights)
        
        # Compute error term from optimized parameters
        predictions = (
            result.x[0] * matrix[:, 0]
            + result.x[1] * matrix[:, 1]
            + result.x[2] * matrix[:, 2]
        )
        residuals = target - predictions
        error_term = weighted_rmse(residuals, weights)
        
        return ToorCoefficients(
            home_advantage=float(result.x[0]),
            home_coeff=float(result.x[1]),
            away_coeff=float(result.x[2]),
            error_term=error_term,
        )
    
    def _fit_coefficients_ols(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray | None,
    ) -> ToorCoefficients:
        """Fit TOOR coefficients using closed-form OLS (legacy method)."""
        
        if self._learn_home_advantage:
            # fit HFA together with strength coefficients
            coeffs = weighted_least_squares(matrix, target, weights=weights)
            predictions = matrix @ coeffs
            residuals = target - predictions
            error_term = weighted_rmse(residuals, weights)
            
            return ToorCoefficients(
                home_advantage=float(coeffs[0]),
                home_coeff=float(coeffs[1]),
                away_coeff=float(coeffs[2]),
                error_term=error_term,
            )
        else:
            # use fixed HFA and fit only strength coefficients
            fixed_hfa = DEFAULT_COEFFICIENTS.home_advantage
            hfa_col = matrix[:, 0]
            strength_matrix = matrix[:, 1:]
            # subtract fixed HFA contribution from the target
            y_adj = target - (fixed_hfa * hfa_col)
            
            coeffs_strength = weighted_least_squares(
                strength_matrix, y_adj, weights=weights
            )
            # reconstruct full predictions to compute residuals / error term
            predictions_full = (strength_matrix @ coeffs_strength) + (fixed_hfa * hfa_col)
            residuals = target - predictions_full
            error_term = weighted_rmse(residuals, weights)
            
            return ToorCoefficients(
                home_advantage=float(fixed_hfa),
                home_coeff=float(coeffs_strength[0]),
                away_coeff=float(coeffs_strength[1]),
                error_term=error_term,
            )
    
    def _build_team_index(self) -> dict[str, int]:
        """Build team name → index mapping from current ratings."""
        strengths = self._rating_model.signed_strengths()
        return {team: idx for idx, team in enumerate(sorted(strengths.keys()))}
    
    def _compute_margin_predictions(
        self,
        home_strengths: np.ndarray,
        away_strengths: np.ndarray,
        neutral_flags: np.ndarray,
    ) -> np.ndarray:
        """Vectorized margin prediction from team strengths.
        
        Args:
            home_strengths: Array of home team strength values
            away_strengths: Array of away team strength values
            neutral_flags: Boolean array indicating neutral venue
            
        Returns:
            Array of predicted margins (home - away)
        """
        coefficients = self._coefficients
        home_adv_contrib = coefficients.home_advantage * (~neutral_flags).astype(float)
        return (
            home_adv_contrib
            + coefficients.home_coeff * home_strengths
            + coefficients.away_coeff * away_strengths
        )
    
    def _compute_margin_sds_vectorized(
        self, 
        pred_margins: np.ndarray,
        sport: str | None = None,
    ) -> np.ndarray:
        """Compute margin standard deviations for all predictions.
        
        Uses conditional SD model if available, otherwise constant error_term.
        """
        cfg = get_validation_config(sport)
        
        if self._conditional_sd_model is not None:
            # Vectorized conditional SD prediction
            abs_margins = np.abs(pred_margins)
            raw_sds = self._conditional_sd_model.intercept + self._conditional_sd_model.slope * abs_margins
            
            # Vectorized guardrail clamping
            sds = np.clip(raw_sds, cfg.margin_sd_min, cfg.margin_sd_max)
            
            # Replace invalid values with fallback
            sds = np.where(np.isfinite(sds), sds, LEAGUE_MARGIN_SD_DEFAULT)
            
            return sds
        else:
            # Constant error term for all predictions
            raw_sd = self._coefficients.error_term
            margin_sd, _ = guardrail_margin_sd(
                raw_sd,
                fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
                guardrail_min=cfg.margin_sd_min,
                guardrail_max=cfg.margin_sd_max,
            )
            return np.full(len(pred_margins), margin_sd)
    
    def _compute_win_probs_vectorized(
        self,
        pred_margins: np.ndarray,
        margin_sds: np.ndarray,
    ) -> tuple[np.ndarray, list[list[dict[str, float]]]]:
        """Compute win probabilities and distributions for all predictions."""
        win_prob_k = self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K
        
        # Vectorized projected spread with bias adjustment
        projected_spreads = -pred_margins
        adjusted_spreads = projected_spreads - self._win_prob_bias
        
        # Vectorized logistic win probability
        p_home_wins = logistic_win_prob(adjusted_spreads, win_prob_k)
        
        # Win probability distributions (still needs per-sample computation)
        win_prob_dists = [
            win_prob_distribution(p_home_win, win_prob_k=win_prob_k, margin_std=margin_sd)
            for p_home_win, margin_sd in zip(p_home_wins, margin_sds)
        ]
        
        return p_home_wins, win_prob_dists

    def _canonical_matchup_prediction(
        self,
        home: str,
        away: str,
        *,
        neutral: bool,
        sport: str | None,
        date: str | None,
        game_id: str | None,
    ) -> dict[str, Any]:
        coefficients = self._coefficients
        win_prob_k = self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K
        safe_game_id = (
            str(game_id)
            if game_id is not None
            else f"{date}_{home}_{away}"
            if date is not None
            else f"{home}_{away}"
        )
        safe_date = str(date) if date is not None else ""
        cfg = get_validation_config(sport)
        strengths = self._rating_model.signed_strengths()
        home_strength = float(strengths.get(home, 0.0))
        away_strength = float(strengths.get(away, 0.0))
        pred_margin = (
            coefficients.home_advantage * (0.0 if neutral else 1.0)
            + coefficients.home_coeff * home_strength
            + coefficients.away_coeff * away_strength
        )
        pred_total = (
            float(self._total_mean)
            if self._total_mean is not None
            else DEFAULT_TOTAL_MEAN_FALLBACK
        )
        total_sd = (
            float(self._total_sd) if self._total_sd is not None else None
        )
        if total_sd is None or not (cfg.total_sd_min <= total_sd <= cfg.total_sd_max):
            total_sd = (
                float(DEFAULT_TOTAL_SD_FALLBACK)
                if DEFAULT_TOTAL_SD_FALLBACK is not None
                else float(20.0)
            )
        projected_home = 0.5 * (pred_total + pred_margin)
        projected_away = 0.5 * (pred_total - pred_margin)
        projected_spread = -pred_margin
        adjusted_spread = align_spread_with_margin(
            pred_margin, projected_spread - self._win_prob_bias
        )
        p_home_win = logistic_win_prob(adjusted_spread, win_prob_k)
        guardrail_context = {
            "model_id": self.model_id,
            "game_id": safe_game_id,
            "date": safe_date,
            "home_team": home,
            "away_team": away,
        }
        if self._conditional_sd_model is not None:
            margin_sd = self._conditional_sd_model.predict(
                pred_margin,
                guardrail_min=cfg.margin_sd_min,
                guardrail_max=cfg.margin_sd_max,
                fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
                logger_override=_LOG,
                log_context=guardrail_context,
                debug_assert=self._debug_assert,
            )
        else:
            raw_sd = coefficients.error_term
            margin_sd, reason = guardrail_margin_sd(
                raw_sd,
                fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
                guardrail_min=cfg.margin_sd_min,
                guardrail_max=cfg.margin_sd_max,
            )
            if reason and _LOG.isEnabledFor(logging.DEBUG):
                stats = self._sd_fit_stats or {}
                parts = [
                    f"reason={reason}",
                    f"raw_sd={raw_sd}",
                    f"applied_sd={margin_sd}",
                    f"date={guardrail_context.get('date')}",
                    f"game_id={guardrail_context.get('game_id')}",
                    f"home={home}",
                    f"away={away}",
                    f"n={stats.get('n')}",
                    f"resid_min={stats.get('residual_min')}",
                    f"resid_max={stats.get('residual_max')}",
                ]
                message = "margin_sd guardrail applied; " + ", ".join(
                    str(part) for part in parts if part is not None
                )
                _LOG.debug(
                    message,
                    extra={
                        **guardrail_context,
                        "reason": reason,
                        "raw_margin_sd": raw_sd,
                        "applied_margin_sd": margin_sd,
                        "sd_sample_size": stats.get("n"),
                        "residual_min": stats.get("residual_min"),
                        "residual_max": stats.get("residual_max"),
                        "pred_margin": pred_margin,
                    },
                )
            if self._debug_assert and margin_sd < cfg.margin_sd_min:
                raise AssertionError(
                    f"margin_sd guardrail violated: applied={margin_sd} raw={raw_sd}"
                )
        win_prob_dist = win_prob_distribution(
            p_home_win,
            win_prob_k=win_prob_k,
            margin_std=margin_sd,
        )
        return {
            "game_id": safe_game_id,
            "date": safe_date,
            "home_team": home,
            "away_team": away,
            "p_home_win": p_home_win,
            "win_prob_dist": win_prob_dist,
            "pred_margin": pred_margin,
            "pred_total": pred_total,
            "margin_mean": pred_margin,
            "total_mean": pred_total,
            "margin_sd": margin_sd,
            "total_sd": total_sd,
            "projected_home_score": projected_home,
            "projected_away_score": projected_away,
            "projected_total": pred_total,
            "projected_win_prob": p_home_win,
            "model_p_home_win": p_home_win,
            "normal_p_home_win": None,
            "win_prob_source": "logistic",
            "margin_dist_assumption": "normal_approx",
            "logistic_home_win_prob": p_home_win,
            "win_prob_k": win_prob_k,
            "winprob_bias": self._win_prob_bias,
            "conditional_sd": self._conditional_sd,
            "conditional_sd_intercept": (
                self._conditional_sd_model.intercept
                if self._conditional_sd_model is not None
                else None
            ),
            "conditional_sd_slope": (
                self._conditional_sd_model.slope
                if self._conditional_sd_model is not None
                else None
            ),
        }

    def project_matchup(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral: bool = False,
        sport: str | None = None,
        date: str | None = None,
        game_id: str | None = None,
    ) -> dict[str, Any]:
        return self._canonical_matchup_prediction(
            home_team,
            away_team,
            neutral=neutral,
            sport=sport,
            date=date,
            game_id=game_id,
        )

    def predict(
        self, 
        upcoming_games_df: Any,
        format: str = "canonical",
        use_vectorized: bool = True,
    ) -> list[GamePrediction] | np.ndarray | pd.DataFrame:
        """Predict outcomes for upcoming games.
        
        Args:
            upcoming_games_df: DataFrame with columns: date, home_team, away_team
            format: Output format - "canonical" (GamePrediction objects), 
                    "array" (numpy arrays), "dataframe" (pandas DataFrame)
            use_vectorized: Use vectorized prediction (faster for large batches)
        
        Returns:
            List of GamePrediction objects, numpy array, or DataFrame based on format
        """
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        
        if len(upcoming_games_df) == 0:
            if format == "array":
                return np.array([]).reshape(0, 5)
            elif format == "dataframe":
                return pd.DataFrame()
            return []
        
        # Use vectorized prediction for better performance
        if use_vectorized and len(upcoming_games_df) > 1:
            return self._predict_vectorized(upcoming_games_df, format=format)
        
        # Fall back to iterative prediction for single games or if vectorization disabled
        return self._predict_iterative(upcoming_games_df, format=format)
    
    def _predict_vectorized(
        self,
        upcoming_games_df: Any,
        format: str = "canonical",
    ) -> list[GamePrediction] | np.ndarray | pd.DataFrame:
        """Vectorized prediction implementation for performance."""
        
        # Extract vectorized inputs
        home_teams = upcoming_games_df["home_team"].astype(str).str.strip().values
        away_teams = upcoming_games_df["away_team"].astype(str).str.strip().values
        neutral_raw = upcoming_games_df.get("neutral", pd.Series([False] * len(upcoming_games_df)))
        neutral_flags = neutral_raw.fillna(False).replace({np.nan: False}).astype(bool).values
        
        # Get sport for validation config (use first non-null or None)
        sport = None
        if "sport" in upcoming_games_df.columns:
            sport_series = upcoming_games_df["sport"].dropna()
            if len(sport_series) > 0:
                sport = str(sport_series.iloc[0])
        
        # Build team index and lookup strengths
        strengths = self._rating_model.signed_strengths()
        
        # Vectorized strength lookup with fallback to 0.0 for unknown teams
        home_strengths = np.array([strengths.get(h, 0.0) for h in home_teams])
        away_strengths = np.array([strengths.get(a, 0.0) for a in away_teams])
        
        # Vectorized margin predictions
        pred_margins = self._compute_margin_predictions(home_strengths, away_strengths, neutral_flags)
        
        # Vectorized total predictions (league average)
        pred_totals = np.full(
            len(upcoming_games_df), 
            self._total_mean if self._total_mean is not None else DEFAULT_TOTAL_MEAN_FALLBACK
        )
        
        # Vectorized margin SD computation
        margin_sds = self._compute_margin_sds_vectorized(pred_margins, sport=sport)
        
        # Total SD (constant league-wide)
        cfg = get_validation_config(sport)
        total_sd = self._total_sd if self._total_sd is not None else None
        if total_sd is None or not (cfg.total_sd_min <= total_sd <= cfg.total_sd_max):
            total_sd = DEFAULT_TOTAL_SD_FALLBACK if DEFAULT_TOTAL_SD_FALLBACK is not None else 20.0
        total_sds = np.full(len(upcoming_games_df), total_sd)
        
        # Vectorized win probabilities
        p_home_wins, win_prob_dists = self._compute_win_probs_vectorized(pred_margins, margin_sds)
        
        # Return based on format
        if format == "array":
            return np.column_stack([
                pred_margins,
                pred_totals,
                p_home_wins,
                margin_sds,
                total_sds,
            ])
        elif format == "dataframe":
            result_df = upcoming_games_df.copy()
            result_df["pred_margin"] = pred_margins
            result_df["pred_total"] = pred_totals
            result_df["p_home_win"] = p_home_wins
            result_df["margin_sd"] = margin_sds
            result_df["total_sd"] = total_sds
            return result_df
        else:  # format == "canonical"
            return self._format_game_predictions_vectorized(
                upcoming_games_df,
                home_teams,
                away_teams,
                neutral_flags,
                pred_margins,
                pred_totals,
                margin_sds,
                total_sds,
                p_home_wins,
                win_prob_dists,
            )
    
    def _format_game_predictions_vectorized(
        self,
        games_df: pd.DataFrame,
        home_teams: np.ndarray,
        away_teams: np.ndarray,
        neutral_flags: np.ndarray,
        pred_margins: np.ndarray,
        pred_totals: np.ndarray,
        margin_sds: np.ndarray,
        total_sds: np.ndarray,
        p_home_wins: np.ndarray,
        win_prob_dists: list[list[dict[str, float]]],
    ) -> list[GamePrediction]:
        """Convert vectorized predictions to canonical GamePrediction objects."""
        predictions = []
        coefficients = self._coefficients
        model_identity = self.metadata().identity_dict()
        win_prob_k = self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K
        
        for i in range(len(games_df)):
            row = games_df.iloc[i]
            game_id = row.get("game_id")
            if game_id is None or pd.isna(game_id):
                game_id = f"{row['date']}_{home_teams[i]}_{away_teams[i]}"
            
            predictions.append(
                GamePrediction(
                    game_id=str(game_id),
                    date=str(row.get("date", "")),
                    home_team=str(home_teams[i]),
                    away_team=str(away_teams[i]),
                    p_home_win=float(p_home_wins[i]),
                    win_prob_dist=win_prob_dists[i] if win_prob_dists else None,
                    pred_margin=float(pred_margins[i]),
                    pred_total=float(pred_totals[i]),
                    margin_sd=float(margin_sds[i]),
                    total_sd=float(total_sds[i]),
                    total_mean=float(pred_totals[i]),
                    win_prob_source="logistic",
                    margin_dist_assumption="normal_approx",
                    metadata=dict(model_identity),
                    extra={
                        "home_advantage": coefficients.home_advantage,
                        "home_coeff": coefficients.home_coeff,
                        "away_coeff": coefficients.away_coeff,
                        "error_term": coefficients.error_term,
                        "win_prob_k": win_prob_k,
                        "winprob_bias": self._win_prob_bias,
                        "conditional_sd": self._conditional_sd,
                        "conditional_sd_intercept": (
                            self._conditional_sd_model.intercept
                            if self._conditional_sd_model is not None
                            else None
                        ),
                        "conditional_sd_slope": (
                            self._conditional_sd_model.slope
                            if self._conditional_sd_model is not None
                            else None
                        ),
                        "projected_home_score": 0.5 * (pred_totals[i] + pred_margins[i]),
                        "projected_away_score": 0.5 * (pred_totals[i] - pred_margins[i]),
                        "projected_total": pred_totals[i],
                        "total_mean": pred_totals[i],
                        "total_sd": total_sds[i],
                        "model_p_home_win": p_home_wins[i],
                        "normal_p_home_win": None,
                        "win_prob_source": "logistic",
                    },
                )
            )
        return predictions
    
    def _predict_iterative(
        self,
        upcoming_games_df: Any,
        format: str = "canonical",
    ) -> list[GamePrediction] | np.ndarray | pd.DataFrame:
        """Iterative prediction implementation (legacy, fallback for single games)."""
        predictions: list[GamePrediction] = []
        coefficients = self._coefficients
        model_identity = self.metadata().identity_dict()

        for row in upcoming_games_df.to_dict(orient="records"):
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            if not home or not away:
                continue

            neutral_raw = row.get("neutral", False)
            neutral = (
                False
                if isinstance(neutral_raw, float) and np.isnan(neutral_raw)
                else bool(neutral_raw)
            )
            sport = row.get("sport") if isinstance(row, dict) else None
            canonical = self._canonical_matchup_prediction(
                home,
                away,
                neutral=neutral,
                sport=sport,
                date=row.get("date"),
                game_id=row.get("game_id"),
            )
            predictions.append(
                GamePrediction(
                    game_id=str(canonical["game_id"]),
                    date=str(canonical["date"]),
                    home_team=home,
                    away_team=away,
                    p_home_win=canonical["p_home_win"],
                    win_prob_dist=canonical["win_prob_dist"],
                    pred_margin=canonical["pred_margin"],
                    pred_total=canonical["pred_total"],
                    margin_sd=canonical["margin_sd"],
                    total_sd=canonical["total_sd"],
                    total_mean=canonical["total_mean"],
                    win_prob_source=canonical["win_prob_source"],
                    margin_dist_assumption=canonical["margin_dist_assumption"],
                    metadata=dict(model_identity),
                    extra={
                        "home_advantage": coefficients.home_advantage,
                        "home_coeff": coefficients.home_coeff,
                        "away_coeff": coefficients.away_coeff,
                        "error_term": coefficients.error_term,
                        "win_prob_k": canonical["win_prob_k"],
                        "winprob_bias": canonical["winprob_bias"],
                        "conditional_sd": canonical["conditional_sd"],
                        "conditional_sd_intercept": canonical["conditional_sd_intercept"],
                        "conditional_sd_slope": canonical["conditional_sd_slope"],
                        "projected_home_score": canonical["projected_home_score"],
                        "projected_away_score": canonical["projected_away_score"],
                        "projected_total": canonical["projected_total"],
                        "total_mean": canonical["total_mean"],
                        "total_sd": canonical["total_sd"],
                        "model_p_home_win": canonical["model_p_home_win"],
                        "normal_p_home_win": canonical["normal_p_home_win"],
                        "win_prob_source": canonical["win_prob_source"],
                    },
                )
            )
        
        # Convert to requested format
        if format == "array":
            return np.array([
                [p.pred_margin, p.pred_total, p.p_home_win, p.margin_sd, p.total_sd]
                for p in predictions
            ])
        elif format == "dataframe":
            return pd.DataFrame([
                {
                    "game_id": p.game_id,
                    "date": p.date,
                    "home_team": p.home_team,
                    "away_team": p.away_team,
                    "pred_margin": p.pred_margin,
                    "pred_total": p.pred_total,
                    "p_home_win": p.p_home_win,
                    "margin_sd": p.margin_sd,
                    "total_sd": p.total_sd,
                }
                for p in predictions
            ])
        
        return predictions
