"""TOOR (Team OLS Optimized Rating) model."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from math import log
from typing import Any, Iterable, Mapping

import numpy as np

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

    def fit(self, games: Iterable[Mapping[str, Any]]) -> None:
        teams: list[str] = []
        seen: set[str] = set()
        samples: list[tuple[str, str, float, float]] = []
        weights: list[float] = []
        fit_end_date = resolve_fit_end_date_from_games(games)

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
        learn_home_advantage: bool = False,
        conditional_sd: bool = False,
        winprob_bias: float = 0.0,
        learn_winprob_bias: bool = False,
    ) -> None:
        """Initialize the backtest model.

        recency_lambda: None disables recency weighting (default = current behavior).
        conditional_sd: when False, uses the constant error_term for margin SDs.
        learn_winprob_bias: when False, uses the provided winprob_bias as-is.
        """
        self._max_iter = max_iter
        self._tol = tol
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
            model_version="1.0",
            params={
                "max_iter": self._max_iter,
                "tol": self._tol,
                "recency_lambda": self._recency_lambda,
                "learn_home_advantage": self._learn_home_advantage,
                "conditional_sd": self._conditional_sd,
                "winprob_bias": self._win_prob_bias,
                "learn_winprob_bias": self._learn_winprob_bias,
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

    def fit(self, games_df: Any) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        games = games_df.to_dict(orient="records")
        # Fit internal OLS team strengths (signed, mean-centered)
        self._rating_model.fit(games)

        design_matrix: list[list[float]] = []
        margins: list[float] = []
        totals: list[float] = []
        weights: list[float] = []
        win_prob_samples: list[tuple[float, int]] = []
        win_prob_spreads: list[float] = []
        win_prob_outcomes: list[int] = []
        residuals_arr = np.array([], dtype=float)

        fit_end_date = resolve_fit_end_date(games_df)
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
            weights.append(
                recency_weight(game.get("date"), fit_end_date, self._recency_lambda)
            )

        # compute recency-weighted league total mean/sd for TOTAL head
        if totals:
            w = np.asarray(weights, dtype=float) if weights else None
            tot_arr = np.asarray(totals, dtype=float)
            if w is None:
                self._total_mean = float(np.mean(tot_arr))
                self._total_sd = float(np.std(tot_arr, ddof=0))
            else:
                denom = float(np.sum(w)) if float(np.sum(w)) > 0 else None
                if denom:
                    self._total_mean = float(np.sum(w * tot_arr) / np.sum(w))
                    residuals = tot_arr - self._total_mean
                    self._total_sd = float(weighted_rmse(residuals, w))
                else:
                    self._total_mean = float(np.mean(tot_arr))
                    self._total_sd = float(np.std(tot_arr, ddof=0))

        if design_matrix:
            matrix = np.asarray(design_matrix, dtype=float)
            target = np.asarray(margins, dtype=float)
            weight_arr = np.asarray(weights, dtype=float)

            if self._learn_home_advantage:
                # fit HFA together with strength coefficients
                coeffs = weighted_least_squares(matrix, target, weights=weight_arr)
                predictions = matrix @ coeffs
                residuals = predictions - target
                residuals_arr = np.asarray(residuals, dtype=float)
                error_term = weighted_rmse(residuals_arr, weight_arr)

                self._coefficients = ToorCoefficients(
                    home_advantage=float(coeffs[0]),
                    home_coeff=float(coeffs[1]),
                    away_coeff=float(coeffs[2]),
                    error_term=error_term,
                )

                if self._conditional_sd:
                    self._conditional_sd_model = fit_conditional_sd(
                        predictions, residuals_arr, weights=weight_arr
                    )
                else:
                    self._conditional_sd_model = None

                predicted_margins_iter = zip(predictions, target)
            else:
                # use fixed HFA and fit only strength coefficients
                fixed_hfa = (
                    DEFAULT_COEFFICIENTS.home_advantage
                )
                hfa_col = matrix[:, 0]
                strength_matrix = matrix[:, 1:]
                # subtract fixed HFA contribution from the target
                y_adj = target - (fixed_hfa * hfa_col)

                coeffs_strength = weighted_least_squares(
                    strength_matrix, y_adj, weights=weight_arr
                )
                # reconstruct full predictions to compute residuals / error term
                predictions_full = (strength_matrix @ coeffs_strength) + (fixed_hfa * hfa_col)
                residuals = predictions_full - target
                residuals_arr = np.asarray(residuals, dtype=float)
                error_term = weighted_rmse(residuals_arr, weight_arr)

                self._coefficients = ToorCoefficients(
                    home_advantage=float(fixed_hfa),
                    home_coeff=float(coeffs_strength[0]),
                    away_coeff=float(coeffs_strength[1]),
                    error_term=error_term,
                )

                if self._conditional_sd:
                    # fit conditional sd on the reconstructed predictions
                    self._conditional_sd_model = fit_conditional_sd(
                        predictions_full, residuals_arr, weights=weight_arr
                    )
                else:
                    self._conditional_sd_model = None

                predicted_margins_iter = zip(predictions_full, target)

            # populate win-prob calibration samples from whichever prediction stream was used
            for predicted_margin, actual_margin in predicted_margins_iter:
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

    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        predictions: list[GamePrediction] = []
        coefficients = self._coefficients
        win_prob_k = self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K
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
            home_advantage = 0.0 if neutral else 1.0

            # Use signed/team-indicator strengths from the internal OLS fit
            strengths = self._rating_model.signed_strengths()
            home_strength = float(strengths.get(home, 0.0))
            away_strength = float(strengths.get(away, 0.0))

            pred_margin = (
                coefficients.home_advantage * home_advantage
                + coefficients.home_coeff * home_strength
                + coefficients.away_coeff * away_strength
            )

            # TOTAL head: provide league total mean/sd (computed at fit)
            pred_total = (
                float(self._total_mean)
                if self._total_mean is not None
                else DEFAULT_TOTAL_MEAN_FALLBACK
            )
            total_sd = (
                float(self._total_sd) if self._total_sd is not None else None
            )

            # Guardrail total_sd for small/unreliable samples to avoid validator failures
            sport = row.get("sport") if isinstance(row, dict) else None
            cfg = get_validation_config(sport)
            if total_sd is None or not (cfg.total_sd_min <= total_sd <= cfg.total_sd_max):
                total_sd = float(DEFAULT_TOTAL_SD_FALLBACK) if DEFAULT_TOTAL_SD_FALLBACK is not None else float(20.0)

            # pred_total is guaranteed numeric (league-level fallback); compute projected scores
            projected_home = 0.5 * (pred_total + pred_margin)
            projected_away = 0.5 * (pred_total - pred_margin)

            projected_spread = -pred_margin
            adjusted_spread = align_spread_with_margin(
                pred_margin, projected_spread - self._win_prob_bias
            )
            p_home_win = logistic_win_prob(adjusted_spread, win_prob_k)
            guardrail_context = {
                "model_id": self.model_id,
                "game_id": row.get("game_id") or f"{row['date']}_{home}_{away}",
                "date": str(row.get("date")),
                "home_team": home,
                "away_team": away,
            }

            if self._conditional_sd_model is not None:
                sport = row.get("sport") if isinstance(row, dict) else None
                cfg = get_validation_config(sport)
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
                sport = row.get("sport") if isinstance(row, dict) else None
                cfg = get_validation_config(sport)
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

            game_id = row.get("game_id") or f"{row['date']}_{home}_{away}"
            predictions.append(
                GamePrediction(
                    game_id=str(game_id),
                    date=str(row["date"]),
                    home_team=home,
                    away_team=away,
                    p_home_win=p_home_win,
                    win_prob_dist=win_prob_dist,
                    pred_margin=pred_margin,
                    pred_total=pred_total,
                    margin_sd=margin_sd,
                    total_sd=total_sd,
                    total_mean=pred_total,
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
                        "projected_home_score": projected_home,
                        "projected_away_score": projected_away,
                        "projected_total": pred_total,
                        "total_mean": pred_total,
                        "total_sd": total_sd,
                        "model_p_home_win": p_home_win,
                        "normal_p_home_win": None,
                        "win_prob_source": "logistic",
                    },
                )
            )
        return predictions
