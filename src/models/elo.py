"""Elo power rating model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isnan
from typing import Any, DefaultDict, Iterable, Mapping

import numpy as np
import pandas as pd

from config import DEFAULT_WIN_PROB_K, DEFAULT_TOTAL_SD_FALLBACK, DEFAULT_TOTAL_MEAN_FALLBACK
from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns
from models.calibration import (
    ConditionalSDModel,
    align_spread_with_margin,
    fit_conditional_sd,
    fit_win_prob_bias,
    recency_weight,
    resolve_fit_end_date,
    weighted_least_squares,
    weighted_rmse,
)
from pipelines.projections import (
    fit_win_prob_scale,
    logistic_win_prob,
    win_prob_distribution,
)


class EloPowerRating:
    """Classic Elo rating model with optional home advantage."""

    def __init__(
        self,
        *,
        k_factor: float = 20.0,
        home_advantage: float = 65.0,
        initial_rating: float = 1500.0,
        min_rating: float = 1.0,
    ) -> None:
        self.model_id = "elo"
        self.model_version = "1.0"
        self.params = {
            "k_factor": k_factor,
            "home_advantage": home_advantage,
            "initial_rating": initial_rating,
            "min_rating": min_rating,
        }
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.initial_rating = float(initial_rating)
        self.min_rating = float(min_rating)
        self._ratings: DefaultDict[str, float] = defaultdict(
            lambda: self.initial_rating
        )

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
        """Fit Elo ratings from game results."""
        ratings: DefaultDict[str, float] = defaultdict(lambda: self.initial_rating)

        for game in games:
            home = str(game.get("home_team", "")).strip()
            away = str(game.get("away_team", "")).strip()
            if not home or not away:
                continue
            try:
                home_score = float(game.get("home_score"))
                away_score = float(game.get("away_score"))
            except Exception:
                continue

            if home_score == away_score:
                outcome_home = 0.5
            elif home_score > away_score:
                outcome_home = 1.0
            else:
                outcome_home = 0.0
            outcome_away = 1.0 - outcome_home

            neutral_raw = game.get("neutral", False)
            if isinstance(neutral_raw, float) and isnan(neutral_raw):
                neutral = False
            else:
                neutral = False if neutral_raw is None else bool(neutral_raw)
            home_advantage = 0.0 if neutral else self.home_advantage

            home_rating = ratings[home]
            away_rating = ratings[away]
            expected_home = self._expected_score(
                home_rating, away_rating, home_advantage
            )
            expected_away = 1.0 - expected_home

            home_rating += self.k_factor * (outcome_home - expected_home)
            away_rating += self.k_factor * (outcome_away - expected_away)

            ratings[home] = max(self.min_rating, home_rating)
            ratings[away] = max(self.min_rating, away_rating)

        self._ratings = ratings

    def update_with_result(self, game: Mapping[str, Any]) -> None:
        """Update ratings with a single game result."""
        home = str(game.get("home_team", "")).strip()
        away = str(game.get("away_team", "")).strip()
        if not home or not away:
            return
        try:
            home_score = float(game.get("home_score"))
            away_score = float(game.get("away_score"))
        except Exception:
            return

        if home_score == away_score:
            outcome_home = 0.5
        elif home_score > away_score:
            outcome_home = 1.0
        else:
            outcome_home = 0.0
        outcome_away = 1.0 - outcome_home

        neutral_raw = game.get("neutral", False)
        if isinstance(neutral_raw, float) and isnan(neutral_raw):
            neutral = False
        else:
            neutral = False if neutral_raw is None else bool(neutral_raw)
        home_advantage = 0.0 if neutral else self.home_advantage

        home_rating = self._ratings[home]
        away_rating = self._ratings[away]
        expected_home = self._expected_score(home_rating, away_rating, home_advantage)
        expected_away = 1.0 - expected_home

        home_rating += self.k_factor * (outcome_home - expected_home)
        away_rating += self.k_factor * (outcome_away - expected_away)

        self._ratings[home] = max(self.min_rating, home_rating)
        self._ratings[away] = max(self.min_rating, away_rating)

    def rankings(self) -> list[tuple[str, float]]:
        """Return ratings ordered from strongest to weakest."""
        return sorted(self._ratings.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def _expected_score(
        team_rating: float, opp_rating: float, home_advantage: float
    ) -> float:
        rating_diff = (opp_rating - (team_rating + home_advantage)) / 400.0
        return 1.0 / (1.0 + 10**rating_diff)


@dataclass
class EloCalibration:
    home_advantage: float
    scale: float
    error_term: float


DEFAULT_CALIBRATION = EloCalibration(home_advantage=0.0, scale=0.0, error_term=0.0)


class EloModel(BaseModel):
    """Backtest-ready Elo wrapper that calibrates margins and win probabilities."""

    def __init__(
        self,
        *,
        k_factor: float = 20.0,
        home_advantage: float = 65.0,
        initial_rating: float = 1500.0,
        min_rating: float = 1.0,
        recency_lambda: float | None = None,
        learn_home_advantage: bool = False,
        conditional_sd: bool = False,
        winprob_bias: float = 0.0,
        learn_winprob_bias: bool = False,
        total_shrinkage: float = 0.5,
        total_team_prior_games: int = 10,
        total_sd_floor: float = 0.5,
    ) -> None:
        """Initialize the backtest model.

        recency_lambda: None disables recency weighting (default = current behavior).
        conditional_sd: when False, uses the constant error_term for margin SDs.
        learn_winprob_bias: when False, uses the provided winprob_bias as-is.
        """
        self._elo = EloPowerRating(
            k_factor=k_factor,
            home_advantage=home_advantage,
            initial_rating=initial_rating,
            min_rating=min_rating,
        )
        self._coefficients = DEFAULT_CALIBRATION
        self._win_prob_k = DEFAULT_WIN_PROB_K
        self._recency_lambda = recency_lambda
        self._learn_home_advantage = learn_home_advantage
        self._conditional_sd = conditional_sd
        self._conditional_sd_model: ConditionalSDModel | None = None
        self._win_prob_bias = float(winprob_bias)
        self._learn_winprob_bias = learn_winprob_bias
        # Totals hyperparameters
        self._total_shrinkage = float(total_shrinkage)
        self._total_team_prior_games = int(total_team_prior_games)
        self._total_sd_floor = float(total_sd_floor)

        # Fitted totals state
        self._total_mean: float | None = None
        self._team_total_adj: dict[str, float] = {}
        self._total_sd: float | None = None

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self._elo.model_id,
            model_version=self._elo.model_version,
            params={
                **self._elo.params,
                "recency_lambda": self._recency_lambda,
                "learn_home_advantage": self._learn_home_advantage,
                "conditional_sd": self._conditional_sd,
                "winprob_bias": self._win_prob_bias,
                "learn_winprob_bias": self._learn_winprob_bias,
                "total_shrinkage": self._total_shrinkage,
                "total_team_prior_games": self._total_team_prior_games,
                "total_sd_floor": self._total_sd_floor,
            },
            supports_margin=True,
            supports_total=True,
            supports_win_prob=True,
            role="primary",
            ensemble_weight=1.0,
            supports_incremental_update=True,
            supports_streaming_backtest=True,
        )

    def fit(self, games_df: Any, *, fit_end_date: pd.Timestamp | None = None) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        games = games_df.to_dict(orient="records")
        self._elo.fit(games)

        design_matrix: list[list[float]] = []
        margins: list[float] = []
        weights: list[float] = []
        win_prob_samples: list[tuple[float, int]] = []
        win_prob_spreads: list[float] = []
        win_prob_outcomes: list[int] = []
        # Totals accumulators
        total_values: list[float] = []
        total_weights: list[float] = []
        team_total_weighted_sum: DefaultDict[str, float] = defaultdict(float)
        team_weight_sum: DefaultDict[str, float] = defaultdict(float)

        # Use explicit fit_end_date when provided; otherwise fall back to resolving from the DataFrame
        fit_end_date = fit_end_date if fit_end_date is not None else resolve_fit_end_date(games_df)
        for game in games:
            home = str(game.get("home_team", "")).strip()
            away = str(game.get("away_team", "")).strip()
            if not home or not away:
                continue
            try:
                margin = float(game.get("home_score")) - float(game.get("away_score"))
            except Exception:
                continue

            neutral_raw = game.get("neutral", False)
            neutral = (
                False
                if isinstance(neutral_raw, float) and np.isnan(neutral_raw)
                else bool(neutral_raw)
            )
            home_advantage_flag = 0.0 if neutral else 1.0

            home_rating = float(self._elo._ratings[home])
            away_rating = float(self._elo._ratings[away])
            rating_diff = home_rating - away_rating

            design_matrix.append([home_advantage_flag, rating_diff])
            margins.append(margin)
            w = recency_weight(game.get("date"), fit_end_date, self._recency_lambda)
            weights.append(w)

            # Totals bookkeeping: each game's total contributes to both teams
            try:
                actual_total = float(game.get("home_score")) + float(game.get("away_score"))
            except Exception:
                continue
            total_values.append(actual_total)
            total_weights.append(w)
            team_total_weighted_sum[home] += w * actual_total
            team_total_weighted_sum[away] += w * actual_total
            team_weight_sum[home] += w
            team_weight_sum[away] += w

        if design_matrix:
            matrix = np.asarray(design_matrix, dtype=float)
            target = np.asarray(margins, dtype=float)
            weight_arr = np.asarray(weights, dtype=float)
            coeffs = weighted_least_squares(matrix, target, weights=weight_arr)
            predictions = matrix @ coeffs
            residuals = predictions - target
            error_term = weighted_rmse(residuals, weight_arr)
            self._coefficients = EloCalibration(
                home_advantage=float(coeffs[0]),
                scale=float(coeffs[1]),
                error_term=error_term,
            )

            if self._conditional_sd:
                self._conditional_sd_model = fit_conditional_sd(
                    predictions, residuals, weights=weight_arr
                )
            else:
                self._conditional_sd_model = None

            for predicted_margin, actual_margin in zip(
                predictions, target, strict=False
            ):
                if actual_margin == 0:
                    continue
                projected_spread = -float(predicted_margin)
                win_prob_samples.append(
                    (projected_spread, 1 if actual_margin > 0 else 0)
                )
                win_prob_spreads.append(projected_spread)
                win_prob_outcomes.append(1 if actual_margin > 0 else 0)
        else:
            self._coefficients = DEFAULT_CALIBRATION
            self._conditional_sd_model = None

        self._win_prob_k = fit_win_prob_scale(
            win_prob_samples, default_k=DEFAULT_WIN_PROB_K
        )
        if self._learn_winprob_bias and win_prob_spreads:
            self._win_prob_bias = fit_win_prob_bias(
                win_prob_spreads,
                win_prob_outcomes,
                win_prob_k=self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K,
            )

        # Fit totals model using weighted league mean and team adjustments
        total_wsum = float(sum(total_weights)) if total_weights else 0.0
        if total_wsum > 0 and len(total_values) >= 3:
            # Weighted league mean
            league_total_mean = float(
                sum(v * w for v, w in zip(total_values, total_weights)) / total_wsum
            )
            # Per-team raw and shrunk adjustments
            raw_adj: dict[str, float] = {}
            adj: dict[str, float] = {}
            for team, wsum in team_weight_sum.items():
                if wsum > 0:
                    team_mean = float(team_total_weighted_sum[team] / wsum)
                    raw = team_mean - league_total_mean
                    raw_adj[team] = raw
                    n_eff = float(wsum)
                    shrink = n_eff / (n_eff + float(self._total_team_prior_games))
                    adj[team] = raw * shrink
                else:
                    raw_adj[team] = 0.0
                    adj[team] = 0.0

            # Build fitted predictions and residuals
            fitted_preds: list[float] = []
            fitted_weights: list[float] = []
            residuals_total: list[float] = []
            # iterate through original games to compute per-game preds
            for game, w in zip(games, total_weights):
                home = str(game.get("home_team", "")).strip()
                away = str(game.get("away_team", "")).strip()
                try:
                    actual_total = float(game.get("home_score")) + float(game.get("away_score"))
                except Exception:
                    continue
                adj_home = adj.get(home, 0.0)
                adj_away = adj.get(away, 0.0)
                pred_total_i = league_total_mean + self._total_shrinkage * 0.5 * (
                    adj_home + adj_away
                )
                fitted_preds.append(pred_total_i)
                fitted_weights.append(w)
                residuals_total.append(pred_total_i - actual_total)

            # compute weighted rmse for totals
            try:
                total_rmse = weighted_rmse(
                    np.asarray(residuals_total, dtype=float),
                    np.asarray(fitted_weights, dtype=float),
                )
            except Exception:
                total_rmse = float(DEFAULT_TOTAL_SD_FALLBACK)

            total_sd = max(self._total_sd_floor, float(total_rmse))

            self._total_mean = float(league_total_mean)
            self._team_total_adj = dict(adj)
            self._total_sd = float(total_sd)
        else:
            # fallback to global defaults
            self._total_mean = float(DEFAULT_TOTAL_MEAN_FALLBACK)
            self._team_total_adj = {}
            self._total_sd = float(DEFAULT_TOTAL_SD_FALLBACK)

    def predict_one(self, row: Mapping[str, Any]) -> GamePrediction:
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        if not home or not away:
            raise ValueError("home_team and away_team are required for prediction.")

        coefficients = self._coefficients
        win_prob_k = self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K
        model_identity = self.metadata().identity_dict()

        neutral_raw = row.get("neutral", False)
        neutral = (
            False
            if isinstance(neutral_raw, float) and np.isnan(neutral_raw)
            else bool(neutral_raw)
        )
        home_advantage_flag = 0.0 if neutral else 1.0

        home_rating = float(self._elo._ratings[home])
        away_rating = float(self._elo._ratings[away])
        rating_diff = home_rating - away_rating

        pred_margin = (
            coefficients.home_advantage * home_advantage_flag
            + coefficients.scale * rating_diff
        )
        projected_spread = -pred_margin
        adjusted_spread = align_spread_with_margin(
            pred_margin, projected_spread - self._win_prob_bias
        )
        p_home_win = logistic_win_prob(adjusted_spread, win_prob_k)
        margin_sd = (
            self._conditional_sd_model.predict(pred_margin)
            if self._conditional_sd_model is not None
            else coefficients.error_term
        )
        win_prob_dist = win_prob_distribution(
            p_home_win,
            win_prob_k=win_prob_k,
            margin_std=margin_sd,
        )

        # Totals prediction
        base_total = (
            float(self._total_mean)
            if self._total_mean is not None
            else float(DEFAULT_TOTAL_MEAN_FALLBACK)
        )
        adj_home = float(self._team_total_adj.get(home, 0.0))
        adj_away = float(self._team_total_adj.get(away, 0.0))
        pred_total = base_total + self._total_shrinkage * 0.5 * (adj_home + adj_away)
        total_sd_raw = (
            float(self._total_sd) if self._total_sd is not None else float(DEFAULT_TOTAL_SD_FALLBACK)
        )
        total_sd = max(self._total_sd_floor, total_sd_raw)

        game_id = row.get("game_id") or f"{row['date']}_{home}_{away}"
        return GamePrediction(
            game_id=str(game_id),
            date=str(row["date"]),
            home_team=home,
            away_team=away,
            p_home_win=p_home_win,
            win_prob_dist=win_prob_dist,
            pred_margin=pred_margin,
            pred_total=pred_total,
            margin_sd=margin_sd,
            total_mean=pred_total,
            total_sd=total_sd,
            win_prob_source="logistic",
            margin_dist_assumption="normal_approx",
            metadata=dict(model_identity),
            extra={
                "home_advantage": coefficients.home_advantage,
                "scale": coefficients.scale,
                "error_term": coefficients.error_term,
                "win_prob_k": win_prob_k,
                "winprob_bias": self._win_prob_bias,
                "conditional_sd": self._conditional_sd,
                "projected_total": pred_total,
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
            },
        )

    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        predictions: list[GamePrediction] = []
        for row in upcoming_games_df.to_dict(orient="records"):
            try:
                predictions.append(self.predict_one(row))
            except ValueError:
                continue
        return predictions

    def update_with_result(self, game_row: Mapping[str, Any]) -> None:
        self._elo.update_with_result(game_row)
