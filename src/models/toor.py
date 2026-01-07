"""TOOR (Team OLS Optimized Rating) model."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Iterable, Mapping

import numpy as np

from config import DEFAULT_WIN_PROB_K
from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns
from models.bradley_terry import BradleyTerry
from models.calibration import (
    ConditionalSDModel,
    align_spread_with_margin,
    fit_conditional_sd,
    fit_win_prob_bias,
    recency_weight,
    resolve_fit_end_date,
    resolve_fit_end_date_from_games,
    weighted_least_squares,
    weighted_rmse,
)
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
        self._ratings = {team: float(np.exp(centered[index[team]])) for team in teams}

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(self._ratings.items(), key=lambda item: item[1], reverse=True)


class TOORModel(BaseModel):
    """Backtest model mapping Bradley-Terry strengths to margins via OLS."""

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
        self._bt_model = BradleyTerry(max_iter=max_iter, tol=tol)
        self._coefficients = DEFAULT_COEFFICIENTS
        self._win_prob_k = DEFAULT_WIN_PROB_K
        self._recency_lambda = recency_lambda
        self._learn_home_advantage = learn_home_advantage
        self._conditional_sd = conditional_sd
        self._conditional_sd_model: ConditionalSDModel | None = None
        self._win_prob_bias = float(winprob_bias)
        self._learn_winprob_bias = learn_winprob_bias

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
            supports_total=False,
            supports_win_prob=True,
            role="primary",
            ensemble_weight=1.0,
        )

    def fit(self, games_df: Any) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        games = games_df.to_dict(orient="records")
        self._bt_model.fit(games)

        design_matrix: list[list[float]] = []
        margins: list[float] = []
        weights: list[float] = []
        win_prob_samples: list[tuple[float, int]] = []
        win_prob_spreads: list[float] = []
        win_prob_outcomes: list[int] = []

        fit_end_date = resolve_fit_end_date(games_df)
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
            home_advantage = 0.0 if neutral else 1.0

            home_rating = self._bt_model.ratings[home]
            away_rating = self._bt_model.ratings[away]
            if home_rating <= 0 or away_rating <= 0:
                continue

            home_log = log(home_rating)
            away_log = log(away_rating)

            design_matrix.append([home_advantage, home_log, away_log])
            margins.append(margin)
            weights.append(
                recency_weight(game.get("date"), fit_end_date, self._recency_lambda)
            )

        if design_matrix:
            matrix = np.asarray(design_matrix, dtype=float)
            target = np.asarray(margins, dtype=float)
            weight_arr = np.asarray(weights, dtype=float)
            coeffs = weighted_least_squares(matrix, target, weights=weight_arr)
            predictions = matrix @ coeffs
            residuals = predictions - target
            error_term = weighted_rmse(residuals, weight_arr)
            self._coefficients = ToorCoefficients(
                home_advantage=float(coeffs[0]),
                home_coeff=float(coeffs[1]),
                away_coeff=float(coeffs[2]),
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

            home_rating = self._bt_model.ratings[home]
            away_rating = self._bt_model.ratings[away]
            home_log = log(home_rating) if home_rating > 0 else 0.0
            away_log = log(away_rating) if away_rating > 0 else 0.0

            pred_margin = (
                coefficients.home_advantage * home_advantage
                + coefficients.home_coeff * home_log
                + coefficients.away_coeff * away_log
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
                    margin_sd=margin_sd,
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
                    },
                )
            )
        return predictions
