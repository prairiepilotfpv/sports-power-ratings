from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns
from models.bradley_terry import BradleyTerry
from models.calibration import (
    ConditionalSDModel,
    fit_conditional_sd,
    recency_weight,
    resolve_fit_end_date,
    weighted_least_squares,
    weighted_rmse,
)


@dataclass
class CalibrationCoefficients:
    home_advantage_points: float
    scale: float
    error_term: float


DEFAULT_CALIBRATION = CalibrationCoefficients(
    home_advantage_points=0.0,
    scale=1.0,
    error_term=0.0,
)


class BradleyTerryCalibratedHFA(BaseModel):
    """Bradley-Terry with calibrated margin mapping in points."""

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
        strict: bool = False,
    ) -> None:
        """Initialize the backtest model.

        recency_lambda: None disables recency weighting (default = current behavior).
        conditional_sd: when False, uses the constant error_term for margin SDs.
        learn_winprob_bias: when False, uses the provided winprob_bias as-is.
        """
        self._max_iter = max_iter
        self._tol = tol
        self._bt_model = BradleyTerry(max_iter=max_iter, tol=tol)
        self._coefficients = DEFAULT_CALIBRATION
        self._recency_lambda = recency_lambda
        self._learn_home_advantage = learn_home_advantage
        self._conditional_sd = conditional_sd
        self._conditional_sd_model: ConditionalSDModel | None = None
        self._win_prob_bias = float(winprob_bias)
        self._learn_winprob_bias = learn_winprob_bias
        self._strict = strict

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="bradley_terry_calibrated_hfa",
            model_version="1.0",
            params={
                "max_iter": self._max_iter,
                "tol": self._tol,
                "recency_lambda": self._recency_lambda,
                "learn_home_advantage": self._learn_home_advantage,
                "conditional_sd": self._conditional_sd,
                "winprob_bias": self._win_prob_bias,
                "learn_winprob_bias": self._learn_winprob_bias,
                "strict": self._strict,
            },
            supports_margin=True,
            supports_total=True,
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
            rating_diff = home_rating - away_rating
            design_matrix.append([home_advantage, rating_diff])
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
            self._coefficients = CalibrationCoefficients(
                home_advantage_points=float(coeffs[0]),
                scale=float(coeffs[1]),
                error_term=error_term,
            )

            if self._conditional_sd:
                self._conditional_sd_model = fit_conditional_sd(
                    predictions, residuals, weights=weight_arr
                )
            else:
                self._conditional_sd_model = None


    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        predictions: list[GamePrediction] = []
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
            projection = self._bt_model.project_matchup(
                home,
                away,
                neutral=neutral,
            )
            pred_margin = projection["margin_mean"]
            margin_sd = projection["margin_sd"]
            total_mean = projection["total_mean"]
            total_sd = projection["total_sd"]
            p_home_win = projection["p_home_win"]

            game_id = row.get("game_id") or f"{row['date']}_{home}_{away}"
            extra = {
                "projected_home_score": projection["projected_home_score"],
                "projected_away_score": projection["projected_away_score"],
                "projected_spread": -pred_margin,
                "model_p_home_win": p_home_win,
                "normal_p_home_win": p_home_win,
                "win_prob_source": "bt_margin_normal",
                "margin_dist_assumption": "normal_approx",
                "logistic_home_win_prob": None,
            }
            self._validate_prediction(
                p_home_win,
                margin_sd,
                total_sd,
                extra["win_prob_source"],
                game_id,
            )
            predictions.append(
                GamePrediction(
                    game_id=str(game_id),
                    date=str(row["date"]),
                    home_team=home,
                    away_team=away,
                    p_home_win=p_home_win,
                    win_prob_dist=None,
                    pred_margin=pred_margin,
                    pred_total=total_mean,
                    margin_sd=margin_sd,
                    total_sd=total_sd,
                    total_mean=total_mean,
                    win_prob_source="logistic",
                    margin_dist_assumption="normal_approx",
                    metadata=dict(model_identity),
                    extra={
                        **extra,
                        "home_advantage_points": self._coefficients.home_advantage_points,
                        "scale": self._coefficients.scale,
                        "error_term": self._coefficients.error_term,
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

    def _validate_prediction(
        self,
        p_home_win: float,
        margin_sd: float,
        total_sd: float,
        win_prob_source: str,
        game_id: str,
    ) -> None:
        errors = []
        if not (0.0 < p_home_win < 1.0):
            errors.append("p_home_win must be between 0 and 1.")
        if margin_sd < 5.0:
            errors.append("margin_sd must be at least 5.")
        if total_sd < 8.0:
            errors.append("total_sd must be at least 8.")
        if win_prob_source == "direct":
            errors.append("win_prob_source cannot be 'direct'.")
        if not errors:
            return
        message = f"Invalid BT prediction for {game_id}: " + "; ".join(errors)
        if self._strict:
            raise ValueError(message)
        import warnings

        warnings.warn(message, RuntimeWarning, stacklevel=2)
