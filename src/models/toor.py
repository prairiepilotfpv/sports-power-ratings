from __future__ import annotations

"""TOOR (Team OLS Optimized Rating) model."""

from dataclasses import dataclass
from math import log
from typing import Any, Iterable, Mapping

import numpy as np

from config import DEFAULT_WIN_PROB_K
from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns
from models.bradley_terry import BradleyTerry
from pipelines.projections import fit_win_prob_scale, logistic_win_prob


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
    """Power rating wrapper that reuses Bradley-Terry logistic strengths."""

    def __init__(self, *, max_iter: int = 500, tol: float = 1e-8) -> None:
        self._model = BradleyTerry(max_iter=max_iter, tol=tol)

    def fit(self, games: Iterable[Mapping[str, Any]]) -> None:
        self._model.fit(games)

    def rankings(self) -> list[tuple[str, float]]:
        return self._model.rankings()


class TOORModel(BaseModel):
    """TOOR backtest model that applies OLS to Bradley-Terry logistic strengths."""

    def __init__(self, *, max_iter: int = 500, tol: float = 1e-8) -> None:
        self._bt_model = BradleyTerry(max_iter=max_iter, tol=tol)
        self._coefficients = DEFAULT_COEFFICIENTS
        self._win_prob_k = DEFAULT_WIN_PROB_K

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="toor",
            version="1.0",
            supports_margin=True,
            supports_total=False,
            supports_win_prob=True,
        )

    def fit(self, games_df: Any) -> None:
        require_columns(games_df, ["home_team", "away_team", "home_score", "away_score"])
        games = games_df.to_dict(orient="records")
        self._bt_model.fit(games)

        design_matrix: list[list[float]] = []
        margins: list[float] = []
        win_prob_samples: list[tuple[float, int]] = []

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
            neutral = False if isinstance(neutral_raw, float) and np.isnan(neutral_raw) else bool(neutral_raw)
            home_advantage = 0.0 if neutral else 1.0

            home_rating = self._bt_model.ratings[home]
            away_rating = self._bt_model.ratings[away]
            if home_rating <= 0 or away_rating <= 0:
                continue

            home_log = log(home_rating)
            away_log = log(away_rating)

            design_matrix.append([home_advantage, home_log, away_log])
            margins.append(margin)

        if design_matrix:
            matrix = np.asarray(design_matrix, dtype=float)
            target = np.asarray(margins, dtype=float)
            coeffs, *_ = np.linalg.lstsq(matrix, target, rcond=None)
            predictions = matrix @ coeffs
            residuals = predictions - target
            error_term = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
            self._coefficients = ToorCoefficients(
                home_advantage=float(coeffs[0]),
                home_coeff=float(coeffs[1]),
                away_coeff=float(coeffs[2]),
                error_term=error_term,
            )

            for predicted_margin, actual_margin in zip(predictions, target, strict=False):
                if actual_margin == 0:
                    continue
                projected_spread = -float(predicted_margin)
                win_prob_samples.append((projected_spread, 1 if actual_margin > 0 else 0))

        self._win_prob_k = fit_win_prob_scale(win_prob_samples, default_k=DEFAULT_WIN_PROB_K)

    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        predictions: list[GamePrediction] = []
        coefficients = self._coefficients
        win_prob_k = self._win_prob_k if self._win_prob_k > 0 else DEFAULT_WIN_PROB_K

        for row in upcoming_games_df.to_dict(orient="records"):
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            if not home or not away:
                continue

            neutral_raw = row.get("neutral", False)
            neutral = False if isinstance(neutral_raw, float) and np.isnan(neutral_raw) else bool(neutral_raw)
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
            p_home_win = logistic_win_prob(projected_spread, win_prob_k)

            game_id = row.get("game_id") or f"{row['date']}_{home}_{away}"
            predictions.append(
                GamePrediction(
                    game_id=str(game_id),
                    date=str(row["date"]),
                    home_team=home,
                    away_team=away,
                    p_home_win=p_home_win,
                    pred_margin=pred_margin,
                    extra={
                        "home_advantage": coefficients.home_advantage,
                        "home_coeff": coefficients.home_coeff,
                        "away_coeff": coefficients.away_coeff,
                        "error_term": coefficients.error_term,
                        "win_prob_k": win_prob_k,
                    },
                )
            )
        return predictions

