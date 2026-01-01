"""Elo power rating model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isnan
from typing import Any, DefaultDict, Iterable, Mapping

import numpy as np

from config import DEFAULT_WIN_PROB_K
from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns
from pipelines.projections import fit_win_prob_scale, logistic_win_prob


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
    ) -> None:
        self._elo = EloPowerRating(
            k_factor=k_factor,
            home_advantage=home_advantage,
            initial_rating=initial_rating,
            min_rating=min_rating,
        )
        self._coefficients = DEFAULT_CALIBRATION
        self._win_prob_k = DEFAULT_WIN_PROB_K

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self._elo.model_id,
            model_version=self._elo.model_version,
            params=self._elo.params,
            supports_margin=True,
            supports_total=False,
            supports_win_prob=True,
        )

    def fit(self, games_df: Any) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        games = games_df.to_dict(orient="records")
        self._elo.fit(games)

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
            home_advantage_flag = 0.0 if neutral else 1.0

            home_rating = float(self._elo._ratings[home])
            away_rating = float(self._elo._ratings[away])
            rating_diff = home_rating - away_rating

            design_matrix.append([home_advantage_flag, rating_diff])
            margins.append(margin)

        if design_matrix:
            matrix = np.asarray(design_matrix, dtype=float)
            target = np.asarray(margins, dtype=float)
            coeffs, *_ = np.linalg.lstsq(matrix, target, rcond=None)
            predictions = matrix @ coeffs
            residuals = predictions - target
            error_term = (
                float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.0
            )
            self._coefficients = EloCalibration(
                home_advantage=float(coeffs[0]),
                scale=float(coeffs[1]),
                error_term=error_term,
            )

            for predicted_margin, actual_margin in zip(
                predictions, target, strict=False
            ):
                if actual_margin == 0:
                    continue
                projected_spread = -float(predicted_margin)
                win_prob_samples.append(
                    (projected_spread, 1 if actual_margin > 0 else 0)
                )
        else:
            self._coefficients = DEFAULT_CALIBRATION

        self._win_prob_k = fit_win_prob_scale(
            win_prob_samples, default_k=DEFAULT_WIN_PROB_K
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
            neutral = False if isinstance(neutral_raw, float) and np.isnan(neutral_raw) else bool(neutral_raw)
            home_advantage_flag = 0.0 if neutral else 1.0

            home_rating = float(self._elo._ratings[home])
            away_rating = float(self._elo._ratings[away])
            rating_diff = home_rating - away_rating

            pred_margin = (
                coefficients.home_advantage * home_advantage_flag
                + coefficients.scale * rating_diff
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
                    metadata=dict(model_identity),
                    extra={
                        "home_advantage": coefficients.home_advantage,
                        "scale": coefficients.scale,
                        "error_term": coefficients.error_term,
                        "win_prob_k": win_prob_k,
                    },
                )
            )

        return predictions
