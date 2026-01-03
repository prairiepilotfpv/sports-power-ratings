"""GSSD (Generalized Scores Standard Deviation) power rating model."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from ssat.frequentist import GSSD

from config import DEFAULT_WIN_PROB_K
from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns
from pipelines.projections import (
    fit_win_prob_scale,
    logistic_win_prob,
    win_prob_distribution,
)


class GSSDPowerRating:
    """Power rating wrapper around ssat's GSSD implementation."""

    def __init__(self) -> None:
        self.model_id = "gssd"
        self.model_version = "1.0"
        self.params: dict[str, Any] = {}
        self._model = GSSD()
        self._ratings: dict[str, float] = {}

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
        rows: list[dict[str, object]] = []
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
            rows.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "home_points": home_score,
                    "away_points": away_score,
                }
            )

        if not rows:
            self._ratings = {}
            return

        df = pd.DataFrame(rows)
        X = df[["home_team", "away_team"]]
        y = df["home_points"] - df["away_points"]
        Z = df[["home_points", "away_points"]]
        self._model.fit(X, y=y, Z=Z)
        self._ratings = self._build_ratings()

    def rankings(self) -> list[tuple[str, float]]:
        """Return ratings ordered from strongest to weakest."""
        return sorted(self._ratings.items(), key=lambda item: item[1], reverse=True)

    def _build_ratings(self) -> dict[str, float]:
        ratings: dict[str, float] = {}
        team_ratings = getattr(self._model, "team_ratings_", None)
        if not team_ratings:
            return ratings

        for team, values in team_ratings.items():
            try:
                pfh, pah, pfa, paa = (float(value) for value in values)
            except Exception:
                continue
            net_rating = 0.5 * ((pfh - pah) + (pfa - paa))
            ratings[str(team)] = exp(net_rating)
        return ratings


@dataclass
class GSSDCalibration:
    home_advantage_points: float
    scale: float
    error_term: float


DEFAULT_CALIBRATION = GSSDCalibration(
    home_advantage_points=0.0, scale=0.0, error_term=0.0
)


class GSSDModel(BaseModel):
    """Backtest wrapper that calibrates GSSD ratings to points and win probs."""

    def __init__(self) -> None:
        self._gssd = GSSDPowerRating()
        self._coefficients = DEFAULT_CALIBRATION
        self._win_prob_k = DEFAULT_WIN_PROB_K

    def metadata(self) -> ModelMetadata:
        meta = self._gssd.metadata()
        return ModelMetadata(
            model_id=meta.model_id,
            model_version=meta.model_version,
            params=meta.params,
            supports_margin=True,
            supports_total=False,
            supports_win_prob=True,
        )

    def fit(self, games_df: Any) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        games = games_df.to_dict(orient="records")
        self._gssd.fit(games)

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
            neutral = (
                False
                if isinstance(neutral_raw, float) and np.isnan(neutral_raw)
                else bool(neutral_raw)
            )
            home_advantage_flag = 0.0 if neutral else 1.0

            home_rating = float(self._gssd._ratings.get(home, 1.0))
            away_rating = float(self._gssd._ratings.get(away, 1.0))
            if home_rating <= 0 or away_rating <= 0:
                continue
            rating_diff = log(home_rating) - log(away_rating)

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
            self._coefficients = GSSDCalibration(
                home_advantage_points=float(coeffs[0]),
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
            neutral = (
                False
                if isinstance(neutral_raw, float) and np.isnan(neutral_raw)
                else bool(neutral_raw)
            )
            home_advantage_flag = 0.0 if neutral else 1.0

            home_rating = float(self._gssd._ratings.get(home, 1.0))
            away_rating = float(self._gssd._ratings.get(away, 1.0))
            if home_rating <= 0 or away_rating <= 0:
                continue

            rating_diff = log(home_rating) - log(away_rating)
            pred_margin = (
                coefficients.home_advantage_points * home_advantage_flag
                + coefficients.scale * rating_diff
            )
            projected_spread = -pred_margin
            p_home_win = logistic_win_prob(projected_spread, win_prob_k)
            win_prob_dist = win_prob_distribution(
                p_home_win,
                win_prob_k=win_prob_k,
                margin_std=coefficients.error_term,
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
                    metadata=dict(model_identity),
                    extra={
                        "home_advantage_points": coefficients.home_advantage_points,
                        "scale": coefficients.scale,
                        "error_term": coefficients.error_term,
                        "win_prob_k": win_prob_k,
                    },
                )
            )
        return predictions
