from __future__ import annotations

from math import log
from typing import Any

from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns
from models.bradley_terry import BradleyTerry


class BradleyTerryHFA(BaseModel):
    """Bradley-Terry with home-field advantage.

    Notes:
        pred_margin is reported as the raw log-odds score
        (log(rating_home) - log(rating_away) + HFA). This is a linear proxy
        for margin and is not calibrated to points.
    """
    def __init__(self, *, max_iter: int = 500, tol: float = 1e-8) -> None:
        self._max_iter = max_iter
        self._tol = tol
        self._model = BradleyTerry(max_iter=max_iter, tol=tol)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="bradley_terry_hfa",
            model_version="1.0",
            params={
                "max_iter": self._max_iter,
                "tol": self._tol,
            },
            supports_margin=True,
            supports_total=False,
            supports_win_prob=True,
        )

    def fit(self, games_df: Any) -> None:
        require_columns(games_df, ["home_team", "away_team", "home_score", "away_score"])
        games = games_df.to_dict(orient="records")
        self._model.fit(games)

    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        predictions: list[GamePrediction] = []
        model_identity = self.metadata().identity_dict()
        for row in upcoming_games_df.to_dict(orient="records"):
            neutral = bool(row.get("neutral", False))
            venue = "neutral" if neutral else "home"
            p_home_win = self._model.predict_probability(
                str(row["home_team"]),
                str(row["away_team"]),
                venue=venue,
            )
            pred_margin = self._score_margin(
                str(row["home_team"]),
                str(row["away_team"]),
                neutral=neutral,
            )
            game_id = row.get("game_id") or f"{row['date']}_{row['home_team']}_{row['away_team']}"
            predictions.append(
                GamePrediction(
                    game_id=str(game_id),
                    date=str(row["date"]),
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    p_home_win=p_home_win,
                    pred_margin=pred_margin,
                    metadata=dict(model_identity),
                )
            )
        return predictions

    def _score_margin(self, home_team: str, away_team: str, *, neutral: bool) -> float:
        rating_home = self._model.ratings[home_team]
        rating_away = self._model.ratings[away_team]
        score = log(rating_home) - log(rating_away)
        if not neutral:
            score += self._model.home_adv
        return score
