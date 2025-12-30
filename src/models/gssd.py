from __future__ import annotations

"""GSSD (Generalized Scores Standard Deviation) power rating model."""

from math import exp
from typing import Any, Iterable, Mapping

import pandas as pd
from ssat.frequentist import GSSD


class GSSDPowerRating:
    """Power rating wrapper around ssat's GSSD implementation."""

    def __init__(self) -> None:
        self.model_id = "gssd"
        self.model_version = "1.0"
        self.params: dict[str, Any] = {}
        self._model = GSSD()
        self._ratings: dict[str, float] = {}

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
