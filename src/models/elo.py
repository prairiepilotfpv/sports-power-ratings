from __future__ import annotations

"""Elo power rating model."""

from collections import defaultdict
from math import isnan
from typing import Any, DefaultDict, Iterable, Mapping


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
        self._ratings: DefaultDict[str, float] = defaultdict(lambda: self.initial_rating)

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
            expected_home = self._expected_score(home_rating, away_rating, home_advantage)
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
    def _expected_score(team_rating: float, opp_rating: float, home_advantage: float) -> float:
        rating_diff = (opp_rating - (team_rating + home_advantage)) / 400.0
        return 1.0 / (1.0 + 10 ** rating_diff)
