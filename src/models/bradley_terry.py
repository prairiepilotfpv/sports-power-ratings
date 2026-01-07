"""Bradley-Terry power rating model for win/loss outcomes."""

from __future__ import annotations

from collections import defaultdict
from math import exp, isnan, log
from typing import Any, DefaultDict, Iterable, Mapping

from models.base import BaseModel, GamePrediction, ModelMetadata, require_columns


class BradleyTerry:
    """Iterative Bradley-Terry solver with optional home advantage term."""

    def __init__(self, *, max_iter: int = 500, tol: float = 1e-8) -> None:
        self.model_id = "bradley-terry"
        self.model_version = "1.0"
        self.params = {"max_iter": max_iter, "tol": tol}
        self.max_iter = max_iter
        self.tol = tol
        self.ratings: DefaultDict[str, float] = defaultdict(lambda: 1.0)
        self.games_played: DefaultDict[str, int] = defaultdict(int)
        self.home_adv = 0.0

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

    @staticmethod
    def _sigmoid(score: float) -> float:
        """Numerically stable sigmoid for logistic probability."""
        if score >= 0:
            z = exp(-score)
            return 1.0 / (1.0 + z)
        z = exp(score)
        return z / (1.0 + z)

    def predict_probability(
        self, team_a: str, team_b: str, venue: str = "neutral"
    ) -> float:
        """Predict win probability for team_a vs team_b with a venue adjustment."""
        rating_a = self.ratings[team_a]
        rating_b = self.ratings[team_b]
        score = log(rating_a) - log(rating_b)
        if venue == "home":
            score += self.home_adv
        elif venue == "away":
            score -= self.home_adv
        return self._sigmoid(score)

    def fit(self, games: Iterable[Mapping[str, Any]]) -> None:
        """Fit Bradley-Terry ratings from game results."""
        teams: set[str] = set()
        games_list: list[tuple[str, str, float, bool]] = []

        for game in games:
            # Drop rows without names or usable scores.
            home = str(game.get("home_team", "")).strip()
            away = str(game.get("away_team", "")).strip()
            if not home or not away:
                continue
            try:
                home_score = int(game.get("home_score"))
                away_score = int(game.get("away_score"))
            except Exception:
                continue

            teams.update([home, away])
            self.games_played[home] += 1
            self.games_played[away] += 1

            if home_score == away_score:
                outcome = 0.5
            elif home_score > away_score:
                outcome = 1.0
            else:
                outcome = 0.0

            neutral_raw = game.get("neutral", False)
            if isinstance(neutral_raw, float) and isnan(neutral_raw):
                neutral = False
            else:
                neutral = False if neutral_raw is None else bool(neutral_raw)
            games_list.append((home, away, outcome, neutral))

        if not teams:
            return

        theta: dict[str, float] = {team: 0.0 for team in teams}
        home_adv = 0.0
        step = 0.1 / max(1.0, len(games_list) / 1000.0)

        for _ in range(self.max_iter):
            grad: dict[str, float] = {team: 0.0 for team in teams}
            grad_home_adv = 0.0
            for home, away, outcome, neutral in games_list:
                # Log-odds include a home advantage term when not neutral.
                score = theta[home] - theta[away] + (0.0 if neutral else home_adv)
                win_prob = self._sigmoid(score)
                diff = outcome - win_prob
                grad[home] += diff
                grad[away] -= diff
                if not neutral:
                    grad_home_adv += diff

            max_delta = 0.0
            for team in teams:
                update = step * grad[team]
                theta[team] += update
                max_delta = max(max_delta, abs(update))
            home_update = step * grad_home_adv
            home_adv += home_update
            max_delta = max(max_delta, abs(home_update))
            if max_delta < self.tol:
                break

        for team in teams:
            self.ratings[team] = exp(theta[team])
        self.home_adv = home_adv

    def rankings(self) -> list[tuple[str, float]]:
        """Return ratings ordered from strongest to weakest."""
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)


class BradleyTerryBacktest(BaseModel):
    """Backtest adapter that reuses the core BradleyTerry implementation."""

    def __init__(self, *, max_iter: int = 500, tol: float = 1e-8) -> None:
        self._model = BradleyTerry(max_iter=max_iter, tol=tol)

    def metadata(self) -> ModelMetadata:
        return self._model.metadata()

    def fit(self, games_df: Any) -> None:
        require_columns(
            games_df, ["home_team", "away_team", "home_score", "away_score"]
        )
        self._model.fit(games_df.to_dict(orient="records"))

    def predict(self, upcoming_games_df: Any) -> list[GamePrediction]:
        require_columns(upcoming_games_df, ["date", "home_team", "away_team"])
        predictions: list[GamePrediction] = []
        metadata = self.metadata().identity_dict()

        for row in upcoming_games_df.to_dict(orient="records"):
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            if not home or not away:
                continue

            neutral_raw = row.get("neutral", False)
            neutral = (
                False
                if isinstance(neutral_raw, float) and isnan(neutral_raw)
                else bool(neutral_raw)
            )
            venue = "neutral" if neutral else "home"
            p_home_win = self._model.predict_probability(home, away, venue=venue)
            pred_margin = self._logit(p_home_win)
            game_id = row.get("game_id") or f"{row['date']}_{home}_{away}"

            predictions.append(
                GamePrediction(
                    game_id=str(game_id),
                    date=str(row["date"]),
                    home_team=home,
                    away_team=away,
                    p_home_win=p_home_win,
                    win_prob_samples=None,
                    pred_margin=pred_margin,
                    metadata=dict(metadata),
                )
            )

        return predictions

    @staticmethod
    def _logit(prob: float, *, epsilon: float = 1e-12) -> float:
        """Convert probability to log-odds with numeric stability."""
        p = min(max(prob, epsilon), 1.0 - epsilon)
        return log(p) - log(1.0 - p)
