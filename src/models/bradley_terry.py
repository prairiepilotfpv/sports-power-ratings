from __future__ import annotations

from collections import defaultdict
from math import exp, isnan
from typing import Any, DefaultDict, Iterable, Mapping


class BradleyTerry:
    def __init__(self, *, max_iter: int = 500, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.ratings: DefaultDict[str, float] = defaultdict(lambda: 1.0)
        self.games_played: DefaultDict[str, int] = defaultdict(int)

    def fit(self, games: Iterable[Mapping[str, Any]]) -> None:
        teams: set[str] = set()
        games_list: list[tuple[str, str, float, bool]] = []

        for game in games:
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
                score = theta[home] - theta[away] + (0.0 if neutral else home_adv)
                win_prob = 1.0 / (1.0 + exp(-score))
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

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)
