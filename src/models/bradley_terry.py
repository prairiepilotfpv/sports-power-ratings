from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Iterable, Mapping


class BradleyTerry:
    def __init__(self, *, max_iter: int = 500, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.ratings: DefaultDict[str, float] = defaultdict(lambda: 1.0)
        self.games_played: DefaultDict[str, int] = defaultdict(int)

    def fit(self, games: Iterable[Mapping[str, Any]]) -> None:
        wins: DefaultDict[str, float] = defaultdict(float)
        matchups: DefaultDict[tuple[str, str], int] = defaultdict(int)
        teams: set[str] = set()

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
                wins[home] += 0.5
                wins[away] += 0.5
            elif home_score > away_score:
                wins[home] += 1.0
            else:
                wins[away] += 1.0

            key = tuple(sorted((home, away)))
            matchups[key] += 1

        for team in teams:
            self.ratings[team] = 1.0

        if not teams:
            return

        for _ in range(self.max_iter):
            max_delta = 0.0
            updated: dict[str, float] = {}
            for team in teams:
                denom = 0.0
                for opponent in teams:
                    if opponent == team:
                        continue
                    key = tuple(sorted((team, opponent)))
                    n_ij = matchups.get(key, 0)
                    if n_ij == 0:
                        continue
                    denom += n_ij / (self.ratings[team] + self.ratings[opponent])
                if denom == 0.0:
                    updated[team] = self.ratings[team]
                else:
                    updated[team] = wins[team] / denom
                max_delta = max(max_delta, abs(updated[team] - self.ratings[team]))

            self.ratings.update(updated)
            if max_delta < self.tol:
                break

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)
