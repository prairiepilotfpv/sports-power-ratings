from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict

H_ELO = 55.0
PTS_PER_ELO = 0.085


def exp_win_prob(rh: float, ra: float, H: float = H_ELO) -> float:
    """Expected win probability for the home team."""

    return 1.0 / (1.0 + 10 ** (-(((rh + H) - ra) / 400.0)))


def margin_scaler(m: float, d: float) -> float:
    """Scale rating adjustments based on score margin and pre-game diff."""

    return min(math.log(m + 1.0) * (2.2 / (2.2 + 0.001 * abs(d))), 1.8)


def k_factor(n: int, base: float = 22.0) -> float:
    """Dynamic K-factor that decays with games played."""

    return base / (1 + max(n, 0)) ** 0.5


class Elo:
    def __init__(self) -> None:
        self.R: DefaultDict[str, float] = defaultdict(lambda: 1500.0)
        self.N: DefaultDict[str, int] = defaultdict(int)

    def predict(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        H = 0.0 if neutral else H_ELO
        diff = (self.R[home] + H) - self.R[away]
        return exp_win_prob(self.R[home], self.R[away], H), diff * PTS_PER_ELO

    def update(self, home: str, away: str, ph: int, pa: int, neutral: bool = False) -> None:
        H = 0.0 if neutral else H_ELO
        rh = self.R[home]
        ra = self.R[away]
        expected = exp_win_prob(rh, ra, H)
        score = 1.0 if ph > pa else 0.0
        pre_game_diff = (rh + H) - ra
        margin = abs(ph - pa)
        g = margin_scaler(margin, pre_game_diff)
        kh = k_factor(self.N[home])
        ka = k_factor(self.N[away])
        delta = g * (score - expected)
        self.R[home] = rh + kh * delta
        self.R[away] = ra - ka * delta
        self.N[home] += 1
        self.N[away] += 1
