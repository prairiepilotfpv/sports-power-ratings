from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

DEFAULT_LOGISTIC_SCALE = 10.0


@dataclass(frozen=True)
class GameProjection:
    margin_neutral: float
    margin: float
    projected_winner: str | None
    projected_spread: float
    projected_win_prob: float | None
    projected_home_score: float | None
    projected_away_score: float | None
    projected_total: float | None


def logistic_win_prob(margin: float, k: float) -> float:
    return 1.0 / (1.0 + math.exp(-margin / k))


def project_game(
    home_rating: float,
    away_rating: float,
    *,
    home_advantage: float,
    neutral: bool,
    k: float = DEFAULT_LOGISTIC_SCALE,
    base_total: float | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
) -> GameProjection:
    margin_neutral = home_rating - away_rating
    margin = margin_neutral + (0.0 if neutral else home_advantage)
    projected_spread = -margin

    projected_winner = None
    if home_team is not None and away_team is not None:
        projected_winner = home_team if margin > 0 else away_team

    projected_win_prob = None
    if k > 0:
        projected_win_prob = logistic_win_prob(margin, k)

    projected_home_score = None
    projected_away_score = None
    projected_total = None
    if base_total is not None:
        projected_total = base_total
        projected_home_score = (base_total + margin) / 2.0
        projected_away_score = (base_total - margin) / 2.0

    return GameProjection(
        margin_neutral=margin_neutral,
        margin=margin,
        projected_winner=projected_winner,
        projected_spread=projected_spread,
        projected_win_prob=projected_win_prob,
        projected_home_score=projected_home_score,
        projected_away_score=projected_away_score,
        projected_total=projected_total,
    )


def average_total_points(rows: Iterable[dict[str, object]]) -> float:
    totals: list[float] = []
    for row in rows:
        try:
            total = float(row.get("home_score")) + float(row.get("away_score"))
        except Exception:
            continue
        totals.append(total)
    if not totals:
        return 0.0
    return sum(totals) / len(totals)


def fit_win_prob_scale(
    samples: Sequence[tuple[float, int]],
    *,
    default_k: float = DEFAULT_LOGISTIC_SCALE,
    min_k: float = 0.5,
    max_k: float = 50.0,
    grid_steps: int = 100,
) -> float:
    if not samples:
        return default_k

    def log_likelihood(k: float) -> float:
        ll = 0.0
        for margin, outcome in samples:
            p = logistic_win_prob(margin, k)
            p = min(max(p, 1e-9), 1.0 - 1e-9)
            ll += outcome * math.log(p) + (1 - outcome) * math.log(1.0 - p)
        return ll

    step = (max_k - min_k) / grid_steps
    grid = [min_k + step * i for i in range(grid_steps + 1)]
    best_k = max(grid, key=log_likelihood)

    lower = max(min_k, best_k - step)
    upper = min(max_k, best_k + step)
    phi = (1 + math.sqrt(5)) / 2
    for _ in range(32):
        left = upper - (upper - lower) / phi
        right = lower + (upper - lower) / phi
        if log_likelihood(left) > log_likelihood(right):
            upper = right
        else:
            lower = left

    return max(min((lower + upper) / 2.0, max_k), min_k)
