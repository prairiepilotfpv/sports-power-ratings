"""Projection helpers for spreads, totals, and win probabilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

try:
    from scipy.stats import norm
except Exception:  # pragma: no cover - scipy optional
    norm = None

from config import DEFAULT_WIN_PROB_K


@dataclass(frozen=True)
class GameProjection:
    """Derived metrics for a single projected game."""

    margin_neutral: float
    margin: float
    projected_winner: str | None
    projected_spread: float
    projected_home_spread: float
    projected_win_prob: float | None
    projected_home_score: float | None
    projected_away_score: float | None
    projected_total: float | None


def _normal_cdf(x: float, *, mean: float, sd: float) -> float:
    """Normal CDF using scipy when available, else an erf fallback."""
    if sd <= 0 or not math.isfinite(sd):
        raise ValueError("Standard deviation must be positive and finite.")
    if norm is not None:
        return float(norm.cdf(x, loc=mean, scale=sd))
    z = (x - mean) / (sd * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _safe_mean(value: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_sd(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        sd = float(value)
    except (TypeError, ValueError):
        return None
    if sd <= 0 or not math.isfinite(sd):
        return None
    return sd


def home_win_prob_from_margin(margin_mean: float | None, margin_sd: float | None) -> float | None:
    """Probability the home team wins given a margin distribution (home - away)."""
    mean = _safe_mean(margin_mean)
    sd = _safe_sd(margin_sd)
    if mean is None or sd is None:
        return None
    return 1.0 - _normal_cdf(0.0, mean=mean, sd=sd)


def cover_prob(
    spread: float | None,
    margin_mean: float | None,
    margin_sd: float | None,
    sign_convention: str = "away_minus_home",
) -> float | None:
    """Price a spread using a normal margin outcome distribution."""
    mean = _safe_mean(margin_mean)
    sd = _safe_sd(margin_sd)
    if mean is None or sd is None or spread is None:
        return None
    if sign_convention not in {"away_minus_home", "home_minus_away"}:
        raise ValueError("sign_convention must be 'away_minus_home' or 'home_minus_away'.")
    threshold = -float(spread) if sign_convention == "away_minus_home" else float(spread)
    return 1.0 - _normal_cdf(threshold, mean=mean, sd=sd)


def over_prob(
    total_line: float | None,
    total_mean: float | None,
    total_sd: float | None,
) -> float | None:
    """Probability the combined score goes over a posted total (outcome distribution)."""
    mean = _safe_mean(total_mean)
    sd = _safe_sd(total_sd)
    if mean is None or sd is None or total_line is None:
        return None
    return 1.0 - _normal_cdf(float(total_line), mean=mean, sd=sd)


def logistic_win_prob(away_minus_home: float | np.ndarray, k: float) -> float | np.ndarray:
    """Convert a spread into a win probability using a logistic curve.
    
    Args:
        away_minus_home: Spread value(s) (negative of home advantage)
        k: Logistic curve steepness parameter
        
    Returns:
        Win probability or array of win probabilities
    """
    # Handle both scalar and array inputs
    if isinstance(away_minus_home, np.ndarray):
        return 1.0 / (1.0 + np.exp(away_minus_home / k))
    else:
        return 1.0 / (1.0 + math.exp(away_minus_home / k))


def win_prob_distribution(
    p_home_win: float,
    *,
    win_prob_k: float | None,
    margin_std: float | None,
) -> list[dict[str, float]]:
    """Return a discrete distribution over win probability outcomes.

    Format: [{"p_home_win": <probability>, "weight": <mass>}].
    """
    if (
        win_prob_k is None
        or margin_std is None
        or win_prob_k <= 0
        or margin_std <= 0
    ):
        return [{"p_home_win": float(p_home_win), "weight": 1.0}]

    p_home_win = min(max(float(p_home_win), 0.0), 1.0)
    slope = (p_home_win * (1.0 - p_home_win)) / win_prob_k
    prob_std = abs(slope) * margin_std
    if prob_std <= 0:
        return [{"p_home_win": p_home_win, "weight": 1.0}]

    delta = prob_std * math.sqrt(2.0)
    delta = min(delta, p_home_win, 1.0 - p_home_win)
    if delta <= 0:
        return [{"p_home_win": p_home_win, "weight": 1.0}]

    return [
        {"p_home_win": p_home_win - delta, "weight": 0.25},
        {"p_home_win": p_home_win, "weight": 0.5},
        {"p_home_win": p_home_win + delta, "weight": 0.25},
    ]


def project_game(
    home_rating: float,
    away_rating: float,
    *,
    home_advantage: float,
    neutral: bool,
    k: float = DEFAULT_WIN_PROB_K,
    base_total: float | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
) -> GameProjection:
    """Project margin, spread, totals, and win prob from power ratings."""
    margin_neutral = home_rating - away_rating
    margin = margin_neutral + (0.0 if neutral else home_advantage)
    # Convention: projected_spread is away_minus_home. Negative => home favored.
    projected_spread = -margin
    projected_home_spread = -projected_spread

    projected_winner = None
    if home_team is not None and away_team is not None:
        projected_winner = home_team if margin > 0 else away_team

    projected_win_prob = None
    if k > 0:
        projected_win_prob = logistic_win_prob(projected_spread, k)

    projected_home_score = None
    projected_away_score = None
    if base_total is not None:
        # Translate expected total + margin into implied team scores.
        projected_home_score = (base_total + margin) / 2.0
        projected_away_score = (base_total - margin) / 2.0
        projected_total = projected_home_score + projected_away_score

    projected_total = None
    if projected_home_score is not None and projected_away_score is not None:
        projected_total = projected_home_score + projected_away_score
    elif base_total is not None:
        projected_total = base_total

    return GameProjection(
        margin_neutral=margin_neutral,
        margin=margin,
        projected_winner=projected_winner,
        projected_spread=projected_spread,
        projected_home_spread=projected_home_spread,
        projected_win_prob=projected_win_prob,
        projected_home_score=projected_home_score,
        projected_away_score=projected_away_score,
        projected_total=projected_total,
    )


def average_total_points(rows: Iterable[dict[str, object]]) -> float:
    """Compute average combined score from completed games."""
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


def fit_total_model(
    rows: Iterable[dict[str, object]],
    ratings: dict[str, float],
) -> tuple[float, float]:
    """Fit a simple linear model for totals based on summed team ratings."""
    points: list[tuple[float, float]] = []
    for row in rows:
        try:
            total = float(row.get("home_score")) + float(row.get("away_score"))
        except Exception:
            continue
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        if not home or not away:
            continue
        home_rating = ratings.get(home)
        away_rating = ratings.get(away)
        if home_rating is None or away_rating is None:
            continue
        points.append((home_rating + away_rating, total))

    if len(points) < 2:
        avg_total = average_total_points(rows)
        return avg_total, 0.0

    xs, ys = zip(*points, strict=False)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0.0:
        return mean_y, 0.0

    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return intercept, slope


def total_from_ratings(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    *,
    intercept: float,
    slope: float,
) -> float | None:
    """Estimate matchup total using the fitted rating-based total model."""
    home_rating = ratings.get(home_team)
    away_rating = ratings.get(away_team)
    if home_rating is None or away_rating is None:
        return None
    total = intercept + slope * (home_rating + away_rating)
    return total if total > 0 else None


def team_scoring_averages(
    rows: Iterable[dict[str, object]],
) -> dict[str, tuple[float, float]]:
    """Compute per-team average points scored and allowed."""
    totals_for: dict[str, float] = {}
    totals_against: dict[str, float] = {}
    counts: dict[str, int] = {}

    for row in rows:
        try:
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            home_score = float(row.get("home_score"))
            away_score = float(row.get("away_score"))
        except Exception:
            continue
        if not home or not away:
            continue

        totals_for[home] = totals_for.get(home, 0.0) + home_score
        totals_against[home] = totals_against.get(home, 0.0) + away_score
        counts[home] = counts.get(home, 0) + 1

        totals_for[away] = totals_for.get(away, 0.0) + away_score
        totals_against[away] = totals_against.get(away, 0.0) + home_score
        counts[away] = counts.get(away, 0) + 1

    averages: dict[str, tuple[float, float]] = {}
    for team, games in counts.items():
        if games <= 0:
            continue
        averages[team] = (totals_for[team] / games, totals_against[team] / games)
    return averages


def matchup_total_from_averages(
    home_team: str,
    away_team: str,
    averages: dict[str, tuple[float, float]],
) -> float | None:
    """Estimate a matchup total using each team's scoring averages."""
    home_avg = averages.get(home_team)
    away_avg = averages.get(away_team)
    if home_avg is None or away_avg is None:
        return None
    home_for, home_against = home_avg
    away_for, away_against = away_avg
    total = (home_for + away_for + home_against + away_against) / 2.0
    return total if total > 0 else None


def fit_win_prob_scale(
    samples: Sequence[tuple[float, int]],
    *,
    default_k: float = DEFAULT_WIN_PROB_K,
    min_k: float = 0.5,
    max_k: float = 50.0,
    grid_steps: int = 100,
) -> float:
    """Fit the win-probability scale using a coarse grid + golden-section search."""
    if not samples:
        return default_k

    def log_likelihood(k: float) -> float:
        ll = 0.0
        for spread, outcome in samples:
            p = logistic_win_prob(spread, k)
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
