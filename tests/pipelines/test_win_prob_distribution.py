from __future__ import annotations

import pytest

from pipelines.projections import win_prob_distribution


def _mean(dist: list[dict[str, float]]) -> float:
    return sum(bucket["p_home_win"] * bucket["weight"] for bucket in dist)


def test_win_prob_distribution_sums_to_one() -> None:
    dist = win_prob_distribution(0.65, win_prob_k=12.0, margin_std=8.0)
    assert sum(bucket["weight"] for bucket in dist) == pytest.approx(1.0)


def test_win_prob_distribution_matches_mean_probability() -> None:
    p_home_win = 0.72
    dist = win_prob_distribution(p_home_win, win_prob_k=9.0, margin_std=6.0)
    assert _mean(dist) == pytest.approx(p_home_win, abs=1e-6)


def test_win_prob_distribution_is_monotonic_with_probability() -> None:
    lower = win_prob_distribution(0.35, win_prob_k=10.0, margin_std=7.0)
    higher = win_prob_distribution(0.65, win_prob_k=10.0, margin_std=7.0)
    assert _mean(higher) > _mean(lower)


def test_win_prob_distribution_falls_back_to_point_mass() -> None:
    dist = win_prob_distribution(0.51, win_prob_k=None, margin_std=None)
    assert dist == [{"p_home_win": 0.51, "weight": 1.0}]
