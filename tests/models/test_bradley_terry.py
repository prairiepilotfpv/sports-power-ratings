from __future__ import annotations

from math import isfinite

import pytest

from models.bradley_terry import BradleyTerry


def test_bradley_terry_rankings_order() -> None:
    games = [
        {
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_score": 100,
            "away_score": 90,
        },
        {
            "home_team": "Beta",
            "away_team": "Gamma",
            "home_score": 80,
            "away_score": 70,
        },
        {
            "home_team": "Alpha",
            "away_team": "Gamma",
            "home_score": 95,
            "away_score": 85,
        },
    ]
    model = BradleyTerry(max_iter=200)
    model.fit(games)

    rankings = model.rankings()
    assert [team for team, _ in rankings] == ["Alpha", "Beta", "Gamma"]


def test_predict_probability_symmetry_neutral() -> None:
    model = BradleyTerry()
    model.ratings.update({"Alpha": 1.7, "Beta": 0.9})

    prob_alpha = model.predict_probability("Alpha", "Beta", venue="neutral")
    prob_beta = model.predict_probability("Beta", "Alpha", venue="neutral")

    assert prob_alpha + prob_beta == pytest.approx(1.0, abs=1e-8)


def test_hfa_directional_effects() -> None:
    model = BradleyTerry()
    model.ratings.update({"Alpha": 1.5, "Beta": 1.2})
    model.home_adv = 0.35

    neutral_prob = model.predict_probability("Alpha", "Beta", venue="neutral")
    home_prob = model.predict_probability("Alpha", "Beta", venue="home")
    away_prob = model.predict_probability("Alpha", "Beta", venue="away")

    assert home_prob >= neutral_prob
    assert away_prob <= neutral_prob


def test_monotonicity_in_team_strength() -> None:
    model = BradleyTerry()
    model.ratings.update({"Alpha": 1.0, "Beta": 1.0})

    base_prob = model.predict_probability("Alpha", "Beta", venue="neutral")

    model.ratings["Alpha"] = 2.0
    stronger_prob = model.predict_probability("Alpha", "Beta", venue="neutral")

    assert stronger_prob > base_prob


def test_extreme_ratings_no_overflow() -> None:
    model = BradleyTerry()
    model.ratings.update({"Alpha": 1e12, "Beta": 1e-12})

    prob_alpha = model.predict_probability("Alpha", "Beta", venue="neutral")
    prob_beta = model.predict_probability("Beta", "Alpha", venue="neutral")

    assert isfinite(prob_alpha)
    assert isfinite(prob_beta)
    assert prob_alpha >= 1.0 - 1e-9
    assert prob_beta <= 1e-9


def test_deterministic_predictions() -> None:
    games = [
        {
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_score": 100,
            "away_score": 90,
        },
        {
            "home_team": "Beta",
            "away_team": "Gamma",
            "home_score": 80,
            "away_score": 70,
        },
        {
            "home_team": "Gamma",
            "away_team": "Alpha",
            "home_score": 65,
            "away_score": 75,
            "neutral": True,
        },
    ]
    model = BradleyTerry(max_iter=200)
    model.fit(games)

    first = model.predict_probability("Alpha", "Beta", venue="neutral")
    second = model.predict_probability("Alpha", "Beta", venue="neutral")

    assert first == pytest.approx(second, abs=0.0)
