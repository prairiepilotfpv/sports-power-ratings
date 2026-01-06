from __future__ import annotations

from datetime import date

import pandas as pd

from models.poisson import PoissonModel, PoissonPowerRating


def _synthetic_games() -> list[dict[str, object]]:
    return [
        {
            "date": date(2024, 1, 1),
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_score": 3,
            "away_score": 1,
            "neutral": False,
        },
        {
            "date": date(2024, 1, 2),
            "home_team": "Beta",
            "away_team": "Gamma",
            "home_score": 2,
            "away_score": 2,
            "neutral": False,
        },
        {
            "date": date(2024, 1, 3),
            "home_team": "Gamma",
            "away_team": "Alpha",
            "home_score": 1,
            "away_score": 4,
            "neutral": True,
        },
    ]


def test_poisson_fit_expected_goals_positive() -> None:
    model = PoissonPowerRating(
        max_iter=500,
        learning_rate=0.1,
        reg_strength=0.1,
        random_seed=7,
    )
    model.fit(_synthetic_games())
    expected = model.expected_goals("Alpha", "Beta", neutral=False)
    assert expected is not None
    home_lambda, away_lambda = expected
    assert home_lambda > 0
    assert away_lambda > 0


def test_poisson_predict_probability_in_range() -> None:
    games = pd.DataFrame(_synthetic_games())
    model = PoissonModel(
        max_iter=500,
        learning_rate=0.1,
        reg_strength=0.1,
        n_simulations=5000,
        random_seed=3,
    )
    model.fit(games)

    upcoming = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 4),
                "home_team": "Alpha",
                "away_team": "Gamma",
                "neutral": False,
            }
        ]
    )
    predictions = model.predict(upcoming)
    assert predictions
    prediction = predictions[0]
    assert 0.0 <= prediction.p_home_win <= 1.0


def test_poisson_rankings_include_all_teams() -> None:
    model = PoissonPowerRating(
        max_iter=500,
        learning_rate=0.1,
        reg_strength=0.1,
        random_seed=11,
    )
    model.fit(_synthetic_games())
    ranked_teams = {team for team, _ in model.rankings()}
    assert ranked_teams == {"Alpha", "Beta", "Gamma"}
