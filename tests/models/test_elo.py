from __future__ import annotations

from models.elo import EloPowerRating


def test_elo_rankings_order() -> None:
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
            "home_score": 95,
            "away_score": 80,
        },
        {
            "home_team": "Alpha",
            "away_team": "Gamma",
            "home_score": 102,
            "away_score": 88,
        },
    ]
    model = EloPowerRating(k_factor=32.0)
    model.fit(games)

    rankings = model.rankings()
    assert [team for team, _ in rankings] == ["Alpha", "Beta", "Gamma"]


def test_elo_fit_produces_positive_ratings() -> None:
    games = [
        {
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_score": 70,
            "away_score": 100,
        },
        {
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_score": 65,
            "away_score": 110,
        },
    ]
    model = EloPowerRating(k_factor=400.0, initial_rating=50.0, min_rating=1.0)
    model.fit(games)

    ratings = dict(model.rankings())
    assert set(ratings) == {"Alpha", "Beta"}
    assert all(rating > 0 for rating in ratings.values())
