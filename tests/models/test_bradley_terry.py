from __future__ import annotations

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
