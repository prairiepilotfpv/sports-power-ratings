from __future__ import annotations

import pytest

from models.elo import Elo


def test_elo_fit_and_rankings() -> None:
    games = [
        {
            "home_team": "Alpha",
            "visitor_team": "Beta",
            "home_pts": 100,
            "visitor_pts": 90,
            "neutral": False,
        },
        {
            "home_team": "Beta",
            "visitor_team": "Gamma",
            "home_pts": 80,
            "visitor_pts": 70,
            "neutral": False,
        },
    ]
    model = Elo()
    model.fit(games)

    rankings = model.rankings()
    assert [team for team, _ in rankings] == ["Alpha", "Beta", "Gamma"]

    prob, spread = model.predict("Alpha", "Gamma", neutral=True)
    assert prob == pytest.approx(0.549226399580097, rel=1e-6)
    assert spread == pytest.approx(2.9169596401516698, rel=1e-6)
