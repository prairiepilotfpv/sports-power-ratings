from __future__ import annotations

import math

import pandas as pd

from pipelines.projections import (
    average_total_points,
    fit_win_prob_scale,
    matchup_total_from_averages,
    project_game,
    team_scoring_averages,
)
from pipelines.schedule import _project_row


def test_margin_neutral_is_rating_diff() -> None:
    projection = project_game(
        5.0, -3.0, home_advantage=2.0, neutral=True, base_total=200.0
    )
    assert projection.margin_neutral == 8.0


def test_margin_adds_home_advantage_when_not_neutral() -> None:
    neutral_projection = project_game(
        4.0, 1.0, home_advantage=2.5, neutral=True, base_total=200.0
    )
    non_neutral_projection = project_game(
        4.0, 1.0, home_advantage=2.5, neutral=False, base_total=200.0
    )
    assert neutral_projection.margin == 3.0
    assert non_neutral_projection.margin == 5.5


def test_projected_spread_is_negative_margin() -> None:
    projection = project_game(
        6.0, 2.0, home_advantage=1.0, neutral=False, base_total=200.0
    )
    assert projection.projected_spread == -projection.margin
    assert projection.projected_home_spread == projection.margin


def test_projected_scores_difference_equals_margin() -> None:
    projection = project_game(
        6.0, 2.0, home_advantage=1.0, neutral=False, base_total=200.0
    )
    diff = projection.projected_home_score - projection.projected_away_score
    assert math.isclose(diff, projection.margin, rel_tol=1e-9)


def test_win_prob_monotonic_and_half_at_zero_margin() -> None:
    low = project_game(
        1.0, 3.0, home_advantage=0.0, neutral=True, k=10.0, base_total=200.0
    )
    mid = project_game(
        2.0, 2.0, home_advantage=0.0, neutral=True, k=10.0, base_total=200.0
    )
    high = project_game(
        3.0, 1.0, home_advantage=0.0, neutral=True, k=10.0, base_total=200.0
    )
    assert low.projected_win_prob < mid.projected_win_prob < high.projected_win_prob
    assert math.isclose(mid.projected_win_prob, 0.5, rel_tol=1e-9)


def test_win_prob_uses_away_minus_home_spread() -> None:
    home_favored = project_game(
        5.0, 1.0, home_advantage=0.0, neutral=True, k=10.0, base_total=200.0
    )
    away_favored = project_game(
        1.0, 5.0, home_advantage=0.0, neutral=True, k=10.0, base_total=200.0
    )
    assert home_favored.projected_spread < 0
    assert away_favored.projected_spread > 0
    assert home_favored.projected_win_prob > 0.5
    assert away_favored.projected_win_prob < 0.5


def test_integration_project_row_sign_conventions() -> None:
    row = pd.Series(
        {
            "date": "2024-01-01",
            "home_team": "Home",
            "away_team": "Away",
            "home_score": None,
            "away_score": None,
            "neutral": False,
            "overtime": False,
            "game_id": "game-1",
        }
    )
    output = _project_row(
        row,
        ratings={"Home": 5.0, "Away": -3.0},
        base_total=200.0,
        scoring_averages={},
        status="scheduled",
        home_advantage=2.0,
        params_source="default",
        tuned_metric_used=None,
        win_prob_k=10.0,
    )
    assert output["projected_winner"] == "Home"
    assert output["projected_spread"] == -10.0
    assert output["projected_home_spread"] == 10.0
    assert output["projected_home_score"] == 105.0
    assert output["projected_away_score"] == 95.0
    assert output["projected_total"] == 200.0


def test_average_total_points_handles_empty() -> None:
    assert average_total_points([]) == 0.0


def test_average_total_points_computes_mean() -> None:
    rows = [
        {"home_score": 100, "away_score": 90},
        {"home_score": 95, "away_score": 105},
    ]
    assert average_total_points(rows) == 195.0


def test_team_scoring_averages_and_matchup_total() -> None:
    rows = [
        {"home_team": "A", "away_team": "B", "home_score": 100, "away_score": 90},
        {"home_team": "B", "away_team": "A", "home_score": 95, "away_score": 105},
    ]
    averages = team_scoring_averages(rows)
    assert set(averages.keys()) == {"A", "B"}

    total = matchup_total_from_averages("A", "B", averages)
    assert total is not None
    assert total > 0
    assert matchup_total_from_averages("A", "C", averages) is None


def test_fit_win_prob_scale_defaults_when_empty() -> None:
    assert fit_win_prob_scale([]) > 0


def test_fit_win_prob_scale_bounds() -> None:
    samples = [(-5.0, 1), (5.0, 0), (-2.5, 1), (2.5, 0)]
    k = fit_win_prob_scale(samples, min_k=0.5, max_k=50.0)
    assert 0.5 <= k <= 50.0
