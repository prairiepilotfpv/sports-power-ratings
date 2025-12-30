from pipelines import projections


def test_average_total_points_skips_invalid_rows() -> None:
    rows = [
        {"home_score": 10, "away_score": 20},
        {"home_score": "15", "away_score": 5},
        {"home_score": None, "away_score": 3},
    ]

    assert projections.average_total_points(rows) == 25.0


def test_team_scoring_averages_tracks_for_and_against() -> None:
    rows = [
        {"home_team": "A", "away_team": "B", "home_score": 10, "away_score": 5},
        {"home_team": "B", "away_team": "A", "home_score": 7, "away_score": 9},
    ]

    averages = projections.team_scoring_averages(rows)

    assert averages["A"] == (9.5, 6.0)
    assert averages["B"] == (6.0, 9.5)


def test_matchup_total_from_averages_returns_expected_total() -> None:
    averages = {
        "A": (9.5, 6.0),
        "B": (6.0, 9.5),
    }

    assert projections.matchup_total_from_averages("A", "B", averages) == 15.5
