from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.bradley_terry import BradleyTerryBacktest as BradleyTerryHFA

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "bradley_terry" / "games.csv"
)


def test_bradley_terry_hfa_fit_fixture_dataset() -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    model = BradleyTerryHFA(max_iter=200)

    model.fit(games_df)


def test_bradley_terry_hfa_predictions_stable() -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    model = BradleyTerryHFA(max_iter=200)
    model.fit(games_df)

    upcoming_df = pd.DataFrame(
        [
            {"date": "2024-01-01", "home_team": "Alpha", "away_team": "Beta"},
            {
                "date": "2024-01-02",
                "home_team": "Gamma",
                "away_team": "Alpha",
                "neutral": True,
            },
        ]
    )

    predictions = model.predict(upcoming_df)

    assert len(predictions) == 2
    for prediction in predictions:
        assert 0.0 < prediction.p_home_win < 1.0
        assert prediction.margin_mean == pytest.approx(prediction.pred_margin)
        assert prediction.margin_sd is not None
        assert prediction.total_mean is not None
        assert prediction.total_sd is not None
        projected_home = prediction.extra["projected_home_score"]
        projected_away = prediction.extra["projected_away_score"]
        assert projected_home - projected_away == pytest.approx(prediction.margin_mean)
        assert projected_home + projected_away == pytest.approx(prediction.total_mean)
        assert prediction.extra["win_prob_source"] == "direct"
        assert prediction.win_prob_source == "direct"
        assert prediction.extra["margin_dist_assumption"] == "normal_approx"
