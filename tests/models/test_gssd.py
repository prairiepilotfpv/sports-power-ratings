from __future__ import annotations

import math

import pandas as pd
import pytest

from models.gssd import GSSDCalibration, GSSDModel, GSSDPowerRating


def test_gssd_build_ratings_uses_net_rating() -> None:
    model = GSSDPowerRating()
    model._model.team_ratings_ = {
        "Alpha": (110.0, 100.0, 108.0, 102.0),
        "Beta": (95.0, 105.0, 98.0, 110.0),
    }

    ratings = model._build_ratings()

    assert ratings["Alpha"] == pytest.approx(8.0)
    assert ratings["Beta"] == pytest.approx(-11.0)


def test_gssd_predict_handles_negative_ratings() -> None:
    model = GSSDModel()
    model._gssd._ratings = {"Alpha": -2.0, "Beta": 1.0}
    model._coefficients = GSSDCalibration(
        home_advantage_points=1.0,
        scale=2.0,
        error_term=3.0,
    )
    model._win_prob_k = 10.0

    df = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "neutral": False,
            }
        ]
    )

    predictions = model.predict(df)

    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.pred_margin == pytest.approx(-5.0)
    expected_prob = 1.0 / (1.0 + math.exp(5.0 / 10.0))
    assert prediction.p_home_win == pytest.approx(expected_prob)
    assert prediction.win_prob_dist
