from __future__ import annotations

import math

import pandas as pd
import pytest

from models.gssd import GSSDCalibration, GSSDModel, GSSDPowerRating


def test_gssd_power_rating_builds_team_stats() -> None:
    model = GSSDPowerRating()
    games = [
        {
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_score": 100,
            "away_score": 90,
        },
        {
            "home_team": "Beta",
            "away_team": "Alpha",
            "home_score": 95,
            "away_score": 105,
        },
    ]

    model.fit(games)

    alpha_stats = model.team_stats("Alpha")
    beta_stats = model.team_stats("Beta")

    assert alpha_stats["pfh"] == pytest.approx(100.0)
    assert alpha_stats["pah"] == pytest.approx(90.0)
    assert alpha_stats["pfa"] == pytest.approx(105.0)
    assert alpha_stats["paa"] == pytest.approx(95.0)
    assert beta_stats["pfh"] == pytest.approx(95.0)
    assert beta_stats["pah"] == pytest.approx(105.0)
    assert beta_stats["pfa"] == pytest.approx(90.0)
    assert beta_stats["paa"] == pytest.approx(100.0)


def test_gssd_predict_uses_coefficients() -> None:
    model = GSSDModel()
    model._gssd._team_stats = {
        "Alpha": {"pfh": 100.0, "pah": 90.0, "pfa": 95.0, "paa": 105.0},
        "Beta": {"pfh": 98.0, "pah": 102.0, "pfa": 92.0, "paa": 108.0},
    }
    model._gssd._league_stats = {
        "pfh": 99.0,
        "pah": 101.0,
        "pfa": 93.0,
        "paa": 107.0,
    }
    model._coefficients = GSSDCalibration(
        intercept=1.0,
        beta_pfh=0.1,
        beta_pah=0.2,
        beta_pfa=-0.05,
        beta_paa=-0.1,
        home_advantage_points=0.0,
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
    expected_margin = (
        1.0
        + 0.1 * 100.0
        + 0.2 * 90.0
        - 0.05 * 92.0
        - 0.1 * 108.0
    )
    assert prediction.pred_margin == pytest.approx(expected_margin)
    expected_prob = 1.0 / (1.0 + math.exp(-expected_margin / 10.0))
    assert prediction.p_home_win == pytest.approx(expected_prob)
    assert prediction.win_prob_dist


def test_gssd_predict_populates_totals_head() -> None:
    model = GSSDModel()
    games = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 100,
                "away_score": 90,
            },
            {
                "date": "2025-01-02",
                "home_team": "Beta",
                "away_team": "Alpha",
                "home_score": 95,
                "away_score": 105,
            },
        ]
    )

    model.fit(games)

    preds = model.predict(
        pd.DataFrame(
            [
                {
                    "date": "2025-01-03",
                    "home_team": "Alpha",
                    "away_team": "Beta",
                }
            ]
        )
    )

    assert len(preds) == 1
    prediction = preds[0]
    # Expected totals from offense/defense blending
    assert prediction.pred_total == pytest.approx(190.0)
    assert prediction.total_mean == pytest.approx(190.0)
    # Fit-derived total SD from training totals [190, 200] => sd = 5
    assert prediction.total_sd is not None
    assert prediction.total_sd == pytest.approx(5.0)
