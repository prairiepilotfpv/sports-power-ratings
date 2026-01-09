import logging

import pandas as pd

from config import (
    LEAGUE_MARGIN_SD_DEFAULT,
    MARGIN_SD_GUARDRAIL_MAX,
    MARGIN_SD_GUARDRAIL_MIN,
)
from models.calibration import ConditionalSDModel
from models.toor import TOORModel


def test_toor_margin_sd_never_below_guardrail():
    games = pd.DataFrame([
        {
            "date": pd.to_datetime("2024-01-01"),
            "home_team": "Lions",
            "away_team": "Tigers",
            "home_score": 110,
            "away_score": 108,
        },
        {
            "date": pd.to_datetime("2024-01-02"),
            "home_team": "Tigers",
            "away_team": "Lions",
            "home_score": 107,
            "away_score": 105,
        },
    ])
    model = TOORModel()
    model.fit(games)

    upcoming = pd.DataFrame([
        {
            "date": pd.to_datetime("2024-01-03"),
            "home_team": "Lions",
            "away_team": "Tigers",
        }
    ])
    predictions = model.predict(upcoming)
    assert predictions
    for pred in predictions:
        assert pred.margin_sd >= MARGIN_SD_GUARDRAIL_MIN


def test_toor_margin_sd_guardrail_logs_reason(caplog):
    games = pd.DataFrame(
        [
            {
                "date": pd.to_datetime("2024-01-01"),
                "home_team": "Lions",
                "away_team": "Tigers",
                "home_score": 110,
                "away_score": 90,
            },
            {
                "date": pd.to_datetime("2024-01-02"),
                "home_team": "Tigers",
                "away_team": "Lions",
                "home_score": 95,
                "away_score": 115,
            },
        ]
    )
    model = TOORModel()
    model.fit(games)
    # Force a pathological conditional SD that would previously yield a 1.0 floor.
    model._conditional_sd_model = ConditionalSDModel(
        intercept=-5.0,
        slope=-1.0,
        min_sd=MARGIN_SD_GUARDRAIL_MIN,
        fallback_sd=LEAGUE_MARGIN_SD_DEFAULT,
        n_samples=2,
        residual_min=-1.0,
        residual_max=1.0,
    )

    caplog.set_level(logging.DEBUG)
    upcoming = pd.DataFrame(
        [
            {
                "date": pd.to_datetime("2024-01-03"),
                "home_team": "Lions",
                "away_team": "Tigers",
            }
        ]
    )
    predictions = model.predict(upcoming)
    assert predictions
    assert predictions[0].margin_sd >= MARGIN_SD_GUARDRAIL_MIN
    assert predictions[0].margin_sd == LEAGUE_MARGIN_SD_DEFAULT
    assert any("margin_sd guardrail applied" in rec.message for rec in caplog.records)


def test_toor_margin_sd_stays_within_bounds_with_conditional():
    games = pd.DataFrame(
        [
            {
                "date": pd.to_datetime("2024-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 120,
                "away_score": 100,
            },
            {
                "date": pd.to_datetime("2024-01-02"),
                "home_team": "B",
                "away_team": "A",
                "home_score": 105,
                "away_score": 102,
            },
            {
                "date": pd.to_datetime("2024-01-03"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 111,
                "away_score": 109,
            },
        ]
    )
    model = TOORModel(conditional_sd=True)
    model.fit(games)

    upcoming = pd.DataFrame(
        [
            {
                "date": pd.to_datetime("2024-01-04"),
                "home_team": "A",
                "away_team": "B",
            },
            {
                "date": pd.to_datetime("2024-01-05"),
                "home_team": "B",
                "away_team": "A",
            },
        ]
    )
    predictions = model.predict(upcoming)
    assert predictions
    for pred in predictions:
        assert MARGIN_SD_GUARDRAIL_MIN <= pred.margin_sd <= MARGIN_SD_GUARDRAIL_MAX
