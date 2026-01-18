import math

import pandas as pd
import pytest

from models.toor import TOORModel, DEFAULT_COEFFICIENTS
from backtest.runner import _predictions_to_frame
from eval.validation import validate_prediction_row


def _simple_hfa_training() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-10-25",
                "home_team": "H1",
                "away_team": "A1",
                "home_score": 110,
                "away_score": 100,
            },
            {
                "date": "2025-10-26",
                "home_team": "A1",
                "away_team": "H1",
                "home_score": 102,
                "away_score": 100,
            },
        ]
    )


def _strong_hfa_training() -> pd.DataFrame:
    # contrived: large consistent home margins to force a learned HFA >> default
    return pd.DataFrame(
        [
            {"date": "2025-10-01", "home_team": "T1", "away_team": "T2", "home_score": 120, "away_score": 90},
            {"date": "2025-10-02", "home_team": "T3", "away_team": "T4", "home_score": 118, "away_score": 92},
            {"date": "2025-10-03", "home_team": "T2", "away_team": "T1", "home_score": 95, "away_score": 94},
            {"date": "2025-10-04", "home_team": "T4", "away_team": "T3", "home_score": 96, "away_score": 95},
        ]
    )


def _upcoming_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": "2026-01-05", "home_team": "T1", "away_team": "T2"}]
    )


def test_toor_hfa_fixed_uses_default_and_predicts():
    model = TOORModel(learn_home_advantage=False)
    tf = _simple_hfa_training()
    model.fit(tf)

    assert math.isfinite(model._coefficients.home_advantage)
    assert model._coefficients.home_advantage == pytest.approx(
        DEFAULT_COEFFICIENTS.home_advantage
    )

    preds = model.predict(_upcoming_frame())
    assert preds
    frame = _predictions_to_frame(preds)
    ok, reasons = validate_prediction_row(frame.iloc[0].to_dict())
    assert ok is True


def test_toor_hfa_learns_when_enabled_and_is_finite():
    model = TOORModel(learn_home_advantage=True)
    model.fit(_strong_hfa_training())

    assert math.isfinite(model._coefficients.home_advantage)
    # with contrived data, learned HFA should differ noticeably from default
    assert abs(model._coefficients.home_advantage - DEFAULT_COEFFICIENTS.home_advantage) > 0.1

    preds = model.predict(_upcoming_frame())
    assert preds
    frame = _predictions_to_frame(preds)
    ok, reasons = validate_prediction_row(frame.iloc[0].to_dict())
    assert ok is True


def test_toor_hfa_both_modes_produce_probabilities():
    for learn in (False, True):
        model = TOORModel(learn_home_advantage=learn)
        model.fit(_simple_hfa_training())
        preds = model.predict(_upcoming_frame())
        assert preds
        p = preds[0].p_home_win
        assert 0.0 < p < 1.0
