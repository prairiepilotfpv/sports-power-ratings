import math

import pandas as pd
import pytest

from backtest.runner import _predictions_to_frame
from eval.validation import validate_prediction_row
from models.gssd import GSSDModel


def _gssd_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-10-25",
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 112,
                "away_score": 105,
            },
            {
                "date": "2025-10-26",
                "home_team": "Away",
                "away_team": "Home",
                "home_score": 99,
                "away_score": 101,
            },
            {
                "date": "2025-10-27",
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 108,
                "away_score": 110,
            },
        ]
    )


def _upcoming_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "home_team": "Home",
                "away_team": "Away",
            }
        ]
    )


def test_gssd_projection_includes_projected_scores() -> None:
    model = GSSDModel()
    model.fit(_gssd_training_frame())
    predictions = model.predict(_upcoming_frame())

    assert len(predictions) == 1
    pred = predictions[0]
    extra = pred.extra

    assert "projected_home_score" in extra
    assert "projected_away_score" in extra
    assert "projected_total" in extra
    assert math.isfinite(pred.pred_margin)
    assert math.isfinite(pred.pred_total)
    assert math.isfinite(pred.p_home_win)
    assert math.isfinite(pred.margin_sd)
    assert math.isfinite(pred.total_sd)

    assert extra["projected_total"] == pytest.approx(pred.pred_total, rel=0)
    summed = extra["projected_home_score"] + extra["projected_away_score"]
    assert math.isclose(summed, pred.pred_total, abs_tol=1e-6)


def test_gssd_prediction_validates_without_total_inconsistent() -> None:
    model = GSSDModel()
    model.fit(_gssd_training_frame())
    predictions = model.predict(_upcoming_frame())
    frame = _predictions_to_frame(predictions)
    assert not frame.empty
    row = frame.iloc[0].to_dict()

    ok, reasons = validate_prediction_row(row)
    assert "total_inconsistent" not in reasons, f"reasons={reasons}"
    projected_home = row.get("projected_home_score")
    projected_away = row.get("projected_away_score")
    projected_total = row.get("projected_total")
    assert projected_home is not None
    assert projected_away is not None
    assert projected_total is not None
    summed = float(projected_home) + float(projected_away)
    assert math.isclose(summed, float(projected_total), abs_tol=1e-6)