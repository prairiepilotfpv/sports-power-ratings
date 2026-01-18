import math

import pandas as pd
import pytest

from backtest.runner import _predictions_to_frame
from eval.validation import validate_prediction_row
from pipelines.tuning import run_tuning_pipeline
from models.toor import TOORModel


def _toor_training_frame() -> pd.DataFrame:
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


def test_toor_projection_includes_projected_scores_and_totals() -> None:
    model = TOORModel()
    model.fit(_toor_training_frame())
    predictions = model.predict(_upcoming_frame())

    assert len(predictions) == 1
    pred = predictions[0]
    extra = pred.extra

    # Contract A: canonical heads present and finite
    assert "projected_home_score" in extra
    assert "projected_away_score" in extra
    assert "projected_total" in extra
    assert math.isfinite(pred.pred_margin)
    assert math.isfinite(pred.pred_total)
    assert math.isfinite(pred.p_home_win)
    assert math.isfinite(pred.margin_sd)
    assert math.isfinite(pred.total_sd)

    # projections must be consistent
    assert extra["projected_total"] == pytest.approx(pred.pred_total, rel=0)
    summed = extra["projected_home_score"] + extra["projected_away_score"]
    assert math.isclose(summed, pred.pred_total, abs_tol=1e-6)


def test_toor_prediction_validates_without_total_inconsistent() -> None:
    model = TOORModel()
    model.fit(_toor_training_frame())
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

    # Contract B: schedule-facing prob stream and source
    assert row.get("model_p_home_win") == pytest.approx(row.get("p_home_win") or row.get("model_p_home_win"))
    assert row.get("win_prob_source") == "logistic"


def test_toor_tuning_produces_numeric_total_metric(tmp_path) -> None:
    # run a small tuning run for the TOTAL metric and ensure numeric outputs
    outputs = run_tuning_pipeline(
        csv_path="tests/fixtures/mini_nba.csv",
        model="toor",
        start_date="2024-10-24",
        end_date="2024-11-05",
        metric="mae_total",
        output_dir=tmp_path,
        sport="nba",
        season="2024-25",
    )

    df = outputs.results
    assert not df.empty
    # mae_total column should exist and contain at least one finite numeric value
    assert "mae_total" in df.columns
    vals = pd.to_numeric(df["mae_total"], errors="coerce")
    assert vals.notna().any()


def test_toor_uses_league_total_mean_fallback_when_missing() -> None:
    """Ensure TOOR always emits a numeric pred_total when _total_mean is None and
    that projected home/away scores are computed from the numeric pred_total.
    """
    import math

    from config import DEFAULT_TOTAL_MEAN_FALLBACK

    model = TOORModel()
    model.fit(_toor_training_frame())

    # simulate the structural failure where the fitted total mean is unavailable
    model._total_mean = None
    model._total_sd = None

    predictions = model.predict(_upcoming_frame())
    assert len(predictions) == 1
    pred = predictions[0]
    extra = pred.extra

    # pred_total must be numeric and equal the league fallback
    assert math.isfinite(pred.pred_total)
    assert pred.pred_total == pytest.approx(DEFAULT_TOTAL_MEAN_FALLBACK)

    # projected scores must be computed from the numeric pred_total and pred_margin
    expected_home = 0.5 * (pred.pred_total + pred.pred_margin)
    expected_away = 0.5 * (pred.pred_total - pred.pred_margin)
    assert extra["projected_home_score"] == pytest.approx(expected_home)
    assert extra["projected_away_score"] == pytest.approx(expected_away)

    # validator should not mark total_inconsistent and downstream MAE_total calculations
    # will receive finite pred_total (covered by other integration tests)
    frame = _predictions_to_frame(predictions)
    ok, reasons = validate_prediction_row(frame.iloc[0].to_dict())
    assert "total_inconsistent" not in reasons, f"reasons={reasons}"