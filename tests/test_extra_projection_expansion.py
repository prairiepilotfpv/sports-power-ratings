import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))

from models.base import GamePrediction
from backtest.runner import _predictions_to_frame
from eval.validation import validate_prediction_row, get_validation_config


def make_prediction_with_extra():
    meta = {
        "model_id": "bradley-terry_hfa",
        "model_version": "1.0",
        "params": {},
    }
    # Build a GamePrediction where projected scores live only in `extra`.
    extra = {
        "projected_home_score": 3.5,
        "projected_away_score": 2.5,
        "projected_total": 6.0,
        "margin_mean": 1.0,
        "margin_sd": 1.5,
        "total_sd": 2.0,
        "win_prob_source": "bt_margin_normal",
    }
    gp = GamePrediction(
        game_id="nhl|2025-10-01|A|B",
        date="2025-10-01",
        home_team="A",
        away_team="B",
        p_home_win=0.7,
        win_prob_samples=None,
        win_prob_dist=None,
        pred_margin=None,
        pred_total=None,
        margin_mean=None,
        margin_sd=None,
        total_mean=None,
        total_sd=None,
        win_prob_source=None,
        margin_dist_assumption=None,
        metadata=meta,
        extra=extra,
    )
    return gp


def test_extra_projection_promoted_and_validated():
    pred = make_prediction_with_extra()
    df = _predictions_to_frame([pred])
    # promoted columns should exist
    assert "projected_home_score" in df.columns or "projected_home_score" in df
    assert "projected_away_score" in df.columns or "projected_away_score" in df
    assert "projected_total" in df.columns or "projected_total" in df

    # Validate prediction row using NHL config — should pass total consistency
    row = df.iloc[0].to_dict()
    config = get_validation_config("nhl")
    ok, reasons = validate_prediction_row(row, config=config, require_score_bounds=True)
    assert ok, f"Validation failed: {reasons}"
