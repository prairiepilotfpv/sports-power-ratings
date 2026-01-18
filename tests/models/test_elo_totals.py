import math

import pandas as pd
import pytest

from models.elo import EloModel
from backtest.runner import _predictions_to_frame
from eval.validation import validate_prediction_row
from pipelines.tuning import run_tuning_pipeline


def _elo_training_frame() -> pd.DataFrame:
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


def test_elo_emits_totals_and_sd() -> None:
    model = EloModel()
    model.fit(_elo_training_frame())
    predictions = model.predict(_upcoming_frame())

    assert len(predictions) == 1
    pred = predictions[0]

    # Ensure totals fields are numeric and respect floor
    assert pred.pred_total is not None and isinstance(pred.pred_total, float)
    assert pred.total_mean is not None and isinstance(pred.total_mean, float)
    assert pred.total_sd is not None and isinstance(pred.total_sd, float)
    assert pred.total_sd >= model._total_sd_floor


def test_elo_tuning_produces_numeric_total_metric(tmp_path) -> None:
    outputs = run_tuning_pipeline(
        csv_path="tests/fixtures/mini_nba.csv",
        model="elo",
        start_date="2024-10-24",
        end_date="2024-11-05",
        metric="mae_total",
        output_dir=tmp_path,
        sport="nba",
        season="2024-25",
    )

    df = outputs.results
    assert not df.empty
    assert "mae_total" in df.columns
    vals = pd.to_numeric(df["mae_total"], errors="coerce")
    assert vals.notna().any()
