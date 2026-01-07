from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.bradley_terry_calibrated_hfa import BradleyTerryCalibratedHFA

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "bradley_terry_hfa" / "games.csv"
)


def test_bradley_terry_calibrated_hfa_predictions_basic() -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    model = BradleyTerryCalibratedHFA(max_iter=200)
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
        assert prediction.pred_margin is not None
        assert np.isfinite(prediction.pred_margin)
        assert 0.0 <= prediction.p_home_win <= 1.0
        assert prediction.total_mean is not None
        assert prediction.total_sd is not None
        assert prediction.extra["win_prob_source"] == "bt_margin_normal"
