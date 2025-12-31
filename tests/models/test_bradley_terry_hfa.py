from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.bradley_terry_hfa import BradleyTerryHFA

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "bradley_terry_hfa" / "games.csv"
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
    assert predictions[0].p_home_win == pytest.approx(0.8979314585495295, abs=1e-12)
    assert predictions[0].pred_margin == pytest.approx(2.1744491759921276, abs=1e-12)
    assert predictions[1].p_home_win == pytest.approx(0.012756215618074503, abs=1e-12)
    assert predictions[1].pred_margin == pytest.approx(-4.348898351984255, abs=1e-12)
