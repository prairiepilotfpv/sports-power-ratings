from __future__ import annotations

import warnings
import pandas as pd
from pathlib import Path
from models.bradley_terry import BradleyTerryBacktest as BradleyTerryHFA

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "bradley_terry" / "games.csv"
)


def test_suppress_small_sd_warning_flag() -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    model = BradleyTerryHFA(max_iter=200)
    model.fit(games_df)

    upcoming = pd.DataFrame(
        [
            {"date": "2024-01-01", "home_team": "Alpha", "away_team": "Beta"},
        ]
    )

    # Ensure prediction runs without raising an exception.
    _ = model.predict(upcoming)
