from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from models.bradley_terry_hfa import BradleyTerryHFA


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "bradley_terry_hfa"
    / "nhl_games.csv"
)


def _build_games_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # adapt Sports-Reference-like CSV to expected columns
    df2 = pd.DataFrame()
    df2["date"] = df["Date"]
    df2["home_team"] = df["Home"]
    df2["away_team"] = df["Visitor"]
    # pick scores from G / G.1 columns
    if "G" in df.columns and "G.1" in df.columns:
        df2["away_score"] = df["G"]
        df2["home_score"] = df["G.1"]
    else:
        df2["away_score"] = df.iloc[:, 3]
        df2["home_score"] = df.iloc[:, 5]
    return df2


def test_nhl_does_not_emit_small_sd_warnings() -> None:
    games_df = _build_games_df(DATA_PATH)
    # Use a modest number of iterations to keep the test fast
    model = BradleyTerryHFA(max_iter=200)
    model.fit(games_df)

    # Build two upcoming matchups
    upcoming = pd.DataFrame(
        [
            {"date": "2025-12-25", "home_team": str(games_df.loc[0, "home_team"]), "away_team": str(games_df.loc[0, "away_team"])},
            {"date": "2025-12-26", "home_team": str(games_df.loc[1, "home_team"]), "away_team": str(games_df.loc[1, "away_team"])},
        ]
    )

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        preds = model.predict(upcoming)

    # Ensure predictions were made
    assert len(preds) == 2

    # Ensure there are no RuntimeWarnings mentioning "Invalid BT prediction"
    assert not any(
        isinstance(w.message, Warning) and "Invalid BT prediction" in str(w.message)
        for w in rec
    )
