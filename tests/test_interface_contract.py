from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.registry import get_backtest_model, list_backtest_models

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "mini_nba.csv"


@pytest.mark.parametrize("model_name", list_backtest_models())
def test_backtest_model_predictions_contract(model_name: str) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date

    cutoff_date = sorted(games_df["date"].unique())[15]
    train_df = games_df[games_df["date"] < cutoff_date]
    upcoming_df = games_df[games_df["date"] >= cutoff_date].drop(
        columns=["home_score", "away_score"],
        errors="ignore",
    )

    model_cls = get_backtest_model(model_name)
    model = model_cls()
    model.fit(train_df)
    predictions = model.predict(upcoming_df)

    assert predictions, "Expected predictions for the fixture dataset."

    game_ids = []
    for prediction in predictions:
        assert prediction.game_id
        assert prediction.date
        assert prediction.home_team
        assert prediction.away_team
        assert prediction.p_home_win is not None
        assert 0.0 <= prediction.p_home_win <= 1.0
        game_ids.append(prediction.game_id)

    assert len(game_ids) == len(set(game_ids)), "game_id values must be unique."
