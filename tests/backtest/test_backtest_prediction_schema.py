from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.runner import run_backtest
from models.registry import get_backtest_model

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mini_nba.csv"


def test_backtest_prediction_schema_smoke(tmp_path: Path) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date

    dates = sorted(games_df["date"].unique())
    start_date = dates[1].isoformat()

    model_cls = get_backtest_model("elo")
    outputs = run_backtest(
        model_factory=model_cls,
        games_df=games_df,
        start_date=start_date,
        end_date=start_date,
        output_dir=tmp_path / "elo",
        model_name="elo",
    )

    required_columns = {
        "game_id",
        "date",
        "home_team",
        "away_team",
        "p_home_win",
        "win_prob_samples",
        "win_prob_dist",
        "pred_margin",
        "pred_total",
        "margin_mean",
        "margin_sd",
        "total_mean",
        "total_sd",
        "model_win_prob",
        "model_id",
    }

    assert not outputs.predictions.empty
    assert required_columns.issubset(set(outputs.predictions.columns))
