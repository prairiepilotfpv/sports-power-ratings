from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.runner import _extract_backtest_win_prob_k, run_backtest
from models.registry import get_backtest_model

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mini_nba.csv"


def test_elo_and_gssd_backtests_emit_win_prob_k(tmp_path: Path) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date

    dates = sorted(games_df["date"].unique())
    start_date = dates[1].isoformat()

    for model_name in ["elo", "gssd"]:
        model_cls = get_backtest_model(model_name)
        outputs = run_backtest(
            model_factory=model_cls,
            games_df=games_df,
            start_date=start_date,
            end_date=start_date,
            output_dir=tmp_path / model_name,
            model_name=model_name,
        )

        assert not outputs.predictions.empty
        win_prob_k = _extract_backtest_win_prob_k(outputs.predictions)
        assert win_prob_k is not None
        assert win_prob_k > 0
