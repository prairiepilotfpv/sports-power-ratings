from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.runner import run_backtest
from models.elo import EloModel
from models.registry import get_backtest_model

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mini_nba.csv"


def test_streaming_backtest_is_deterministic_for_elo(tmp_path: Path, monkeypatch) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date
    dates = sorted(games_df["date"].unique())
    start_date = dates[1].isoformat()
    end_date = dates[-1].isoformat()

    monkeypatch.delenv("ELO_STREAM_REFIT_DAYS", raising=False)
    monkeypatch.delenv("ELO_STREAM_REFIT_GAMES", raising=False)

    fit_calls: list[int] = []
    original_fit = EloModel.fit

    def counting_fit(self, games_df: pd.DataFrame) -> None:
        fit_calls.append(int(len(games_df)))
        original_fit(self, games_df)

    monkeypatch.setattr(EloModel, "fit", counting_fit)

    model_cls = get_backtest_model("elo")
    outputs1 = run_backtest(
        model_factory=model_cls,
        games_df=games_df,
        start_date=start_date,
        end_date=end_date,
        output_dir=tmp_path / "run1",
        model_name="elo",
    )
    outputs2 = run_backtest(
        model_factory=model_cls,
        games_df=games_df,
        start_date=start_date,
        end_date=end_date,
        output_dir=tmp_path / "run2",
        model_name="elo",
    )

    assert not outputs1.predictions.empty
    assert not outputs2.predictions.empty
    pd.testing.assert_frame_equal(outputs1.metrics_overall, outputs2.metrics_overall)
    assert len(fit_calls) == 2
