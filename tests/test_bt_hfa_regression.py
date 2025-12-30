from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.runner import run_backtest
from models.bradley_terry_hfa import BradleyTerryHFA

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "mini_nba.csv"

# If this test fails due to expected model changes, recompute the
# metrics on the fixture dataset and update the thresholds below.
LOG_LOSS_THRESHOLD = 1.3
BRIER_SCORE_THRESHOLD = 0.35


def test_bt_hfa_backtest_metrics(tmp_path: Path) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    outputs = run_backtest(
        lambda: BradleyTerryHFA(max_iter=200),
        games_df,
        output_dir=tmp_path,
        model_name="bradley_terry_hfa",
    )

    metrics = outputs.metrics_overall.iloc[0]
    assert metrics["log_loss"] is not None
    assert metrics["brier_score"] is not None
    assert metrics["log_loss"] <= LOG_LOSS_THRESHOLD
    assert metrics["brier_score"] <= BRIER_SCORE_THRESHOLD
