from __future__ import annotations

import pandas as pd
from pathlib import Path

import pytest
pytest.importorskip("sklearn")

from backtest.runner import run_backtest
from models.registry import get_backtest_model


def _make_games() -> pd.DataFrame:
    # Two dates: first date will be used as initial test fold (no training),
    # second date will have prior predictions available for calibration training.
    dates = [pd.to_datetime("2025-10-07"), pd.to_datetime("2025-10-08")]
    rows = []
    for d in dates:
        rows.append({"date": d, "home_team": "A", "away_team": "B", "home_score": 3, "away_score": 2})
        rows.append({"date": d, "home_team": "C", "away_team": "D", "home_score": 1, "away_score": 4})
    return pd.DataFrame(rows)


def test_run_backtest_with_calibration(tmp_path: Path) -> None:
    games = _make_games()
    model_cls = get_backtest_model("elo")
    outputs = run_backtest(
        model_factory=lambda: model_cls(),
        games_df=games,
        start_date="2025-10-07",
        end_date="2025-10-08",
        output_dir=tmp_path / "bt",
        model_name="elo",
        calibrate=True,
        calib_dir=tmp_path / "calib",
    )

    # Accept any evidence that calibration ran: calibrated column present, or
    # calibration eval rows were persisted, or calibration summary returned.
    calibrated_present = (
        ("p_home_win_calibrated" in outputs.predictions.columns)
        and outputs.predictions["p_home_win_calibrated"].notna().any()
    )
    calib_summary_nonempty = not outputs.calibration.empty
    import os

    calib_files_exist = any(
        name.startswith("calibration_eval_")
        for name in os.listdir("outputs/calibrators")
    ) if os.path.exists("outputs/calibrators") else False

    assert calibrated_present or calib_summary_nonempty or calib_files_exist
