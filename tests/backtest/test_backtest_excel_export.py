from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.runner import run_backtest
from models.registry import get_backtest_model

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mini_nba.csv"


def test_backtest_exports_excel(tmp_path: Path) -> None:
    games_df = pd.read_csv(FIXTURE_PATH)
    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date

    dates = sorted(games_df["date"].unique())
    assert len(dates) > 1, "Fixture must include at least two dates."

    start_date = dates[1].isoformat()
    model_cls = get_backtest_model("bradley_terry_hfa")

    output_dir = tmp_path / "backtest"
    run_backtest(
        model_factory=model_cls,
        games_df=games_df,
        start_date=start_date,
        end_date=start_date,
        output_dir=output_dir,
        model_name="bradley_terry_hfa",
    )

    excel_files = list(output_dir.glob("backtest_*.xlsx"))
    assert len(excel_files) == 1

    excel = pd.ExcelFile(excel_files[0])
    assert set(excel.sheet_names) == {
        "predictions",
        "metrics_by_date",
        "metrics_overall",
        "calibration",
    }
