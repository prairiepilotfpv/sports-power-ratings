from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.runner import BacktestOutputs


def export_backtest_outputs_excel(outputs: BacktestOutputs, *, run_id: str) -> Path:
    """Write a backtest Excel workbook and return the path."""
    output_path = outputs.output_dir / f"backtest_{run_id}.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        outputs.predictions.to_excel(writer, sheet_name="predictions", index=False)
        outputs.metrics_by_date.to_excel(
            writer, sheet_name="metrics_by_date", index=False
        )
        outputs.metrics_overall.to_excel(
            writer, sheet_name="metrics_overall", index=False
        )
        outputs.calibration.to_excel(writer, sheet_name="calibration", index=False)

    return output_path
