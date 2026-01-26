import sqlite3
from pathlib import Path

import pandas as pd

from src.calibration.base import BaseCalibrator
from src.calibration.io import save_calibrator
from src.pipelines.validation_report import build_validation_report


class DummyCalibrator(BaseCalibrator):
    def fit(self, df: pd.DataFrame) -> None:  # pragma: no cover - test helper
        return None

    def transform(self, probs: pd.Series) -> pd.Series:  # pragma: no cover - test helper
        return probs


def _write_dummy_calibrator(base: Path, market: str) -> None:
    calib = DummyCalibrator()
    save_calibrator(
        calibrator=calib,
        output_dir=base,
        market=market,
        metadata={"n": 10, "method": "dummy", "market": market},
    )


def test_validation_report_includes_calibration_status(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    # Minimal DB file for validation report to read.
    db_path = tmp_path / "test.db"
    sqlite3.connect(db_path).close()

    # Create calibrator artifacts for all markets.
    base = Path("outputs") / "calibrators" / "nba" / "2025-26" / "historical"
    _write_dummy_calibrator(base / "ML", "ML")
    _write_dummy_calibrator(base / "spread", "spread")
    _write_dummy_calibrator(base / "total", "total")

    report, frames = build_validation_report(
        db_path=db_path,
        sport="nba",
        season="2025-26",
        run_backtests=False,
        run_calibration=False,
        calibration_source_id="historical",
    )

    assert "calibration_status" in frames
    status = frames["calibration_status"]
    assert set(status["market"]) == {"ML", "spread", "total"}
    assert status["exists"].all()
    summary = report.get("summary") or {}
    assert summary.get("calibration_missing") == 0
    assert summary.get("calibration_markets") == 3
