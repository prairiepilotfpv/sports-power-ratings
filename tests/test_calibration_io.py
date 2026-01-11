from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.calibration.base import BaseCalibrator
from src.calibration.io import load_latest_calibrator


class DummyCalibrator(BaseCalibrator):
    def fit(self, df: pd.DataFrame) -> None:
        return None

    def transform(self, probs: pd.Series) -> pd.Series:
        return probs


def _save_calibrator(path: Path, *, tag: str, mtime: int) -> None:
    calibrator = DummyCalibrator()
    calibrator.metadata["tag"] = tag
    calibrator.save(path)
    os.utime(path, (mtime, mtime))


def test_load_latest_calibrator_prefers_highest_priority(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "outputs" / "calibrators"
    season_dir = base / "nba" / "2024-25" / "elo"
    sport_dir = base / "nba" / "elo"

    season_dir.mkdir(parents=True, exist_ok=True)
    sport_dir.mkdir(parents=True, exist_ok=True)

    _save_calibrator(season_dir / "older.joblib", tag="season", mtime=100)
    _save_calibrator(sport_dir / "newer.joblib", tag="sport", mtime=200)

    loaded = load_latest_calibrator(
        sport="nba",
        season="2024-25",
        model="elo",
        market="ML",
    )

    assert loaded is not None
    assert loaded.metadata.get("tag") == "season"
