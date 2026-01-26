from pathlib import Path

import time

from src.calibration.base import BaseCalibrator
from src.calibration.io import load_latest_calibrator


class DummyCalibrator(BaseCalibrator):
    def fit(self, df):  # pragma: no cover - test helper
        return None

    def transform(self, probs):  # pragma: no cover - test helper
        return probs


def test_load_latest_calibrator_prefers_market_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = Path("outputs") / "calibrators"
    # Ensure directories exist; create a market-specific file and a fallback file.
    sport = "TEST"
    season = "2025"
    model = "mymodel"
    market = "ML"

    market_dir = base / sport / season / model / market
    fallback_dir = base / sport / season / model
    market_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir.mkdir(parents=True, exist_ok=True)

    # Create two calibrator files; market one is newer
    old = DummyCalibrator()
    old.metadata["tag"] = "fallback"
    old.save(fallback_dir / "old.joblib")
    time.sleep(0.01)
    new = DummyCalibrator()
    new.metadata["tag"] = "market"
    new.save(market_dir / "new.joblib")

    val = load_latest_calibrator(sport=sport, season=season, model=model, market=market)
    assert val is not None
    assert getattr(val, "metadata", {}).get("tag") == "market"
