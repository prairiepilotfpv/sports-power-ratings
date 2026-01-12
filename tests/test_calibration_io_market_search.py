from pathlib import Path

import time

from src.calibration.io import load_latest_calibrator


def test_load_latest_calibrator_prefers_market_dir(tmp_path, monkeypatch):
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

    # Create two joblib files; market one is newer
    f1 = fallback_dir / "old.joblib"
    f1.write_text("x")
    time.sleep(0.01)
    f2 = market_dir / "new.joblib"
    f2.write_text("y")

    # Monkeypatch BaseCalibrator.load to return a sentinel based on path
    class Dummy:
        def __init__(self, path):
            self._p = path

        @classmethod
        def load(cls, path):
            return f"loaded:{path}"

    monkeypatch.setattr("src.calibration.io.BaseCalibrator", Dummy)

    val = load_latest_calibrator(sport=sport, season=season, model=model, market=market)
    assert isinstance(val, str)
    assert "new.joblib" in val
