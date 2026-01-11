from calibration.io import load_latest_calibrator


def test_load_latest_calibrator_returns_none_when_no_files():
    cal = load_latest_calibrator(sport="nonexistent", season="no_season", model="no_model", market="ML")
    assert cal is None
