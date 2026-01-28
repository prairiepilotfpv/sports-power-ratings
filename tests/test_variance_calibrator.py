import pandas as pd
import pytest
from itertools import cycle, islice

from calibration.distribution import VarianceCalibrator


def _build_dataset(sd_values: list[float] | None = None, n: int = 5) -> pd.DataFrame:
    if sd_values is None:
        sd_values = [float(1 + i) for i in range(n)]
    sd_values = list(islice(cycle(sd_values), n))
    return pd.DataFrame(
        {
            "pred_mean": [float(i) for i in range(n)],
            "pred_sd": sd_values,
            "actual_value": [float(i) + 0.1 for i in range(n)],
        }
    )


def test_variance_calibrator_basic_fit():
    calibrator = VarianceCalibrator()
    df = _build_dataset()
    calibrator.fit(df)
    result = calibrator.transform(df)

    assert {"calibrated_mean", "calibrated_sd"}.issubset(result.columns)
    assert calibrator.c_min <= calibrator.c <= calibrator.c_max
    assert calibrator.tau >= 0
    assert calibrator.metadata["method"] == "variance_distribution"

    # Means should pass through unchanged and calibrator should produce finite SDs
    pd.testing.assert_series_equal(
        result["calibrated_mean"], df["pred_mean"].astype(float), check_names=False
    )
    assert result["calibrated_sd"].notna().all()


def test_variance_calibrator_marks_unhealthy_when_guardrail_clips():
    calibrator = VarianceCalibrator(
        guardrail_min=10.0,
        clip_rate_threshold=0.01,
    )
    df = _build_dataset(sd_values=[1.0])
    calibrator.fit(df)

    assert not calibrator.healthy
    assert calibrator.clip_rate > calibrator.clip_rate_threshold
    assert calibrator.metadata["healthy"] is False


def test_variance_calibrator_falls_back_to_identity_on_optimizer_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("calibration.distribution.minimize", _raise)
    calibrator = VarianceCalibrator()
    df = _build_dataset()
    calibrator.fit(df)

    assert calibrator.metadata["method"] == "variance_identity"
    assert not calibrator.healthy

    transformed = calibrator.transform(df)
    pd.testing.assert_series_equal(
        transformed["calibrated_sd"], df["pred_sd"].astype(float), check_names=False
    )
