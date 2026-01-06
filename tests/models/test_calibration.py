from __future__ import annotations

from models.calibration import ConditionalSDModel


def test_conditional_sd_increases_with_margin() -> None:
    model = ConditionalSDModel(intercept=1.0, slope=0.2)
    low_sd = model.predict(1.0)
    high_sd = model.predict(5.0)

    assert high_sd > low_sd
