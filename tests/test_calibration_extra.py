from __future__ import annotations

import pytest
pytest.importorskip("sklearn")

from src.calibration.platt import PlattScalingCalibrator
from src.calibration.isotonic import IsotonicCalibrator

def test_placeholder():
    # This file exists to provide compatibility in CI environments that
    # already have a similarly-named test module; the substantive tests
    # live in `tests/test_calibrators.py`.
    assert True
