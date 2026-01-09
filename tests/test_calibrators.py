from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import pytest
pytest.importorskip("sklearn")

from src.calibration.platt import PlattScalingCalibrator
from src.calibration.isotonic import IsotonicCalibrator


def test_platt_and_isotonic_fit_transform_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    p = np.linspace(0.05, 0.95, 200)
    y = rng.binomial(1, p)
    df = pd.DataFrame({"p_home_win": p, "home_win": y})

    # Platt
    platt = PlattScalingCalibrator()
    platt.fit(df)
    preds_platt = platt.transform(pd.Series([0.2, 0.5, 0.8]))
    assert len(preds_platt) == 3
    assert (preds_platt >= 0.0).all() and (preds_platt <= 1.0).all()

    # Isotonic
    iso = IsotonicCalibrator()
    iso.fit(df)
    sample = pd.Series(np.linspace(0.1, 0.9, 50))
    preds_iso = iso.transform(sample)
    assert len(preds_iso) == len(sample)
    # monotonic non-decreasing
    diffs = np.diff(preds_iso.values)
    assert (diffs >= -1e-8).all()

    # Save/load roundtrip
    out = tmp_path / "calib.joblib"
    platt.save(out)
    loaded = PlattScalingCalibrator.load(out)
    assert isinstance(loaded, PlattScalingCalibrator)
