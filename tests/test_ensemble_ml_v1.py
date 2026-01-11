import json
from pathlib import Path

import pandas as pd
import pytest

from ensemble.ml_v1 import MLWeightedAverageEnsemble
from ensemble.io import load_ml_weights


def test_ml_weighted_equal_weights(tmp_path, monkeypatch):
    # Ensure no weights file exists for the test sport/season
    out_dir = Path("outputs") / "ensembles" / "TESTSPORT" / "2025"
    if out_dir.exists():
        for f in out_dir.glob("*"):
            f.unlink()
    ens = MLWeightedAverageEnsemble("TESTSPORT", "2025")
    df = pd.DataFrame(
        [{"model_name": "a", "p_home_win": 0.6}, {"model_name": "b", "p_home_win": 0.4}]
    )
    p, comps = ens.combine(df)
    assert pytest.approx(p, rel=1e-6) == 0.5
    comps_list = json.loads(comps)
    assert isinstance(comps_list, list)
    assert len(comps_list) == 2


def test_ml_weighted_respects_weights(tmp_path):
    # Write a small weights file and ensure renormalization works
    base = Path("outputs") / "ensembles" / "NBA" / "2025-26"
    base.mkdir(parents=True, exist_ok=True)
    weights_path = base / "ensemble_ml_v1.json"
    weights_path.write_text(json.dumps({"m1": 2, "m2": 1}))
    ens = MLWeightedAverageEnsemble("NBA", "2025-26")
    df = pd.DataFrame(
        [{"model_name": "m1", "p_home_win": 0.6}, {"model_name": "m2", "p_home_win": 0.4}]
    )
    p, comps = ens.combine(df)
    assert pytest.approx(p, rel=1e-6) == pytest.approx((0.6 * 2 + 0.4 * 1) / 3)
    comps_list = json.loads(comps)
    assert any(c.get("weight") == pytest.approx(2 / 3) for c in comps_list)
