import json
from pathlib import Path

import pandas as pd
import pytest

from ensemble.total_v1 import TotalWeightedAverageEnsemble


def test_total_equal_weights_when_no_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ens = TotalWeightedAverageEnsemble("TESTSPORT", "2025")
    df = pd.DataFrame(
        [
            {"model_name": "a", "total_mean": 200.0, "total_sd": 10.0},
            {"model_name": "b", "total_mean": 210.0, "total_sd": 14.0},
        ]
    )
    mean, sd, comps = ens.combine(df)
    assert mean == pytest.approx(205.0, rel=1e-6)
    assert sd == pytest.approx((0.5 * 100.0 + 0.5 * 196.0) ** 0.5, rel=1e-6)
    comps_list = json.loads(comps)
    weight_sum = sum(c["w"] for c in comps_list if c["total_mean"] is not None)
    assert weight_sum == pytest.approx(1.0, rel=1e-6)


def test_total_renormalizes_missing_total_mean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = Path("outputs") / "ensembles" / "TEST" / "2025" / "TOTAL"
    base.mkdir(parents=True, exist_ok=True)
    (base / "ensemble_total_v1.json").write_text(json.dumps({"a": 1, "b": 1}))

    ens = TotalWeightedAverageEnsemble("TEST", "2025")
    df = pd.DataFrame(
        [
            {"model_name": "a", "total_mean": 200.0, "total_sd": 10.0},
            {"model_name": "b", "total_mean": float("nan"), "total_sd": 12.0},
        ]
    )
    mean, sd, comps = ens.combine(df)
    assert mean == pytest.approx(200.0, rel=1e-6)
    comps_list = json.loads(comps)
    weights = {c["model"]: c["w"] for c in comps_list}
    assert weights["a"] == pytest.approx(1.0, rel=1e-6)
    assert weights["b"] == pytest.approx(0.0, rel=1e-6)
    assert sd == pytest.approx(10.0, rel=1e-6)


def test_total_sd_weighted_rms_uses_only_sd_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ens = TotalWeightedAverageEnsemble("TEST", "2025")
    df = pd.DataFrame(
        [
            {"model_name": "a", "total_mean": 200.0, "total_sd": 10.0},
            {"model_name": "b", "total_mean": 210.0, "total_sd": 20.0},
            {"model_name": "c", "total_mean": 220.0, "total_sd": float("nan")},
        ]
    )
    mean, sd, _ = ens.combine(df)
    assert mean == pytest.approx(210.0, rel=1e-6)
    assert sd == pytest.approx((0.5 * 100.0 + 0.5 * 400.0) ** 0.5, rel=1e-6)
