import json
from pathlib import Path

import pandas as pd
import pytest

from ensemble.spread_v1 import SpreadWeightedAverageEnsemble


def test_spread_equal_weights_when_no_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ens = SpreadWeightedAverageEnsemble("TESTSPORT", "2025")
    df = pd.DataFrame(
        [
            {"model_name": "a", "margin_mean": 3.0, "margin_sd": 2.0},
            {"model_name": "b", "margin_mean": 1.0, "margin_sd": 4.0},
        ]
    )
    mean, sd, comps = ens.combine(df)
    assert mean == pytest.approx(2.0, rel=1e-6)
    comps_list = json.loads(comps)
    weight_sum = sum(c["w"] for c in comps_list if c["margin_mean"] is not None)
    assert weight_sum == pytest.approx(1.0, rel=1e-6)


def test_spread_renormalizes_missing_margin_mean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = Path("outputs") / "ensembles" / "TEST" / "2025" / "SPREAD"
    base.mkdir(parents=True, exist_ok=True)
    (base / "ensemble_spread_v1.json").write_text(json.dumps({"a": 1, "b": 1}))

    ens = SpreadWeightedAverageEnsemble("TEST", "2025")
    df = pd.DataFrame(
        [
            {"model_name": "a", "margin_mean": 4.0, "margin_sd": 1.0},
            {"model_name": "b", "margin_mean": float("nan"), "margin_sd": 2.0},
        ]
    )
    mean, sd, comps = ens.combine(df)
    assert mean == pytest.approx(4.0, rel=1e-6)
    comps_list = json.loads(comps)
    weights = {c["model"]: c["w"] for c in comps_list}
    assert weights["a"] == pytest.approx(1.0, rel=1e-6)
    assert weights["b"] == pytest.approx(0.0, rel=1e-6)


def test_spread_margin_sd_weighted_rms_uses_only_sd_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ens = SpreadWeightedAverageEnsemble("TEST", "2025")
    df = pd.DataFrame(
        [
            {"model_name": "a", "margin_mean": 3.0, "margin_sd": 2.0},
            {"model_name": "b", "margin_mean": 1.0, "margin_sd": 4.0},
            {"model_name": "c", "margin_mean": 2.0, "margin_sd": float("nan")},
        ]
    )
    mean, sd, _ = ens.combine(df)
    assert mean == pytest.approx(2.0, rel=1e-6)
    assert sd == pytest.approx((0.5 * 4.0 + 0.5 * 16.0) ** 0.5, rel=1e-6)


def test_spread_components_weights_sum_to_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ens = SpreadWeightedAverageEnsemble("TEST", "2025")
    df = pd.DataFrame(
        [
            {"model_name": "a", "margin_mean": 3.0, "margin_sd": 2.0},
            {"model_name": "b", "margin_mean": 1.0, "margin_sd": 4.0},
            {"model_name": "c", "margin_mean": float("nan"), "margin_sd": 5.0},
        ]
    )
    _, _, comps = ens.combine(df)
    comps_list = json.loads(comps)
    weight_sum = sum(c["w"] for c in comps_list if c["margin_mean"] is not None)
    assert weight_sum == pytest.approx(1.0, rel=1e-6)
