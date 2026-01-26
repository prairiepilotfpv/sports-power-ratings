import json
from pathlib import Path

import pandas as pd
import pytest

from ensemble.total_v1 import TotalWeightedAverageEnsemble


def test_total_equal_weights_when_no_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ens = TotalWeightedAverageEnsemble("TESTSPORT", "2025", weights={"a": 1.0, "b": 1.0})
    df = pd.DataFrame(
        [
            {"model_name": "a", "total_mean": 200.0, "total_sd": 10.0},
            {"model_name": "b", "total_mean": 210.0, "total_sd": 14.0},
        ]
    )
    mean, sd, comps = ens.combine(df)
    assert mean == pytest.approx(205.0, rel=1e-6)
    assert sd == pytest.approx((0.5 * 100.0 + 0.5 * 196.0 + 25.0) ** 0.5, rel=1e-6)
    comps_list = json.loads(comps)
    weight_sum = sum(c["weight"] for c in comps_list if c["value"] is not None)
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
    weights = {c["model"]: c["weight"] for c in comps_list}
    assert weights["a"] == pytest.approx(1.0, rel=1e-6)
    assert weights["b"] == pytest.approx(0.0, rel=1e-6)
    assert sd == pytest.approx(10.0, rel=1e-6)


def test_total_sd_weighted_rms_uses_only_sd_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ens = TotalWeightedAverageEnsemble("TEST", "2025", weights={"a": 1.0, "b": 1.0, "c": 1.0}, include_between_model_variance=False)
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


def test_total_between_variance_agreement_is_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ens = TotalWeightedAverageEnsemble("TEST", "2025", weights={"a": 1.0, "b": 1.0}, include_between_model_variance=True)
    df = pd.DataFrame(
        [
            {"model_name": "a", "total_mean": 205.0, "total_sd": 10.0},
            {"model_name": "b", "total_mean": 205.0, "total_sd": 12.0},
        ]
    )
    _, sd, _ = ens.combine(df)
    expected_within = (0.5 * (10.0 ** 2) + 0.5 * (12.0 ** 2)) ** 0.5
    assert sd == pytest.approx(expected_within, rel=1e-6)


def test_total_between_variance_disagreement_increases_sd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    baseline = TotalWeightedAverageEnsemble("TEST", "2025", weights={"a": 1.0, "b": 1.0, "c": 1.0}, include_between_model_variance=False)
    adjusted = TotalWeightedAverageEnsemble("TEST", "2025", weights={"a": 1.0, "b": 1.0}, include_between_model_variance=True)
    df = pd.DataFrame(
        [
            {"model_name": "a", "total_mean": 198.0, "total_sd": 10.0},
            {"model_name": "b", "total_mean": 222.0, "total_sd": 14.0},
        ]
    )
    _, sd_baseline, _ = baseline.combine(df)
    _, sd_adjusted, _ = adjusted.combine(df)
    assert sd_baseline == pytest.approx((0.5 * 100.0 + 0.5 * 196.0) ** 0.5, rel=1e-6)
    assert sd_adjusted == pytest.approx((0.5 * 100.0 + 0.5 * 196.0 + 144.0) ** 0.5, rel=1e-6)
    assert sd_adjusted > sd_baseline


def test_total_between_variance_handles_missing_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ens = TotalWeightedAverageEnsemble("TEST", "2025", weights={"a": 1.0, "b": 1.0}, include_between_model_variance=True)
    df = pd.DataFrame(
        [
            {"model_name": "a", "total_mean": 200.0, "total_sd": 10.0},
            {"model_name": "b", "total_mean": float("nan"), "total_sd": 30.0},
            {"model_name": "c", "total_mean": 240.0, "total_sd": float("nan")},
        ]
    )
    mean, sd, _ = ens.combine(df)
    assert mean == pytest.approx(220.0, rel=1e-6)
    assert sd == pytest.approx(10.0, rel=1e-6)


def test_total_between_variance_skipped_when_single_var_component(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ens = TotalWeightedAverageEnsemble("TEST", "2025", weights={"a": 1.0, "b": 1.0}, include_between_model_variance=True)
    df = pd.DataFrame(
        [
            {"model_name": "a", "total_mean": 210.0, "total_sd": 10.0},
            {"model_name": "b", "total_mean": 230.0, "total_sd": float("nan")},
        ]
    )
    mean, sd, _ = ens.combine(df)
    assert mean == pytest.approx(220.0, rel=1e-6)
    assert sd == pytest.approx(10.0, rel=1e-6)
