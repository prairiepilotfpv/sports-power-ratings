import json
import math
from pathlib import Path

import pandas as pd
import pytest

from ensemble.ml_v1 import MLWeightedAverageEnsemble, _logit_pool
from ensemble.io import load_ml_weights


def _expected_logit_pooled(probs: list[float], weights: list[float]) -> float:
    """Helper to compute expected logit-pooled probability for test assertions."""
    z_sum = 0.0
    for p, w in zip(probs, weights):
        z_sum += w * math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-z_sum))


def test_ml_weighted_equal_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Provide explicit equal weights (Issue #2 fix: models default to 0.0 without weights)
    ens = MLWeightedAverageEnsemble("TESTSPORT", "2025", weights={"a": 1.0, "b": 1.0})
    df = pd.DataFrame(
        [{"model_name": "a", "p_home_win": 0.6}, {"model_name": "b", "p_home_win": 0.4}]
    )
    p, comps = ens.combine(df)
    # With logit pooling: z1 = log(0.6/0.4), z2 = log(0.4/0.6) = -z1
    # Equal weights (0.5 each) -> z = 0.5*z1 + 0.5*(-z1) = 0 -> p = 0.5
    assert pytest.approx(p, rel=1e-6) == 0.5
    comps_list = json.loads(comps)
    assert isinstance(comps_list, list)
    assert len(comps_list) == 2


def test_ml_weighted_respects_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
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
    # With logit pooling and weights 2:1 (normalized to 2/3, 1/3)
    expected = _expected_logit_pooled([0.6, 0.4], [2/3, 1/3])
    assert pytest.approx(p, rel=1e-4) == expected
    comps_list = json.loads(comps)
    assert any(c.get("weight") == pytest.approx(2 / 3) for c in comps_list)


def test_ml_logit_pooling_known_values(tmp_path, monkeypatch):
    """Verify logit pooling produces expected output for known inputs."""
    monkeypatch.chdir(tmp_path)
    ens = MLWeightedAverageEnsemble("TESTSPORT", "2025", weights={"a": 1.0, "b": 1.0})
    df = pd.DataFrame([
        {"model_name": "a", "p_home_win": 0.7},
        {"model_name": "b", "p_home_win": 0.3},
    ])
    p, comps = ens.combine(df)

    # With symmetric inputs (0.7 and 0.3 are equidistant from 0.5 in log-odds):
    # z_0.7 = log(0.7/0.3) = 0.847
    # z_0.3 = log(0.3/0.7) = -0.847
    # Equal weights: z = 0.5 * 0.847 + 0.5 * (-0.847) = 0
    # p = 1/(1+exp(0)) = 0.5
    assert pytest.approx(p, rel=1e-4) == 0.5


def test_ml_logit_pooling_extreme_values(tmp_path, monkeypatch):
    """Verify clipping prevents log(0) or log(inf) for extreme probabilities."""
    monkeypatch.chdir(tmp_path)
    ens = MLWeightedAverageEnsemble("TESTSPORT", "2025", weights={"a": 1.0})

    # Test very high probability
    df_high = pd.DataFrame([{"model_name": "a", "p_home_win": 0.999999999}])
    p_high, _ = ens.combine(df_high)
    assert 0.0 < p_high < 1.0  # Should not be exactly 1 due to clipping
    assert p_high > 0.99  # But should still be very high

    # Test very low probability
    df_low = pd.DataFrame([{"model_name": "a", "p_home_win": 0.000000001}])
    p_low, _ = ens.combine(df_low)
    assert 0.0 < p_low < 1.0  # Should not be exactly 0 due to clipping
    assert p_low < 0.01  # But should still be very low


def test_ml_logit_pooling_single_model(tmp_path, monkeypatch):
    """Single model returns its probability unchanged (identity property)."""
    monkeypatch.chdir(tmp_path)
    ens = MLWeightedAverageEnsemble("TESTSPORT", "2025", weights={"a": 1.0})

    for test_prob in [0.3, 0.5, 0.7, 0.9]:
        df = pd.DataFrame([{"model_name": "a", "p_home_win": test_prob}])
        p, _ = ens.combine(df)
        # With single model at weight 1.0, logit pooling returns the same probability
        assert pytest.approx(p, rel=1e-6) == test_prob


def test_logit_pool_function_directly():
    """Direct unit test of the _logit_pool helper function."""
    # Test symmetric case: equal weights with symmetric probs -> 0.5
    result = _logit_pool([0.7, 0.3], [0.5, 0.5])
    assert pytest.approx(result, rel=1e-6) == 0.5

    # Test single probability -> identity
    result = _logit_pool([0.6], [1.0])
    assert pytest.approx(result, rel=1e-6) == 0.6

    # Test clipping: extreme value should not cause overflow
    result = _logit_pool([1e-9, 0.5], [0.5, 0.5])
    assert 0.0 < result < 1.0

    result = _logit_pool([1.0 - 1e-9, 0.5], [0.5, 0.5])
    assert 0.0 < result < 1.0
