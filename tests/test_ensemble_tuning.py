import numpy as np

from pipelines.ensemble_tuning import optimize_ensemble_weights


def test_optimize_weights_sum_to_one():
    probs = np.array(
        [
            [0.6, 0.4],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=float,
    )
    targets = np.array([1, 1, 0, 1], dtype=float)
    weights = optimize_ensemble_weights(probs, targets, models=["a", "b"])
    assert set(weights.keys()) == {"a", "b"}
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6
    assert all(weight >= 0 for weight in weights.values())


def test_optimize_single_model():
    probs = np.array([[0.55], [0.45], [0.7]], dtype=float)
    targets = np.array([1, 0, 1], dtype=float)
    weights = optimize_ensemble_weights(probs, targets, models=["solo"])
    assert weights == {"solo": 1.0}
