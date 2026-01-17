import numpy as np
from tools import ensemble_optimize as eo


def test_projection_and_optimize_simple():
    # Create synthetic probs for 3 models over 100 games
    np.random.seed(1)
    n = 200
    m = 3
    # true weights
    true_w = np.array([0.6, 0.3, 0.1])
    # base signals
    s = np.random.randn(n, m) * 0.5
    probs = 1 / (1 + np.exp(-s))
    # ensemble prob with true weights
    ens = probs.dot(true_w)
    # generate labels
    y = (ens > 0.5).astype(int)
    # run optimizer
    w, loss = eo.optimize_weights(probs, y, lr=0.5, max_iter=1000, restarts=2)
    assert w.shape[0] == m
    assert np.all(w >= -1e-8)
    # sums to (approximately) 1
    assert abs(w.sum() - 1) < 1e-6
    # recovered weights should be reasonably close to true weights
    assert np.linalg.norm(w - true_w) < 0.5
