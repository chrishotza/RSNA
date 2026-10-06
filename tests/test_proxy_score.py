import numpy as np

from src.proxy_score import macro_auc, rank_percentile


def test_perfect_predictions():
    y = np.tile([0.0, 1.0], (12, 1)).T
    p = np.tile([0.01, 0.99], (12, 1)).T
    macro, aucs = macro_auc(y, p)
    assert macro == 1.0
    assert np.allclose(aucs, 1.0)


def test_inverse_predictions():
    y = np.tile([0.0, 1.0], (12, 1)).T
    p = np.tile([0.99, 0.01], (12, 1)).T
    macro, aucs = macro_auc(y, p)
    assert macro == 0.0
    assert np.allclose(aucs, 0.0)


def test_rank_percentile_preserves_order():
    x = np.array([10.0, 20.0, 20.0, 5.0])
    r = rank_percentile(x)
    assert r[3] < r[0] < r[1]
    assert r[1] == r[2]
