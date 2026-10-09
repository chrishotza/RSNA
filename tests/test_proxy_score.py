import numpy as np
import pandas as pd
import pytest

from src.proxy_score import LABELS, macro_auc, rank_percentile, score_oof_file


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


def test_nonfinite_prediction_on_labeled_row_fails_closed():
    y = np.tile([0.0, 0.0, 1.0, 1.0], (12, 1)).T
    p = np.tile([0.1, 0.2, 0.8, 0.9], (12, 1)).T
    p[0, 0] = np.nan
    with pytest.raises(ValueError, match="prediction coverage is incomplete"):
        macro_auc(y, p)


def test_nan_target_can_represent_unaddressed_label():
    y = np.tile([0.0, 0.0, 1.0, 1.0], (12, 1)).T
    p = np.tile([0.1, 0.2, 0.8, 0.9], (12, 1)).T
    y[0, 0] = np.nan
    macro, aucs = macro_auc(y, p)
    assert macro == 1.0
    assert np.allclose(aucs, 1.0)


def test_infinite_target_fails_closed():
    y = np.tile([0.0, 0.0, 1.0, 1.0], (12, 1)).T
    p = np.tile([0.1, 0.2, 0.8, 0.9], (12, 1)).T
    y[0, 0] = np.inf
    with pytest.raises(ValueError, match="targets contain infinity"):
        macro_auc(y, p)


def test_oof_file_requires_study_and_fold_provenance(tmp_path):
    data = {}
    y = np.tile([0.0, 1.0], (12, 1)).T
    p = np.tile([0.1, 0.9], (12, 1)).T
    for j, label in enumerate(LABELS):
        data["y_" + label] = y[:, j]
        data["p_" + label] = p[:, j]
    path = tmp_path / "missing_provenance.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing OOF provenance columns"):
        score_oof_file(path)


def test_valid_oof_file_scores_and_checks_unique_studies(tmp_path):
    data = {
        "StudyInstanceUID": ["study-a", "study-b"],
        "fold": [0, 1],
    }
    y = np.tile([0.0, 1.0], (12, 1)).T
    p = np.tile([0.1, 0.9], (12, 1)).T
    for j, label in enumerate(LABELS):
        data["y_" + label] = y[:, j]
        data["p_" + label] = p[:, j]
    path = tmp_path / "valid_oof.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    result = score_oof_file(path)
    assert result["macro_auc"] == 1.0
    assert result["n_studies"] == 2
