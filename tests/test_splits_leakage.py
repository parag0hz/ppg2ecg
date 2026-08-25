import numpy as np

from ppg2ecg.data.dalia import SUBJECTS
from ppg2ecg.data.leakage import check_subject_disjoint, check_window_disjoint, check_windowwise_normalization
from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows
from ppg2ecg.data.splits import make_holdout_split, make_kfold_splits


def test_holdout_split_is_disjoint_and_deterministic():
    a, b = make_holdout_split(seed=42), make_holdout_split(seed=42)
    assert a == b
    rep = check_subject_disjoint(a, SUBJECTS)
    assert rep["ok"], rep
    assert rep["sizes"] == {"train": 13, "val": 1, "test": 1}
    assert make_holdout_split(seed=43) != a


def test_kfold_covers_every_subject_once():
    folds = make_kfold_splits(seed=42)
    tests = [s for f in folds for s in f["test"]]
    assert sorted(tests, key=lambda s: int(s[1:])) == list(SUBJECTS)
    for f in folds:
        rep = check_subject_disjoint(f, SUBJECTS)
        assert rep["ok"], rep


def test_subject_overlap_is_detected():
    bad = {"train": ["S1", "S2"], "val": ["S2"], "test": ["S3"]}
    rep = check_subject_disjoint(bad)
    assert not rep["ok"] and rep["overlaps"]["train∩val"] == ["S2"]


def test_window_overlap_is_detected():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((10, 32)), rng.standard_normal((10, 32))
    assert check_window_disjoint({"train": a, "test": b})["ok"]
    b[3] = a[7]
    rep = check_window_disjoint({"train": a, "test": b})
    assert not rep["ok"] and rep["overlaps"]["train∩test"] == 1


def test_preprocessing_is_windowwise():
    x = np.random.default_rng(1).standard_normal((16, 256)).cumsum(axis=1)
    for kw in (PPG_KW, ECG_KW):
        rep = check_windowwise_normalization(lambda a: preprocess_windows(a, 128, 4, **kw), x)
        assert rep["ok"], rep


def test_preprocess_output_range_and_shape():
    x = np.random.default_rng(2).standard_normal((5, 2800)).cumsum(axis=1)
    y = preprocess_windows(x, 128, 4, **ECG_KW)
    assert y.shape == (5, 512)
    assert np.allclose(y.min(axis=1), -1) and np.allclose(y.max(axis=1), 1, atol=1e-6)
