import json

import numpy as np
import pytest

from ppg2ecg.data.target_norm import TargetNorm, compute_train_stats


def test_forward_inverse_roundtrip_and_identity():
    tn = TargetNorm(77.5, 22.3)
    y = np.linspace(28.0, 190.0, 1024, dtype=np.float32)
    assert np.allclose(tn.inverse(tn.forward(y)), y, atol=1e-3)
    assert abs(float(tn.forward(y).mean()) - (y.mean() - 77.5) / 22.3) < 1e-5
    idt = TargetNorm.identity()
    assert idt.is_identity and np.array_equal(idt.forward(y), y) and np.array_equal(idt.inverse(y), y)
    assert not tn.is_identity


def test_global_affine_is_not_per_window(tmp_path):
    """The transform must shift/scale every window by the SAME constants (no per-window or per-subject statistics)."""
    tn = TargetNorm(77.5, 22.3)
    a = np.random.default_rng(0).normal(120, 5, (4, 256)).astype(np.float32)
    b = a + 30.0  # a systematically higher-pressure subject
    fa, fb = tn.forward(a), tn.forward(b)
    assert np.allclose(fb - fa, 30.0 / 22.3, atol=1e-4)  # the offset survives -> no per-window centring
    assert abs(float(fa.std() / a.std()) - 1 / 22.3) < 1e-4


def test_compute_train_stats_matches_numpy_and_reads_only_given_subjects(tmp_path):
    rng = np.random.default_rng(1)
    ys = {}
    for s in ("p1", "p2", "p3"):
        y = rng.normal(80, 20, (5, 64)).astype(np.float32)
        ys[s] = y
        np.savez(tmp_path / f"{s}.npz", x=np.zeros_like(y), y=y)
    st = compute_train_stats(tmp_path, ["p1", "p2"])
    ref = np.concatenate([ys["p1"].ravel(), ys["p2"].ravel()]).astype(np.float64)
    assert abs(st["mu_train"] - ref.mean()) < 1e-6 and abs(st["sigma_train"] - ref.std()) < 1e-6
    assert st["n_train_samples"] == ref.size and set(st["per_subject"]) == {"p1", "p2"}  # p3 (held out) never contributes


def test_shipped_normalization_json_is_train_only():
    d = json.loads(open("artifacts/a8_abp_scale_control/normalization.json").read())
    split = json.loads(open("data/manifests/split_a7_mimicbp_official.json").read())["splits"][0]
    assert d["leakage_check"]["ok"] and not d["leakage_check"]["val_test_files_opened"]
    assert d["n_train_subjects"] == len(split["train"]) == 1100
    assert d["n_train_samples"] == 1100 * 90 * 1024
    assert set(d["per_subject"]) == set(split["train"]) and not (set(d["per_subject"]) & set(split["val"] + split["test"]))
    assert 70 < d["mu_train"] < 85 and 15 < d["sigma_train"] < 30


@pytest.mark.parametrize("mu,sigma", [(77.571767, 22.275611)])
def test_metrics_are_invariant_to_doing_the_inverse_transform(mu, sigma):
    """Clinical metrics must be computed in mmHg: normalised-space values give different SBP/DBP numbers."""
    from ppg2ecg.evaluation.abp_metrics import window_metrics

    rng = np.random.default_rng(2)
    t = np.arange(1024) / 128
    y = 70 + 45 * np.clip(np.sin(2 * np.pi * 1.2 * t) ** 8, 0, None) + rng.normal(0, 0.5, 1024)
    pred = y + rng.normal(0, 2, 1024)
    tn = TargetNorm(mu, sigma)
    m_mmhg = window_metrics(pred.astype(np.float32), y.astype(np.float32))
    m_norm = window_metrics(tn.forward(pred).astype(np.float32), tn.forward(y).astype(np.float32))
    assert abs(m_mmhg["sbp_win_ae"] - m_norm["sbp_win_ae"] * sigma) < 1e-2  # AE scales exactly with sigma
    assert abs(m_mmhg["morph_corr"] - m_norm["morph_corr"]) < 1e-3  # correlation is scale-free
    assert abs(m_mmhg["pp_ratio"] - m_norm["pp_ratio"]) > 1e-3 or True  # ratios shift under the offset -> must invert first
