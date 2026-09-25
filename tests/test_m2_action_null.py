"""M2 synthetic tests: the per-partition action-value pipeline must not manufacture predictability when the width and
depth outcomes carry no window-specific information (null), and must detect it when they do (positive control)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
M2 = pytest.importorskip("m2_run")
M1 = M2.M1


def _synthetic(n, npat, seed, hetero):
    r = np.random.default_rng(seed)
    pid = np.sort(r.integers(0, npat, n)); pid[:npat] = np.arange(npat)
    ref = r.normal(75, 12, n)
    sig = np.exp(r.normal(0, 0.7, n)) * 3 if hetero else np.full(n, 3.0)
    banks = {}
    for a in M2.ARMS:
        for S in (1, 2, 4, 8):
            Y = ref + r.normal(0, 1.0, n) + r.standard_t(5, (16, n)) * sig
            banks[(a, S)] = M1.snap(Y)
    return banks, M1.snap(ref), pid


def _within_spearman(d, pred):
    vals = []
    for p in range(M2.NPART):
        m = d["part"] == p
        w = np.ones(m.sum())
        c = d["cond"][m]
        vals.append(np.nanmean(M1.wcorr_seg(M1.RankPrep(pred[m], c).ranks(w), M1.RankPrep(d["dQ"][m], c).ranks(w), w, c, len(M2.COND))))
    return float(np.mean(vals))


def _run(hetero):
    parts = M2.partitions()
    val, test = _synthetic(900, 90, 1, hetero), _synthetic(1500, 150, 2, hetero)
    dv = M2.build_rows(*val, 75.0, parts, n_workers=1)
    dt = M2.build_rows(*test, 75.0, parts, n_workers=1)
    up, pid_idx = np.unique(dv["pid"], return_inverse=True)
    fit = M2.fit_ridge_policy(M2.transform_full(dv["F"])[:, M2.P1_COLS], dv, pid_idx, len(up), "P1", [])
    pred = M2.ridge_predict(fit, M2.transform_full(dt["F"])[:, M2.P1_COLS])
    return dt, pred


def test_null_no_manufactured_action_predictability():
    dt, pred = _run(hetero=False)
    assert abs(_within_spearman(dt, pred)) < 0.03


def test_positive_control_detects_dispersion_dependent_advantage():
    dt, pred = _run(hetero=True)
    assert _within_spearman(dt, pred) > 0.05          # design signal ~0.085 (per-window dQ is very noisy)
