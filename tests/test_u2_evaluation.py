"""U2 evaluation gates (docs/U2_PAIRED_IMEANFLOW_VS_PENGUIN_PREREGISTRATION.md §7-§9)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "scripts/u2_evaluate.py").read_text()
_ns: dict = {"__file__": str(ROOT / "scripts/u2_evaluate.py")}
exec(compile(SRC.split("def main()")[0], "u2_evaluate", "exec"), _ns)   # helpers only, no argparse
resp_wave_corr = _ns["resp_wave_corr"]
paired_cluster_bootstrap = _ns["paired_cluster_bootstrap"]
fd_subset = _ns["fd_subset"]
FS = 128


def _breath(n_windows=1, bpm=15.0, phase=0.0, T=60 * FS):
    t = np.arange(T) / FS
    return np.tile(np.sin(2 * np.pi * bpm / 60 * t + phase), (n_windows, 1))


@pytest.mark.parametrize("phase,expected", [(0.0, 1.0), (np.pi, -1.0), (np.pi / 2, 0.0)])
def test_resp_corr_is_phase_sensitive(phase, expected):
    """The whole point of the co-primary: upstream's FFT-argmax RespRateError scores a
    phase-inverted prediction perfectly, this must not."""
    got = float(resp_wave_corr(_breath(1, phase=phase), _breath(1))[0])
    assert got == pytest.approx(expected, abs=0.02)


def test_resp_corr_rejects_noise():
    noise = np.random.default_rng(0).standard_normal((1, 60 * FS))
    assert abs(float(resp_wave_corr(noise, _breath(1))[0])) < 0.2


def test_resp_corr_filters_both_signals_not_just_the_prediction():
    """Upstream low-passes the prediction only (help_func.py:191); U2's version is symmetric,
    so swapping the arguments can only flip nothing."""
    a, b = _breath(1, bpm=15.0), _breath(1, bpm=15.0, phase=0.3)
    assert float(resp_wave_corr(a, b)[0]) == pytest.approx(float(resp_wave_corr(b, a)[0]), abs=1e-9)


# ------------------------------------------------------------------ paired cluster bootstrap
def test_orientation_makes_positive_mean_arm_I_better():
    subj = np.array(["a"] * 10 + ["b"] * 10)
    c = np.full(20, 2.0)
    i = np.full(20, 1.0)                       # arm I has the smaller error
    pt, lo, hi, ns = paired_cluster_bootstrap(c, i, subj, "lower_better")
    assert pt == pytest.approx(1.0) and lo > 0 and ns == 2
    pt2, *_ = paired_cluster_bootstrap(c, i, subj, "higher_better")
    assert pt2 == pytest.approx(-1.0)          # same numbers, opposite meaning


def test_cluster_is_the_subject_not_the_window():
    """One subject with many windows must not outvote another with few."""
    subj = np.array(["a"] * 100 + ["b"] * 2)
    c = np.concatenate([np.full(100, 1.0), np.full(2, 1.0)])
    i = np.concatenate([np.full(100, 0.0), np.full(2, 10.0)])   # b is much worse
    pt, *_ = paired_cluster_bootstrap(c, i, subj, "lower_better")
    assert pt == pytest.approx((1.0 + -9.0) / 2)                # equal subject weight, not window weight


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(3)
    subj = np.repeat([f"s{i}" for i in range(6)], 20)
    c, i = rng.standard_normal(120), rng.standard_normal(120)
    assert paired_cluster_bootstrap(c, i, subj, "lower_better") == paired_cluster_bootstrap(c, i, subj, "lower_better")


# ------------------------------------------------------------------ FD subsampling
def test_fd_subset_is_exact_linspace_never_a_stride():
    """A stride silently halves small corpora -- the defect D1 had to fix."""
    assert fd_subset(500).tolist() == list(range(500))
    s = fd_subset(10_000)
    assert len(s) == 3000 and s[0] == 0 and s[-1] == 9999 and len(np.unique(s)) == len(s)
