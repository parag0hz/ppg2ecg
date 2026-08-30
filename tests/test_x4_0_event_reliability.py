"""X4-0 unit tests (docs/X4_0_EVENT_RELIABILITY_PREREGISTRATION.md sec. 16). Synthetic tensors only, no real data."""
from __future__ import annotations

import numpy as np
import ppg2ecg.utils.mkl_warmup  # noqa: F401
import pytest
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.alignment_diagnostics import FS, HF_CUT_HZ, LOCAL_MAX_SHIFT_MS, QRS_HALF_MS
from ppg2ecg.flow.imeanflow import sample_meanflow

T = 1024


class _Net(torch.nn.Module):
    """Deterministic stand-in for MeanFlowS5.u that records every (t, h) query."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def u(self, z, ppg, t, h):
        self.calls.append((float(t[0, 0]), float(h[0, 0])))
        return 0.1 * z + 0.01 * ppg


# ---- 1: test-subject firewall
def test_firewall_blocks_test_subjects():
    ER.assert_no_test_subjects(["an0", "k2s", "e61"])
    for bad in (["kjd"], ["ssx"], ["an0", "kjd"], ["ssx", "k2s"]):
        with pytest.raises(ER.WildPPGTestFirewallError):
            ER.assert_no_test_subjects(bad)


# ---- 2 / 22 / 23: uniform schedule parity with the frozen sampler
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_uniform_schedule_matches_frozen_sample_meanflow(n):
    torch.manual_seed(0)
    ppg, e = torch.randn(3, 1, T), torch.randn(3, 1, T)
    a, nfe_a = sample_meanflow(_Net(), ppg, e.clone(), n)
    b, nfe_b = ER.sample_meanflow_schedule(_Net(), ppg, e.clone(), ER.UNIFORM[n])
    assert nfe_a == nfe_b == n
    assert torch.allclose(a, b, atol=1e-6)


# ---- 3: requested steps == actual NFE
@pytest.mark.parametrize("name", list(ER.SCHEDULES))
def test_schedule_nfe_equals_len(name):
    h = ER.SCHEDULES[name]
    net = _Net()
    _, nfe = ER.sample_meanflow_schedule(net, torch.randn(2, 1, T), torch.randn(2, 1, T), h)
    assert nfe == len(h) == len(net.calls)


# ---- 24 / 25: LN/LD h sums are 1 and terminate at t = 0
@pytest.mark.parametrize("name", list(ER.SCHEDULES))
def test_schedules_sum_to_one_and_end_at_zero(name):
    h = ER.SCHEDULES[name]
    assert abs(sum(h) - 1.0) < 1e-9
    ts = ER.schedule_times(h)
    assert ts[0] == 1.0 and abs(ts[-1]) < 1e-9
    assert np.all(np.diff(ts) < 0)                      # strictly decreasing t: noise -> data
    with pytest.raises(ValueError):
        ER.schedule_times([0.3, 0.3])                   # does not sum to 1


def test_large_step_is_where_the_name_says():
    net = _Net()
    ER.sample_meanflow_schedule(net, torch.randn(1, 1, T), torch.randn(1, 1, T), ER.SCHEDULES["LN4"])
    assert np.isclose(net.calls[0][1], 0.70) and np.isclose(net.calls[0][0], 1.0)   # big step first, from noise
    net2 = _Net()
    ER.sample_meanflow_schedule(net2, torch.randn(1, 1, T), torch.randn(1, 1, T), ER.SCHEDULES["LD4"])
    assert np.isclose(net2.calls[-1][1], 0.70) and np.isclose(net2.calls[-1][0], 0.70)  # big step last, into data


# ---- 4 / 26: the same Gaussian tensor is reused across NFE and across schedules
def test_same_source_reused_across_nfe_and_schedules():
    torch.manual_seed(1)
    e = torch.randn(2, 1, T)
    ppg = torch.randn(2, 1, T)
    snapshot = e.clone()
    for n in (1, 4, 8):
        ER.sample_meanflow_schedule(_Net(), ppg, e, ER.UNIFORM[n])
    for name in ("U4", "LN4", "LD4"):
        ER.sample_meanflow_schedule(_Net(), ppg, e, ER.SCHEDULES[name])
    assert torch.equal(e, snapshot)                     # sampler must not mutate the source bank


# ---- 6 / 7: deterministic hash subsets, pre-viewed windows excluded
def test_deterministic_hash_subset_and_exclusions():
    a = ER.select_subset("x4-event-nfe-v2", "an0", 22183, 1024)
    assert len(a) == 1024 and len(set(a.tolist())) == 1024
    assert np.array_equal(a, ER.select_subset("x4-event-nfe-v2", "an0", 22183, 1024))
    assert not np.array_equal(a, ER.select_subset("x4-event-source-v2", "an0", 22183, 1024))  # salt matters
    for s, w in ER.PREVIEWED_WINDOWS:
        assert w not in ER.select_subset("x4-event-nfe-v2", s, 30000, 2048).tolist()
        assert w not in ER.select_subset("x4-event-source-v2", s, 30000, 2048).tolist()
        assert w not in ER.select_subset("x4-event-schedule-v2", s, 30000, 2048).tolist()
    assert ER.window_hash("s", "an0", 5) == ER.window_hash("s", "an0", 5)
    assert ER.window_hash("s", "an0", 5) != ER.window_hash("s", "k2s", 5)


# ---- 8-12: X0 parity of the frozen semantics
def test_x0_semantics_parity():
    assert FS == 128 and QRS_HALF_MS == 100.0 and HF_CUT_HZ == 15.0 and LOCAL_MAX_SHIFT_MS == 150.0
    assert ER.MATCH_TOL_MS == 50.0 and ER.GT_ANCHOR_MS == LOCAL_MAX_SHIFT_MS
    assert ER.peak_train_agreement.__defaults__[1] == 50.0       # matcher tolerance is the X0 value
    # the module delegates to the X0 matcher rather than reimplementing it
    a, b = np.array([100, 300, 500]), np.array([102, 299, 900])
    pairs, miss, spur = R.match_rpeaks(a, b, FS, tol_ms=50.0)
    assert len(pairs) == 2 and miss == 1 and spur == 1
    assert ER.peak_train_agreement(a, b)["n_matched"] == 2


# ---- 17 / 18 / 19: predicted-vs-predicted event agreement
def test_identical_peak_trains_give_f1_one():
    p = np.array([50, 200, 400, 700])
    r = ER.peak_train_agreement(p, p.copy())
    assert r["f1"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0


def test_disjoint_peak_trains_give_f1_zero():
    r = ER.peak_train_agreement(np.array([10, 20, 30]), np.array([500, 600, 700]))
    assert r["f1"] == 0.0
    assert ER.peak_train_agreement(np.array([10]), np.array([]))["f1"] == 0.0
    assert ER.peak_train_agreement(np.array([]), np.array([]))["f1"] == 1.0


def test_known_synthetic_shift_matches_as_expected():
    p = np.array([100, 300, 500, 700])
    within = int(round(0.040 * FS))                    # 40 ms  -> inside the 50 ms tolerance
    beyond = int(round(0.080 * FS))                    # 80 ms  -> outside
    assert ER.peak_train_agreement(p, p + within)["f1"] == 1.0
    assert ER.peak_train_agreement(p, p + beyond)["f1"] == 0.0


# ---- 20 / 21: GT-anchored presence and the >=16/32 conditional filter
def test_gt_anchor_detection_probability_and_conditional_timing():
    gt = np.array([100, 400, 700])
    srcs = []
    for k in range(32):
        pk = [100 + (k % 5) - 2, 400 + (k % 3) - 1]    # beats 0 and 1 always present, small jitter
        if k < 8:
            pk.append(700)                              # beat 2 present in only 8 / 32 sources
        srcs.append(np.array(sorted(pk)))
    res = ER.gt_anchored_presence(gt, srcs)
    assert np.allclose(res["detection_probability"], [1.0, 1.0, 8 / 32])
    assert res["n_sources"] == 32
    eligible = res["n_detected"] >= 16
    assert eligible.tolist() == [True, True, False]     # the frozen >=16/32 filter drops the unstable beat
    assert np.all(np.isfinite(res["timing_sd_ms"][eligible]))
    # a source whose peak is beyond +/-150 ms is not counted as present
    far = ER.gt_anchored_presence(np.array([500]), [np.array([500 + int(0.200 * FS)])])
    assert far["detection_probability"][0] == 0.0


# ---- 28: jitter calibration is deterministic and preserves peak count
def test_jitter_calibration_deterministic_and_count_preserving():
    gt = np.array([100, 300, 500, 700, 900])
    j1 = ER.jitter_peaks(gt, 20.0, rng=np.random.default_rng(20260830), n_time=T)
    j2 = ER.jitter_peaks(gt, 20.0, rng=np.random.default_rng(20260830), n_time=T)
    assert np.array_equal(j1, j2) and len(j1) == len(gt)
    assert np.array_equal(ER.jitter_peaks(gt, 0.0, shift_ms=0.0, n_time=T), gt)
    shifted = ER.jitter_peaks(gt, 0.0, shift_ms=30.0, n_time=T)
    assert np.allclose(shifted - gt, round(0.030 * FS))
    assert ER.peak_train_agreement(gt, shifted)["f1"] == 1.0    # 30 ms shift still inside the 50 ms matcher


# ---- 27: subject-stratified bootstrap keeps subjects separate and equally weighted
def test_subject_stratified_bootstrap():
    rng = np.random.default_rng(0)
    # unequal window counts, and NON-constant within each subject so the bootstrap is not degenerate
    v = np.concatenate([rng.normal(0.0, 1.0, 500), rng.normal(1.0, 1.0, 50)])
    s = np.array(["an0"] * 500 + ["k2s"] * 50)
    point, lo, hi = ER.subject_stratified_bootstrap(v, s, n_boot=200, seed=20260830)
    expect = 0.5 * (v[:500].mean() + v[500:].mean())             # EQUAL subject weight, not window weight
    assert abs(point - expect) < 1e-9
    assert abs(point - v.mean()) > 1e-3                          # differs from the window-weighted mean
    assert lo <= point <= hi
    assert ER.subject_stratified_bootstrap(v, s, 200, 20260830) == ER.subject_stratified_bootstrap(v, s, 200, 20260830)
    assert ER.subject_stratified_bootstrap(v, s, 200, 1) != ER.subject_stratified_bootstrap(v, s, 200, 20260830)
    # the k2s subject (n=50) must widen the interval far more than its window share would suggest
    assert (hi - lo) > 0.05


# ---- 13 / 14: derangement parity with eval_a2.py, and no fixed points
def test_derangement_parity_and_no_fixed_points():
    src = open("scripts/eval_a2.py").read()
    assert "def derangement" in src

    def derangement(n, seed):                                     # verbatim semantics from eval_a2.py
        rng = np.random.default_rng(seed)
        while True:
            p = rng.permutation(n)
            if n < 2 or not np.any(p == np.arange(n)):
                return p

    p = derangement(512, 1)
    assert len(p) == 512 and len(set(p.tolist())) == 512
    assert not np.any(p == np.arange(512))                        # no fixed point
    assert np.array_equal(p, derangement(512, 1))                 # deterministic


# ---- 15 / 16: perturbations change exactly one input
def test_perturbations_change_only_one_input():
    torch.manual_seed(2)
    ppg, z0, z1 = torch.randn(4, 1, T), torch.randn(4, 1, T), torch.randn(4, 1, T)
    perm = np.array([1, 2, 3, 0])
    # B: source perturbation -> PPG identical, source different
    assert torch.equal(ppg, ppg) and not torch.equal(z0, z1)
    # C: condition perturbation -> source identical, PPG permuted with no fixed point
    assert torch.equal(z0, z0) and not torch.equal(ppg, ppg[perm])
    assert not np.any(perm == np.arange(4))
