"""D0 unit tests on synthetic data only (docs/D0_EVENT_CELL_SUPPORT_PREREGISTRATION.md §7)."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.beatfirst import beats as BB
from ppg2ecg.beatfirst import eventcell as EC
from ppg2ecg.beatfirst import render as BR

ROOT = Path(__file__).resolve().parents[1]
T = 512


@pytest.fixture(scope="module")
def d0():
    sys.path.insert(0, str(ROOT / "scripts"))
    import d0_event_cell
    return d0_event_cell


def gauss_beat(ghost_at=None, amp=1.0, ghost_amp=1.0, offset=0.0):
    k = np.arange(BB.SEG_LEN)
    b = amp * np.exp(-0.5 * ((k - BB.R_INDEX) / 1.5) ** 2) + offset
    if ghost_at is not None:
        b = b + ghost_amp * np.exp(-0.5 * ((k - ghost_at) / 1.5) ** 2)
    return b


# ------------------------------------------------------------------------------------------ cells (tests 2-5)
def test_cells_are_ordered_and_interior_limits_are_midpoints():
    P = np.array([60, 170, 270, 395])
    L, U = EC.nominal_cells(P, rr_single=100.0)
    assert np.all(L < U) and np.all(np.diff(L) > 0)
    np.testing.assert_allclose(U[:-1], (P[:-1] + P[1:]) / 2)
    np.testing.assert_allclose(L[1:], U[:-1])


def test_first_last_extrapolation_and_singleton_rule():
    L, U = EC.nominal_cells([100, 180, 300], rr_single=100.0)
    assert L[0] == 100 - 0.5 * 80 and U[-1] == 300 + 0.5 * 120
    Ls, Us = EC.nominal_cells([200], rr_single=96.0)
    assert (Ls[0], Us[0]) == (152.0, 248.0)
    assert EC.nominal_cells([], 100.0)[0].size == 0


def test_singleton_rr_comes_from_train_and_is_frozen_in_the_renderer_config(d0):
    src_x = inspect.getsource(d0.stage_crossfade)
    assert "rr_single = rr_median * FS" in src_x and "frozen_bf0()" in src_x       # BF0 template.npz: TRAIN median RR
    src_e = inspect.getsource(d0.stage_evaluate)
    assert 'cfg["rr_single_samples"]' in src_e and 'cfg["crossfade_width_samples"]' in src_e


# ------------------------------------------------------------------------------------------ weights (tests 6-11)
@pytest.mark.parametrize("P", [[60, 170, 270, 395], [40, 200, 420], [100], [5, 150, 506], [100, 110, 300]])
@pytest.mark.parametrize("w", [4, 8, 16])
def test_partition_of_unity_and_zero_outside_support(P, w):
    W, rec = EC.cell_weights(P, T, w, rr_single=100.0)
    s = W.sum(axis=0)
    cov = s > EC.TOL
    assert np.max(np.abs(s[cov] - 1.0)) < 1e-12                                   # tests 9-10
    t = np.arange(T)
    for i, p in enumerate(P):                                                       # test 8
        assert np.all(W[i][(t < p - BB.N_BEFORE) | (t > p + BB.N_AFTER)] == 0)
        assert W[i][p] == 1.0                                                       # own R sample fully owned
    for i in range(len(P) - 1):                                                     # nothing of beat i at the next R
        assert W[i][P[i + 1]] == 0 and W[i + 1][P[i]] == 0
        if P[i + 1] - P[i] >= 14 + w:                                               # ... nor in its +-6-sample QRS
            assert np.all(W[i][P[i + 1] - 6:P[i + 1] + 7] == 0)


def test_crossfade_moves_only_as_far_as_the_beat_supports_require():
    P = [100, 240]                                       # RR 140 > 128: the midpoint 170 is outside beat 2's head (176)
    W, rec = EC.cell_weights(P, T, 16, 100.0)
    c = rec["centres"][0]
    assert c == 240 - BB.N_BEFORE + 8 and rec["shifted"] == 1 and rec["widths"][0] == 16
    assert np.all(W[0][int(c + 8) + 1:] == 0) and np.all(W[1][:int(c - 8)] == 0)
    W2, rec2 = EC.cell_weights([100, 200], T, 16, 100.0)
    assert rec2["centres"][0] == 150.0 and rec2["shifted"] == 0
    W3, rec3 = EC.cell_weights([100, 250], T, 16, 100.0)   # supports overlap by 15 < 16: crossfade over the overlap
    assert rec3["centres"][0] == (186 + 201) / 2 and rec3["widths"][0] == 15
    W4, rec4 = EC.cell_weights([100, 300], T, 16, 100.0)   # no overlap (201 < 236): a gap, BF0 fill in between
    assert rec4["gaps"] == 1 and np.all(W4.sum(axis=0)[202:236] == 0)


def test_no_time_warping_and_central_r_not_shifted():
    P = [70, 180, 300, 410]
    beats = [gauss_beat(offset=0.1 * i) for i in range(4)]
    y, info = EC.render(beats, P, T, 16, 100.0, 0.0)
    W, _ = EC.cell_weights(P, T, 16, 100.0)
    for i, p in enumerate(P):
        own = np.flatnonzero(W[i] == 1.0)
        np.testing.assert_allclose(y[own], beats[i][own - p + BB.R_INDEX], atol=1e-12)    # absolute offsets kept
        assert int(np.argmax(y[p - 5:p + 6])) == 5                                          # test 20


def test_render_does_not_modify_beat_arrays():
    beats = np.stack([gauss_beat() for _ in range(3)])
    before = beats.copy()
    EC.render(beats, [80, 200, 330], T, 16, 100.0, 0.0)
    np.testing.assert_array_equal(beats, before)                                    # test 7


def test_outside_cells_uses_bf0_nearest_covered_value_rule():
    P = [250]
    y, info = EC.render([gauss_beat(offset=0.3)], P, T, 16, 100.0, -9.0)
    lo, hi = info["left_limit"], info["right_limit"]
    assert (lo, hi) == (200.0, 300.0)
    assert np.all(y[:200] == y[200]) and np.all(y[301:] == y[300])                 # test 11
    assert info["flat_fill_fraction"] == pytest.approx(1 - 101 / T)
    cov = np.zeros(10, bool)
    cov[[3, 7]] = True
    filled = EC.nearest_fill(np.arange(10.0), cov)
    assert filled[5] == 3.0 and filled[6] == 7.0 and filled[0] == 3.0 and filled[9] == 7.0   # ties to the left
    assert np.all(EC.render([], [], T, 16, 100.0, -0.5)[0] == -0.5)


# ------------------------------------------------------------------------------------------ mechanism (tests 19, 21)
def test_synthetic_neighbour_ghost_is_exposed_by_bf0_and_removed_by_event_cells():
    P = [100, 190, 300]                                  # RR 90 then 110; each beat carries a ghost R at +90 samples
    beats = [gauss_beat(ghost_at=BB.R_INDEX + 90) for _ in P]
    y_orig = BR.assemble(beats, P, T, 0.0)[0]
    y_cell, info = EC.render(beats, P, T, 16, 100.0, 0.0)
    assert y_orig[390] > 0.9                             # the last beat's ghost: only that beat covers sample 390
    assert abs(y_cell[390]) < 1e-6                       # beyond the last cell (U = 355): BF0 fill of a flat value
    assert y_cell[300] == pytest.approx(y_orig[300], abs=0.05)


def test_boundary_statistics_and_audit_catch_a_step():
    P = [100, 200]
    beats = [np.zeros(BB.SEG_LEN), np.ones(BB.SEG_LEN)]                             # a level step between beats
    y0, info0 = EC.render(beats, P, T, 0, 100.0, 0.0)
    y16, info16 = EC.render(beats, P, T, 16, 100.0, 0.0)
    assert np.max(np.abs(np.diff(y0))) == pytest.approx(1.0)
    assert np.max(np.abs(np.diff(y16))) < 0.12                                      # pi / (2 * 16) ~ 0.098
    assert EC.boundary_near(152, info16["centres"], (info16["left_limit"], info16["right_limit"]), 16) == "crossfade"
    assert EC.boundary_near(60, info16["centres"], (info16["left_limit"], info16["right_limit"]), 16) == "edge_limit"
    assert EC.boundary_near(300, [150.0], (None, None), 16) is None


def test_boundary_stats_function(d0):
    y = np.concatenate([np.zeros(50), np.ones(50)])
    a, b = d0.boundary_stats(y, 49.5, 4)
    assert a == 1.0 and b == 1.0
    a2, _ = d0.boundary_stats(np.linspace(0, 1, 100), 50, 8)
    assert a2 == pytest.approx(1 / 99)


# ------------------------------------------------------------------------------------------ zones (test 22) and FP (16)
def test_far_classification_separates_edge_gap_and_interior():
    P = [100, 200, 300]
    assert EC.fp_zone(102, P)[0] == "A_within_50ms"
    assert EC.fp_zone(212, P)[0] == "B_after_50_150ms"
    assert EC.fp_zone(245, P)[0] == "C_after_150_450ms"
    assert EC.fp_zone(270, P)[0] == "D_before_300_50ms"                             # -234 ms
    assert EC.fp_zone(260, P)[:2] == ("E_far", "interior")                         # -312.5 ms
    assert EC.fp_zone(255, P)[:2] == ("E_far", "interior")                         # 351 ms before the next R
    assert EC.fp_zone(30, P)[:2] == ("E_far", "before_first")
    assert EC.fp_zone(470, P)[:2] == ("E_far", "after_last")
    assert EC.fp_zone(370, [50, 150, 250, 480])[:2] == ("E_far", "gap_missed_event")


def test_fp_matching_uses_reference_r_within_50ms(d0):
    assert d0.fp_detections([100, 300], [106, 300]) == []                          # 46.9 ms: matched
    assert d0.fp_detections([100, 300], [107, 300]) == [107]                       # 54.7 ms: false
    assert d0.fp_detections([100], [99, 101]) == [101]                              # one-to-one


# ------------------------------------------------------------------------------------------ selection (tests 12-13)
def test_width_selection_rule(d0):
    real = {w: {"A_p95": 0.05, "B_p95": 0.04} for w in (4, 8, 12, 16)}
    mk = lambda a: {w: {"A_p95": a[w][0], "B_p95": a[w][1]} for w in (4, 8, 12, 16)}  # noqa: E731
    ok8 = {4: (0.2, 0.1), 8: (0.05, 0.04), 12: (0.03, 0.02), 16: (0.02, 0.01)}
    assert d0.select_width({"S": mk(ok8), "D": mk(ok8)}, real) == (8, True)
    late = {4: (0.2, 0.1), 8: (0.2, 0.1), 12: (0.2, 0.1), 16: (0.04, 0.03)}
    assert d0.select_width({"S": mk(late), "D": mk(ok8)}, real) == (16, True)       # every arm must pass
    never = {w: (1.0, 1.0) for w in (4, 8, 12, 16)}
    assert d0.select_width({"S": mk(never), "D": mk(ok8)}, real) == (16, False)


def test_crossfade_selection_never_touches_validation_or_detection(d0):
    src = inspect.getsource(d0.stage_crossfade)
    assert 'load_role("train")' in src and 'load_role("val")' not in src
    for banned in ("B._peaks", "detect_rpeaks", "rpeak_prf_at", "kanflow_fd", "beat_level_metrics", "render_val"):
        assert banned not in src


# ------------------------------------------------------------------------------------------ frozen inputs (1, 14, 15, 18)
def test_frozen_beats_and_hard_stops(d0):
    src = inspect.getsource(d0.regenerate)
    assert "B.sample_stochastic(nets[\"stochastic\"], ppg, rr, own, 0, dev)" in src                # seed 0, BF0 seeds
    assert "B.predict_deterministic(nets[\"deterministic\"], ppg, rr, dev)" in src
    assert "B.conditions(X, pos, rr_median)" in src
    for fn in (d0.stage_reproduce, d0.stage_evaluate):
        s = inspect.getsource(fn)
        assert "SystemExit" in s and "assemble_all" in s                            # bit-exact original render or STOP
    assert "bf0_g1_reproduced" in inspect.getsource(d0.stage_evaluate) and "1e-12" in inspect.getsource(d0.stage_evaluate)


def test_original_renderer_is_deterministic():
    beats = [gauss_beat(offset=0.2), gauss_beat(), gauss_beat(offset=-0.1)]
    a = BR.assemble(beats, [90, 200, 320], T, 0.0)[0]
    b = BR.assemble([x.copy() for x in beats], [90, 200, 320], T, 0.0)[0]
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------------------------------ statistics and verdict (17)
def test_gap_recovery_uses_paired_patient_resamples(d0):
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(30), 4)
    placed = rng.uniform(0.7, 0.9, size=120)
    orig = placed - rng.uniform(0.02, 0.05, size=120)
    full = d0.gap_recovery(placed, orig, placed, pid, n_rep=100)
    none = d0.gap_recovery(orig, orig, placed, pid, n_rep=100)
    assert full["point"] == pytest.approx(1.0) and full["ci95"] == pytest.approx([1.0, 1.0])
    assert none["point"] == pytest.approx(0.0) and none["ci95"] == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize("flags,expected", [
    ({"D1": True, "D2": True, "D3": True, "D4": True}, "STRONGLY SUPPORTED"),
    ({"D1": True, "D2": False, "D3": True, "D4": True}, "PARTIALLY SUPPORTED"),
    ({"D1": True, "D2": True, "D3": False, "D4": True}, "PARTIALLY SUPPORTED"),
    ({"D1": True, "D2": True, "D3": True, "D4": False}, "PARTIALLY SUPPORTED"),
    ({"D1": False, "D2": True, "D3": True, "D4": True}, "NOT SUPPORTED"),
    ({"D1": True, "D2": True, "D3": True, "D4": True, "flat_fill_increase": 0.3}, "NOT SUPPORTED"),
    ({"D1": True, "D2": True, "D3": True, "D4": True, "boundary_new_fp_frac": 0.6}, "NOT SUPPORTED"),
    ({"D1": True, "D2": True, "D3": True, "D4": True, "boundary_new_fp_frac": 0.3}, "PARTIALLY SUPPORTED"),
    ({"D1": True, "D2": True, "D3": True, "D4": True, "fd_increase": 7.0}, "PARTIALLY SUPPORTED"),
])
def test_verdict_rules(d0, flags, expected):
    base = {"flat_fill_increase": 0.05, "boundary_new_fp_frac": 0.05, "fd_increase": 1.0, "fd_increase_max": 6.46}
    assert d0.d0_verdict(base | flags) == expected


def test_sensitivity_renderer_never_enters_the_verdict(d0):
    src = inspect.getsource(d0.stage_evaluate)
    flags_block = src[src.index("flags = {"):src.index("verdict = d0_verdict(flags)")]
    assert "sens" not in flags_block and "SENS_WIDTH" not in flags_block
    assert d0.SENS_WIDTH == max(d0.CANDIDATE_WIDTHS)
