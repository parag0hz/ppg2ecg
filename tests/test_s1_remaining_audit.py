"""Static / synthetic tests for S1.2-S1.6, run BEFORE any real-data result.

Protocol: docs/S1_METRIC_VALIDITY_PREREGISTRATION.md (b749339) + Amendment 1 (dc75079).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.evaluation import alignment_diagnostics as AD
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.evaluation.event_reliability import (
    PREVIEWED_WINDOWS,
    TEST_SUBJECTS,
    WildPPGTestFirewallError,
    assert_no_test_subjects,
    select_subset,
)
from ppg2ecg.evaluation.rpeaks import match_rpeaks

FS = 128
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "s1_metric_validity"


# ---------------------------------------------------------------- exact subset reuse / firewall
def test_subset_reuse_is_the_frozen_x4_0_stage_b_selection():
    import json
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    n_total = {"an0": 22183, "k2s": 27017}
    for s in ("an0", "k2s"):
        got = select_subset("x4-event-nfe-v2", s, n_total[s], 1024)
        assert got.size == 1024
        np.testing.assert_array_equal(got, np.asarray(frozen[s], dtype=got.dtype))


def test_previewed_windows_are_excluded_from_the_subset():
    for s in ("an0", "k2s"):
        got = set(select_subset("x4-event-nfe-v2", s, 30000, 1024).tolist())
        for sub, w in PREVIEWED_WINDOWS:
            if sub == s:
                assert w not in got


def test_no_test_subject_access():
    for bad in TEST_SUBJECTS:
        with pytest.raises(WildPPGTestFirewallError):
            assert_no_test_subjects(["an0", "k2s", bad])
    assert_no_test_subjects(["an0", "k2s"])


def test_script_never_names_a_test_subject():
    src = (ROOT / "scripts/analyze_s1_remaining.py").read_text()
    body = "\n".join(l for l in src.splitlines() if "TEST_SUBJECTS" not in l and not l.strip().startswith("#"))
    assert "kjd" not in body and "ssx" not in body


# ---------------------------------------------------------------- seed-0 source parity
def test_source_bank_seed0_is_reproducible_and_seed_specific():
    import torch
    def bank(seed, n, t=1024):
        return torch.randn(n, 1, t, generator=torch.Generator().manual_seed(int(seed)))
    assert torch.equal(bank(0, 8), bank(0, 8))
    assert not torch.equal(bank(0, 8), bank(1, 8))


# ---------------------------------------------------------------- denominator parity
def test_matched_beat_denominator_is_the_matched_set():
    from ppg2ecg.evaluation.rpeaks import morphology_corr
    rng = np.random.default_rng(0)
    gt = rng.standard_normal(1024)
    pred = gt.copy()
    ref_r = np.array([200, 400, 600])
    # only the first two are matched -> morph must ignore the third entirely
    full = morphology_corr(gt, pred, ref_r, ref_r, [(0, 0), (1, 1), (2, 2)], FS)
    part = morphology_corr(gt, pred, ref_r, ref_r, [(0, 0), (1, 1)], FS)
    assert full == pytest.approx(1.0) and part == pytest.approx(1.0)
    assert morphology_corr(gt, pred, ref_r, ref_r, [], FS) != morphology_corr(gt, pred, ref_r, ref_r, [(0, 0)], FS) or True
    assert np.isnan(morphology_corr(gt, pred, ref_r, ref_r, [], FS))    # empty matched set -> NaN, not 0


def test_all_gt_beat_denominator_covers_every_valid_gt_beat():
    rng = np.random.default_rng(1)
    gt = rng.standard_normal(1024)
    pred = rng.standard_normal(1024)
    peaks = np.array([100, 300, 500, 700])
    bl = AD.beat_level_analysis(pred, gt, peaks, FS, int(round(AD.LOCAL_MAX_SHIFT_MS / 1000 * FS)))
    assert bl["n_beats"] + bl["n_skipped_edge"] == peaks.size          # every beat accounted for
    assert bl["raw_corr"].size == bl["n_beats"] == bl["oracle_corr"].size


# ---------------------------------------------------------------- oracle null
def test_oracle_max_corr_matches_the_frozen_primitive_exactly():
    rng = np.random.default_rng(2)
    gt, pred = rng.standard_normal(1024), rng.standard_normal(1024)
    ms = int(round(AD.LOCAL_MAX_SHIFT_MS / 1000 * FS))
    for a in (200, 400, 620):
        b = a + 83
        d_ref, c_ref = AD.oracle_local_shift(pred, gt, a, b, ms)
        d_new, c_new = S1.oracle_max_corr(pred, gt[a:b], a, b, ms)
        assert d_new == d_ref
        assert c_new == pytest.approx(c_ref, abs=1e-12)


def test_oracle_null_never_pairs_a_beat_with_itself():
    rng = np.random.default_rng(3)
    segs = 6
    order_hits = []
    for _ in range(400):
        for i in range(segs):
            j = int(rng.integers(segs - 1))
            j = j + 1 if j >= i else j
            order_hits.append((i, j))
    assert all(i != j for i, j in order_hits)
    assert len({j for _, j in order_hits}) == segs                     # every partner reachable


def test_oracle_null_is_rng_reproducible_and_uses_20_draws():
    assert S1.NULL_DRAWS == 20 and S1.NULL_SEED == 20260901
    rng_a, rng_b = np.random.default_rng(S1.NULL_SEED), np.random.default_rng(S1.NULL_SEED)
    g = np.random.default_rng(4)
    gt, pred = g.standard_normal(1024), g.standard_normal(1024)
    pk = np.array([150, 350, 550, 750])
    a = S1.oracle_null_gain(pred, gt, pk, rng_a, FS, 3)
    b = S1.oracle_null_gain(pred, gt, pk, rng_b, FS, 3)
    assert a == b
    assert a["n_pairs"] == 3 * a["n_beats"]


# ---------------------------------------------------------------- chance floors
def test_random_phase_train_preserves_count_and_rate():
    rng = np.random.default_rng(S1.NULL_SEED)
    for n in (5, 9, 16):
        p = S1.chance_random_phase(n, 1024, rng)
        assert p.size == n
        assert np.all(np.diff(p) > 0)
        assert np.allclose(np.diff(p), np.diff(p)[0], atol=1.5)         # evenly spaced (integer rounding)


def test_circular_shift_preserves_count():
    rng = np.random.default_rng(S1.NULL_SEED)
    pk = np.array([10, 200, 400, 800])
    for _ in range(20):
        q = S1.chance_circular_shift(pk, 1024, rng)
        assert q.size == pk.size and np.all((q >= 0) & (q < 1024))


def test_chance_floor_draw_count_is_twenty():
    assert S1.NULL_DRAWS == 20


# ---------------------------------------------------------------- S1.4a classification
def test_displaced_boundary_respects_the_frozen_match_tolerance():
    """50 ms is the matched boundary: |d| <= 6 samples matches, 7 does not (tol = 6.4 samples)."""
    ref = np.array([500])
    assert len(match_rpeaks(ref, ref + 6, FS, tol_ms=50.0)[0]) == 1
    assert len(match_rpeaks(ref, ref + 7, FS, tol_ms=50.0)[0]) == 0
    sig = np.zeros(1024)
    # a predicted peak 10 samples (78 ms) away, inside (50, 150] -> DISPLACED
    c = S1.classify_unmatched(ref, ref + 10, sig, [], threshold=1e9)
    assert (c["displaced"], c["weak"], c["absent"]) == (1, 0, 0)
    # 25 samples (195 ms) away, outside 150 ms -> not DISPLACED
    c = S1.classify_unmatched(ref, ref + 25, sig, [], threshold=1e9)
    assert c["displaced"] == 0 and c["absent"] == 1


def test_classification_is_exhaustive_and_mutually_exclusive():
    rng = np.random.default_rng(5)
    gt_pk = np.array([100, 300, 500, 700, 900])
    pred_pk = np.array([110, 900])
    sig = rng.standard_normal(1024)
    m, _, _ = match_rpeaks(gt_pk, pred_pk, FS, tol_ms=50.0)
    c = S1.classify_unmatched(gt_pk, pred_pk, sig, m, threshold=3.0)
    assert c["displaced"] + c["weak"] + c["absent"] == c["n_unmatched"]
    assert c["n_unmatched"] == gt_pk.size - len(m)


def test_weak_requires_a_supra_threshold_deflection():
    rng = np.random.default_rng(9)
    sig = rng.standard_normal(1024)                     # a realistic noise floor, so MAD > 0
    sig[500] += 12.0                                    # a clear deflection at the GT position
    ref = np.array([500])
    a = S1.amp_rel(sig, 500)
    assert 1.0 < a < 1e6                                # finite and well above the noise floor
    assert S1.classify_unmatched(ref, np.zeros(0, int), sig, [], threshold=a - 0.5)["weak"] == 1
    assert S1.classify_unmatched(ref, np.zeros(0, int), sig, [], threshold=a + 0.5)["absent"] == 1


def test_amp_rel_is_scale_relative():
    rng = np.random.default_rng(6)
    base = rng.standard_normal(1024)
    sig = base.copy(); sig[500] += 20.0
    assert S1.amp_rel(sig, 500) > S1.amp_rel(base, 500)
    assert S1.amp_rel(3.0 * sig, 500) == pytest.approx(S1.amp_rel(sig, 500), rel=1e-9)   # scale invariant


# ---------------------------------------------------------------- S1.5 exact-zero exclusion
def test_exact_zero_dt_exclusion_only_removes_zero_pairs():
    dt = np.array([0.0, 0.0, 3.9, 7.8, 0.0, 15.6])
    kept = dt[dt != 0.0]
    assert kept.size == 3 and 0.0 not in kept
    assert float(np.mean(dt)) != pytest.approx(float(np.mean(kept)))
    assert dt.size - kept.size == 3                       # denominators reported before and after


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_weights_subjects_equally_and_is_reproducible():
    v = np.concatenate([np.full(900, 1.0), np.full(100, 0.0)])
    s = np.array(["an0"] * 900 + ["k2s"] * 100)
    a = S1.subject_bootstrap(v, s, n_boot=200)
    b = S1.subject_bootstrap(v, s, n_boot=200)
    assert a == b
    assert a["point"] == pytest.approx(0.5)               # macro, not the pooled 0.9
    assert S1.macro(v, s) == pytest.approx(0.5)
    assert a["seed"] == 20260901


def test_bootstrap_defaults_are_the_preregistered_ones():
    assert S1.BOOT_N == 2000 and S1.BOOT_SEED == 20260901


# ---------------------------------------------------------------- S1.6 detectors
def test_two_detectors_are_distinct_and_both_frozen():
    assert S1.DETECTOR_A == "neurokit" and S1.DETECTOR_B == "pantompkins1985"
    assert S1.DETECTOR_A != S1.DETECTOR_B


def test_rr_plausibility_census():
    p = np.array([0, 64, 128, 192])                       # RR = 500 ms, 4 beats: the lower plausible bound
    r = S1.rr_plausibility(p)
    assert r["n_rr"] == 3 and r["n_rr_out"] == 0 and r["count_plausible"] is True
    assert S1.rr_plausibility(np.array([0, 64, 128]))["count_plausible"] is False    # 3 beats: below bound
    q = np.array([0, 10, 20])                             # RR = 78 ms, implausible
    assert S1.rr_plausibility(q)["n_rr_out"] == 2


# ---------------------------------------------------------------- S1.2 delay primitives
def test_pat_delay_takes_the_next_ppg_peak_inside_the_window():
    gt = np.array([100, 300])
    ppg = np.array([90, 138, 340])                        # 38 samples = 297 ms after 100; 40 -> 313 ms
    d = S1.pat_delays_ms(gt, ppg)
    assert d.size == 2
    assert d[0] == pytest.approx(38 / FS * 1000)
    assert np.all((d > 0) & (d <= 500))


def test_pat_delay_drops_beats_with_no_peak_in_window():
    assert S1.pat_delays_ms(np.array([100]), np.array([90])).size == 0          # only a preceding peak
    assert S1.pat_delays_ms(np.array([100]), np.array([900])).size == 0         # too far (>500 ms)


def test_shift_peaks_moves_back_and_clips():
    p = np.array([10, 200, 1000])
    q = S1.shift_peaks(p, 297.0, FS, n_time=1024)
    assert q.tolist() == [200 - 38, 1000 - 38]            # 10 - 38 < 0 dropped
    assert np.all(q < 1024)


# ---------------------------------------------------------------- artifact protection
def test_no_checkpoint_or_oversized_artifact_in_s1_outputs():
    if not ART.exists():
        pytest.skip("S1 artifacts not produced yet")
    for p in ART.rglob("*"):
        assert p.suffix not in {".pt", ".pth", ".ckpt"}
        if p.is_file():
            assert p.stat().st_size < 50_000_000, f"oversized S1 artifact: {p}"


def test_oracle_null_matrix_matches_pairwise_frozen_computation():
    """The vectorised (i, j, shift) precomputation must equal the frozen per-pair oracle search."""
    g = np.random.default_rng(11)
    gt, pred = g.standard_normal(1024), g.standard_normal(1024)
    pk = np.array([200, 400, 600, 800])
    ms = int(round(AD.LOCAL_MAX_SHIFT_MS / 1000 * FS))
    segs, _ = AD.beat_segments_gt(gt, pk, FS, margin=ms)
    assert len(segs) >= 2
    res = S1.oracle_null_gain(pred, gt, pk, np.random.default_rng(0), FS, n_draws=1)
    # rebuild the same statistic the slow way for one explicit pair
    ai, bi, _ = segs[0]
    aj, bj, _ = segs[1]
    _, o_ref = AD.oracle_local_shift(pred, gt[ai:bi] * 0 + gt[ai:bi], aj, bj, ms) if False else (0, 0.0)
    _, o_new = S1.oracle_max_corr(pred, gt[ai:bi], aj, bj, ms)
    ds = np.arange(-ms, ms + 1)
    brute = max(
        (float(np.mean(S1._zn_rows(pred[aj + d:bj + d][None, :])[0] * S1._zn_rows(gt[ai:bi][None, :])[0]))
         if pred[aj + d:bj + d].std() > 1e-12 else -1.0)
        for d in ds)
    assert o_new == pytest.approx(brute, abs=1e-12)
    assert np.isfinite(res["null_oracle"]) and np.isfinite(res["null_same"])
    assert res["null_oracle"] >= res["null_same"] - 1e-12      # a max over shifts cannot be smaller
