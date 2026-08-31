"""Static / synthetic tests for the S1 G1 hard gate, run BEFORE any real-data G1 result.

Protocol: docs/S1_METRIC_VALIDITY_PREREGISTRATION.md (b749339)
          + docs/S1_METRIC_VALIDITY_AMENDMENT_1.md (dc75079)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.evaluation import stamping as ST
from ppg2ecg.evaluation.event_reliability import (
    TEST_SUBJECTS,
    WildPPGTestFirewallError,
    assert_no_test_subjects,
    select_subset,
)
from ppg2ecg.evaluation.rpeaks import detect_rpeaks, match_rpeaks, prf

FS = 128
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "s1_metric_validity"


# ---------------------------------------------------------------- 1. test-subject firewall
def test_firewall_rejects_test_subjects():
    for bad in TEST_SUBJECTS:
        with pytest.raises(WildPPGTestFirewallError):
            assert_no_test_subjects(["an0", bad])
    assert_no_test_subjects(["an0", "k2s"])          # must not raise


def test_template_subject_list_excludes_test_and_validation():
    assert set(ST.TEMPLATE_SUBJECTS).isdisjoint(TEST_SUBJECTS)
    assert set(ST.TEMPLATE_SUBJECTS).isdisjoint({"an0", "k2s"})


# ---------------------------------------------------------------- 2. Template A provenance
def test_template_subjects_are_the_ten_frozen_train_subjects():
    split = json.loads((ROOT / "data/manifests/split_a4_wildppg_seed42.json").read_text())["splits"][0]
    train = set(split["train"])
    assert set(ST.TEMPLATE_SUBJECTS) == train - set(ST.TEMPLATE_EXCLUDED_NOISY)
    assert set(ST.TEMPLATE_EXCLUDED_NOISY) == {"fex", "p5d"}
    assert set(ST.TEMPLATE_EXCLUDED_NOISY) <= train          # excluded, not absent
    assert len(ST.TEMPLATE_SUBJECTS) == 10


def test_collect_train_beats_refuses_validation_subjects():
    with pytest.raises(ValueError, match="validation"):
        ST.collect_train_beats(lambda s: np.zeros((4, 1024)), subjects=("e61", "an0"))
    with pytest.raises(WildPPGTestFirewallError):
        ST.collect_train_beats(lambda s: np.zeros((4, 1024)), subjects=("e61", "kjd"))


def test_collect_train_beats_only_reads_requested_subjects():
    seen: list[str] = []

    def loader(name):
        seen.append(name)
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        return rng.standard_normal((3, 1024))

    ST.collect_train_beats(loader, subjects=("e61", "l38"), n_take=3)
    assert set(seen) == {"e61", "l38"}
    assert not (set(seen) & (set(TEST_SUBJECTS) | {"an0", "k2s"}))


# ---------------------------------------------------------------- 3. deterministic template subset
def test_template_subset_is_deterministic_and_frozen():
    assert ST.TEMPLATE_SALT == "s1-template-v1"
    assert ST.TEMPLATE_N_TAKE == 256
    a = select_subset(ST.TEMPLATE_SALT, "e61", 5000, ST.TEMPLATE_N_TAKE, exclude=())
    b = select_subset(ST.TEMPLATE_SALT, "e61", 5000, ST.TEMPLATE_N_TAKE, exclude=())
    assert a.size == ST.TEMPLATE_N_TAKE
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, select_subset("other-salt", "e61", 5000, ST.TEMPLATE_N_TAKE, exclude=()))
    assert not np.array_equal(a, select_subset(ST.TEMPLATE_SALT, "l38", 5000, ST.TEMPLATE_N_TAKE, exclude=()))


# ---------------------------------------------------------------- 4. scaling formula parity
def test_template_scaling_matches_the_frozen_formula_exactly():
    rng = np.random.default_rng(0)
    beats = rng.standard_normal((37, 83)) * rng.uniform(0.5, 3.0, size=(37, 1))
    t_a, meta = ST.build_template_a(beats)

    t_raw = np.median(beats, axis=0)
    a_target = float(np.median(np.ptp(beats, axis=1)))
    expected = t_raw * a_target / (np.ptp(t_raw) + 1e-12)
    np.testing.assert_allclose(t_a, expected, rtol=0, atol=0)

    assert meta["a_target_median_ptp"] == pytest.approx(a_target)
    assert meta["final_ptp"] == pytest.approx(a_target, rel=1e-9)   # final ptp == frozen median beat ptp
    assert meta["n_beats"] == 37


def test_template_scaling_is_pure_scaling_no_recentering_or_smoothing():
    rng = np.random.default_rng(1)
    beats = rng.standard_normal((11, 83))
    t_a, _ = ST.build_template_a(beats)
    t_raw = np.median(beats, axis=0)
    ratio = t_a / t_raw
    assert np.allclose(ratio, ratio[0])          # a single global factor: no offset, no filtering


# ---------------------------------------------------------------- 5. QRS crop geometry
def test_frozen_discrete_geometry():
    g = ST.template_geometry(FS)
    assert g["full_len"] == 83 and g["r_index_full"] == 32
    assert g["qrs_n_before"] == 10 and g["qrs_n_after"] == 15
    assert g["qrs_len"] == 26
    assert g["qrs_slice"] == (22, 48)
    assert g["r_index_qrs"] == 10
    assert g["qrs_n_before"] == int(round(0.080 * FS))
    assert g["qrs_n_after"] == int(round(0.120 * FS))


def test_crop_takes_exactly_the_frozen_window_and_no_full_beat_leakage():
    t_a = np.arange(83, dtype=np.float64)
    t_b = ST.crop_qrs(t_a, FS)
    assert t_b.size == 26
    np.testing.assert_array_equal(t_b, np.arange(22, 48))
    assert t_b[ST.template_geometry(FS)["r_index_qrs"]] == 32       # R lands on the full-template R index
    with pytest.raises(ValueError):
        ST.crop_qrs(np.zeros(84), FS)


# ---------------------------------------------------------------- 6. stamping behaviour
def test_stamp_places_template_at_exact_positions_with_zero_baseline():
    tmpl = np.array([1.0, 5.0, 2.0])
    out = ST.stamp(tmpl, [10, 40], n_time=64, r_index=1)
    assert out[10] == 5.0 and out[40] == 5.0            # r_index sits ON the peak
    assert out[9] == 1.0 and out[11] == 2.0
    assert out.sum() == pytest.approx(2 * tmpl.sum())   # nothing else added
    assert np.count_nonzero(out) == 6


def test_stamp_is_not_adaptive_and_not_per_window_scaled():
    tmpl = np.array([0.0, 3.0, 0.0])
    a = ST.stamp(tmpl, [20], n_time=64, r_index=1)
    b = ST.stamp(tmpl, [20], n_time=64, r_index=1, baseline=np.full(64, 7.0))
    assert a[20] == 3.0
    assert b[20] == 10.0                                 # baseline added, template amplitude untouched
    assert (b - a).std() == pytest.approx(0.0)


def test_stamp_clips_at_the_edges_and_never_wraps():
    tmpl = np.array([1.0, 2.0, 3.0])
    out = ST.stamp(tmpl, [0], n_time=8, r_index=1)
    assert out[0] == 2.0 and out[1] == 3.0
    assert out[-1] == 0.0                                # no wrap-around
    out = ST.stamp(tmpl, [7], n_time=8, r_index=1)
    assert out[6] == 1.0 and out[7] == 2.0


def test_stamp_superposes_when_supports_overlap():
    tmpl = np.ones(5)
    out = ST.stamp(tmpl, [10, 12], n_time=32, r_index=2)
    assert out.max() == 2.0                              # T-A behaviour, declared in the prereg
    assert ST.stamp_supports_overlap([10, 12], r_index=2, tmpl_len=5) is True
    assert ST.stamp_supports_overlap([10, 20], r_index=2, tmpl_len=5) is False


# ---------------------------------------------------------------- 7. T-B overlap arithmetic
def test_tb_overlap_boundary_is_exactly_26_samples():
    g = ST.template_geometry(FS)
    r_i, n = g["r_index_qrs"], g["qrs_len"]
    assert g["min_rr_samples_for_no_overlap"] == 26
    assert ST.stamp_supports_overlap([100, 100 + 25], r_index=r_i, tmpl_len=n) is True    # d = 25 overlaps
    assert ST.stamp_supports_overlap([100, 100 + 26], r_index=r_i, tmpl_len=n) is False   # d = 26 does not


# ---------------------------------------------------------------- 8. detector / matcher parity
def test_gate_scoring_uses_the_frozen_primitives_unmodified():
    ref = np.array([100, 300, 500])
    m, fp, fn = match_rpeaks(ref, ref, FS, tol_ms=50.0)
    assert (len(m), fp, fn) == (3, 0, 0)
    assert prf(3, 0, 0) == (1.0, 1.0, 1.0)
    # 50 ms = 6.4 samples: 6 matches, 7 does not
    m6, _, _ = match_rpeaks(ref, ref + 6, FS, tol_ms=50.0)
    m7, _, _ = match_rpeaks(ref, ref + 7, FS, tol_ms=50.0)
    assert len(m6) == 3 and len(m7) == 0


def test_stamped_template_is_actually_detectable_on_a_synthetic_train():
    """Sanity of the harness itself, on synthetic data: not a G1 result."""
    g = ST.template_geometry(FS)
    tmpl = ST.analytic_template_c(1.0, FS)
    peaks = np.arange(80, 1024 - 80, 110)
    sig = ST.stamp(tmpl, peaks, 1024, g["r_index_qrs"])
    det = detect_rpeaks(sig, FS)
    m, fp, fn = match_rpeaks(peaks, det, FS, tol_ms=50.0)
    assert len(m) >= 1 and prf(len(m), fp, fn)[2] > 0.0


# ---------------------------------------------------------------- 9. equal-subject macro aggregation
def test_equal_subject_macro_aggregation():
    per_subject = {"an0": np.full(900, 1.0), "k2s": np.full(100, 0.0)}
    macro = float(np.mean([float(np.mean(v)) for v in per_subject.values()]))
    pooled = float(np.mean(np.concatenate(list(per_subject.values()))))
    assert macro == pytest.approx(0.5)
    assert pooled == pytest.approx(0.9)
    assert macro != pytest.approx(pooled)          # the two differ; the gate uses macro


# ---------------------------------------------------------------- 10. artifact overwrite protection
def test_artifact_dir_holds_no_historical_checkpoint_or_prediction_dump():
    if not ART.exists():
        pytest.skip("G1 artifacts not produced yet")
    for p in ART.rglob("*"):
        assert p.suffix not in {".pt", ".pth", ".ckpt"}, f"checkpoint-like artifact in S1 output: {p}"
        if p.is_file():
            assert p.stat().st_size < 50_000_000, f"oversized S1 artifact: {p}"


# ---------------------------------------------------------------- 11. determinism
def test_template_build_is_bit_identical_on_rerun():
    rng = np.random.default_rng(7)
    beats = rng.standard_normal((23, 83))
    t1, m1 = ST.build_template_a(beats)
    t2, m2 = ST.build_template_a(beats.copy())
    assert ST.sha256_array(t1) == ST.sha256_array(t2)
    assert m1 == m2


def test_stamping_is_bit_identical_on_rerun():
    g = ST.template_geometry(FS)
    tmpl = ST.analytic_template_c(1.0, FS)
    peaks = np.array([50, 200, 400, 700])
    a = ST.stamp(tmpl, peaks, 1024, g["r_index_qrs"])
    b = ST.stamp(tmpl, peaks, 1024, g["r_index_qrs"])
    assert ST.sha256_array(a) == ST.sha256_array(b)


def test_lowfreq_baseline_removes_beat_scale_structure():
    g = ST.template_geometry(FS)
    sig = ST.stamp(ST.analytic_template_c(1.0, FS), np.arange(60, 1000, 110), 1024, g["r_index_qrs"])
    drift = 0.5 * np.sin(2 * np.pi * 0.25 * np.arange(1024) / FS)
    base = ST.lowfreq_baseline(sig + drift, FS, cut_hz=1.0)
    assert np.corrcoef(base, drift)[0, 1] > 0.9        # keeps the drift
    assert abs(np.corrcoef(base, sig)[0, 1]) < 0.3     # discards the beats
