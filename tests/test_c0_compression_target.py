"""Static / synthetic validation for C0, run BEFORE any real-data C0 result.

Protocol: docs/C0_IMF_COMPRESSION_TARGET_PREREGISTRATION.md (5df1a33).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.evaluation import alignment_diagnostics as AD
from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.evaluation.paired_stats import BOOT_N, BOOT_SEED, paired_subject_bootstrap
from ppg2ecg.evaluation.rpeaks import match_rpeaks, prf

FS, T = 128, 1024
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_c0_compression_target.py"
ART = ROOT / "artifacts/c0_imf_compression_target"
SRC = SCRIPT.read_text()


# ---------------------------------------------------------------- 1 exact frozen subset
def test_population_is_the_frozen_x4_0_stage_b_subset():
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    for s, n_total in (("an0", 22183), ("k2s", 27017)):
        got = ER.select_subset("x4-event-nfe-v2", s, n_total, 1024)
        assert got.size == 1024
        np.testing.assert_array_equal(got, np.asarray(frozen[s], dtype=got.dtype))
    assert 'SALT, TAKE = "x4-event-nfe-v2", 1024' in SRC


# ---------------------------------------------------------------- 2 test firewall
def test_test_firewall():
    for bad in ER.TEST_SUBJECTS:
        with pytest.raises(ER.WildPPGTestFirewallError):
            ER.assert_no_test_subjects(["an0", "k2s", bad])
    body = "\n".join(l for l in SRC.splitlines() if not l.strip().startswith("#"))
    assert "kjd" not in body and "ssx" not in body
    assert "assert_no_test_subjects(VAL)" in SRC


# ---------------------------------------------------------------- 3 one source bank, reused
def test_source_bank_is_drawn_once_and_reused_across_nfe():
    # the script must draw e0 exactly once, outside the NFE loop, and slice it per batch
    assert len(re.findall(r"torch\.randn\(", SRC)) == 1, "source must be drawn exactly once"
    assert re.search(r"e0 = torch\.randn\(len\(X\), 1, T_LEN, generator=torch\.Generator\(\)\.manual_seed\(SRC_SEED\)\)", SRC)
    assert "e0[i:i + BATCH]" in SRC
    i_src = SRC.index("e0 = torch.randn")
    i_loop = SRC.index("for n in NFES:", SRC.index("preds, nfe_seen"))
    assert i_src < i_loop, "the source bank must be created before the NFE loop"


def test_source_bank_is_deterministic_and_seed_specific():
    def bank(seed, n=8):
        return torch.randn(n, 1, T, generator=torch.Generator().manual_seed(int(seed)))
    assert torch.equal(bank(0), bank(0))
    assert not torch.equal(bank(0), bank(1))


# ---------------------------------------------------------------- 4 NFE parity
def test_nfe_grid_is_frozen_and_parity_is_asserted():
    assert "NFES = (1, 2, 4, 8)" in SRC
    assert "50" not in re.search(r"NFES = \([^)]*\)", SRC).group(0)
    assert 'assert got == {n}, f"NFE parity violated' in SRC


def test_uniform_schedules_have_the_requested_step_count():
    for n in (1, 2, 4, 8):
        h = ER.UNIFORM[n]
        assert len(h) == n
        assert sum(h) == pytest.approx(1.0)


# ---------------------------------------------------------------- 5 GT-fixed extraction parity
def test_raw_keys_come_from_the_frozen_x0_primitive():
    rng = np.random.default_rng(0)
    gt, pred = rng.standard_normal(T), rng.standard_normal(T)
    pk = np.array([200, 400, 600, 800])
    bl = AD.beat_level_analysis(pred, gt, pk, FS, int(round(AD.LOCAL_MAX_SHIFT_MS / 1000 * FS)))
    for k in ("raw_corr", "raw_qrs_energy_ratio", "raw_slope_ratio", "raw_p2p_ratio",
              "raw_qrs_rmse", "raw_rmse"):
        assert k in bl and bl[k].size == bl["n_beats"]
    assert bl["n_beats"] + bl["n_skipped_edge"] == pk.size          # every GT beat accounted for


def test_raw_stats_are_shift_free():
    """raw_* must be the same when the oracle radius changes; only oracle_* may move."""
    rng = np.random.default_rng(1)
    gt, pred = rng.standard_normal(T), rng.standard_normal(T)
    pk = np.array([300, 500, 700])
    a = AD.beat_level_analysis(pred, gt, pk, FS, 19)
    b = AD.beat_level_analysis(pred, gt, pk, FS, 5)
    np.testing.assert_allclose(a["raw_corr"], b["raw_corr"], atol=0)
    np.testing.assert_allclose(a["raw_qrs_rmse"], b["raw_qrs_rmse"], atol=0)


# ---------------------------------------------------------------- 6/7 no prediction detector, no shift, in the primary path
def test_primary_metrics_never_use_a_prediction_detector_or_a_shift_search():
    block = SRC[SRC.index("PRIMARY = ["):SRC.index("RATIO_RAW =")]
    assert "oracle" not in block
    assert all(k.startswith("raw_") for k in re.findall(r'"(raw_[a-z_]+)"', block))
    # the counted set is exactly the six preregistered names
    names = re.findall(r'\("([a-z0-9_]+)",\s+"raw_', block)
    assert names == ["raw_corr", "qrs_e_dev", "slope_dev", "p2p_dev", "raw_qrs_rmse", "raw_rmse"]
    # no oracle_* value is ever written into the primary table or the decision
    dec = SRC[SRC.index("# ---------------- frozen decision rule"):SRC.index('(OUT / "decision.json")')]
    assert "oracle" not in dec, "no oracle-derived value may enter the decision"


# ---------------------------------------------------------------- 8 orientation
def test_orientation_makes_positive_mean_better():
    s = np.array(["an0"] * 50 + ["k2s"] * 50)
    worse, better = np.full(100, 0.9), np.full(100, 0.95)
    r = paired_subject_bootstrap(worse, better, s, "higher_better", 100)
    assert r["point"] == pytest.approx(0.05) and r["verdict"] == "improves"
    hi_dev, lo_dev = np.full(100, 0.40), np.full(100, 0.10)
    r = paired_subject_bootstrap(hi_dev, lo_dev, s, "lower_better", 100)
    assert r["point"] == pytest.approx(0.30) and r["verdict"] == "improves"
    r = paired_subject_bootstrap(lo_dev, hi_dev, s, "lower_better", 100)
    assert r["verdict"] == "worsens"
    with pytest.raises(ValueError):
        paired_subject_bootstrap(worse, better, s, "bigger_is_nicer", 10)


def test_deviation_is_absolute_distance_from_one():
    for ratio, want in ((1.0, 0.0), (0.6, 0.4), (1.4, 0.4)):
        assert abs(ratio - 1.0) == pytest.approx(want)
    assert 'abs(v - 1.0) if is_dev else v' in SRC


# ---------------------------------------------------------------- 9 paired bootstrap
def test_bootstrap_is_paired_not_independent():
    """Perfectly correlated arms with a constant offset must give a zero-width CI when paired."""
    rng = np.random.default_rng(3)
    a = rng.normal(0, 5, 400)
    b = a + 0.2
    s = np.array(["an0"] * 200 + ["k2s"] * 200)
    r = paired_subject_bootstrap(a, b, s, "higher_better", 500)
    assert r["point"] == pytest.approx(0.2)
    assert (r["hi"] - r["lo"]) < 1e-9, "independent resampling would widen this interval"
    assert r["verdict"] == "improves"


def test_bootstrap_settings_and_reproducibility():
    assert BOOT_N == 2000 and BOOT_SEED == 20260901
    s = np.array(["an0"] * 30 + ["k2s"] * 30)
    a = np.random.default_rng(4).normal(size=60); b = a + np.random.default_rng(5).normal(size=60)
    assert paired_subject_bootstrap(a, b, s, "higher_better", 200) == \
           paired_subject_bootstrap(a, b, s, "higher_better", 200)


def test_bootstrap_weights_subjects_equally():
    s = np.array(["an0"] * 900 + ["k2s"] * 100)
    d = np.concatenate([np.zeros(900), np.ones(100)])
    r = paired_subject_bootstrap(np.zeros(1000), d, s, "higher_better", 100)
    assert r["point"] == pytest.approx(0.5)                 # macro, not the pooled 0.1


def test_bootstrap_rejects_unpaired_shapes():
    with pytest.raises(ValueError):
        paired_subject_bootstrap(np.zeros(10), np.zeros(11), np.array(["an0"] * 10), "higher_better", 10)


# ---------------------------------------------------------------- 10 chance-floor parity with S1
def test_chance_floor_uses_the_s1_construction():
    assert "S1.chance_random_phase" in SRC
    assert "S1.NULL_DRAWS" in SRC and "S1.NULL_SEED" in SRC
    assert S1.NULL_DRAWS == 20 and S1.NULL_SEED == 20260901
    rng = np.random.default_rng(S1.NULL_SEED)
    p = S1.chance_random_phase(9, T, rng)
    assert p.size == 9 and np.all(np.diff(p) > 0)
    gt = np.array([100, 300, 500])
    m, fp, fn = match_rpeaks(gt, gt, FS, S1.MATCH_TOL_MS)
    assert prf(len(m), fp, fn) == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------- 11/12 hygiene
def test_no_training_construct_and_no_checkpoint_write():
    for bad in (".backward(", "optimizer", "optim.", "zero_grad", ".train()", "torch.save"):
        assert bad not in SRC, f"training construct in C0 script: {bad}"
    assert "requires_grad_(False)" in SRC and "torch.no_grad()" in SRC


def test_decision_rule_matches_the_frozen_gates():
    assert "gate_a = (len(a_imp) >= 2) and (len(a_wor) == 0) and (not a_collapse)" in SRC
    assert "gate_b = gate_a and (len(b_imp) >= 2) and (len(b_wor) == 0) and (not b_degr)" in SRC
    assert "COMPRESSION PREMISE NOT ESTABLISHED / INCONCLUSIVE" in SRC


def test_artifacts_hold_no_checkpoint_or_oversized_dump():
    if not ART.exists():
        pytest.skip("C0 artifacts not produced yet")
    for p in ART.rglob("*"):
        assert p.suffix not in {".pt", ".pth", ".ckpt"}
        if p.is_file():
            assert p.stat().st_size < 50_000_000
