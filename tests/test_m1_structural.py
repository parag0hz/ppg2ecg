"""Static / synthetic validation for M1, run BEFORE any M1 prediction.

Protocol: docs/M1_C1_STRUCTURAL_MECHANISM_AUDIT_PREREGISTRATION.md (959eb60).
"""
from __future__ import annotations

import ast as _ast
import json
from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import m1_structural as M
from ppg2ecg.evaluation.c2_cohort import ATLAS_SALT, PER_STRATUM, SITES, atlas_cohort
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts/analyze_m1_structural.py").read_text()


# ---------------------------------------------------------------- firewall / population
def test_no_test_subject_access():
    for bad in ER.TEST_SUBJECTS:
        with pytest.raises(ER.WildPPGTestFirewallError):
            ER.assert_no_test_subjects(["an0", "k2s", bad])
    body = "\n".join(l for l in SCRIPT.splitlines() if not l.strip().startswith("#"))
    assert "kjd" not in body and "ssx" not in body
    assert "assert_no_test_subjects(VAL)" in SCRIPT


def test_population_is_the_frozen_2048_window_subset():
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    tot = 0
    for s, n in (("an0", 22183), ("k2s", 27017)):
        got = ER.select_subset("x4-event-nfe-v2", s, n, 1024)
        np.testing.assert_array_equal(got, np.asarray(frozen[s], dtype=got.dtype))
        tot += got.size
    assert tot == 2048
    assert 'VAL, SALT, TAKE, NFES, SRC_SEED = ("an0", "k2s"), "x4-event-nfe-v2", 1024, (2, 4), 0' in SCRIPT


def test_same_gaussian_source_across_arms_and_nfe():
    assert SCRIPT.count("torch.randn(") == 1
    assert "e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))" in SCRIPT
    i_src, i_arm = SCRIPT.index("e0 = torch.randn"), SCRIPT.index("for arm in ARMS:")
    assert i_src < i_arm, "the source bank must be built before the arm loop"


# ---------------------------------------------------------------- ms -> sample conversion
def test_frozen_ms_to_sample_conversion():
    assert M.FS == 128
    assert M.CORE == int(round(80 / 1000 * 128)) == 10
    assert M.PERI == int(round(250 / 1000 * 128)) == 32
    assert M.PROFILE_TAU.size == 65
    assert M.PROFILE_TAU[0] == -32 and M.PROFILE_TAU[-1] == 32


# ---------------------------------------------------------------- region masks
def test_region_masks_partition_every_sample_exactly_once():
    tau = M.tau_map(1024, np.array([200, 500, 800]))
    m = M.region_masks(tau)
    stack = np.stack([m[k] for k in ("qrs_core", "peri_qrs", "background")])
    assert np.all(stack.sum(axis=0) == 1)
    assert stack.sum() == 1024


def test_region_boundaries_are_inclusive_as_preregistered():
    tau = np.array([-33.0, -32.0, -11.0, -10.0, 0.0, 10.0, 11.0, 32.0, 33.0])
    m = M.region_masks(tau)
    assert m["qrs_core"].tolist() == [False, False, False, True, True, True, False, False, False]
    assert m["peri_qrs"].tolist() == [False, True, True, False, False, False, True, True, False]
    assert m["background"].tolist() == [True, False, False, False, False, False, False, False, True]


def test_tau_map_uses_nearest_peak_and_is_signed():
    t = M.tau_map(20, np.array([5, 15]))
    assert t[5] == 0 and t[15] == 0
    assert t[4] == -1 and t[6] == 1
    assert t[10] in (-5.0, 5.0)                       # equidistant: some nearest peak, magnitude 5
    assert np.all(np.isinf(M.tau_map(10, np.zeros(0, dtype=int))))


# ---------------------------------------------------------------- no translation / no oracle
def _executable_source(path: Path) -> str:
    """Module source with every docstring stripped, so disclaimers cannot satisfy a purity scan."""
    tree = _ast.parse(path.read_text())
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.Module, _ast.ClassDef)):
            node.body = [n for n in node.body if not (isinstance(n, _ast.Expr)
                         and isinstance(n.value, _ast.Constant) and isinstance(n.value.value, str))]
    return _ast.unparse(tree).lower()


def test_no_alignment_or_oracle_anywhere_in_the_primary_path():
    code = _executable_source(ROOT / "src/ppg2ecg/evaluation/m1_structural.py")
    for forbidden in ("oracle", "argmax", "np.roll", "shift", "align", "match_rpeaks", "beat_level_analysis"):
        assert forbidden not in code, forbidden
    script = _executable_source(ROOT / "scripts/analyze_m1_structural.py")
    for forbidden in ("oracle_corr", "oracle_absent", "oracle_qrs", "beat_level_analysis"):
        assert forbidden not in script, forbidden


def test_metrics_are_invariant_to_nothing_but_the_signals_themselves():
    """A shifted prediction must NOT score like an aligned one -- proof that nothing re-aligns."""
    g = np.random.default_rng(0)
    gt = np.zeros(1024); pk = np.array([200, 400, 600, 800])
    for r in pk:
        gt[r - 3:r + 4] += np.array([0, .2, .6, 1., .6, .2, 0])
    gt += 0.01 * g.standard_normal(1024)
    exact = M.region_errors(gt.copy(), gt, pk)["qrs_core__a2_sq"]
    shifted = M.region_errors(np.roll(gt, 5), gt, pk)["qrs_core__a2_sq"]
    assert shifted > exact * 5, "a translated prediction must be penalised at fixed coordinates"


# ---------------------------------------------------------------- derivative / curvature definitions
def test_frozen_difference_operators():
    x = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    np.testing.assert_allclose(M.d1(x), [1, 3, 5, 7])
    np.testing.assert_allclose(M.d2(x), [2, 2, 2])       # x[n+1]-2x[n]+x[n-1]


# ---------------------------------------------------------------- spectral bands
def test_frozen_spectral_bands_and_nyquist_edge():
    assert [b[0] for b in M.BANDS] == ["F1", "F2", "F3", "F4"]
    assert [(b[1], b[2]) for b in M.BANDS] == [(0.5, 4.0), (4.0, 8.0), (8.0, 15.0), (15.0, 64.0)]
    assert M.BANDS[-1][2] == M.FS / 2
    assert M.WELCH == dict(fs=128, nperseg=256, noverlap=128, window="hann", detrend="constant")
    # a pure tone must land in exactly one band
    t = np.arange(1024) / M.FS
    e = M.band_energy(np.sin(2 * np.pi * 10.0 * t))
    assert max(e, key=e.get) == "F3"
    e = M.band_energy(np.sin(2 * np.pi * 25.0 * t))
    assert max(e, key=e.get) == "F4"


def test_spectral_metrics_shape_and_ratio_orientation():
    g = np.random.default_rng(1)
    gt = g.standard_normal(1024)
    s = M.spectral_metrics(gt.copy(), gt)
    assert len(s) == 16
    for name, _lo, _hi in M.BANDS:
        assert s[f"{name}__ratio_dev"] == pytest.approx(0.0, abs=1e-9)   # identical -> deviation 0
        assert s[f"{name}__err_energy"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- bootstrap orientation
def test_positive_oriented_difference_convention():
    s = np.array(["an0"] * 40 + ["k2s"] * 40)
    worse, better = np.full(80, 0.5), np.full(80, 0.2)          # lower is better
    r = paired_subject_bootstrap(worse, better, s, "lower_better", 200)
    assert r["point"] == pytest.approx(0.3) and r["verdict"] == "improves"
    r = paired_subject_bootstrap(np.full(80, 0.2), np.full(80, 0.5), s, "higher_better", 200)
    assert r["verdict"] == "improves"
    assert "HIGHER_BETTER = {\"raw_corr\", \"f1_excess\"}" in SCRIPT


def test_equal_subject_bootstrap_weighting():
    s = np.array(["an0"] * 900 + ["k2s"] * 100)
    a = np.zeros(1000); b = np.concatenate([np.zeros(900), np.ones(100)])
    r = paired_subject_bootstrap(a, b, s, "higher_better", 200)
    assert r["point"] == pytest.approx(0.5)                      # macro, not pooled 0.1


# ---------------------------------------------------------------- frozen visual cohort
def test_atlas_cohort_is_the_frozen_metadata_only_c2_cohort():
    assert ATLAS_SALT == "c2-visual-atlas-v1" and PER_STRATUM == 8
    tot = 0
    for s in ("an0", "k2s"):
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        idx = ER.select_subset("x4-event-nfe-v2", s, len(d["x"]), 1024)
        c = atlas_cohort(s, d["site"][idx], d["window_index"][idx])
        assert set(c) == set(SITES)
        for site in SITES:
            assert c[site].size == 8
            assert np.all(np.asarray(d["site"][idx])[c[site]] == site)
        tot += sum(v.size for v in c.values())
    assert tot == 64


def test_atlas_selection_uses_no_model_derived_quantity():
    tree = _ast.parse((ROOT / "src/ppg2ecg/evaluation/c2_cohort.py").read_text())
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.Module, _ast.ClassDef)):
            node.body = [n for n in node.body if not (isinstance(n, _ast.Expr)
                         and isinstance(n.value, _ast.Constant) and isinstance(n.value.value, str))]
    code = _ast.unparse(tree).lower()
    for forbidden in ("f1", "morph", "corr", "rmse", "error", "predict", "model", "rpeak"):
        assert forbidden not in code, forbidden


def test_script_verifies_the_cohort_size_before_use():
    assert "assert ncoh == 64" in SCRIPT


# ---------------------------------------------------------------- no training
def test_no_training_construct():
    for bad in (".backward(", "optimizer", "optim.", "zero_grad", ".train()", "torch.save("):
        assert bad not in SCRIPT, bad
    assert "requires_grad_(False)" in SCRIPT and "torch.no_grad()" in SCRIPT
