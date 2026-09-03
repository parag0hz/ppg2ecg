"""O1 tests — docs/O1_ECG_COMPONENT_EXTRACTABILITY_PREREGISTRATION.md section 19.

Static source audits + numerical checks. No GPU, no probe training, no test subject.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import m1_structural as M1
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import q1_corruption as Q
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.probes import r1_cohort as C

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o1_component_extractability"
TRAIN_SRC = (ROOT / "scripts/o1_train_probes.py").read_text()
EVAL_SRC = (ROOT / "scripts/o1_evaluate.py").read_text()
BUILD_SRC = (ROOT / "scripts/o1_targets_build.py").read_text()
AUDIT_SRC = (ROOT / "scripts/o1_four_site_audit.py").read_text()
MOD_SRC = (ROOT / "src/ppg2ecg/evaluation/o1_targets.py").read_text()
PREREG = (ROOT / "docs/O1_ECG_COMPONENT_EXTRACTABILITY_PREREGISTRATION.md").read_text()
RNG = np.random.default_rng(11)


def _code(src: str, start: str, end: str) -> str:
    """Source of one region with docstrings and comments removed (ast round-trip)."""
    body = src[src.index(start):src.index(end)]
    tree = ast.parse(body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                    and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


# --------------------------------------------------------------------------------- firewall / roles
def test_test_subject_firewall():
    with pytest.raises(ER.WildPPGTestFirewallError):
        ER.assert_no_test_subjects(("an0", "kjd"))
    for src in (TRAIN_SRC, EVAL_SRC, BUILD_SRC, AUDIT_SRC, MOD_SRC):
        assert "kjd" not in src and "ssx" not in src
        assert "assert_no_test_subjects" in src or src is MOD_SRC


def test_exact_r1_subject_split():
    split = C.internal_dev_split()
    assert split["internal_dev"] == ("u7y", "e61") or set(split["internal_dev"]) == {"u7y", "e61"}
    assert set(split["probe_train"]) == {"fex", "l38", "n31", "ngh", "p5d", "p9p", "qm9", "trh", "tz8", "w4p"}
    assert C.VAL == ("an0", "k2s")
    assert "C.internal_dev_split()" in TRAIN_SRC and "C.cohort_positions" in BUILD_SRC
    assert C.COHORT_SALT == "r1-global-rhythm-observability-v1" and (C.N_TRAIN_PER, C.N_VAL_PER) == (2048, 1024)


def test_training_never_touches_validation():
    assert 'assert not (set(subs) & set(C.VAL))' in TRAIN_SRC
    body = _code(TRAIN_SRC, "def load_role", "def train_one")
    assert 'subs = PROBE_TRAIN if role == \'probe_train\' else INTERNAL_DEV' in body
    assert "C.VAL" in body                                   # only as the guard asserted above
    assert TRAIN_SRC.count("C.VAL") == 1
    assert "wildppg_8s/{s}.npz" in body and "an0" not in body and "k2s" not in body
    assert "TRAIN_ROLES = (\"probe_train\", \"internal_dev\")" in TRAIN_SRC
    # checkpoint selection is internal-dev only
    assert 'dev_mae < best' in TRAIN_SRC and '"selection": "internal_dev standardized MAE"' in TRAIN_SRC
    assert "validation" not in TRAIN_SRC.split("def main")[1].split("if args.preflight")[0]


def test_cohort_manifest_matches_the_frozen_r1_cohort():
    rows = list(csv.DictReader(open(ART / "cohort_manifest.csv")))
    assert len(rows) == 106_496
    by_role = {}
    for r in rows:
        by_role.setdefault(r["role"], set()).add(r["subject"])
    assert by_role["validation"] == {"an0", "k2s"} and len(by_role["probe_train"]) == 10
    val = [r for r in rows if r["role"] == "validation"]
    assert len(val) == 8192 and len({(r["subject"], r["site"]) for r in val}) == 8


# --------------------------------------------------------------------------------- four-site identity
def test_four_site_ecg_identity_established():
    a = json.loads((ART / "four_site_target_audit.json").read_text())
    assert a["identity_established"] is True
    rows = list(csv.DictReader(open(ART / "four_site_target_audit.csv")))
    assert len(rows) == 14
    for r in rows:
        assert int(r["n_groups_waveform_identical"]) == int(r["n_multi_site_groups"])
        assert int(r["n_deep_identical"]) == int(r["n_deep_checked"]) == 512


# --------------------------------------------------------------------------------- targets
def test_targets_use_frozen_primitives_and_are_deterministic():
    y = np.sin(2 * np.pi * np.arange(1024) / 128.0 * 1.2) + 0.3 * RNG.standard_normal(1024)
    a, b = OT.window_targets(y), OT.window_targets(y)
    for t in OT.TARGETS:
        assert (np.isnan(a[t]) and np.isnan(b[t])) or a[t] == b[t]
    assert "R.detect_rpeaks" in MOD_SRC and "M1.d1" in MOD_SRC and "M1.d2" in MOD_SRC
    assert "R.qrs_width_ms" in MOD_SRC and "hf_energy_ratio" in MOD_SRC
    assert "M1.CORE" in MOD_SRC and M1.CORE == 10
    assert "np.median" in MOD_SRC                      # per-window aggregation is the median
    assert len(OT.TARGETS) == 9 and OT.TARGET_IDS["median_QRS_width_ms"] == "T8"


def test_target_beat_validity_matches_m1():
    i = MOD_SRC.index("def window_targets")
    body = MOD_SRC[i:MOD_SRC.index("# ---", i)]
    assert "if lo - 1 < 0 or hi + 2 > y.size" in body
    assert "lo, hi = r - M1.CORE, r + M1.CORE" in body


def test_validation_statistics_never_used_for_scaling():
    sc = json.loads((ART / "target_scaling.json").read_text())
    assert "TRAIN median" in sc["rule"] and "validation statistics are never used" in sc["rule"]
    assert sc["computed_on"] == "unique ECG windows of the probe_train subjects"
    i = BUILD_SRC.index("med, iqr = float(np.nanmedian(v_tr))")
    assert "v_tr" in BUILD_SRC[i:i + 200] and "v_val" not in BUILD_SRC[i:i + 120]
    for t, v in sc["targets"].items():
        assert v["scale_train_IQR"] > 1e-9, t


def test_target_validity_and_variability_artifacts():
    val = {r["target"]: r for r in csv.DictReader(open(ART / "target_validity.csv"))}
    var = {r["target"]: r for r in csv.DictReader(open(ART / "target_variability.csv"))}
    assert set(val) == set(OT.TARGETS) == set(var)
    for t in OT.TARGETS:
        assert float(val[t]["valid_fraction_validation"]) >= OT.MIN_VALID_FRACTION
        assert float(var[t]["within_subject_variance"]) > 0


# --------------------------------------------------------------------------------- probe
def test_probe_has_no_ecg_and_no_gt_r_input():
    i = MOD_SRC.index("class ComponentGlobalTCN")
    body = MOD_SRC[i:MOD_SRC.index("def receptive_field")]
    src = ast.unparse(ast.parse(body))
    for bad in ("ecg", "rpeak", "peaks", "site"):
        assert bad not in src.lower(), bad
    net = OT.build_probe(42)
    import inspect
    assert list(inspect.signature(net.forward).parameters) == ["x"]
    assert tuple(net(torch.zeros(2, 1, 1024)).shape) == (2,)


def test_identical_architecture_and_parameter_count_across_targets_and_seeds():
    counts = {(t, s): OT.n_params(OT.build_probe(s)) for t in OT.TARGETS for s in OT.SEEDS}
    assert len(set(counts.values())) == 1
    n = next(iter(counts.values()))
    assert n == 328_897 and n <= OT.MAX_PARAMS
    assert "def build_probe(seed" in MOD_SRC and "target" not in MOD_SRC[MOD_SRC.index("def build_probe"):MOD_SRC.index("def n_params")]


def test_receptive_field_covers_the_window():
    assert OT.receptive_field() == 2041 >= 1024
    assert OT.DILATIONS == (1, 2, 4, 8, 16, 32, 64, 128)


def test_exact_seeds_and_frozen_hyperparameters():
    assert OT.SEEDS == (40, 42, 44)
    assert (OT.LR, OT.WEIGHT_DECAY, OT.BATCH) == (1e-3, 1e-4, 128)
    assert (OT.MAX_EPOCHS, OT.PATIENCE, OT.HUBER_BETA) == (30, 5, 1.0)
    assert OT.BUDGET_GPU_HOURS == 4.0 and OT.PREFLIGHT_STEPS == 100
    assert "SmoothL1Loss(beta=OT.HUBER_BETA)" in TRAIN_SRC
    assert "AdamW(net.parameters(), lr=OT.LR, weight_decay=OT.WEIGHT_DECAY)" in TRAIN_SRC
    for tok in ("40, 42, 44", "beta = 1.0", "patience 5", "batch **128**"):
        assert tok in PREREG or tok.replace("**", "") in PREREG


# --------------------------------------------------------------------------------- baselines
def test_b2_uses_rhythm_and_site_features_only():
    body = _code(EVAL_SRC, "def _ppg_rhythm", "def rhythm_features")
    assert "dsp_ppg_peaks" in body
    for bad in ("ptp", "std(", "mean(", "max(", "fft", "morph", "sum("):
        assert bad not in body, bad
    feats = EVAL_SRC[EVAL_SRC.index("def rhythm_features"):EVAL_SRC.index("def ridge_fit")]
    assert "C.SITES" in feats and "onehot" in feats
    assert f"alpha=OT.RIDGE_ALPHA" in EVAL_SRC and OT.RIDGE_ALPHA == 1.0
    assert "alpha search" not in EVAL_SRC.lower() and "GridSearch" not in EVAL_SRC


# --------------------------------------------------------------------------------- shuffles
def test_ss_shuffle_is_a_fixed_point_free_bijection_within_subject_site():
    sub = np.array(["an0"] * 64 + ["k2s"] * 64)
    site = np.array((["sternum"] * 32 + ["wrist"] * 32) * 2)
    wi = np.arange(128)
    p = OT.same_subject_shuffle(sub, site, wi)
    OT.assert_derangement(p)
    assert np.all(sub[p] == sub) and np.all(site[p] == site)
    assert OT.SS_SALT == "o1-same-subject-shuffle-v1"
    assert np.array_equal(p, OT.same_subject_shuffle(sub, site, wi))


def test_xs_shuffle_is_cross_subject_and_same_site():
    sub = np.array(["an0"] * 64 + ["k2s"] * 64)
    site = np.array((["sternum"] * 32 + ["wrist"] * 32) * 2)
    wi = np.arange(128)
    p = OT.cross_subject_shuffle(sub, site, wi)
    OT.assert_cross_subject(p, sub, site)
    assert OT.XS_SALT == "o1-cross-subject-shuffle-v1"
    assert np.array_equal(p, OT.cross_subject_shuffle(sub, site, wi))


# --------------------------------------------------------------------------------- bootstrap
def test_cluster_bootstrap_resamples_ecg_windows_with_all_site_rows_together():
    i = EVAL_SRC.index("def cluster_bootstrap")
    body = EVAL_SRC[i:EVAL_SRC.index("def metrics_row")]
    assert "UNDERLYING ECG WINDOW" in body and "np.unique(cl, return_inverse=True)" in body
    assert "groups[i]" in body and "rng.integers(0, len(groups), len(groups))" in body
    assert 'CLUSTER = np.array([f"{a}|{b}" for a, b in zip(SUB, WI)])' in EVAL_SRC
    assert "cluster_bootstrap(d_ss, SUB, CLUSTER)" in EVAL_SRC
    assert (OT.BOOT_N, OT.BOOT_SEED) == (2000, 20260903)


# --------------------------------------------------------------------------------- corruption reuse
def test_q1_corruption_functions_are_reused_unchanged():
    assert "from ppg2ecg.evaluation import q1_corruption as Q" in EVAL_SRC
    assert "Q.corrupt_block(Xv, cond" in EVAL_SRC
    assert set(("LP_1.25Hz", "SNR_0dB", "DROP_2.0s", Q.SHUFFLED, Q.NULL)) <= set(Q.CONDITIONS)
    x = np.sin(2 * np.pi * np.arange(1024) / 128.0 * 1.1)
    a = Q.corrupt_row(x, "SNR_0dB", "an0", "wrist", 3)
    assert abs(Q.achieved_snr_db(x, a)) < 1e-6
    assert "def corrupt_row" not in EVAL_SRC and "def apply_noise" not in EVAL_SRC


def test_clean_classification_is_frozen_before_the_corruption_analysis():
    i_cls = EVAL_SRC.index('wcsv(ART / "component_classification.csv", cls_rows)')
    i_dec = EVAL_SRC.index('(ART / "decision.json").write_text')
    i_corr = EVAL_SRC.index("secondary: corruption transfer")
    assert i_cls < i_dec < i_corr
    assert '"frozen_before_secondary_analyses": True' in EVAL_SRC


# --------------------------------------------------------------------------------- classification / verdict
def test_classification_implements_the_preregistration():
    assert (OT.SKILL_R_STRONG, OT.RHO_STRONG) == (0.10, 0.30)
    strong = OT.classify_component(0.2, 0.01, 0.5, True, True, True)
    assert strong["class"] == OT.CLASS_A
    assert OT.classify_component(0.05, 0.01, 0.5, True, True, True)["class"] == OT.CLASS_B      # Skill_R < 0.10
    assert OT.classify_component(0.2, 0.01, 0.1, True, True, True)["class"] == OT.CLASS_B       # rho < 0.30
    assert OT.classify_component(0.2, 0.01, 0.5, False, True, True)["class"] == OT.CLASS_B      # not all seeds beat B0
    assert OT.classify_component(0.2, -0.01, 0.5, True, True, True)["class"] == OT.CLASS_C      # SS unresolved
    assert OT.classify_component(-0.2, -0.01, 0.5, True, True, False)["class"] == OT.CLASS_C    # does not beat B2
    assert OT.classify_component(-0.2, -0.01, 0.1, False, False, False)["class"] == OT.CLASS_D
    assert "UNOBSERVABLE" not in MOD_SRC.upper().replace("NO CLEAR EXTRACTABILITY", "")


def test_verdict_implements_the_preregistration():
    prim = tuple(OT.TARGETS)
    ok = {t: True for t in prim}
    het = {t: (OT.CLASS_A if t in ("beat_count", "median_RR_ms") else OT.CLASS_D) for t in prim}
    r = OT.decide_o1(het, True, prim, ok)                                    # Amendment 1: rhythm-only high set
    assert r["verdict"] == OT.VERDICT_B and r["amendment_1_applied"] is True
    mixed = {t: (OT.CLASS_A if t in ("beat_count", "median_RR_ms", "median_QRS_p2p", "median_QRS_energy")
                 else OT.CLASS_C) for t in prim}
    assert OT.decide_o1(mixed, True, prim, ok)["verdict"] == OT.VERDICT_A
    broad = {t: OT.CLASS_B for t in prim}
    assert OT.decide_o1(broad, True, prim, ok)["verdict"] == OT.VERDICT_C
    assert OT.decide_o1(broad, False, prim, ok)["verdict"] == OT.VERDICT_D  # positive control gate
    for v in (OT.VERDICT_A, OT.VERDICT_B, OT.VERDICT_C, OT.VERDICT_D):
        assert v in PREREG


def test_positive_control_targets_and_gate():
    assert OT.POSITIVE_CONTROLS == ("beat_count", "median_RR_ms")
    assert ">= 0.70" in EVAL_SRC or "0.70" in EVAL_SRC
    assert "positive_control_ok" in EVAL_SRC


def test_generator_utilization_never_classifies_direct_extractability():
    i_cls = EVAL_SRC.index("cls = OT.classify_component")
    i_cw = EVAL_SRC.index("# ---------------- generator-utilization crosswalk")
    assert i_cls < i_cw
    block = EVAL_SRC[EVAL_SRC.index("# ---------------- skill, bootstrap, classification"):i_cw]
    for bad in ("q1art", "crosswalk", "utilization", "generator_fidelity"):
        assert bad not in block.lower(), bad
    sig = ast.unparse(ast.parse(MOD_SRC[MOD_SRC.index("def classify_component"):MOD_SRC.index("VERDICT_A =")]))
    for bad in ("generator", "utilization", "f1_excess", "qrs_e_dev"):
        assert bad not in sig.lower()


def test_amendment_1_is_documented_and_pre_result():
    doc = (ROOT / "docs/O1_PREREGISTRATION_AMENDMENT_1.md").read_text()
    for tok in ("Amendment 1", "before any probe was trained", "is evaluated before verdict A",
                "No threshold, no classification rule"):
        assert tok.lower() in doc.lower(), tok
    assert "amendment_1_applied" in MOD_SRC and "Amendment 1" in MOD_SRC
