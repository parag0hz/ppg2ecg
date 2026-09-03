"""O2 tests — docs/O2_ORACLE_EVENT_CANONICALIZATION_PREREGISTRATION.md section 14.

Stage 0 rejected the canonicalization operator, so no generator was trained; these tests cover everything that
exists: the warp operator, leakage by construction, the cohort/firewall, and the frozen gate/verdict logic.
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
from ppg2ecg.evaluation import o2_warp as W
from ppg2ecg.evaluation import rpeaks as R

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2_oracle_canonicalization"
MOD_SRC = (ROOT / "src/ppg2ecg/evaluation/o2_warp.py").read_text()
S0_SRC = (ROOT / "scripts/o2_stage0_roundtrip.py").read_text()
PREREG = (ROOT / "docs/O2_ORACLE_EVENT_CANONICALIZATION_PREREGISTRATION.md").read_text()
AUDIT = (ROOT / "docs/O2_CANONICAL_WARP_AUDIT.md").read_text()
RNG = np.random.default_rng(3)


def _code(src: str, start: str, end: str) -> str:
    """Source of one region with docstrings removed (ast round-trip)."""
    tree = ast.parse(src[src.index(start):src.index(end)])
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                    and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)
R5 = np.array([120, 340, 551, 790, 1000])


# --------------------------------------------------------------------------------- firewall / pins / integrity
def test_test_subject_firewall_and_no_test_names():
    with pytest.raises(ER.WildPPGTestFirewallError):
        ER.assert_no_test_subjects(("an0", "kjd"))
    for src in (MOD_SRC, S0_SRC):
        assert "kjd" not in src and "ssx" not in src
    assert "assert_no_test_subjects" in S0_SRC


def test_pins_and_c2_untouched():
    sub = (ROOT / ".gitmodules").read_text()
    assert "PENGUIN" in sub
    assert not list((ROOT / "outputs").glob("*c2*")), "C2 must remain untrained"
    assert (ROOT / "docs/C2_DEFERRED_BEFORE_TRAINING.md").exists()


def test_evaluation_cohort_is_the_frozen_2048_subset():
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    for s, n_total in (("an0", 22183), ("k2s", 27017)):
        got = ER.select_subset("x4-event-nfe-v2", s, n_total, 1024)
        assert got.size == 1024 and np.array_equal(got, np.asarray(frozen[s], dtype=got.dtype))
    assert 'VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024' in S0_SRC
    assert 'len(X) != 2048 or n_beats != 19834' in S0_SRC


def test_gt_detector_is_unchanged():
    assert "R.detect_rpeaks" in S0_SRC and "detect_rpeaks(np.asarray(sig, dtype=np.float64), OT.FS)" in S0_SRC
    assert "method=" not in S0_SRC


# --------------------------------------------------------------------------------- warp mechanics
def test_canonical_schedule_definition():
    q = W.canonical_positions(R5)
    assert q[0] == R5[0] and q[-1] == R5[-1]
    assert np.allclose(np.diff(q), (R5[-1] - R5[0]) / (len(R5) - 1))
    assert W.canonical_positions(np.array([100, 500])) is None          # K < 3 -> identity
    assert W.MIN_BEATS == 3 and W.ANCHOR_W == 10


def test_map_is_monotone_with_monotone_inverse_and_exact_boundaries():
    w = W.EventWarp(R5)
    assert w.valid() and not w.identity
    t = np.arange(1024, dtype=float)
    tau = w.forward(t)
    assert np.all(np.diff(tau) > 0)
    assert np.all(np.diff(w.inverse(t)) > 0)
    assert w.forward(0.0) == 0.0 and w.forward(1023.0) == 1023.0
    assert w.inverse(0.0) == 0.0 and w.inverse(1023.0) == 1023.0
    assert np.all(np.isfinite(tau)) and np.all(w.slopes() > 0)


def test_r_maps_to_q_and_back():
    w = W.EventWarp(R5)
    np.testing.assert_allclose(w.forward(R5), w.q, atol=1e-9)
    np.testing.assert_allclose(w.inverse(w.q), R5.astype(float), atol=1e-9)


def test_local_slope_is_one_across_the_qrs_core():
    w = W.EventWarp(R5)
    np.testing.assert_allclose(w.core_slopes(), 1.0, atol=1e-9)
    assert M1.CORE == W.ANCHOR_W == 10                                   # the anchor half-width IS the QRS core


def test_identity_warp_is_bit_exact_and_used_for_k_lt_3():
    x = torch.from_numpy(RNG.standard_normal((3, 1, 1024)).astype(np.float32))
    warps = [W.EventWarp(np.array([100, 500])), W.EventWarp(R5), W.EventWarp(np.array([300]))]
    assert warps[0].identity and warps[0].status == "K<3" and warps[2].identity
    out = W.apply_warp(x, warps, "to_canonical")
    assert torch.equal(out[0], x[0]) and torch.equal(out[2], x[2])
    assert not torch.equal(out[1], x[1])
    assert torch.equal(W.round_trip(x, [warps[0]] * 3)[0], x[0])


def test_no_amplitude_jacobian_and_no_renormalisation():
    body = _code(MOD_SRC, "def resample_at", "def warp_positions")
    for bad in ("jacobian", "slope *", "/ scale", "normalize", "zscore", "std(", "ptp("):
        assert bad not in body.lower(), bad
    x = torch.from_numpy((3.0 * RNG.standard_normal((1, 1, 1024))).astype(np.float32))
    w = [W.EventWarp(R5)]
    y = W.apply_warp(x, w, "to_canonical")
    assert float(y.abs().max()) <= float(x.abs().max()) + 1e-6           # interpolation never inflates amplitude
    assert abs(float(y.std()) - float(x.std())) < 0.5 * float(x.std())   # no re-standardisation


def test_same_warp_object_is_applied_to_ppg_and_ecg():
    assert "apply_warp" in S0_SRC or "round_trip" in S0_SRC
    assert "def apply_warp(x: torch.Tensor, warps" in MOD_SRC          # one entry point, modality-agnostic
    x = torch.from_numpy(RNG.standard_normal((2, 1, 1024)).astype(np.float32))
    w = [W.EventWarp(R5), W.EventWarp(R5)]
    a, b = W.apply_warp(x, w, "to_canonical"), W.apply_warp(x, w, "to_canonical")
    assert torch.equal(a, b)                                            # deterministic


def test_warp_uses_gt_r_only_as_coordinates():
    """No GT ECG sample value and no event feature can reach a model tensor through this module."""
    tree = ast.parse(MOD_SRC)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for bad in ("soft_event_field", "detect_rpeaks", "gate", "embedding"):
        assert bad not in names
    code = _code(MOD_SRC, "FS, T_LEN", "ROUNDTRIP_GATE").lower()   # the whole operator, docstrings removed
    code = code.replace("eventwarp", "").replace("centeronlywarp", "")   # class names only
    for bad in ("event", "phase", "beat_count", "rr_", "token", "field", "embed", "ecg", "soft_", "sigmoid"):
        assert bad not in code, bad
    import inspect
    assert list(inspect.signature(W.EventWarp.__init__).parameters)[1] == "r"   # takes R positions, not the ECG


# --------------------------------------------------------------------------------- gate / verdict logic
def test_roundtrip_gate_matches_the_preregistration():
    assert W.ROUNDTRIP_GATE == {"raw_rmse": 0.020, "T6": 0.020, "T7": 0.020, "T4": 0.020, "T8": 0.020,
                                "f1_at_50": 0.98, "beat_count_diff": 0}
    good = {"raw_rmse": 0.01, "T6": 0.01, "T7": 0.01, "T4": 0.01, "T8": 0.01, "f1_at_50": 0.99, "beat_count_diff": 0}
    assert W.roundtrip_gate(good)["passed"]
    for k, bad in (("raw_rmse", 0.03), ("T6", 0.03), ("T7", 0.03), ("T4", 0.03), ("T8", 0.03),
                   ("f1_at_50", 0.97), ("beat_count_diff", 1)):
        assert not W.roundtrip_gate(dict(good, **{k: bad}))["passed"], k
    for tok in ("0.020", "0.98", "R0-1", "R0-6"):
        assert tok in PREREG or tok in AUDIT


def test_verdicts_match_the_preregistration():
    j = {k: True for k in ("J1", "J2", "J3", "J4", "J5", "J6", "J7")}
    assert W.decide_o2(j)["verdict"] == W.VERDICT_A
    assert W.decide_o2(dict(j, J2=False))["verdict"] == W.VERDICT_B
    assert W.decide_o2(dict(j, J1=False, morphology_improves=True))["verdict"] == W.VERDICT_C
    assert W.decide_o2({k: False for k in ("J1", "J2", "J3", "J4", "J5", "J6", "J7")})["verdict"] == W.VERDICT_D
    assert (W.NONINF_MARGIN, W.F1_EXCESS_MIN) == (0.020, 0.10)
    for v in (W.VERDICT_REJECT, W.VERDICT_A, W.VERDICT_B, W.VERDICT_C, W.VERDICT_D):
        assert v in PREREG


# --------------------------------------------------------------------------------- stage-0 artifacts
def test_stage0_artifacts_are_consistent_with_the_rejection():
    d = json.loads((ART / "stage0_result.json").read_text())
    assert d["n_windows"] == 2048 and d["n_gt_beats"] == 19834
    assert d["all_warps_valid"] is True
    assert d["other_fallback_fraction"] <= W.FALLBACK_BUDGET
    assert abs(d["core_slope_min"] - 1.0) < 1e-9 and abs(d["core_slope_max"] - 1.0) < 1e-9
    assert d["gate"]["passed"] is False                                   # Stage 0 rejected the operator
    med = d["qrs_preserving_medians"]
    assert med["raw_rmse"] <= 0.020 and med["f1_at_50"] >= 0.98 and med["beat_count_diff"] == 0
    assert med["T6"] > 0.020 and med["T7"] > 0.020                        # the failing checks
    rows = list(csv.DictReader(open(ART / "warp_roundtrip_metrics.csv")))
    assert len(rows) == 2048 and {r["variant"] for r in rows} == {"qrs_preserving"}
    assert len(list(csv.DictReader(open(ART / "center_only_roundtrip_metrics.csv")))) == 2048
    assert not (ROOT / "outputs/o2_canon_oracle_seed42").exists()          # no generator was trained


def test_o1_aligned_scaling_reuses_the_frozen_o1_iqr():
    sc = json.loads((ROOT / "artifacts/o1_component_extractability/target_scaling.json").read_text())["targets"]
    d = json.loads((ART / "stage0_result.json").read_text())
    for t, v in d["o1_train_iqr"].items():
        assert abs(v - sc[t]["scale_train_IQR"]) < 1e-12
    assert 'target_scaling.json' in S0_SRC and "OT.window_targets" in S0_SRC
    assert set(d["o1_train_iqr"]) == {"median_QRS_p2p", "median_QRS_max_abs_derivative",
                                      "median_QRS_curvature_energy", "median_QRS_width_ms"}
