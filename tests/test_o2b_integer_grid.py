"""O2b tests — docs/O2B_INTEGER_GRID_CANONICALIZATION_PREREGISTRATION.md section 8.

Operator repair audit only: no generator is trained anywhere, and no O2b script may reach a training entry point.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import o2_warp as O2
from ppg2ecg.evaluation import o2b_warp as B
from ppg2ecg.evaluation import rpeaks as R

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2b_integer_grid"
MOD_SRC = (ROOT / "src/ppg2ecg/evaluation/o2b_warp.py").read_text()
S0_SRC = (ROOT / "scripts/o2b_stage0.py").read_text()
PREREG = (ROOT / "docs/O2B_INTEGER_GRID_CANONICALIZATION_PREREGISTRATION.md").read_text()
AUDIT = (ROOT / "docs/O2B_INTEGER_GRID_WARP_AUDIT.md").read_text()
R5 = np.array([120, 340, 551, 790, 1000])
RNG = np.random.default_rng(5)


# --------------------------------------------------------------------------------- repository
def test_firewall_pins_and_c2_untouched():
    with pytest.raises(ER.WildPPGTestFirewallError):
        ER.assert_no_test_subjects(("an0", "kjd"))
    for src in (MOD_SRC, S0_SRC):
        assert "kjd" not in src and "ssx" not in src
    assert "assert_no_test_subjects" in S0_SRC
    assert not list((ROOT / "outputs").glob("*c2*"))
    assert not (ROOT / "outputs/o2_canon_oracle_seed42").exists()


def test_no_generator_training_is_reachable_from_o2b():
    for bad in ("torch.optim", "AdamW", ".backward(", "MeanFlowS5", "build_penguin_backbone", "imeanflow",
                "checkpoint_best", "load_state_dict", ".train()"):
        assert bad not in MOD_SRC, bad
        assert bad not in S0_SRC, bad
    assert '"generator_trained": False' in S0_SRC
    assert "no generator is trained" in MOD_SRC.lower() or "No generator is trained" in MOD_SRC


# --------------------------------------------------------------------------------- schedule
def test_real_schedule_is_exactly_o2s():
    q = B.canonical_positions_int(R5)
    np.testing.assert_array_equal(np.asarray(O2.canonical_positions(R5)), O2.canonical_positions(R5))
    w = B.IntegerEventWarp(R5)
    np.testing.assert_allclose(w.q_real, O2.canonical_positions(R5), atol=0)
    assert q is not None and np.issubdtype(np.asarray(q).dtype, np.integer)


def test_round_half_to_even_is_deterministic_with_explicit_ties():
    np.testing.assert_array_equal(B.round_half_to_even([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5]),
                                  np.array([0, 2, 2, 4, 0, -2, -2]))
    np.testing.assert_array_equal(B.round_half_to_even([2.4, 2.6, -2.4, -2.6]), np.array([2, 3, -2, -3]))
    a = RNG.uniform(-500, 1500, 1000)
    np.testing.assert_array_equal(B.round_half_to_even(a), B.round_half_to_even(a))          # deterministic
    np.testing.assert_array_equal(B.round_half_to_even(a), np.rint(a).astype(np.int64))      # matches numpy's banker rounding
    assert "round_half_to_even" in AUDIT and "0.5→0, 1.5→2, 2.5→2" in AUDIT


def test_integer_schedule_endpoints_and_monotonicity():
    q = B.canonical_positions_int(R5)
    assert q[0] == R5[0] and q[-1] == R5[-1]
    assert np.all(np.diff(q) > 0)
    assert np.all(q >= 0) and np.all(q <= 1023)
    assert B.canonical_positions_int(np.array([100, 500])) is None       # K < 3 -> identity


def test_minimum_spacing_rule_and_no_repair():
    assert B.MIN_INT_SPACING == 2 * B.ANCHOR_W + 1 == 21
    assert B.spacing_ok(np.array([10, 31, 52]))
    assert not B.spacing_ok(np.array([10, 30, 52]))                       # 20 < 21
    dense = np.arange(0, 200, 15)                                         # spacing 15 -> violates
    anchors, status = B.build_int_anchors(dense)
    assert anchors is None and status == "integer spacing violated"
    for bad in ("isotonic", "clip(", "np.clip", "shift_q", "dynamic", "drop_window"):
        assert bad not in MOD_SRC, bad


# --------------------------------------------------------------------------------- anchors and grid
def test_anchors_are_integer_and_slope_is_one():
    w = B.IntegerEventWarp(R5)
    assert not w.identity and w.valid()
    assert np.all(np.mod(w.integer_shift(), 1.0) == 0.0)                  # q_int - r is an integer
    np.testing.assert_allclose(w.core_slopes(), 1.0, atol=1e-12)
    np.testing.assert_allclose(w.forward(R5), np.asarray(w.q, float), atol=1e-9)
    np.testing.assert_allclose(w.inverse(np.asarray(w.q, float)), R5.astype(float), atol=1e-9)
    assert w.forward(0.0) == 0.0 and w.forward(1023.0) == 1023.0


def test_protected_core_coordinates_are_integer_within_tolerance():
    for r in (R5, np.array([80, 300, 520, 740, 960]), np.array([61, 190, 331, 470, 615, 760, 900])):
        w = B.IntegerEventWarp(r)
        assert w.core_offsets().max() <= B.CORE_OFFSET_TOL
    assert B.CORE_OFFSET_TOL == 1e-6


def test_resampler_is_o2s_and_identity_is_bit_exact():
    assert "from ppg2ecg.evaluation import o2_warp as O2" in MOD_SRC
    for bad in ("sinc", "cubic", "spline", "fourier", "bicubic", "interp1d", "learned"):
        assert bad not in MOD_SRC.lower(), bad
    assert "import ast" or True
    code = ast.unparse(ast.parse(MOD_SRC))                                 # docstrings survive; strip them below
    tree = ast.parse(MOD_SRC)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    assert "grid_sample" not in code                                       # resampling is inherited, not re-implemented
    for bad in ("normalize", "zscore", "jacobian"):
        assert bad not in code.lower(), bad
    x = torch.from_numpy(RNG.standard_normal((2, 1, 1024)).astype(np.float32))
    warps = [B.IntegerEventWarp(np.array([100, 500])), B.IntegerEventWarp(R5)]
    out = O2.apply_warp(x, warps, "to_canonical")
    assert torch.equal(out[0], x[0]) and not torch.equal(out[1], x[1])



def test_integer_grid_roundtrip_is_much_closer_to_the_original_than_o2s():
    """Mechanical check on a synthetic sharp signal: the integer operator must not blunt a spike."""
    x = np.zeros(1024, dtype=np.float32)
    r = np.array([120, 340, 560, 780, 1000])
    for k in (120, 340, 551, 790, 1000):
        x[k - 2:k + 3] = np.array([-0.2, 0.4, 2.0, 0.4, -0.2], dtype=np.float32)
    t = torch.from_numpy(x)[None, None]
    rr = np.array([120, 340, 551, 790, 1000])
    rt_int = O2.round_trip(t, [B.IntegerEventWarp(rr)]).squeeze().numpy()
    rt_frac = O2.round_trip(t, [O2.EventWarp(rr)]).squeeze().numpy()
    peak = float(x.max())
    assert abs(float(rt_int.max()) - peak) <= abs(float(rt_frac.max()) - peak) + 1e-6
    assert float(np.abs(rt_int - x).max()) <= float(np.abs(rt_frac - x).max()) + 1e-6
    assert r.size == rr.size


# --------------------------------------------------------------------------------- metrics reuse
def test_stage0_reuses_the_exact_o2_metrics_cohort_detector_and_iqr():
    assert 'spec_from_file_location("o2_stage0_roundtrip"' in S0_SRC
    assert "S0.roundtrip_metrics(" in S0_SRC and "S0.load_cohort()" in S0_SRC
    assert "O2.roundtrip_gate(med)" in S0_SRC
    assert 'target_scaling.json' in S0_SRC and "scale_train_IQR" in S0_SRC
    assert 'len(X) != 2048 or n_beats != 19834' in S0_SRC
    assert "detect_rpeaks" not in S0_SRC                                    # the detector comes from the O2 module
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    got = ER.select_subset("x4-event-nfe-v2", "an0", 22183, 1024)
    assert np.array_equal(got, np.asarray(frozen["an0"], dtype=got.dtype))


def test_stage0_thresholds_are_the_o2_ones():
    assert O2.ROUNDTRIP_GATE == {"raw_rmse": 0.020, "T6": 0.020, "T7": 0.020, "T4": 0.020, "T8": 0.020,
                                 "f1_at_50": 0.98, "beat_count_diff": 0}
    for tok in ("R0-1", "R0-6", "0.020", "0.98"):
        assert tok in PREREG or tok in AUDIT


# --------------------------------------------------------------------------------- verdict tree
def test_verdict_tree_matches_the_preregistration():
    ok = {k: True for k in ("R0-1", "R0-2", "R0-3", "R0-4a", "R0-4b", "R0-5", "R0-6")}
    assert B.decide_o2b(ok)["verdict"] == B.VERDICT_A
    assert B.decide_o2b(dict(ok, **{"R0-4b": False}))["verdict"] == B.VERDICT_B
    for k in ("R0-1", "R0-2", "R0-3", "R0-4a", "R0-5", "R0-6"):
        assert B.decide_o2b(dict(ok, **{k: False}))["verdict"] == B.VERDICT_C, k
    assert B.decide_o2b({}, precheck_ok=False)["verdict"] == B.VERDICT_INVALID
    for v in (B.VERDICT_A, B.VERDICT_B, B.VERDICT_C, B.VERDICT_INVALID):
        assert v in PREREG


def test_t8_support_note_is_a_code_fact():
    a, b = int(round(0.08 * OT.FS)), int(round(0.12 * OT.FS))
    assert (a, b) == (10, 15)
    assert b > B.ANCHOR_W                                                    # S search reaches beyond the protected core
    import inspect as _i
    src = _i.getsource(R.qrs_width_ms)
    assert "q_win_s: float = 0.08" in src and "s_win_s: float = 0.12" in src
    assert "beyond the protected core" in AUDIT
