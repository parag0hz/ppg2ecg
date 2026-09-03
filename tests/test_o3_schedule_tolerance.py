"""O3 tests — docs/O3_SCHEDULE_ERROR_TOLERANCE_PREREGISTRATION.md section 27.

O3 trains nothing: every generator and probe is frozen, and no optimizer may be reachable from any O3 source.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import o2_warp as O2
from ppg2ecg.evaluation import o2b_warp as BW
from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.evaluation import rpeaks as RP
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.probes import rhythm_tcn as RTCN

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o3_schedule_tolerance"
MOD_SRC = (ROOT / "src/ppg2ecg/evaluation/o3_schedule.py").read_text()
SRCS = {n: (ROOT / f"scripts/o3_{n}.py").read_text()
        for n in ("common", "synthetic", "r1_bridge", "preflight", "figures")}
SRCS["module"] = MOD_SRC
PREREG = (ROOT / "docs/O3_SCHEDULE_ERROR_TOLERANCE_PREREGISTRATION.md").read_text()
R5 = np.array([120, 340, 551, 790, 1000])
R9 = np.array([60, 170, 281, 395, 505, 618, 730, 845, 960])


def _code(src: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


CODE = {k: _code(v) for k, v in SRCS.items()}


# --------------------------------------------------------------------------------- repository
def test_firewall_pins_a4_and_c2_untouched():
    for sub, pin in (("external/PENGUIN", "6cd70cdefb91f10efeb8dce34019b5067cb25344"),
                     ("external/iMeanFlow", "bf60cd7cb653f6628e59d48034b333c5eba445e2")):
        head = subprocess.run(["git", "-C", str(ROOT / sub), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert head == pin, f"{sub} moved: {head}"
    with pytest.raises(ER.WildPPGTestFirewallError):
        ER.assert_no_test_subjects(("an0", "kjd"))
    for n, s in SRCS.items():
        assert "kjd" not in s and "ssx" not in s, n
    assert hashlib.md5((ROOT / "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt").read_bytes()).hexdigest() == \
        "31c042d291052fbb6dc15263ad316be2"
    assert not list((ROOT / "outputs").glob("*c2*"))
    assert 'ER.assert_no_test_subjects(VAL)' in SRCS["common"]


def test_o3_trains_nothing_and_constructs_no_optimizer():
    for n, code in CODE.items():
        for bad in ("torch.optim", "AdamW", ".backward(", "loss.backward", ".train()", "requires_grad_(True)",
                    "imeanflow_loss", "batch_rounds", "torch.save", "sample_tr_c1", "seed_everything",
                    "DataLoader", "count_params"):
            assert bad not in code, f"{n}: {bad}"
    assert CODE["common"].count("requires_grad_(False)") >= 2
    assert "assert not any(p.requires_grad for p in base.parameters())" in SRCS["common"]
    assert "assert not any(p.requires_grad for p in o2c.parameters())" in SRCS["common"]
    assert "not tcn.training" in SRCS["common"]
    assert '"training_performed": False' in SRCS["synthetic"] and '"training_performed": False' in SRCS["r1_bridge"]
    assert "NO TRAINING" in PREREG


# --------------------------------------------------------------------------------- frozen components
def test_frozen_component_hashes():
    def fsha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert fsha(ROOT / "outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt").startswith("557c7054")
    assert fsha(ROOT / "outputs/o2c_canon_oracle_seed42/checkpoint_final.pt").startswith("5aab09be")
    assert fsha(ROOT / "outputs/r1_global_tcn_seed42/checkpoint_best.pt").startswith("bfe76ea6")
    assert fsha(ROOT / "src/ppg2ecg/evaluation/o2b_warp.py").startswith("cb4d1866")
    assert fsha(ROOT / "src/ppg2ecg/evaluation/o2_warp.py").startswith("046becfb")
    ck = torch.load(ROOT / "outputs/o2c_canon_oracle_seed42/checkpoint_final.pt", map_location="cpu",
                    weights_only=False)
    assert int(ck["step"]) == 10046 and ck["state_dict_sha256"].startswith("f1cc44b3")
    assert RT.EXPECTED_GENERATOR_STATE_SHA.startswith("47d7ccb9")
    assert RT.EXPECTED_RHYTHM_STATE_SHA.startswith("0986a7af")


def test_operator_is_imported_unchanged():
    assert O3.MIN_INT_SPACING is BW.MIN_INT_SPACING == 21
    assert O3.CORE_OFFSET_TOL is BW.CORE_OFFSET_TOL == 1e-6
    assert BW.ANCHOR_W == O2.ANCHOR_W == 10 and O2.MIN_BEATS == 3
    assert O3.NONINF_MARGIN is O2.NONINF_MARGIN == 0.020
    assert O3.F1_EXCESS_MIN is O2.F1_EXCESS_MIN == 0.10
    for n, code in CODE.items():
        for bad in ("def canonical_positions", "def build_int_anchors", "def build_anchors", "class EventWarp",
                    "class IntegerEventWarp", "grid_sample", "ANCHOR_W = 10", "MIN_INT_SPACING = 2",
                    "MIN_INT_SPACING = 21", "CORE_OFFSET_TOL = 1e", "isotonic", "interp1d", "spline", "sinc"):
            assert bad not in code, f"{n}: {bad}"
    assert "def round_half_to_even" not in CODE["module"] and "BW.round_half_to_even" in MOD_SRC


def test_same_supplied_schedule_warps_the_ppg_and_the_inverse():
    src = SRCS["common"]
    assert 'xcan = warp_block(X, warps, "to_canonical", dev)' in src
    assert 'return warp_block(can, warps, "to_raw", dev), can' in src
    assert CODE["common"].count("'to_raw'") == 1 and CODE["common"].count("'to_canonical'") == 1
    for n in ("synthetic", "r1_bridge"):
        assert "o2c_predict(" in SRCS[n], n
        assert "to_canonical" not in SRCS[n] and "to_raw" not in SRCS[n], n


# --------------------------------------------------------------------------------- perturbations
def test_jitter_levels_salt_and_integer_only():
    assert O3.JITTER_LEVELS == (0, 1, 2, 4, 6, 8) and O3.REPS == (0, 1, 2)
    assert O3.JITTER_SALT == "o3-jitter-v1" and O3.MISS_SALT == "o3-miss-v1" and O3.EXTRA_SALT == "o3-extra-v1"
    for j in O3.JITTER_LEVELS:
        s = O3.jitter_schedule(R9, 0, j, "an0", "wrist", 11)
        d = np.asarray(s, np.int64) - R9
        assert np.issubdtype(np.asarray(s).dtype, np.integer)
        assert np.all(np.abs(d) <= j), (j, d)
    assert np.array_equal(O3.jitter_schedule(R9, 0, 0, "an0", "wrist", 11), R9)          # J0 == GT
    assert not np.array_equal(O3.jitter_schedule(R9, 0, 2, "an0", "wrist", 11),
                              O3.jitter_schedule(R9, 1, 2, "an0", "wrist", 11))          # reps differ
    assert not np.array_equal(O3.jitter_schedule(R9, 0, 2, "an0", "wrist", 11),
                              O3.jitter_schedule(R9, 0, 2, "an0", "head", 11))           # site enters the salt


def test_perturbations_are_deterministic_and_use_no_global_rng():
    for _ in range(3):
        assert np.array_equal(O3.jitter_schedule(R9, 2, 6, "k2s", "ankle", 5),
                              O3.jitter_schedule(R9, 2, 6, "k2s", "ankle", 5))
        assert np.array_equal(O3.miss_schedule(R9, 1, 2, "k2s", "ankle", 5),
                              O3.miss_schedule(R9, 1, 2, "k2s", "ankle", 5))
        assert np.array_equal(O3.extra_schedule(R9, 1, 2, "k2s", "ankle", 5),
                              O3.extra_schedule(R9, 1, 2, "k2s", "ankle", 5))
    np.random.seed(7); a = O3.jitter_schedule(R9, 0, 4, "an0", "head", 3)
    np.random.seed(99); b = O3.jitter_schedule(R9, 0, 4, "an0", "head", 3)
    np.testing.assert_array_equal(a, b)
    assert "np.random" not in _code(MOD_SRC) and "default_rng" not in _code(MOD_SRC)
    assert "hashlib.sha256" in MOD_SRC and 'digest()[:8]' in MOD_SRC


def test_no_jitter_repair_and_precheck_is_a_hard_stop():
    for bad in ("repair", "isotonic", "np.clip(np.diff", "drop_window", "merge", "reorder", "np.sort(s"):
        assert bad not in _code(MOD_SRC), bad
    dense = np.array([100, 118, 140])                        # spacing 18 < 21
    pc = O3.precheck_schedule(dense)
    assert not pc["passed"] and not pc["checks"]["spacing_ok"]
    assert "PERTURBATION DESIGN INVALID" in SRCS["synthetic"]
    assert O3.VERDICT_JITTER_INVALID in PREREG and O3.VERDICT_EXTRA_INVALID in PREREG
    assert "raise RuntimeError(f\"{v} (STOP)" in SRCS["synthetic"]


def test_miss_counts_and_never_deletes_first_or_last():
    for rep in O3.REPS:
        for n in (1, 2):
            s = O3.miss_schedule(R9, rep, n, "an0", "sternum", 4)
            assert s.size == R9.size - n
            assert s[0] == R9[0] and s[-1] == R9[-1]
            assert set(s.tolist()) < set(R9.tolist())
    assert np.array_equal(O3.miss_schedule(R9, 0, 0, "an0", "sternum", 4), R9)
    order = O3.miss_interior_order(R9, 0, "an0", "sternum", 4)
    assert set(order.tolist()) == set(range(1, R9.size - 1))
    assert np.array_equal(order, O3.miss_interior_order(R9, 0, "an0", "sternum", 4))


def test_extra_midpoint_rule_and_eligibility():
    assert O3.extra_midpoint(100, 200) == 150
    assert O3.extra_midpoint(100, 101) == 100                      # 100.5 -> banker's rounding to 100
    assert O3.extra_midpoint(101, 102) == 102                      # 101.5 -> 102
    tight = np.array([0, 40, 200])                                 # first interval too short (both halves < 21)
    assert 0 not in O3.extra_eligible(tight).tolist()
    assert 1 in O3.extra_eligible(tight).tolist()
    for n in (1, 2):
        s = O3.extra_schedule(R9, 0, n, "an0", "head", 9)
        assert s.size == R9.size + n and np.all(np.diff(s) >= O3.MIN_INT_SPACING)
        assert set(R9.tolist()) < set(s.tolist())
    with pytest.raises(ValueError):
        O3.extra_schedule(np.array([0, 30, 60]), 0, 2, "an0", "head", 9)


def test_selection_never_depends_on_a_result():
    for n, code in CODE.items():
        for bad in ("f1_excess >", "if f1", "sort(key=lambda r: r['f1", "best_rep", "argmax(f1", "choose_best"):
            assert bad not in code, f"{n}: {bad}"
    assert "_u64(f\"{base}|{i}\")" in MOD_SRC                       # rank depends only on the salt and the index


# --------------------------------------------------------------------------------- schedule invariants
def test_supplied_schedules_are_sorted_unique_integer_and_in_bounds():
    for fam, lv in (("JITTER", 8), ("MISS", 2), ("EXTRA", 2)):
        for rep in O3.REPS:
            s = O3.supplied_schedule(fam, lv, rep, R9, "k2s", "wrist", 2)
            assert np.issubdtype(s.dtype, np.integer)
            assert np.all(np.diff(s) > 0) and s.size == np.unique(s).size
            assert s.min() >= 0 and s.max() <= O3.T_LEN - 1
            assert np.all(np.diff(s) >= O3.MIN_INT_SPACING)
            pc = O3.precheck_schedule(s)
            assert pc["passed"] and not pc["identity"], (fam, lv, rep, pc)


def test_no_gt_correction_after_the_schedule_is_created():
    for n in ("synthetic", "r1_bridge"):
        code = CODE[n]
        for bad in ("gt_pk[i] +", "snap", "align_to_gt", "offset_correction", "np.roll", "shift_pred",
                    "replace_peaks", "smooth_schedule"):
            assert bad not in code, f"{n}: {bad}"
    assert "GT R is NOT substituted back" in PREREG or "never substituted back" in PREREG
    # the R1 arm builds its warp from S only
    assert "warps = C.build_warps(S)" in SRCS["r1_bridge"]
    assert "gt_fallback_used" in SRCS["r1_bridge"] and '"gt_fallback_used": False' in SRCS["r1_bridge"]


# --------------------------------------------------------------------------------- evaluation
def test_cohort_nfe_and_source_seed_are_the_frozen_ones():
    assert O3.NFE == 4 and O3.SRC_SEED == 0
    assert "len(X) != 2048 or n_beats != 19834" in SRCS["common"]
    assert "!= 1922" in SRCS["common"]
    e0 = torch.randn(2048, 1, 1024, generator=torch.Generator().manual_seed(0))
    assert hashlib.sha256(e0.numpy().tobytes()).hexdigest() == \
        "868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f"
    assert "ER.UNIFORM[NFE]" in SRCS["common"] and "assert got == {NFE}" in SRCS["common"]


def test_frozen_regression_constants_match_the_o2c_result():
    import importlib.util
    sp = importlib.util.spec_from_file_location("o3_common_t", ROOT / "scripts/o3_common.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    assert abs(m.FROZEN_B["f1_excess"] - 0.3175618683270061) < 1e-12
    assert abs(m.FROZEN_O2C["f1_excess"] - 0.8592510052638713) < 1e-12
    for k, v in (("nAE_T4", 0.40715406781869296), ("nAE_T6", 0.40190969840528823),
                 ("nAE_T7", 0.41699228519117515), ("nAE_T8", 0.41387939453125)):
        assert abs(m.FROZEN_O2C[k] - v) < 1e-12, k
    assert m.REG_TOL == 1e-6
    o2c = json.loads((ROOT / "artifacts/o2c_oracle_integer_grid/decision.json").read_text())
    assert abs(o2c["J_detail"]["f1_excess"]["O2C"] - m.FROZEN_O2C["f1_excess"]) < 1e-12
    assert O3.VERDICT_REGRESSION in PREREG


def test_o1_targets_and_train_iqrs_are_the_frozen_ones():
    scal = json.loads((ROOT / "artifacts/o1_component_extractability/target_scaling.json").read_text())["targets"]
    for t, v in (("median_QRS_p2p", 0.5053170621395111), ("median_QRS_max_abs_derivative", 0.22995377704501152),
                 ("median_QRS_curvature_energy", 0.033796727107052275), ("median_QRS_width_ms", 31.25)):
        assert scal[t]["scale_train_IQR"] == v and t in O3.ALIGNED
    assert OT.TARGET_IDS["median_QRS_p2p"] == "T4" and OT.TARGET_IDS["median_QRS_width_ms"] == "T8"
    y = np.zeros(1024); y[500] = 1.0; y[499] = 0.4; y[501] = 0.4
    bp = O3.beat_primitives(y, 500)
    assert bp is not None and set(bp) == set(O3.ALIGNED)
    assert O3.beat_primitives(y, 3) is None                            # core must fit, exactly as in O1


def test_bootstrap_is_ecg_window_clustered_with_the_frozen_o3_seed():
    assert O3.BOOT_N == 2000 and O3.BOOT_SEED == 20260904
    assert 'cluster = np.array([f"{a}|{b}" for a, b in zip(SUB, WI)])' in SRCS["common"]
    assert "n_boot=O3.BOOT_N, seed=O3.BOOT_SEED" in SRCS["common"]
    assert "20260904" in PREREG


def test_three_replicates_are_not_pooled():
    assert O3.level_survives([{"survives": True}] * 3)
    assert not O3.level_survives([{"survives": True}] * 2)              # fewer than three reps never passes
    assert not O3.level_survives([{"survives": True}, {"survives": True}, {"survives": False}])
    assert "O3.level_survives(" in SRCS["synthetic"]
    assert CODE["synthetic"].count("C.paired_boot(rows_b, al_b, rows_c, al_c, SUB, CLUSTER)") == 1
    assert "for name, fam, lv, rep in CONDS" in SRCS["synthetic"]
    assert "not pooled" in PREREG or "not** pooled" in PREREG


def test_retention_is_unclipped():
    assert O3.retention(0.5, 0.0, 1.0, True) == 0.5
    assert O3.retention(-0.5, 0.0, 1.0, True) == -0.5                   # negative values are reported as-is
    assert O3.retention(2.0, 0.0, 1.0, True) == 2.0                     # > 1 values are reported as-is
    assert O3.retention(0.25, 1.0, 0.0, False) == 0.75
    for bad in ("np.clip(ret", "clip(0", "min(1.0", "max(0.0"):
        assert bad not in _code(MOD_SRC), bad


# --------------------------------------------------------------------------------- R1
def test_r1_uses_the_exact_frozen_operating_point():
    assert O3.R1_THRESHOLD == 0.35
    assert RTCN.REFRACTORY_SAMPLES == 32 and RTCN.REFRACTORY_MS == 250.0
    assert "extract_events(prob[j], threshold)" in SRCS["common"]
    assert "threshold: float = O3.R1_THRESHOLD" in SRCS["common"]
    assert "torch.sigmoid(tcn(pp))" in SRCS["common"]
    for bad in ("thresh_grid", "THRESH_GRID", "for th in", "best_threshold", "site_delay", "delay_ms",
                "phase_correct", "smooth", "median_filter"):
        assert bad not in CODE["common"], bad
        assert bad not in CODE["r1_bridge"], bad
    p = np.zeros(1024); p[100] = 0.9; p[110] = 0.8; p[400] = 0.4; p[500] = 0.30
    ev = RTCN.extract_events(p, 0.35)
    assert ev.tolist() == [100, 400]        # 110 suppressed by the 32-sample refractory, 0.30 below threshold


def test_r1_runs_only_after_the_synthetic_curve_is_frozen():
    src = SRCS["r1_bridge"]
    assert 'if not FROZEN.exists():' in src and "stage A must complete first (STOP)" in src
    assert "synthetic_curve_frozen.json" in src
    assert 'bad = {f: [h, C.fsha(ART / f)] for f, h in frozen["artifact_sha256"].items()' in src
    assert "synthetic artifacts changed after the freeze (STOP)" in src
    i_frozen = src.index("FROZEN.exists()")
    assert i_frozen < src.index("C.r1_schedules(")
    assert "R1 BRIDGE FAILS PRECHECK" in src and "R1 BRIDGE FAILS PRECHECK" in PREREG


# --------------------------------------------------------------------------------- gates and verdict
def test_joint_gates_use_the_unchanged_margins():
    def r(point, lo, hi):
        v = "improves" if lo > 0 else ("worsens" if hi < 0 else "unresolved")
        return {"point": point, "lo": lo, "hi": hi, "verdict": v}
    good = {"f1_excess": r(0.5, 0.4, 0.6), "nAE_T4": r(0.2, 0.1, 0.3), "nAE_T6": r(0.2, 0.1, 0.3),
            "nAE_T7": r(0.2, 0.1, 0.3), "nAE_T8": r(0.2, 0.1, 0.3), "qrs_deriv_rmse": r(0.1, 0.05, 0.15),
            "qrs_curvature_err": r(0.1, 0.05, 0.15)}
    assert O3.joint_gates(good)["survives"]
    assert not O3.joint_gates({**good, "f1_excess": r(0.05, 0.01, 0.09)})["G1"]        # point below +0.10
    assert not O3.joint_gates({**good, "f1_excess": r(0.5, -0.1, 0.9)})["G1"]          # CI touches 0
    assert not O3.joint_gates({**good, "nAE_T6": r(-0.05, -0.09, -0.01)})["G2"]        # below the -0.020 margin
    assert O3.joint_gates({**good, "nAE_T6": r(-0.005, -0.015, 0.005)})["G2"]          # inside the margin
    assert not O3.joint_gates({**good, "nAE_T6": r(0.0, -0.01, 0.01), "nAE_T7": r(0.0, -0.01, 0.01)})["G4"]
    assert not O3.joint_gates({**good, "qrs_deriv_rmse": r(-0.2, -0.3, -0.1)})["G5"]
    assert not O3.joint_gates({**good, "nAE_T8": r(-0.2, -0.3, -0.1)})["G6"]
    assert O3.F1_EXCESS_MIN == 0.10 and O3.NONINF_MARGIN == 0.020


def test_bridge_gate_and_verdict_tree():
    g = {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True, "G6": True, "survives": True}
    pos, neg = {"lo": 0.1}, {"lo": -0.1}
    assert O3.bridge_gate(g, pos, pos)["supported"]
    assert not O3.bridge_gate(g, pos, neg)["supported"]
    assert not O3.bridge_gate({**g, "survives": False}, pos, pos)["supported"]
    lp_all = {0: True, 1: True, 2: True, 4: True, 6: True, 8: True}
    assert O3.decide_o3(lp_all, True, True)["verdict"] == O3.VERDICT_A
    assert O3.decide_o3(lp_all, False, True)["verdict"] == O3.VERDICT_B
    assert O3.decide_o3({0: True, 1: True, 2: True, 4: False, 6: False, 8: False}, False, True)["verdict"] == O3.VERDICT_C
    assert O3.decide_o3({0: True, 1: True, 2: False, 4: False, 6: False, 8: False}, False, True)["verdict"] == O3.VERDICT_D
    assert O3.decide_o3(lp_all, True, False)["verdict"] == O3.VERDICT_B       # invalid R1 precheck cannot give A
    for v in (O3.VERDICT_A, O3.VERDICT_B, O3.VERDICT_C, O3.VERDICT_D, O3.VERDICT_REGRESSION, O3.VERDICT_BRITTLE):
        assert v in PREREG, v
    assert O3.j_max({0: True, 1: True, 2: True, 4: False, 6: False, 8: False}) == 2
    assert O3.j_max({j: False for j in O3.JITTER_LEVELS}) is None


def test_miss_and_extra_cannot_select_the_global_verdict():
    fn = next(n for n in ast.walk(ast.parse(MOD_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == "decide_o3")
    fn = ast.parse(ast.unparse(fn)).body[0]
    if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]                                    # the docstring is prose, not logic
    tree = ast.unparse(fn).replace("miss_extra_can_select_verdict", "")
    for bad in ("miss", "extra", "MISS", "EXTRA"):
        assert bad not in tree, bad
    assert O3.decide_o3({0: True, 1: True, 2: True, 4: True, 6: True, 8: True}, False, True)["verdict"] == O3.VERDICT_B
    assert "MISS and EXTRA never select the global verdict" in PREREG
    assert '"miss_extra_can_select_verdict": False' in MOD_SRC


def test_shape_only_diagnostic_cannot_enter_the_gates():
    assert "shape_only" not in CODE["module"].split("def joint_gates")[1].split("def bridge_gate")[0]
    assert "SHAPE-ONLY" in PREREG and "cannot enter G1-G6" in PREREG.replace("–", "-")
    for k in ("G1", "G2", "G3", "G4", "G5", "G6"):
        assert k not in _code(SRCS["common"]).split("def shape_only")[1]
    pairs = O3.retained_pairs("EXTRA", 1, 0, R9, O3.extra_schedule(R9, 0, 1, "an0", "head", 9), "an0", "head", 9)
    assert pairs.shape == (R9.size, 2)                 # inserted beats are excluded, originals all retained
    m = O3.retained_pairs("MISS", 2, 0, R9, O3.miss_schedule(R9, 0, 2, "an0", "head", 9), "an0", "head", 9)
    assert m.shape == (R9.size - 2, 2)


def test_claim_boundaries_are_stated():
    flat = " ".join(PREREG.split())
    for banned in ("exact R timing is observable from PPG", "phase is solved", "deployability established",
                   "calibrated uncertainty", "clinical validity", "causal bottleneck proof",
                   "information-theoretic limit"):
        assert banned in flat, banned
    assert "confidence-calibrated" not in PREREG
    assert "ORACLE DIAGNOSTIC" in PREREG and "supervised with ECG R labels" in flat


# --------------------------------------------------------------------------------- review-driven regressions
def test_operator_floor_label_follows_the_measured_floor_for_every_family():
    src = SRCS["synthetic"]
    assert '"floor_exceeds_0.020": bool(fmed["T6"] > 0.020 or fmed["T7"] > 0.020)' in src
    assert '"OPERATOR-CONFOUNDED" if (lv > 0 and (fmed["T6"] > 0.020 or fmed["T7"] > 0.020))' in src
    assert 'fam in ("MISS", "EXTRA") and lv > 0 and' not in src        # the family guard must not gate the label
    assert '"stops_run": bool(fam == "JITTER" and lv == 1' in src      # only J1 stops the run, per section 10
    assert "OPERATOR-CONFOUNDED" in PREREG


def test_sweep_requires_a_passing_runtime_preflight():
    src = SRCS["synthetic"]
    assert 'pf = ART / "runtime_preflight.json"' in src and 'if pfj.get("stop"):' in src
    assert src.index('pf = ART / "runtime_preflight.json"') < src.index("15-17. generator sweep")
    assert "BUDGET_GPU_HOURS" in SRCS["preflight"] and O3.BUDGET_GPU_HOURS == 2.0
    assert "BOTH stage-B arms" in SRCS["preflight"]                    # both 8-source arms are charged
    assert "2.0 GPU-hour" in PREREG or "2.0 GPU-hours" in PREREG


def test_precheck_is_written_before_the_first_generator_call():
    src = SRCS["synthetic"]
    i_pc = src.index('C.wcsv(ART / "schedule_precheck.csv"')
    assert i_pc < src.index("C.R2E.gen_plain(base, X, e0")
    assert "section 8: before ANY generator call" in src
    assert "the GT (ORACLE) schedule fails the section 8 precheck (STOP)" in src


def test_feasibility_shortfall_takes_the_preregistered_verdict():
    src = SRCS["synthetic"]
    assert "except ValueError as exc:" in src and "O3.VERDICT_EXTRA_INVALID if fam ==" in src
    assert '(ART / "decision.json").write_text' in src.split("except ValueError as exc:")[1][:600]


def test_paired_bootstrap_columns_use_the_equal_subject_macro():
    assert '"B": macro(a, SUB), "arm": macro(b, SUB)' in SRCS["common"]
    assert "np.nanmean(a)" not in CODE["common"]


def test_shape_only_separates_match_rate_from_primitive_usability():
    src = SRCS["common"]
    assert '"primitive_usable_beats"' in src and '"primitive_usable_fraction"' in src
    assert 'u["match"] += int(len(m))' in src
    y = np.zeros((2, 1024), dtype=np.float32)
    for i in range(2):
        for c in (200, 400, 600, 800):
            y[i, c - 1:c + 2] = [0.4, 1.0, 0.4]
    sup = [np.array([200, 400, 600, 800], np.int64)] * 2
    gt = [np.array([200, 400, 600, 800], np.int64)] * 2
    ret = [np.stack([np.arange(4), np.arange(4)], axis=1)] * 2
    import importlib.util
    sp = importlib.util.spec_from_file_location("o3_common_s", ROOT / "scripts/o3_common.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    out = m.shape_only(y, y.astype(np.float64), sup, gt, ret, {t: 1.0 for t in O3.ALIGNED},
                       np.array(["an0", "k2s"]))
    assert out["retained_beats"] == 8 and out["matched_beats"] <= 8
    assert 0.0 <= out["matched_coverage"] <= 1.0
    assert out["primitive_usable_beats"] <= out["matched_beats"]


def test_stage_b_append_is_idempotent(tmp_path):
    import importlib.util
    sp = importlib.util.spec_from_file_location("o3_r1_t", ROOT / "scripts/o3_r1_bridge.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    p = tmp_path / "rows.csv"
    m.C.wcsv(p, [{"stage": "synthetic", "condition": "JITTER_4", "v": 1}])
    row = [{"stage": "R1", "condition": "O2C-R1-SCHEDULE", "v": 2}]
    m._append(p, row)
    m._append(p, row)
    import csv as _csv
    rows = list(_csv.DictReader(open(p)))
    assert len(rows) == 2 and sum(r["stage"] == "R1" for r in rows) == 1


def test_j_max_quality_is_not_faked_when_nothing_survives():
    assert 'q_at = None if jm is None else' in SRCS["synthetic"]
    assert O3.j_max({j: False for j in O3.JITTER_LEVELS}) is None
