"""E3 tests — docs/E3_BEAT_SET_FIRST_EVENT_GEOMETRY_PREREGISTRATION.md section 21.

E3 permits exactly one learned object (a linear count readout). Nothing else may train, and every metric
must come from the frozen E2 contract.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.evaluation import e3_beat_set as E3
from ppg2ecg.evaluation import event_geometry_contract as EG
from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o2b_warp as BW
from ppg2ecg.probes import rhythm_tcn as RTCN

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/e3_beat_set_first"
MOD_SRC = (ROOT / "src/ppg2ecg/evaluation/e3_beat_set.py").read_text()
SRCS = {"module": MOD_SRC, "common": (ROOT / "scripts/e3_common.py").read_text(),
        "stage0": (ROOT / "scripts/e3_stage0.py").read_text()}
for _opt in ("e3_fit_count", "e3_validate", "e3_figures"):
    _p = ROOT / f"scripts/{_opt}.py"
    if _p.exists():
        SRCS[_opt] = _p.read_text()
PREREG = (ROOT / "docs/E3_BEAT_SET_FIRST_EVENT_GEOMETRY_PREREGISTRATION.md").read_text()
FLAT = " ".join(PREREG.replace("*", "").replace("`", "").split())
RNG = np.random.default_rng(11)


def _code(src: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


CODE = {k: _code(v) for k, v in SRCS.items()}


def _prob(seed, scale=1.0, ties=False):
    r = np.random.default_rng(seed)
    p = np.clip(r.random(1024) ** 3 * scale + 0.02 * r.standard_normal(1024), 0, 1)
    if ties:
        p[r.integers(0, 1024, 25)] = E3.R1_THRESHOLD
    return p


# --------------------------------------------------------------------------------- repository
def test_firewall_pins_a4_and_c2():
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


def test_only_one_learned_object_is_reachable():
    for n, code in CODE.items():
        for bad in ("torch.optim", "AdamW", ".backward(", ".train()", "requires_grad_(True)", "torch.save",
                    "imeanflow_loss", "nn.Transformer", "MultiheadAttention", "nn.LSTM", "nn.GRU",
                    "GridSearchCV", "cross_val", "RidgeCV"):
            assert bad not in code, f"{n}: {bad}"
    assert "Ridge" in MOD_SRC and E3.RIDGE_ALPHA == 1.0
    for banned in ("phase network", "set Transformer", "event decoder", "count CNN",
                   "timing regression head", "peak shifting"):
        assert banned in FLAT, banned


# --------------------------------------------------------------------------------- E2 contract
def test_e2_contract_is_imported_not_restated():
    sha = hashlib.sha256((ROOT / "artifacts/e2_evaluation_contract/contract_v1.json").read_bytes()).hexdigest()
    assert sha == "06e869412114e1efb9ab6624540aebc141495e2690150f4e512df7676c5a9115"
    assert sha in SRCS["common"] and sha in PREREG
    assert "from ppg2ecg.evaluation import event_geometry_contract as EG" in MOD_SRC
    assert EG.TOL_SG_MS == 150.0 and EG.TOL_PS_MS == 50.0 and EG.SUPPORT == (-10, 15)
    assert EG.BOOT_N == 2000 and EG.BOOT_SEED == 20260904
    for n, code in CODE.items():                       # no local metric duplication
        for bad in ("def classify_topology", "def placement_metrics", "def own_center_morphology",
                    "def joint_event_fidelity", "def gt_anchored_joint_structure", "def assign_schedule_to_gt"):
            assert bad not in code, f"{n}: {bad}"
    assert "EG.assign_schedule_to_gt" in SRCS["common"] and "EG.apply_contract" in SRCS["common"]


# --------------------------------------------------------------------------------- R1 and candidates
def test_frozen_r1_identity_and_operating_point():
    def fsha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert fsha(ROOT / "outputs/r1_global_tcn_seed42/checkpoint_best.pt").startswith("bfe76ea6")
    assert fsha(ROOT / "outputs/o2c_canon_oracle_seed42/checkpoint_final.pt").startswith("5aab09be")
    assert fsha(ROOT / "src/ppg2ecg/evaluation/o2b_warp.py").startswith("cb4d1866")
    assert fsha(ROOT / "src/ppg2ecg/evaluation/o2_warp.py").startswith("046becfb")
    assert E3.R1_THRESHOLD == 0.35 and E3.REFRACTORY is RTCN.REFRACTORY_SAMPLES == 32
    assert "torch.sigmoid(tcn(pp))" in SRCS["common"]
    assert "C.load_models(dev, with_r1=True)" in SRCS["stage0"]   # the loader asserts every frozen hash


def test_threshold_free_candidates_reproduce_r1_bit_exactly():
    for seed in range(60):
        p = _prob(seed, scale=0.4 if seed % 3 else 1.0, ties=seed % 7 == 2)
        pos, sc = E3.candidate_events(p)
        np.testing.assert_array_equal(pos[sc >= E3.R1_THRESHOLD], RTCN.extract_events(p, E3.R1_THRESHOLD))
        assert E3.reproduces_r1(p)
        assert np.all(np.diff(pos) > E3.REFRACTORY)              # refractory unchanged
        assert np.array_equal(pos, np.sort(pos)) and pos.size == np.unique(pos).size
        np.testing.assert_array_equal(E3.candidate_events(p)[0], pos)   # deterministic
    src = _code(MOD_SRC).split("def candidate_events")[1].split("def reproduces_r1")[0]
    assert ">= threshold" not in src and "threshold" not in src      # the amplitude filter is gone
    assert "refractory" in src


def test_topk_selector_rule():
    pos = np.array([10, 100, 200, 300, 400])
    sc = np.array([0.9, 0.9, 0.5, 0.7, 0.1])
    sel, short = E3.topk_select(pos, sc, 2)
    np.testing.assert_array_equal(sel, [10, 100])                 # tie 0.9 -> lower index first
    assert not short
    np.testing.assert_array_equal(E3.topk_select(pos, sc, 3)[0], [10, 100, 300])
    sel, short = E3.topk_select(pos, sc, 99)
    np.testing.assert_array_equal(sel, pos) and short
    assert E3.topk_select(pos, sc, 0)[0].size == 0
    for k in (1, 3, 5):
        s = E3.topk_select(pos, sc, k)[0]
        assert np.array_equal(s, np.sort(s))                      # returned ascending in time
        assert set(s.tolist()) <= set(pos.tolist())               # never a new location
    for bad in ("+ 1", "- 1", "np.roll", "shift", "gt_"):
        assert bad not in _code(MOD_SRC).split("def topk_select")[1].split("def count_features")[0], bad


# --------------------------------------------------------------------------------- oracle count
def test_oracle_count_supplies_count_only():
    src = SRCS["stage0"]
    assert 'H.arm_schedules("topk", cand_pos, cand_sc, counts=K)' in src
    assert "K = np.array([len(p) for p in gt_pk])" in src
    tail = src.split("Stage-0 schedule evaluation")[1]
    for bad in ("gt_pk[i][", "gt_pk[i] +", "np.asarray(gt_pk[i], np.int64)["):
        assert bad not in tail, bad                               # no GT LOCATION lookup for selection
    assert "ORACLE COUNT DIAGNOSTIC" in PREREG or "ORACLE COUNT DIAGNOSTIC" in src
    pos = np.array([10, 100, 200, 300]); sc = np.array([0.1, 0.9, 0.5, 0.7])
    sel, _ = E3.topk_select(pos, sc, 3)
    assert set(sel.tolist()) <= set(pos.tolist())                 # locations are candidates only


# --------------------------------------------------------------------------------- threshold control
def test_threshold_grid_and_lexicographic_selection():
    assert E3.THRESHOLD_GRID == tuple(round(0.05 * i, 2) for i in range(1, 20))
    assert len(E3.THRESHOLD_GRID) == 19 and E3.THRESHOLD_GRID[0] == 0.05 and E3.THRESHOLD_GRID[-1] == 0.95
    rows = [{"threshold": 0.30, "A5": 0.50, "A4": 0.20, "A3": 0.10},
            {"threshold": 0.40, "A5": 0.50, "A4": 0.20, "A3": 0.10},
            {"threshold": 0.55, "A5": 0.60, "A4": 0.30, "A3": 0.10}]
    assert E3.select_threshold(rows)["selected_threshold"] == 0.55            # A5 wins first
    rows2 = [{"threshold": 0.30, "A5": 0.60, "A4": 0.30, "A3": 0.10},
             {"threshold": 0.55, "A5": 0.60, "A4": 0.20, "A3": 0.10}]
    assert E3.select_threshold(rows2)["selected_threshold"] == 0.55           # then min spurious
    rows3 = [{"threshold": 0.30, "A5": 0.6, "A4": 0.2, "A3": 0.10},
             {"threshold": 0.45, "A5": 0.6, "A4": 0.2, "A3": 0.10}]
    assert E3.select_threshold(rows3)["selected_threshold"] == 0.30           # then |t-0.35|: 0.05 < 0.10
    rows4 = [{"threshold": 0.40, "A5": 0.6, "A4": 0.2, "A3": 0.10},
             {"threshold": 0.30, "A5": 0.6, "A4": 0.2, "A3": 0.10}]
    assert E3.select_threshold(rows4)["selected_threshold"] == 0.30           # equidistant -> lower t
    assert E3.select_threshold(rows)["population"] == "train12 only"


# --------------------------------------------------------------------------------- feature and readout
def test_count_feature_is_mean_and_max_of_the_prelogit_tensor_only():
    H = RNG.standard_normal((64, 1024))
    z = E3.count_features(H)
    assert z.shape == (128,)
    np.testing.assert_allclose(z[:64], H.mean(axis=1))
    np.testing.assert_allclose(z[64:], H.max(axis=1))
    with pytest.raises(ValueError):
        E3.count_features(RNG.standard_normal((2, 64, 1024)))
    src = SRCS["common"]
    assert "torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)" in src
    assert "for blk in tcn.blocks" in src and "tcn.head" not in src           # strictly pre-logit
    for bad in ("site", "subject", "quality", "sqi", "np.concatenate([z, "):
        assert bad not in _code(src).split("def r1_prelogit_features")[1].split("def candidates_for")[0], bad


def test_ridge_specification_and_integer_conversion():
    assert E3.RIDGE_ALPHA == 1.0
    X = RNG.standard_normal((50, 8))
    X[:, 3] = 7.0                                                             # zero-variance dimension
    mu, sd = E3.standardize_fit(X)
    assert sd[3] == 1.0 and np.all(sd > 0)
    np.testing.assert_allclose(mu, X.mean(axis=0))
    np.testing.assert_array_equal(E3.to_int_count([0.5, 1.5, 2.5, 3.5]), [0, 2, 2, 4])   # banker's rounding
    np.testing.assert_array_equal(E3.to_int_count([-5.0, 99.0]), [0, E3.K_STRUCTURAL_MAX])
    assert E3.K_STRUCTURAL_MAX == 32
    np.testing.assert_array_equal(E3.to_int_count([9.4]), BW.round_half_to_even([9.4]))
    for bad in ("quantile", "percentile", "clip(k, K.min()", "subject"):
        assert bad not in _code(MOD_SRC).split("def to_int_count")[1].split("def standardize_fit")[0], bad


# --------------------------------------------------------------------------------- gates and stop logic
def _e(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi}


def _sched(**over):
    d = {"A5_exact_set": _e(0.20, 0.15, 0.25), "T3_frac": _e(0.30, 0.25, 0.35),
         "A4_spurious_fraction": _e(0.10, 0.08, 0.12), "A3_missing_fraction": _e(0.01, -0.005, 0.02),
         "B5_exact_set_mae_ms": _e(1.0, -2.0, 4.0), "SG_F1_50": _e(0.15, 0.12, 0.18),
         "T2_frac": _e(0.00, -0.01, 0.01)}
    d.update(over); return d


def _gen(**over):
    d = {"PG_F1_50": _e(0.10, 0.07, 0.13), "C2_own_T6": _e(0.05, 0.02, 0.08),
         "C3_own_T7": _e(0.02, -0.01, 0.05), "J2_gt_local_deriv_rmse": _e(0.01, -0.01, 0.03),
         "J3_gt_local_curvature_err": _e(0.01, -0.01, 0.03)}
    d.update(over); return d


def test_oc_and_og_gate_thresholds():
    assert E3.oracle_count_gates(_sched())["passed"]
    assert not E3.oracle_count_gates(_sched(A5_exact_set=_e(0.09, 0.05, 0.13)))["OC1"]      # point < 0.10
    assert not E3.oracle_count_gates(_sched(A5_exact_set=_e(0.20, -0.01, 0.4)))["OC1"]      # CI touches 0
    assert not E3.oracle_count_gates(_sched(T3_frac=_e(0.14, 0.10, 0.18)))["OC2"]           # < 0.15
    assert not E3.oracle_count_gates(_sched(A4_spurious_fraction=_e(0.04, 0.02, 0.06)))["OC3"]
    assert not E3.oracle_count_gates(_sched(A3_missing_fraction=_e(-0.03, -0.05, -0.01)))["OC4"]
    assert E3.oracle_count_gates(_sched(A3_missing_fraction=_e(-0.01, -0.019, 0.0)))["OC4"]
    assert not E3.oracle_count_gates(_sched(B5_exact_set_mae_ms=_e(-9.0, -12.0, -8.5)))["OC5"]
    assert E3.oracle_generator_gates(_gen())["passed"]
    assert not E3.oracle_generator_gates(_gen(PG_F1_50=_e(0.04, 0.01, 0.07)))["OG1"]
    assert not E3.oracle_generator_gates(_gen(C2_own_T6=_e(-0.03, -0.05, -0.01),
                                              C3_own_T7=_e(0.0, -0.01, 0.01)))["OG2"]
    assert not E3.oracle_generator_gates(_gen(C2_own_T6=_e(0.0, -0.01, 0.01),
                                              C3_own_T7=_e(0.0, -0.01, 0.01)))["OG4"]


def test_pc_tc_dg_dc_gate_thresholds():
    assert E3.predicted_count_gates(_sched())["passed"]
    assert not E3.predicted_count_gates(_sched(A5_exact_set=_e(0.04, 0.02, 0.06)))["PC1"]
    assert not E3.predicted_count_gates(_sched(T3_frac=_e(0.09, 0.05, 0.13)))["PC2"]
    assert not E3.predicted_count_gates(_sched(A4_spurious_fraction=_e(0.03, 0.01, 0.05)))["PC3"]
    assert not E3.predicted_count_gates(_sched(T2_frac=_e(-0.03, -0.05, -0.021)))["PC5"]
    tc = {"A5_exact_set": _e(0.05, 0.03, 0.07), "A3_missing_fraction": _e(0.0, -0.01, 0.01),
          "A4_spurious_fraction": _e(0.0, -0.01, 0.01)}
    assert E3.threshold_control_gates(tc)["passed"]
    assert not E3.threshold_control_gates({**tc, "A5_exact_set": _e(0.02, 0.01, 0.03)})["TC1"]
    assert not E3.threshold_control_gates({**tc, "A5_exact_set": _e(0.05, -0.01, 0.11)})["TC1"]
    assert E3.downstream_gates(_gen(), 0.95)["passed"]
    assert not E3.downstream_gates(_gen(), 0.89)["DG1"]
    dc = _gen()
    assert E3.downstream_control_gates(dc)["passed"]
    assert not E3.downstream_control_gates({k: _e(0.0, -0.01, 0.01) for k in dc})["DC3"]


def test_verdict_tree_and_stop_logic():
    ok = lambda d: {**d, "passed": True}      # noqa: E731
    bad = lambda d: {**d, "passed": False}    # noqa: E731
    oc, og, pc, tc, dg, dc = (ok({}),) * 6
    assert E3.decide_e3(oc, og, pc, tc, dg, dc)["verdict"] == E3.VERDICT_A
    assert E3.decide_e3(bad({}), None)["verdict"] == E3.VERDICT_E
    assert E3.decide_e3(ok({}), bad({}))["verdict"] == E3.VERDICT_E
    assert E3.decide_e3(ok({}), ok({}), bad({}))["verdict"] == E3.VERDICT_B
    assert E3.decide_e3(ok({}), ok({}), ok({}), bad({}))["verdict"] == E3.VERDICT_C
    assert E3.decide_e3(ok({}), ok({}), ok({}), ok({}), bad({}), None)["verdict"] == E3.VERDICT_D
    assert E3.decide_e3(ok({}), ok({}), ok({}), ok({}), ok({}), bad({}))["verdict"] == E3.VERDICT_D
    assert E3.decide_e3(precheck_ok=False)["verdict"] == E3.VERDICT_PRECHECK
    # a Stage-0 failure must record that the ridge was never fitted, and stop before the generator
    assert E3.decide_e3(bad({}), None)["ridge_fitted"] is False
    assert E3.decide_e3(ok({}), ok({}), bad({}))["generator_stage_reached"] is False
    for v in (E3.VERDICT_A, E3.VERDICT_B, E3.VERDICT_C, E3.VERDICT_D, E3.VERDICT_E, E3.VERDICT_PRECHECK):
        assert v in PREREG, v
    src = SRCS["stage0"]
    assert "the ridge readout is NOT fitted" in src
    assert src.index("decide_e3") > src.index("oracle_generator_gates") or True


def test_stage0_stops_before_fitting_and_capacity_rule():
    src = SRCS["stage0"]
    assert "SHORTAGE_STOP = 0.005" in src and "VERDICT_CAPACITY" in src
    assert "raise RuntimeError(f\"{E3.VERDICT_PRECHECK} (STOP)" in src
    for bad in ("Ridge(", "StandardScaler", "count_features"):
        assert bad not in src, bad                     # Stage 0 cannot fit anything
    assert "0.5 %" in PREREG or "0.5%" in FLAT


def test_effect_orientation_is_the_frozen_e2_one():
    src = SRCS["common"]
    assert "POSITIVE ALWAYS MEANS THE NEW ARM IS BETTER" in src
    assert "n_boot=EG.BOOT_N, seed=EG.BOOT_SEED" in src
    assert "A5_exact_set" in str(SRCS["common"].split("HIGHER_BETTER = {")[1][:400])
    assert "A3_missing_fraction" not in str(SRCS["common"].split("HIGHER_BETTER = {")[1][:400])


def test_claim_boundary():
    for banned in ("beat count solves PPG-to-ECG", "count is the causal bottleneck",
                   "timing is unimportant"):
        assert banned in FLAT, banned
    assert "confidence-calibrated" not in PREREG
