"""E1 tests — docs/E1_EVENT_PLACEMENT_MORPHOLOGY_DECOMPOSITION_PREREGISTRATION.md section 22.

E1 trains nothing and tunes no threshold: every network is frozen and no optimizer may be reachable.
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

from ppg2ecg.evaluation import e1_decompose as E1
from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import m1_structural as M1
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.evaluation import rpeaks as RP

ROOT = Path(__file__).resolve().parents[1]
MOD_SRC = (ROOT / "src/ppg2ecg/evaluation/e1_decompose.py").read_text()
SRCS = {"module": MOD_SRC, "main": (ROOT / "scripts/e1_decompose.py").read_text(),
        "figures": (ROOT / "scripts/e1_figures.py").read_text()}
PREREG = (ROOT / "docs/E1_EVENT_PLACEMENT_MORPHOLOGY_DECOMPOSITION_PREREGISTRATION.md").read_text()
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


def test_e1_trains_nothing_and_tunes_nothing():
    for n, code in CODE.items():
        for bad in ("torch.optim", "AdamW", ".backward(", ".train()", "requires_grad_(True)", "torch.save",
                    "imeanflow_loss", "THRESH_GRID", "best_threshold", "for th in", "seed_everything"):
            assert bad not in code, f"{n}: {bad}"
    assert '"training_performed": False' in SRCS["main"]
    assert "NO TRAINING" in PREREG and "NO THRESHOLD TUNING" in PREREG


def test_frozen_component_hashes():
    def fsha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert fsha(ROOT / "outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt").startswith("557c7054")
    assert fsha(ROOT / "outputs/o2c_canon_oracle_seed42/checkpoint_final.pt").startswith("5aab09be")
    assert fsha(ROOT / "outputs/r1_global_tcn_seed42/checkpoint_best.pt").startswith("bfe76ea6")
    assert fsha(ROOT / "src/ppg2ecg/evaluation/o2b_warp.py").startswith("cb4d1866")
    assert fsha(ROOT / "src/ppg2ecg/evaluation/o2_warp.py").startswith("046becfb")
    assert "C.load_models(dev, with_r1=True)" in SRCS["main"]      # the frozen loader asserts every hash


FLAT = " ".join(PREREG.replace("*", "").replace("`", "").split())


def test_post_hoc_status_is_declared():
    flat = FLAT
    assert "E1 was designed AFTER the O3 results were known" in flat
    assert "frozen post-hoc diagnostic criterion" in flat
    assert "not independent preregistered confirmation" in flat
    assert "POST-O3 DIAGNOSTIC" in MOD_SRC


# --------------------------------------------------------------------------------- cohort
def test_exact_o3_cohort():
    assert "len(X) != 2048 or n_beats != 19834" in (ROOT / "scripts/o3_common.py").read_text()
    assert "!= 1922" in (ROOT / "scripts/o3_common.py").read_text()
    assert "C.load_cohort()" in SRCS["main"]
    assert E1.PRIMARY_ARMS == ("B", "ORACLE", "JITTER_2", "JITTER_4", "JITTER_8", "MISS1", "EXTRA1",
                               "R1-SCHEDULE")
    assert E1.SYNTHETIC_O2C_ARMS == ("ORACLE", "JITTER_2", "JITTER_4", "JITTER_8", "MISS1", "EXTRA1")
    for bad in ("MISS_2", "EXTRA_2", "JITTER_1", "JITTER_6"):
        assert f'"{bad}"' not in CODE["main"].split("ARMS = ")[1][:400], bad


# --------------------------------------------------------------------------------- identity
def test_synthetic_identity_is_exact():
    for j in (0, 2, 4, 8):
        s = O3.jitter_schedule(R9, 0, j, "an0", "wrist", 3)
        p = O3.retained_pairs("JITTER", j, 0, R9, s, "an0", "wrist", 3)
        np.testing.assert_array_equal(p[:, 0], np.arange(R9.size))
        np.testing.assert_array_equal(p[:, 1], np.arange(R9.size))
    s = O3.miss_schedule(R9, 0, 1, "an0", "wrist", 3)
    p = O3.retained_pairs("MISS", 1, 0, R9, s, "an0", "wrist", 3)
    assert p.shape == (R9.size - 1, 2)
    dropped = set(range(R9.size)) - set(p[:, 1].tolist())
    assert len(dropped) == 1 and 0 not in dropped and R9.size - 1 not in dropped
    s = O3.extra_schedule(R9, 0, 1, "an0", "wrist", 3)
    p = O3.retained_pairs("EXTRA", 1, 0, R9, s, "an0", "wrist", 3)
    assert p.shape == (R9.size, 2)                                  # inserted beat carries NO gt identity
    assert set(range(s.size)) - set(p[:, 0].tolist()) != set()
    assert "inserted beat has `gt_beat_id = NONE`" in PREREG
    assert "if si not in ident:" in SRCS["main"] and "inserted beat: no GT identity" in SRCS["main"]


def test_r1_matcher_is_monotonic_one_to_one_with_the_frozen_objective():
    assert E1.TOL_IDENTITY_MS == 150.0
    ref, pred = np.array([100, 118]), np.array([110, 128])
    greedy = RP.match_rpeaks(ref, pred, E1.FS, 150.0)[0]
    dp, ur, up = E1.dp_match(ref, pred, E1.FS, 150.0)
    assert len(greedy) == 1 and len(dp) == 2                        # cardinality is maximised first
    assert dp == [(0, 0), (1, 1)] and ur == 0 and up == 0
    rng = np.random.default_rng(3)
    for _ in range(40):
        g = np.sort(rng.choice(np.arange(50, 1000), size=int(rng.integers(3, 12)), replace=False))
        s = np.sort(rng.choice(np.arange(50, 1000), size=int(rng.integers(3, 12)), replace=False))
        m, a, b = E1.dp_match(g, s)
        assert all(m[i][0] < m[i + 1][0] and m[i][1] < m[i + 1][1] for i in range(len(m) - 1))   # monotone
        assert len({i for i, _ in m}) == len(m) and len({j for _, j in m}) == len(m)             # one-to-one
        assert all(abs(int(g[i]) - int(s[j])) <= 150.0 / 1000 * E1.FS for i, j in m)             # tolerance
        assert a == len(g) - len(m) and b == len(s) - len(m)
        assert E1.dp_match(g, s)[0] == m                                                         # deterministic
    assert "TARGET-DERIVED DIAGNOSTIC MATCHING" in PREREG
    assert "never" in PREREG.split("TARGET-DERIVED DIAGNOSTIC MATCHING")[1][:400]


def test_r1_identity_never_modifies_the_supplied_schedule():
    tail = SRCS["main"].split("ONLY NOW the R1")[-1]
    for bad in ("S_r1[i] =", "snap", "align_to_gt", "np.roll", "shift_schedule", "correct_schedule"):
        assert bad not in tail, bad
    assert SRCS["main"].count("S_r1 = C.r1_schedules(") == 1
    assert "Never modifies either schedule" in MOD_SRC


# --------------------------------------------------------------------------------- chain
def test_chain_uses_the_exact_frozen_fifty_millisecond_matcher():
    assert E1.TOL_CHAIN_MS == 50.0
    assert "RP.match_rpeaks(s, p, E1.FS, E1.TOL_CHAIN_MS)" in SRCS["main"]
    assert "exact frozen O3 adherence matcher" in FLAT
    for bad in ("gt_pk[i] +", "np.roll", "shift_pred", "align_to_gt", "snap_to", "np.correlate"):
        assert bad not in CODE["main"], bad


# --------------------------------------------------------------------------------- morphology
def test_own_center_support_and_eligibility_are_frozen():
    assert (E1.WIN_LO, E1.WIN_HI) == (-10, 15) and (E1.ELIG_LO, E1.ELIG_HI) == (11, 15)
    assert E1.eligible(11) and not E1.eligible(10)
    assert E1.eligible(E1.T_LEN - 1 - 15) and not E1.eligible(E1.T_LEN - 15)
    assert "[-10, +15]" in PREREG or "`[-10, +15]`" in PREREG
    for bad in ("correlate", "argmax(np.correlate", "dtw", "best_shift", "for sh in range", "np.roll"):
        assert bad not in _code(MOD_SRC), bad
    assert "No local cross-correlation shift" in PREREG


def test_own_center_uses_each_waveforms_own_event():
    y = np.zeros(1024)
    for c in (200, 400, 600, 800):
        y[c - 2:c + 3] = [-0.2, 0.5, 2.0, 0.5, -0.2]
    shifted = np.roll(y, 7)                                          # same shape, displaced by 7 samples
    iqr = {t: 1.0 for t in E1.ALIGNED}
    same = E1.beat_shape(shifted, y, 407, 400, iqr)                  # own centres: shape is identical
    anchored = E1.beat_shape(shifted, y, 400, 400, iqr)              # GT-anchored: placement is enforced
    assert same is not None and anchored is not None
    for t in E1.ALIGNED:
        assert same[E1.NAE[t]] == pytest.approx(0.0, abs=1e-9), t     # own-centre sees no damage
    assert anchored["local_raw_rmse"] > same["local_raw_rmse"]        # GT-anchored does
    assert same["local_corr"] == pytest.approx(1.0, abs=1e-9)
    assert E1.beat_shape(shifted, y, 5, 400, iqr) is None             # ineligible centre is excluded


def test_primitives_are_the_exact_o1_ones():
    scal = json.loads((ROOT / "artifacts/o1_component_extractability/target_scaling.json").read_text())["targets"]
    for t, v in (("median_QRS_p2p", 0.5053170621395111), ("median_QRS_max_abs_derivative", 0.22995377704501152),
                 ("median_QRS_curvature_energy", 0.033796727107052275), ("median_QRS_width_ms", 31.25)):
        assert scal[t]["scale_train_IQR"] == v and t in E1.ALIGNED
    assert "O3.beat_primitives(" in MOD_SRC and "def beat_primitives" not in MOD_SRC
    assert "M1.d1(" in MOD_SRC and "M1.d2(" in MOD_SRC
    assert M1.CORE == 10 and OT.TARGET_IDS["median_QRS_max_abs_derivative"] == "T6"


def test_coverage_travels_with_every_morphology_row():
    src = SRCS["main"]
    assert "def coverage(" in src
    for k in ("C1_schedule_to_gt_identity", "C2_generated_to_supplied_adherence", "C3_full_chain_P_S_G",
              "C4_gt_beats_excluded", "C5_generated_beats_excluded"):
        assert k in src, k
    assert "COVERAGE_MIN = 0.80" in MOD_SRC and E1.COVERAGE_MIN == 0.80
    assert 'cov_ok = all(COV[a]["C3_full_chain_P_S_G"] >= E1.COVERAGE_MIN' in src
    assert "No morphology-only metric may appear without its coverage" in PREREG


# --------------------------------------------------------------------------------- bootstrap
def test_bootstrap_aggregates_beats_to_windows_first():
    assert "def window_median" in MOD_SRC and "Per-window MEDIAN over eligible beats" in MOD_SRC
    src = SRCS["main"]
    assert "w = E1.window_median(beats, keys)" in src
    assert 'C.O1E.cluster_bootstrap(d, SUB, CLUSTER, n_boot=E1.BOOT_N, seed=E1.BOOT_SEED)' in src
    assert E1.BOOT_N == 2000 and E1.BOOT_SEED == 20260904
    assert "a = np.array([r[key] for r in a_rows], float)" in src         # window-level vectors only
    beats = [{"x": 1.0}, {"x": 3.0}, {"x": np.nan}]
    assert E1.window_median(beats, ["x"])["x"] == 2.0
    assert np.isnan(E1.window_median([], ["x"])["x"])


def test_damage_orientation_is_never_reversed():
    assert "Damage(A,B) = Error_A - Error_B" in SRCS["main"]
    assert "positive = FIRST arm worse" in SRCS["main"]
    assert "a positive effect always means the FIRST arm is worse" in PREREG.replace("**", "")


# --------------------------------------------------------------------------------- strata
def test_topology_classes_and_bins_are_exact():
    g = np.array([100, 300, 500, 700])
    assert E1.topology(g, g)["topology_class"] == E1.T0
    assert E1.topology(g, g[:-1])["topology_class"] == E1.T2
    assert E1.topology(g, np.sort(np.append(g, 200)))["topology_class"] == E1.T3
    far = np.array([100, 300, 500, 900])                     # same count, one event beyond +-150 ms
    assert E1.topology(g, far)["topology_class"] == E1.T1
    assert E1.TIMING_BINS == ((0.0, 16.0, "A"), (16.0, 32.0, "B"), (32.0, float("inf"), "C"))
    assert E1.timing_bin(16.0) == "A" and E1.timing_bin(16.001) == "B"
    assert E1.timing_bin(32.0) == "B" and E1.timing_bin(32.001) == "C"
    assert E1.timing_bin(float("nan")) is None
    assert "16 ms" in PREREG and "32 ms" in PREREG


def test_stratum_minimum_coverage_is_frozen():
    assert (E1.MIN_STRATUM_WINDOWS, E1.MIN_STRATUM_PER_SUBJECT) == (100, 30)
    assert E1.stratum_sufficient(100, {"an0": 30, "k2s": 70})
    assert not E1.stratum_sufficient(99, {"an0": 30, "k2s": 69})
    assert not E1.stratum_sufficient(200, {"an0": 29, "k2s": 171})
    assert not E1.stratum_sufficient(200, {})
    assert "INSUFFICIENT STRATUM COVERAGE" in PREREG
    assert "the threshold is never lowered" in FLAT


def test_adherence_label_cannot_enter_the_verdict():
    assert E1.ADHERENCE_HIGH == 0.90
    fn = next(n for n in ast.walk(ast.parse(MOD_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == "decide_e1")
    body = ast.unparse(fn)
    for bad in ("ADHERENCE", "adherence", "HIGH ADHERENCE"):
        assert bad not in body, bad
    assert "does not enter the global verdict" in PREREG.replace("**", "")


# --------------------------------------------------------------------------------- gates and verdict
def _b(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi,
            "verdict": "improves" if lo > 0 else ("worsens" if hi < 0 else "unresolved")}


def _res(**over):
    r = {"gt_anchored_local_deriv_rmse_J4_vs_ORACLE": _b(0.1, 0.05, 0.15),
         "gt_anchored_local_curvature_err_J4_vs_ORACLE": _b(0.1, 0.05, 0.15),
         "T6_gt_minus_own_damage_J4": _b(0.1, 0.05, 0.15), "T7_gt_minus_own_damage_J4": _b(0.1, 0.05, 0.15),
         "MISS1_damage_T6": _b(0.5, 0.4, 0.6), "MISS1_damage_T7": _b(0.3, 0.2, 0.4),
         "EXTRA1_damage_T6": _b(0.3, 0.2, 0.4), "EXTRA1_damage_T7": _b(0.2, 0.1, 0.3),
         "excess_MISS_T6": _b(0.4, 0.3, 0.5), "excess_MISS_T7": _b(0.25, 0.15, 0.35),
         "excess_EXTRA_T6": _b(0.2, 0.1, 0.3), "excess_EXTRA_T7": _b(0.1, 0.02, 0.2),
         "JITTER8_damage_T6": _b(0.1, 0.05, 0.15), "JITTER8_damage_T7": _b(0.05, 0.02, 0.09)}
    r.update(over)
    return r


def test_gates_and_verdict_tree():
    res = _res()
    pg, tg = E1.placement_gates(res), E1.topology_gates(res)
    assert pg["supported"] and tg["supported"]
    assert E1.decide_e1(pg, tg, res, True, True)["verdict"] == E1.VERDICT_A
    # placement gate P3/P4 use the same-functional GT-anchored comparison
    r2 = _res(T6_gt_minus_own_damage_J4=_b(-0.1, -0.2, -0.02))
    assert not E1.placement_gates(r2)["P3"]
    # fine-placement priority: no topology excess damage and jitter damage dominates
    r3 = _res(excess_MISS_T6=_b(-0.2, -0.3, -0.1), excess_MISS_T7=_b(-0.2, -0.3, -0.1),
              JITTER8_damage_T6=_b(0.9, 0.8, 1.0), JITTER8_damage_T7=_b(0.9, 0.8, 1.0))
    assert E1.decide_e1(E1.placement_gates(r3), E1.topology_gates(r3), r3, True, True)["verdict"] == E1.VERDICT_B
    # mixed: both damage, neither dominates
    r4 = _res(excess_MISS_T6=_b(-0.1, -0.2, -0.02), excess_MISS_T7=_b(0.1, 0.02, 0.2),
              JITTER8_damage_T6=_b(0.2, 0.1, 0.3), JITTER8_damage_T7=_b(0.05, 0.02, 0.09))
    assert E1.decide_e1(E1.placement_gates(r4), E1.topology_gates(r4), r4, True, True)["verdict"] == E1.VERDICT_C
    # coverage failure always wins
    assert E1.decide_e1(pg, tg, res, False, True)["verdict"] == E1.VERDICT_D
    assert E1.decide_e1(pg, tg, res, True, False)["verdict"] == E1.VERDICT_D
    for v in (E1.VERDICT_A, E1.VERDICT_B, E1.VERDICT_C, E1.VERDICT_D):
        assert v in PREREG, v


def test_r1_strata_cannot_override_the_synthetic_evidence():
    fn = next(n for n in ast.walk(ast.parse(MOD_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == "decide_e1")
    fn = ast.parse(ast.unparse(fn)).body[0]
    if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]                                    # the docstring is prose, not logic
    body = ast.unparse(fn).replace("r1_strata_can_override", "")
    for bad in ("r1_", "R1", "stratum", "bin"):
        assert bad not in body, bad
    assert E1.decide_e1(E1.placement_gates(_res()), E1.topology_gates(_res()), _res(), True, True)[
        "r1_strata_can_override"] is False
    assert "can never override conflicting synthetic intervention evidence" in PREREG.replace("**", "")


def test_r1_runs_only_after_the_synthetic_contrasts_are_written():
    src = SRCS["main"]
    i_syn = src.index('C.wcsv(ART / "topology_excess_damage.csv"')
    i_r1 = src.index("S_r1 = C.r1_schedules(")
    assert i_syn < i_r1
    assert "ONLY NOW the R1 diagnostic assignment" in src
    assert src.index('C.wcsv(ART / "synthetic_contrasts.csv"') < i_r1


def test_claim_limits_are_stated():
    for banned in ("Beat count is the causal bottleneck", "PPG lacks fine timing information",
                   "A beat detector will solve PPG-to-ECG", "Shape is independent of timing"):
        assert banned in FLAT, banned
    assert "confidence-calibrated" not in PREREG
    assert "not causal" in SRCS["main"] or "observational" in SRCS["main"]


def test_reconstruction_gate_treats_an_undefined_quantity_as_agreement():
    import importlib.util
    sp = importlib.util.spec_from_file_location("e1_main_t", ROOT / "scripts/e1_decompose.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    nan = float("nan")
    assert m.agree(nan, nan, 1e-9)                      # MISS/EXTRA leave max_abs_shift undefined in O3 too
    assert not m.agree(nan, 1.0, 1e-9) and not m.agree(1.0, nan, 1e-9)
    assert m.agree(1.0, 1.0 + 1e-12, 1e-9) and not m.agree(1.0, 1.001, 1e-9)
    assert "the exact O3 expression" in SRCS["main"]     # the shift is recomputed, never copied from O3


def test_chaining_uses_gt_sample_positions_not_beat_indices():
    """A GT beat index is not a time. Coverage collapses to a few percent if the two are confused."""
    src = SRCS["main"]
    assert "gpos = np.asarray(gt_pk[i], dtype=np.int64)" in src
    assert "gp = int(gpos[g])" in src and "the GT beat's SAMPLE POSITION" in src
    assert "E1.beat_shape(gen[i], Yd[i], int(p[pj]), gp, iqr)" in src
    assert "E1.beat_shape(gen[i], Yd[i], gp, gp, iqr)" in src
    assert "int(p[pj]) - gp" in src
    assert "analyse_arm(gen, Yd, S, P, pairs, gt_pk, iqr" in src

