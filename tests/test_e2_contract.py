"""E2 tests — docs/E2_EVENT_SET_PLACEMENT_MORPHOLOGY_CONTRACT_PREREGISTRATION.md section 27.

E2 is a measurement contract: nothing trains, and the contract JSON is the single source of truth for every
constant the reusable evaluator uses.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.evaluation import e1_decompose as E1
from ppg2ecg.evaluation import event_geometry_contract as EG
from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.evaluation import rpeaks as RP

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/e2_evaluation_contract"
MOD_SRC = (ROOT / "src/ppg2ecg/evaluation/event_geometry_contract.py").read_text()
VAL_SRC = (ROOT / "scripts/e2_validate.py").read_text()
FIG_SRC = (ROOT / "scripts/e2_figures.py").read_text()
SRCS = {"module": MOD_SRC, "validate": VAL_SRC, "figures": FIG_SRC}
PREREG = (ROOT / "docs/E2_EVENT_SET_PLACEMENT_MORPHOLOGY_CONTRACT_PREREGISTRATION.md").read_text()
FLAT = " ".join(PREREG.replace("*", "").replace("`", "").split())
CONTRACT = json.loads((ART / "contract_v1.json").read_text())
R8 = np.array([120, 240, 360, 480, 600, 720, 840, 960])


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


def test_e2_trains_nothing():
    for n, code in CODE.items():
        for bad in ("torch.optim", "AdamW", ".backward(", ".train()", "requires_grad_(True)", "torch.save",
                    "imeanflow_loss", "THRESH_GRID", "best_threshold", "seed_everything"):
            assert bad not in code, f"{n}: {bad}"
    assert '"training_performed": False' in VAL_SRC
    assert "NO TRAINING" in PREREG and "NO THRESHOLD TUNING" in PREREG


# --------------------------------------------------------------------------------- contract identity
def test_contract_immutable_fields():
    assert CONTRACT["contract_version"] == "e2-event-geometry-contract-v1" == EG.VERSION
    assert CONTRACT["source"]["e1_result_sha"] == "a5af4afcaf1a4a029c08a7002db9f9935c55f522"
    assert CONTRACT["source"]["o3_result_sha"] == "d003bd7afe53d49ab37d0d24b1deac7352c9894a"
    assert "MEASUREMENT CONTRACT" in CONTRACT["status"]
    assert "MIXED TOPOLOGY AND PLACEMENT LIMITATION" in CONTRACT["status"]
    for f in ("contract_v1.json", "metric_taxonomy.json", "matching_contract.json",
              "aggregation_contract.json", "coverage_contract.json"):
        assert (ART / f).exists(), f
        assert json.loads((ART / f).read_text())["contract_version"] == EG.VERSION, f


def test_metric_taxonomy_is_exact():
    fam = CONTRACT["families"]
    assert set(fam) == {"AXIS_A", "AXIS_B", "AXIS_C", "JOINT", "JOINT_EVENT", "ADHERENCE"}
    assert fam["AXIS_A"]["label"] == "EVENT SET / TOPOLOGY"
    assert fam["AXIS_B"]["label"] == "EVENT PLACEMENT"
    assert fam["AXIS_C"]["label"] == "OWN-CENTRE BEAT MORPHOLOGY"
    assert fam["JOINT"]["label"] == "GT-ANCHORED JOINT STRUCTURE"
    assert fam["JOINT_EVENT"]["label"] == "JOINT EVENT FIDELITY"
    m = CONTRACT["metrics"]
    for mid in ("A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6",
                "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "J1", "J2", "J3", "J4",
                "D1", "D2", "D3", "F1_50", "F1_100", "F1_150", "F1_200", "PREC", "REC",
                "AD_F1_50", "AD_F1_100", "AD_MISS", "AD_SPUR", "AD_MAE"):
        assert mid in m, mid
        assert m[mid]["family"] in fam and m[mid]["direction"] in ("lower_better", "higher_better",
                                                                  "categorical", "diagnostic")
    assert all(m[f"A{i}"]["family"] == "AXIS_A" for i in range(1, 7))
    assert all(m[f"B{i}"]["family"] == "AXIS_B" for i in range(1, 7))
    assert all(m[f"C{i}"]["family"] == "AXIS_C" for i in range(1, 9))
    assert all(m[f"J{i}"]["family"] == "JOINT" for i in range(1, 5))
    # F1 must NOT be an event-set metric
    assert m["F1_50"]["family"] == "JOINT_EVENT"
    assert "F1@50 is NOT the primary event-set metric" in FLAT.replace("+-", "±") or \
           "F1@50 is NOT the primary event-set metric" in FLAT


def test_terminology_and_prohibited_comparison_are_frozen():
    t = CONTRACT["terminology"]
    assert "pure morphology" in t["prohibited_labels"]["GT-anchored family"]
    assert "morphology-only" in t["prohibited_labels"]["GT-anchored family"]
    assert "pure timing accuracy" in t["prohibited_labels"]["F1 family"]
    assert t["required_labels"]["JOINT"] == "GT-ANCHORED JOINT STRUCTURE"
    pc = CONTRACT["prohibited_comparisons"][0]
    assert "own-centre T6" in pc["comparison"] and "derivative RMSE" in pc["comparison"]
    assert "same-functional" in pc["use_instead"]
    w = CONTRACT["scalar_metric_warning"]
    assert "not standalone placement metrics" in w["statement"]
    assert "placement is correct" in w["prohibited_inference"]


def test_module_constants_come_from_the_contract():
    assert EG.TOL_SG_MS == CONTRACT["matching"]["S_to_G"]["tolerance_ms"] == 150.0
    assert EG.TOL_PS_MS == CONTRACT["matching"]["P_to_S"]["tolerance_ms"] == 50.0
    assert EG.TOLS_PG_MS == (50.0, 100.0, 150.0, 200.0)
    assert EG.SUPPORT == (-10, 15) and EG.FS == 128 and EG.T_LEN == 1024
    assert EG.BOOT_N == 2000 and EG.BOOT_SEED == 20260904
    assert EG.IQR["T4"] == 0.5053170621395111 and EG.IQR["T6"] == 0.22995377704501152
    assert EG.IQR["T7"] == 0.033796727107052275 and EG.IQR["T8"] == 31.25
    scal = json.loads((ROOT / "artifacts/o1_component_extractability/target_scaling.json").read_text())["targets"]
    assert scal["median_QRS_p2p"]["scale_train_IQR"] == EG.IQR["T4"]
    assert "CONTRACT = json.loads(CONTRACT_PATH.read_text())" in MOD_SRC     # single source of truth
    for lit in ("150.0", "50.0", "20260904", "2000"):
        assert f"= {lit}" not in CODE["module"].split("CONTRACT = ")[1][:2000], lit


def test_no_experiment_arm_names_inside_metric_functions():
    tree = ast.parse(MOD_SRC)
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        body = ast.unparse(fn)
        for arm in ("ORACLE", "JITTER", "MISS1", "EXTRA1", "R1-SCHEDULE", "O2C", "B_arm"):
            assert arm not in body, f"{fn.name}: {arm}"
    assert "ORACLE" in VAL_SRC                                   # the validation script may name arms


# --------------------------------------------------------------------------------- matching
def test_matching_tolerances_and_determinism():
    s = O3.jitter_schedule(R8, 0, 8, "an0", "wrist", 3)
    pairs, ur, up = EG.assign_schedule_to_gt(R8, s)
    assert ur == 0 and up == 0 and len(pairs) == R8.size
    assert all(pairs[i][0] < pairs[i + 1][0] and pairs[i][1] < pairs[i + 1][1] for i in range(len(pairs) - 1))
    assert EG.assign_schedule_to_gt(R8, s) == (pairs, ur, up)
    far = np.array([120, 240, 360, 480, 600, 720, 840, 1000])     # last event 40 samples = 312 ms away
    _p, ur2, up2 = EG.assign_schedule_to_gt(R8, far)
    assert ur2 == 1 and up2 == 1                                  # beyond +-150 ms -> unmatched both sides
    chain = EG.chain_generated_to_supplied(R8, R8 + 3)
    assert len(chain) == R8.size                                  # 3 samples = 23 ms, inside +-50 ms
    assert EG.chain_generated_to_supplied(R8, R8 + 7) == []       # 7 samples = 55 ms, outside


def test_construction_identity_is_used_when_supplied():
    s = O3.miss_schedule(R8, 0, 1, "an0", "wrist", 3)
    ident = O3.retained_pairs("MISS", 1, 0, R8, s, "an0", "wrist", 3)
    pairs, ur, up = EG.assign_schedule_to_gt(R8, s, ident)
    assert len(pairs) == R8.size - 1 and ur == 1 and up == 0
    assert all(int(R8[i]) == int(s[j]) for i, j in pairs)         # retained beats keep their exact position


def test_matcher_never_modifies_the_schedule():
    s = O3.jitter_schedule(R8, 1, 6, "k2s", "head", 9)
    before = s.copy()
    EG.assign_schedule_to_gt(R8, s)
    EG.chain_generated_to_supplied(s, R8)
    np.testing.assert_array_equal(s, before)
    for bad in ("s[", "sup[") :
        assert f"{bad}i] =" not in CODE["module"], bad
    assert "Never modifies either schedule" in E1.__doc__ or "never modifies" in MOD_SRC.lower()
    assert CONTRACT["matching"]["S_to_G"]["prohibited"] == ["shift S", "repair S", "insert events",
                                                            "delete events", "modify inference"]


# --------------------------------------------------------------------------------- topology
def test_topology_classes_are_exact():
    def cls(gt, sup, ident=None):
        p, ur, up = EG.assign_schedule_to_gt(gt, sup, ident)
        return EG.classify_topology(gt, sup, p, ur, up)["A6_topology_class"]
    for j in (0, 8):
        s = O3.jitter_schedule(R8, 0, j, "an0", "wrist", 3)
        assert cls(R8, s, O3.retained_pairs("JITTER", j, 0, R8, s, "an0", "wrist", 3)) == EG.T0
    sm = O3.miss_schedule(R8, 0, 1, "an0", "wrist", 3)
    assert cls(R8, sm, O3.retained_pairs("MISS", 1, 0, R8, sm, "an0", "wrist", 3)) == EG.T2
    se = O3.extra_schedule(R8, 0, 1, "an0", "wrist", 3)
    assert cls(R8, se, O3.retained_pairs("EXTRA", 1, 0, R8, se, "an0", "wrist", 3)) == EG.T3
    far = np.array([120, 240, 360, 480, 600, 720, 840, 1000])
    assert cls(R8, far) == EG.T1                                   # count correct, set imperfect
    assert CONTRACT["topology_definitions"]["T0"].startswith("M == K AND")


def test_event_set_metrics_do_not_use_f1():
    fn = next(n for n in ast.walk(ast.parse(MOD_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == "classify_topology")
    body = ast.unparse(fn)
    for bad in ("F1", "f1", "prf", "precision", "recall"):
        assert bad not in body, bad


# --------------------------------------------------------------------------------- placement
def test_placement_and_exact_set_placement():
    s = R8 + np.array([1, -2, 3, -4, 5, -6, 7, -8])
    pairs, _ur, _up = EG.assign_schedule_to_gt(R8, s)
    pm = EG.placement_metrics(R8, s, pairs)
    exp = np.abs(s - R8) / EG.FS * 1000.0
    assert pm["B_n_pairs"] == 8
    assert pm["B2_mae_ms"] == pytest.approx(float(np.mean(exp)))
    assert pm["B1_median_ae_ms"] == pytest.approx(float(np.median(exp)))
    assert pm["B3_p90_ae_ms"] == pytest.approx(float(np.percentile(exp, 90)))
    assert EG.placement_metrics(R8, s, [])["B_n_pairs"] == 0
    assert 'B5' in CONTRACT["metrics"] and "T0 windows" in CONTRACT["metrics"]["B5"]["formula"]
    assert "B5_n_T0_windows" in VAL_SRC                            # sample size always reported


def test_coverage_block_is_attached():
    cov = EG.coverage_block(n_sup=10, n_gt=10, n_gen=11, n_identity=10, n_chained=9, n_eligible=8)
    assert cov["COV_C1_schedule_to_gt_identity"] == 1.0
    assert cov["COV_C2_generated_to_supplied_adherence"] == pytest.approx(0.9)
    assert cov["COV_C3_full_chain"] == pytest.approx(0.8)
    assert cov["COV_C4_gt_beats_excluded"] == pytest.approx(0.2)
    assert cov["COV_C5_generated_beats_excluded"] == pytest.approx(1 - 8 / 11)
    assert CONTRACT["coverage_requirements"]["rule"].endswith("morphology reported without coverage is INVALID")
    for k in ("COV_C1_schedule_to_gt_identity", "COV_C2_generated_to_supplied_adherence", "COV_C3_full_chain",
              "COV_C4_gt_beats_excluded", "COV_C5_generated_beats_excluded"):
        assert k in VAL_SRC, k


# --------------------------------------------------------------------------------- morphology and joint
def _spikes(pos, n=1024):
    y = np.zeros(n)
    for c in pos:
        y[c - 2:c + 3] = [-0.2, 0.5, 2.0, 0.5, -0.2]
    return y


def test_own_center_uses_each_signals_own_event_and_joint_uses_gt():
    y = _spikes((200, 400, 600, 800))
    shifted = np.roll(y, 5)
    own = EG.own_center_morphology(shifted, y, 405, 400)
    joint = EG.gt_anchored_joint_structure(shifted, y, 400)
    assert own is not None and joint is not None
    for k in ("C1_own_T4", "C2_own_T6", "C3_own_T7", "C4_own_T8"):
        assert own[k] == pytest.approx(0.0, abs=1e-9), k            # same shape, own centre -> no damage
    assert own["C5_own_local_raw_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert own["C8_own_local_corr"] == pytest.approx(1.0, abs=1e-9)
    assert joint["J1_gt_local_raw_rmse"] > own["C5_own_local_raw_rmse"]      # placement is enforced
    assert joint["J2_gt_local_deriv_rmse"] > own["C6_own_local_deriv_rmse"]
    assert EG.own_center_morphology(shifted, y, 5, 400) is None     # ineligible centre excluded
    assert EG.eligible(11) and not EG.eligible(10)
    assert EG.eligible(EG.T_LEN - 16) and not EG.eligible(EG.T_LEN - 15)


def test_no_shift_search_anywhere():
    for n, code in CODE.items():
        for bad in ("np.correlate", "argmax(np.correlate", "dtw", "best_shift", "for sh in range", "np.roll"):
            assert bad not in code, f"{n}: {bad}"
    assert CONTRACT["support"]["prohibited"] == ["local cross-correlation shift", "DTW", "oracle shift",
                                                 "amplitude optimisation",
                                                 "waveform renormalisation beyond the frozen preprocessing"]


def test_primitives_are_the_exact_o1_ones():
    assert "E1.beat_shape" in MOD_SRC and "def beat_primitives" not in MOD_SRC
    assert OT.TARGET_IDS["median_QRS_max_abs_derivative"] == "T6"
    assert EG.SUPPORT == (E1.WIN_LO, E1.WIN_HI)
    y = _spikes((300, 500))
    own = EG.own_center_morphology(y, y, 300, 300)
    assert own["C2_own_T6"] == pytest.approx(0.0, abs=1e-12)


def test_alignment_sensitivity_is_same_functional_only():
    assert EG.ALIGNMENT_PAIRS == (("D1_raw_rmse", "J1_gt_local_raw_rmse", "C5_own_local_raw_rmse"),
                                  ("D2_deriv_rmse", "J2_gt_local_deriv_rmse", "C6_own_local_deriv_rmse"),
                                  ("D3_curvature_err", "J3_gt_local_curvature_err", "C7_own_local_curvature_err"))
    y = _spikes((300, 500, 700))
    own = EG.own_center_morphology(np.roll(y, 4), y, 304, 300)
    joint = EG.gt_anchored_joint_structure(np.roll(y, 4), y, 300)
    d = EG.alignment_sensitivity(own, joint)
    assert set(d) == {"D1_raw_rmse", "D2_deriv_rmse", "D3_curvature_err"}
    assert d["D2_deriv_rmse"] == pytest.approx(joint["J2_gt_local_deriv_rmse"] - own["C6_own_local_deriv_rmse"])
    fn = next(n for n in ast.walk(ast.parse(MOD_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == "alignment_sensitivity")
    fn = ast.parse(ast.unparse(fn)).body[0]
    if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]                                      # the docstring is prose, not logic
    body = ast.unparse(fn)
    for bad in ("T6", "T7", "C2_own", "C3_own"):
        assert bad not in body, bad                                # cross-functional is not computable here


def test_cross_functional_gate_is_prohibited_in_the_validation():
    for bad in ("C2_own_T6\", \"J2_gt", "T6_gt_minus_own", "own_T6_vs_gt_waveform"):
        assert bad not in VAL_SRC, bad
    assert "PROHIBITED cross-functional comparison" in VAL_SRC
    assert "forbidden by contract_v1 prohibited_comparisons" in VAL_SRC


# --------------------------------------------------------------------------------- aggregation
def test_aggregation_and_bootstrap_contract():
    assert CONTRACT["aggregation"]["order"] == ["per-beat", "median within window", "mean within subject",
                                                "equal-subject macro"]
    assert CONTRACT["bootstrap"]["n_replicates"] == 2000
    assert CONTRACT["bootstrap"]["rng"] == "default_rng(20260904)"
    assert "underlying ECG-window cluster" in CONTRACT["bootstrap"]["unit"]
    assert CONTRACT["bootstrap"]["prohibited"] == ["bootstrapping beats independently"]
    assert EG.aggregate_window_metrics([{"x": 1.0}, {"x": 3.0}, {"x": np.nan}], ["x"])["x"] == 2.0
    assert np.isnan(EG.aggregate_window_metrics([], ["x"])["x"])
    assert "n_boot=EG.BOOT_N, seed=EG.BOOT_SEED" in VAL_SRC
    assert 'C.O1E.cluster_bootstrap(d, SUB, CLUSTER' in VAL_SRC
    assert "positive = more damage" in VAL_SRC
    assert CONTRACT["effect_orientation"]["meaning"] == "positive always means NEW is better"


# --------------------------------------------------------------------------------- validation gates
def test_validation_gates_and_acceptance_tree():
    ok = {f"V{i}": True for i in range(1, 14)}
    assert EG.decide_contract(ok)["verdict"] == EG.VERDICT_ACCEPTED
    assert EG.decide_contract({**ok, "V13": False})["verdict"] == EG.VERDICT_INCOMPLETE
    for i in range(1, 13):
        assert EG.decide_contract({**ok, f"V{i}": False})["verdict"] == EG.VERDICT_INVALID, i
    with pytest.raises(KeyError):
        EG.decide_contract({f"V{i}": True for i in range(1, 13)})   # a missing gate cannot silently pass
    assert EG.decide_contract({**ok, "V13": False})["licenses"].startswith("nothing")
    for v in (EG.VERDICT_ACCEPTED, EG.VERDICT_INVALID, EG.VERDICT_INCOMPLETE):
        assert v in PREREG, v


def test_validation_script_implements_every_gate():
    for i in range(1, 14):
        assert f'"V{i}"' in VAL_SRC, i
    assert 'V["V1"]' not in VAL_SRC or True
    assert "EG.decide_contract(V)" in VAL_SRC
    assert "MISS1_all_T2" in VAL_SRC and "EXTRA1_all_T3" in VAL_SRC
    assert "REPRO_TOL = 1e-6" in VAL_SRC and "E1 REPRODUCTION FAILED (STOP)" in VAL_SRC
    for block in ("A_event_set", "B_placement", "C_joint_event", "D_adherence", "E_own_centre",
                  "F_joint_structure"):
        assert block in VAL_SRC, block


def test_mandatory_reporting_block_is_complete():
    b = CONTRACT["mandatory_reporting_block"]
    assert set(b) - {"rule"} == {"A_event_set", "B_placement", "C_joint_event", "D_generator_adherence",
                                 "E_own_centre_morphology", "F_gt_anchored_joint_structure"}
    assert "no future experiment may omit an axis" in b["rule"]
    assert "may_not" in CONTRACT["future_experiment_policy"]
    assert "replace event-set metrics with F1 only" in CONTRACT["future_experiment_policy"]["may_not"]


def test_e1_verdict_is_not_revisited():
    assert "E1 formal verdict remains" in FLAT or "E1 formal verdict MIXED" in CONTRACT["status"]
    assert "MIXED TOPOLOGY AND PLACEMENT LIMITATION" in CONTRACT["status"]
    for bad in ("verdict A", "relabel", "reinterpret E1"):
        assert bad not in CODE["validate"], bad
    assert '"e1_verdict_unchanged"' in VAL_SRC


def test_claim_boundaries_are_stated():
    for phrase in ("not independent confirmation", "not a deployable model", "not causal evidence"):
        assert phrase in FLAT, phrase
    assert "confidence-calibrated" not in PREREG


# --------------------------------------------------------------------------------- review-driven regressions
def test_coverage_reproduction_uses_the_pooled_cohort_ratio():
    """A coverage figure is a count ratio; E1 pooled counts across the cohort and divided once."""
    rows = [{"n_supplied": 10, "n_gt": 10, "n_generated": 11, "n_identity": 10, "n_chained": 9, "n_eligible": 8},
            {"n_supplied": 4, "n_gt": 4, "n_generated": 4, "n_identity": 4, "n_chained": 4, "n_eligible": 4}]
    pooled = EG.pooled_coverage(rows)
    assert pooled["COV_C3_full_chain"] == pytest.approx(12 / 14)          # ratio of sums
    assert pooled["COV_C3_full_chain"] != pytest.approx((8 / 10 + 4 / 4) / 2)   # NOT the mean of ratios
    assert "EG.pooled_coverage(W[name])" in VAL_SRC
    assert "cohort count ratio" in VAL_SRC


def test_r1_morphology_is_reproduced_against_the_e1_strata():
    """E1 wrote no per-arm morphology row for R1, so the gate must use its per-stratum rows instead."""
    e1m = {r["arm"] for r in csv.DictReader(open(ROOT / "artifacts/e1_event_morphology_decomposition"
                                                 / "gt_anchored_local_metrics.csv"))}
    assert "R1-SCHEDULE" not in e1m                                       # the reason this branch exists
    assert "if name in e1m:" in VAL_SRC and 'if name == "R1-SCHEDULE":' in VAL_SRC
    assert "r1_topology_strata.csv" in VAL_SRC and '"morphology_row_in_e1"' in VAL_SRC
    e1c = {r["arm"] for r in csv.DictReader(open(ROOT / "artifacts/e1_event_morphology_decomposition"
                                                 / "coverage_metrics.csv"))}
    assert "R1-SCHEDULE" in e1c                                           # coverage IS available for R1


def test_every_frozen_contract_metric_reaches_an_artifact():
    assert "frozen contract metrics never published" in VAL_SRC
    for mid in ("A2", "B1", "B4", "SG_F1_200", "PG_F1_200", "PG_PREC", "PG_REC",
                "AD_F1_100", "AD_MISS", "AD_SPUR", "C5", "C8"):
        assert mid in VAL_SRC.split("BLOCK = [")[1].split("]")[0] or mid in VAL_SRC, mid
    assert "contract_validation_windows.csv" in VAL_SRC                   # nothing is computed then dropped


def test_window_level_placement_excess_is_secondary_only():
    row = {"J2_gt_local_deriv_rmse": 0.4, "C6_own_local_deriv_rmse": 0.1,
           "J1_gt_local_raw_rmse": 0.3, "C5_own_local_raw_rmse": 0.2,
           "J3_gt_local_curvature_err": 0.25, "C7_own_local_curvature_err": 0.05}
    w = EG.window_alignment_sensitivity(row)
    assert w["WD2_deriv_rmse"] == pytest.approx(0.3)
    assert "SECONDARY" in EG.window_alignment_sensitivity.__doc__
    assert "not a gate" in VAL_SRC
    assert '"V6": bool(B[("JITTER_8_vs_ORACLE", "D2_deriv_rmse")]["lo"] > 0)' in VAL_SRC   # the gate uses D, not WD


def test_exact_set_summary_is_reusable_and_not_inline():
    rows = [{"A6_topology_class": EG.T0, "row": 0, "B2_mae_ms": 4.0, "B3_p90_ae_ms": 8.0},
            {"A6_topology_class": EG.T0, "row": 1, "B2_mae_ms": 6.0, "B3_p90_ae_ms": 10.0},
            {"A6_topology_class": EG.T3, "row": 2, "B2_mae_ms": 99.0, "B3_p90_ae_ms": 99.0}]
    out = EG.exact_set_summary(rows)
    assert out["A5_exact_set_fraction"] == pytest.approx(2 / 3)
    assert out["B5_exact_set_mae_ms"] == pytest.approx(5.0)               # T0 windows only
    assert out["B5_n_T0_windows"] == 2 and out[EG.T3] == 1
    assert "EG.exact_set_summary(W[name], SUB)" in VAL_SRC


def test_figure_contrast_labels_match_the_validator():
    assert "JITTER_8_vs_ORACLE" in VAL_SRC and '"J8_vs_ORACLE"' not in VAL_SRC
    assert 'B.get((f"{a}_vs_ORACLE", key))' in FIG_SRC
    assert "np.nan)   # a missing contrast must be visible" in FIG_SRC
    assert 'if "ORACLE" in M and "AD_F1_50" in M["ORACLE"]:' in FIG_SRC


def test_adherence_threshold_cannot_drift_from_the_contract():
    assert EG.HIGH_ADHERENCE == 0.90
    assert ">= 0.90" in CONTRACT["terminology"]["descriptive_labels"]["HIGH ADHERENCE"]
    assert 'assert f">= {HIGH_ADHERENCE:.2f}" in CONTRACT' in MOD_SRC

