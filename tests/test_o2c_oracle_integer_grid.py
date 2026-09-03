"""O2c tests — docs/O2C_ORACLE_INTEGER_GRID_MEANFLOW_PREREGISTRATION.md section 19.

The operator is imported unchanged from the accepted O2b implementation; the only training-side novelty is the
canonical coordinate. GT R may enter the coordinate construction and nothing else.
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
from ppg2ecg.evaluation import o2b_warp as B
from ppg2ecg.evaluation import q1_corruption as Q
from ppg2ecg.flow import rhythm_transfer as RT

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2c_oracle_integer_grid"
OUT = ROOT / "outputs/o2c_canon_oracle_seed42"
TRAIN_SRC = (ROOT / "scripts/o2c_train.py").read_text()
AUDIT_SRC = (ROOT / "scripts/o2c_train_corpus_audit.py").read_text()
GUARD_SRC = (ROOT / "scripts/o2c_stage0_regression.py").read_text()
EVAL_SRC = (ROOT / "scripts/o2c_evaluate.py").read_text()
PREREG = (ROOT / "docs/O2C_ORACLE_INTEGER_GRID_MEANFLOW_PREREGISTRATION.md").read_text()
O2PRE = (ROOT / "docs/O2_ORACLE_EVENT_CANONICALIZATION_PREREGISTRATION.md").read_text()
SRCS = {"train": TRAIN_SRC, "audit": AUDIT_SRC, "guard": GUARD_SRC, "eval": EVAL_SRC}
IDENT = json.loads((ART / "operator_identity.json").read_text())
R5 = np.array([120, 340, 551, 790, 1000])
RNG = np.random.default_rng(11)


def _code(src: str) -> str:
    """Source with every docstring removed, so token scans cannot be satisfied by prose."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


CODE = {k: _code(v) for k, v in SRCS.items()}


# --------------------------------------------------------------------------------- repository
def test_firewall_pins_a4_checkpoint_and_c2_untouched():
    for sub, pin in (("external/PENGUIN", "6cd70cdefb91f10efeb8dce34019b5067cb25344"),
                     ("external/iMeanFlow", "bf60cd7cb653f6628e59d48034b333c5eba445e2")):
        head = subprocess.run(["git", "-C", str(ROOT / sub), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert head == pin, f"{sub} moved: {head}"
    with pytest.raises(ER.WildPPGTestFirewallError):
        ER.assert_no_test_subjects(("an0", "kjd"))
    for name, src in SRCS.items():
        assert "kjd" not in src and "ssx" not in src, name
        assert "assert_no_test_subjects" in src, name
    assert hashlib.md5((ROOT / "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt").read_bytes()).hexdigest() == \
        "31c042d291052fbb6dc15263ad316be2"
    assert not list((ROOT / "outputs").glob("*c2*"))


def test_training_never_touches_validation_or_test_subjects():
    assert 'assert not (set(split["train"]) & {"an0", "k2s"})' in TRAIN_SRC
    assert 'assert not (set(split["train"]) & {"an0", "k2s"})' in AUDIT_SRC
    for bad in ("checkpoint_best", "patience", "val_loader", "best_epoch", "min_delta", "no_improve",
                "load_arrays(ROOT / PROCESSED, split['val']"):
        assert bad not in CODE["train"], bad
    assert '"validation_loaded": False' in TRAIN_SRC and '"validation_selection": False' in TRAIN_SRC
    assert '"early_stopping": False' in TRAIN_SRC


# --------------------------------------------------------------------------------- operator identity
def test_operator_source_is_byte_identical_to_the_accepted_o2b_implementation():
    for key in ("o2b_warp", "o2_warp"):
        p = ROOT / IDENT["files"][key]["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == IDENT["files"][key]["sha256"], key
        blob = subprocess.run(["git", "-C", str(ROOT), "hash-object", str(p)],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert blob == IDENT["files"][key]["git_blob"], key
    assert IDENT["modification"].startswith("none")
    assert IDENT["config"] == {"ANCHOR_W": B.ANCHOR_W, "MIN_BEATS": B.MIN_BEATS, "EPS": O2.EPS,
                               "MIN_INT_SPACING": B.MIN_INT_SPACING, "CORE_OFFSET_TOL": B.CORE_OFFSET_TOL,
                               "FS": 128, "T_LEN": 1024,
                               "rounding": IDENT["config"]["rounding"], "resampler": IDENT["config"]["resampler"],
                               "identity_rows": IDENT["config"]["identity_rows"],
                               "post_warp_renormalisation": False, "amplitude_jacobian": False}


def test_o2c_scripts_import_the_operator_and_redefine_nothing():
    for name in ("train", "eval", "audit", "guard"):
        assert "o2b_warp" in SRCS[name], name
    for name, code in CODE.items():
        for bad in ("def canonical_positions", "def build_int_anchors", "def round_half_to_even", "def spacing_ok",
                    "class IntegerEventWarp", "class EventWarp", "grid_sample", "ANCHOR_W =", "MIN_INT_SPACING =",
                    "isotonic", "np.clip", "interp1d", "sinc", "spline"):
            assert bad not in code, f"{name}: {bad}"
    assert "BW.IntegerEventWarp(" in TRAIN_SRC and "O2W.apply_warp(" in TRAIN_SRC
    assert "BW.IntegerEventWarp(" in EVAL_SRC and "O2W.apply_warp(" in EVAL_SRC
    assert B.MIN_INT_SPACING == 21 and B.ANCHOR_W == 10 and B.CORE_OFFSET_TOL == 1e-6


def test_integer_schedule_rounding_inverse_and_identity_are_the_accepted_ones():
    w = B.IntegerEventWarp(R5)
    assert not w.identity and w.valid()
    q = np.asarray(w.q, np.int64)
    assert q[0] == R5[0] and q[-1] == R5[-1] and np.all(np.diff(q) >= B.MIN_INT_SPACING)
    np.testing.assert_array_equal(B.round_half_to_even([0.5, 1.5, 2.5, -0.5]), np.array([0, 2, 2, 0]))
    np.testing.assert_allclose(w.core_slopes(), 1.0, atol=1e-12)
    assert w.core_offsets().max() <= B.CORE_OFFSET_TOL
    np.testing.assert_allclose(w.inverse(np.asarray(w.q, float)), R5.astype(float), atol=1e-9)
    x = torch.from_numpy(RNG.standard_normal((2, 1, 1024)).astype(np.float32))
    warps = [B.IntegerEventWarp(np.array([100, 500])), w]                  # K < 3 -> identity, bit-exact
    out = O2.apply_warp(x, warps, "to_canonical")
    assert torch.equal(out[0], x[0]) and not torch.equal(out[1], x[1])


# --------------------------------------------------------------------------------- train-corpus audit
def test_train_audit_covers_the_whole_corpus_with_the_frozen_stop_rules():
    assert "load_arrays(ROOT / PROCESSED, split[\"train\"], None)" in AUDIT_SRC     # no limit, no subsample
    assert "K3_BUDGET = 0.005" in AUDIT_SRC
    assert 'out["stop_A_invalid_warp"]' in AUDIT_SRC and 'out["stop_B_k3_fraction"]' in AUDIT_SRC \
        and 'out["stop_C_spacing_violation"]' in AUDIT_SRC
    assert 'raise RuntimeError("TRAIN-CORPUS CANONICALIZATION INVALID (STOP)")' in AUDIT_SRC
    for bad in ("except", "continue  # skip", "drop", "repair", "W =", "ANCHOR_W ="):
        assert bad not in CODE["audit"], bad
    assert "MeanFlowS5" not in AUDIT_SRC and "backward" not in AUDIT_SRC


def test_audit_reports_k_lt_3_spacing_and_integer_core_facts():
    from importlib import util
    sp = util.spec_from_file_location("o2c_audit_t", ROOT / "scripts/o2c_train_corpus_audit.py")
    m = util.module_from_spec(sp); sp.loader.exec_module(m)
    K2 = m.audit_one(np.array([100, 500]))
    assert K2[1] == 1 and K2[2] == "K<3"
    dense = m.audit_one(np.arange(0, 200, 15))
    assert dense[1] == 1 and dense[2] == "integer spacing violated"
    ok = m.audit_one(R5)
    assert ok[1] == 0 and ok[3] >= B.MIN_INT_SPACING and ok[4] <= B.CORE_OFFSET_TOL and ok[6] == 1


def test_stage0_regression_guard_reuses_the_exact_o2_gate_and_o2b_reference():
    assert 'spec_from_file_location("o2_stage0_roundtrip"' in GUARD_SRC
    assert "S0.roundtrip_metrics(" in GUARD_SRC and "S0.load_cohort()" in GUARD_SRC
    assert "O2.roundtrip_gate(med)" in GUARD_SRC
    assert 'O2BART / "decision.json"' in GUARD_SRC and "REF_TOL = 1e-12" in GUARD_SRC
    assert 'VERDICT_FAIL = "OPERATOR REGRESSION DETECTED"' in GUARD_SRC and "OPERATOR REGRESSION DETECTED" in PREREG
    assert "MeanFlowS5" not in GUARD_SRC and "backward" not in GUARD_SRC
    assert O2.ROUNDTRIP_GATE == {"raw_rmse": 0.020, "T6": 0.020, "T7": 0.020, "T4": 0.020, "T8": 0.020,
                                 "f1_at_50": 0.98, "beat_count_diff": 0}
    ref = json.loads((ROOT / "artifacts/o2b_integer_grid/decision.json").read_text())["medians"]
    assert abs(ref["raw_rmse"] - 0.001696) < 5e-7 and ref["T4"] == 0.0 and ref["T6"] == 0.0 and ref["T8"] == 0.0
    assert ref["f1_at_50"] == 1.0 and ref["beat_count_diff"] == 0.0


# --------------------------------------------------------------------------------- leakage boundary
def test_gt_r_enters_only_the_coordinate_and_never_a_model_input_or_loss():
    for name in ("train", "eval"):
        code = CODE[name]
        for bad in ("phase", "event_map", "r_channel", "peak_channel", "torch.cat", "aux_loss", "qrs_loss",
                    "event_loss", "contrastive", "snap", "shift_pred", "np.roll"):
            assert bad not in code, f"{name}: {bad}"
    # the generator sees exactly one conditioning tensor: the (canonical) PPG
    assert "imeanflow_loss(net, ecg_c.unsqueeze(1), ppg_c.unsqueeze(1), e, t, r," in TRAIN_SRC
    assert "sample_meanflow_schedule(net, pp, bank[i:i + BATCH].to(dev), ER.UNIFORM[NFE])" in EVAL_SRC
    assert "make_ppg2" not in TRAIN_SRC                                     # no event field anywhere in O2c training
    assert EVAL_SRC.count("make_ppg2") == 1 and "RT.make_ppg2(pp, sf)" in EVAL_SRC.split("# ---------------- 27.")[1]
    # the only consumer of the cached GT R schedule is the warp constructor
    assert "BW.IntegerEventWarp(p) for p in pks" in TRAIN_SRC and "canonicalize(x, y, pks, dev)" in TRAIN_SRC
    for ln in TRAIN_SRC.splitlines() + EVAL_SRC.splitlines():
        if "pks" in ln or "gt_pk" in ln:
            for bad in ("net(", "imeanflow_loss", "torch.randn", "sample_meanflow_schedule", "unsqueeze(1), e,"):
                assert bad not in ln, ln.strip()


def test_both_modalities_share_one_map_and_the_source_is_canonical():
    assert 'O2W.apply_warp(xb, wl, "to_canonical")' in TRAIN_SRC and 'O2W.apply_warp(yb, wl, "to_canonical")' in TRAIN_SRC
    assert "e = torch.randn(Bc, 1, ecg_c.shape[1], device=dev)" in TRAIN_SRC
    assert "source drawn IN canonical coordinates" in TRAIN_SRC
    x = torch.from_numpy(RNG.standard_normal((1, 1, 1024)).astype(np.float32))
    w = [B.IntegerEventWarp(R5)]
    fwd = O2.apply_warp(x, w, "to_canonical")
    back = O2.apply_warp(fwd, w, "to_raw")
    assert torch.allclose(back, O2.round_trip(x, w))


# --------------------------------------------------------------------------------- model / training
def test_model_objective_optimizer_and_rng_match_the_c1_arm_b_replay():
    meta = json.loads((ROOT / "outputs/c1_imf_baseline_replay_seed42/train_meta.json").read_text())
    a = meta["args"]
    from importlib import util
    sp = util.spec_from_file_location("o2c_train_t", ROOT / "scripts/o2c_train.py")
    m = util.module_from_spec(sp); sp.loader.exec_module(m)
    for k in ("seed", "batch_size", "micro_batch", "lr", "weight_decay", "val_every_steps", "p_mean", "p_std",
              "data_proportion", "norm_p", "norm_eps", "jvp_mode", "cond_mode", "h_scale", "sample_rate",
              "h_dim", "ssm_ratio", "mlp_ratio", "c1_arm"):
        assert m.CFG[k] == a[k], k
    assert m.CFG["blocks"] == a["blocks"]
    assert meta["params"]["total"] == 4_568_707
    assert "seed_everything(CFG[\"seed\"], deterministic=True)" in TRAIN_SRC
    assert "gen.manual_seed(CFG[\"seed\"])" in TRAIN_SRC and "tr_gen.manual_seed(CFG[\"seed\"] + 1)" in TRAIN_SRC
    assert "torch.optim.AdamW(net.parameters(), lr=CFG[\"lr\"], weight_decay=CFG[\"weight_decay\"])" in TRAIN_SRC
    assert "batch_rounds(loader, CFG[\"val_every_steps\"])" in TRAIN_SRC
    assert "sample_tr_c1(Bc, tr_gen, arm=CFG[\"c1_arm\"], **tr_kw)" in TRAIN_SRC
    assert "(loss * (Bc / B)).backward()" in TRAIN_SRC and "opt.step()" in TRAIN_SRC
    for bad in ("scheduler", "lr_scheduler", "ema_", "clip_grad", "autocast", "GradScaler", "attention", "adapter",
                "Parameter(", "nn.Linear", "dropout"):
        assert bad not in CODE["train"], bad


def test_exactly_the_compute_matched_step_count_and_a_fresh_seed_42_init():
    res = json.loads((ART / "baseline_step_resolution.json").read_text())
    assert res["exact_optimizer_steps_at_best_round"] == 10046 and res["exact_optimizer_steps_total"] == 14409
    assert "steps_target != 10046" in TRAIN_SRC and "budget = 100 if args.preflight else steps_target" in TRAIN_SRC
    assert "10,046" in PREREG
    assert 'raise RuntimeError(f"parameter count {params[\'total\']} != frozen B 4568707")' in TRAIN_SRC
    assert "initialization_hash.json" in TRAIN_SRC and "init_sha = state_sha(net.state_dict())" in TRAIN_SRC
    for bad in ("load_state_dict", "checkpoint_last", "resume"):
        assert bad not in CODE["train"], bad
    assert "checkpoint_final.pt" in TRAIN_SRC
    assert "checkpoint_frac" not in TRAIN_SRC                              # final checkpoint only


def test_preflight_budget_is_the_frozen_three_gpu_hours():
    assert '"budget_gpu_hours": 3.0' in TRAIN_SRC and 'bool(proj_h > 3.0)' in TRAIN_SRC
    assert "preflight state discarded" in TRAIN_SRC
    assert "3 GPU-h" in PREREG or "3.0" in PREREG


# --------------------------------------------------------------------------------- evaluation
def test_evaluation_uses_the_frozen_cohort_source_bank_and_nfe():
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    for s, n_total in (("an0", 22183), ("k2s", 27017)):
        got = ER.select_subset("x4-event-nfe-v2", s, n_total, 1024)
        np.testing.assert_array_equal(got, np.asarray(frozen[s], dtype=got.dtype))
    assert "S0.load_cohort()" in EVAL_SRC and "len(X) != 2048 or n_beats != 19834" in EVAL_SRC
    assert "FS, T_LEN, BATCH, NFE, SRC_SEED = 128, 1024, 64, 4, 0" in EVAL_SRC
    e0 = torch.randn(2048, 1, 1024, generator=torch.Generator().manual_seed(0))
    assert hashlib.sha256(e0.numpy().tobytes()).hexdigest() == \
        "868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f"
    assert "SRC_BANK_SHA" in EVAL_SRC and "ER.UNIFORM[NFE]" in EVAL_SRC and "assert got == {NFE}" in EVAL_SRC


def test_baseline_regression_gate_is_the_frozen_row_at_the_preregistered_tolerance():
    assert "BASELINE_TOL = 1e-6" in EVAL_SRC
    frozen = {"f1": 0.4367, "chance_f1": 0.1192, "f1_excess": 0.3176, "missing": 0.5662, "spurious": 0.5154,
              "beats_ratio_dev": 0.1067}
    from importlib import util
    sp = util.spec_from_file_location("o2c_eval_t", ROOT / "scripts/o2c_evaluate.py")
    src = ast.parse(EVAL_SRC)
    got = {}
    for node in ast.walk(src):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FROZEN_B":
            got = ast.literal_eval(node.value)
    assert got and set(got) == set(frozen)
    for k, v in frozen.items():
        assert abs(got[k] - v) <= 5e-5, k
    assert 'raise RuntimeError(f"BASELINE REGRESSION FAILED (STOP): {bad}")' in EVAL_SRC
    assert sp is not None


def test_o1_functionals_and_train_iqrs_are_the_frozen_ones():
    scal = json.loads((ROOT / "artifacts/o1_component_extractability/target_scaling.json").read_text())["targets"]
    for t, v in (("median_QRS_p2p", 0.5053170621395111), ("median_QRS_max_abs_derivative", 0.22995377704501152),
                 ("median_QRS_curvature_energy", 0.033796727107052275), ("median_QRS_width_ms", 31.25)):
        assert scal[t]["scale_train_IQR"] == v, t
        assert t in OT.TARGETS
    assert "scale_train_IQR" in EVAL_SRC and "S0._targets" in EVAL_SRC
    assert "ALIGNED = S0.ALIGNED" in EVAL_SRC
    assert OT.TARGET_IDS["median_QRS_p2p"] == "T4" and OT.TARGET_IDS["median_QRS_width_ms"] == "T8"


def test_no_prediction_shift_or_oracle_post_processing_in_evaluation():
    for bad in ("np.roll", "snap", "align_to_gt", "shift", "argmax_align", "median_filter", "savgol",
                "gt_pk[i] +", "replace_peaks", "post_process"):
        assert bad not in CODE["eval"], bad
    assert 'warp_block(can, warps, "to_raw", dev)' in EVAL_SRC          # the ONLY transform applied to O2c output
    assert CODE["eval"].count("'to_raw'") == 1


def test_bootstrap_is_ecg_window_clustered_and_subject_stratified():
    assert 'CLUSTER = np.array([f"{a}|{b}" for a, b in zip(SUB, WI)])' in EVAL_SRC
    assert "O1E.cluster_bootstrap(" in EVAL_SRC
    assert OT.BOOT_N == 2000 and OT.BOOT_SEED == 20260903
    assert "cluster_bootstrap(d, SUB, CLUSTER)" in EVAL_SRC


def test_multisource_uses_the_exact_q1_512_cohort_and_seeds_0_to_7():
    assert Q.UNC_SEEDS == tuple(range(8)) and Q.N_UNCERTAINTY_PER_STRATUM == 64
    assert "Q.uncertainty_positions(SUB, SITE, WI)" in EVAL_SRC and "assert len(unc) == 512" in EVAL_SRC
    assert "Q.uncertainty_from_samples(" in EVAL_SRC and "for sd in Q.UNC_SEEDS" in EVAL_SRC
    assert "manual_seed(int(sd))" in EVAL_SRC


def test_secondary_arms_are_frozen_and_labelled_as_leakage_diagnostics():
    p = ROOT / "outputs/r3_gtf_oracle_seed42/module_step2200.pt"
    assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest().startswith("24dbeaf65f7e")
    assert 'gmod_sd["arm"] == "gtf_oracle"' in EVAL_SRC
    assert 'gmod_sd["generator_state_sha256"] == gmeta["state_dict_sha256"]' in EVAL_SRC
    assert RT.EXPECTED_GENERATOR_STATE_SHA.startswith("47d7ccb9")
    assert hashlib.sha256((ROOT / RT.GENERATOR_CKPT).read_bytes()).hexdigest().startswith("557c7054")
    assert "TARGET LEAKAGE DIAGNOSTIC" in EVAL_SRC
    assert "requires_grad_(False)" in EVAL_SRC and EVAL_SRC.count("requires_grad_(False)") >= 3


def test_operator_floor_is_reported_and_never_subtracted():
    assert "NEVER subtracted from any generator error" in EVAL_SRC
    for bad in ("- floor", "-= floor", "floor_corrected", "minus_floor"):
        assert bad not in CODE["eval"], bad
    assert "It is **never** subtracted from generator error" in PREREG


# --------------------------------------------------------------------------------- decision
def test_gates_are_the_frozen_o2_ones():
    assert O2.NONINF_MARGIN == 0.020 and O2.F1_EXCESS_MIN == 0.10
    assert 'BT["f1_excess"]["lo"] > 0 and BT["f1_excess"]["point"] >= O2W.F1_EXCESS_MIN' in EVAL_SRC
    assert "-O2W.NONINF_MARGIN" in EVAL_SRC
    assert "O2W.decide_o2(j)" in EVAL_SRC
    for tok in ("J1", "J2", "J3", "J4", "J5", "J6", "J7", "S1", "S2"):
        assert tok in PREREG and tok in O2PRE
    for tok in ("0.10", "0.020"):
        assert tok in PREREG


def test_verdict_tree_is_the_frozen_o2_code():
    ok = {f"J{i}": True for i in range(1, 8)}
    assert O2.decide_o2(ok)["verdict"] == O2.VERDICT_A
    assert O2.decide_o2(dict(ok, J7=False))["verdict"] == O2.VERDICT_B
    assert O2.decide_o2(dict(ok, J1=False, morphology_improves=True))["verdict"] == O2.VERDICT_C
    assert O2.decide_o2({f"J{i}": False for i in range(1, 8)})["verdict"] == O2.VERDICT_D
    for v in (O2.VERDICT_A, O2.VERDICT_B, O2.VERDICT_C):
        assert v in PREREG, v
    # verdict D: the O2c preregistration restates it as "... ORACLE FACTORIZATION BENEFIT"; the executed code is
    # the frozen O2 string, and section 15 fixes precedence ("the frozen O2 preregistration wins").
    assert O2.VERDICT_D == "NO MATERIAL ORACLE CANONICALIZATION BENEFIT"
    assert "NO MATERIAL ORACLE FACTORIZATION BENEFIT" in PREREG
    assert "the frozen O2\npreregistration wins" in PREREG
    assert "PRETRAIN STOP — OPERATOR REGRESSION DETECTED" in PREREG
    assert "PRETRAIN STOP — TRAIN-CORPUS CANONICALIZATION INVALID" in PREREG


def test_secondary_analyses_cannot_alter_the_primary_verdict():
    assert '"secondary_analyses_cannot_alter_the_verdict": True' in EVAL_SRC
    j = ast.parse(EVAL_SRC)
    body = EVAL_SRC.split("j = {")[1].split("}\n")[0]
    for bad in ("canonical", "site", "gtf", "GTF", "mac_can", "f150"):
        assert bad not in body, bad
    assert j is not None
    assert "never alter the primary verdict" in PREREG or "never alter" in PREREG


def test_claim_boundaries_are_stated_in_the_preregistration():
    for phrase in ("NOT deployable", "oracle", "problem-discovery"):
        assert phrase.lower() in PREREG.lower(), phrase
    assert "confidence-calibrated" not in PREREG
    flat = " ".join(O2PRE.split())
    for banned in ("PPG-derived phase solves the problem", "event timing is deterministic from PPG",
                   "WHAT should be stochastic and WHEN deterministic"):
        assert banned in flat, banned
