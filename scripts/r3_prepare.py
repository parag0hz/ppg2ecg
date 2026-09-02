"""R3 pre-training preparation (prereg 3d779fc sections 2, 4, 6-9): frozen-checkpoint manifest with all hashes,
parameter manifest, per-arm initialisation hashes, the R2 shuffle manifest (copied, re-asserted), the R2
oracle-cache and loader-order provenance (verified, not duplicated). No optimizer step, no data window read.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.flow import rhythm_fusion as RF
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/r3_rhythm_fusion"
R2ART = ROOT / "artifacts/r2_rhythm_transfer"
VAL = ("an0", "k2s")


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    git = git_sha(ROOT)
    # ---- frozen checkpoints ----
    ck = torch.load(ROOT / RT.GENERATOR_CKPT, map_location="cpu", weights_only=False)
    gsha = RT.state_dict_sha256(ck["state_dict"]); assert gsha == RT.EXPECTED_GENERATOR_STATE_SHA
    ckt = torch.load(ROOT / RT.RHYTHM_CKPT, map_location="cpu", weights_only=False)
    tsha = RT.state_dict_sha256(ckt["state_dict"]); assert tsha == RT.EXPECTED_RHYTHM_STATE_SHA
    assert RT.file_sha256(ROOT / RT.GENERATOR_CKPT).startswith(RF.EXPECTED_GENERATOR_FILE_SHA_PREFIX)
    assert RT.file_sha256(ROOT / RT.RHYTHM_CKPT).startswith(RF.EXPECTED_RHYTHM_FILE_SHA_PREFIX)
    r2prov = json.loads((R2ART / "provenance.json").read_text())
    assert r2prov["adapters"]["true"].endswith(RF.R2_ADD_CKPT) and r2prov["adapters"]["oracle"].endswith(RF.R2_ORACLE_CKPT), "R2 adapter paths differ from R2 provenance"
    r2 = {}
    for name, path, prefix in (("r2_add", RF.R2_ADD_CKPT, RF.EXPECTED_R2_ADD_STATE_SHA_PREFIX), ("r2_oracle", RF.R2_ORACLE_CKPT, RF.EXPECTED_R2_ORACLE_STATE_SHA_PREFIX)):
        ad = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        ssha = RT.state_dict_sha256(ad["state_dict"])
        assert ssha.startswith(prefix), (name, ssha)
        assert ad["step"] == RT.STEPS and ad["generator_state_sha256"] == gsha and ad["rhythm_state_sha256"] == tsha
        fsha = RT.file_sha256(ROOT / path)
        assert fsha.startswith(RF.EXPECTED_R2_ADD_FILE_SHA_PREFIX if name == "r2_add" else RF.EXPECTED_R2_ORACLE_FILE_SHA_PREFIX), (name, fsha)
        r2[name] = {"path": path, "file_sha256": fsha, "state_dict_sha256": ssha, "arm": ad["arm"], "step": ad["step"],
                    "adapter_l2": float(torch.linalg.vector_norm(ad["state_dict"]["proj.weight"])), "git": ad["git"]["commit"], "probe_hash": ad["probe_hash"]}
    man = {"generator": {"path": RT.GENERATOR_CKPT, "file_sha256": RT.file_sha256(ROOT / RT.GENERATOR_CKPT), "state_dict_sha256": gsha, "round": int(ck["epoch"]),
                         "n_params": int(sum(v.numel() for v in ck["state_dict"].values()))},
           "rhythm_tcn": {"path": RT.RHYTHM_CKPT, "file_sha256": RT.file_sha256(ROOT / RT.RHYTHM_CKPT), "state_dict_sha256": tsha, "params": int(ckt["params"]), "rf": int(ckt["rf"])},
           **r2,
           "r2_shuffle_manifest": {"path": "artifacts/r2_rhythm_transfer/shuffle_manifest.csv", "file_sha256": RT.file_sha256(R2ART / "shuffle_manifest.csv")},
           "r2_oracle_cache": {"path": "artifacts/r2_rhythm_transfer/_cache_oracle_train.npz", **json.loads((R2ART / "cache_build.json").read_text())},
           "r2_loader_order": json.loads((R2ART / "loader_order_provenance.json").read_text()), "r2_probe_hash": RF.R2_PROBE_HASH,
           "git": git, "prereg": "3d779fc", "utc": datetime.now(timezone.utc).isoformat()}
    (ART / "frozen_checkpoint_manifest.json").write_text(json.dumps(man, indent=2, default=str))
    print(f"[M] generator {gsha[:16]} round {man['generator']['round']} | tcn {tsha[:16]} | r2 add {r2['r2_add']['state_dict_sha256'][:16]} | r2 oracle {r2['r2_oracle']['state_dict_sha256'][:16]}", flush=True)

    # ---- parameter manifest + initialisation hashes (fixed construction order, seed 42) ----
    h_dim = int(ck["model_cfg"]["h_dim"])
    arms, fam_hash = {}, {}
    for arm in RF.TRAINED_ARMS:
        m = RF.build_r3_module(RF.ARM_FAMILY[arm], RF.ARM_GATE_MODE[arm], c_hidden=h_dim, seed=RF.SEED)
        arms[arm] = {"family": RF.ARM_FAMILY[arm], "gate_mode": RF.ARM_GATE_MODE[arm], "n_params": RF.n_params(m),
                     "full": RF.params_sha256(m), "fusion_subset": RF.params_sha256(m, only_prefix="fusion."),
                     "gate_subset": RF.params_sha256(m, only_prefix="gate.") if m.gate is not None else None,
                     "param_names": list(RF.param_names(m))}
        fam_hash.setdefault(RF.ARM_FAMILY[arm], set()).add(arms[arm]["full"])
    assert all(len(v) == 1 for v in fam_hash.values()), "initialisation differs within a family"
    assert len({arms[a]["fusion_subset"] for a in arms}) == 1, "fusion subset differs across families"
    tf, gtf = arms["tf_true"], arms["gtf_true"]
    assert tf["n_params"] == RF.EXPECTED_TF_PARAMS and gtf["n_params"] == RF.EXPECTED_GTF_PARAMS
    assert tuple(tf["param_names"]) == RF.TF_PARAM_NAMES and tuple(gtf["param_names"]) == RF.GTF_PARAM_NAMES
    (ART / "initialization_hashes.json").write_text(json.dumps({"seed": RF.SEED, "construction_order": "fusion(tok, q, attn.q/k/v/o, out) then gate", "arms": arms}, indent=2))
    (ART / "parameter_manifest.json").write_text(json.dumps({
        "generator_params_state_dict": RF.GENERATOR_PARAMS, "generator_params_effective": 4_304_513, "rhythm_tcn_params": int(ckt["params"]),
        "ADD": {"trainable": 128, "names": ["rhythm_adapter.proj.weight"], "fusion": "additive 1x1 at pre_conv_ppg output (R2, frozen)"},
        "TF": {"trainable": tf["n_params"], "names": tf["param_names"], "fusion": "target-side cross-attention at z_e", "gate": None,
               "ratio_state_dict": tf["n_params"] / RF.GENERATOR_PARAMS, "ratio_effective": tf["n_params"] / 4_304_513, "budget_ok": tf["n_params"] <= RF.PARAM_BUDGET and tf["n_params"] / RF.GENERATOR_PARAMS < RF.PARAM_BUDGET_FRAC},
        "GTF": {"trainable": gtf["n_params"], "names": gtf["param_names"], "fusion": "target-side cross-attention at z_e", "gate": "adaptive rhythm gate (81 params; CONST control identical)",
                "gate_params": gtf["n_params"] - tf["n_params"], "gate_budget_ok": (gtf["n_params"] - tf["n_params"]) < RF.GATE_PARAM_BUDGET},
        "GTF_TRUE_eq_GTF_CONST_eq_GTF_SHUFFLE_eq_GTF_ORACLE": len({arms[a]["n_params"] for a in ("gtf_true", "gtf_const", "gtf_shuffle", "gtf_oracle")}) == 1}, indent=2))
    print(f"[P] TF {tf['n_params']} GTF {gtf['n_params']} gate {gtf['n_params']-tf['n_params']} | init hashes: tf {tf['full'][:12]} gtf {gtf['full'][:12]} fusion-subset shared {tf['fusion_subset'][:12]}", flush=True)

    # ---- shuffle manifest: copy R2's and re-assert every population against the rule ----
    rows = list(csv.DictReader(open(R2ART / "shuffle_manifest.csv")))
    for pop, key, n_exp in (("train", "partner_train_row", 293271), ("eval", "partner_pop_row", 2048), ("viz", "partner_pop_row", 64)):
        rs = [r for r in rows if r["population"] == pop]
        assert len(rs) == n_exp, (pop, len(rs))
        sub = np.array([r["subject"] for r in rs]); site = np.array([r["site"] for r in rs]); wi = np.array([int(r["window_index"]) for r in rs])
        partner = np.array([int(r[key]) for r in rs]); RT.assert_derangement(partner)
        assert np.array_equal(partner, RT.shuffle_partner(sub, site, wi)), pop
    shutil.copyfile(R2ART / "shuffle_manifest.csv", ART / "shuffle_manifest.csv")
    print("[S] R2 shuffle manifest re-asserted (train 293271 / eval 2048 / viz 64) and copied", flush=True)

    # ---- oracle cache sha (verified; not duplicated) ----
    field = np.load(R2ART / "_cache_oracle_train.npz")["field"]
    sha = hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()
    assert sha == man["r2_oracle_cache"]["oracle_cache_sha256"], "oracle cache sha mismatch"
    del field
    (ART / "prepare_provenance.json").write_text(json.dumps({"git": git, "prereg": "3d779fc", "oracle_cache_sha256": sha, "utc": datetime.now(timezone.utc).isoformat(),
                                                              "test_subjects_loaded": []}, indent=2))
    print(f"[done] R3 prepared; oracle cache {sha[:16]} verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
