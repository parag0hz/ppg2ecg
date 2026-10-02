"""DP3 — MIMIC-BP ECG-target-blind cross-task external validation of the frozen DualReadout-ECG S1 MIDDLE
(docs/DP3_MIMICBP_TARGET_BLIND_PREREGISTRATION.md).

Stages (exact order): target_audit (A), integrity (B), train <point|gen|dual> <43|44> (C), internal (D), compute (E),
interface (F), freeze (G), eval_external (H), summarize (I).
Seed 42 = the frozen DP0 checkpoints (outputs/dp0_dualreadout). Seeds 43 / 44 = the DP0 protocols with only the seed changed,
trained on DP-TRAIN only. MIMIC-BP waveform values are read only by eval_external, after the committed final freeze.
The frozen DP0 module (scripts/dp0_dualreadout.py) is imported and never modified.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/dp3_mimicbp.py <stage> [args]
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import gc
import hashlib
import json
import re
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bf0_run as B  # noqa: E402
import c0_coherentbeat as C0  # noqa: E402
import dp0_dualreadout as F  # noqa: E402
from ppg2ecg.anchorflow import fastfd as FF  # noqa: E402
from ppg2ecg.beatfirst import render as BR  # noqa: E402
from ppg2ecg.coherentbeat import ablation as AB  # noqa: E402
from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows  # noqa: E402
from ppg2ecg.dualreadout import model as D  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as RP  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import RhythmTCN  # noqa: E402
from ppg2ecg.rhythmfield import model as RM  # noqa: E402
from ppg2ecg.scaleflow import model as SM  # noqa: E402

PREREG = "docs/DP3_MIMICBP_TARGET_BLIND_PREREGISTRATION.md"
CODE_FILES = ("scripts/dp3_mimicbp.py", "tests/test_dp3_mimicbp.py")
ART = ROOT / "artifacts/dp3_mimicbp"
OUT = ROOT / "outputs/dp3_mimicbp"
DP0_OUT = ROOT / "outputs/dp0_dualreadout"
DP0_ART = ROOT / "artifacts/dp0_dualreadout"
FREEZE = ART / "final_freeze_manifest.json"
MIMIC = ROOT / "data/raw/MIMIC-BP"
SEEDS = (42, 43, 44)
NEW_SEEDS = (43, 44)
ROLES = ("point", "gen", "S1")
FS, T = 128, 512
FS_RAW, N_SEG, SEG_LEN = 125, 30, 3750
WIN_S = 4
WIN_RAW = FS_RAW * WIN_S                      # 500 samples per 4 s window at 125 Hz
WIN_PER_SEG = SEG_LEN // WIN_RAW              # 7 windows per 30 s segment; the last 2 s are dropped
HR_RANGE = (30.0, 200.0)                      # frozen V1 / DP0 reference-ECG validity rule
BOOT_N, BOOT_SEED, SHUF_SEED = 2000, 20261002, 20261002
N_K16, K, K16_SALT = 2000, 16, "dp3-k16-v1"
EVIDENCE_LABEL = "ECG-target-blind cross-task external cohort"
DISCLOSURE = ("MIMIC-BP PPG had previously been used in unrelated PPG-to-ABP experiments, whereas ECG reconstruction targets were "
              "withheld from the PPG-to-ECG architecture-development process.")


def write_json(name, obj):
    p = ART / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(B.clean(obj), indent=1, ensure_ascii=False))


def read_json(name):
    return json.loads((ART / name).read_text())


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def ci(v, pid):
    return C0.cluster_ci(v, pid, BOOT_N, BOOT_SEED)


# ----------------------------------------------------------------------------------------------- checkpoints
def ckpt_file(role: str, seed: int) -> Path:
    """Seed 42 = the frozen DP0 checkpoints; seeds 43 / 44 = DP3 checkpoints."""
    name = {"point": "point", "gen": "gen", "S1": "S1"}[role]
    return DP0_OUT / f"{name}.pt" if seed == 42 else OUT / f"seed{seed}" / f"{name}.pt"


def build(role):
    return F.build({"point": "point", "gen": "gen", "S1": "S1"}[role])


def load_model(role, seed, dev):
    ck = torch.load(ckpt_file(role, seed), map_location="cpu")
    net = build(role)
    net.load_state_dict(ck["state_dict"])
    return net.to(dev).eval()


def load_detector(dev):
    ck = torch.load(DP0_OUT / "detector.pt", map_location="cpu")
    net = RhythmTCN()
    net.load_state_dict(ck["state_dict"])
    return net.to(dev).eval()


# ----------------------------------------------------------------------------------------------- Stage A: target-blind audit
ALIAS = re.compile(r"mimic[-_ ]?bp", re.I)
ECG_TERMS = re.compile(r"ecg|ppg2ecg|ppg-to-ecg|ppg→ecg|ppg->ecg|reconstruct|morpholog|r-peak|rpeak", re.I)
ECG_LOAD = re.compile(r"MIMIC-BP/ecg|_ecg\.npy|/ \"ecg\" /|/ 'ecg' /", re.I)
AUDIT_SELF = {"scripts/dp3_mimicbp.py", "tests/test_dp3_mimicbp.py", "scripts/dp2_dataset_audit.py", "tests/test_dp2_dataset_audit.py"}
SCAN_DIRS = ("docs", "scripts", "src", "artifacts", "configs", "data/manifests", "tests", "external/PENGUIN/src", "external/PENGUIN/config")
ADJUDICATED = {   # every same-line MIMIC-BP + ECG-term co-mention found by the scan, with its reading
    "docs/FIGURES_BY_STAGE.md": "figure index: 'MIMIC-BP (ABP)' next to a WildPPG ECG panel; MIMIC-BP target is ABP",
    "docs/D3_PENGUIN_SIX_DATASET_PREREGISTRATION.md": "reuse of frozen 'ECG/MIMIC-BP arms' = ECG-dataset arms and the MIMIC-BP ABP arms",
    "docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md": "MIMIC-BP EXCLUDED from the ECG benchmark: 'target is ABP, not ECG'",
    "docs/A7_ABP_DATASET_AUDIT.md": "dataset audit lists the channels (incl. ECG) of the PPG->ABP corpus; ABP processing only",
    "scripts/predict_a7.py": "'no ECG metrics (ABP metrics ...)' on MIMIC-BP",
    "scripts/preflight_a0.py": "target preprocessing switches to raw ABP when dataset == MIMIC-BP (ECG_KW only for ECG datasets)",
    "docs/PROGRAM_SUMMARY_ALL_STAGES.md": "figure data line: 'MIMIC-BP (ABP), WildPPG (global-z ECG)'",
    "docs/TECHNICAL_SPEC_MODEL_DATA_TRAINING_2026-09-03.md": "raw layout listing {ppg,abp,ecg,resp,labels}; loader reads ppg/abp/labels only",
    "src/ppg2ecg/data/mimicbp.py": "raw layout docstring; windows_for_subject loads ppg, abp and labels only",
    "docs/PROJECT_STATUS_SUMMARY_FOR_LLM.md": "A7 PPG->ABP results contrasted with ECG findings on other datasets",
    "src/ppg2ecg/evaluation/abp_metrics.py": "ABP metric definitions (MIMIC-BP label definition)",
    "scripts/build_processed_mimicbp.py": "imports the ppg/abp loader; builds PPG -> ABP windows",
    "docs/DP2_LOCAL_DATASET_AUDIT.md": "DP2 audit inventory (metadata)",
    "artifacts/dp2_external/primary_dataset_selection.json": "DP2 audit record (metadata)",
    "artifacts/dp2_external/table_dataset_inventory.csv": "DP2 audit record (metadata)",
    "artifacts/dp2_external/dataset_inventory.json": "DP2 audit record (metadata)",
    "artifacts/dp2_external/dataset_eligibility.json": "DP2 audit record (metadata)",
}
PRIOR_ABP_USE = [("84223f0", "A7 dataset audit + preregistration (PPG -> ABP, MIMIC-BP)"),
                 ("fed5b8c", "A7 result: one-step structural attenuation on ABP (train 1,100 / val 195 / test 229 subjects)"),
                 ("9ff77b5", "A8: ABP target-scale sensitivity"), ("2842c11", "U1: upstream PENGUIN incl. MIMIC-BP (ABP task)"),
                 ("e09ee6d", "U2: MIMIC-BP evaluated (ABP target)"), ("eddbe60", "EXP-D preregistration (MIMIC-BP SBP / DBP / MAP)"),
                 ("b1eb66a", "EXP-D ABP result (U1 PENGUIN ABP checkpoint, 190 test subjects)")]


def stage_target_audit(ex, dev):
    lines, ecg_lines, ecg_loads = [], [], []
    for d in SCAN_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            rel = str(p.relative_to(ROOT))
            if not p.is_file() or p.suffix.lower() not in (".py", ".md", ".json", ".csv", ".txt", ".yaml", ".yml", ".sh", ".toml") or rel in AUDIT_SELF:
                continue
            if rel.startswith("artifacts/dp3_mimicbp"):
                continue
            for i, ln in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if ALIAS.search(ln):
                    lines.append(rel)
                    if ECG_TERMS.search(ln):
                        ecg_lines.append({"file": rel, "line": i, "text": ln.strip()[:240]})
                if p.suffix == ".py" and ECG_LOAD.search(ln) and ALIAS.search(p.read_text(errors="replace")):
                    ecg_loads.append({"file": rel, "line": i, "text": ln.strip()[:240]})
    unadjudicated = sorted({x["file"] for x in ecg_lines} - set(ADJUDICATED))
    keys = {}
    for d in ("data/processed/mimicbp_8s", "data/processed/u2_mimicbp"):
        fs = sorted((ROOT / d).glob("*.npz"))
        keys[d] = {"files": len(fs), "array_names": sorted(set().union(*[set(zipfile.ZipFile(f).namelist()) for f in fs[:20]]))}
    up = (ROOT / "external/PENGUIN/src/utils/load_data.py").read_text()
    m = re.search(r"def load_MIMIC_BP\(.*?\n(.*?)\n\n", up, re.S)
    upstream_kinds = sorted(set(re.findall(r"MIMIC-BP/(\w+)/", m.group(0)))) if m else []
    loader_kinds = sorted(set(re.findall(r'raw / "(\w+)" / f"\{pid\}_', (ROOT / "src/ppg2ecg/data/mimicbp.py").read_text())))
    u2_target = re.search(r'"MIMIC-BP":\s*\("u2_mimicbp",\s*"(\w+)"', (ROOT / "scripts/u2_build.py").read_text()).group(1)
    outs = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "outputs").glob("*") if "mimic" in p.name.lower())
    out_ecg = sorted(str(p.relative_to(ROOT)) for o in outs for p in (ROOT / o).rglob("*") if "ecg" in p.name.lower()) if outs else []
    gitlog = subprocess.run(["git", "log", "--all", "--format=%h %s"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    mimic_commits = [ln[:160] for ln in gitlog if ALIAS.search(ln) or "mimic" in ln.lower()]
    found = bool(ecg_loads) or bool(unadjudicated) or bool(out_ecg) or "ecg" in upstream_kinds or "ecg" in loader_kinds or u2_target != "ABP" \
        or any("ecg" in n.lower() for k in keys.values() for n in k["array_names"])
    out = {"verdict": "FAIL" if found else "PASS", "prior_mimicbp_ecg_target_use": "FOUND" if found else "NONE",
           "evidence_label": EVIDENCE_LABEL if not found else "INELIGIBLE", "required_disclosure": DISCLOSURE,
           "method": "metadata-only: repository text scan, code-path inspection, processed-file array names (zip listing), output path names, git log; "
                     "no MIMIC-BP waveform value was read",
           "scan": {"dirs": SCAN_DIRS, "excluded_self": sorted(AUDIT_SELF), "files_mentioning_mimicbp": len(set(lines)),
                    "same_line_ecg_term_comentions": ecg_lines, "adjudication": ADJUDICATED, "unadjudicated_files": unadjudicated},
           "code_paths": {"project_loader_kinds (src/ppg2ecg/data/mimicbp.py)": loader_kinds, "upstream_penguin_loader_kinds": upstream_kinds,
                          "u2_build_target": u2_target, "python_lines_loading_mimicbp_ecg_arrays": ecg_loads},
           "processed_files": keys, "outputs_with_mimic_in_name": outs, "output_files_with_ecg_in_name_under_mimic_outputs": out_ecg,
           "git_commits_mentioning_mimic": mimic_commits, "prior_ppg_abp_use": [{"commit": c, "description": d} for c, d in PRIOR_ABP_USE],
           "checked_uses": {u: "NOT FOUND" if not found else "SEE EVIDENCE" for u in (
               "training", "model selection", "checkpoint selection", "architecture design", "loss design", "threshold selection",
               "reconstruction performance", "ECG morphology analysis", "ECG-derived event analysis", "PPG->ECG hypothesis formation")}}
    write_json("target_blind_audit.json", out)
    print(f"[dp3] target-blind audit: {out['verdict']} (ECG target use {out['prior_mimicbp_ecg_target_use']})", flush=True)
    if found:
        print("DP3 TARGET-BLIND STATUS: FAILED", flush=True)


# ----------------------------------------------------------------------------------------------- Stage B: frozen S1 integrity
def stage_integrity(ex, dev):
    acc = F.accounting()
    s1 = acc["S1"]
    net = F.build("S1")
    own = net.ownership()
    h = json.loads((DP0_ART / "checkpoint_hashes.json").read_text())
    hashes = {k: {"dp0": h[k], "now": B.sha256_file(DP0_OUT / f"{k}.pt")} for k in ("detector", "point", "gen", "S1")}
    prep_commits = subprocess.run(["git", "log", "--format=%h", "--", "src/ppg2ecg/data/preprocess.py", "src/ppg2ecg/evaluation/rpeaks.py"],
                                  cwd=ROOT, capture_output=True, text=True).stdout.split()
    det = json.loads((DP0_ART / "detector_config.json").read_text())
    checks = {"total_params": s1["total"] == 943_372, "shared_params": s1["shared"] == 246_720, "adapter_params": s1["adapters"] == 8_320,
              "separate_params": acc["separate_waveform_params"] == 1_191_910, "saving_2085": round(100 * s1["saving"], 2) == 20.85,
              "shared_blocks_1_6": len(net.shared_blocks) == 6 and [b.c1.dilation[0] for b in net.shared_blocks] == [1, 2, 4, 8, 16, 32],
              "private_blocks_7_8": len(net.point_tower) == len(net.flow_tower) == 2,
              "widths_71_30": (acc["widths"]["w_point"], acc["widths"]["w_flow"]) == (71, 30),
              "checkpoint_hashes": all(v["dp0"] == v["now"] for v in hashes.values()),
              "detector_rule": (F.DET["threshold"], F.DET["refractory"]) == (0.35, 32) and det["protocol"]["threshold"] == 0.35,
              "dp0_lock_freeze_intact": _dp0_freeze_ok(), "preprocess_unchanged_since_v1": prep_commits == ["a15b354"]}
    out = {"checks": checks, "pass": all(checks.values()), "accounting": s1, "separate": acc["separate_waveform_params"],
           "ownership_counts": {k: sum(v == k for v in own.values()) for k in ("shared", "point", "flow")}, "hashes": hashes,
           "preprocessing": {"file": "src/ppg2ecg/data/preprocess.py", "sha256": B.sha256_file(ROOT / "src/ppg2ecg/data/preprocess.py"),
                             "commits": prep_commits, "PPG_KW": PPG_KW, "ECG_KW": ECG_KW},
           "detector": {"threshold": F.DET["threshold"], "refractory": F.DET["refractory"], "sigma_ms": F.DET["sigma_ms"]}}
    write_json("s1_integrity_audit.json", out)
    print(f"[dp3] integrity: {'PASS' if out['pass'] else 'FAIL'} {json.dumps(checks)}", flush=True)
    if not out["pass"]:
        raise SystemExit("STOP: frozen S1 integrity mismatch")


def _dp0_freeze_ok():
    try:
        F.check_lock_freeze()
        return True
    except PermissionError:
        return False


# ----------------------------------------------------------------------------------------------- Stage C: seeds 43 / 44
def _saver(seed):
    def _save(name, net, secs, nan_steps, extra=None):
        d = OUT / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{name}.pt"
        meta = {"seed": seed, "train_seconds": secs, "n_params": D.n_params(net), "nan_steps": nan_steps,
                "peak_gpu_mem_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "optimizer": "AdamW",
                "git_commit": git_head()} | (extra or {})
        torch.save({"state_dict": net.state_dict()} | meta, f)
        meta["sha256"] = B.sha256_file(f)
        write_json(f"checkpoints/seed{seed}_{name}.json", meta)
    return _save


def config_spec(role):
    """Seed-independent training configuration of a role (hashed; seeds 43 / 44 must match DP0's)."""
    net = build(role)
    base = {"role": role, "class": type(net).__name__, "params": D.n_params(net), "protocol": F.PROTO, "training_role": "dp_train",
            "training_raster": "reference-R raster", "split_sha256": json.loads((DP0_ART / "split_hashes.json").read_text())["dp_train"]}
    if role == "S1":
        base |= {"share": "S1", "widths": F.widths(), "schedule": "round-robin, odd POINT->FLOW / even FLOW->POINT", "updates": [F.PROTO["steps"]] * 2}
    if role == "point":
        base |= {"loss": "L1"}
    if role == "gen":
        base |= {"loss": "flow-matching MSE", "solver": "Euler", "nfe": F.NFE}
    return base


def config_hash(role):
    return hashlib.sha256(json.dumps(config_spec(role), sort_keys=True, default=str).encode()).hexdigest()


def stage_train(ex, dev, role, seed):
    seed = int(seed)
    if seed not in NEW_SEEDS or role not in ("point", "gen", "dual"):
        raise SystemExit("STOP: only point / gen / dual for seeds 43 / 44")
    F.SEED = seed                                  # every DP0 trainer reads the module-level SEED (init, batch stream, FM noise)
    F.OUT = OUT / f"seed{seed}"
    F._save = _saver(seed)
    {"point": F.stage_train_point, "gen": F.stage_train_gen}.get(role, lambda e, d: F.stage_train_dual(e, d, "S1"))(ex, dev)
    name = {"point": "point", "gen": "gen", "dual": "S1"}[role]
    meta = read_json(f"checkpoints/seed{seed}_{name}.json")
    man = read_json("training_seed_manifest.json") if (ART / "training_seed_manifest.json").exists() else {"jobs": []}
    man["jobs"] = [j for j in man["jobs"] if not (j["seed"] == seed and j["role"] == name)] + [{
        "seed": seed, "role": name, "git_commit": meta["git_commit"], "config_sha256": config_hash(name if name != "S1" else "S1"),
        "split_sha256": config_spec(name)["split_sha256"], "params": meta["n_params"], "optimizer": "AdamW", "lr": F.PROTO["lr"],
        "batch": F.PROTO["batch"], "updates": meta.get("updates", F.PROTO["steps"]), "train_seconds": meta["train_seconds"], "gpu": meta["gpu"],
        "peak_gpu_mem_mib": meta["peak_gpu_mem_mib"], "nan_steps": meta["nan_steps"], "checkpoint_sha256": meta["sha256"]}]
    write_json("training_seed_manifest.json", man)


# ----------------------------------------------------------------------------------------------- shared evaluation helpers
@torch.no_grad()
def point_infer(net, X, R, dev, bs=1024):
    f = net if isinstance(net, AB.WWDet) else net.point
    return np.concatenate([f(torch.from_numpy(X[i:i + bs]).to(dev), torch.from_numpy(R[i:i + bs]).to(dev)).cpu().numpy()
                           for i in range(0, len(X), bs)]).astype(np.float64)


@torch.no_grad()
def gen_infer(net, X, R, noise, dev, bs=1024):
    out = []
    for i in range(0, len(X), bs):
        x0, x, r = (torch.from_numpy(a[i:i + bs]).to(dev) for a in (noise, X, R))
        out.append((SM.euler(net, x0, x, r, F.NFE) if isinstance(net, SM.ScaleFM) else D.euler(net, x0, x, r, F.NFE)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def evaluate_seed(ctx, seed, ex, dev, with_shuffle=True):
    """P, S1 point, G, S1 generative (and the PPG-shuffled generative heads) of one seed on one population; paired bootstrap."""
    X, Y, Pid, R, ref, pairs, perm = ctx["X"], ctx["Y"], ctx["Pid"], ctx["R"], ctx["ref"], ctx["pairs"], ctx["perm"]
    nets = {r: load_model(r, seed, dev) for r in ROLES}
    pts = {"P": point_infer(nets["point"], X, R, dev), "S1": point_infer(nets["S1"], X, R, dev)}
    gens = {"G": gen_infer(nets["gen"], X, R, ctx["noise"], dev), "S1": gen_infer(nets["S1"], X, R, ctx["noise"], dev)}
    shuf = {k: gen_infer(nets[{"G": "gen", "S1": "S1"}[k]], X[perm], R, ctx["noise"], dev) for k in ("G", "S1")} if with_shuffle else {}
    del nets
    torch.cuda.empty_cache()
    pm_p = {k: F.point_metrics(v, Y, ref, Pid, pairs, ex) for k, v in pts.items()}
    pm_g = {k: F.point_metrics(v, Y, ref, Pid, pairs, ex) for k, v in gens.items()}
    pm_s = {k: F.point_metrics(v, Y, ref, Pid, pairs, ex) for k, v in shuf.items()}
    res = B.patient_resamples(np.unique(Pid).size, BOOT_N, BOOT_SEED)
    arms = {"G": gens["G"], "S1": gens["S1"]} | ({"G_shuf": shuf["G"], "S1_shuf": shuf["S1"]} if with_shuffle else {})
    prs = [("S1", "G")] + ([("S1_shuf", "S1"), ("G_shuf", "G")] if with_shuffle else [])
    fdb = FF.fd_bootstrap(arms, Y, Pid, prs, res)
    pats = pm_p["P"]["pm"]["patients"]
    assert all(np.array_equal(m["pm"]["patients"], pats) for m in list(pm_p.values()) + list(pm_g.values()) + list(pm_s.values()))
    d = {"corr": ci(pm_p["S1"]["corr"] - pm_p["P"]["corr"], Pid),
         "fp": C0.cluster_ci(pm_p["S1"]["pm"]["fp_rate"] - pm_p["P"]["pm"]["fp_rate"], pats, BOOT_N, BOOT_SEED),
         "recall": C0.cluster_ci(pm_p["S1"]["pm"]["recall"] - pm_p["P"]["pm"]["recall"], pats, BOOT_N, BOOT_SEED),
         "fd": fdb["S1-G"], "f1_desc": C0.cluster_ci(pm_p["S1"]["pm"]["f1"] - pm_p["P"]["pm"]["f1"], pats, BOOT_N, BOOT_SEED),
         "mae_desc": ci(pm_p["S1"]["mae"] - pm_p["P"]["mae"], Pid), "hr_desc": ci(pm_p["S1"]["hr"] - pm_p["P"]["hr"], Pid)}
    if with_shuffle:
        d |= {"fd_shuf": fdb["S1_shuf-S1"], "corr_shuf": ci(pm_s["S1"]["corr"] - pm_g["S1"]["corr"], Pid),
              "g_fd_shuf": fdb["G_shuf-G"], "g_corr_shuf": ci(pm_s["G"]["corr"] - pm_g["G"]["corr"], Pid)}
    acc = F.accounting()
    gates = D.gates(d, acc["S1"]["total"], acc["separate_waveform_params"]) if with_shuffle else None
    mu_P = pts["P"]
    real_dev = float(np.sqrt(((Y - mu_P) ** 2).mean()))
    out = {"seed": seed, "windows": int(len(Y)), "patients": int(np.unique(Pid).size),
           "point": {k: F.summarize_point(m, Pid) | {"fd_descriptive": float(PMX.kanflow_fd(pts[k], Y))} for k, m in pm_p.items()},
           "gen": {k: {"fd": fdb["fd"][k], "nfe": F.NFE, "single_sample": F.summarize_point(m, Pid),
                       "diversity_ratio_rms_dev_from_P": float(np.sqrt(((gens[k] - mu_P) ** 2).mean()) / real_dev),
                       "spectral_discrepancy": F.spectral_discrepancy(gens[k], Y)} for k, m in pm_g.items()},
           "comparisons": d, "gates": gates,
           "condition_shuffle": {k: {"fd_conditioned": fdb["fd"][k], "fd_shuffled": fdb["fd"][f"{k}_shuf"],
                                     "corr_conditioned": ci(pm_g[k]["corr"], Pid), "corr_shuffled": ci(pm_s[k]["corr"], Pid)} for k in shuf},
           "patient_level": {"patients": pats.tolist(), "delta_fp": (pm_p["S1"]["pm"]["fp_rate"] - pm_p["P"]["pm"]["fp_rate"]).tolist(),
                             "delta_corr": _per_patient(pm_p["S1"]["corr"] - pm_p["P"]["corr"], Pid, pats),
                             "delta_hr_mae": _per_patient(pm_p["S1"]["hr"] - pm_p["P"]["hr"], Pid, pats)}}
    return out, pts["P"]


def _per_patient(v, pid, pats):
    v = np.asarray(v, float)
    return [float(np.nanmean(v[pid == p])) if np.isfinite(v[pid == p]).any() else None for p in pats]


# ----------------------------------------------------------------------------------------------- Stage D: internal multi-seed
def stage_internal(ex, dev):
    for s in NEW_SEEDS:
        for r in ROLES:
            if not ckpt_file(r, s).exists():
                raise SystemExit(f"STOP: missing seed {s} {r}")
    res = {}
    for role in ("dp_dev", "af_lock"):
        ctx = F.context(role, ex, dev)
        res[role] = {}
        for s in SEEDS:
            out, _ = evaluate_seed(ctx, s, ex, dev, with_shuffle=False)
            out.pop("patient_level")
            res[role][s] = out
            c = out["comparisons"]
            print(f"[dp3] internal {role} seed {s}: dcorr {c['corr'][0]:+.4f} dFP {c['fp'][0]:+.4f} drec {c['recall'][0]:+.4f} dFD {c['fd'][0]:+.3f}", flush=True)
        del ctx
        gc.collect()
    write_json("internal_multiseed_metrics.json", {"note": "internal characterization only (DP-DEV development population; AF-LOCK already known from "
                                                           "DP0); never used to select a seed, change the model or decide on MIMIC-BP",
                                                   "results": res})
    summ = {}
    for role, rr in res.items():
        summ[role] = {}
        for key, get in (("delta_corr", lambda o: o["comparisons"]["corr"][0]), ("delta_fp", lambda o: o["comparisons"]["fp"][0]),
                         ("delta_recall", lambda o: o["comparisons"]["recall"][0]), ("delta_fd", lambda o: o["comparisons"]["fd"][0]),
                         ("P_corr", lambda o: o["point"]["P"]["corr"][0]), ("S1_corr", lambda o: o["point"]["S1"]["corr"][0]),
                         ("P_fp", lambda o: o["point"]["P"]["pm_fp_rate"][0]), ("S1_fp", lambda o: o["point"]["S1"]["pm_fp_rate"][0]),
                         ("P_recall", lambda o: o["point"]["P"]["pm_recall"][0]), ("S1_recall", lambda o: o["point"]["S1"]["pm_recall"][0]),
                         ("G_fd", lambda o: o["gen"]["G"]["fd"]), ("S1_fd", lambda o: o["gen"]["S1"]["fd"])):
            v = np.array([get(rr[s]) for s in SEEDS])
            summ[role][key] = {"by_seed": dict(zip(map(str, SEEDS), v.tolist())), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                               "min": float(v.min()), "max": float(v.max())}
    write_json("internal_multiseed_summary.json", summ)


# ----------------------------------------------------------------------------------------------- Stage E: fair compute
def haar_cond(ppg, raster):
    return SM.haar(ppg), SM.haar(raster)


def scaleflow_cached(net, xt, t, hp, hr):
    """ScaleFM.forward with Haar(PPG) / Haar(raster) supplied from a cache: the same operations in the same order, so the
    output is identical to net(xt, t, ppg, raster)."""
    xc, xm, xf = SM.haar(xt)
    (pc, pm, pf), (ec, em, ef) = hp, hr
    e = net.temb(t)
    hc, vc = net.coarse(torch.stack([xc, pc, ec], dim=1), e)
    hm, vm = net.mid(torch.cat([torch.stack([xm, pm, em], dim=1), net.proj_cm(hc)], dim=1), e)
    _, vf = net.fine(torch.cat([torch.stack([xf, pf, ef], dim=1), SM.up2(net.proj_mf(hm)), SM.up2(net.proj_cf(hc))], dim=1), e)
    return SM.ihaar(vc, vm, vf)


@torch.no_grad()
def euler_cached(net, x0, ppg, raster, nfe=8):
    hp, hr = haar_cond(ppg, raster)
    x = x0.clone()
    dt = 1.0 / nfe
    for i in range(nfe):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * scaleflow_cached(net, x, t, hp, hr)
    return x


def stage_compute(ex, dev):
    from torch.utils.flop_counter import FlopCounterMode
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, _, Pid_, wid_, _ = F.load_role("dp_dev", ex)
    pick = B.salted_rank("dp3-latency-v1", range(len(X)))[:50]
    torch.set_num_threads(4)
    acc = F.accounting()
    det_params = D.n_params(RhythmTCN())
    lat, mem, flops, eq = {}, {}, {}, {}
    reps = {"cuda": 200, "cpu": 60}
    for dname in ("cuda", "cpu"):
        d = torch.device(dname)
        det = load_detector(d)
        P, G, S1 = (load_model(r, 42, d) for r in ROLES)

        def raster(xi):
            p = torch.sigmoid(det(xi[:, None])[:, 0]).float().cpu().numpy()[0]
            return torch.from_numpy(AB.event_raster([F._extract(p)])).to(d)

        def req(system, mode, xi, x0, ri=None):
            if system == "naive":
                if mode == "point":
                    return P(xi, raster(xi) if ri is None else ri)
                if mode == "gen":
                    return SM.euler(G, x0, xi, raster(xi) if ri is None else ri, F.NFE)
                return P(xi, raster(xi) if ri is None else ri), SM.euler(G, x0, xi, raster(xi) if ri is None else ri, F.NFE)
            if system == "cached":
                r = raster(xi) if ri is None else ri
                if mode == "point":
                    return P(xi, r)
                if mode == "gen":
                    return euler_cached(G, x0, xi, r, F.NFE)
                return P(xi, r), euler_cached(G, x0, xi, r, F.NFE)
            r = raster(xi) if ri is None else ri
            if mode == "point":
                return S1.point(xi, r)
            if mode == "gen":
                return D.euler(S1, x0, xi, r, F.NFE)
            mu, c = S1.both(xi, r)
            return mu, D.euler(S1, x0, xi, r, F.NFE, cond=c)

        for system in ("naive", "cached", "s1"):
            for mode in ("point", "gen", "both"):
                for with_det in (True, False):
                    ts = []
                    n = reps[dname]
                    for rep in range(n + 10):
                        i = int(pick[rep % len(pick)])
                        xi = torch.from_numpy(X[i:i + 1]).to(d)
                        x0 = torch.from_numpy(SM.window_noise(Pid_[i:i + 1], wid_[i:i + 1])).to(d)
                        with torch.no_grad():
                            ri = None if with_det else raster(xi)
                            if dname == "cuda":
                                torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            req(system, mode, xi, x0, ri)
                            if dname == "cuda":
                                torch.cuda.synchronize()
                        if rep >= 10:
                            ts.append((time.perf_counter() - t0) * 1000)
                    q = np.percentile(ts, [25, 50, 75])
                    lat[f"{dname}_{system}_{mode}" + ("_with_detector" if with_det else "_waveform_only")] = {"median": float(q[1]), "q25": float(q[0]),
                                                                                                           "q75": float(q[2]), "n": n}
                if dname == "cuda":
                    xi = torch.from_numpy(X[:1]).to(d)
                    x0 = torch.from_numpy(SM.window_noise(Pid_[:1], wid_[:1])).to(d)
                    with torch.no_grad():
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                        req(system, mode, xi, x0); torch.cuda.synchronize()
                        mem[f"{system}_{mode}"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        if dname == "cpu":
            xi = torch.from_numpy(X[:1])
            with torch.no_grad():
                ri = raster(xi)
                x0 = torch.from_numpy(SM.window_noise(Pid_[:1], wid_[:1]))
                eq["cached_equals_naive_generation"] = bool(torch.equal(SM.euler(G, x0, xi, ri, F.NFE), euler_cached(G, x0, xi, ri, F.NFE)))
                eq["cached_point_equals_naive"] = True
            z, t0_ = torch.zeros(1, T), torch.zeros(1)

            def fl(fn):
                fc = FlopCounterMode(display=False)
                with fc, torch.no_grad():
                    fn()
                return int(fc.get_total_flops())
            f_det = fl(lambda: det(xi[:, None]))
            f_P = fl(lambda: P(xi, ri))
            f_G = fl(lambda: G(z, t0_, xi, ri))
            hp, hr = haar_cond(xi, ri)
            f_Gc = fl(lambda: scaleflow_cached(G, z, t0_, hp, hr))
            with torch.no_grad():
                h = S1.trunk(xi, ri)
                cond = S1.flow_features(h)
            f_tr = fl(lambda: S1.trunk(xi, ri))
            f_pb = fl(lambda: S1.point_dec(S1.point_features(h)))
            f_ff = fl(lambda: S1.flow_features(h))
            f_v = fl(lambda: S1.velocity(z, t0_, cond))
            nfe = F.NFE
            flops = {"detector": f_det, "per_component": {"P": f_P, "G_per_nfe": f_G, "G_cached_per_nfe": f_Gc, "S1_trunk": f_tr, "S1_point_branch": f_pb,
                                                          "S1_flow_features": f_ff, "S1_velocity_per_nfe": f_v},
                     "waveform_only": {"naive": {"point": f_P, "gen": nfe * f_G, "both": f_P + nfe * f_G},
                                       "cached": {"point": f_P, "gen": nfe * f_Gc, "both": f_P + nfe * f_Gc},
                                       "s1": {"point": f_tr + f_pb, "gen": f_tr + f_ff + nfe * f_v, "both": f_tr + f_pb + f_ff + nfe * f_v}},
                     "with_detector": {"naive": {"point": f_det + f_P, "gen": f_det + nfe * f_G, "both": 2 * f_det + f_P + nfe * f_G},
                                       "cached": {"point": f_det + f_P, "gen": f_det + nfe * f_Gc, "both": f_det + f_P + nfe * f_Gc},
                                       "s1": {"point": f_det + f_tr + f_pb, "gen": f_det + f_tr + f_ff + nfe * f_v,
                                              "both": f_det + f_tr + f_pb + f_ff + nfe * f_v}}}
    params = {"naive": {"waveform": acc["separate_waveform_params"], "full_pipeline": acc["separate_waveform_params"] + 2 * det_params,
                        "note": "two independent pipelines, each with its own detector pass (same detector weights)"},
              "cached": {"waveform": acc["separate_waveform_params"], "full_pipeline": acc["separate_waveform_params"] + det_params},
              "s1": {"waveform": acc["S1"]["total"], "full_pipeline": acc["S1"]["total"] + det_params}}
    proto = {"systems": {"naive": "P and G as two independent pipelines: each runs the detector + event extraction + raster; G recomputes Haar(PPG) and "
                                  "Haar(raster) inside every vector-field evaluation",
                         "cached": "one detector + raster shared by P and G; G's Haar(PPG) / Haar(raster) computed once per window and reused by all 8 NFE "
                                   "(numerically identical, checked); no weight merged, no network altered. P and G have no other shareable computation: "
                                   "G mixes x_t into every layer from its first convolution, so its condition path cannot be precomputed",
                         "s1": "one detector + raster; S1 shared trunk once; point branch; flow features once; 8 velocity evaluations"},
             "timing": "batch-1, 10 warm-up calls then 200 (GPU) / 60 (CPU, 4 threads) calls over 50 salted DP-DEV windows; median and IQR",
             "flops": "torch.utils.flop_counter on one window (CPU); element-wise Haar arithmetic is not counted by the counter",
             "memory": "CUDA max_memory_allocated during one batch-1 request minus the resident baseline (weights)", "nfe": F.NFE}
    write_json("compute_protocol.json", proto)
    for system, fname in (("naive", "compute_naive.json"), ("cached", "compute_cached.json"), ("s1", "compute_s1.json")):
        write_json(fname, {"system": system, "params": params[system], "flops_waveform_only": flops["waveform_only"][system],
                           "flops_with_detector": flops["with_detector"][system],
                           "latency_ms": {k: v for k, v in lat.items() if f"_{system}_" in k},
                           "gpu_peak_inference_mib_above_weights": {k: v for k, v in mem.items() if k.startswith(system)},
                           "numerical_equivalence": eq if system == "cached" else None})
    with open(ART / "table_compute.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "waveform_params", "pipeline_params", "flops_point", "flops_gen", "flops_both", "gpu_point_ms", "gpu_gen_ms", "gpu_both_ms",
                    "gpu_both_iqr", "cpu_both_ms", "cpu_both_iqr", "peak_mem_both_mib"])
        for s in ("naive", "cached", "s1"):
            L = lambda k: lat[k]  # noqa: E731
            w.writerow([s, params[s]["waveform"], params[s]["full_pipeline"], *[flops["with_detector"][s][m] for m in ("point", "gen", "both")],
                        *[f"{L(f'cuda_{s}_{m}_with_detector')['median']:.2f}" for m in ("point", "gen", "both")],
                        f"{L(f'cuda_{s}_both_with_detector')['q25']:.2f}-{L(f'cuda_{s}_both_with_detector')['q75']:.2f}",
                        f"{L(f'cpu_{s}_both_with_detector')['median']:.2f}",
                        f"{L(f'cpu_{s}_both_with_detector')['q25']:.2f}-{L(f'cpu_{s}_both_with_detector')['q75']:.2f}", f"{mem[f'{s}_both']:.3f}"])
    print(f"[dp3] compute: GPU both naive {lat['cuda_naive_both_with_detector']['median']:.2f} cached {lat['cuda_cached_both_with_detector']['median']:.2f} "
          f"S1 {lat['cuda_s1_both_with_detector']['median']:.2f} ms; equivalence {eq}", flush=True)


# ----------------------------------------------------------------------------------------------- Stage F: MIMIC-BP interface (metadata only)
def subject_ids():
    return sorted(p.name.split("_")[0] for p in (MIMIC / "ppg").glob("p*_ppg.npy"))


def pid_int(pid: str) -> int:
    return int(pid[1:])


def window_ids(n_subjects_before: int, seg: int, k: int) -> int:
    """Global window id = subject rank * 210 + segment * 7 + window (deterministic, independent of exclusions)."""
    return n_subjects_before * N_SEG * WIN_PER_SEG + seg * WIN_PER_SEG + k


def stage_interface(ex, dev):
    ids = subject_ids()
    shapes = {}
    for kind in ("ppg", "ecg", "abp", "resp", "labels"):
        sh = {}
        for pid in ids:
            a = np.load(MIMIC / kind / f"{pid}_{kind}.npy", mmap_mode="r")          # header only (shape / dtype)
            sh.setdefault(f"{list(a.shape)} {a.dtype}", 0)
            sh[f"{list(a.shape)} {a.dtype}"] += 1
        shapes[kind] = sh
    fs = int(re.search(r"fs\s*=\s*(\d+)", (MIMIC / "read_data.py").read_text()).group(1))
    lists = {k: json.loads((MIMIC / f"{k}_subjects.txt").read_text().replace("'", '"')) for k in ("train", "val", "test")}
    write_json("dataset_schema.json", {"path": str(MIMIC), "subjects": len(ids), "files_per_kind": {k: sum(v.values()) for k, v in shapes.items()},
                                       "array_shapes": shapes, "sampling_rate_hz": fs, "fs_source": "data/raw/MIMIC-BP/read_data.py ('fs = 125')",
                                       "channels": {"ppg": "one channel per file (p<ID>_ppg.npy)", "ecg": "one channel per file (p<ID>_ecg.npy); lead "
                                                    "identity not stated in local metadata (UNKNOWN)", "abp": "not used", "resp": "not used"},
                                       "segments": f"{N_SEG} x {SEG_LEN} samples = 30 s per segment", "official_split_lists": {k: len(v) for k, v in lists.items()},
                                       "synchronisation": "PPG and ECG of a segment share sample indices (one MIMIC-III record); curated by the dataset "
                                                          "authors (ECG/PPG fundamental frequency within 0.3 Hz, pulse-arrival-time consistency; A7 audit)",
                                       "access": "file names, mmap shapes / dtypes and loader text only; no waveform value read"})
    adapter = {"evidence_label": EVIDENCE_LABEL, "disclosure": DISCLOSURE, "cohort": "all 1,524 subjects (no calibration split)",
               "ppg_channel": "the single ppg file per subject", "ecg_channel": "the single ecg file per subject (lead identity UNKNOWN; rule: exactly one ECG "
                                                                                "channel exists -> use it)",
               "windows": {"length_s": WIN_S, "raw_samples": WIN_RAW, "start_rule": "segment start (t = 0) of each 30 s segment", "stride_s": WIN_S,
                           "overlap": 0, "per_segment": WIN_PER_SEG, "edge": "the last 2 s of each 30 s segment are dropped (no padding, no cross-segment window)",
                           "order": "subjects sorted by id, then segment, then window", "window_id": "subject_rank * 210 + segment * 7 + k",
                           "patient_id_for_noise": "int(pid[1:])"},
               "resampling": "scipy.signal.resample (FFT) 500 -> 512 samples per window inside ppg2ecg.data.preprocess.preprocess_windows (the V1 / DP0 "
                             "implementation, unchanged since a15b354)",
               "normalization": {"ppg": PPG_KW, "ecg": ECG_KW, "semantics": "per-window statistics only (nothing fitted); identical to the DP0 corpus"},
               "temporal_alignment": "none applied: PPG and ECG windows take identical sample indices of the same segment; no shift, no PTT compensation, "
                                     "no offset search",
               "detector": {"checkpoint": "outputs/dp0_dualreadout/detector.pt", "threshold": F.DET["threshold"], "refractory": F.DET["refractory"],
                            "raster": "Gaussian sigma 20 ms (ablation.event_raster)"},
               "reference_r": "neurokit (ppg2ecg.evaluation.rpeaks.detect_rpeaks) on the preprocessed 128 Hz ECG window",
               "noise": "scaleflow.window_noise(int pid, window_id) (sha256 'patient:window:20261002')"}
    write_json("dataset_adapter.json", adapter)
    rules = {"order": ["R1 unreadable / missing file or shape != (30, 3750) -> subject excluded",
                       "R2 raw window: any non-finite PPG or ECG sample, or zero standard deviation of PPG or ECG -> window excluded",
                       "R3 after preprocessing: any non-finite value in x or y -> window excluded",
                       "R4 frozen V1 / DP0 reference-ECG validity rule: neurokit HR of the preprocessed ECG window not finite or outside [30, 200] bpm -> "
                       "window excluded",
                       "R5 subject with no remaining window -> subject excluded"],
             "source": "R2-R4 are the scripts/v1_build_vitaldb.py rules that defined every DP0 window; nothing depends on any model output",
             "not_allowed": ["model correlation", "ECG morphology", "FP", "F1", "FD", "subject difficulty"], "v1_candidate_subsampling": "not applied (all windows)"}
    write_json("external_exclusion_rules.json", rules)
    man = {"subjects": ids, "n_subjects": len(ids), "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
           "pid_int": {p: pid_int(p) for p in ids}, "max_windows": len(ids) * N_SEG * WIN_PER_SEG,
           "k16_rule": f"salted rank '{K16_SALT}' over the global window ids of eligible windows, first {N_K16}",
           "shuffle": "default_rng(20261002).permutation over eligible windows (PPG only)"}
    write_json("external_subject_manifest.json", man)
    print(f"[dp3] interface: {len(ids)} subjects, shapes {json.dumps(shapes['ppg'])} / {json.dumps(shapes['ecg'])}, fs {fs}", flush=True)


# ----------------------------------------------------------------------------------------------- Stage G: final freeze
def freeze_files():
    files = [PREREG, *CODE_FILES, "scripts/dp0_dualreadout.py", "src/ppg2ecg/dualreadout/model.py", "src/ppg2ecg/scaleflow/model.py",
             "src/ppg2ecg/data/preprocess.py", "src/ppg2ecg/evaluation/rpeaks.py", "src/ppg2ecg/evaluation/paper_metrics.py",
             "src/ppg2ecg/anchorflow/fastfd.py", "src/ppg2ecg/probes/rhythm_tcn.py", "src/ppg2ecg/coherentbeat/ablation.py", "scripts/bf0_run.py",
             "scripts/c0_coherentbeat.py", "artifacts/dp3_mimicbp/dataset_adapter.json", "artifacts/dp3_mimicbp/external_exclusion_rules.json",
             "artifacts/dp3_mimicbp/external_subject_manifest.json", "artifacts/dp3_mimicbp/target_blind_audit.json",
             "artifacts/dp0_dualreadout/split_manifest.json", "outputs/dp0_dualreadout/detector.pt"]
    files += [str(ckpt_file(r, s).relative_to(ROOT)) for s in SEEDS for r in ROLES]
    return files


def stage_freeze(ex, dev):
    if read_json("target_blind_audit.json")["verdict"] != "PASS":
        raise SystemExit("STOP: target-blind audit did not pass")
    write_json("final_freeze_manifest.json", {"evidence_label": EVIDENCE_LABEL, "primary_seed": 42, "seeds": list(SEEDS), "nfe": F.NFE,
                                              "bootstrap": {"replicates": BOOT_N, "seed": BOOT_SEED}, "shuffle_seed": SHUF_SEED,
                                              "sha256": {f: B.sha256_file(ROOT / f) for f in freeze_files()}})
    write_json("checkpoint_hashes.json", {f"{r}{s}": B.sha256_file(ckpt_file(r, s)) for s in SEEDS for r in ROLES} |
               {"detector": B.sha256_file(DP0_OUT / "detector.pt")})


def check_final_freeze():
    """MIMIC-BP waveform values are read only after a committed, unchanged final freeze whose hashed files are unchanged."""
    if not FREEZE.exists():
        raise PermissionError("DP3: MIMIC-BP sealed (no final freeze manifest)")
    if not (ROOT / PREREG).exists():
        raise PermissionError("DP3: MIMIC-BP sealed (no preregistration)")
    fm = json.loads(FREEZE.read_text())
    for f, h in fm["sha256"].items():
        if B.sha256_file(ROOT / f) != h:
            raise PermissionError(f"DP3: frozen file changed: {f}")
    for rel in (str(FREEZE.relative_to(ROOT)), PREREG):
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
        if not (tracked and clean):
            raise PermissionError(f"DP3: {rel} not committed")


# ----------------------------------------------------------------------------------------------- Stage H: external evaluation
def _subject_windows(args):
    """Load one subject (only after the freeze), apply R1-R4, return preprocessed windows and an exclusion log."""
    rank, pid = args
    log = []
    try:
        ppg = np.load(MIMIC / "ppg" / f"{pid}_ppg.npy")
        ecg = np.load(MIMIC / "ecg" / f"{pid}_ecg.npy")
        assert ppg.shape == ecg.shape == (N_SEG, SEG_LEN)
    except Exception as e:                                                   # R1
        return pid, None, [(pid, -1, -1, "R1 unreadable or wrong shape", str(e)[:80])]
    P = ppg[:, :WIN_PER_SEG * WIN_RAW].reshape(N_SEG * WIN_PER_SEG, WIN_RAW).astype(np.float64)
    E = ecg[:, :WIN_PER_SEG * WIN_RAW].reshape(N_SEG * WIN_PER_SEG, WIN_RAW).astype(np.float64)
    seg = np.repeat(np.arange(N_SEG), WIN_PER_SEG)
    k = np.tile(np.arange(WIN_PER_SEG), N_SEG)
    ok = np.isfinite(P).all(1) & np.isfinite(E).all(1)
    with np.errstate(invalid="ignore"):
        ok &= (np.nan_to_num(P).std(1) > 0) & (np.nan_to_num(E).std(1) > 0)
    log += [(pid, int(seg[j]), int(k[j]), "R2 raw non-finite or constant", "") for j in np.flatnonzero(~ok)]
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return pid, None, log + [(pid, -1, -1, "R5 no remaining window", "")]
    x = preprocess_windows(P[idx], FS, WIN_S, **PPG_KW)
    y = preprocess_windows(E[idx], FS, WIN_S, **ECG_KW)
    ok2 = np.isfinite(x).all(1) & np.isfinite(y).all(1)
    log += [(pid, int(seg[idx[j]]), int(k[idx[j]]), "R3 non-finite after preprocessing", "") for j in np.flatnonzero(~ok2)]
    peaks = [np.asarray(RP.detect_rpeaks(row, FS, "neurokit"), int) if good else np.zeros(0, int) for row, good in zip(y, ok2)]
    hr = np.array([RP.hr_bpm(pk, FS) if good else np.nan for pk, good in zip(peaks, ok2)])
    ok3 = ok2 & np.isfinite(hr) & (hr >= HR_RANGE[0]) & (hr <= HR_RANGE[1])
    log += [(pid, int(seg[idx[j]]), int(k[idx[j]]), "R4 reference HR not finite or outside [30, 200] bpm", f"{hr[j]:.1f}") for j in np.flatnonzero(ok2 & ~ok3)]
    keep = np.flatnonzero(ok3)
    if keep.size == 0:
        return pid, None, log + [(pid, -1, -1, "R5 no remaining window", "")]
    wid = np.array([window_ids(rank, int(seg[idx[j]]), int(k[idx[j]])) for j in keep], np.int64)
    return pid, {"x": x[keep].astype(np.float32), "y": y[keep].astype(np.float64), "wid": wid, "ref": [peaks[j] for j in keep]}, log


def external_context(ex, dev):
    check_final_freeze()
    ids = read_json("external_subject_manifest.json")["subjects"]
    assert ids == subject_ids()
    X, Y, Pid, W, ref, log = [], [], [], [], [], []
    for pid, data, lg in ex.map(_subject_windows, list(enumerate(ids)), chunksize=8):
        log += lg
        if data is not None:
            X.append(data["x"]); Y.append(data["y"]); W.append(data["wid"]); ref += data["ref"]
            Pid.append(np.full(len(data["wid"]), pid_int(pid)))
    X, Y, Pid, W = np.concatenate(X), np.concatenate(Y), np.concatenate(Pid), np.concatenate(W)
    with open(ART / "external_exclusion_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "segment", "window", "rule", "detail"])
        w.writerows(log)
    det = load_detector(dev)
    with torch.no_grad():
        prob = [torch.sigmoid(det(torch.from_numpy(X[i:i + 4096]).to(dev)[:, None])[:, 0]).float().cpu().numpy() for i in range(0, len(X), 4096)]
    ev = [np.asarray(e, int) for e in ex.map(F._extract, list(np.concatenate(prob)), chunksize=512)]
    del det, prob
    R = AB.event_raster(ev)
    S = np.sort(np.argsort([hashlib.sha256(f"{K16_SALT}:{int(w)}".encode()).hexdigest() for w in W], kind="stable")[:N_K16])
    return {"X": X, "Y": Y, "Pid": Pid, "wid": W, "ref": ref, "ev": ev, "R": R, "pairs": [BR.matched_pairs(ref[i], ev[i], T) for i in range(len(Y))],
            "noise": SM.window_noise(Pid, W), "S": S, "perm": np.random.default_rng(SHUF_SEED).permutation(len(Y)), "log": log, "ids": ids}


def stage_eval_external(ex, dev):
    check_final_freeze()
    ctx = external_context(ex, dev)
    X, Y, Pid = ctx["X"], ctx["Y"], ctx["Pid"]
    excl_subj = sorted({r[0] for r in ctx["log"] if r[3].startswith(("R1", "R5"))})
    from collections import Counter
    cohort = {"subjects_total": len(ctx["ids"]), "subjects_eligible": int(np.unique(Pid).size), "subjects_excluded": len(excl_subj),
              "windows_possible": len(ctx["ids"]) * N_SEG * WIN_PER_SEG, "windows_eligible": int(len(Y)),
              "exclusions_by_rule": dict(Counter(r[3].split(" ")[0] for r in ctx["log"]))}
    write_json("external_cohort.json", cohort)
    print(f"[dp3] external cohort: {json.dumps(cohort)}", flush=True)
    placed = PMX.rpeak_prf_at(Y, Y, FS, 50.0, peaks=(ctx["ref"], ctx["ev"]))
    pmp = RM.patient_macro_rows(placed["n_tp"], placed["n_fp"], placed["n_fn"], Pid)
    bl = PMX.beat_level_metrics(Y, Y, FS, 50.0, peaks=(ctx["ref"], ctx["ev"]))
    write_json("external_detector_metrics.json", {**{f"pm_{k}": C0.cluster_ci(pmp[k], pmp["patients"], BOOT_N, BOOT_SEED)
                                                     for k in ("fp_rate", "recall", "precision", "f1")}, "pooled": RM.pooled_prf(**pmp["pooled"]),
                                                  "rr_mae_ms": ci(bl["rr_mae_ms"], Pid), "hr_mae_bpm": ci(bl["hr_abs_err"], Pid),
                                                  "note": "frozen DP0 detector vs neurokit reference R on the preprocessed ECG; descriptive only"})
    allres, muP42 = {}, None
    for s in SEEDS:
        out, muP = evaluate_seed(ctx, s, ex, dev, with_shuffle=True)
        if s == 42:
            muP42 = muP
        pl = out.pop("patient_level")
        write_json(f"external_seed{s}_metrics.json", {k: v for k, v in out.items() if k not in ("comparisons", "gates")} | {"cohort": cohort})
        write_json(f"external_seed{s}_bootstrap.json", {"unit": "patient", "replicates": BOOT_N, "seed": BOOT_SEED, "comparisons": out["comparisons"]})
        write_json(f"external_seed{s}_gates.json", out["gates"] | {"T5_saving": F.accounting()["S1"]["saving"]})
        if s == 42:
            write_json("external_seed42_patient_level.json", pl)
        allres[s] = out
        g = out["gates"]
        print(f"[dp3] external seed {s}: {json.dumps(g)}", flush=True)
        gc.collect()
    write_json("external_condition_shuffle.json", {s: {"S1": {"fd_shuffled_minus_conditioned": r["comparisons"]["fd_shuf"],
                                                              "corr_shuffled_minus_conditioned": r["comparisons"]["corr_shuf"]} | r["condition_shuffle"]["S1"],
                                                       "G_descriptive": {"fd_shuffled_minus_conditioned": r["comparisons"]["g_fd_shuf"],
                                                                         "corr_shuffled_minus_conditioned": r["comparisons"]["g_corr_shuf"]} | r["condition_shuffle"]["G"]}
                                                   for s, r in allres.items()})
    k16 = {}
    for role, key in (("gen", "G42"), ("S1", "S1-42")):
        net = load_model(role, 42, dev)
        S = ctx["S"]
        Ks = np.stack([gen_infer(net, X[S], ctx["R"][S], SM.window_noise(Pid[S], ctx["wid"][S], k), dev) for k in range(K)])
        k16[key] = F.k16_block(ctx, Ks, muP42[S], ex)
        del net
    write_json("k16_external.json", {"windows": int(len(ctx["S"])), "K": K, "rule": f"salted rank '{K16_SALT}' over eligible window ids", "results": k16})
    _multiseed_summary(allres)


def _multiseed_summary(allres):
    passes = {s: all(allres[s]["gates"][k] for k in ("P1", "P2", "G1", "CONDITION")) for s in SEEDS}
    n = sum(passes.values())
    rob = "ROBUST-3/3" if n == 3 else "ROBUST-2/3" if n == 2 else "SEED-SENSITIVE"
    summ = {}
    for key, get in (("P_corr", lambda o: o["point"]["P"]["corr"][0]), ("S1_corr", lambda o: o["point"]["S1"]["corr"][0]),
                     ("delta_corr", lambda o: o["comparisons"]["corr"][0]), ("P_fp", lambda o: o["point"]["P"]["pm_fp_rate"][0]),
                     ("S1_fp", lambda o: o["point"]["S1"]["pm_fp_rate"][0]), ("delta_fp", lambda o: o["comparisons"]["fp"][0]),
                     ("P_recall", lambda o: o["point"]["P"]["pm_recall"][0]), ("S1_recall", lambda o: o["point"]["S1"]["pm_recall"][0]),
                     ("delta_recall", lambda o: o["comparisons"]["recall"][0]), ("G_fd", lambda o: o["gen"]["G"]["fd"]),
                     ("S1_fd", lambda o: o["gen"]["S1"]["fd"]), ("delta_fd", lambda o: o["comparisons"]["fd"][0]),
                     ("shuffle_fd", lambda o: o["comparisons"]["fd_shuf"][0]), ("shuffle_corr", lambda o: o["comparisons"]["corr_shuf"][0])):
        v = np.array([get(allres[s]) for s in SEEDS], float)
        summ[key] = {"by_seed": dict(zip(map(str, SEEDS), v.tolist())), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()),
                     "max": float(v.max())}
    primary = allres[42]["gates"]["QUALIFIED"]
    write_json("external_multiseed_summary.json", {"seed_passes_T1_T4": passes, "robustness": rob, "primary_seed42": "CONFIRMED" if primary else "FAILED",
                                                   "verdict": "TARGET-BLIND EXTERNAL CONFIRMED" if primary else "TARGET-BLIND EXTERNAL FAILED",
                                                   "paper_readiness": ("STRONG GO" if primary and rob == "ROBUST-3/3" else "GO WITH SEED LIMITATION"
                                                                       if primary and rob == "ROBUST-2/3" else "INTERNAL + CROSS-TASK EVIDENCE ONLY"
                                                                       if not primary else "INTERNAL + CROSS-TASK EVIDENCE ONLY (seed-sensitive)"),
                                                   "summary": summ, "note": "descriptive across n = 3 seeds; no inferential test over seeds"})


STAGES = {"target_audit": stage_target_audit, "integrity": stage_integrity, "internal": stage_internal, "compute": stage_compute,
          "interface": stage_interface, "freeze": stage_freeze, "eval_external": stage_eval_external}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + ["train"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(12) as ex:
        if a.stage == "train":
            stage_train(ex, dev, *a.args)
        else:
            STAGES[a.stage](ex, dev)


if __name__ == "__main__":
    main()
