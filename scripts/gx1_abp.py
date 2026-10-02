"""GX1 — cross-target generalization of partial DualReadout sharing, PPG -> ABP (docs/GX1_ABP_GENERALIZATION_PREREGISTRATION.md).

The frozen DP0 S1 MIDDLE topology (shared stem + blocks 1-6; private blocks 7-8 + decoder per readout) is transferred
without any sharing-depth search. Only the target interface changes: the target is MIMIC-BP ABP under the frozen A8 global
TRAIN-only z normalization (artifacts/a8_abp_scale_control/normalization.json). Data: the pre-existing official MIMIC-BP split
(train 1,100 / val 195 / test 229) as GX-TRAIN / GX-DEV / GX-LOCK; GX-LOCK is read only after the committed final freeze.
Every trainer is the frozen DP0 trainer (scripts/dp0_dualreadout.py, imported, never modified), fed GX-TRAIN tensors.

Stages: build <train|dev>, margins, train_detector, train <point|gen|dual> <42|43|44>, dev, compute, freeze, eval_lock,
        train_diag <S0|S2>, eval_diag, summarize
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/gx1_abp.py <stage> [args]
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bf0_run as B  # noqa: E402
import c0_coherentbeat as C0  # noqa: E402
import dp0_dualreadout as F  # noqa: E402
import dp3_mimicbp as DP3  # noqa: E402
from ppg2ecg.anchorflow import fastfd as FF  # noqa: E402
from ppg2ecg.coherentbeat import ablation as AB  # noqa: E402
from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows  # noqa: E402
from ppg2ecg.data.target_norm import TargetNorm  # noqa: E402
from ppg2ecg.dualreadout import model as D  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as RP  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import RhythmTCN  # noqa: E402
from ppg2ecg.rhythmfield import model as RM  # noqa: E402
from ppg2ecg.scaleflow import model as SM  # noqa: E402

PREREG = "docs/GX1_ABP_GENERALIZATION_PREREGISTRATION.md"
CODE_FILES = ("scripts/gx1_abp.py", "tests/test_gx1_abp.py")
ART = ROOT / "artifacts/gx1_abp"
OUT = ROOT / "outputs/gx1_abp"
FREEZE = ART / "final_freeze_manifest.json"
MIMIC = ROOT / "data/raw/MIMIC-BP"
SPLIT = ROOT / "data/manifests/split_a7_mimicbp_official.json"
NORM = ROOT / "artifacts/a8_abp_scale_control/normalization.json"
ROLE_KEY = {"train": "train", "dev": "val", "lock": "test"}
SEEDS = (42, 43, 44)
FS, T, FS_RAW, N_SEG, SEG_LEN, WIN_S = 128, 512, 125, 30, 3750, 4
WIN_RAW = FS_RAW * WIN_S
WIN_PER_SEG = SEG_LEN // WIN_RAW
HR_RANGE = (30.0, 200.0)
BOOT_N, BOOT_SEED, SHUF_SEED = 2000, 20261002, 20261002
M_CORR = 0.02                 # G1 (DP0 corr margin)
EVIDENCE_LABEL = "CROSS-TASK GENERALIZATION (historically used ABP dataset; not fresh external confirmation)"


def write_json(name, obj):
    p = ART / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(B.clean(obj), indent=1, ensure_ascii=False))


def read_json(name):
    return json.loads((ART / name).read_text())


def ci(v, pid):
    return C0.cluster_ci(v, pid, BOOT_N, BOOT_SEED)


def tnorm() -> TargetNorm:
    return TargetNorm.load(NORM)


# ----------------------------------------------------------------------------------------------- seals
def check_final_freeze():
    """GX-LOCK subjects are read only after a committed, unchanged final freeze and preregistration."""
    if not FREEZE.exists() or not (ROOT / PREREG).exists():
        raise PermissionError("GX1: GX-LOCK sealed (no final freeze / preregistration)")
    fm = json.loads(FREEZE.read_text())
    for f, h in fm["sha256"].items():
        if B.sha256_file(ROOT / f) != h:
            raise PermissionError(f"GX1: frozen file changed: {f}")
    for rel in (str(FREEZE.relative_to(ROOT)), PREREG):
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
        if not (tracked and clean):
            raise PermissionError(f"GX1: {rel} not committed")


def role_subjects(role):
    sp = json.loads(SPLIT.read_text())["splits"][0]
    return sorted(sp[ROLE_KEY[role]])


# ----------------------------------------------------------------------------------------------- data adapter
def _subject(args):
    """One subject: 4 s windows (7 per 30 s segment), rules R1-R5, PPG (PPG_KW), ABP (FFT resample only, mmHg), ECG
    (ECG_KW, only for the reference-R raster / detector targets / R4)."""
    rank, pid = args
    try:
        ppg, abp, ecg = (np.load(MIMIC / k / f"{pid}_{k}.npy") for k in ("ppg", "abp", "ecg"))
        assert ppg.shape == abp.shape == ecg.shape == (N_SEG, SEG_LEN)
    except Exception as e:
        return pid, None, [(pid, -1, -1, "R1 unreadable or wrong shape", str(e)[:80])]
    cut = lambda a: a[:, :WIN_PER_SEG * WIN_RAW].reshape(N_SEG * WIN_PER_SEG, WIN_RAW).astype(np.float64)  # noqa: E731
    P, A, E = cut(ppg), cut(abp), cut(ecg)
    seg, k = np.repeat(np.arange(N_SEG), WIN_PER_SEG), np.tile(np.arange(WIN_PER_SEG), N_SEG)
    ok = np.isfinite(P).all(1) & np.isfinite(A).all(1) & np.isfinite(E).all(1)
    with np.errstate(invalid="ignore"):
        ok &= (np.nan_to_num(P).std(1) > 0) & (np.nan_to_num(A).std(1) > 0) & (np.nan_to_num(E).std(1) > 0)
    log = [(pid, int(seg[j]), int(k[j]), "R2 raw non-finite or constant", "") for j in np.flatnonzero(~ok)]
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return pid, None, log + [(pid, -1, -1, "R5 no remaining window", "")]
    x = preprocess_windows(P[idx], FS, WIN_S, **PPG_KW)
    a = signal.resample(A[idx], FS * WIN_S, axis=1)                          # ABP: resample only (A7 / A8 / EXP-D convention)
    e = preprocess_windows(E[idx], FS, WIN_S, **ECG_KW)
    ok2 = np.isfinite(x).all(1) & np.isfinite(a).all(1) & np.isfinite(e).all(1)
    log += [(pid, int(seg[idx[j]]), int(k[idx[j]]), "R3 non-finite after preprocessing", "") for j in np.flatnonzero(~ok2)]
    peaks = [np.asarray(RP.detect_rpeaks(row, FS, "neurokit"), int) if g else np.zeros(0, int) for row, g in zip(e, ok2)]
    hr = np.array([RP.hr_bpm(pk, FS) if g else np.nan for pk, g in zip(peaks, ok2)])
    ok3 = ok2 & np.isfinite(hr) & (hr >= HR_RANGE[0]) & (hr <= HR_RANGE[1])
    log += [(pid, int(seg[idx[j]]), int(k[idx[j]]), "R4 reference HR not finite or outside [30, 200] bpm", f"{hr[j]:.1f}") for j in np.flatnonzero(ok2 & ~ok3)]
    keep = np.flatnonzero(ok3)
    if keep.size == 0:
        return pid, None, log + [(pid, -1, -1, "R5 no remaining window", "")]
    wid = np.array([rank * N_SEG * WIN_PER_SEG + int(seg[idx[j]]) * WIN_PER_SEG + int(k[idx[j]]) for j in keep], np.int64)
    return pid, {"x": x[keep].astype(np.float32), "abp_mmhg": a[keep].astype(np.float32), "wid": wid, "ref": [peaks[j] for j in keep]}, log


def build_role(role, ex):
    """GX-TRAIN / GX-DEV are built before the freeze; GX-LOCK only through load_role('lock') after it."""
    if role == "lock":
        check_final_freeze()
    ids = role_subjects(role)
    X, A, Pid, W, ref, log = [], [], [], [], [], []
    for pid, d, lg in ex.map(_subject, list(enumerate(ids)), chunksize=8):
        log += lg
        if d is not None:
            X.append(d["x"]); A.append(d["abp_mmhg"]); W.append(d["wid"]); ref += d["ref"]; Pid.append(np.full(len(d["wid"]), int(pid[1:])))
    X, A, Pid, W = np.concatenate(X), np.concatenate(A), np.concatenate(Pid), np.concatenate(W)
    idx, off = B.pack_list(ref)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"data_{role}.npz", x=X, abp_mmhg=A, pid=Pid, wid=W, ref_idx=idx, ref_off=off)
    with open(ART / f"exclusion_log_{role}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "segment", "window", "rule", "detail"])
        w.writerows(log)
    from collections import Counter
    return {"subjects": len(ids), "subjects_eligible": int(np.unique(Pid).size), "windows": int(len(X)),
            "windows_possible": len(ids) * N_SEG * WIN_PER_SEG, "exclusions": dict(Counter(r[3].split(" ")[0] for r in log))}


def load_role(role):
    if role == "lock":
        check_final_freeze()
    d = np.load(OUT / f"data_{role}.npz")
    ref = [np.asarray(r, int) for r in B.split_list(d["ref_idx"], d["ref_off"])]
    return d["x"], d["abp_mmhg"].astype(np.float64), d["pid"], d["wid"], ref


def stage_build(ex, dev, role):
    if role not in ("train", "dev"):
        raise SystemExit("STOP: build only train / dev before the freeze (lock is built by eval_lock)")
    info = build_role(role, ex)
    sm = read_json("split_manifest.json") if (ART / "split_manifest.json").exists() else {}
    sm.setdefault("built", {})[role] = info
    write_json("split_manifest.json", sm)
    print(f"[gx1] built {role}: {json.dumps(info)}", flush=True)


# ----------------------------------------------------------------------------------------------- margins (TRAIN only)
def stage_margins(ex, dev):
    """G2: 0.02 x TRAIN SD of the normalized ABP target. G3: the ECG FD margin (+1.0) times the ratio of TRAIN target total
    variances tr(Cov) in the units D_ABP is computed (mmHg^2) and DP0's ECG FD units. TRAIN data only."""
    _, A, _, _, _ = load_role("train")
    tn = tnorm()
    z = (A - tn.mu) / tn.sigma
    tr_abp = float(A.var(axis=0, ddof=1).sum())
    _, Ye, Pa = C0.load_arch("train")
    dp_train = json.loads((ROOT / "artifacts/dp0_dualreadout/split_manifest.json").read_text())["roles"]["dp_train"]["patients"]
    ye = Ye[np.isin(Pa, dp_train)]
    tr_ecg = float(ye.var(axis=0, ddof=1).sum())
    m = {"G1_corr_margin": -M_CORR,
         "G2_mae_margin_z": M_CORR * float(z.std()), "G2_mae_margin_mmhg_equiv": M_CORR * float(z.std()) * tn.sigma,
         "G2_derivation": "the ECG point gate used 0.02 on a unit-free correlation; the same 0.02 is applied as a fraction of the GX-TRAIN SD of the "
                          "normalized (A8 global z) ABP target, the unit of the normalized-waveform MAE",
         "train_target_sd_z": float(z.std()), "G3_trace_abp_mmhg2": tr_abp, "G3_trace_ecg_dp_train": tr_ecg,
         "G3_margin_mmhg2": 1.0 * tr_abp / tr_ecg,
         "G3_derivation": "DP0 FD margin +1.0 (ECG units) x tr(Cov_GX-TRAIN ABP windows, mmHg) / tr(Cov_DP-TRAIN ECG windows); FD scales with the "
                          "square of the signal unit, so the trace ratio converts the margin; TRAIN data only",
         "n_train_windows_abp": int(len(A)), "n_train_windows_ecg": int(len(ye))}
    write_json("margins.json", m)
    print(f"[gx1] margins: {json.dumps(m)}", flush=True)


# ----------------------------------------------------------------------------------------------- training (frozen DP0 trainers)
def _gx_train_data(ex, dev):
    X, A, _, _, ref = load_role("train")
    tn = tnorm()
    y = ((A - tn.mu) / tn.sigma).astype(np.float32)
    return (torch.from_numpy(X).to(dev), torch.from_numpy(y).to(dev), torch.from_numpy(AB.event_raster(ref)).to(dev), ref)


def _saver(tag):
    def _save(name, net, secs, nan_steps, extra=None):
        d = OUT / tag
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{name}.pt"
        meta = {"seed": F.SEED, "train_seconds": secs, "n_params": D.n_params(net), "nan_steps": nan_steps,
                "peak_gpu_mem_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "optimizer": "AdamW",
                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                "target": "MIMIC-BP ABP, A8 global z", "training_role": "GX-TRAIN"} | (extra or {})
        torch.save({"state_dict": net.state_dict()} | meta, f)
        meta["sha256"] = B.sha256_file(f)
        write_json(f"checkpoints/{tag}_{name}.json", meta)
    return _save


def stage_train_detector(ex, dev):
    """RD1 / C0 detector protocol (as DP0 on DP-TRAIN) on GX-TRAIN, seed 42: conditioning infrastructure, not one of the 9 jobs."""
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, _, _, _, ref = load_role("train")
    Fd = torch.from_numpy(np.stack(list(ex.map(C0._field, ref, chunksize=512))).astype(np.float16)).to(dev)
    Xt = torch.from_numpy(X).to(dev)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    net = RhythmTCN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=F.DET["lr"], weight_decay=F.DET["wd"])
    lossf = nn.BCEWithLogitsLoss()
    batches, t0, nan_steps = C0._epoch_batches(len(Xt), F.DET["batch"], torch.Generator().manual_seed(42)), time.time(), 0
    net.train()
    for _ in range(F.DET["steps"]):
        b = next(batches).to(dev)
        loss = lossf(net(Xt[b][:, None])[:, 0], Fd[b].float())
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward(); opt.step()
    F.SEED = 42
    _saver("detector")("detector", net, time.time() - t0, nan_steps, {"protocol": F.DET})


def stage_train(ex, dev, role, seed):
    seed = int(seed)
    if seed not in SEEDS or role not in ("point", "gen", "dual"):
        raise SystemExit("STOP: point / gen / dual for seeds 42 / 43 / 44 only")
    F.SEED, F.OUT, F._save, F._train_data = seed, OUT / f"seed{seed}", _saver(f"seed{seed}"), _gx_train_data
    {"point": F.stage_train_point, "gen": F.stage_train_gen}.get(role, lambda e, d: F.stage_train_dual(e, d, "S1"))(ex, dev)
    _record_job(seed, {"point": "point", "gen": "gen", "dual": "S1"}[role])


def stage_train_diag(ex, dev, share):
    """Optional diagnostic controls (seed 42), trained only after the primary freeze; never replace S1."""
    if share not in ("S0", "S2") or not FREEZE.exists():
        raise SystemExit("STOP: diagnostics S0 / S2 only after the primary final freeze")
    F.SEED, F.OUT, F._save, F._train_data = 42, OUT / "diag42", _saver("diag42"), _gx_train_data
    F.stage_train_dual(ex, dev, share)
    _record_job(42, share, diag=True)


def _record_job(seed, name, diag=False):
    tag = "diag42" if diag else f"seed{seed}"
    meta = read_json(f"checkpoints/{tag}_{name}.json")
    man = read_json("training_manifest.json") if (ART / "training_manifest.json").exists() else {"jobs": []}
    man["jobs"] = [j for j in man["jobs"] if not (j["seed"] == seed and j["role"] == name and j.get("diagnostic") == diag)] + [{
        "seed": seed, "role": name, "diagnostic": diag, "git_commit": meta["git_commit"], "params": meta["n_params"], "lr": F.PROTO["lr"],
        "batch": F.PROTO["batch"], "updates": meta.get("updates", F.PROTO["steps"]), "train_seconds": meta["train_seconds"],
        "nan_steps": meta["nan_steps"], "peak_gpu_mem_mib": meta["peak_gpu_mem_mib"], "checkpoint_sha256": meta["sha256"]}]
    write_json("training_manifest.json", man)


def ckpt(role, seed, diag=False):
    return OUT / ("diag42" if diag else f"seed{seed}") / f"{role}.pt"


def load_model(role, seed, dev, diag=False):
    c = torch.load(ckpt(role, seed, diag), map_location="cpu")
    net = F.build(role)
    net.load_state_dict(c["state_dict"])
    return net.to(dev).eval()


def load_detector(dev):
    c = torch.load(OUT / "detector" / "detector.pt", map_location="cpu")
    net = RhythmTCN()
    net.load_state_dict(c["state_dict"])
    return net.to(dev).eval()


# ----------------------------------------------------------------------------------------------- evaluation
def context(role, ex, dev):
    X, A, Pid, W, ref = load_role(role)
    det = load_detector(dev)
    with torch.no_grad():
        prob = [torch.sigmoid(det(torch.from_numpy(X[i:i + 4096]).to(dev)[:, None])[:, 0]).float().cpu().numpy() for i in range(0, len(X), 4096)]
    ev = [np.asarray(e, int) for e in ex.map(F._extract, list(np.concatenate(prob)), chunksize=512)]
    tn = tnorm()
    return {"X": X, "A": A, "Z": (A - tn.mu) / tn.sigma, "Pid": Pid, "wid": W, "ref": ref, "ev": ev, "R": AB.event_raster(ev),
            "noise": SM.window_noise(Pid, W), "perm": np.random.default_rng(SHUF_SEED).permutation(len(X))}


def wave_corr(p, y):
    p, y = p - p.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    den = np.sqrt((p ** 2).sum(1) * (y ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (p * y).sum(1) / den, np.nan)


def bp_errors(pred_mmhg, A):
    return {"sbp": np.abs(pred_mmhg.max(1) - A.max(1)), "dbp": np.abs(pred_mmhg.min(1) - A.min(1)), "map": np.abs(pred_mmhg.mean(1) - A.mean(1))}


def summarize(pz, ctx):
    tn = tnorm()
    pm = pz * tn.sigma + tn.mu
    c, mae = wave_corr(pz, ctx["Z"]), np.abs(pz - ctx["Z"]).mean(1)
    bp = bp_errors(pm, ctx["A"])
    Pid = ctx["Pid"]
    return {"corr": c, "mae": mae, **bp}, {"corr": ci(c, Pid), "mae_z": ci(mae, Pid), "mae_mmhg": ci(np.abs(pm - ctx["A"]).mean(1), Pid),
                                           **{f"{k}_mae_mmhg": ci(v, Pid) for k, v in bp.items()}}


def evaluate_pair(ctx, nets, ex, dev, with_shuffle=True):
    """P / S1 point and G / S1 generative of one seed (or diagnostic S1-family net) on one population."""
    X, R, Pid, perm = ctx["X"], ctx["R"], ctx["Pid"], ctx["perm"]
    tn = tnorm()
    pts = {k: DP3.point_infer(n, X, R, dev) for k, n in nets.items() if k in ("P", "S1")}
    gens = {k: DP3.gen_infer(n, X, R, ctx["noise"], dev) for k, n in (("G", nets["G"]), ("S1", nets["S1"]))}
    shuf = {k: DP3.gen_infer(n, X[perm], R, ctx["noise"], dev) for k, n in (("G", nets["G"]), ("S1", nets["S1"]))} if with_shuffle else {}
    raw_p = {k: summarize(v, ctx) for k, v in pts.items()}
    raw_g = {k: summarize(v, ctx) for k, v in gens.items()}
    raw_s = {k: summarize(v, ctx) for k, v in shuf.items()}
    mm = lambda z: z * tn.sigma + tn.mu  # noqa: E731
    arms = {"G": mm(gens["G"]), "S1": mm(gens["S1"])} | ({"G_shuf": mm(shuf["G"]), "S1_shuf": mm(shuf["S1"])} if with_shuffle else {})
    prs = [("S1", "G")] + ([("S1_shuf", "S1"), ("G_shuf", "G")] if with_shuffle else [])
    fdb = FF.fd_bootstrap(arms, ctx["A"], Pid, prs, B.patient_resamples(np.unique(Pid).size, BOOT_N, BOOT_SEED))
    d = {"corr": ci(raw_p["S1"][0]["corr"] - raw_p["P"][0]["corr"], Pid), "mae_z": ci(raw_p["S1"][0]["mae"] - raw_p["P"][0]["mae"], Pid),
         "D": fdb["S1-G"], **{f"{k}_mae_desc": ci(raw_p["S1"][0][k] - raw_p["P"][0][k], Pid) for k in ("sbp", "dbp", "map")}}
    if with_shuffle:
        d |= {"D_shuf": fdb["S1_shuf-S1"], "corr_shuf": ci(raw_s["S1"][0]["corr"] - raw_g["S1"][0]["corr"], Pid),
              "G_D_shuf": fdb["G_shuf-G"], "G_corr_shuf": ci(raw_s["G"][0]["corr"] - raw_g["G"][0]["corr"], Pid)}
    out = {"windows": int(len(X)), "patients": int(np.unique(Pid).size), "point": {k: v[1] for k, v in raw_p.items()},
           "gen": {k: v[1] | {"D_mmhg2": fdb["fd"][k]} for k, v in raw_g.items()}, "comparisons": d}
    if with_shuffle:
        out["shuffle"] = {k: v[1] | {"D_mmhg2": fdb["fd"][f"{k}_shuf"]} for k, v in raw_s.items()}
    return out


def gates(d, margins):
    acc = F.accounting()
    g = {"G1": d["corr"][1] > -M_CORR, "G2": d["mae_z"][2] < margins["G2_mae_margin_z"], "G3": d["D"][2] < margins["G3_margin_mmhg2"],
         "G4": d["D_shuf"][1] > 0 and d["corr_shuf"][2] < 0, "G5": D.saving(acc["S1"]["total"], acc["separate_waveform_params"]) >= 0.15}
    g = {k: bool(v) for k, v in g.items()}
    g["PASS"] = all(g.values())
    return g


def nets_for(seed, dev):
    return {"P": load_model("point", seed, dev), "G": load_model("gen", seed, dev), "S1": load_model("S1", seed, dev)}


def stage_dev(ex, dev):
    """GX-DEV characterization (descriptive only; no decision depends on it)."""
    ctx = context("dev", ex, dev)
    res = {s: evaluate_pair(ctx, nets_for(s, dev), ex, dev, with_shuffle=False) for s in SEEDS}
    write_json("dev_characterization.json", {"note": "GX-DEV, descriptive only", "results": res})
    for s, r in res.items():
        c = r["comparisons"]
        print(f"[gx1] dev seed {s}: P corr {r['point']['P']['corr'][0]:.4f} S1 {r['point']['S1']['corr'][0]:.4f} dcorr {c['corr'][0]:+.4f} "
              f"dMAE {c['mae_z'][0]:+.4f} D G {r['gen']['G']['D_mmhg2']:.1f} S1 {r['gen']['S1']['D_mmhg2']:.1f}", flush=True)


def stage_compute(ex, dev):
    """Separate-naive / separate-cached / S1_ABP (seed 42), the DP3 protocol on GX-DEV windows."""
    from torch.utils.flop_counter import FlopCounterMode
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, _, Pid_, wid_, _ = load_role("dev")
    pick = B.salted_rank("gx1-latency-v1", range(len(X)))[:50]
    torch.set_num_threads(4)
    acc = F.accounting()
    lat, mem, eq, fl = {}, {}, {}, {}
    for dname in ("cuda", "cpu"):
        d = torch.device(dname)
        det = load_detector(d)
        P, G, S1 = load_model("point", 42, d), load_model("gen", 42, d), load_model("S1", 42, d)

        def raster(xi):
            p = torch.sigmoid(det(xi[:, None])[:, 0]).float().cpu().numpy()[0]
            return torch.from_numpy(AB.event_raster([F._extract(p)])).to(d)

        def req(system, mode, xi, x0):
            if system == "naive":
                if mode == "point":
                    return P(xi, raster(xi))
                if mode == "gen":
                    return SM.euler(G, x0, xi, raster(xi), F.NFE)
                return P(xi, raster(xi)), SM.euler(G, x0, xi, raster(xi), F.NFE)
            r = raster(xi)
            if system == "cached":
                return {"point": lambda: P(xi, r), "gen": lambda: DP3.euler_cached(G, x0, xi, r, F.NFE),
                        "both": lambda: (P(xi, r), DP3.euler_cached(G, x0, xi, r, F.NFE))}[mode]()
            if mode == "point":
                return S1.point(xi, r)
            if mode == "gen":
                return D.euler(S1, x0, xi, r, F.NFE)
            mu, c = S1.both(xi, r)
            return mu, D.euler(S1, x0, xi, r, F.NFE, cond=c)

        n = 200 if dname == "cuda" else 60
        for system in ("naive", "cached", "s1"):
            for mode in ("point", "gen", "both"):
                ts = []
                for rep in range(n + 10):
                    i = int(pick[rep % len(pick)])
                    xi = torch.from_numpy(X[i:i + 1]).to(d)
                    x0 = torch.from_numpy(SM.window_noise(Pid_[i:i + 1], wid_[i:i + 1])).to(d)
                    with torch.no_grad():
                        if dname == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        req(system, mode, xi, x0)
                        if dname == "cuda":
                            torch.cuda.synchronize()
                    if rep >= 10:
                        ts.append((time.perf_counter() - t0) * 1000)
                q = np.percentile(ts, [25, 50, 75])
                lat[f"{dname}_{system}_{mode}"] = {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]), "n": n}
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
                eq["cached_equals_naive_generation"] = bool(torch.equal(SM.euler(G, x0, xi, ri, F.NFE), DP3.euler_cached(G, x0, xi, ri, F.NFE)))
                h = S1.trunk(xi, ri)
                cond = S1.flow_features(h)
            z, t0_ = torch.zeros(1, T), torch.zeros(1)

            def flc(fn):
                fc = FlopCounterMode(display=False)
                with fc, torch.no_grad():
                    fn()
                return int(fc.get_total_flops())
            f_det, f_P, f_G = flc(lambda: det(xi[:, None])), flc(lambda: P(xi, ri)), flc(lambda: G(z, t0_, xi, ri))
            hp, hr = DP3.haar_cond(xi, ri)
            f_Gc = flc(lambda: DP3.scaleflow_cached(G, z, t0_, hp, hr))
            f_tr, f_pb = flc(lambda: S1.trunk(xi, ri)), flc(lambda: S1.point_dec(S1.point_features(h)))
            f_ff, f_v = flc(lambda: S1.flow_features(h)), flc(lambda: S1.velocity(z, t0_, cond))
            k = F.NFE
            fl = {"naive": {"point": f_det + f_P, "gen": f_det + k * f_G, "both": 2 * f_det + f_P + k * f_G},
                  "cached": {"point": f_det + f_P, "gen": f_det + k * f_Gc, "both": f_det + f_P + k * f_Gc},
                  "s1": {"point": f_det + f_tr + f_pb, "gen": f_det + f_tr + f_ff + k * f_v, "both": f_det + f_tr + f_pb + f_ff + k * f_v}}
    det_p = D.n_params(RhythmTCN())
    params = {"naive": acc["separate_waveform_params"], "cached": acc["separate_waveform_params"], "s1": acc["S1"]["total"]}
    for s in ("naive", "cached", "s1"):
        write_json(f"compute_{s}.json", {"system": s, "waveform_params": params[s], "pipeline_params": params[s] + (2 if s == "naive" else 1) * det_p,
                                         "flops_with_detector": fl[s], "latency_ms": {kk: v for kk, v in lat.items() if f"_{s}_" in kk},
                                         "gpu_peak_inference_mib_above_weights": {kk: v for kk, v in mem.items() if kk.startswith(s)},
                                         "numerical_equivalence": eq if s == "cached" else None})
    g6 = {"flops_lower": fl["s1"]["both"] < fl["cached"]["both"], "gpu_lower": lat["cuda_s1_both"]["median"] < lat["cuda_cached_both"]["median"],
          "cpu_lower": lat["cpu_s1_both"]["median"] < lat["cpu_cached_both"]["median"]}
    g6["G6_SUPPORTED"] = bool(g6["flops_lower"] and (g6["gpu_lower"] or g6["cpu_lower"]))
    write_json("compute_g6.json", g6)
    with open(ART / "table_efficiency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "waveform_params", "flops_both", "gpu_both_ms", "gpu_both_iqr", "cpu_both_ms", "cpu_both_iqr", "peak_mem_both_mib"])
        for s in ("naive", "cached", "s1"):
            g, c = lat[f"cuda_{s}_both"], lat[f"cpu_{s}_both"]
            w.writerow([s, params[s], fl[s]["both"], f"{g['median']:.2f}", f"{g['q25']:.2f}-{g['q75']:.2f}", f"{c['median']:.2f}",
                        f"{c['q25']:.2f}-{c['q75']:.2f}", f"{mem[f'{s}_both']:.3f}"])
    print(f"[gx1] compute: {json.dumps(g6)} GPU both {lat['cuda_naive_both']['median']:.2f} / {lat['cuda_cached_both']['median']:.2f} / "
          f"{lat['cuda_s1_both']['median']:.2f} ms", flush=True)


def freeze_files():
    files = [PREREG, *CODE_FILES, "scripts/dp0_dualreadout.py", "scripts/dp3_mimicbp.py", "src/ppg2ecg/dualreadout/model.py",
             "src/ppg2ecg/scaleflow/model.py", "src/ppg2ecg/data/preprocess.py", "src/ppg2ecg/data/target_norm.py", "src/ppg2ecg/evaluation/rpeaks.py",
             "src/ppg2ecg/evaluation/paper_metrics.py", "src/ppg2ecg/anchorflow/fastfd.py", "src/ppg2ecg/probes/rhythm_tcn.py", "scripts/bf0_run.py",
             "scripts/c0_coherentbeat.py", "data/manifests/split_a7_mimicbp_official.json", "artifacts/a8_abp_scale_control/normalization.json",
             "artifacts/gx1_abp/margins.json", "artifacts/gx1_abp/evaluator_manifest.json", "outputs/gx1_abp/detector/detector.pt"]
    files += [str(ckpt(r, s).relative_to(ROOT)) for s in SEEDS for r in ("point", "gen", "S1")]
    return files


def stage_freeze(ex, dev):
    write_json("final_freeze_manifest.json", {"evidence_label": EVIDENCE_LABEL, "primary_seed": 42, "seeds": list(SEEDS), "nfe": F.NFE,
                                              "bootstrap": {"replicates": BOOT_N, "seed": BOOT_SEED}, "shuffle_seed": SHUF_SEED,
                                              "sha256": {f: B.sha256_file(ROOT / f) for f in freeze_files()}})
    write_json("checkpoint_hashes.json", {f"{r}{s}": B.sha256_file(ckpt(r, s)) for s in SEEDS for r in ("point", "gen", "S1")} |
               {"detector": B.sha256_file(OUT / "detector" / "detector.pt")})


def stage_eval_lock(ex, dev):
    check_final_freeze()
    info = build_role("lock", ex)
    write_json("lock_cohort.json", info)
    ctx = context("lock", ex, dev)
    margins = read_json("margins.json")
    placed = PMX.rpeak_prf_at(ctx["Z"], ctx["Z"], FS, 50.0, peaks=(ctx["ref"], ctx["ev"]))
    pmp = RM.patient_macro_rows(placed["n_tp"], placed["n_fp"], placed["n_fn"], ctx["Pid"])
    write_json("detector_metrics.json", {f"pm_{k}": C0.cluster_ci(pmp[k], pmp["patients"], BOOT_N, BOOT_SEED) for k in ("fp_rate", "recall", "precision", "f1")}
               | {"note": "GX detector (GX-TRAIN) vs neurokit ECG R on GX-LOCK; descriptive"})
    allres = {}
    for s in SEEDS:
        r = evaluate_pair(ctx, nets_for(s, dev), ex, dev, with_shuffle=True)
        r["gates"] = gates(r["comparisons"], margins)
        write_json(f"seed{s}_metrics.json", {k: v for k, v in r.items() if k not in ("comparisons", "gates")})
        write_json(f"seed{s}_bootstrap.json", {"unit": "patient", "replicates": BOOT_N, "seed": BOOT_SEED, "comparisons": r["comparisons"]})
        write_json(f"seed{s}_gates.json", r["gates"] | {"margins": margins})
        allres[s] = r
        print(f"[gx1] GX-LOCK seed {s}: {json.dumps(r['gates'])}", flush=True)
        gc.collect()
    write_json("condition_shuffle.json", {s: {"S1": {"D_shuffled_minus_conditioned": r["comparisons"]["D_shuf"], "corr_shuffled_minus_conditioned":
                                                     r["comparisons"]["corr_shuf"]},
                                              "G_diagnostic": {"D_shuffled_minus_conditioned": r["comparisons"]["G_D_shuf"],
                                                               "corr_shuffled_minus_conditioned": r["comparisons"]["G_corr_shuf"]}} for s, r in allres.items()})
    passes = {s: allres[s]["gates"]["PASS"] for s in SEEDS}
    n = sum(passes.values())
    cls = "GENERALIZES-3/3" if n == 3 else "GENERALIZES-2/3" if n == 2 else "SEED-SENSITIVE"
    g6 = read_json("compute_g6.json")["G6_SUPPORTED"]
    p42 = passes[42]
    verdict = ("CROSS-TARGET PRINCIPLE STRONGLY SUPPORTED" if p42 and n == 3 and g6 else "CROSS-TARGET PRINCIPLE SUPPORTED" if p42 and n >= 2
               else "NOT SUPPORTED")
    write_json("multiseed_summary.json", {"seed_pass": passes, "classification": cls, "seed42_pass": p42, "G6_compute_supported": g6,
                                          "principle_verdict": verdict, "evidence_label": EVIDENCE_LABEL})
    print(f"[gx1] {cls} | {verdict}", flush=True)


def stage_eval_diag(ex, dev):
    """Diagnostic S0 / S2 controls (seed 42) on GX-LOCK, after the primary verdict; explanatory only."""
    check_final_freeze()
    if not (ART / "multiseed_summary.json").exists():
        raise SystemExit("STOP: diagnostics only after the primary verdict")
    ctx = context("lock", ex, dev)
    res = {}
    for share in ("S0", "S2"):
        nets = {"P": load_model("point", 42, dev), "G": load_model("gen", 42, dev), "S1": load_model(share, 42, dev, diag=True)}
        r = evaluate_pair(ctx, nets, ex, dev, with_shuffle=False)
        acc = F.accounting()[share]
        res[share] = r | {"params": acc["total"], "saving": acc["saving"]}
        print(f"[gx1] diag {share}: dcorr {r['comparisons']['corr'][0]:+.4f} dMAE {r['comparisons']['mae_z'][0]:+.4f} dD {r['comparisons']['D'][0]:+.2f}", flush=True)
    write_json("diagnostic_sharing_depth.json", {"note": "explanatory diagnostics after the primary verdict; cannot replace S1 or change any verdict", "results": res})


def _cs(v, nd=4):
    return "n/a" if v is None or any(x is None for x in v) else f"{v[0]:+.{nd}f} [{v[1]:+.{nd}f}, {v[2]:+.{nd}f}]"


def stage_summarize(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    acc = F.accounting()
    mg = read_json("margins.json")
    met = {s: read_json(f"seed{s}_metrics.json") for s in SEEDS}
    bt = {s: read_json(f"seed{s}_bootstrap.json")["comparisons"] for s in SEEDS}
    gt = {s: read_json(f"seed{s}_gates.json") for s in SEEDS}
    with open(ART / "table_main.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "role", "params", "corr", "mae_z", "mae_mmhg", "sbp_mae", "dbp_mae", "map_mae", "D_mmhg2", "same_seed_delta", "gates"])
        for s in SEEDS:
            m, b, g = met[s], bt[s], gt[s]
            gs = " ".join(f"{k}:{'P' if g[k] else 'F'}" for k in ("G1", "G2", "G3", "G4", "G5"))
            for role, par, r, dd in (("P_ABP", acc["specialist_point"], m["point"]["P"], ""), ("S1_ABP point", acc["S1"]["point_path"], m["point"]["S1"],
                                     f"dcorr {_cs(b['corr'])}; dMAE {_cs(b['mae_z'])}"),
                                     ("G_ABP", acc["specialist_gen"], m["gen"]["G"], ""), ("S1_ABP gen", acc["S1"]["total"], m["gen"]["S1"], f"dD {_cs(b['D'], 1)}")):
                w.writerow([s, role, par, f"{r['corr'][0]:.4f}", f"{r['mae_z'][0]:.4f}", f"{r['mae_mmhg'][0]:.2f}", f"{r['sbp_mae_mmhg'][0]:.2f}",
                            f"{r['dbp_mae_mmhg'][0]:.2f}", f"{r['map_mae_mmhg'][0]:.2f}", f"{r.get('D_mmhg2', float('nan')):.1f}" if "D_mmhg2" in r else "",
                            dd, gs if role.startswith("S1") else ""])
    ecg = json.loads((ROOT / "artifacts/dp3_mimicbp/internal_multiseed_metrics.json").read_text())["results"]["af_lock"]
    ecg_c = {k: json.loads((ROOT / f"artifacts/dp3_mimicbp/compute_{k}.json").read_text()) for k in ("cached", "s1")}
    abp_c = {k: read_json(f"compute_{k}.json") for k in ("cached", "s1")}
    cols = {42: "#4a3aa7", 43: "#2a7f3f", 44: "#d59a54"}
    fig, ax = plt.subplots(2, 4, figsize=(24, 10))
    for row, (task, src, key_c, key_g, m_g, glab) in enumerate((
            ("PPG -> ECG (internal AF-LOCK, DP0 / DP3)", {int(k): v["comparisons"] for k, v in ecg.items()}, "corr", "fd", 1.0, "delta FD"),
            ("PPG -> ABP (GX-LOCK, transferred S1)", bt, "corr", "D", mg["G3_margin_mmhg2"], "delta D_ABP (mmHg^2)"))):
        for j, (key, title, marg) in enumerate(((key_c, "delta point corr (S1 - P)", -M_CORR), (key_g, f"{glab} (S1 - G)", m_g))):
            a = ax[row, j]
            for i, s in enumerate(SEEDS):
                v = src[s][key]
                a.plot([i, i], [v[1], v[2]], color=cols[s], lw=4)
                a.plot([i], [v[0]], "o", color=cols[s], ms=7)
            a.axhline(marg, color="k", ls="--", lw=1)
            a.axhline(0, color="0.75", lw=0.6)
            a.set_xticks(range(3), [f"seed {s}" for s in SEEDS])
            a.set_title(f"{task}\n{title}; dashed = NI margin", loc="left", fontsize=9)
        a = ax[row, 2]
        a.bar([0, 1], [acc["separate_waveform_params"], acc["S1"]["total"]], color=["0.55", "#4a3aa7"])
        a.set_xticks([0, 1], ["separate P + G", "S1 (same topology)"])
        a.set_title(f"parameters: saving {100 * acc['S1']['saving']:.2f} %", loc="left", fontsize=9)
        a = ax[row, 3]
        cc = ecg_c if row == 0 else abp_c
        if row == 0:
            fl = [cc[k]["flops_with_detector"]["both"] / 1e9 for k in ("cached", "s1")]
            gl = [cc[k]["latency_ms"][f"cuda_{k}_both_with_detector"]["median"] for k in ("cached", "s1")]
        else:
            fl = [cc[k]["flops_with_detector"]["both"] / 1e9 for k in ("cached", "s1")]
            gl = [cc[k]["latency_ms"][f"cuda_{k}_both"]["median"] for k in ("cached", "s1")]
        a.bar([0, 1], gl, color=["#d59a54", "#4a3aa7"])
        for i in range(2):
            a.text(i, gl[i], f"{gl[i]:.2f} ms\n{fl[i]:.2f} GFLOPs", ha="center", va="bottom", fontsize=8)
        a.set_xticks([0, 1], ["separate-cached", "S1"])
        a.set_ylim(0, max(gl) * 1.35)
        a.set_title(f"both outputs, GPU batch-1 (compute saving {100 * (1 - fl[1] / fl[0]):.0f} % FLOPs)", loc="left", fontsize=9)
    ms = read_json("multiseed_summary.json")
    fig.suptitle("SAME sharing topology (shared stem + blocks 1-6; private blocks 7-8 + decoders) transferred from PPG->ECG to PPG->ABP, "
                 f"no sharing-depth search | ABP: {ms['classification']} | {ms['principle_verdict']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(ART / "figure_generalization.png", dpi=105)


STAGES = {"margins": stage_margins, "train_detector": stage_train_detector, "dev": stage_dev, "compute": stage_compute, "freeze": stage_freeze,
          "eval_lock": stage_eval_lock, "eval_diag": stage_eval_diag, "summarize": stage_summarize}
ARG_STAGES = {"build": stage_build, "train": stage_train, "train_diag": stage_train_diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + list(ARG_STAGES))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(12) as ex:
        if a.stage in ARG_STAGES:
            ARG_STAGES[a.stage](ex, dev, *a.args)
        else:
            STAGES[a.stage](ex, dev)


if __name__ == "__main__":
    main()
