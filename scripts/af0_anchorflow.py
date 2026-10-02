"""AF0 — AnchorFlow-ECG development program (docs/AF0_ANCHORFLOW_DEVELOPMENT_PROTOCOL.md).

Nested split of SF-TRAIN (3,037 patients, seed 20261002): AF-TRAIN 2,400 / AF-DEV 300 / AF-LOCK 337. Detector and the
deterministic anchor are trained once on AF-TRAIN and frozen. Residual flows model x - mu(c) (AF-TRAIN only). AF-DEV is
the adaptive development population (bounded, predeclared search); AF-LOCK opens once, after a committed freeze, for a
single winner. The old V1 TEST is never loaded by AF0.

Stages: split, audit, manifest, train_detector, train_anchor, prep, train <job>, eval_baselines, eval_cand <id>, nfe <id>,
        freeze <id>, eval_lock, compute, figure_dev, seeds <id>
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/af0_anchorflow.py <stage> [arg]
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bf0_run as B  # noqa: E402
import c0_coherentbeat as C0  # noqa: E402
import c0a_ablation as CA  # noqa: E402
import sf0_scaleflow as SF  # noqa: E402
from ppg2ecg.anchorflow import fastfd as FF  # noqa: E402
from ppg2ecg.anchorflow import model as A  # noqa: E402
from ppg2ecg.beatfirst import render as BR  # noqa: E402
from ppg2ecg.coherentbeat import ablation as AB  # noqa: E402
from ppg2ecg.coherentbeat import split as SP  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, extract_events  # noqa: E402
from ppg2ecg.rhythmfield import model as RM  # noqa: E402
from ppg2ecg.scaleflow import model as SM  # noqa: E402

PROTOCOL = "docs/AF0_ANCHORFLOW_DEVELOPMENT_PROTOCOL.md"
CODE_FILES = ("scripts/af0_anchorflow.py", "src/ppg2ecg/anchorflow/__init__.py", "src/ppg2ecg/anchorflow/model.py",
              "src/ppg2ecg/anchorflow/fastfd.py", "tests/test_af0_anchorflow.py")
ART = ROOT / "artifacts/af0_anchorflow"
OUT = ROOT / "outputs/af0_anchorflow"
LOCK_FREEZE = ART / "lock_freeze_manifest.json"
FS, T = 128, 512
SEED, SPLIT_SEED, BOOT_SEED, BOOT_N = 42, 20261002, 20261002, 2000
COUNTS = {"af_dev": 300, "af_lock": 337, "af_train": 2400}
PROTO = {"steps": 20000, "batch": 64, "lr": 1e-3, "wd": 0.01, "clip": 1.0}
DET = dict(C0.DET)
N_K16, K = 2000, 16
MAX_CANDIDATES, MAX_JOBS = 10, 30
MARG = {"fd_vs_sf": 1.0, "corr": 0.02, "fp": 0.05, "recall": 0.01, "div": (0.5, 1.5)}
BASE_CFG = {"norm": "A0", "anchor": "B0", "coupling": "C1", "width_mult": 1.0, "nfe": 8, "amp": "F0", "ppg_dropout": 0.0}
HISTORY_COLS = ("candidate_id", "parent_id", "change", "params", "NFE", "FD", "residual_FD", "mean16_corr", "mean16_FP", "mean16_recall",
                "diversity_ratio", "PPG_shuffle_effect", "anchor_shuffle_effect", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "overall")


def write_json(name, obj):
    p = ART / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(B.clean(obj), indent=1))


def read_json(name):
    return json.loads((ART / name).read_text())


def ci(v, pid):
    return C0.cluster_ci(v, pid, BOOT_N, BOOT_SEED)


# ----------------------------------------------------------------------------------------------- split / data
def af_split(sf_train_patients) -> dict:
    """default_rng(20261002).permutation of the sorted SF-TRAIN patients: first 300 -> AF-DEV, next 337 -> AF-LOCK, rest AF-TRAIN."""
    p = np.array(sorted(int(x) for x in sf_train_patients))
    perm = np.random.default_rng(SPLIT_SEED).permutation(p.size)
    dev = sorted(int(x) for x in p[perm[:300]])
    lock = sorted(int(x) for x in p[perm[300:637]])
    train = sorted(int(x) for x in p[perm[637:]])
    return {"af_train": train, "af_dev": dev, "af_lock": lock}


def check_lock_freeze():
    """AF-LOCK stays sealed until a committed, unchanged lock-freeze manifest names exactly one development winner."""
    if not LOCK_FREEZE.exists():
        raise PermissionError("AF0: AF-LOCK sealed (no lock-freeze manifest)")
    fm = json.loads(LOCK_FREEZE.read_text())
    if not fm.get("winner"):
        raise PermissionError("AF0: AF-LOCK sealed (no development winner)")
    for f, h in fm["sha256"].items():
        if B.sha256_file(ROOT / f) != h:
            raise PermissionError(f"AF0: frozen file changed: {f}")
    rel = str(LOCK_FREEZE.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise PermissionError("AF0: lock-freeze manifest not committed")


def load_role(role, ex):
    """(X, Y, Pid, wid, ref) for af_train / af_dev / af_lock (af_lock only after the committed freeze)."""
    if role not in COUNTS:
        raise ValueError(role)
    if role == "af_lock":
        check_lock_freeze()
    X, Y, Pid = C0.load_arch("train")
    ref = C0.reference_peaks("train", Y, ex)
    pats = read_json("split_manifest.json")["roles"][role]["patients"]
    wid = np.flatnonzero(np.isin(Pid, pats))
    return X[wid], Y[wid], Pid[wid], wid, [ref[i] for i in wid]


# ----------------------------------------------------------------------------------------------- budget ledger
def ledger():
    p = ART / "training_jobs.json"
    return json.loads(p.read_text()) if p.exists() else {"jobs": []}


def register_job(name):
    lg = ledger()
    if len(lg["jobs"]) >= MAX_JOBS:
        raise SystemExit(f"STOP: training-job budget {MAX_JOBS} exhausted")
    lg["jobs"].append({"name": name, "time": time.strftime("%F %T")})
    write_json("training_jobs.json", lg)


def candidate_ids():
    d = ART / "candidate_configs"
    return sorted(p.stem for p in d.glob("c*.json")) if d.exists() else []


# ----------------------------------------------------------------------------------------------- stages: split / audit / manifest
def stage_split(ex, dev):
    sf = json.loads((ROOT / "artifacts/sf0_scaleflow/split_manifest.json").read_text())["roles"]
    sp = af_split(sf["sf_train"]["patients"])
    c0r = C0.split_info()["roles"]
    old = SP.old_heldout_patients(C0.manifest())
    allp = set().union(*map(set, sp.values()))
    checks = {"counts": {k: len(v) for k, v in sp.items()} == COUNTS, "union_is_sf_train": allp == set(sf["sf_train"]["patients"]),
              "disjoint": sum(len(v) for v in sp.values()) == len(allp),
              "no_arch_val_holdout": not allp & (set(c0r["val"]["patients"]) | set(c0r["holdout"]["patients"])),
              "no_sf_val": not allp & set(sf["sf_val"]["patients"]), "no_old_val_test": not allp & (set(old["val"]) | set(old["test"]))}
    if not all(checks.values()):
        raise SystemExit(f"STOP: split checks {checks}")
    X, Y, Pid = C0.load_arch("train")
    roles = {r: {"patients": sp[r], "n_patients": len(sp[r]), "n_windows": int(np.isin(Pid, sp[r]).sum())} for r in sp}
    write_json("split_manifest.json", {"source": "SF-TRAIN (artifacts/sf0_scaleflow/split_manifest.json)", "seed": SPLIT_SEED,
                                       "rule": "default_rng(20261002).permutation of sorted SF-TRAIN patients: [0:300] AF-DEV, [300:637] AF-LOCK, rest AF-TRAIN",
                                       "roles": roles, "checks": checks})
    write_json("split_hashes.json", {r: {"patients_sha256": hashlib.sha256(json.dumps(sp[r]).encode()).hexdigest(),
                                         "window_index_sha256": hashlib.sha256(np.flatnonzero(np.isin(Pid, sp[r])).astype(np.int64).tobytes()).hexdigest()}
                                     for r in sp})
    print(json.dumps({r: (roles[r]["n_patients"], roles[r]["n_windows"]) for r in roles}), flush=True)


def stage_audit(ex, dev):
    w_b2, p_b2 = A.width_for(lambda w: A.VanillaResidualFM(w))
    w_b3, p_b3 = A.width_for(lambda w: A.ResidualScaleFM(w, None, "B0"))
    w_b4, p_b4 = A.width_for(lambda w: A.ResidualScaleFM(w, "C1", "B0"))
    write_json("baseline_configs.json", {
        "B0_anchor": {"class": "WWDet(72, 5)", "params": A.n_params(SM.ww_l1()), "loss": "L1", "training_raster": "reference R (SF0 WW-L1 protocol)"},
        "B1_full_scaleflow": {"class": "scaleflow ScaleFM(50, coupled=True)", "params": A.n_params(SM.ScaleFM(50, True)), "protocol": "SF0 unchanged on AF-TRAIN"},
        "B2_vanilla_residual": {"class": "anchorflow VanillaResidualFM", "dec_width": w_b2, "params": p_b2, "norm": "waveform (global mean / std)",
                                "inputs": "[h(PPG), raster, mu, y_t]"},
        "B3_independent_residual": {"class": "anchorflow ResidualScaleFM(coupling=None, anchor=B0)", "width": w_b3, "params": p_b3, "norm": "A0"},
        "B4_base": {"class": "anchorflow ResidualScaleFM(coupling=C1, anchor=B0)", "width": w_b4, "params": p_b4, "norm": "A0"}})
    write_json("search_space.json", {
        "stage_A_norm": ["A0 per-band TRAIN mean/std", "A1 per-band TRAIN median/IQR", "A2 global TRAIN mean/std"],
        "stage_B_anchor_injection": ["B0 concat mu_s", "B1 FiLM from mu_s", "B2 concat + FiLM"],
        "stage_C_coupling": ["C0 additive projected features", "C1 concatenative fusion (SF0 form, base)", "C2 gated additive (learned channel-wise sigmoid gates, init 0)"],
        "stage_D_width_mult": [0.75, 1.0, 1.25], "stage_E_nfe": [4, 8, 16], "stage_F_amp": ["F0 none", "F1 global alpha", "F2 per-band alpha"],
        "stage_G_ppg_dropout": [0.0, 0.05, 0.10], "base": BASE_CFG, "max_candidates": MAX_CANDIDATES, "max_training_jobs": MAX_JOBS,
        "width_rule": "every structural variant is re-matched to ~600k by parameter count (16..160); stage D multiplies the matched width",
        "nfe_rule": "NFE variants of a trained candidate are evaluated only if its NFE-8 run passes D1-D7; lowest passing NFE kept"})
    write_json("haar_config.json", json.loads((ROOT / "artifacts/sf0_scaleflow/haar_config.json").read_text()) | {"source": "SF0, unchanged"})
    write_json("detector_config.json", {"family": "RD1 RhythmTCN", "protocol": DET, "training": "AF-TRAIN only, seed 42",
                                        "events": "threshold 0.35, refractory 32; one event sequence per window, shared by every model"})
    write_json("anchor_config.json", {"class": "WWDet(72, 5) (SF0 WW-L1)", "loss": "L1", "protocol": PROTO, "training": "AF-TRAIN, reference-R raster, seed 42",
                                      "inference": "AF detector events raster", "frozen": "after training; never updated during the search"})
    write_json("development_protocol.json", {"adaptive_population": "AF-DEV (300 patients)", "locked_population": "AF-LOCK (337), opened once after freeze",
                                             "old_test": "never opened in AF0", "gates": "D1-D7 (protocol §8)", "bootstrap": {"replicates": BOOT_N, "seed": BOOT_SEED},
                                             "k16": {"windows": N_K16, "K": K, "subset": "salted rank 'af0-k16-v1' over AF-DEV windows"},
                                             "noise": "scaleflow.window_noise (sha256 'patient:window:20261002', ':k{k}' for k > 0)",
                                             "budget": {"candidates": MAX_CANDIDATES, "training_jobs": MAX_JOBS}})


def stage_manifest(ex, dev):
    write_json("protocol_manifest.json", {"protocol": {PROTOCOL: B.sha256_file(ROOT / PROTOCOL)}, "code": {f: B.sha256_file(ROOT / f) for f in CODE_FILES},
                                          "design": {f: B.sha256_file(ART / f) for f in ("split_manifest.json", "split_hashes.json", "baseline_configs.json",
                                                                                         "search_space.json", "development_protocol.json", "anchor_config.json",
                                                                                         "detector_config.json", "haar_config.json")},
                                          "written_before_any_af_dev_metric": True, "software": B.software()})


# ----------------------------------------------------------------------------------------------- training
def _save(name, net, secs, nan_steps, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{name}.pt"
    meta = {"seed": extra.pop("seed", SEED) if extra else SEED, "train_seconds": secs, "n_params": A.n_params(net), "nan_steps": nan_steps,
            "checkpoint": "last step", "peak_gpu_mem_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "optimizer": "AdamW"} | (extra or {})
    torch.save({"state_dict": net.state_dict()} | meta, f)
    meta["sha256"] = B.sha256_file(f)
    write_json(f"checkpoints/{name}.json", meta)


def stage_train_detector(ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    register_job("detector")
    X, _, _, _, ref = load_role("af_train", ex)
    Fd = torch.from_numpy(np.stack(list(ex.map(C0._field, ref, chunksize=512))).astype(np.float16)).to(dev)
    Xt = torch.from_numpy(X).to(dev)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = RhythmTCN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=DET["lr"], weight_decay=DET["wd"])
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    batches, t0, nan_steps = C0._epoch_batches(len(Xt), DET["batch"], g), time.time(), 0
    net.train()
    for step in range(1, DET["steps"] + 1):
        b = next(batches).to(dev)
        loss = lossf(net(Xt[b][:, None])[:, 0], Fd[b].float())
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward(); opt.step()
    _save("detector", net, time.time() - t0, nan_steps, {"protocol": DET})


def stage_train_anchor(ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    register_job("anchor")
    X, Y, _, _, ref = load_role("af_train", ex)
    Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y.astype(np.float32)).to(dev)
    Rt = torch.from_numpy(AB.event_raster(ref)).to(dev)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = SM.ww_l1().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=PROTO["lr"], weight_decay=PROTO["wd"])
    g = torch.Generator().manual_seed(SEED)
    batches, t0, nan_steps = C0._epoch_batches(len(Xt), PROTO["batch"], g), time.time(), 0
    net.train()
    for step in range(1, PROTO["steps"] + 1):
        bd = next(batches).to(dev)
        loss = (net(Xt[bd], Rt[bd]) - Yt[bd]).abs().mean()
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), PROTO["clip"])
        opt.step()
    _save("anchor", net, time.time() - t0, nan_steps, PROTO | {"loss": "L1"})
    write_json("anchor_checkpoint_hash.json", {"anchor.pt": B.sha256_file(OUT / "anchor.pt")})


def load_ckpt(name, dev, net=None):
    ck = torch.load(OUT / f"{name}.pt", map_location="cpu")
    if net is None:
        net = RhythmTCN() if name == "detector" else SM.ww_l1() if name == "anchor" else SM.ScaleFM(50, True) if name == "b1" else None
    net.load_state_dict(ck["state_dict"])
    return net.to(dev).eval()


def _extract(p):
    return extract_events(p, DET["threshold"], DET["refractory"])


@torch.no_grad()
def detector_events(X, dev, ex):
    det = load_ckpt("detector", dev)
    out = [torch.sigmoid(det(torch.from_numpy(X[i:i + 2048]).to(dev)[:, None])[:, 0]).float().cpu().numpy() for i in range(0, len(X), 2048)]
    return [np.asarray(e, int) for e in ex.map(_extract, list(np.concatenate(out)), chunksize=256)]


@torch.no_grad()
def anchor_out(X, R, dev, bs=1024):
    net = load_ckpt("anchor", dev)
    return np.concatenate([net(torch.from_numpy(X[i:i + bs]).to(dev), torch.from_numpy(R[i:i + bs]).to(dev)).cpu().numpy()
                           for i in range(0, len(X), bs)])


def stage_prep(ex, dev):
    """Frozen detector events and frozen anchor outputs for AF-TRAIN and AF-DEV; AF-TRAIN-only residual statistics."""
    h = json.loads((ART / "anchor_checkpoint_hash.json").read_text())["anchor.pt"]
    assert B.sha256_file(OUT / "anchor.pt") == h
    stats = {}
    for role in ("af_train", "af_dev"):
        X, Y, Pid, wid, ref = load_role(role, ex)
        ev = detector_events(X, dev, ex)
        R = AB.event_raster(ev)
        mu = anchor_out(X, R, dev)
        idx, off = B.pack_list(ev)
        np.savez(OUT / f"{role}_prep.npz", mu=mu.astype(np.float32), ev_idx=idx, ev_off=off, wid=wid)
        if role == "af_train":
            r = (Y - mu.astype(np.float64))
            for kind in ("A0", "A1", "A2", "waveform"):
                stats[kind] = A.Normalizer.fit(kind, r).to_json()
            Z = A.to_bands(torch.from_numpy(r)).numpy()
            stats["descriptive"] = {b: {"mean": float(Z[:, s:e].mean()), "std": float(Z[:, s:e].std()), "rms": float(np.sqrt((Z[:, s:e] ** 2).mean()))}
                                    for b, s, e in A.BANDS}
    write_json("residual_stats.json", stats | {"source": "AF-TRAIN residuals r = ECG - mu (frozen anchor at the frozen detector's AF-TRAIN events)"})


def load_prep(role):
    d = np.load(OUT / f"{role}_prep.npz")
    return d["mu"].astype(np.float32), [np.asarray(e, int) for e in B.split_list(d["ev_idx"], d["ev_off"])]


def build_job(job: str):
    """job in b2 / b3 / candidate id -> (net, normalizer, cfg)."""
    st = read_json("residual_stats.json")
    if job == "b2":
        w, _ = A.width_for(lambda w: A.VanillaResidualFM(w))
        return A.VanillaResidualFM(w), A.Normalizer.from_json(st["waveform"]), {"kind": "B2", "ppg_dropout": 0.0}
    if job == "b3":
        w, _ = A.width_for(lambda w: A.ResidualScaleFM(w, None, "B0"))
        return A.ResidualScaleFM(w, None, "B0"), A.Normalizer.from_json(st["A0"]), {"kind": "B3", "ppg_dropout": 0.0}
    cfg = read_json(f"candidate_configs/{job}.json")
    w0, p0 = A.width_for(lambda w: A.ResidualScaleFM(w, cfg["coupling"], cfg["anchor"]))
    w = int(w0 * cfg["width_mult"])                                              # floor: stage D never exceeds the 1.5x cap
    net = A.ResidualScaleFM(w, cfg["coupling"], cfg["anchor"])
    if A.n_params(net) > 1.5 * p0:
        raise SystemExit(f"STOP: width {w} exceeds 1.5x the matched residual-flow parameters ({A.n_params(net)} > {1.5 * p0})")
    return net, A.Normalizer.from_json(st[cfg["norm"]]), cfg


def stage_train(ex, dev, job):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    seed = SEED
    if "@seed" in job:
        job, seed = job.split("@seed")[0], int(job.split("@seed")[1])
    register_job(job if seed == SEED else f"{job}@seed{seed}")
    X, Y, _, _, ref = load_role("af_train", ex)
    Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y.astype(np.float32)).to(dev)
    if job == "b1":                                                               # SF0 protocol unchanged
        Rt = torch.from_numpy(AB.event_raster(ref)).to(dev)
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        net = SM.ScaleFM(50, True).to(dev)
    else:
        mu, ev = load_prep("af_train")
        Mt, Rt = torch.from_numpy(mu).to(dev), torch.from_numpy(AB.event_raster(ev)).to(dev)
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        net, norm, cfg = build_job(job)
        net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=PROTO["lr"], weight_decay=PROTO["wd"])
    g = torch.Generator().manual_seed(seed)
    gd = torch.Generator(device=dev).manual_seed(seed)
    batches, t0, acc, nan_steps = C0._epoch_batches(len(Xt), PROTO["batch"], g), time.time(), [], 0
    net.train()
    for step in range(1, PROTO["steps"] + 1):
        bd = next(batches).to(dev)
        if job == "b1":
            loss = SM.fm_loss(net, Yt[bd], Xt[bd], Rt[bd], gd)
        else:
            y1 = norm.encode(Yt[bd] - Mt[bd])                                    # residual of the FROZEN anchor (no gradient path)
            loss = A.fm_loss(net, y1, Xt[bd], Rt[bd], Mt[bd], gd, cfg.get("ppg_dropout", 0.0))
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), PROTO["clip"])
        opt.step(); acc.append(loss.item())
        if step % 2000 == 0:
            print(f"[af0] {job} seed {seed} step {step} loss {np.mean(acc):.5f}", flush=True); acc = []
    _save(job if seed == SEED else f"{job}_seed{seed}", net, time.time() - t0, nan_steps, PROTO | {"seed": seed, "loss": "flow-matching MSE"})


# ----------------------------------------------------------------------------------------------- generation
@torch.no_grad()
def gen_residual(job, X, R, mu, noise, dev, nfe=8, bs=512, ckpt=None):
    net, norm, _ = build_job(job)
    net = load_ckpt(ckpt or job, dev, net)
    out = []
    for i in range(0, len(X), bs):
        sl = slice(i, i + bs)
        y = A.euler(net, torch.from_numpy(noise[sl]).to(dev), torch.from_numpy(X[sl]).to(dev), torch.from_numpy(R[sl]).to(dev),
                    torch.from_numpy(mu[sl]).to(dev), nfe)
        out.append(norm.decode(y.double()).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def gen_full_sf(X, R, noise, dev, nfe=8, bs=512):
    net = load_ckpt("b1", dev)
    return np.concatenate([SM.euler(net, torch.from_numpy(noise[i:i + bs]).to(dev), torch.from_numpy(X[i:i + bs]).to(dev),
                                    torch.from_numpy(R[i:i + bs]).to(dev), nfe).cpu().numpy() for i in range(0, len(X), bs)]).astype(np.float64)


def k16_subset(n):
    return np.sort(B.salted_rank("af0-k16-v1", range(n))[:N_K16])


# ----------------------------------------------------------------------------------------------- metrics
def point_metrics(w, Y, ref, Pid, pairs, ex):
    """Deterministic / center metrics: per-window corr on the given pairs, MAE, patient-macro FP / recall / precision / F1, RR / HR."""
    evaluable = np.array([len(r) > 0 for r in ref])
    det = list(ex.map(B._peaks, list(w), chunksize=256))
    prf = PMX.rpeak_prf_at(w, Y, FS, 50.0, peaks=(ref, det))
    bl = PMX.beat_level_metrics(w, Y, FS, 50.0, peaks=(ref, det))
    corr = np.array([np.nanmean(BR.pair_correlations(Y[i], w[i], pairs[i])) if len(pairs[i]) else np.nan for i in range(len(Y))])
    pm = RM.patient_macro_rows(prf["n_tp"], prf["n_fp"], prf["n_fn"], Pid)
    return {"corr": corr, "mae": np.abs(w - Y).mean(axis=1), "pm": pm, "rr": bl["rr_mae_ms"], "hr": bl["hr_abs_err"], "det": det,
            "f1_window": np.where(evaluable, prf["rpeak_f1"], np.nan)}


def summarize_point(m, Pid):
    s = m["pm"]["patients"]
    return {"corr": ci(m["corr"], Pid), "mae": ci(m["mae"], Pid), "rr_mae_ms": ci(m["rr"], Pid), "hr_mae_bpm": ci(m["hr"], Pid),
            **{f"pm_{k}": C0.cluster_ci(m["pm"][k], s, BOOT_N, BOOT_SEED) for k in ("fp_rate", "recall", "precision", "f1")},
            "pooled": RM.pooled_prf(**m["pm"]["pooled"])}


def band_stats(r):
    Z = A.to_bands(torch.from_numpy(np.asarray(r, np.float64))).numpy()
    return {b: {"mean": float(Z[:, s:e].mean()), "var": float(Z[:, s:e].var()), "rms": float(np.sqrt((Z[:, s:e] ** 2).mean()))} for b, s, e in A.BANDS}


def spectral_discrepancy(a, b):
    """Mean |log PSD ratio| between generated and real residual sets (Hann, rFFT, bins 1..256)."""
    pa = (np.abs(np.fft.rfft(a * np.hanning(T), axis=1)) ** 2).mean(axis=0)[1:]
    pb = (np.abs(np.fft.rfft(b * np.hanning(T), axis=1)) ** 2).mean(axis=0)[1:]
    return float(np.mean(np.abs(np.log((pa + 1e-12) / (pb + 1e-12)))))


def dev_context(ex, dev):
    X, Y, Pid, wid, ref = load_role("af_dev", ex)
    mu, ev = load_prep("af_dev")
    R = AB.event_raster(ev)
    pairs = [BR.matched_pairs(ref[i], ev[i], T) for i in range(len(Y))]
    return {"X": X, "Y": Y, "Pid": Pid, "wid": wid, "ref": ref, "mu": mu.astype(np.float64), "ev": ev, "R": R, "pairs": pairs,
            "noise": SM.window_noise(Pid, wid), "S": k16_subset(len(Y)), "perm": np.random.default_rng(SPLIT_SEED).permutation(len(Y))}


def generative_block(ctx, x_single, Ks, ex, name):
    """Population single-sample metrics and K16 conditional-set metrics of one generative arm. Ks: [K, |S|, T] samples."""
    Y, mu, S = ctx["Y"], ctx["mu"], ctx["S"]
    r_gen, r_real = x_single - mu, Y - mu
    out = {"fd": float(PMX.kanflow_fd(x_single, Y)), "residual_fd": float(PMX.kanflow_fd(r_gen, r_real)),
           "diversity_ratio": float(r_gen.std() / r_real.std()), "residual_spectral_discrepancy": spectral_discrepancy(r_gen, r_real),
           "residual_band_stats_gen": band_stats(r_gen), "residual_band_stats_real": band_stats(r_real)}
    xbar, xmed = Ks.mean(axis=0), np.median(Ks, axis=0)
    rk = Ks - mu[S][None]
    out["k16_within_condition_diversity"] = float(np.mean([BR.mean_pairwise_rms(rk[:, i, :]) for i in range(len(S))]))
    out["k16_within_diversity_over_real_residual_rms"] = out["k16_within_condition_diversity"] / float(np.sqrt((r_real[S] ** 2).mean()))
    bd = {}
    for b, s, e in A.BANDS:
        Z = A.to_bands(torch.from_numpy(rk.reshape(-1, T))).numpy().reshape(rk.shape[0], rk.shape[1], T)[:, :, s:e]
        bd[b] = float(np.mean(Z.std(axis=0)))
    out["k16_scale_diversity_sd"] = bd
    out["k16_beat_aligned_diversity"] = B.seed_diversity(Ks, [ctx["ev"][i] for i in S])[0]
    out["k16_center_minus_anchor_mae"] = float(np.abs(xbar - mu[S]).mean())
    out["k16_center_minus_anchor_band_mae"] = {b: float(np.abs(A.to_bands(torch.from_numpy(xbar - mu[S])).numpy()[:, s:e]).mean()) for b, s, e in A.BANDS}
    peaks = [list(ex.map(B._peaks, list(Ks[k]), chunksize=128)) for k in range(Ks.shape[0])]
    out["k16_r_time_seed_sd_ms"] = B.timing_sd([ctx["ref"][i] for i in S], [[peaks[k][j] for k in range(Ks.shape[0])] for j in range(len(S))])[0]
    out["k16_events_per_window_mean"] = float(np.mean([len(p) for pk in peaks for p in pk]))
    return out, xbar, xmed


def stage_eval_baselines(ex, dev):
    if not (ART / "protocol_manifest.json").exists():
        raise SystemExit("STOP: AF-DEV metrics only after the committed development protocol")
    ctx = dev_context(ex, dev)
    Y, Pid, S = ctx["Y"], ctx["Pid"], ctx["S"]
    res = {"windows": len(Y), "patients": int(np.unique(Pid).size), "k16_windows": int(len(S))}
    pm_anchor = point_metrics(ctx["mu"], Y, ctx["ref"], Pid, ctx["pairs"], ex)
    res["anchor_point"] = summarize_point(pm_anchor, Pid)
    res["anchor_fd"] = float(PMX.kanflow_fd(ctx["mu"], Y))
    placed = PMX.rpeak_prf_at(Y, Y, FS, 50.0, peaks=(ctx["ref"], ctx["ev"]))
    pmp = RM.patient_macro_rows(placed["n_tp"], placed["n_fp"], placed["n_fn"], Pid)
    res["detector_events"] = {f"pm_{k}": C0.cluster_ci(pmp[k], pmp["patients"], BOOT_N, BOOT_SEED) for k in ("fp_rate", "recall", "precision", "f1")}
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {"anchor": ctx["mu"].astype(np.float32)}
    for name in ("b1", "b2", "b3"):
        if name == "b1":
            xs = gen_full_sf(ctx["X"], ctx["R"], ctx["noise"], dev)
            Ks = np.stack([gen_full_sf(ctx["X"][S], ctx["R"][S], SM.window_noise(Pid[S], ctx["wid"][S], k), dev) for k in range(K)])
        else:
            xs = ctx["mu"] + gen_residual(name, ctx["X"], ctx["R"], ctx["mu"].astype(np.float32), ctx["noise"], dev)
            Ks = np.stack([ctx["mu"][S] + gen_residual(name, ctx["X"][S], ctx["R"][S], ctx["mu"][S].astype(np.float32),
                                                       SM.window_noise(Pid[S], ctx["wid"][S], k), dev) for k in range(K)])
        g, xbar, _ = generative_block(ctx, xs, Ks, ex, name)
        cm = point_metrics(xbar, Y[S], [ctx["ref"][i] for i in S], Pid[S], [ctx["pairs"][i] for i in S], ex)
        g["mean16"] = summarize_point(cm, Pid[S])
        res[name] = g
        arrays[name] = xs.astype(np.float32)
        print(f"[af0] baseline {name}: FD {g['fd']:.3f} residual FD {g['residual_fd']:.3f} div {g['diversity_ratio']:.3f}", flush=True)
    np.savez(OUT / "dev_baseline_outputs.npz", **arrays)
    am = point_metrics(ctx["mu"][S], Y[S], [ctx["ref"][i] for i in S], Pid[S], [ctx["pairs"][i] for i in S], ex)
    np.savez(OUT / "dev_anchor_k16subset_metrics.npz", corr=am["corr"], fp=am["pm"]["fp_rate"], recall=am["pm"]["recall"], patients=am["pm"]["patients"])
    res["anchor_on_k16_subset"] = summarize_point(am, Pid[S])
    write_json("dev_baselines.json", res)


def gates(d) -> dict:
    g = {"D1": d["fd_vs_anchor"][2] < 0 and d["fd_vs_fullsf"][2] < MARG["fd_vs_sf"], "D2": d["corr16_vs_anchor"][1] > -MARG["corr"],
         "D3": d["fp16_vs_anchor"][2] < MARG["fp"] and d["recall16_vs_anchor"][1] > -MARG["recall"], "D4": d["rfd_vs_b2"][2] < 0,
         "D5": d["rfd_vs_b3"][2] < 0, "D6": d["rfd_s1_minus_cond"][1] > 0 and d["rfd_anchorshuf_minus_cond"][1] > 0,
         "D7": MARG["div"][0] <= d["diversity_ratio"] <= MARG["div"][1]}
    g = {k: bool(v) for k, v in g.items()}
    g["overall"] = all(g.values())
    return g


def stage_eval_cand(ex, dev, cid, nfe=None, role="af_dev"):
    if role == "af_dev" and not (ART / "dev_baselines.json").exists():
        raise SystemExit("STOP: baselines first")
    cfg = read_json(f"candidate_configs/{cid}.json")
    nfe = int(nfe or cfg["nfe"])
    ctx = dev_context(ex, dev) if role == "af_dev" else lock_context(ex, dev)
    X, Y, Pid, R, mu, S, perm = ctx["X"], ctx["Y"], ctx["Pid"], ctx["R"], ctx["mu"], ctx["S"], ctx["perm"]
    mu32 = mu.astype(np.float32)
    base = np.load(OUT / ("dev_baseline_outputs.npz" if role == "af_dev" else "lock_baseline_outputs.npz"))
    xs = mu + gen_residual(cid, X, R, mu32, ctx["noise"], dev, nfe)
    Ks = np.stack([mu[S] + gen_residual(cid, X[S], R[S], mu32[S], SM.window_noise(Pid[S], ctx["wid"][S], k), dev, nfe) for k in range(K)])
    g, xbar, xmed = generative_block(ctx, xs, Ks, ex, cid)
    x_s1 = mu + gen_residual(cid, X[perm], R, mu32, ctx["noise"], dev, nfe)                                    # S1: PPG shuffled in the flow only
    mu_s2 = anchor_out(X[perm], R, dev).astype(np.float64)                                                     # S2: whole pipeline on shuffled PPG
    x_s2 = mu_s2 + gen_residual(cid, X[perm], R, mu_s2.astype(np.float32), ctx["noise"], dev, nfe)
    x_as = mu + gen_residual(cid, X, R, mu32[perm], ctx["noise"], dev, nfe)                                    # anchor shuffled in the flow condition
    r_real = Y - mu
    res = B.patient_resamples(np.unique(Pid).size, BOOT_N, BOOT_SEED)
    fd_ecg = FF.fd_bootstrap({"cand": xs, "anchor": base["anchor"].astype(np.float64), "b1": base["b1"].astype(np.float64)}, Y, Pid,
                             [("cand", "anchor"), ("cand", "b1")], res)
    fd_res = FF.fd_bootstrap({"cand": xs - mu, "b2": base["b2"].astype(np.float64) - mu, "b3": base["b3"].astype(np.float64) - mu,
                              "s1": x_s1 - mu, "as": x_as - mu}, r_real, Pid, [("cand", "b2"), ("cand", "b3"), ("s1", "cand"), ("as", "cand")], res)
    cm = point_metrics(xbar, Y[S], [ctx["ref"][i] for i in S], Pid[S], [ctx["pairs"][i] for i in S], ex)
    medm = point_metrics(xmed, Y[S], [ctx["ref"][i] for i in S], Pid[S], [ctx["pairs"][i] for i in S], ex)
    am = np.load(OUT / ("dev_anchor_k16subset_metrics.npz" if role == "af_dev" else "lock_anchor_k16subset_metrics.npz"))
    assert np.array_equal(am["patients"], cm["pm"]["patients"])
    subs = cm["pm"]["patients"]
    d = {"fd_vs_anchor": fd_ecg["cand-anchor"], "fd_vs_fullsf": fd_ecg["cand-b1"],
         "corr16_vs_anchor": ci(cm["corr"] - am["corr"], Pid[S]),
         "fp16_vs_anchor": C0.cluster_ci(cm["pm"]["fp_rate"] - am["fp"], subs, BOOT_N, BOOT_SEED),
         "recall16_vs_anchor": C0.cluster_ci(cm["pm"]["recall"] - am["recall"], subs, BOOT_N, BOOT_SEED),
         "rfd_vs_b2": fd_res["cand-b2"], "rfd_vs_b3": fd_res["cand-b3"], "rfd_s1_minus_cond": fd_res["s1-cand"],
         "rfd_anchorshuf_minus_cond": fd_res["as-cand"], "diversity_ratio": g["diversity_ratio"]}
    gt = gates(d)
    sp = point_metrics(xs, Y, ctx["ref"], Pid, ctx["pairs"], ex)
    s2 = point_metrics(x_s2, Y, ctx["ref"], Pid, ctx["pairs"], ex)
    out = {"candidate": cid, "role": role, "nfe": nfe, "config": cfg, "params": A.n_params(build_job(cid)[0]), "comparisons": d, "gates": gt,
           "fd": fd_ecg["fd"], "residual_fd": fd_res["fd"], "generative": g, "mean16": summarize_point(cm, Pid[S]), "median16": summarize_point(medm, Pid[S]),
           "single_sample": summarize_point(sp, Pid),
           "condition_controls": {"S1_ppg_flow_only": {"residual_fd": fd_res["fd"]["s1"], "fd": float(PMX.kanflow_fd(x_s1, Y))},
                                  "S2_ppg_whole_pipeline": {"fd": float(PMX.kanflow_fd(x_s2, Y)), "point": summarize_point(s2, Pid)},
                                  "anchor_shuffle": {"residual_fd": fd_res["fd"]["as"], "fd": float(PMX.kanflow_fd(x_as, Y))}}}
    tag = f"{cid}" + ("" if nfe == cfg["nfe"] else f"_nfe{nfe}")
    write_json(f"candidate_metrics/{tag}.json" if role == "af_dev" else f"lock/{tag}.json", out)
    if role == "af_dev":
        append_history(cid, cfg, out, tag)
    print(f"[af0] {role} {tag}: gates {json.dumps(gt)} FD {fd_ecg['fd']['cand']:.3f} rFD {fd_res['fd']['cand']:.3f} "
          f"corr16 {d['corr16_vs_anchor'][0]:+.4f} fp16 {d['fp16_vs_anchor'][0]:+.4f} div {g['diversity_ratio']:.3f}", flush=True)
    return out


def append_history(cid, cfg, out, tag):
    p = ART / "search_history.csv"
    new = not p.exists()
    d, g = out["comparisons"], out["gates"]
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HISTORY_COLS)
        w.writerow([tag, cfg.get("parent", ""), cfg.get("change", ""), out["params"], out["nfe"], f"{out['fd']['cand']:.4f}",
                    f"{out['residual_fd']['cand']:.4f}", f"{out['mean16']['corr'][0]:.4f}", f"{out['mean16']['pm_fp_rate'][0]:.4f}",
                    f"{out['mean16']['pm_recall'][0]:.4f}", f"{d['diversity_ratio']:.4f}", f"{d['rfd_s1_minus_cond'][0]:.4f}",
                    f"{d['rfd_anchorshuf_minus_cond'][0]:.4f}"] + ["PASS" if g[k] else "FAIL" for k in ("D1", "D2", "D3", "D4", "D5", "D6", "D7")] +
                   ["PASS" if g["overall"] else "FAIL"])


def stage_nfe(ex, dev, cid):
    """Stage E: only for a candidate whose NFE-8 evaluation passed D1-D7; lowest passing NFE is kept."""
    m = read_json(f"candidate_metrics/{cid}.json")
    if not m["gates"]["overall"]:
        raise SystemExit("STOP: NFE tuning cannot rescue a failing architecture")
    for nfe in (4, 16):
        stage_eval_cand(ex, dev, cid, nfe)


# ----------------------------------------------------------------------------------------------- lock (sealed)
def lock_context(ex, dev):
    X, Y, Pid, wid, ref = load_role("af_lock", ex)
    ev = detector_events(X, dev, ex)
    R = AB.event_raster(ev)
    mu = anchor_out(X, R, dev).astype(np.float64)
    pairs = [BR.matched_pairs(ref[i], ev[i], T) for i in range(len(Y))]
    return {"X": X, "Y": Y, "Pid": Pid, "wid": wid, "ref": ref, "mu": mu, "ev": ev, "R": R, "pairs": pairs, "noise": SM.window_noise(Pid, wid),
            "S": k16_subset(len(Y)), "perm": np.random.default_rng(SPLIT_SEED).permutation(len(Y))}


def stage_freeze(ex, dev, cid):
    m = read_json(f"candidate_metrics/{cid}.json")
    if not m["gates"]["overall"]:
        raise SystemExit("STOP: only a development winner can be frozen")
    files = [PROTOCOL, *CODE_FILES, "outputs/af0_anchorflow/detector.pt", "outputs/af0_anchorflow/anchor.pt", f"outputs/af0_anchorflow/{cid}.pt",
             "outputs/af0_anchorflow/b1.pt", "outputs/af0_anchorflow/b2.pt", "outputs/af0_anchorflow/b3.pt", "artifacts/af0_anchorflow/residual_stats.json",
             f"artifacts/af0_anchorflow/candidate_configs/{cid}.json", "artifacts/af0_anchorflow/split_manifest.json",
             "artifacts/af0_anchorflow/development_winner.json", "src/ppg2ecg/scaleflow/model.py", "src/ppg2ecg/evaluation/paper_metrics.py",
             "src/ppg2ecg/evaluation/rpeaks.py", "scripts/c0_coherentbeat.py", "scripts/bf0_run.py"]
    write_json("lock_freeze_manifest.json", {"winner": cid, "nfe": m["nfe"], "sha256": {f: B.sha256_file(ROOT / f) for f in files}})


def stage_eval_lock(ex, dev):
    check_lock_freeze()
    fm = json.loads(LOCK_FREEZE.read_text())
    cid = fm["winner"]
    ctx = lock_context(ex, dev)
    S, Pid, Y = ctx["S"], ctx["Pid"], ctx["Y"]
    arrays = {"anchor": ctx["mu"].astype(np.float32)}
    res = {"anchor_point": summarize_point(point_metrics(ctx["mu"], Y, ctx["ref"], Pid, ctx["pairs"], ex), Pid), "anchor_fd": float(PMX.kanflow_fd(ctx["mu"], Y))}
    for name in ("b1", "b2", "b3"):
        xs = gen_full_sf(ctx["X"], ctx["R"], ctx["noise"], dev) if name == "b1" else ctx["mu"] + gen_residual(name, ctx["X"], ctx["R"], ctx["mu"].astype(np.float32), ctx["noise"], dev)
        arrays[name] = xs.astype(np.float32)
        res[name] = {"fd": float(PMX.kanflow_fd(xs, Y)), "residual_fd": float(PMX.kanflow_fd(xs - ctx["mu"], Y - ctx["mu"])),
                     "diversity_ratio": float((xs - ctx["mu"]).std() / (Y - ctx["mu"]).std())}
    np.savez(OUT / "lock_baseline_outputs.npz", **arrays)
    am = point_metrics(ctx["mu"][S], Y[S], [ctx["ref"][i] for i in S], Pid[S], [ctx["pairs"][i] for i in S], ex)
    np.savez(OUT / "lock_anchor_k16subset_metrics.npz", corr=am["corr"], fp=am["pm"]["fp_rate"], recall=am["pm"]["recall"], patients=am["pm"]["patients"])
    write_json("lock_metrics.json", res)
    out = stage_eval_cand(ex, dev, cid, fm["nfe"], role="af_lock")
    write_json("lock_bootstrap.json", {"unit": "patient", "replicates": BOOT_N, "seed": BOOT_SEED, "comparisons": out["comparisons"]})
    write_json("lock_gates.json", out["gates"] | {"verdict": "CONFIRMED" if out["gates"]["overall"] else "FAILED"})


def stage_winner(ex, dev, cid):
    """Record the development winner (prereg §7 lexicographic rule applied by the operator over passing candidates)."""
    m = read_json(f"candidate_metrics/{cid}.json")
    if not m["gates"]["overall"]:
        raise SystemExit("STOP: not a passing candidate")
    write_json("development_winner.json", {"winner": cid, "nfe": m["nfe"], "config": m["config"], "params": m["params"], "gates": m["gates"],
                                           "fd": m["fd"]["cand"], "residual_fd": m["residual_fd"]["cand"]})


def stage_summarize(ex, dev):
    """Aggregate the development record: winner (or NONE), controls, residual distribution, diversity, checkpoints, compute,
    figure_dev.png and table_dev.csv. Reads stored AF-DEV results only (plus one latency / FLOP measurement)."""
    from torch.utils.flop_counter import FlopCounterMode
    base = read_json("dev_baselines.json")
    cands = {p.stem: json.loads(p.read_text()) for p in sorted((ART / "candidate_metrics").glob("c*.json"))}
    passing = [c for c, m in cands.items() if m["gates"]["overall"]]
    if passing:
        win = sorted(passing, key=lambda c: (cands[c]["residual_fd"]["cand"], cands[c]["fd"]["cand"], -cands[c]["mean16"]["corr"][0], cands[c]["nfe"], cands[c]["params"]))[0]
        write_json("development_winner.json", {"winner": win})
    else:
        write_json("development_winner.json", {"winner": None, "verdict": "AF0 DEVELOPMENT FAILED", "candidates_evaluated": sorted(cands),
                                               "training_jobs": len(ledger()["jobs"])})
    write_json("condition_shuffle.json", {c: {"S1_ppg_flow_only_residual_fd_minus_cond": m["comparisons"]["rfd_s1_minus_cond"],
                                              "S1_fd": m["condition_controls"]["S1_ppg_flow_only"]["fd"], "S2_whole_pipeline_fd": m["condition_controls"]["S2_ppg_whole_pipeline"]["fd"],
                                              "S2_point_corr": m["condition_controls"]["S2_ppg_whole_pipeline"]["point"]["corr"]} for c, m in cands.items()})
    write_json("anchor_shuffle.json", {c: {"residual_fd_minus_cond": m["comparisons"]["rfd_anchorshuf_minus_cond"],
                                           "fd": m["condition_controls"]["anchor_shuffle"]["fd"]} for c, m in cands.items()})
    keys = ("fd", "residual_fd", "diversity_ratio", "residual_spectral_discrepancy", "residual_band_stats_gen", "residual_band_stats_real")
    write_json("residual_distribution.json", {**{b: {k: base[b][k] for k in keys} for b in ("b1", "b2", "b3")},
                                              **{c: {k: m["generative"][k] for k in keys} for c, m in cands.items()}})
    dk = ("diversity_ratio", "k16_within_condition_diversity", "k16_within_diversity_over_real_residual_rms", "k16_scale_diversity_sd",
          "k16_beat_aligned_diversity", "k16_r_time_seed_sd_ms", "k16_events_per_window_mean", "k16_center_minus_anchor_mae", "k16_center_minus_anchor_band_mae")
    write_json("diversity.json", {**{b: {k: base[b][k] for k in dk} for b in ("b1", "b2", "b3")}, **{c: {k: m["generative"][k] for k in dk} for c, m in cands.items()}})
    cks = {p.stem: json.loads(p.read_text()) for p in sorted((ART / "checkpoints").glob("*.json"))}
    write_json("candidate_checkpoints.json", {k: {"sha256": v["sha256"], "params": v["n_params"], "train_seconds": v["train_seconds"],
                                                   "peak_gpu_mem_mib": v["peak_gpu_mem_mib"], "nan_steps": v["nan_steps"]} for k, v in cks.items()})
    best = min(cands, key=lambda c: (cands[c]["residual_fd"]["cand"], cands[c]["fd"]["cand"]))
    X = load_role("af_dev", ex)[0][:64]
    mu, ev = load_prep("af_dev")
    R = AB.event_raster(ev[:64]).astype(np.float32)
    lat, flops = {}, {}
    torch.set_num_threads(4)
    for dname in ("cpu", "cuda"):
        d = torch.device(dname)
        anc, det = load_ckpt("anchor", d), load_ckpt("detector", d)
        arms = {"anchor": None, "b1": load_ckpt("b1", d)} | {k: load_ckpt(k, d, build_job(k)[0]) for k in ("b2", "b3", best)}
        for k, net in arms.items():
            ts = []
            for i in range(23):
                xi, ri = torch.from_numpy(X[i:i + 1]).to(d), torch.from_numpy(R[i:i + 1]).to(d)
                if dname == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    p = torch.sigmoid(det(xi[:, None])[:, 0])
                    m = anc(xi, ri)
                    if k == "b1":
                        SM.euler(net, torch.zeros(1, T, device=d), xi, ri, 8)
                    elif k != "anchor":
                        A.euler(net, torch.zeros(1, T, device=d), xi, ri, m, 8)
                if dname == "cuda":
                    torch.cuda.synchronize()
                if i >= 3:
                    ts.append((time.perf_counter() - t0) * 1000)
            lat[f"{dname}_{k}"] = float(np.median(ts))
            if dname == "cpu" and net is not None:
                xi, ri = torch.from_numpy(X[:1]), torch.from_numpy(R[:1])
                with torch.no_grad():
                    m1 = anc(xi, ri)                                                   # anchor pass counted separately
                fc = FlopCounterMode(display=False)
                with fc, torch.no_grad():
                    net(torch.zeros(1, T), torch.zeros(1), xi, ri) if k == "b1" else net(torch.zeros(1, T), torch.zeros(1), xi, ri, m1)
                flops[k] = {"per_vector_field_eval": int(fc.get_total_flops()), "nfe": 8}
        if dname == "cpu":
            for name, net in (("anchor", anc), ("detector", det)):
                fc = FlopCounterMode(display=False)
                with fc, torch.no_grad():
                    net(torch.from_numpy(X[:1]), torch.from_numpy(R[:1])) if name == "anchor" else net(torch.from_numpy(X[:1])[:, None])
                flops[name] = {"per_pass": int(fc.get_total_flops())}
    write_json("compute_accounting.json", {"checkpoints": {k: {"params": v["n_params"], "train_seconds": v["train_seconds"], "peak_gpu_mem_mib": v["peak_gpu_mem_mib"]}
                                                           for k, v in cks.items()},
                                           "batch1_latency_ms_full_pipeline": lat, "flops": flops, "best_candidate_measured": best,
                                           "notes": "pipeline = detector + anchor (+ NFE-8 residual flow); FLOPs from torch.utils.flop_counter (one window, CPU)",
                                           "training_jobs": ledger()["jobs"], "software": B.software()})
    _figure_dev(base, cands)


def _figure_dev(base, cands):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted(cands)
    fig, ax = plt.subplots(2, 4, figsize=(25, 10.5))
    a = ax[0, 0]
    a.axis("off")
    a.set_title("A  AnchorFlow pipeline", loc="left")
    a.text(0.0, 0.97, "PPG + event raster\n   |\n frozen WW anchor -> mu(c)  (point output)\n   |\n residual r = x - mu\n   |\n fixed Haar"
           " -> coarse -> mid -> fine\n   (scale-coupled residual FM,\n    conditioned on PPG_s, event_s, mu_s)\n   |\n residual sample r_k\n   |\n"
           " x_k = mu + r_k  (conditional samples)", va="top", family="monospace", fontsize=10)
    a = ax[0, 1]
    ap = base["anchor_point"]
    vals = [ap["corr"][0], ap["pm_recall"][0], ap["pm_precision"][0], ap["pm_f1"][0], ap["pm_fp_rate"][0]]
    a.bar(range(5), vals, color="#0e7c86")
    a.set_xticks(range(5), ["corr", "recall", "precision", "F1", "FP / win"])
    a.set_title("B  anchor point estimate (AF-DEV)", loc="left")
    a = ax[0, 2]
    arms = ["anchor", "B1 full SF", "B2 vanilla res", "B3 indep res"] + names
    fds = [base["anchor_fd"], base["b1"]["fd"], base["b2"]["fd"], base["b3"]["fd"]] + [cands[c]["fd"]["cand"] for c in names]
    a.bar(range(len(arms)), fds, color=["0.4", "#a23b52", "#d59a54", "#7a6fd0"] + ["#4a3aa7"] * len(names))
    a.axhline(base["b1"]["fd"] + 1.0, color="k", ls="--", lw=0.8)
    a.set_xticks(range(len(arms)), arms, rotation=60, fontsize=7)
    a.set_title("C  population FD (dashed: full SF + 1.0)", loc="left")
    a = ax[0, 3]
    rf = [base["b2"]["residual_fd"], base["b3"]["residual_fd"]] + [cands[c]["residual_fd"]["cand"] for c in names]
    a.bar(range(len(rf)), rf, color=["#d59a54", "#7a6fd0"] + ["#4a3aa7"] * len(names))
    a.set_xticks(range(len(rf)), ["B2", "B3"] + names, rotation=60, fontsize=8)
    a.set_title("D  residual FD (D4 vs B2, D5 vs B3)", loc="left")
    def ciplot(a, key, title, margin=None, ylab=""):
        for i, c in enumerate(names):
            v = cands[c]["comparisons"][key]
            a.plot([i, i], [v[1], v[2]], color="k", lw=3)
            a.plot([i], [v[0]], "o", color="#4a3aa7")
        if margin is not None:
            a.axhline(margin, color="k", ls="--", lw=0.8)
        a.axhline(0, color="0.7", lw=0.6)
        a.set_xticks(range(len(names)), names, rotation=60, fontsize=8)
        a.set_title(title, loc="left")
        a.set_ylabel(ylab)
    ciplot(ax[1, 0], "corr16_vs_anchor", "E  D2: corr(mean16) - corr(anchor) (margin -0.02)", -0.02)
    ciplot(ax[1, 1], "fp16_vs_anchor", "E'  D3: FP(mean16) - FP(anchor) (margin +0.05)", 0.05)
    a = ax[1, 2]
    for i, c in enumerate(names):
        for j, (k, col) in enumerate((("rfd_s1_minus_cond", "#2a7f3f"), ("rfd_anchorshuf_minus_cond", "#a23b52"))):
            v = cands[c]["comparisons"][k]
            a.plot([i + (j - 0.5) * 0.3] * 2, [v[1], v[2]], color=col, lw=3, label=("PPG-only shuffle S1" if j == 0 else "anchor shuffle") if i == 0 else None)
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(range(len(names)), names, rotation=60, fontsize=8)
    a.set_title("F  D6: residual FD shuffled - conditioned (must be > 0)", loc="left")
    a.legend(fontsize=8)
    a = ax[1, 3]
    dv = [base["b1"]["diversity_ratio"], base["b2"]["diversity_ratio"], base["b3"]["diversity_ratio"]] + [cands[c]["generative"]["diversity_ratio"] for c in names]
    a.bar(range(len(dv)), dv, color=["#a23b52", "#d59a54", "#7a6fd0"] + ["#4a3aa7"] * len(names))
    a.axhline(0.5, color="k", ls="--", lw=0.8); a.axhline(1.5, color="k", ls="--", lw=0.8)
    a.set_xticks(range(len(dv)), ["B1", "B2", "B3"] + names, rotation=60, fontsize=8)
    a.set_title("G  D7: residual diversity ratio (0.5-1.5)", loc="left")
    win = read_json("development_winner.json")["winner"]
    fig.suptitle(f"AF0 AnchorFlow — AF-DEV adaptive development (AF-LOCK not opened). Development winner: {win or 'NONE (AF0 DEVELOPMENT FAILED)'}", fontsize=12)
    fig.tight_layout()
    fig.savefig(ART / "figure_dev.png", dpi=105)
    with open(ART / "table_dev.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "params", "nfe", "fd", "residual_fd", "mean16_corr", "mean16_fp", "mean16_recall", "diversity_ratio"])
        w.writerow(["anchor (point)", 593577, 1, f"{base['anchor_fd']:.3f}", "", f"{base['anchor_point']['corr'][0]:.4f}", f"{base['anchor_point']['pm_fp_rate'][0]:.4f}",
                    f"{base['anchor_point']['pm_recall'][0]:.4f}", ""])
        for b, lab in (("b1", "B1 full-signal ScaleFlow"), ("b2", "B2 vanilla residual FM"), ("b3", "B3 independent residual FM")):
            g = base[b]
            w.writerow([lab, "", 8, f"{g['fd']:.3f}", f"{g['residual_fd']:.3f}", f"{g['mean16']['corr'][0]:.4f}", f"{g['mean16']['pm_fp_rate'][0]:.4f}",
                        f"{g['mean16']['pm_recall'][0]:.4f}", f"{g['diversity_ratio']:.4f}"])
        for c in names:
            m = cands[c]
            w.writerow([c, m["params"], m["nfe"], f"{m['fd']['cand']:.3f}", f"{m['residual_fd']['cand']:.3f}", f"{m['mean16']['corr'][0]:.4f}",
                        f"{m['mean16']['pm_fp_rate'][0]:.4f}", f"{m['mean16']['pm_recall'][0]:.4f}", f"{m['generative']['diversity_ratio']:.4f}"])


def new_candidate(cid: str, parent: str, change: str, hypothesis: str, **overrides):
    """Write candidate_configs/<cid>.json (base config + overrides); refuses beyond the 10-candidate budget."""
    if cid not in candidate_ids() and len(candidate_ids()) >= MAX_CANDIDATES:
        raise SystemExit("STOP: candidate budget exhausted")
    parent_cfg = read_json(f"candidate_configs/{parent}.json") if parent else dict(BASE_CFG)
    cfg = {k: parent_cfg[k] for k in BASE_CFG} | overrides | {"id": cid, "parent": parent, "change": change, "hypothesis": hypothesis}
    write_json(f"candidate_configs/{cid}.json", cfg)
    return cfg


STAGES = {"split": stage_split, "audit": stage_audit, "manifest": stage_manifest, "train_detector": stage_train_detector,
          "train_anchor": stage_train_anchor, "prep": stage_prep, "eval_baselines": stage_eval_baselines, "eval_lock": stage_eval_lock, "summarize": stage_summarize}
ARG_STAGES = {"train": stage_train, "eval_cand": stage_eval_cand, "nfe": stage_nfe, "freeze": stage_freeze, "winner": stage_winner}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + list(ARG_STAGES))
    ap.add_argument("arg", nargs="?")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(10) as ex:
        if args.stage in ARG_STAGES:
            ARG_STAGES[args.stage](ex, dev, args.arg)
        else:
            STAGES[args.stage](ex, dev)


if __name__ == "__main__":
    main()
