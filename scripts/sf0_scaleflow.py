"""SF0 — ScaleFlow-ECG feasibility (docs/SF0_SCALEFLOW_PREREGISTRATION.md).

New split of the former ARCH-TRAIN (3,470 patients, seed 20261002): SF-TRAIN 3,037 / SF-VAL 433. Timing detector
retrained on SF-TRAIN. Four waveform models trained once on SF-TRAIN (seed 42, 20,000 x 64, last checkpoint):
WW-L1, WW-FM, SCALE-FM-INDEPENDENT, SCALEFLOW-COUPLED. ARCH-VAL / ARCH-HOLDOUT are never used. The old V1 TEST opens only
after SF-VAL qualification, a clean freshness audit and a committed final freeze.

Stages: split, audit, manifest, train_detector, train_wwl1, train_wwfm, train_ind, train_sf, evaluate_val, nfe,
        stochastic, compute, figure_val, freeze, evaluate_test
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/sf0_scaleflow.py <stage>
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
from ppg2ecg.beatfirst import render as BR  # noqa: E402
from ppg2ecg.coherentbeat import ablation as AB  # noqa: E402
from ppg2ecg.coherentbeat import split as SP  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as RP  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, extract_events  # noqa: E402
from ppg2ecg.rhythmfield import model as RM  # noqa: E402
from ppg2ecg.scaleflow import model as M  # noqa: E402

PREREG = "docs/SF0_SCALEFLOW_PREREGISTRATION.md"
CODE_FILES = ("scripts/sf0_scaleflow.py", "src/ppg2ecg/scaleflow/__init__.py", "src/ppg2ecg/scaleflow/model.py", "tests/test_sf0_scaleflow.py")
ART = ROOT / "artifacts/sf0_scaleflow"
OUT = ROOT / "outputs/sf0_scaleflow"
FREEZE = ART / "final_test_freeze_manifest.json"
FS, T = 128, 512
SPLIT_SEED, BOOT_SEED, BOOT_N, SEED = 20261002, 20261002, 2000, 42
N_SF_VAL = 433
PROTO = {"steps": 20000, "batch": 64, "lr": 1e-3, "wd": 0.01, "clip": 1.0}
DET = dict(C0.DET)                                   # RD1 / C0 detector protocol: 14,000 x 64, AdamW 1e-3 / 0.01, BCE, sigma 20 ms
ARMS = ("WWL1", "WWFM", "IND", "SF")
FM_ARMS = ("WWFM", "IND", "SF")
NFE = 8
N_STOCH, K_STOCH = 2000, 16
LABEL = {"WWL1": "WW-L1", "WWFM": "WW-FM", "IND": "SCALE-FM-INDEPENDENT", "SF": "SCALEFLOW-COUPLED"}


def write_json(name, obj):
    ART.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(json.dumps(B.clean(obj), indent=1))


def ci(v, pid):
    return C0.cluster_ci(v, pid, BOOT_N, BOOT_SEED)


def fd_diff_ci(a, b, Y, pid):
    """C0's in-replicate FD bootstrap with the SF0 seed (patients resampled, FD recomputed on matched windows)."""
    jobs = B.patient_bootstrap_indices(pid, BOOT_N, BOOT_SEED)
    assert min(len(j) for j in jobs) >= PMX.FD_SMALL_SET
    with ProcessPoolExecutor(16, initializer=C0._fd_init, initargs=(a, b, Y)) as pool:
        draws = np.array(list(pool.map(C0._fd_draw, jobs, chunksize=8)))
    return [float(PMX.kanflow_fd(a, Y) - PMX.kanflow_fd(b, Y)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


# ----------------------------------------------------------------------------------------------- split / data
def sf_split(arch_train_patients) -> dict:
    """Deterministic: permutation of the sorted ARCH-TRAIN patients with default_rng(20261002); first 433 -> SF-VAL."""
    p = np.array(sorted(int(x) for x in arch_train_patients))
    perm = np.random.default_rng(SPLIT_SEED).permutation(p.size)
    val = sorted(int(x) for x in p[perm[:N_SF_VAL]])
    train = sorted(int(x) for x in p[perm[N_SF_VAL:]])
    return {"sf_train": train, "sf_val": val}


def ids_sha(ids) -> str:
    return hashlib.sha256(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()


def load_role(role, ex):
    """(X, Y, Pid, wid, ref) of SF-TRAIN / SF-VAL; wid = window index in the frozen ARCH-TRAIN concatenation."""
    if role not in ("sf_train", "sf_val"):
        raise ValueError(role)
    X, Y, Pid = C0.load_arch("train")
    ref = C0.reference_peaks("train", Y, ex)
    pats = json.loads((ART / "split_manifest.json").read_text())["roles"][role]["patients"]
    wid = np.flatnonzero(np.isin(Pid, pats))
    return X[wid], Y[wid], Pid[wid], wid, [ref[i] for i in wid]


def check_final_freeze():
    if not FREEZE.exists():
        raise PermissionError("SF0: old V1 TEST sealed (no final-test freeze manifest)")
    fm = json.loads(FREEZE.read_text())
    if fm.get("qualified") is not True or fm.get("freshness_audit") != "CLEAN":
        raise PermissionError("SF0: old V1 TEST sealed (not qualified or freshness audit not clean)")
    for f, h in fm["sha256"].items():
        if B.sha256_file(ROOT / f) != h:
            raise PermissionError(f"SF0: frozen file changed: {f}")
    rel = str(FREEZE.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise PermissionError("SF0: final-test freeze manifest not committed")


def load_test():
    check_final_freeze()
    X, Y, P, W = [], [], [], []
    poc = C0.manifest()["extra"]["patient_of_case"]
    off = 0
    for c in C0.manifest()["splits"][0]["test"]:
        d = np.load(C0.DATA / f"{c}.npz")
        assert int(d["subjectid"]) == int(poc[c]), c
        n = len(d["x"])
        X.append(d["x"].astype(np.float32)); Y.append(d["y"].astype(np.float64)); P.append(np.full(n, int(d["subjectid"])))
        W.append(np.arange(off, off + n)); off += n
    return np.concatenate(X), np.concatenate(Y), np.concatenate(P), np.concatenate(W)


# ----------------------------------------------------------------------------------------------- stages: split / audit / manifest
def stage_split(ex, dev):
    m = C0.manifest()
    arch = C0.split_info()["roles"]
    sp = sf_split(arch["train"]["patients"])
    old = SP.old_heldout_patients(m)
    other = set(arch["val"]["patients"]) | set(arch["holdout"]["patients"])
    checks = {"disjoint": not (set(sp["sf_train"]) & set(sp["sf_val"])),
              "union_is_arch_train": set(sp["sf_train"]) | set(sp["sf_val"]) == set(int(p) for p in arch["train"]["patients"]),
              "no_old_val_or_test": not ((set(sp["sf_train"]) | set(sp["sf_val"])) & (set(old["val"]) | set(old["test"]))),
              "no_arch_val_or_holdout": not ((set(sp["sf_train"]) | set(sp["sf_val"])) & other),
              "counts": [len(sp["sf_train"]), len(sp["sf_val"])] == [3037, 433]}
    if not all(checks.values()):
        raise SystemExit(f"STOP: split checks {checks}")
    X, Y, Pid = C0.load_arch("train")
    roles = {}
    for r in ("sf_train", "sf_val"):
        w = np.flatnonzero(np.isin(Pid, sp[r]))
        roles[r] = {"patients": sp[r], "n_patients": len(sp[r]), "n_windows": int(w.size)}
    write_json("split_manifest.json", {"source": "former ARCH-TRAIN (C0 split_manifest.json)", "seed": SPLIT_SEED,
                                       "rule": "default_rng(20261002).permutation of sorted ARCH-TRAIN patients; first 433 -> SF-VAL",
                                       "roles": roles, "checks": checks})
    write_json("split_hashes.json", {r: {"patients_sha256": ids_sha(sp[r]),
                                         "window_index_sha256": hashlib.sha256(np.flatnonzero(np.isin(Pid, sp[r])).astype(np.int64).tobytes()).hexdigest()}
                                     for r in sp})
    print(json.dumps({r: roles[r]["n_windows"] for r in roles}), flush=True)


def stage_audit(ex, dev):
    pm = M.param_match()
    write_json("parameter_match.json", pm | {"rule": "widths by parameter count only; depths fixed (WW-FM decoder 5 blocks as WW-DET; "
                                                     "6 blocks per scale branch, dilations 1..32); SCALEFLOW closest to 600k; "
                                                     "INDEPENDENT and WW-FM closest to SCALEFLOW's count", "grid": [M.WIDTH_GRID[0], M.WIDTH_GRID[-1]]})
    write_json("model_configs.json", {
        "WWL1": {"class": "coherentbeat.ablation.WWDet(72, 5)", "inputs": "PPG, event raster", "loss": "L1", "params": pm["WWL1"]["params"]},
        "WWFM": {"class": "scaleflow.model.WWFM", "dec_width": pm["WWFM"]["dec_width"], "encoder": "C0 Encoder 64 ch, 8 blocks (PPG)",
                 "decoder": "1x1 on [h, raster, x_t] -> 5 time-conditioned residual blocks (dil 1,2,4,8,16) -> 1x1", "params": pm["WWFM"]["params"]},
        "IND": {"class": "scaleflow.model.ScaleFM(coupled=False)", "width": pm["IND"]["width"], "branches": "coarse 128 / mid 128 / fine 256; "
                "input [x_t, PPG, raster] coefficient of the scale; 6 time-conditioned residual blocks", "params": pm["IND"]["params"]},
        "SF": {"class": "scaleflow.model.ScaleFM(coupled=True)", "width": pm["SF"]["width"], "coupling": f"1x1 projections ({M.PROJ} ch): "
               "coarse -> mid input; mid and coarse -> fine input (nearest x2 upsampling); no fine -> coarse", "params": pm["SF"]["params"]},
        "time_embedding": {"sinusoidal_dim": M.T_SIN, "mlp": f"Linear({M.T_SIN},{M.T_HID}) GELU Linear({M.T_HID},{M.T_HID})", "injection":
                           "additive Linear projection after the first conv of every residual block", "scale": "1000 t"}})
    write_json("haar_config.json", {"transform": "fixed two-level orthonormal Haar (no parameters)", "level1": "low=(x[2k]+x[2k+1])/sqrt2, high=(x[2k]-x[2k+1])/sqrt2",
                                    "coarse": "low_2 (128)", "mid": "high_2 (128)", "fine": "high_1 (256)", "applied_to": ["x_t", "PPG", "event raster"],
                                    "loss_space": "velocity coefficients are inverse-transformed to the 512-sample waveform before the MSE"})
    write_json("flow_config.json", {"path": "x_t = (1 - t) x0 + t x1, x0 ~ N(0, I), t ~ U(0, 1)", "target": "u = x1 - x0", "loss": "MSE(v, u), waveform space",
                                    "inference": "Euler, NFE = 8, t = 0 -> 1 (secondary NFE 4 / 16 after the primary verdict)",
                                    "noise": "one N(0, I) 512-vector per window from sha256('patient:window:20261002'), identical for WW-FM / IND / SF",
                                    "primary": "exactly one sample per window", "normalization": "stored ECG / PPG units, as WW-L1"})
    write_json("detector_config.json", {"family": "RD1 RhythmTCN (C0 protocol)", "protocol": DET, "training": "SF-TRAIN only, seed 42",
                                        "target": "Gaussian sigma 20 ms at SF-TRAIN reference R, BCE with logits", "events": "threshold 0.35, refractory 32"})
    write_json("training_manifest.json", {"role": "SF-TRAIN only", "seed": SEED, "protocol": PROTO, "checkpoint": "last step",
                                          "training_raster": "Gaussian event raster at SF-TRAIN reference R (WW-DET convention), all four models",
                                          "evaluation_raster": "the SF detector's events (threshold 0.35, refractory 32), identical for all four models",
                                          "WWL1": "L1", "FM": "MSE flow matching, torch.Generator(cuda) seeded 42 for x0 and t"})
    print(json.dumps(pm), flush=True)


def stage_manifest(ex, dev):
    write_json("prereg_manifest.json", {"prereg": {PREREG: B.sha256_file(ROOT / PREREG)}, "code": {f: B.sha256_file(ROOT / f) for f in CODE_FILES},
                                        "frozen_design": {f: B.sha256_file(ART / f) for f in ("split_manifest.json", "split_hashes.json", "parameter_match.json",
                                                                                              "model_configs.json", "haar_config.json", "flow_config.json",
                                                                                              "detector_config.json", "training_manifest.json")},
                                        "written_before_any_sf0_training_or_sf_val_metric": True, "software": B.software()})


# ----------------------------------------------------------------------------------------------- training
def _save(name, net, secs, nan_steps, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{name}.pt"
    meta = {"seed": SEED, "train_seconds": secs, "n_params": M.n_params(net), "nan_steps": nan_steps, "checkpoint": "last step (no selection)",
            "peak_gpu_mem_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "optimizer": "AdamW"} | (extra or {})
    torch.save({"state_dict": net.state_dict()} | meta, f)
    meta["sha256"] = B.sha256_file(f)
    write_json(f"checkpoint_{name}.json", meta)


def stage_train_detector(ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, _, _, _, ref = load_role("sf_train", ex)
    Fd = torch.from_numpy(np.stack(list(ex.map(C0._field, ref, chunksize=512))).astype(np.float16)).to(dev)
    Xt = torch.from_numpy(X).to(dev)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = RhythmTCN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=DET["lr"], weight_decay=DET["wd"])
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    batches, t0, acc, nan_steps = C0._epoch_batches(len(Xt), DET["batch"], g), time.time(), [], 0
    net.train()
    for step in range(1, DET["steps"] + 1):
        b = next(batches).to(dev)
        loss = lossf(net(Xt[b][:, None])[:, 0], Fd[b].float())
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward(); opt.step(); acc.append(loss.item())
        if step % 1000 == 0:
            print(f"[sf0] detector step {step} BCE {np.mean(acc):.5f}", flush=True); acc = []
    _save("detector", net, time.time() - t0, nan_steps, {"protocol": DET, "lr": DET["lr"], "weight_decay": DET["wd"], "batch": DET["batch"],
                                                         "steps": DET["steps"]})


def build(name):
    pm = json.loads((ART / "parameter_match.json").read_text())
    if name == "wwl1":
        return M.ww_l1()
    if name == "wwfm":
        return M.WWFM(pm["WWFM"]["dec_width"])
    return M.ScaleFM(pm["IND" if name == "ind" else "SF"]["width"], coupled=(name == "sf"))


def _train(name, ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, Y, _, _, ref = load_role("sf_train", ex)
    Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y.astype(np.float32)).to(dev)
    Rt = torch.from_numpy(AB.event_raster(ref)).to(dev)                     # reference-R raster (WW-DET convention)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = build(name).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=PROTO["lr"], weight_decay=PROTO["wd"])
    g = torch.Generator().manual_seed(SEED)
    gd = torch.Generator(device=dev).manual_seed(SEED)
    batches, t0, acc, nan_steps = C0._epoch_batches(len(Xt), PROTO["batch"], g), time.time(), [], 0
    net.train()
    for step in range(1, PROTO["steps"] + 1):
        bd = next(batches).to(dev)
        if name == "wwl1":
            loss = (net(Xt[bd], Rt[bd]) - Yt[bd]).abs().mean()
        else:
            loss = M.fm_loss(net, Yt[bd], Xt[bd], Rt[bd], gd)
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), PROTO["clip"])
        opt.step(); acc.append(loss.item())
        if step % 1000 == 0:
            print(f"[sf0] {name} step {step} loss {np.mean(acc):.5f}", flush=True); acc = []
    _save(name, net, time.time() - t0, nan_steps, PROTO | {"loss": "L1" if name == "wwl1" else "flow-matching MSE"})


def stage_train_wwl1(ex, dev):
    _train("wwl1", ex, dev)


def stage_train_wwfm(ex, dev):
    _train("wwfm", ex, dev)


def stage_train_ind(ex, dev):
    _train("ind", ex, dev)


def stage_train_sf(ex, dev):
    _train("sf", ex, dev)


# ----------------------------------------------------------------------------------------------- inference
def load_net(name, dev):
    ck = torch.load(OUT / f"{name}.pt", map_location="cpu")
    net = RhythmTCN() if name == "detector" else build(name)
    net.load_state_dict(ck["state_dict"])
    return net.to(dev).eval()


@torch.no_grad()
def detector_events(X, dev, ex):
    det = load_net("detector", dev)
    out = [torch.sigmoid(det(torch.from_numpy(X[i:i + 2048]).to(dev)[:, None])[:, 0]).float().cpu().numpy() for i in range(0, len(X), 2048)]
    return [np.asarray(e, int) for e in ex.map(_extract, list(np.concatenate(out)), chunksize=256)]


def _extract(p):
    return extract_events(p, DET["threshold"], DET["refractory"])


@torch.no_grad()
def generate(name, X, R, noise, dev, nfe=NFE, bs=512):
    net = load_net(name, dev)
    out = []
    for i in range(0, len(X), bs):
        x, r = torch.from_numpy(X[i:i + bs]).to(dev), torch.from_numpy(R[i:i + bs]).to(dev)
        if name == "wwl1":
            out.append(net(x, r).cpu().numpy())
        else:
            out.append(M.euler(net, torch.from_numpy(noise[i:i + bs]).to(dev), x, r, nfe).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ----------------------------------------------------------------------------------------------- evaluation
def shuffle_perm(n) -> np.ndarray:
    return np.random.default_rng(SPLIT_SEED).permutation(n)


def evaluate(role, ex, dev):
    if role == "val":
        X, Y, Pid, wid, ref = load_role("sf_val", ex)
    else:
        X, Y, Pid, wid = load_test()
        ref = list(ex.map(B._peaks, list(Y), chunksize=256))
    N = len(Y)
    evaluable = np.array([len(r) > 0 for r in ref])
    events = detector_events(X, dev, ex)
    R = AB.event_raster(events)
    noise = M.window_noise(Pid, wid)
    perm = shuffle_perm(N)
    waves = {"WWL1": generate("wwl1", X, R, noise, dev)}
    for k, n in (("WWFM", "wwfm"), ("IND", "ind"), ("SF", "sf")):
        waves[k] = generate(n, X, R, noise, dev)
    waves["SF_PPGSHUF"] = generate("sf", X[perm], R, noise, dev)                 # PPG permuted; noise, raster, model fixed
    waves["SF_EVSHUF"] = generate("sf", X, R[perm], noise, dev)                  # rasters permuted; PPG, noise, model fixed
    all_arms = ARMS + ("SF_PPGSHUF", "SF_EVSHUF")
    det = {k: list(ex.map(B._peaks, list(waves[k]), chunksize=256)) for k in all_arms}
    per = {}
    for k in all_arms:
        per[k] = CA.arm_metrics(waves[k], Y, ref, det[k], evaluable, ex)
        per[k]["n_tp"] = PMX.rpeak_prf_at(waves[k], Y, FS, 50.0, peaks=(ref, det[k]))["n_tp"]
    pprf = PMX.rpeak_prf_at(Y, Y, FS, 50.0, peaks=(ref, events))
    pbl = PMX.beat_level_metrics(Y, Y, FS, 50.0, peaks=(ref, events))
    per["PLACED"] = {"f1": np.where(evaluable, pprf["rpeak_f1"], np.nan), "precision": pprf["rpeak_precision"], "recall": pprf["rpeak_recall"],
                     "fp": pprf["n_fp"], "fn": pprf["n_fn"], "n_tp": pprf["n_tp"], "rr_mae_ms": pbl["rr_mae_ms"], "hr_abs_err": pbl["hr_abs_err"]}
    pairs = [BR.matched_pairs(ref[i], events[i], T) for i in range(N)]
    corr = {k: [BR.pair_correlations(Y[i], waves[k][i], pairs[i]) for i in range(N)] for k in all_arms}
    wm = B.window_pair_means(corr, all_arms)
    for k in all_arms:
        per[k]["beat_corr"] = wm[k]
    pm = {k: RM.patient_macro_rows(per[k]["n_tp"], per[k]["fp"], per[k]["fn"], Pid) for k in ("PLACED",) + all_arms}
    subs = pm["SF"]["patients"]
    summary = {}
    for k in pm:
        s = {f"pm_{m}": C0.cluster_ci(pm[k][m], subs, BOOT_N, BOOT_SEED) for m in ("precision", "recall", "f1", "fp_rate")}
        s["pooled"] = RM.pooled_prf(**pm[k]["pooled"])
        for m in ("rr_mae_ms", "hr_abs_err"):
            s[m] = ci(per[k][m], Pid)
        if k != "PLACED":
            for m in ("mae", "pcc", "s4", "s5", "spectral_ratio_dev", "beat_corr"):
                s[m] = ci(per[k][m], Pid)
            s["fd"] = float(PMX.kanflow_fd(waves[k], Y))
        summary[k] = s
    pci = lambda a, b, m: C0.cluster_ci(pm[a][m] - pm[b][m], subs, BOOT_N, BOOT_SEED)  # noqa: E731
    wci = lambda a, b, m: ci(per[a][m] - per[b][m], Pid)  # noqa: E731
    best_fd = min(("WWL1", "WWFM"), key=lambda k: summary[k]["fd"])
    best_corr = max(("WWL1", "WWFM"), key=lambda k: summary[k]["beat_corr"][0])
    fdc = {}
    fdc["WWFM-WWL1"] = fd_diff_ci(waves["WWFM"], waves["WWL1"], Y, Pid)
    fdc["SF-WWFM"] = fd_diff_ci(waves["SF"], waves["WWFM"], Y, Pid)
    fdc["SF-WWL1"] = fd_diff_ci(waves["SF"], waves["WWL1"], Y, Pid) if best_fd == "WWL1" else None
    fdc["SF-IND"] = fd_diff_ci(waves["SF"], waves["IND"], Y, Pid)
    fdc["PPGSHUF-SF"] = fd_diff_ci(waves["SF_PPGSHUF"], waves["SF"], Y, Pid)
    fdc["EVSHUF-SF"] = fd_diff_ci(waves["SF_EVSHUF"], waves["SF"], Y, Pid)
    comps = {"G1": {"fd": fdc["WWFM-WWL1"], "corr": wci("WWFM", "WWL1", "beat_corr"), "fp": pci("WWFM", "WWL1", "fp_rate"),
                    "recall": pci("WWFM", "WWL1", "recall"), "mae": wci("WWFM", "WWL1", "mae")},
             "fd_sf_wwfm": fdc["SF-WWFM"], "best_fd_baseline": best_fd,
             "fd_sf_best": fdc["SF-WWFM"] if best_fd == "WWFM" else fdc["SF-WWL1"],
             "best_corr_baseline": best_corr, "corr_sf_bestcorr": wci("SF", best_corr, "beat_corr"),
             "fp_sf_wwl1": pci("SF", "WWL1", "fp_rate"), "recall_sf_wwl1": pci("SF", "WWL1", "recall"),
             "fd_sf_ind": fdc["SF-IND"], "corr_sf_ind": wci("SF", "IND", "beat_corr"),
             "fd_shuf_cond": fdc["PPGSHUF-SF"], "corr_shuf_cond": wci("SF_PPGSHUF", "SF", "beat_corr"),
             "descriptive": {"fp_sf_wwfm": pci("SF", "WWFM", "fp_rate"), "fp_sf_ind": pci("SF", "IND", "fp_rate"), "corr_sf_wwfm": wci("SF", "WWFM", "beat_corr"),
                             "f1_sf_wwl1": pci("SF", "WWL1", "f1"), "mae_sf_wwfm": wci("SF", "WWFM", "mae"), "mae_sf_wwl1": wci("SF", "WWL1", "mae")}}
    g1 = M.g1_label(comps["G1"])
    gts = M.gates(comps)
    shuffle = {"ppg_shuffle": {"fd_shuf_minus_cond": fdc["PPGSHUF-SF"], "corr_shuf_minus_cond": comps["corr_shuf_cond"],
                               "fp_shuf_minus_cond": pci("SF_PPGSHUF", "SF", "fp_rate"), "f1_shuf_minus_cond": pci("SF_PPGSHUF", "SF", "f1"),
                               "mae_shuf_minus_cond": wci("SF_PPGSHUF", "SF", "mae")},
               "event_shuffle_descriptive": {"fd_shuf_minus_cond": fdc["EVSHUF-SF"], "corr_shuf_minus_cond": wci("SF_EVSHUF", "SF", "beat_corr"),
                                             "fp_shuf_minus_cond": pci("SF_EVSHUF", "SF", "fp_rate"), "f1_shuf_minus_cond": pci("SF_EVSHUF", "SF", "f1")},
               "seed": SPLIT_SEED, "rule": "default_rng(20261002).permutation over the role's windows; noise, model and the other input fixed"}
    scale = scale_diagnostics(waves, Y, ref)
    raw = {k: {"fp": int(np.nansum(per[k]["fp"])), "fn": int(np.nansum(per[k]["fn"]))} for k in per}
    return {"role": role, "windows": N, "patients": int(subs.size), "windows_without_events": int(sum(len(e) == 0 for e in events)),
            "matched_pairs": wm["_n_pairs"], "summary": summary, "comparisons": comps, "g1": g1, "gates": gts, "shuffle": shuffle,
            "scale": scale, "raw": raw}


def scale_diagnostics(waves, Y, ref) -> dict:
    """Descriptive band errors: each waveform split into its coarse / mid / fine Haar band components (which sum to the
    waveform); RMS of (arm band - reference band) overall and within +-10 samples of reference R; PPG-shuffle and coupling
    effects per band (RMS of SF - SF_PPGSHUF and SF - IND)."""
    def bands(a):
        c, m, f = M.haar(torch.from_numpy(np.asarray(a, np.float64)))
        z = torch.zeros_like
        return {"coarse": M.ihaar(c, z(m), z(f)).numpy(), "mid": M.ihaar(z(c), m, z(f)).numpy(), "fine": M.ihaar(z(c), z(m), f).numpy()}
    q = np.zeros(Y.shape, bool)
    for i, r in enumerate(ref):
        for p in np.asarray(r, int):
            q[i, max(0, p - 10):p + 11] = True
    by = bands(Y)
    out = {}
    rms = lambda d, msk=None: float(np.sqrt(np.mean(d ** 2))) if msk is None else float(np.sqrt(np.mean(d[msk] ** 2)))  # noqa: E731
    bw = {k: bands(waves[k]) for k in ("WWL1", "WWFM", "IND", "SF", "SF_PPGSHUF")}
    for k, b in bw.items():
        out[f"error_vs_reference_{k}"] = {s: {"all": rms(b[s] - by[s]), "qrs": rms(b[s] - by[s], q), "non_qrs": rms(b[s] - by[s], ~q)} for s in b}
    out["ppg_shuffle_effect_SF"] = {s: {"all": rms(bw["SF"][s] - bw["SF_PPGSHUF"][s]), "qrs": rms(bw["SF"][s] - bw["SF_PPGSHUF"][s], q)} for s in by}
    out["coupling_effect_SF_minus_IND"] = {s: {"all": rms(bw["SF"][s] - bw["IND"][s]), "qrs": rms(bw["SF"][s] - bw["IND"][s], q)} for s in by}
    out["reference_band_rms"] = {s: {"all": rms(by[s]), "qrs": rms(by[s], q)} for s in by}
    out["cumulative_SF_error"] = {"coarse_only": rms(bw["SF"]["coarse"] - by["coarse"]),
                                  "coarse_mid": rms(bw["SF"]["coarse"] + bw["SF"]["mid"] - by["coarse"] - by["mid"]),
                                  "full": rms(waves["SF"] - Y)}
    return out


def stage_evaluate_val(ex, dev):
    if not (ART / "prereg_manifest.json").exists():
        raise SystemExit("STOP: SF-VAL outcomes only after the committed preregistration")
    r = evaluate("val", ex, dev)
    write_json("val_metrics.json", {k: r[k] for k in ("role", "windows", "patients", "windows_without_events", "matched_pairs", "summary", "raw")})
    write_json("val_bootstrap.json", {"unit": "patient", "replicates": BOOT_N, "seed": BOOT_SEED, "comparisons": r["comparisons"]})
    write_json("val_gates.json", {"G1": r["g1"], **r["gates"], "failure_categories": [] if r["gates"]["QUALIFIED"] else M.failure_categories(r["gates"], r["g1"])})
    write_json("condition_shuffle_val.json", r["shuffle"])
    write_json("scale_diagnostics.json", {"val": r["scale"]})
    print(f"[sf0] G1 {r['g1']} gates {json.dumps(r['gates'])}", flush=True)


# ----------------------------------------------------------------------------------------------- secondary (after the primary verdict)
def stage_nfe(ex, dev):
    if not (ART / "val_gates.json").exists():
        raise SystemExit("STOP: NFE analysis only after the primary NFE = 8 verdict is frozen")
    X, Y, Pid, wid, ref = load_role("sf_val", ex)
    events = detector_events(X, dev, ex)
    R = AB.event_raster(events)
    noise = M.window_noise(Pid, wid)
    evaluable = np.array([len(r) > 0 for r in ref])
    pairs = [BR.matched_pairs(ref[i], events[i], T) for i in range(len(Y))]
    out = {}
    for name in ("wwfm", "sf"):
        for nfe in (4, 8, 16):
            w = generate(name, X, R, noise, dev, nfe=nfe)
            dt = list(ex.map(B._peaks, list(w), chunksize=256))
            prf = PMX.rpeak_prf_at(w, Y, FS, 50.0, peaks=(ref, dt))
            pmr = RM.patient_macro_rows(prf["n_tp"], prf["n_fp"], prf["n_fn"], Pid)
            cr = [np.nanmean(BR.pair_correlations(Y[i], w[i], pairs[i])) if len(pairs[i]) else np.nan for i in range(len(Y))]
            net = load_net(name, dev)
            x1 = torch.from_numpy(X[:1]).to(dev); r1 = torch.from_numpy(R[:1]).to(dev); n1 = torch.from_numpy(noise[:1]).to(dev)
            ts = []
            for rep in range(23):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                M.euler(net, n1, x1, r1, nfe)
                torch.cuda.synchronize()
                if rep >= 3:
                    ts.append((time.perf_counter() - t0) * 1000)
            out[f"{name}_nfe{nfe}"] = {"fd": float(PMX.kanflow_fd(w, Y)), "beat_corr": ci(np.asarray(cr, float), Pid)[0],
                                       "fp_rate": float(np.nanmean(pmr["fp_rate"])), "recall": float(np.nanmean(pmr["recall"])),
                                       "gpu_ms_batch1_waveform_model": float(np.median(ts))}
            print(f"[sf0] nfe {name} {nfe}: {json.dumps(B.clean(out[f'{name}_nfe{nfe}']))}", flush=True)
    write_json("nfe_analysis.json", out | {"note": "secondary, after the frozen NFE = 8 verdict; FD point estimates; beat_corr on each window's own pairs"})


def stage_stochastic(ex, dev):
    if not (ART / "val_gates.json").exists():
        raise SystemExit("STOP: secondary analyses only after the primary verdict")
    X, Y, Pid, wid, ref = load_role("sf_val", ex)
    sub = np.sort(B.salted_rank("sf0-stoch-v1", range(len(Y)))[:N_STOCH])
    X, Y, Pid, wid = X[sub], Y[sub], Pid[sub], wid[sub]
    ref = [ref[i] for i in sub]
    events = detector_events(X, dev, ex)
    R = AB.event_raster(events)
    out = {"windows": int(N_STOCH), "K": K_STOCH, "subset": "salted rank 'sf0-stoch-v1' over SF-VAL windows"}
    real_div = np.array([BR.within_window_diversity(Y[i], events[i]) for i in range(len(Y))])
    hr_ref = np.array([RP.hr_bpm(np.asarray(r, int), FS) for r in ref])
    for name, key in (("wwfm", "WWFM"), ("ind", "IND"), ("sf", "SF")):
        S = np.stack([generate(name, X, R, M.window_noise(Pid, wid, k), dev) for k in range(K_STOCH)])     # [K, N, T]
        ww_div = np.array([BR.mean_pairwise_rms(S[:, i, :]) for i in range(len(Y))])
        beat_div, n_anchor = B.seed_diversity(S, events)
        gen_div = np.array([np.nanmean([BR.within_window_diversity(S[k, i], events[i]) for k in range(K_STOCH)]) for i in range(len(Y))])
        peaks = [list(ex.map(B._peaks, list(S[k]), chunksize=256)) for k in range(K_STOCH)]
        rsd, n_sd = B.timing_sd(ref, [[peaks[k][i] for k in range(K_STOCH)] for i in range(len(Y))])
        hr_k = np.array([[RP.hr_bpm(peaks[k][i], FS) for k in range(K_STOCH)] for i in range(len(Y))])
        with np.errstate(all="ignore"):
            hr_med = np.nanmedian(hr_k, axis=1)
        out[key] = {"within_window_waveform_diversity": ci(ww_div, Pid), "beat_aligned_diversity": beat_div, "beat_anchors": n_anchor,
                    "generated_over_real_beat_diversity_ratio": float(np.nanmean(gen_div) / np.nanmean(real_div)),
                    "r_time_seed_sd_ms_median": rsd, "r_time_sd_beats": n_sd,
                    "k16_consensus_hr_mae": ci(np.abs(hr_med - hr_ref), Pid), "single_sample_hr_mae": ci(np.abs(hr_k[:, 0] - hr_ref), Pid)}
        print(f"[sf0] stochastic {key}: {json.dumps(B.clean(out[key]))[:300]}", flush=True)
    write_json("stochastic_secondary_val.json", out)


def stage_compute(ex, dev):
    from torch.utils.flop_counter import FlopCounterMode
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, _, Pid, wid, _ = load_role("sf_val", ex)
    pick = B.salted_rank("sf0-latency-v1", range(len(X)))[:50]
    torch.set_num_threads(4)
    lat, flops = {}, {}
    for dname in ("cpu", "cuda"):
        d = torch.device(dname)
        detn = load_net("detector", d)
        nets = {k: load_net(n, d) for k, n in (("WWL1", "wwl1"), ("WWFM", "wwfm"), ("IND", "ind"), ("SF", "sf"))}

        def run(k, i):
            xi = torch.from_numpy(X[i:i + 1]).to(d)
            with torch.no_grad():
                p = torch.sigmoid(detn(xi[:, None])[:, 0]).float().cpu().numpy()[0]
                r = torch.from_numpy(AB.event_raster([_extract(p)])).to(d)
                if k == "WWL1":
                    return nets[k](xi, r)
                return M.euler(nets[k], torch.from_numpy(M.window_noise(Pid[i:i + 1], wid[i:i + 1])).to(d), xi, r, NFE)

        for k in ARMS:
            ts = []
            for rep, i in enumerate(list(pick[:3]) + list(pick)):
                if dname == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                run(k, int(i))
                if dname == "cuda":
                    torch.cuda.synchronize()
                if rep >= 3:
                    ts.append((time.perf_counter() - t0) * 1000)
            lat[f"{dname}_{k}"] = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}
            if dname == "cpu":
                xi = torch.from_numpy(X[:1])
                r = torch.zeros(1, T)
                fc = FlopCounterMode(display=False)
                with fc, torch.no_grad():
                    nets[k](xi, r) if k == "WWL1" else nets[k](xi.clone(), torch.zeros(1), xi, r)
                fd = FlopCounterMode(display=False)
                with fd, torch.no_grad():
                    detn(xi[:, None])
                flops[k] = {"per_vector_field_eval": int(fc.get_total_flops()), "detector": int(fd.get_total_flops()),
                            "total_pipeline": int(fd.get_total_flops() + fc.get_total_flops() * (1 if k == "WWL1" else NFE)), "nfe": 1 if k == "WWL1" else NFE}
    ck = {n: json.loads((ART / f"checkpoint_{n}.json").read_text()) for n in ("detector", "wwl1", "wwfm", "ind", "sf")}
    write_json("compute_accounting.json", {
        "params": {k: ck[n]["n_params"] for k, n in (("WWL1", "wwl1"), ("WWFM", "wwfm"), ("IND", "ind"), ("SF", "sf"))} | {"detector": ck["detector"]["n_params"]},
        "pipeline_params": {k: ck[n]["n_params"] + ck["detector"]["n_params"] for k, n in (("WWL1", "wwl1"), ("WWFM", "wwfm"), ("IND", "ind"), ("SF", "sf"))},
        "train_seconds": {n: ck[n]["train_seconds"] for n in ck}, "peak_gpu_mem_mib": {n: ck[n]["peak_gpu_mem_mib"] for n in ck},
        "nan_steps": {n: ck[n]["nan_steps"] for n in ck}, "batch1_latency_ms_full_pipeline": lat, "flops": flops,
        "notes": "full pipeline = SF detector + waveform model (NFE 8 for FM arms); FLOPs from torch.utils.flop_counter on one window (CPU), "
                 "a lower bound; parameter-matched, not compute-matched", "software": B.software()})
    write_json("checkpoint_hashes.json", {n: ck[n]["sha256"] for n in ck})


# ----------------------------------------------------------------------------------------------- figure / table
def stage_figure_val(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = json.loads((ART / "val_metrics.json").read_text())["summary"]
    b = json.loads((ART / "val_bootstrap.json").read_text())["comparisons"]
    g = json.loads((ART / "val_gates.json").read_text())
    sh = json.loads((ART / "condition_shuffle_val.json").read_text())
    cc = json.loads((ART / "compute_accounting.json").read_text()) if (ART / "compute_accounting.json").exists() else None
    col = {"WWL1": "#0e7c86", "WWFM": "#a23b52", "IND": "#d59a54", "SF": "#4a3aa7"}
    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
    a = ax[0, 0]
    a.axis("off")
    a.set_title("A  ScaleFlow-coupled (vector field)", loc="left")
    a.text(0.02, 0.95, "PPG / event raster / x_t\n        |\n   fixed Haar (2 levels)\n        |\n coarse(128) -> mid(128) -> fine(256)\n"
           "   1x1 projections, nearest x2 up\n   (no fine -> coarse path)\n        |\n   inverse Haar\n        |\n whole-window velocity v(x_t, t)\n\n"
           "Euler, NFE = 8, one fixed noise per window", va="top", family="monospace", fontsize=10)
    def bars(a, keys, labels, title):
        for i, k in enumerate(keys):
            v = b[k] if isinstance(b[k], list) else b[k]
            a.plot([i, i], [v[1], v[2]], color="k", lw=3)
            a.plot([i], [v[0]], "o", color="k")
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xticks(range(len(keys)), labels, fontsize=8)
        a.set_title(title, loc="left")
    a = ax[0, 1]
    for i, (k, lab) in enumerate((("fd", "FD"), ("corr", "beat corr"), ("fp", "FP / win"))):
        v = b["G1"][k]
        a.plot([i, i], [v[1], v[2]], color=col["WWFM"], lw=3)
        a.plot([i], [v[0]], "o", color=col["WWFM"])
    a.axhline(0, color="0.6", lw=0.8)
    a.set_xticks(range(3), ["FD", "beat corr", "FP / win"])
    a.set_title(f"B  WW-FM - WW-L1 (G1: {g['G1']})", loc="left")
    a = ax[0, 2]
    bars(a, ["fd_sf_wwfm", "fd_sf_best", "fd_sf_ind"], ["SF - WW-FM (G2)", f"SF - best ({b['best_fd_baseline']}, G3)", "SF - IND (G6)"],
         "C  FD effects (CI must be < 0)")
    a = ax[1, 0]
    for k in ARMS:
        a.scatter([m[k]["fd"]], [m[k]["pm_fp_rate"][0]], s=80, color=col[k])
        a.annotate(LABEL[k], (m[k]["fd"], m[k]["pm_fp_rate"][0]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    a.set_xlabel("FD (lower = better)")
    a.set_ylabel("false R / window (patient mean)")
    a.set_title("D  FD vs FP (SF-VAL; no combined score)", loc="left")
    a = ax[1, 1]
    nn_ = lambda v: [np.nan if x is None else x for x in v]  # noqa: E731
    for i, k in enumerate(ARMS):
        v = nn_(m[k]["beat_corr"])
        a.bar(i, v[0], color=col[k])
        a.plot([i, i], [v[1], v[2]], color="k")
    pts = [nn_(m[k]["beat_corr"])[0] for k in ARMS]
    if np.isfinite(pts).any():
        a.set_ylim(np.nanmin(pts) - 0.05, np.nanmax(pts) + 0.02)
    a.set_xticks(range(4), [LABEL[k] for k in ARMS], fontsize=7)
    a.set_title("E  beat-aligned morphology correlation", loc="left")
    a = ax[1, 2]
    v1, v2 = sh["ppg_shuffle"]["fd_shuf_minus_cond"], sh["ppg_shuffle"]["corr_shuf_minus_cond"]
    a.plot([0, 0], [v1[1], v1[2]], color=col["SF"], lw=3); a.plot([0], [v1[0]], "o", color=col["SF"])
    a2 = a.twinx()
    a2.plot([1, 1], [v2[1], v2[2]], color="0.4", lw=3); a2.plot([1], [v2[0]], "o", color="0.4")
    a.axhline(0, color="0.7", lw=0.6)
    a.set_xticks([0, 1], ["FD shuffled - conditioned\n(must be > 0)", "corr shuffled - conditioned\n(must be < 0, right axis)"], fontsize=8)
    a.set_title(f"F  PPG-shuffle control (G7 {'PASS' if g['G7'] else 'FAIL'})", loc="left")
    fig.suptitle(f"SF0 ScaleFlow — SF-VAL only (TEST not opened). Gates: " + ", ".join(f"{k} {'PASS' if g[k] else 'FAIL'}" for k in ("G2", "G3", "G4", "G5", "G6", "G7"))
                 + f" -> {'QUALIFIED' if g['QUALIFIED'] else 'NOT QUALIFIED'}", fontsize=11)
    fig.tight_layout()
    fig.savefig(ART / "figure_val.png", dpi=110)
    with open(ART / "table_val.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "method", "params", "flops_pipeline", "nfe", "fd", "morph_corr", "fp_per_window", "recall", "f1", "rr_mae_ms", "hr_mae_bpm",
                    "mae", "gpu_latency_ms"])
        fmt = lambda v, spec: "" if v is None else format(v, spec)  # noqa: E731
        for k in ARMS:
            s = m[k]
            w.writerow(["SF-VAL", LABEL[k], cc["params"][k] if cc else "", cc["flops"][k]["total_pipeline"] if cc else "", 1 if k == "WWL1" else NFE,
                        fmt(s["fd"], ".2f"), fmt(s["beat_corr"][0], ".4f"), fmt(s["pm_fp_rate"][0], ".4f"), fmt(s["pm_recall"][0], ".4f"),
                        fmt(s["pm_f1"][0], ".4f"), fmt(s["rr_mae_ms"][0], ".3f"), fmt(s["hr_abs_err"][0], ".2f"), fmt(s["mae"][0], ".4f"),
                        fmt(cc["batch1_latency_ms_full_pipeline"][f"cuda_{k}"]["median_ms"], ".2f") if cc else ""])


# ----------------------------------------------------------------------------------------------- final test (sealed)
def stage_freeze(ex, dev):
    g = json.loads((ART / "val_gates.json").read_text())
    if not g["QUALIFIED"]:
        raise SystemExit("SF0: not qualified on SF-VAL; TEST stays closed")
    audit = (ART / "final_test_freshness_audit.md").read_text()
    files = [PREREG, *CODE_FILES, *[f"outputs/sf0_scaleflow/{n}.pt" for n in ("detector", "wwl1", "wwfm", "ind", "sf")], "scripts/c0_coherentbeat.py",
             "scripts/c0a_ablation.py", "scripts/bf0_run.py", "src/ppg2ecg/coherentbeat/model.py", "src/ppg2ecg/coherentbeat/ablation.py",
             "src/ppg2ecg/evaluation/rpeaks.py", "src/ppg2ecg/evaluation/paper_metrics.py", "src/ppg2ecg/probes/rhythm_tcn.py",
             "src/ppg2ecg/rhythmfield/model.py", "artifacts/sf0_scaleflow/split_manifest.json", "artifacts/sf0_scaleflow/val_gates.json",
             "artifacts/sf0_scaleflow/final_test_freshness_audit.md", "data/manifests/split_v1_vitaldb_seed42.json"]
    write_json("final_test_freeze_manifest.json", {"qualified": True, "freshness_audit": "CLEAN" if "VERDICT: CLEAN" in audit else "NOT CLEAN",
                                                   "sha256": {f: B.sha256_file(ROOT / f) for f in files}})


def stage_evaluate_test(ex, dev):
    r = evaluate("test", ex, dev)
    write_json("test_metrics.json", {k: r[k] for k in ("role", "windows", "patients", "windows_without_events", "matched_pairs", "summary", "raw")})
    write_json("test_bootstrap.json", {"unit": "patient", "replicates": BOOT_N, "seed": BOOT_SEED, "comparisons": r["comparisons"]})
    reversal = r["comparisons"]["fd_sf_wwfm"][0] >= 0
    verdict = "STRONG" if r["gates"]["QUALIFIED"] else ("FAILED" if reversal else "PARTIAL")
    write_json("test_gates.json", {"G1": r["g1"], **r["gates"], "major_reversal_G2_point": bool(reversal), "verdict": verdict})
    write_json("condition_shuffle_test.json", r["shuffle"])


STAGES = {"split": stage_split, "audit": stage_audit, "manifest": stage_manifest, "train_detector": stage_train_detector,
          "train_wwl1": stage_train_wwl1, "train_wwfm": stage_train_wwfm, "train_ind": stage_train_ind, "train_sf": stage_train_sf,
          "evaluate_val": stage_evaluate_val, "nfe": stage_nfe, "stochastic": stage_stochastic, "compute": stage_compute,
          "figure_val": stage_figure_val, "freeze": stage_freeze, "evaluate_test": stage_evaluate_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(10) as ex:
        STAGES[args.stage](ex, dev)


if __name__ == "__main__":
    main()
