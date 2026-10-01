"""R1 — RhythmField-WW, the final architecture experiment of this branch (docs/R1_RHYTHMFIELD_WW_PREREGISTRATION.md).

Frozen and never modified: C0 split, timing detector, placed events, CoherentBeat-C0, WW-DET (C0-A), metric code.
New, trained once on ARCH-TRAIN (seed 42, last checkpoint): SOFT-RHYTHM-WW, JOINT-RHYTHMFIELD-WW.
ARCH-VAL qualifies and selects; ARCH-HOLDOUT is not used; the old V1 TEST opens only after a qualifying candidate, a clean
freshness audit and a committed final freeze.

Stages: audit, manifest, train_soft, train_joint, evaluate_val, compute, figure_val, freeze, evaluate_test, figure_test
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/r1_rhythmfield.py <stage>
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
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
from ppg2ecg.eventaudit import taxonomy as TX  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import extract_events, soft_event_field  # noqa: E402
from ppg2ecg.rhythmfield import model as RM  # noqa: E402

PREREG = "docs/R1_RHYTHMFIELD_WW_PREREGISTRATION.md"
CODE_FILES = ("scripts/r1_rhythmfield.py", "src/ppg2ecg/rhythmfield/__init__.py", "src/ppg2ecg/rhythmfield/model.py",
              "tests/test_r1_rhythmfield.py")
ART = ROOT / "artifacts/r1_rhythmfield_ww"
OUT = ROOT / "outputs/r1_rhythmfield_ww"
FREEZE = ART / "final_test_freeze_manifest.json"
FS, T = 128, 512
SEED = 42
PROTO = dict(C0.C0T)                                  # WW-DET protocol: 20,000 x 64 windows, AdamW 1e-3 / 0.01, clip 1.0
ARMS = ("C0", "WW", "SOFT", "JOINT")
RANDOM_FREE_MS = 150.0                                # event-free positions: >= 150 ms from every reference R and placed event
N_RANDOM_FREE = 20000


def write_json(name, obj):
    ART.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(json.dumps(B.clean(obj), indent=1))


# ----------------------------------------------------------------------------------------------- frozen inputs
@torch.no_grad()
def detector_field(X, dev) -> np.ndarray:
    """The frozen C0 timing detector's PRE-THRESHOLD output: sigmoid(logits), [N, 512] at the ECG sample grid (the
    detector is fully convolutional at 128 Hz, so no interpolation). Identical to the tensor C0.detect_events thresholds."""
    det = C0.load_net("detector", dev)
    out = []
    for i in range(0, len(X), 2048):
        out.append(torch.sigmoid(det(torch.from_numpy(X[i:i + 2048]).to(dev)[:, None])[:, 0]).float().cpu().numpy())
    return np.concatenate(out)


def rhythm_target(ref) -> np.ndarray:
    """C0 detector target convention: max of Gaussians, sigma = 20 ms, at ARCH-TRAIN reference R (BCE with logits)."""
    return np.stack([soft_event_field(np.asarray(r, float), T, RM.RHYTHM_SIGMA_MS / 1000.0 * FS) for r in ref]).astype(np.float32)


def old_test_cases():
    return C0.manifest()["splits"][0]["test"]


def check_final_freeze():
    """The old V1 TEST is sealed until a committed final-test freeze manifest exists and every frozen file is unchanged."""
    if not FREEZE.exists():
        raise PermissionError("R1: old V1 TEST sealed (no final-test freeze manifest)")
    fm = json.loads(FREEZE.read_text())
    if fm.get("selected") not in ("SOFT", "JOINT") or fm.get("freshness_audit") != "CLEAN":
        raise PermissionError("R1: old V1 TEST sealed (no qualifying candidate or freshness audit not clean)")
    for f, h in fm["sha256"].items():
        if B.sha256_file(ROOT / f) != h:
            raise PermissionError(f"R1: frozen file changed: {f}")
    rel = str(FREEZE.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise PermissionError("R1: final-test freeze manifest not committed")


def load_test():
    check_final_freeze()
    X, Y, P = [], [], []
    poc = C0.manifest()["extra"]["patient_of_case"]
    for c in old_test_cases():
        d = np.load(C0.DATA / f"{c}.npz")
        assert int(d["subjectid"]) == int(poc[c]), c
        X.append(d["x"].astype(np.float32)); Y.append(d["y"].astype(np.float64)); P.append(np.full(len(d["x"]), int(d["subjectid"])))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(P)


# ----------------------------------------------------------------------------------------------- audit / manifest
def stage_audit(ex, dev):
    h = {f: B.sha256_file(ROOT / f) for f in ("outputs/c0_coherentbeat/detector.pt", "outputs/c0_coherentbeat/c0.pt",
                                             "outputs/c0a_coherentbeat_ablation/ww.pt", "outputs/c0_coherentbeat/val_renders.npz",
                                             "artifacts/c0_coherentbeat/split_manifest.json", "artifacts/c0a_coherentbeat_ablation/val_metrics.json",
                                             "artifacts/c0a_coherentbeat_ablation/model_configs.json")}
    c0h = json.loads((C0.ART / "checkpoint_hashes.json").read_text())
    cah = json.loads((CA.ART / "checkpoint_hashes.json").read_text())
    ok = {"detector": h["outputs/c0_coherentbeat/detector.pt"] == c0h["detector"], "c0": h["outputs/c0_coherentbeat/c0.pt"] == c0h["c0"],
          "ww": h["outputs/c0a_coherentbeat_ablation/ww.pt"] == cah["ww"]}
    if not all(ok.values()):
        raise SystemExit(f"STOP: frozen checkpoint mismatch {ok}")
    write_json("input_hashes.json", {"sha256": h, "checkpoints_equal_recorded": ok})
    soft, joint = RM.soft_rhythm_ww(), RM.JointRhythmFieldWW()
    write_json("model_configs.json", {
        "SOFT": {"class": "ppg2ecg.coherentbeat.ablation.WWDet (WW-DET, unchanged)", "dec_ch": RM.WW_DEC_CH, "n_dec": RM.WW_N_DEC,
                 "conditioning": "frozen detector sigmoid(logits) [B, 512] in place of the Gaussian event raster", "params": RM.n_params(soft)},
        "JOINT": {"class": "ppg2ecg.rhythmfield.model.JointRhythmFieldWW", "encoder": "C0 Encoder (64 ch, 8 blocks)",
                  "rhythm_head_def": "1x1 64->64, GELU, 1x1 64->1 -> z_R [B, 512]; p_R = sigmoid(z_R), not detached",
                  "decoder": "WW-DET decoder (1x1 on [h, p_R] -> 5 residual blocks, dilations 1,2,4,8,16 -> 1x1)", "params": RM.param_breakdown(joint)},
        "WW_DET_params": 593577, "detector_params": sum(p.numel() for p in C0.load_net("detector", "cpu").parameters())})
    write_json("rhythm_field_config.json", {
        "soft_field": "sigmoid of the frozen C0 RhythmTCN logits, the exact tensor C0.detect_events thresholds; no temperature, "
                      "recalibration, sharpening or threshold", "resolution": "detector output length = 512 = ECG window, no interpolation",
        "joint_target": {"convention": "C0 timing detector (RD1 family)", "field": "max of Gaussians at ARCH-TRAIN reference R",
                         "sigma_ms": RM.RHYTHM_SIGMA_MS, "loss": "BCEWithLogitsLoss (mean), lambda = 1.0"},
        "diagnostic_extraction": {"threshold": RM.DET_THRESHOLD, "refractory_samples": RM.DET_REFRACTORY}})
    write_json("training_manifest.json", {"role": "ARCH-TRAIN only", "seed": SEED, "protocol": PROTO, "checkpoint": "last step",
                                          "SOFT": {"loss": "full-window L1", "input_field": "frozen detector field on ARCH-TRAIN PPG"},
                                          "JOINT": {"loss": "full-window L1 + 1.0 x BCE(z_R, Gaussian target)"}})
    print(json.dumps(ok), flush=True)


def stage_manifest(ex, dev):
    write_json("prereg_manifest.json", {"prereg": {PREREG: B.sha256_file(ROOT / PREREG)}, "code": {f: B.sha256_file(ROOT / f) for f in CODE_FILES},
                                        "frozen_design": {f: B.sha256_file(ART / f) for f in ("model_configs.json", "rhythm_field_config.json",
                                                                                              "training_manifest.json", "input_hashes.json")},
                                        "written_before_any_r1_training_or_metric": True, "software": B.software()})


# ----------------------------------------------------------------------------------------------- training
def _save(name, net, secs, nan_steps, extra):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{name}.pt"
    meta = {"seed": SEED, "train_seconds": secs, "n_params": RM.n_params(net), "nan_steps": nan_steps, "checkpoint": "last step (no selection)",
            "peak_gpu_mem_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "optimizer": "AdamW",
            "lr": PROTO["lr"], "weight_decay": PROTO["wd"], "batch": PROTO["batch"], "steps": PROTO["steps"], "clip": PROTO["clip"]} | extra
    torch.save({"state_dict": net.state_dict()} | meta, f)
    meta["sha256"] = B.sha256_file(f)
    write_json(f"checkpoint_{name}.json", meta)


def _train(name, ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, Y, _ = C0.load_arch("train")
    ref = C0.reference_peaks("train", Y, ex)
    Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y.astype(np.float32)).to(dev)
    if name == "soft":
        Ft = torch.from_numpy(detector_field(X, dev).astype(np.float16)).to(dev)          # frozen detector, inference only
        net = RM.soft_rhythm_ww()
    else:
        Ft = torch.from_numpy(rhythm_target(ref).astype(np.float16)).to(dev)              # ARCH-TRAIN reference R only
        net = RM.JointRhythmFieldWW()
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=PROTO["lr"], weight_decay=PROTO["wd"])
    bce = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    batches, t0, acc, accr, nan_steps = C0._epoch_batches(len(Xt), PROTO["batch"], g), time.time(), [], [], 0
    net.train()
    for step in range(1, PROTO["steps"] + 1):
        bd = next(batches).to(dev)
        if name == "soft":
            loss = (net(Xt[bd], Ft[bd].float()) - Yt[bd]).abs().mean()
            lr_ = torch.zeros(())
        else:
            y, z = net(Xt[bd])
            lr_ = bce(z, Ft[bd].float())
            loss = (y - Yt[bd]).abs().mean() + RM.LAMBDA_RHYTHM * lr_
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), PROTO["clip"])
        opt.step(); acc.append(loss.item()); accr.append(lr_.item())
        if step % 1000 == 0:
            print(f"[r1] {name} step {step} loss {np.mean(acc):.5f} rhythm {np.mean(accr):.5f}", flush=True); acc, accr = [], []
    _save(name, net, time.time() - t0, nan_steps, {"loss": "L1" if name == "soft" else "L1 + 1.0 x BCE"})


def stage_train_soft(ex, dev):
    _train("soft", ex, dev)


def stage_train_joint(ex, dev):
    _train("joint", ex, dev)


def load_new(name, dev):
    ck = torch.load(OUT / f"{name}.pt", map_location="cpu")
    net = RM.soft_rhythm_ww() if name == "soft" else RM.JointRhythmFieldWW()
    net.load_state_dict(ck["state_dict"])
    return net.to(dev).eval()


@torch.no_grad()
def render_soft(X, field, dev, bs=256):
    net = load_new("soft", dev)
    return np.concatenate([net(torch.from_numpy(X[i:i + bs]).to(dev), torch.from_numpy(field[i:i + bs]).to(dev)).cpu().numpy()
                           for i in range(0, len(X), bs)]).astype(np.float64)


@torch.no_grad()
def render_joint(X, dev, bs=256):
    net = load_new("joint", dev)
    ys, ps = [], []
    for i in range(0, len(X), bs):
        y, _, p = net(torch.from_numpy(X[i:i + bs]).to(dev), return_field=True)
        ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy())
    return np.concatenate(ys).astype(np.float64), np.concatenate(ps)


# ----------------------------------------------------------------------------------------------- evaluation
def _extract(p):
    return extract_events(p, RM.DET_THRESHOLD, RM.DET_REFRACTORY)


def evaluate(role, ex, dev) -> dict:
    if role == "val":
        X, Y, Pid = C0.load_arch("val")
        ref = C0.reference_peaks("val", Y, ex)
    else:
        X, Y, Pid = load_test()
        ref = list(ex.map(B._peaks, list(Y), chunksize=256))
    N = len(Y)
    evaluable = np.array([len(r) > 0 for r in ref])
    fdet = detector_field(X, dev)
    events = [np.asarray(e, int) for e in ex.map(_extract, list(fdet), chunksize=256)]
    waves = {}
    waves["C0"], _, _ = C0.render_c0("c0", X, events, dev, C0.train_rr_median())
    waves["WW"] = CA.render_ww(X, events, dev)
    repro = {}
    if role == "val":
        st = np.load(C0.OUT / "val_renders.npz")
        stored_ev = [np.asarray(e, int) for e in B.split_list(st["events_idx"], st["events_off"])]
        if not all(np.array_equal(a, b) for a, b in zip(events, stored_ev)):
            raise SystemExit("STOP: frozen events not reproduced from the pre-threshold field")
        if np.max(np.abs(waves["C0"].astype(np.float32) - st["C0"])) != 0.0:
            raise SystemExit("STOP: frozen C0 render not reproduced")
    waves["SOFT"] = render_soft(X, fdet, dev)
    waves["JOINT"], fjoint = render_joint(X, dev)
    det = {k: list(ex.map(B._peaks, list(waves[k]), chunksize=256)) for k in ARMS}
    per = {}
    for k in ARMS:
        per[k] = CA.arm_metrics(waves[k], Y, ref, det[k], evaluable, ex)
        prf = PMX.rpeak_prf_at(waves[k], Y, FS, 50.0, peaks=(ref, det[k]))
        per[k]["n_tp"] = prf["n_tp"]
    pprf = PMX.rpeak_prf_at(Y, Y, FS, 50.0, peaks=(ref, events))
    pbl = PMX.beat_level_metrics(Y, Y, FS, 50.0, peaks=(ref, events))
    per["PLACED"] = {"f1": np.where(evaluable, pprf["rpeak_f1"], np.nan), "precision": pprf["rpeak_precision"], "recall": pprf["rpeak_recall"],
                     "fp": pprf["n_fp"], "fn": pprf["n_fn"], "n_tp": pprf["n_tp"], "rr_mae_ms": pbl["rr_mae_ms"], "hr_abs_err": pbl["hr_abs_err"]}
    hd = [np.asarray(e, int) for e in ex.map(_extract, list(fjoint), chunksize=256)]
    hprf = PMX.rpeak_prf_at(Y, Y, FS, 50.0, peaks=(ref, hd))
    hbl = PMX.beat_level_metrics(Y, Y, FS, 50.0, peaks=(ref, hd))
    per["JOINT_HEAD"] = {"f1": np.where(evaluable, hprf["rpeak_f1"], np.nan), "precision": hprf["rpeak_precision"], "recall": hprf["rpeak_recall"],
                         "fp": hprf["n_fp"], "fn": hprf["n_fn"], "n_tp": hprf["n_tp"], "rr_mae_ms": hbl["rr_mae_ms"], "hr_abs_err": hbl["hr_abs_err"]}
    pairs = [BR.matched_pairs(ref[i], events[i], T) for i in range(N)]
    corr = {k: [BR.pair_correlations(Y[i], waves[k][i], pairs[i]) for i in range(N)] for k in ARMS}
    wm = B.window_pair_means(corr, ARMS)
    for k in ARMS:
        per[k]["beat_corr"] = wm[k]
    # ---- frozen reference arms must reproduce C0-A's stored window-level event metrics exactly (ARCH-VAL)
    if role == "val":
        stored = json.loads((CA.ART / "val_metrics.json").read_text())["summary"]
        for k in ("PLACED", "C0", "WW"):
            for m in ("f1", "precision", "recall", "fp", "fn", "rr_mae_ms", "hr_abs_err") + (("mae", "pcc", "s4", "s5", "spectral_ratio_dev") if k != "PLACED" else ()):
                a = C0.cluster_ci(per[k][m], Pid)
                dv = max(abs(x - y) for x, y in zip(a, stored[k][m]))
                repro[f"{k}.{m}"] = dv
        for k in ("C0", "WW"):
            repro[f"{k}.fd"] = abs(float(PMX.kanflow_fd(waves[k], Y)) - stored[k]["fd"])
        if max(repro.values()) > 1e-9:
            raise SystemExit(f"STOP: frozen C0 / WW-DET metrics not reproduced: {max(repro.values())}")
    # ---- patient-macro event metrics (prereg §6) and pooled / window-level continuity
    pm = {k: RM.patient_macro_rows(per[k]["n_tp"], per[k]["fp"], per[k]["fn"], Pid) for k in ("PLACED", "JOINT_HEAD") + ARMS}
    subs = pm["WW"]["patients"]
    pci = lambda v: C0.cluster_ci(v, subs)  # noqa: E731
    summary = {}
    for k in pm:
        s = {f"pm_{m}": pci(pm[k][m]) for m in ("precision", "recall", "f1", "fp_rate")}
        s["pooled"] = RM.pooled_prf(**pm[k]["pooled"])
        s["window_f1_historical"] = C0.cluster_ci(per[k]["f1"], Pid)
        s["rr_mae_ms"] = C0.cluster_ci(per[k]["rr_mae_ms"], Pid)
        s["hr_abs_err"] = C0.cluster_ci(per[k]["hr_abs_err"], Pid)
        if k in ARMS:
            for m in ("mae", "pcc", "s4", "s5", "spectral_ratio_dev", "beat_corr"):
                s[m] = C0.cluster_ci(per[k][m], Pid)
            s["fd"] = float(PMX.kanflow_fd(waves[k], Y))
        summary[k] = s
    comps = {}
    for cand in ("SOFT", "JOINT"):
        d = lambda m, o: pci(pm[cand][m] - pm[o][m])  # noqa: E731
        comps[cand] = {"fp_vs_ww": d("fp_rate", "WW"), "fp_vs_c0": d("fp_rate", "C0"), "recall_vs_ww": d("recall", "WW"),
                       "recall_vs_c0": d("recall", "C0"), "precision_vs_ww": d("precision", "WW"), "f1_vs_ww": d("f1", "WW"),
                       "f1_vs_c0": d("f1", "C0"), "corr_vs_ww": C0.cluster_ci(per[cand]["beat_corr"] - per["WW"]["beat_corr"], Pid),
                       "corr_vs_c0": C0.cluster_ci(per[cand]["beat_corr"] - per["C0"]["beat_corr"], Pid),
                       "mae_vs_ww": C0.cluster_ci(per[cand]["mae"] - per["WW"]["mae"], Pid),
                       "fd_vs_ww": C0.fd_diff_ci(waves[cand], waves["WW"], Y, Pid), "fd_vs_c0": C0.fd_diff_ci(waves[cand], waves["C0"], Y, Pid)}
        print(f"[r1] {role} {cand}: {json.dumps(B.clean({k: comps[cand][k] for k in ('fp_vs_ww', 'fp_vs_c0', 'recall_vs_ww', 'fd_vs_ww', 'fd_vs_c0', 'corr_vs_ww')}))}", flush=True)
    tax = taxonomy(ref, events, det, N)
    diag = field_diagnostics(ref, events, det, fdet, fjoint, tax["_cls"])
    tax.pop("_cls")
    raw = {k: {"fp": int(np.nansum(per[k]["fp"])), "fn": int(np.nansum(per[k]["fn"]))} for k in per}
    return {"role": role, "windows": N, "patients": int(subs.size), "windows_without_events": int(sum(len(e) == 0 for e in events)),
            "matched_pairs": wm["_n_pairs"], "summary": summary, "comparisons": comps, "raw": raw, "reproduction_max_abs_dev": repro,
            "taxonomy": tax, "field_diagnostics": diag}


def taxonomy(ref, events, det, N) -> dict:
    """E0 four-way taxonomy (descriptive), placed = the frozen detector events, for every waveform arm."""
    out, cls = {}, {}
    for k in ARMS:
        cls[k] = [TX.classify(ref[i], events[i], det[k][i]) for i in range(N)]
        c = {t: int(sum(np.sum(x["type"] == t) for x in cls[k])) for t in TX.TYPES}
        reg = {r: {t: 0 for t in ("C", "D")} for r in TX.REGIONS}
        for i in range(N):
            for j, t in enumerate(det[k][i]):
                ty = cls[k][i]["type"][j]
                if ty in ("C", "D"):
                    reg[TX.localize(int(t), events[i])["region"]][ty] += 1
        out[k] = {"counts": c, "per_window": {t: c[t] / N for t in TX.TYPES}, "C_D_by_region": reg}
    out["_cls"] = cls
    return out


def _vals(field, idx_by_window):
    return np.concatenate([field[i][np.asarray(ix, int)] for i, ix in enumerate(idx_by_window) if len(ix)]) if any(len(ix) for ix in idx_by_window) else np.zeros(0)


def _desc(v):
    v = np.asarray(v, float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "frac_ge_threshold": float(np.mean(v >= RM.DET_THRESHOLD))}


def field_diagnostics(ref, events, det, fdet, fjoint, cls) -> dict:
    """Dense-field values (rhythm score, not a calibrated probability) at reference R, correctly placed events, missed
    reference R, random event-free positions, and at the rendered A / B / C / D detections of each candidate."""
    N = len(ref)
    rp = [dict(TX.RP.match_rpeaks(np.asarray(ref[i], int), np.asarray(events[i], int), FS, 50.0)[0]) for i in range(N)]
    correct = [[int(events[i][j]) for j in rp[i].values()] for i in range(N)]
    missed = [[int(r) for a, r in enumerate(ref[i]) if a not in rp[i]] for i in range(N)]
    rng = np.random.default_rng(20261001)
    free = [[] for _ in range(N)]
    tol = RANDOM_FREE_MS / 1000.0 * FS
    for _ in range(N_RANDOM_FREE * 3):
        i, t = int(rng.integers(0, N)), int(rng.integers(0, T))
        occ = np.concatenate([np.asarray(ref[i], float), np.asarray(events[i], float)])
        if occ.size == 0 or np.min(np.abs(occ - t)) >= tol:
            free[i].append(t)
        if sum(len(f) for f in free) >= N_RANDOM_FREE:
            break
    out = {}
    for name, field in (("detector_field", fdet), ("joint_field", fjoint)):
        out[name] = {"reference_R": _desc(_vals(field, ref)), "correctly_placed_events": _desc(_vals(field, correct)),
                     "missed_reference_R": _desc(_vals(field, missed)), "random_event_free": _desc(_vals(field, free))}
    for k, field in (("SOFT", fdet), ("JOINT", fjoint), ("WW", fdet)):
        for ty in TX.TYPES:
            ix = [[int(det[k][i][j]) for j in range(len(det[k][i])) if cls[k][i]["type"][j] == ty] for i in range(N)]
            out.setdefault(f"{k}_rendered_types", {})[ty] = _desc(_vals(field, ix))
    ix = [[int(det["JOINT"][i][j]) for j in range(len(det["JOINT"][i])) if cls["JOINT"][i]["type"][j] in ("C", "D")] for i in range(N)]
    out["JOINT_spontaneous_CD_detector_field"] = _desc(_vals(fdet, ix))
    return out


def stage_evaluate_val(ex, dev):
    if not (ART / "prereg_manifest.json").exists():
        raise SystemExit("STOP: R1 metrics only after the committed preregistration")
    r = evaluate("val", ex, dev)
    qual = {c: RM.val_gates(r["comparisons"][c]) for c in ("SOFT", "JOINT")}
    sel = RM.select_candidate({c: qual[c]["QUALIFIED"] for c in qual}, {c: r["summary"][c]["fd"] for c in qual},
                              {c: r["summary"][c]["pm_fp_rate"][0] for c in qual})
    write_json("val_metrics.json", {k: r[k] for k in ("role", "windows", "patients", "windows_without_events", "matched_pairs", "summary", "raw",
                                                       "reproduction_max_abs_dev")})
    write_json("val_bootstrap.json", {"unit": "patient", "replicates": C0.BOOT_N, "seed": C0.BOOT_SEED, "comparisons": r["comparisons"]})
    write_json("val_qualification.json", qual)
    write_json("candidate_selection.json", {"selected": sel, "rule": "prereg §8 (lower FD; |dFD| < 0.25 -> lower FP; |dFP| < 0.01 -> SOFT)",
                                            "fd": {c: r["summary"][c]["fd"] for c in qual}, "fp": {c: r["summary"][c]["pm_fp_rate"][0] for c in qual}})
    write_json("event_taxonomy_val.json", r["taxonomy"])
    write_json("rhythm_field_diagnostics.json", {"val": r["field_diagnostics"]})
    print(f"[r1] qualification {json.dumps(qual)} selected {sel}", flush=True)


# ----------------------------------------------------------------------------------------------- compute
def stage_compute(ex, dev):
    from torch.utils.flop_counter import FlopCounterMode
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, _, _ = C0.load_arch("val")
    rr_edge = C0.train_rr_median()
    pick = B.salted_rank("r1-latency-v1", range(len(X)))[:50]
    torch.set_num_threads(4)
    lat, flops = {}, {}
    for dname in ("cpu", "cuda"):
        d = torch.device(dname)
        detn = C0.load_net("detector", d)
        nets = {"C0": C0.load_net("c0", d), "WW": CA.load_new("ww", d), "SOFT": load_new("soft", d), "JOINT": load_new("joint", d)}

        def run(kind, i):
            xi = torch.from_numpy(X[i:i + 1]).to(d)
            if kind == "JOINT":
                return nets["JOINT"](xi)
            p = torch.sigmoid(detn(xi[:, None])[:, 0])
            if kind == "SOFT":
                return nets["SOFT"](xi, p)
            ev = C0._extract(p.float().cpu().numpy()[0])
            if kind == "WW":
                return nets["WW"](xi, torch.from_numpy(CA.AB.event_raster([ev])).to(d))
            return nets["C0"](xi, [ev], rr_edge)

        for kind in ("C0", "WW", "SOFT", "JOINT"):
            ts = []
            for rep, i in enumerate(list(pick[:3]) + list(pick)):
                if dname == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    run(kind, int(i))
                if dname == "cuda":
                    torch.cuda.synchronize()
                if rep >= 3:
                    ts.append((time.perf_counter() - t0) * 1000)
            lat[f"{dname}_{kind}"] = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}
            if dname == "cpu":
                fc = FlopCounterMode(display=False)
                with fc, torch.no_grad():
                    run(kind, int(pick[0]))
                flops[kind] = int(fc.get_total_flops())
    ck = {n: json.loads((ART / f"checkpoint_{n}.json").read_text()) for n in ("soft", "joint")}
    det_p = sum(p.numel() for p in C0.load_net("detector", "cpu").parameters())
    jb = RM.param_breakdown(RM.JointRhythmFieldWW())
    write_json("compute_accounting.json", {
        "params": {"SOFT_waveform": ck["soft"]["n_params"], "SOFT_full_pipeline_incl_frozen_detector": ck["soft"]["n_params"] + det_p,
                   "JOINT_total": jb["total"], "JOINT_waveform": jb["waveform"], "JOINT_rhythm_head": jb["rhythm_head"], "JOINT_full_pipeline": jb["total"],
                   "WW_DET_waveform": 593577, "WW_DET_full_pipeline": 593577 + det_p, "C0_waveform": 592770, "C0_full_pipeline": 592770 + det_p,
                   "detector": det_p},
        "train_seconds": {n: ck[n]["train_seconds"] for n in ck}, "peak_gpu_mem_mib": {n: ck[n]["peak_gpu_mem_mib"] for n in ck},
        "nan_steps": {n: ck[n]["nan_steps"] for n in ck}, "batch1_latency_ms_full_pipeline": lat, "flops_one_window_full_pipeline": flops,
        "notes": "full pipeline = detector + waveform model, except JOINT (integrated rhythm head, no external detector); FLOPs from "
                 "torch.utils.flop_counter on one salted ARCH-VAL window (CPU), a lower bound; parameter-matched, not compute-matched",
        "software": B.software()})
    write_json("checkpoint_hashes.json", {n: ck[n]["sha256"] for n in ck})


# ----------------------------------------------------------------------------------------------- figures
def stage_figure_val(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = json.loads((ART / "val_metrics.json").read_text())["summary"]
    b = json.loads((ART / "val_bootstrap.json").read_text())["comparisons"]
    q = json.loads((ART / "val_qualification.json").read_text())
    sel = json.loads((ART / "candidate_selection.json").read_text())["selected"]
    ca = json.loads((CA.ART / "val_metrics.json").read_text())["summary"]
    tx = json.loads((ART / "event_taxonomy_val.json").read_text())
    fd = json.loads((ART / "rhythm_field_diagnostics.json").read_text())["val"]
    col = {"C0": "#0e7c86", "WW": "#a23b52", "SOFT": "#d59a54", "JOINT": "#4a3aa7", "BF0": "0.55", "PM": "0.75"}
    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
    a = ax[0, 0]
    pts = {"BF0-DET (C0-A, historical)": (ca["BF0"]["fd"], ca["BF0"]["fp"][0], col["BF0"]), "PM-BF0 (C0-A, historical)": (ca["PM"]["fd"], ca["PM"]["fp"][0], col["PM"])}
    for k, lab in (("C0", "CoherentBeat-C0"), ("WW", "WW-DET"), ("SOFT", "SOFT-RHYTHM-WW"), ("JOINT", "JOINT-RHYTHMFIELD-WW")):
        pts[lab] = (m[k]["fd"], m[k]["pm_fp_rate"][0], col[k])
    for lab, (x, y, c) in pts.items():
        a.scatter([x], [y], s=70, color=c, zorder=3)
        a.annotate(lab, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    a.set_xlabel("FD (lower = better)")
    a.set_ylabel("false R detections / window (patient mean; lower = better)")
    a.set_title("A  FP vs FD on ARCH-VAL (no combined score)", loc="left")
    names = [("fp_vs_ww", "FP - WW"), ("fp_vs_c0", "FP - C0"), ("recall_vs_ww", "recall - WW"), ("corr_vs_ww", "corr - WW")]
    a = ax[0, 1]
    for j, cand in enumerate(("SOFT", "JOINT")):
        for i, (k, lab) in enumerate(names):
            v = b[cand][k]
            x = i + (j - 0.5) * 0.3
            a.plot([x, x], [v[1], v[2]], color=col[cand], lw=3)
            a.plot([x], [v[0]], "o", color=col[cand], label=cand if i == 0 else None)
    for i, mg in enumerate((0.0, 0.03, -0.005, -0.02)):
        a.plot([i - 0.4, i + 0.4], [mg, mg], "k--", lw=0.8)
    a.axhline(0, color="0.7", lw=0.6)
    a.set_xticks(range(4), [l for _, l in names])
    a.set_title("B  gates V1 / V2 / V3 / V5 (dashed = margin)", loc="left")
    a.legend(fontsize=8)
    a = ax[0, 2]
    for j, cand in enumerate(("SOFT", "JOINT")):
        for i, k in enumerate(("fd_vs_ww", "fd_vs_c0")):
            v = b[cand][k]
            x = i + (j - 0.5) * 0.3
            a.plot([x, x], [v[1], v[2]], color=col[cand], lw=3)
            a.plot([x], [v[0]], "o", color=col[cand], label=cand if i == 0 else None)
    a.plot([-0.4, 0.4], [1.0, 1.0], "k--", lw=0.8)
    a.plot([0.6, 1.4], [0.0, 0.0], "k--", lw=0.8)
    a.set_xticks([0, 1], ["FD - WW (margin +1.0)", "FD - C0 (must be < 0)"])
    a.set_title("C  gate V4", loc="left")
    a.legend(fontsize=8)
    a = ax[1, 0]
    xs = np.arange(4)
    for j, k in enumerate(ARMS):
        a.bar(xs + (j - 1.5) * 0.2, [tx[k]["per_window"][t] for t in TX.TYPES], 0.2, color=col[k], label=k)
    a.set_yscale("log")
    a.set_xticks(xs, ["A", "B", "C", "D"])
    a.set_ylabel("detections / window (log)")
    a.set_title("D  E0 taxonomy (placed = frozen detector events; descriptive)", loc="left")
    a.legend(fontsize=8)
    a = ax[1, 1]
    cats = ("reference_R", "correctly_placed_events", "missed_reference_R", "random_event_free")
    for j, (f, c) in enumerate((("detector_field", col["SOFT"]), ("joint_field", col["JOINT"]))):
        a.bar(np.arange(4) + (j - 0.5) * 0.35, [fd[f][k].get("median", np.nan) for k in cats], 0.35, color=c,
              label=("frozen detector field (SOFT input)" if f == "detector_field" else "JOINT p_R"))
    a.axhline(RM.DET_THRESHOLD, color="k", ls="--", lw=0.8)
    a.set_xticks(range(4), ["reference R", "correct placed", "missed ref R", "random event-free"], fontsize=8)
    a.set_ylabel("median rhythm score")
    a.set_title("E  dense rhythm field (rhythm score, not calibrated)", loc="left")
    a.legend(fontsize=8)
    a = ax[1, 2]
    a.axis("off")
    rows = [["", "V1", "V2", "V3", "V4", "V5", "QUALIFIED"]] + [[c] + ["PASS" if q[c][g] else "FAIL" for g in ("V1", "V2", "V3", "V4", "V5")] +
                                                                ["YES" if q[c]["QUALIFIED"] else "NO"] for c in ("SOFT", "JOINT")]
    tb = a.table(cellText=rows, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(11)
    tb.scale(1, 2.5)
    a.set_title(f"F  ARCH-VAL qualification — selected: {sel}", loc="left")
    fig.suptitle("R1 RhythmField-WW — ARCH-VAL (development / qualification; ARCH-HOLDOUT not used)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ART / "figure_val.png", dpi=110)
    fig2, a = plt.subplots(figsize=(8, 6))
    for lab, (x, y, c) in pts.items():
        a.scatter([x], [y], s=80, color=c, zorder=3)
        a.annotate(lab, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=9)
    a.set_xlabel("FD (lower = better)")
    a.set_ylabel("false R detections / window (lower = better)")
    a.set_title("FP vs FD, ARCH-VAL (ideal: WW-DET FD and CoherentBeat FP)")
    fig2.tight_layout()
    fig2.savefig(ART / "pareto.png", dpi=110)


# ----------------------------------------------------------------------------------------------- final test (sealed)
def stage_freeze(ex, dev):
    sel = json.loads((ART / "candidate_selection.json").read_text())["selected"]
    if sel not in ("SOFT", "JOINT"):
        raise SystemExit("R1: no qualifying candidate; TEST stays closed")
    audit = (ART / "test_freshness_audit.md").read_text()
    status = "CLEAN" if "VERDICT: CLEAN" in audit else "NOT CLEAN"
    files = [PREREG, *CODE_FILES, f"outputs/r1_rhythmfield_ww/{sel.lower()}.pt", "outputs/c0_coherentbeat/detector.pt", "outputs/c0_coherentbeat/c0.pt",
             "outputs/c0a_coherentbeat_ablation/ww.pt", "scripts/c0_coherentbeat.py", "scripts/c0a_ablation.py", "scripts/bf0_run.py",
             "src/ppg2ecg/coherentbeat/model.py", "src/ppg2ecg/coherentbeat/ablation.py", "src/ppg2ecg/evaluation/rpeaks.py",
             "src/ppg2ecg/evaluation/paper_metrics.py", "src/ppg2ecg/probes/rhythm_tcn.py", "src/ppg2ecg/eventaudit/taxonomy.py",
             "artifacts/r1_rhythmfield_ww/val_qualification.json", "artifacts/r1_rhythmfield_ww/candidate_selection.json",
             "artifacts/r1_rhythmfield_ww/test_freshness_audit.md", "data/manifests/split_v1_vitaldb_seed42.json"]
    write_json("final_test_freeze_manifest.json", {"selected": sel, "freshness_audit": status, "sha256": {f: B.sha256_file(ROOT / f) for f in files}})


def stage_evaluate_test(ex, dev):
    r = evaluate("test", ex, dev)
    sel = json.loads(FREEZE.read_text())["selected"]
    v = RM.test_verdict(r["comparisons"][sel])
    write_json("test_metrics.json", {k: r[k] for k in ("role", "windows", "patients", "windows_without_events", "matched_pairs", "summary", "raw")})
    write_json("test_bootstrap.json", {"unit": "patient", "replicates": C0.BOOT_N, "seed": C0.BOOT_SEED, "selected": sel,
                                       "comparisons": {sel: r["comparisons"][sel]}})
    write_json("test_verdict.json", {"selected": sel} | v)
    write_json("event_taxonomy_test.json", r["taxonomy"])
    print(f"[r1] TEST verdict {v}", flush=True)


STAGES = {"audit": stage_audit, "manifest": stage_manifest, "train_soft": stage_train_soft, "train_joint": stage_train_joint,
          "evaluate_val": stage_evaluate_val, "compute": stage_compute, "figure_val": stage_figure_val, "freeze": stage_freeze,
          "evaluate_test": stage_evaluate_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(10) as ex:
        STAGES[args.stage](ex, dev)


if __name__ == "__main__":
    main()
