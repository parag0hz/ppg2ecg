"""C0 — CoherentBeat deterministic feasibility (docs/C0_COHERENTBEAT_PREREGISTRATION.md).

New prospective architecture split of the V1 VitalDB TRAIN patient pool (ARCH-TRAIN / ARCH-VAL / ARCH-HOLDOUT). Every
learned component is trained on ARCH-TRAIN only. Stage A qualifies on ARCH-VAL; ARCH-HOLDOUT can be loaded only after a
committed holdout-freeze manifest exists and the ARCH-VAL verdict is QUALIFIED.

Stages: split, audit, train_detector, train_bf0det, train_c0, train_localonly, evaluate_val, freeze, evaluate_holdout,
        compute, figure
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/c0_coherentbeat.py <stage>
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
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
from ppg2ecg.beatfirst import beats as BB  # noqa: E402
from ppg2ecg.beatfirst import model as BM  # noqa: E402
from ppg2ecg.beatfirst import render as BR  # noqa: E402
from ppg2ecg.coherentbeat import geometry as G  # noqa: E402
from ppg2ecg.coherentbeat import model as CM  # noqa: E402
from ppg2ecg.coherentbeat import split as SP  # noqa: E402
from ppg2ecg.evaluation import metrics as MET  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as RP  # noqa: E402
from ppg2ecg.evaluation.m1_structural import spectral_metrics  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, extract_events, soft_event_field  # noqa: E402

PREREG = "docs/C0_COHERENTBEAT_PREREGISTRATION.md"
CODE_FILES = ("scripts/c0_coherentbeat.py", "src/ppg2ecg/coherentbeat/__init__.py", "src/ppg2ecg/coherentbeat/split.py",
              "src/ppg2ecg/coherentbeat/geometry.py", "src/ppg2ecg/coherentbeat/model.py", "tests/test_c0_coherentbeat.py")
ART = ROOT / "artifacts/c0_coherentbeat"
OUT = ROOT / "outputs/c0_coherentbeat"
MANIFEST = ROOT / "data/manifests/split_v1_vitaldb_seed42.json"
DATA = ROOT / "data/processed/v1_vitaldb"
RD1_TRAIN_PEAKS = ROOT / "outputs/rd1_detector/train_rpeaks.npz"
FREEZE = ART / "holdout_freeze_manifest.json"
FS, T = 128, 512
SEED = 42
DET = {"steps": 14000, "batch": 64, "lr": 1e-3, "wd": 0.01, "sigma_ms": 20.0, "threshold": 0.35, "refractory": 32}
BEAT = {"steps": 20000, "batch": 256, "lr": 1e-3, "wd": 0.01, "clip": 1.0}          # BF0 deterministic beat protocol
C0T = {"steps": 20000, "batch": 64, "lr": 1e-3, "wd": 0.01, "clip": 1.0}            # window-level (RD1 batch convention)
BOOT_N, BOOT_SEED = 2000, 20261001
SALT_CHECK, SALT_EXAMPLE, SALT_LAT = "c0-peakcheck-v1", "c0-example-v1", "c0-latency-v1"
N_PEAK_CHECK, N_LAT = 200, 50
MARGIN = {"g1_f1": 0.02, "g1_rr_ms": 2.0, "g4_corr": 0.02}
BOUNDARY_NEAR = 2                                   # samples, frozen before ARCH-VAL results
ARMS = ("BF0", "LOCAL", "C0")                       # BF0-DET-RETRAIN, C0-LOCAL-ONLY, CoherentBeat-C0


# ----------------------------------------------------------------------------------------------- pure helpers
def cluster_ci(d, pid, n_rep: int = BOOT_N, seed: int = BOOT_SEED):
    """Equal-patient-weight mean of per-window values (nan skipped) and the patient-bootstrap 95% CI."""
    d = np.asarray(d, dtype=np.float64)
    pid = np.asarray(pid)
    subs = np.unique(pid)
    per = np.array([np.nanmean(d[pid == s]) if np.isfinite(d[pid == s]).any() else np.nan for s in subs])
    if not np.isfinite(per).any():
        return [float("nan")] * 3
    draws = np.array([np.nanmean(per[r]) for r in B.patient_resamples(len(subs), n_rep, seed)])
    return [float(np.nanmean(per)), float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]


def gates(d: dict) -> dict:
    """Prereg §8 qualification rules on [point, CI low, CI high] triples."""
    return {"G1": bool(d["dF1"][1] > -MARGIN["g1_f1"] and d["dRR"][2] < MARGIN["g1_rr_ms"]),
            "G2": bool(d["dFP"][2] < 0), "G3": bool(d["dFD"][2] < 0), "G4": bool(d["dCorr"][1] > -MARGIN["g4_corr"])}


def verdict(g: dict, stage: str) -> str:
    if g["G1"] and g["G2"] and g["G3"]:
        return ("QUALIFIED" if stage == "val" else "STRONG") if g["G4"] else "PARTIAL"
    return "FAILED"


def boundary_points(events, rr_edge):
    L, R = G.supports(events, rr_edge)
    ev = np.asarray(events, dtype=np.float64)
    return np.concatenate([ev - L, ev + R]) if ev.size else np.zeros(0)


def fp_detections(ref, det) -> list[int]:
    det = np.asarray(det, int)
    matched = {j for _, j in RP.match_rpeaks(np.asarray(ref, int), det, FS, 50.0)[0]}
    return [int(det[j]) for j in range(det.size) if j not in matched]


# ----------------------------------------------------------------------------------------------- data
def manifest():
    return json.loads(MANIFEST.read_text())


def split_info():
    return json.loads((ART / "split_manifest.json").read_text())


def check_freeze():
    """ARCH-HOLDOUT is sealed until a committed freeze manifest exists, matches every frozen file, and ARCH-VAL QUALIFIED."""
    if not FREEZE.exists():
        raise PermissionError("ARCH-HOLDOUT is sealed: no holdout-freeze manifest")
    fm = json.loads(FREEZE.read_text())
    if json.loads((ART / "qualification.json").read_text())["verdict"] != "QUALIFIED":
        raise PermissionError("ARCH-HOLDOUT is sealed: ARCH-VAL did not qualify")
    for f, h in fm["sha256"].items():
        if B.sha256_file(ROOT / f) != h:
            raise PermissionError(f"ARCH-HOLDOUT is sealed: frozen file changed: {f}")
    rel = str(FREEZE.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise PermissionError("ARCH-HOLDOUT is sealed: the freeze manifest is not committed")


def load_arch(role: str):
    """X float32 [N, 512], Y float64, patient id per window; cases in V1 manifest order. HOLDOUT needs the freeze."""
    if role not in ("train", "val", "holdout"):
        raise ValueError(role)
    if role == "holdout":
        check_freeze()
    cases = split_info()["roles"][role]["cases"]
    X, Y, P = [], [], []
    poc = manifest()["extra"]["patient_of_case"]
    for c in cases:
        d = np.load(DATA / f"{c}.npz")
        assert int(d["subjectid"]) == int(poc[c]), c
        X.append(d["x"].astype(np.float32)); Y.append(d["y"].astype(np.float64)); P.append(np.full(len(d["x"]), int(d["subjectid"])))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(P)


def reference_peaks(role: str, Y, ex):
    """The RD1 neurokit cache sliced to the role's windows, re-verified on 200 salted windows of the role."""
    if role == "holdout":
        check_freeze()
    offs = SP.window_offsets(manifest())
    d = np.load(RD1_TRAIN_PEAKS)
    allp = B.split_list(d["idx"], d["off"])
    peaks = []
    for c in split_info()["roles"][role]["cases"]:
        a, b = offs[c]
        peaks += [np.asarray(p, int) for p in allp[a:b]]
    assert len(peaks) == len(Y)
    chk = B.salted_rank(f"{SALT_CHECK}-{role}", range(len(Y)))[:N_PEAK_CHECK]
    fresh = list(ex.map(B._peaks, [Y[i] for i in chk]))
    for i, f in zip(chk, fresh):
        assert np.array_equal(peaks[i], f), f"reference peak cache differs at {role} window {i}"
    return peaks


def write_json(name, obj):
    ART.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(json.dumps(B.clean(obj), indent=1))


def train_rr_median():
    return json.loads((ART / "audit.json").read_text())["arch_train"]["median_reference_rr_samples"]


# ----------------------------------------------------------------------------------------------- split / audit
def stage_split(ex, dev):
    m = manifest()
    sp = SP.arch_split(m)
    old = SP.old_heldout_patients(m)
    roles = {r: set(sp[r]["patients"]) for r in sp}
    checks = {"train_val": len(roles["train"] & roles["val"]), "train_holdout": len(roles["train"] & roles["holdout"]),
              "val_holdout": len(roles["val"] & roles["holdout"]),
              "old_val_in_arch": len(set(old["val"]) & set().union(*roles.values())),
              "old_test_in_arch": len(set(old["test"]) & set().union(*roles.values()))}
    assert all(v == 0 for v in checks.values()), checks
    wpc = m["extra"]["windows_per_case"]
    info = {"seed": SP.SPLIT_SEED, "source": "V1 VitalDB TRAIN patient pool (data/manifests/split_v1_vitaldb_seed42.json)",
            "rule": "patients sorted numerically, default_rng(20261001).permutation; first 434 HOLDOUT, next 433 VAL, rest TRAIN",
            "roles": {r: {"n_patients": len(sp[r]["patients"]), "n_cases": len(sp[r]["cases"]),
                          "n_windows": int(sum(int(wpc[c]) for c in sp[r]["cases"])),
                          "patients": sp[r]["patients"], "cases": sp[r]["cases"]} for r in sp},
            "overlap_checks": checks}
    write_json("split_manifest.json", info)
    write_json("split_hashes.json", {r: {"patients_sha256": SP.ids_sha256(sp[r]["patients"]),
                                         "cases_sha256": hashlib.sha256(json.dumps(sp[r]["cases"]).encode()).hexdigest()}
                                     for r in sp} | {"overlap_checks": checks})
    print({r: (info["roles"][r]["n_patients"], info["roles"][r]["n_windows"]) for r in sp}, checks)


def stage_audit(ex, dev):
    out = {}
    for role in ("train", "val"):
        X, Y, P = load_arch(role)
        ref = reference_peaks(role, Y, ex)
        rr = np.concatenate([np.diff(p) for p in ref if len(p) >= 2])
        out[f"arch_{role}"] = {"windows": int(len(X)), "patients": int(len(np.unique(P))), "reference_peaks": int(sum(len(p) for p in ref)),
                               "windows_without_reference_peaks": int(sum(len(p) == 0 for p in ref)),
                               "median_reference_rr_samples": float(np.median(rr))}
    out["arch_holdout"] = {"windows_from_manifest": split_info()["roles"]["holdout"]["n_windows"], "loaded": False}
    out["test_or_old_val_loaded"] = False
    write_json("audit.json", out)
    write_json("input_hashes.json", {"v1_manifest": B.sha256_file(MANIFEST), "rd1_train_peaks_cache": B.sha256_file(RD1_TRAIN_PEAKS),
                                     "bf0_beatfirst_code": {f: B.sha256_file(ROOT / f) for f in B.CODE_FILES[:6]},
                                     "software": B.software()})
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------------------------- training
def _field(r):
    return soft_event_field(r, T, DET["sigma_ms"] / 1000.0 * FS)


def _epoch_batches(n, batch, g):
    perm, pos = torch.randperm(n, generator=g), 0
    while True:
        if pos + batch > len(perm):
            perm, pos = torch.randperm(n, generator=g), 0
        yield perm[pos:pos + batch]
        pos += batch


def _meta(name, net, secs, nan_steps, extra=None):
    f = OUT / f"{name}.pt"
    meta = {"steps_seed": SEED, "train_seconds": secs, "n_params": sum(p.numel() for p in net.parameters()),
            "nan_steps": nan_steps, "checkpoint": "last step (no selection)",
            "peak_gpu_mem_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} | (extra or {})
    torch.save({"state_dict": net.state_dict()} | meta, f)
    meta["sha256"] = B.sha256_file(f)
    write_json(f"checkpoint_{name}.json", meta)
    return meta


def stage_train_detector(ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, Y, _ = load_arch("train")
    ref = reference_peaks("train", Y, ex)
    Fd = torch.from_numpy(np.stack(list(ex.map(_field, ref, chunksize=512))).astype(np.float16)).to(dev)
    Xt = torch.from_numpy(X).to(dev)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = RhythmTCN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=DET["lr"], weight_decay=DET["wd"])
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    batches, t0, acc, nan_steps = _epoch_batches(len(Xt), DET["batch"], g), time.time(), [], 0
    net.train()
    for step in range(1, DET["steps"] + 1):
        b = next(batches).to(dev)
        loss = lossf(net(Xt[b][:, None])[:, 0], Fd[b].float())
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward(); opt.step(); acc.append(loss.item())
        if step % 1000 == 0:
            print(f"[c0] detector step {step} BCE {np.mean(acc):.5f}", flush=True); acc = []
    OUT.mkdir(parents=True, exist_ok=True)
    _meta("detector", net, time.time() - t0, nan_steps, {"protocol": DET, "family": "RD1 RhythmTCN (Global-TCN)"})


def stage_train_bf0det(ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, Y, _ = load_arch("train")
    ref = reference_peaks("train", Y, ex)
    E, P, RR, _ = BB.extract_training_beats(X, Y, ref)
    template = np.median(E, axis=0)
    rr_median = train_rr_median() / FS
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "bf0det_template.npz", template=template, rr_median=rr_median)
    torch.cuda.reset_peak_memory_stats()
    net, secs, nan_steps = B._train_beat_model("deterministic", E, P, RR, dev)     # BF0 protocol: 20,000 x 256, seed 42
    _meta("bf0det", net, secs, nan_steps, {"protocol": BEAT, "n_train_beats": int(len(E)), "family": "BF0 BeatFlowNet, deterministic L1"})


def _train_c0(name, use_global, ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, Y, _ = load_arch("train")
    ref = reference_peaks("train", Y, ex)
    rr_edge = train_rr_median()
    Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y.astype(np.float32)).to(dev)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = CM.CoherentBeat(use_global=use_global).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=C0T["lr"], weight_decay=C0T["wd"])
    g = torch.Generator().manual_seed(SEED)
    batches, t0, acc, nan_steps = _epoch_batches(len(Xt), C0T["batch"], g), time.time(), [], 0
    net.train()
    for step in range(1, C0T["steps"] + 1):
        b = next(batches)
        bd = b.to(dev)
        out = net(Xt[bd], [ref[int(i)] for i in b], rr_edge)          # training events: reference R
        loss = (out - Yt[bd]).abs().mean()
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), C0T["clip"])
        opt.step(); acc.append(loss.item())
        if step % 1000 == 0:
            print(f"[c0] {name} step {step} L1 {np.mean(acc):.5f}", flush=True); acc = []
    OUT.mkdir(parents=True, exist_ok=True)
    _meta(name, net, time.time() - t0, nan_steps, {"protocol": C0T, "use_global": use_global, "training_events": "reference R"})


def stage_train_c0(ex, dev):
    _train_c0("c0", True, ex, dev)


def stage_train_localonly(ex, dev):
    _train_c0("c0_localonly", False, ex, dev)


# ----------------------------------------------------------------------------------------------- inference
def load_net(name, dev):
    ck = torch.load(OUT / f"{name}.pt", map_location="cpu")
    if name == "detector":
        net = RhythmTCN()
    elif name == "bf0det":
        net = BM.BeatFlowNet()
    else:
        net = CM.CoherentBeat(use_global=(name == "c0"))
    net.load_state_dict(ck["state_dict"])
    return net.to(dev).eval()


@torch.no_grad()
def detect_events(X, dev, ex):
    det = load_net("detector", dev)
    out = []
    for i in range(0, len(X), 2048):
        out.append(torch.sigmoid(det(torch.from_numpy(X[i:i + 2048]).to(dev)[:, None])[:, 0]).float().cpu().numpy())
    field = np.concatenate(out)
    return [np.asarray(e, int) for e in ex.map(_extract, list(field), chunksize=256)]


def _extract(p):
    return extract_events(p, DET["threshold"], DET["refractory"])


@torch.no_grad()
def render_c0(name, X, events, dev, rr_edge, bs=256):
    net = load_net(name, dev)
    out, gs, ls = [], [], []
    for i in range(0, len(X), bs):
        y, g, loc = net(torch.from_numpy(X[i:i + bs]).to(dev), events[i:i + bs], rr_edge, return_parts=True)
        out.append(y.cpu().numpy()); gs.append(g.cpu().numpy()); ls.append(loc.cpu().numpy())
    return np.concatenate(out).astype(np.float64), np.concatenate(gs).astype(np.float64), np.concatenate(ls).astype(np.float64)


def render_bf0det(X, events, dev):
    tz = np.load(OUT / "bf0det_template.npz")
    template, rr_median = tz["template"], float(tz["rr_median"])
    fill = float(np.median(np.concatenate([template[:8], template[-8:]])))
    ppg, rr, _ = B.conditions(X, events, rr_median)
    beats = B.predict_deterministic(load_net("bf0det", dev), ppg, rr, dev)
    return B.assemble_all(beats, events, fill).astype(np.float64)


# ----------------------------------------------------------------------------------------------- evaluation
_FD: dict = {}


def _fd_init(a, b, Y):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    _FD["a"], _FD["b"], _FD["Y"] = a, b, Y


def _fd_draw(idx):
    return PMX.kanflow_fd(_FD["a"][idx], _FD["Y"][idx]) - PMX.kanflow_fd(_FD["b"][idx], _FD["Y"][idx])


def fd_diff_ci(a, b, Y, pid):
    jobs = B.patient_bootstrap_indices(pid, BOOT_N, BOOT_SEED)
    assert min(len(j) for j in jobs) >= PMX.FD_SMALL_SET
    with ProcessPoolExecutor(16, initializer=_fd_init, initargs=(a, b, Y)) as pool:
        draws = np.array(list(pool.map(_fd_draw, jobs, chunksize=8)))
    return [float(PMX.kanflow_fd(a, Y) - PMX.kanflow_fd(b, Y)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def evaluate(role, ex, dev):
    X, Y, Pid = load_arch(role)
    N = len(Y)
    ref = reference_peaks(role, Y, ex)
    evaluable = np.array([len(r) > 0 for r in ref])
    rr_edge = train_rr_median()
    events = detect_events(X, dev, ex)                                # ONE event sequence for every arm
    waves = {"BF0": render_bf0det(X, events, dev)}
    waves["LOCAL"], _, loc_lo = render_c0("c0_localonly", X, events, dev, rr_edge)
    waves["C0"], g_c0, loc_c0 = render_c0("c0", X, events, dev, rr_edge)
    waves["GLOBAL_ONLY"] = g_c0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{role}_renders.npz", **{k: v.astype(np.float32) for k, v in waves.items()},
             events_idx=B.pack_list(events)[0], events_off=B.pack_list(events)[1])
    det = {k: list(ex.map(B._peaks, list(w), chunksize=256)) for k, w in waves.items()}
    per = {}
    for k, w in waves.items():
        prf = PMX.rpeak_prf_at(w, Y, FS, 50.0, peaks=(ref, det[k]))
        bl = PMX.beat_level_metrics(w, Y, FS, 50.0, peaks=(ref, det[k]))
        sm = MET.signal_metrics(w, Y)
        st = np.array(list(ex.map(B._structure, [(w[i], Y[i], ref[i]) for i in range(N)], chunksize=256)))
        spec = np.array([np.mean([v for kk, v in spectral_metrics(w[i], Y[i]).items() if kk.endswith("__ratio_dev")]) for i in range(N)])
        per[k] = {"f1": np.where(evaluable, prf["rpeak_f1"], np.nan), "precision": prf["rpeak_precision"], "recall": prf["rpeak_recall"],
                  "fp": prf["n_fp"], "fn": prf["n_fn"], "rr_mae_ms": bl["rr_mae_ms"], "hr_abs_err": bl["hr_abs_err"],
                  "mae": sm["mae"], "pcc": sm["pcc"], "s4": st[:, 0], "s5": st[:, 1], "spectral_ratio_dev": spec}
    pprf = PMX.rpeak_prf_at(Y, Y, FS, 50.0, peaks=(ref, events))
    pbl = PMX.beat_level_metrics(Y, Y, FS, 50.0, peaks=(ref, events))
    per["PLACED"] = {"f1": np.where(evaluable, pprf["rpeak_f1"], np.nan), "precision": pprf["rpeak_precision"],
                     "recall": pprf["rpeak_recall"], "fp": pprf["n_fp"], "fn": pprf["n_fn"], "rr_mae_ms": pbl["rr_mae_ms"],
                     "hr_abs_err": pbl["hr_abs_err"]}
    pairs = [BR.matched_pairs(ref[i], events[i], T) for i in range(N)]
    corr = {k: [BR.pair_correlations(Y[i], waves[k][i], pairs[i]) for i in range(N)] for k in ARMS}
    wm = B.window_pair_means(corr, ARMS)
    for k in ARMS:
        per[k]["beat_corr"] = wm[k]
    summary = {k: {m: cluster_ci(v, Pid) for m, v in d.items()} for k, d in per.items()}
    fd = {k: float(PMX.kanflow_fd(w, Y)) for k, w in waves.items()}
    for k in fd:
        summary[k]["fd"] = fd[k]
    detail = {"dF1": cluster_ci(per["C0"]["f1"] - per["PLACED"]["f1"], Pid),
              "dRR": cluster_ci(per["C0"]["rr_mae_ms"] - per["PLACED"]["rr_mae_ms"], Pid),
              "dFP": cluster_ci(per["C0"]["fp"] - per["BF0"]["fp"], Pid),
              "dFD": fd_diff_ci(waves["C0"], waves["BF0"], Y, Pid),
              "dCorr": cluster_ci(per["C0"]["beat_corr"] - per["BF0"]["beat_corr"], Pid)}
    local_only = {"dFP_vs_BF0": cluster_ci(per["LOCAL"]["fp"] - per["BF0"]["fp"], Pid),
                  "dF1_vs_placed": cluster_ci(per["LOCAL"]["f1"] - per["PLACED"]["f1"], Pid),
                  "dFD_vs_BF0": fd_diff_ci(waves["LOCAL"], waves["BF0"], Y, Pid),
                  "dCorr_vs_BF0": cluster_ci(per["LOCAL"]["beat_corr"] - per["BF0"]["beat_corr"], Pid),
                  "C0_minus_LOCAL_fp": cluster_ci(per["C0"]["fp"] - per["LOCAL"]["fp"], Pid)}
    g = gates(detail)
    v = verdict(g, role)
    # ---- coherence: boundary-near FPs, boundary jumps, decomposition
    coh = {}
    for k, loc in (("C0", loc_c0), ("LOCAL", loc_lo)):
        near, tot, vj, dj, rj, rv, rdj = 0, 0, [], [], [], [], []
        for i in range(N):
            bpts = boundary_points(events[i], rr_edge)
            fps = fp_detections(ref[i], det[k][i])
            tot += len(fps)
            near += sum(1 for d in fps if bpts.size and np.min(np.abs(bpts - d)) <= BOUNDARY_NEAR)
            w, y = waves[k][i], Y[i]
            for b in bpts:
                k0 = int(np.floor(b))
                if 1 <= k0 and k0 + 2 < T:
                    vj.append(abs(w[k0 + 1] - w[k0])); dj.append(abs((w[k0 + 2] - w[k0 + 1]) - (w[k0] - w[k0 - 1])))
                    rv.append(abs(y[k0 + 1] - y[k0])); rdj.append(abs((y[k0 + 2] - y[k0 + 1]) - (y[k0] - y[k0 - 1])))
                    rj.append(max(abs(loc[i][k0]), abs(loc[i][k0 + 1])))
        q = lambda a: np.percentile(a, [50, 95, 99]).tolist() if len(a) else None  # noqa: E731
        coh[k] = {"fp_total": tot, "boundary_near_fp": near, "boundary_near_fraction": near / max(tot, 1),
                  "local_residual_at_boundary_q50_95_99": q(rj), "value_jump_q50_95_99": q(vj),
                  "first_difference_jump_q50_95_99": q(dj), "reference_ecg_value_jump_q50_95_99": q(rv),
                  "reference_ecg_first_difference_jump_q50_95_99": q(rdj)}
    qrs = np.zeros((N, T), bool)
    for i in range(N):
        for r in events[i]:
            qrs[i, max(0, r - 10):min(T, r + 11)] = True
    gac = g_c0 - g_c0.mean(axis=1, keepdims=True)
    e_l, e_g = loc_c0 ** 2, gac ** 2
    decomp = {"global_rms": float(np.sqrt(np.mean(g_c0 ** 2))), "global_ac_rms": float(np.sqrt(np.mean(gac ** 2))),
              "local_rms": float(np.sqrt(np.mean(loc_c0 ** 2))),
              "qrs_energy_share_local": float(e_l[qrs].sum() / (e_l[qrs].sum() + e_g[qrs].sum())),
              "non_qrs_energy_share_global": float(e_g[~qrs].sum() / (e_l[~qrs].sum() + e_g[~qrs].sum())),
              "definition": "QRS region = +-10 samples of placed events; global energy uses g minus its window mean"}
    miss = {"windows": N, "windows_without_events": int(sum(len(e) == 0 for e in events)),
            "windows_without_reference_peaks": int((~evaluable).sum()), "events": int(sum(len(e) for e in events)),
            "matched_pairs_for_corr": wm["_n_pairs"],
            "nan_windows": {k: {m: int(np.sum(~np.isfinite(np.asarray(v, float)))) for m, v in d.items()} for k, d in per.items()}}
    raw = {k: {m: int(np.nansum(per[k][m])) for m in ("fp", "fn")} for k in per}
    return {"role": role, "summary": summary, "detail": detail, "gates": g, "verdict": v, "local_only": local_only,
            "coherence": coh, "decomposition": decomp, "missingness": miss, "raw": raw,
            "detector_alone": summary["PLACED"]}


def stage_evaluate_val(ex, dev):
    r = evaluate("val", ex, dev)
    write_json("val_metrics.json", {k: r[k] for k in ("summary", "local_only", "missingness", "raw", "detector_alone")})
    write_json("val_bootstrap.json", {"unit": "patient", "replicates": BOOT_N, "seed": BOOT_SEED, "detail": r["detail"],
                                      "local_only": r["local_only"]})
    write_json("qualification.json", {"stage": "ARCH-VAL", "gates": r["gates"], "detail": r["detail"], "margins": MARGIN,
                                      "verdict": r["verdict"]})
    write_json("boundary_metrics.json", r["coherence"])
    write_json("decomposition_metrics.json", r["decomposition"])
    print(json.dumps(B.clean({"gates": r["gates"], "verdict": r["verdict"], "detail": r["detail"]}), indent=1))


def stage_freeze(ex, dev):
    q = json.loads((ART / "qualification.json").read_text())
    if q["verdict"] != "QUALIFIED":
        raise SystemExit(f"ARCH-VAL verdict {q['verdict']}: no holdout freeze")
    files = [PREREG, *CODE_FILES] + [B.rel(OUT / f"{n}.pt") for n in ("detector", "bf0det", "c0", "c0_localonly")]
    files += [B.rel(OUT / "bf0det_template.npz")] + [B.rel(ART / f) for f in ("split_manifest.json", "audit.json", "qualification.json")]
    files += ["src/ppg2ecg/evaluation/paper_metrics.py", "src/ppg2ecg/evaluation/rpeaks.py", "src/ppg2ecg/beatfirst/render.py",
              "scripts/bf0_run.py"]
    write_json("holdout_freeze_manifest.json", {"sha256": {f: B.sha256_file(ROOT / f) for f in files},
                                                "frozen_before_holdout_access": True, "arch_val_verdict": q["verdict"]})


def stage_evaluate_holdout(ex, dev):
    check_freeze()
    r = evaluate("holdout", ex, dev)
    write_json("holdout_metrics.json", {k: r[k] for k in ("summary", "local_only", "missingness", "raw", "coherence", "decomposition")})
    write_json("holdout_bootstrap.json", {"unit": "patient", "replicates": BOOT_N, "seed": BOOT_SEED, "detail": r["detail"],
                                          "gates": r["gates"], "verdict": r["verdict"], "local_only": r["local_only"]})
    print(json.dumps(B.clean({"gates": r["gates"], "verdict": r["verdict"], "detail": r["detail"]}), indent=1))


@torch.no_grad()
def stage_compute(ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, _, _ = load_arch("val")
    rr_edge = train_rr_median()
    pick = B.salted_rank(SALT_LAT, range(len(X)))[:N_LAT]
    torch.set_num_threads(4)
    lat = {}
    for dname in ("cpu", "cuda"):
        d = torch.device(dname)
        det = load_net("detector", d)
        nets = {"c0": load_net("c0", d), "bf0det": load_net("bf0det", d)}
        tz = np.load(OUT / "bf0det_template.npz")
        for kind in ("c0", "bf0det"):
            ts = []
            for rep, i in enumerate(list(pick[:3]) + list(pick)):
                if dname == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                p = torch.sigmoid(det(torch.from_numpy(X[i:i + 1]).to(d)[:, None])[:, 0]).float().cpu().numpy()[0]
                ev = _extract(p)
                if kind == "c0":
                    nets["c0"](torch.from_numpy(X[i:i + 1]).to(d), [ev], rr_edge)
                elif len(ev):
                    ppg, rr, _ = B.conditions(X[i:i + 1], [ev], float(tz["rr_median"]))
                    beats = B.predict_deterministic(nets["bf0det"], ppg, rr, d)
                    BR.assemble(beats, ev, T, 0.0)
                if dname == "cuda":
                    torch.cuda.synchronize()
                if rep >= 3:
                    ts.append((time.perf_counter() - t0) * 1000)
            lat[f"{dname}_{kind}"] = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}
    ck = {n: json.loads((ART / f"checkpoint_{n}.json").read_text()) for n in ("detector", "bf0det", "c0", "c0_localonly")}
    write_json("compute_accounting.json", {
        "params": {n: ck[n]["n_params"] for n in ck}, "c0_over_bf0det_params": ck["c0"]["n_params"] / ck["bf0det"]["n_params"],
        "train_seconds": {n: ck[n]["train_seconds"] for n in ck}, "peak_gpu_mem_mib": {n: ck[n]["peak_gpu_mem_mib"] for n in ck},
        "nan_steps": {n: ck[n]["nan_steps"] for n in ck}, "batch1_latency_ms": lat,
        "latency_protocol": f"{N_LAT} salted ARCH-VAL windows, 3 warm-up, CPU 4 threads; detector + waveform model",
        "software": B.software()})
    write_json("checkpoint_hashes.json", {n: ck[n]["sha256"] for n in ck})


def stage_figure(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    holdout = (ART / "holdout_metrics.json").exists()
    role = "holdout" if holdout else "val"
    if holdout:
        check_freeze()
        hm, hb = json.loads((ART / "holdout_metrics.json").read_text()), json.loads((ART / "holdout_bootstrap.json").read_text())
        S, Dt, gt, vd, coh = hm["summary"], hb["detail"], hb["gates"], hb["verdict"], hm["coherence"]
    else:
        vm, q = json.loads((ART / "val_metrics.json").read_text()), json.loads((ART / "qualification.json").read_text())
        S, Dt, gt, vd = vm["summary"], q["detail"], q["gates"], q["verdict"]
        coh = json.loads((ART / "boundary_metrics.json").read_text())
    X, Y, _ = load_arch(role)
    rd = np.load(OUT / f"{role}_renders.npz")
    events = B.split_list(rd["events_idx"], rd["events_off"])
    C = {"BF0": "#4A3AA7", "LOCAL": "#B0621B", "C0": "#0E7A86", "PLACED": "#8A94A3", "ref": "0.55"}
    pf = lambda k: "PASS" if gt[k] else "FAIL"  # noqa: E731
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    a = fig.add_subplot(gs[0, 0]); a.axis("off")
    a.text(0.02, 0.92, "A  BF0 / D0 failure", fontsize=10, weight="bold")
    for k, s in enumerate(["independent absolute-level beats", "long support: neighbour-QRS ghosts at window edges (BF0)",
                           "short cells: level steps between beats (D0)", "-> false R detections"]):
        a.text(0.04, 0.78 - 0.13 * k, s, fontsize=9)
    a.text(0.02, 0.22, "B  CoherentBeat-C0", fontsize=10, weight="bold")
    a.text(0.04, 0.08, "PPG -> shared encoder -> global spline g(t) (17 coef)\n"
                       "                    + sum_i bump_i(t) q_i(t - r_i)  (compact, C-inf)", fontsize=8.5, family="monospace")
    c = fig.add_subplot(gs[0, 1:])
    pick = [i for i in B.salted_rank(SALT_EXAMPLE, range(len(Y))) if len(events[i]) >= 3][:2]
    t = np.arange(T) / FS
    off = 0.0
    for i in pick:
        c.plot(t, Y[i] + off, color=C["ref"], lw=1.4)
        for k, lw in (("BF0", 0.8), ("LOCAL", 0.8), ("C0", 1.0)):
            c.plot(t, rd[k][i] + off, color=C[k], lw=lw, label=k if off == 0 else None)
        c.plot(np.asarray(events[i]) / FS, np.full(len(events[i]), 1.15 + off), "v", color="k", ms=5)
        off -= 2.6
    c.legend(fontsize=8, ncol=3, loc="upper right")
    c.set_yticks([])
    c.set_title(f"C  first salted {role.upper()} windows with >= 3 events (not curated); grey = reference, ▼ placed events", loc="left", fontsize=9)
    d_ax = fig.add_subplot(gs[1, 0])
    ks = ["PLACED", "BF0", "LOCAL", "C0"]
    for j, (m, lab) in enumerate((("f1", "R-peak F1"), ("fp", "FP / window"))):
        vals = [S[k][m] for k in ks]
        xs = np.arange(4) + j * 5
        d_ax.bar(xs, [v[0] for v in vals], color=[C[k] for k in ks], width=0.7)
        for x_, v_ in zip(xs, vals):
            d_ax.errorbar(x_, v_[0], yerr=[[v_[0] - v_[1]], [v_[2] - v_[0]]], color="k", capsize=3, lw=1)
        d_ax.text(xs.mean(), max(v_[2] for v_ in vals) * 1.04, lab, fontsize=8, ha="center")
    d_ax.set_xticks(list(range(4)) + list(range(5, 9)), ks * 2, fontsize=7, rotation=30)
    d_ax.set_title(f"D  events: G1 {pf('G1')}, G2 {pf('G2')}", loc="left", fontsize=9)
    e_ax = fig.add_subplot(gs[1, 1])
    ks3 = ["BF0", "LOCAL", "C0"]
    e_ax.bar(range(3), [S[k]["fd"] for k in ks3], color=[C[k] for k in ks3], width=0.6)
    e_ax.set_xticks(range(3), ks3, fontsize=8)
    e_ax.set_ylabel("FD (lower = better)")
    e_ax.set_title(f"E  FD: G3 {pf('G3')} (C0-BF0 {Dt['dFD'][0]:+.2f} [{Dt['dFD'][1]:+.2f}, {Dt['dFD'][2]:+.2f}])\n"
                   f"beat corr: G4 {pf('G4')} (C0-BF0 {Dt['dCorr'][0]:+.3f} [{Dt['dCorr'][1]:+.3f}, {Dt['dCorr'][2]:+.3f}])",
                   loc="left", fontsize=9)
    f_ax = fig.add_subplot(gs[1, 2])
    f_ax.bar(range(2), [coh[k]["boundary_near_fp"] for k in ("LOCAL", "C0")], color=[C["LOCAL"], C["C0"]], width=0.6)
    f_ax.set_xticks(range(2), ["LOCAL-ONLY", "C0"], fontsize=8)
    f_ax.set_ylabel(f"false detections within ±{BOUNDARY_NEAR} samples of a support boundary")
    f_ax.set_title("F  boundary-near false detections", loc="left", fontsize=9)
    t_ax = fig.add_subplot(gs[1, 3]); t_ax.axis("off")
    rows = [["", "BF0-DET", "LOCAL", "C0"]] + [[lab] + [f"{S[k][m][0]:.3f}" for k in ks3] for m, lab in
                                                (("f1", "F1"), ("fp", "FP/win"), ("rr_mae_ms", "RR-MAE"), ("beat_corr", "beat corr"),
                                                 ("mae", "MAE"), ("pcc", "PCC"))] + [["FD"] + [f"{S[k]['fd']:.2f}" for k in ks3]]
    tb = t_ax.table(cellText=rows, loc="center", cellLoc="center")
    tb.scale(1, 1.5)
    t_ax.set_title(f"placed-event F1 {S['PLACED']['f1'][0]:.3f}\nΔF1 C0 − placed {Dt['dF1'][0]:+.4f} "
                   f"[{Dt['dF1'][1]:+.4f}, {Dt['dF1'][2]:+.4f}]", loc="left", fontsize=9)
    label = "ARCH-HOLDOUT (prospective)" if holdout else "ARCH-VAL (development / qualification only)"
    fig.suptitle(f"C0 CoherentBeat deterministic — {label} — {vd}", fontsize=11)
    fig.tight_layout()
    fig.savefig(ART / "figure.png", dpi=150)


def stage_manifest(ex, dev):
    write_json("model_config.json", {
        "encoder": {"channels": CM.CH, "kernel": CM.KERNEL, "dilations": list(CM.ENC_DILATIONS), "stem": "1x1 conv 1->64"},
        "global": {"basis": "centred cardinal cubic B-spline", "n_coef": G.N_COEF, "knot_spacing_samples": G.KNOT_SPACING,
                   "control_points": "t = 0, 32, ..., 512", "coef_head": "avg over +-16 samples -> 1x1 conv -> GELU -> 1x1 conv"},
        "local": {"tau_grid": [G.TAU_LO, G.TAU_HI], "L_max_samples": G.L_MAX, "R_max_samples": G.R_MAX, "rr_fraction": G.RR_FRAC,
                  "film_dilations": list(CM.LOCAL_DILATIONS), "cond": "MLP([window encoder mean, RR_prev s, RR_next s]) -> 128",
                  "envelope": "b(u) = exp(1 - 1/(1 - u^2)), u = -tau/L (tau<0), tau/R (tau>=0)"},
        "rr_edge_samples": train_rr_median(), "params": {"c0": CM.n_params(CM.CoherentBeat(True)),
                                                         "c0_localonly": CM.n_params(CM.CoherentBeat(False)),
                                                         "bf0det": BM.n_params(BM.BeatFlowNet())}})
    write_json("timing_detector_config.json", DET | {"family": "RD1 RhythmTCN", "seed": SEED, "train_role": "ARCH-TRAIN",
                                                     "targets": "Gaussian at reference R", "loss": "BCE"})
    write_json("training_manifest.json", {"seed": SEED, "train_role": "ARCH-TRAIN", "c0": C0T | {"loss": "L1 full window",
                                          "training_events": "reference R"}, "c0_localonly": C0T | {"global": "g = 0"},
                                          "bf0det": BEAT | {"family": "BeatFlowNet deterministic", "training_events": "reference R"},
                                          "detector": DET, "checkpoint": "last step", "early_stopping": "none"})
    write_json("prereg_manifest.json", {
        "prereg": PREREG, "sha256": {f: B.sha256_file(ROOT / f) for f in (PREREG, *CODE_FILES)},
        "split_files": {f: B.sha256_file(ART / f) for f in ("split_manifest.json", "split_hashes.json", "audit.json",
                                                           "model_config.json", "timing_detector_config.json",
                                                           "training_manifest.json")},
        "parent_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "written_before_any_c0_training_or_arch_val_metric": True})


STAGES = {"split": stage_split, "manifest": stage_manifest, "audit": stage_audit, "train_detector": stage_train_detector,
          "train_bf0det": stage_train_bf0det, "train_c0": stage_train_c0, "train_localonly": stage_train_localonly,
          "evaluate_val": stage_evaluate_val, "freeze": stage_freeze, "evaluate_holdout": stage_evaluate_holdout,
          "compute": stage_compute, "figure": stage_figure}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(10) as ex:
        STAGES[args.stage](ex, dev)


if __name__ == "__main__":
    main()
