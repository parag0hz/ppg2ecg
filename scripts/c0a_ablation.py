"""C0-A — CoherentBeat architecture-attribution ablation pack (docs/C0A_COHERENTBEAT_ABLATION_PREREGISTRATION.md).

Frozen from C0 (never modified): split, timing detector, placed events, BF0-DET-RETRAIN, C0-LOCAL-ONLY, CoherentBeat-C0,
metric and bootstrap code. New, trained once on ARCH-TRAIN: PM-BF0-DET, CONST-GLOBAL-LOCAL, WW-DET.
ARCH-VAL first; the previously opened ARCH-HOLDOUT only after a committed C0-A freeze (frozen secondary replication).

Stages: params, audit, manifest, train_pm, train_const, train_ww, evaluate_val, freeze, evaluate_holdout, compute, figure
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/c0a_ablation.py <stage>
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bf0_run as B  # noqa: E402
import c0_coherentbeat as C0  # noqa: E402
from ppg2ecg.beatfirst import beats as BB  # noqa: E402
from ppg2ecg.beatfirst import model as BM  # noqa: E402
from ppg2ecg.beatfirst import render as BR  # noqa: E402
from ppg2ecg.coherentbeat import ablation as AB  # noqa: E402
from ppg2ecg.evaluation import metrics as MET  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation.m1_structural import spectral_metrics  # noqa: E402

PREREG = "docs/C0A_COHERENTBEAT_ABLATION_PREREGISTRATION.md"
CODE_FILES = ("scripts/c0a_ablation.py", "src/ppg2ecg/coherentbeat/ablation.py", "tests/test_c0a_ablation.py")
ART = ROOT / "artifacts/c0a_coherentbeat_ablation"
OUT = ROOT / "outputs/c0a_coherentbeat_ablation"
FREEZE = ART / "holdout_ablation_freeze.json"
FS, T = 128, 512
SEED = C0.SEED
MARGIN = {"f1": 0.02, "rr_ms": 2.0, "corr": 0.02}
WAVE_ARMS = ("BF0", "PM", "LOCAL", "CONST", "WW", "C0")
LABEL = {"PLACED": "Placed events", "BF0": "BF0-DET", "PM": "PM-BF0-DET", "LOCAL": "LOCAL-ONLY", "CONST": "CONST-GLOBAL-LOCAL",
         "WW": "WW-DET", "C0": "CoherentBeat-C0"}
REPRO_KEYS = ("f1", "precision", "recall", "fp", "fn", "rr_mae_ms", "hr_abs_err", "mae", "pcc", "s4", "s5", "spectral_ratio_dev")


# ----------------------------------------------------------------------------------------------- claim rules (pure)
def claim_m1(d: dict) -> str:
    """C0 vs PM-BF0-DET: FP and FD lower (CI < 0), morphology non-inferior (-0.02), C0 inside the placed-event rhythm margins."""
    ok = (d["fp"][2] < 0 and d["fd"][2] < 0 and d["corr"][1] > -MARGIN["corr"]
          and d["rhythm_f1"][1] > -MARGIN["f1"] and d["rhythm_rr"][2] < MARGIN["rr_ms"])
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def claim_m2(d: dict) -> str:
    """C0 vs CONST: FD lower (CI < 0) and no meaningful event / morphology disadvantage (F1 or correlation lower CI <= -0.02)."""
    disadvantage = d["f1"][1] <= -MARGIN["f1"] or d["corr"][1] <= -MARGIN["corr"]
    return "SUPPORTED" if (d["fd"][2] < 0 and not disadvantage) else "NOT SUPPORTED"


def claim_m3(d: dict) -> str:
    """C0 vs WW-DET: STRONG if FP and FD both lower (CI < 0) with morphology non-inferior; PARTIAL if one is lower and the
    other not significantly worse, morphology non-inferior; otherwise NOT SUPPORTED."""
    corr_ok = d["corr"][1] > -MARGIN["corr"]
    fp_b, fd_b = d["fp"][2] < 0, d["fd"][2] < 0
    fp_w, fd_w = d["fp"][1] > 0, d["fd"][1] > 0
    if fp_b and fd_b and corr_ok:
        return "STRONG"
    if corr_ok and ((fp_b and not fd_w) or (fd_b and not fp_w)):
        return "PARTIAL"
    return "NOT SUPPORTED"


def claim_carrier(d: dict) -> str:
    """CONST vs LOCAL-ONLY: a shared absolute context carrier is supported if CONST has lower FD and lower FP (CI < 0)."""
    return "SUPPORTED" if (d["fd"][2] < 0 and d["fp"][2] < 0) else "NOT SUPPORTED"


PRIMARY_EFFECTS = {"M1": ("fp", "fd"), "M2": ("fd",), "M3": ("fp", "fd"), "CARRIER": ("fd", "fp")}


def replication(val_dec: str, hold_dec: str, val_d: dict, hold_d: dict, keys) -> str:
    same = all(np.sign(val_d[k][0]) == np.sign(hold_d[k][0]) for k in keys)
    if not same:
        return "NOT REPLICATED"
    return "REPLICATED" if val_dec == hold_dec else "DIRECTIONALLY CONSISTENT"


def final_matrix(val: dict, rep: dict) -> dict:
    ok = lambda k: rep[k] != "NOT REPLICATED"  # noqa: E731
    m3 = {"STRONG": "SUPPORTED", "PARTIAL": "PARTIAL", "NOT SUPPORTED": "NOT SUPPORTED"}[val["M3"]]
    return {"capacity_only_explanation": "DISFAVORED" if (val["M1"] == "SUPPORTED" and ok("M1")) else "REMAINS PLAUSIBLE",
            "shared_absolute_context_carrier": "SUPPORTED" if (val["CARRIER"] == "SUPPORTED" and ok("CARRIER")) else "NOT SUPPORTED",
            "time_varying_global_field": "SUPPORTED" if (val["M2"] == "SUPPORTED" and ok("M2")) else "NOT SUPPORTED",
            "explicit_global_local_decomposition": m3 if ok("M3") else "NOT SUPPORTED"}


def c1_decision(val: dict, rep: dict) -> str:
    go = (val["M1"] == "SUPPORTED" and val["M3"] in ("STRONG", "PARTIAL")
          and rep["M1"] != "NOT REPLICATED" and rep["M3"] != "NOT REPLICATED")
    return "GO (experimental design only; C1 is not trained here)" if go else "NO-GO"


# ----------------------------------------------------------------------------------------------- io / guards
def write_json(name, obj):
    ART.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(json.dumps(B.clean(obj), indent=1))


def check_c0a_freeze():
    """The previously opened ARCH-HOLDOUT is used by C0-A only after a committed C0-A freeze that still matches."""
    if not FREEZE.exists():
        raise PermissionError("C0-A: ARCH-HOLDOUT sealed until holdout_ablation_freeze.json exists")
    if not (ART / "claim_matrix_val.json").exists():
        raise PermissionError("C0-A: ARCH-VAL claim matrix missing")
    for f, h in json.loads(FREEZE.read_text())["sha256"].items():
        if B.sha256_file(ROOT / f) != h:
            raise PermissionError(f"C0-A: frozen file changed: {f}")
    rel = str(FREEZE.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise PermissionError("C0-A: the freeze manifest is not committed")


def load_role(role):
    if role == "holdout":
        check_c0a_freeze()
    return C0.load_arch(role)


def configs():
    return json.loads((ART / "model_configs.json").read_text())


def build(name):
    cfg = configs()
    if name == "pm_bf0":
        return BM.BeatFlowNet(ch=cfg["PM"]["ch"])
    if name == "const":
        return AB.ConstGlobalLocal()
    if name == "ww":
        return AB.WWDet(cfg["WW"]["dec_ch"], cfg["WW"]["n_dec"])
    raise ValueError(name)


def load_new(name, dev):
    net = build(name)
    net.load_state_dict(torch.load(OUT / f"{name}.pt", map_location="cpu")["state_dict"])
    return net.to(dev).eval()


def save_ckpt(name, net, secs, nan_steps, extra):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"seed": SEED, "train_seconds": secs, "n_params": BM.n_params(net), "nan_steps": nan_steps,
            "checkpoint": "last step (no selection)", "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_mem_mib": torch.cuda.max_memory_allocated() / 2 ** 20} | extra
    torch.save({"state_dict": net.state_dict()} | meta, OUT / f"{name}.pt")
    meta["sha256"] = B.sha256_file(OUT / f"{name}.pt")
    write_json(f"checkpoint_{name}.json", meta)


# ----------------------------------------------------------------------------------------------- params / audit / manifest
def stage_params(ex, dev):
    pm_key, pm_n, pm_mis = AB.select_closest(AB.pm_bf0_candidates())
    ww_key, ww_n, ww_mis = AB.select_closest(AB.ww_candidates())
    const_n = BM.n_params(AB.ConstGlobalLocal())
    write_json("parameter_match.json", {
        "target_c0_params": AB.C0_PARAMS, "criterion": "parameter count only; closest to the target, ties to the first",
        "PM": {"grid": "BeatFlowNet channel width 32..128 (all other BF0 settings unchanged)", "selected_ch": pm_key,
               "params": pm_n, "mismatch": pm_mis, "within_2pct": abs(pm_mis) <= 0.02,
               "neighbours": {c: n for c, n in AB.pm_bf0_candidates().items() if abs(c - pm_key) <= 2}},
        "WW": {"grid": "decoder width {32..96 step 8} x depth 1..12 on the C0 encoder", "selected": {"dec_ch": ww_key[0], "n_dec": ww_key[1]},
               "params": ww_n, "mismatch": ww_mis, "within_5pct": abs(ww_mis) <= 0.05},
        "CONST": {"params": const_n, "mismatch": (const_n - AB.C0_PARAMS) / AB.C0_PARAMS,
                  "note": "the scalar head has the same layers as C0's spline-coefficient head (1x1 conv == linear on one position)"}})
    write_json("model_configs.json", {
        "PM": {"family": "BF0 BeatFlowNet deterministic (x_t = 0, t = 0)", "ch": pm_key, "kernel": BM.KERNEL,
               "dilations": list(BM.DILATIONS), "t_dim": BM.T_DIM, "cond_hidden": BM.COND_HIDDEN, "segment": [-64, 101],
               "render": "BF0 Hann overlap-add + renormalization, BF0 conditioning, ARCH-TRAIN template baseline"},
        "CONST": {"family": "CoherentBeat-C0 with g(t) = a_global", "global": "Linear(64,64) -> GELU -> Linear(64,1) on the window encoder mean",
                  "local": "identical to C0 (encoder, FiLM local residual, supports, C-infinity bump)"},
        "WW": {"family": "event-conditioned whole-window predictor", "encoder": "C0 Encoder (64 ch, 8 blocks)",
               "dec_ch": ww_key[0], "n_dec": ww_key[1], "dec_dilations": list(AB.WW_DEC_DILATIONS[:ww_key[1]]),
               "raster": "max of Gaussians, sigma = 20 ms (RD1 / C0 detector target convention), at the window's events",
               "output": "512 samples directly; no decomposition, supports or stitching"}})


def stage_audit(ex, dev):
    c0h = json.loads((C0.ART / "checkpoint_hashes.json").read_text())
    now = {n: B.sha256_file(C0.OUT / f"{n}.pt") for n in ("detector", "bf0det", "c0", "c0_localonly")}
    split_ok = json.loads((C0.ART / "split_hashes.json").read_text()) == {
        r: {"patients_sha256": C0.SP.ids_sha256(C0.split_info()["roles"][r]["patients"]),
            "cases_sha256": C0.hashlib.sha256(json.dumps(C0.split_info()["roles"][r]["cases"]).encode()).hexdigest()}
        for r in ("holdout", "val", "train")} | {"overlap_checks": C0.split_info()["overlap_checks"]}
    X, _, _ = C0.load_arch("val")
    st = np.load(C0.OUT / "val_renders.npz")
    stored = B.split_list(st["events_idx"], st["events_off"])
    fresh = C0.detect_events(X, dev, ex)
    ev_ok = len(stored) == len(fresh) and all(np.array_equal(a, b) for a, b in zip(stored, fresh))
    res = {"c0_checkpoints_unchanged": now == c0h, "checkpoint_sha256": now, "c0_split_hashes_reproduced": split_ok,
           "val_events_reproduced_from_frozen_detector": ev_ok, "val_windows": int(len(X)),
           "c0_freeze_manifest_commit": "dddaa4b", "holdout_loaded": False}
    assert res["c0_checkpoints_unchanged"] and split_ok and ev_ok, res
    write_json("audit.json", res)
    write_json("train_manifest.json", {
        "role": "ARCH-TRAIN only", "seed": SEED, "checkpoint": "last step",
        "PM": {"protocol": "BF0 deterministic beat: 20,000 steps x 256 beats, AdamW 1e-3 / 0.01, clip 1.0",
               "training_events": "reference R"},
        "CONST": {"protocol": C0.C0T, "loss": "L1 full window", "training_events": "reference R"},
        "WW": {"protocol": C0.C0T, "loss": "L1 full window", "training_raster": "reference R"}})
    print(json.dumps(res, indent=1))


def stage_manifest(ex, dev):
    write_json("prereg_manifest.json", {
        "prereg": PREREG, "sha256": {f: B.sha256_file(ROOT / f) for f in (PREREG, *CODE_FILES)},
        "frozen_design": {f: B.sha256_file(ART / f) for f in ("parameter_match.json", "model_configs.json", "audit.json", "train_manifest.json")},
        "c0_frozen": {"prereg": "3b6ea98", "holdout_freeze": "dddaa4b", "result": "f9e25b5"},
        "parent_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "written_before_any_c0a_training_or_metric": True})


# ----------------------------------------------------------------------------------------------- training
def stage_train_pm(ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, Y, _ = C0.load_arch("train")
    ref = C0.reference_peaks("train", Y, ex)
    E, P, RR, _ = BB.extract_training_beats(X, Y, ref)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = build("pm_bf0").to(dev)                                   # BF0 `_train_beat_model`, deterministic branch, wider
    opt = torch.optim.AdamW(net.parameters(), lr=B.LR, weight_decay=B.WD)
    g = torch.Generator().manual_seed(SEED)
    Ed, Pd, RRd = (torch.from_numpy(a).to(dev) for a in (E, P, RR))
    t0, acc, nan_steps = time.time(), [], 0
    net.train()
    for s in range(1, B.BEAT_STEPS + 1):
        idx = torch.randint(0, len(Ed), (B.BEAT_BATCH,), generator=g).to(dev)
        loss = BM.l1_loss(net, Ed[idx], Pd[idx], RRd[idx])
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), B.CLIP)
        opt.step(); acc.append(loss.item())
        if s % 1000 == 0:
            print(f"[c0a] pm_bf0 step {s} L1 {np.mean(acc):.5f}", flush=True); acc = []
    save_ckpt("pm_bf0", net, time.time() - t0, nan_steps, {"n_train_beats": int(len(E)), "steps": B.BEAT_STEPS, "batch": B.BEAT_BATCH})


def _train_window_model(name, ex, dev):
    """C0's `_train_c0` loop: full-window L1, 20,000 x 64 windows, epoch-permutation batches, seed 42, clip 1.0."""
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, Y, _ = C0.load_arch("train")
    ref = C0.reference_peaks("train", Y, ex)
    rr_edge = C0.train_rr_median()
    Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y.astype(np.float32)).to(dev)
    Rt = torch.from_numpy(AB.event_raster(ref)).to(dev) if name == "ww" else None
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = build(name).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=C0.C0T["lr"], weight_decay=C0.C0T["wd"])
    g = torch.Generator().manual_seed(SEED)
    batches, t0, acc, nan_steps = C0._epoch_batches(len(Xt), C0.C0T["batch"], g), time.time(), [], 0
    net.train()
    for step in range(1, C0.C0T["steps"] + 1):
        b = next(batches)
        bd = b.to(dev)
        out = net(Xt[bd], Rt[bd]) if name == "ww" else net(Xt[bd], [ref[int(i)] for i in b], rr_edge)
        loss = (out - Yt[bd]).abs().mean()
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), C0.C0T["clip"])
        opt.step(); acc.append(loss.item())
        if step % 1000 == 0:
            print(f"[c0a] {name} step {step} L1 {np.mean(acc):.5f}", flush=True); acc = []
    save_ckpt(name, net, time.time() - t0, nan_steps, {"steps": C0.C0T["steps"], "batch": C0.C0T["batch"]})


def stage_train_const(ex, dev):
    _train_window_model("const", ex, dev)


def stage_train_ww(ex, dev):
    _train_window_model("ww", ex, dev)


# ----------------------------------------------------------------------------------------------- inference
def render_pm(X, events, dev):
    tz = np.load(C0.OUT / "bf0det_template.npz")
    template, rr_median = tz["template"], float(tz["rr_median"])
    fill = float(np.median(np.concatenate([template[:8], template[-8:]])))
    ppg, rr, _ = B.conditions(X, events, rr_median)
    beats = B.predict_deterministic(load_new("pm_bf0", dev), ppg, rr, dev)
    return B.assemble_all(beats, events, fill).astype(np.float64)


@torch.no_grad()
def render_const(X, events, dev, rr_edge, bs=256):
    net = load_new("const", dev)
    out, gs, ls = [], [], []
    for i in range(0, len(X), bs):
        y, g, loc = net(torch.from_numpy(X[i:i + bs]).to(dev), events[i:i + bs], rr_edge, return_parts=True)
        out.append(y.cpu().numpy()); gs.append(g.cpu().numpy()); ls.append(loc.cpu().numpy())
    return np.concatenate(out).astype(np.float64), np.concatenate(gs).astype(np.float64), np.concatenate(ls).astype(np.float64)


@torch.no_grad()
def render_ww(X, events, dev, bs=256):
    net = load_new("ww", dev)
    R = AB.event_raster(events)
    out = [net(torch.from_numpy(X[i:i + bs]).to(dev), torch.from_numpy(R[i:i + bs]).to(dev)).cpu().numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out).astype(np.float64)


def arm_metrics(w, Y, ref, det, evaluable, ex):
    """Exactly C0's per-window metric block."""
    N = len(Y)
    prf = PMX.rpeak_prf_at(w, Y, FS, 50.0, peaks=(ref, det))
    bl = PMX.beat_level_metrics(w, Y, FS, 50.0, peaks=(ref, det))
    sm = MET.signal_metrics(w, Y)
    st = np.array(list(ex.map(B._structure, [(w[i], Y[i], ref[i]) for i in range(N)], chunksize=256)))
    spec = np.array([np.mean([v for kk, v in spectral_metrics(w[i], Y[i]).items() if kk.endswith("__ratio_dev")]) for i in range(N)])
    return {"f1": np.where(evaluable, prf["rpeak_f1"], np.nan), "precision": prf["rpeak_precision"], "recall": prf["rpeak_recall"],
            "fp": prf["n_fp"], "fn": prf["n_fn"], "rr_mae_ms": bl["rr_mae_ms"], "hr_abs_err": bl["hr_abs_err"],
            "mae": sm["mae"], "pcc": sm["pcc"], "s4": st[:, 0], "s5": st[:, 1], "spectral_ratio_dev": spec}


def c0_reproduction_dev(new: dict, stored: dict) -> float:
    """Max |new - stored| over the frozen C0 summary entries (REPRO_KEYS and FD); nan / missing counts as a failure (inf)."""
    devs = []
    for m in [k for k in REPRO_KEYS if k in stored] + (["fd"] if "fd" in stored else []):
        a = new[m] if isinstance(new[m], list) else [new[m]]
        b = stored[m] if isinstance(stored[m], list) else [stored[m]]
        for x, y in zip(a, b):
            if y is None and (x is None or not np.isfinite(x)):
                continue
            devs.append(abs(float(x) - float(y)) if (y is not None and x is not None and np.isfinite(x)) else float("inf"))
    return max(devs) if devs else float("inf")


def evaluate(role, ex, dev):
    X, Y, Pid = load_role(role)
    N = len(Y)
    ref = C0.reference_peaks(role, Y, ex)
    evaluable = np.array([len(r) > 0 for r in ref])
    rr_edge = C0.train_rr_median()
    st = np.load(C0.OUT / f"{role}_renders.npz")
    events = [np.asarray(e, int) for e in B.split_list(st["events_idx"], st["events_off"])]
    fresh = C0.detect_events(X, dev, ex)
    if not all(np.array_equal(a, b) for a, b in zip(events, fresh)):
        raise SystemExit(f"STOP: frozen C0 events not reproduced on {role}")
    waves = {"BF0": C0.render_bf0det(X, events, dev)}
    waves["LOCAL"], _, loc_lo = C0.render_c0("c0_localonly", X, events, dev, rr_edge)
    waves["C0"], g_c0, loc_c0 = C0.render_c0("c0", X, events, dev, rr_edge)
    for k in ("BF0", "LOCAL", "C0"):
        if np.max(np.abs(waves[k].astype(np.float32) - st[k])) != 0.0:
            raise SystemExit(f"STOP: frozen {k} render not reproduced on {role}")
    waves["PM"] = render_pm(X, events, dev)
    waves["CONST"], g_const, loc_const = render_const(X, events, dev, rr_edge)
    waves["WW"] = render_ww(X, events, dev)
    det = {k: list(ex.map(B._peaks, list(waves[k]), chunksize=256)) for k in WAVE_ARMS}
    per = {k: arm_metrics(waves[k], Y, ref, det[k], evaluable, ex) for k in WAVE_ARMS}
    pprf = PMX.rpeak_prf_at(Y, Y, FS, 50.0, peaks=(ref, events))
    pbl = PMX.beat_level_metrics(Y, Y, FS, 50.0, peaks=(ref, events))
    per["PLACED"] = {"f1": np.where(evaluable, pprf["rpeak_f1"], np.nan), "precision": pprf["rpeak_precision"],
                     "recall": pprf["rpeak_recall"], "fp": pprf["n_fp"], "fn": pprf["n_fn"], "rr_mae_ms": pbl["rr_mae_ms"],
                     "hr_abs_err": pbl["hr_abs_err"]}
    pairs = [BR.matched_pairs(ref[i], events[i], T) for i in range(N)]
    corr = {k: [BR.pair_correlations(Y[i], waves[k][i], pairs[i]) for i in range(N)] for k in WAVE_ARMS}
    wm = B.window_pair_means(corr, WAVE_ARMS)
    for k in WAVE_ARMS:
        per[k]["beat_corr"] = wm[k]
    summary = {k: {m: C0.cluster_ci(v, Pid) for m, v in d.items()} for k, d in per.items()}
    for k in WAVE_ARMS:
        summary[k]["fd"] = float(PMX.kanflow_fd(waves[k], Y))
    # ---- frozen C0 results must reproduce exactly (event / waveform metrics and FD of the frozen arms)
    stored = json.loads((C0.ART / ("val_metrics.json" if role == "val" else "holdout_metrics.json")).read_text())["summary"]
    repro = {k: c0_reproduction_dev(summary[k], stored[k]) for k in ("PLACED", "BF0", "LOCAL", "C0")}
    if not all(np.isfinite(v) and v <= 1e-9 for v in repro.values()):
        raise SystemExit(f"STOP: frozen C0 metrics not reproduced on {role}: {repro}")
    repro["beat_corr_max_abs_dev_vs_c0_population"] = max(abs(summary[k]["beat_corr"][0] - stored[k]["beat_corr"][0]) for k in ("BF0", "LOCAL", "C0"))
    # ---- paired comparisons (same patients, same windows, same resamples)
    d = lambda a, b, m: C0.cluster_ci(per[a][m] - per[b][m], Pid)  # noqa: E731
    comps = {}
    for name, other in (("M1", "PM"), ("M2", "CONST"), ("M3", "WW")):
        comps[name] = {"fp": d("C0", other, "fp"), "f1": d("C0", other, "f1"), "corr": d("C0", other, "beat_corr"),
                       "mae": d("C0", other, "mae"), "rr": d("C0", other, "rr_mae_ms"),
                       "fd": C0.fd_diff_ci(waves["C0"], waves[other], Y, Pid)}
    comps["M1"]["rhythm_f1"] = d("C0", "PLACED", "f1")
    comps["M1"]["rhythm_rr"] = d("C0", "PLACED", "rr_mae_ms")
    comps["CARRIER"] = {"fp": d("CONST", "LOCAL", "fp"), "f1": d("CONST", "LOCAL", "f1"), "corr": d("CONST", "LOCAL", "beat_corr"),
                        "fd": C0.fd_diff_ci(waves["CONST"], waves["LOCAL"], Y, Pid)}
    for other in ("PM", "CONST", "WW"):
        comps[f"{other}_rhythm"] = {"f1": d(other, "PLACED", "f1"), "rr": d(other, "PLACED", "rr_mae_ms")}
    decisions = {"M1": claim_m1(comps["M1"]), "M2": claim_m2(comps["M2"]), "M3": claim_m3(comps["M3"]),
                 "CARRIER": claim_carrier(comps["CARRIER"])}
    # ---- coherence (CONST shares C0's supports)
    coh = {}
    for k, loc in (("C0", loc_c0), ("CONST", loc_const)):
        near, tot = 0, 0
        for i in range(N):
            bp = C0.boundary_points(events[i], rr_edge)
            fps = C0.fp_detections(ref[i], det[k][i])
            tot += len(fps)
            near += sum(1 for x in fps if bp.size and np.min(np.abs(bp - x)) <= C0.BOUNDARY_NEAR)
        coh[k] = {"fp_total": tot, "boundary_near_fp": near}
    raw = {k: {m: int(np.nansum(per[k][m])) for m in ("fp", "fn")} for k in per}
    miss = {"windows": N, "windows_without_events": int(sum(len(e) == 0 for e in events)), "matched_pairs": wm["_n_pairs"],
            "nan_windows": {k: {m: int(np.sum(~np.isfinite(np.asarray(v, float)))) for m, v in dd.items()} for k, dd in per.items()}}
    return {"role": role, "summary": summary, "comparisons": comps, "decisions": decisions, "coherence": coh, "raw": raw,
            "missingness": miss, "c0_reproduction_max_abs_dev": repro}


def stage_evaluate_val(ex, dev):
    r = evaluate("val", ex, dev)
    write_json("val_metrics.json", {k: r[k] for k in ("summary", "coherence", "raw", "missingness", "c0_reproduction_max_abs_dev")})
    write_json("val_bootstrap.json", {"unit": "patient", "replicates": C0.BOOT_N, "seed": C0.BOOT_SEED, "comparisons": r["comparisons"]})
    write_json("claim_matrix_val.json", {"stage": "ARCH-VAL (development / qualification)", "decisions": r["decisions"]})
    print(json.dumps(B.clean({"decisions": r["decisions"], "comparisons": r["comparisons"]}), indent=1))


def stage_freeze(ex, dev):
    if not (ART / "claim_matrix_val.json").exists():
        raise SystemExit("no ARCH-VAL claim matrix")
    files = [PREREG, *CODE_FILES, *C0.CODE_FILES, "src/ppg2ecg/coherentbeat/ablation.py"]
    files += [B.rel(OUT / f"{n}.pt") for n in ("pm_bf0", "const", "ww")]
    files += [B.rel(C0.OUT / f"{n}.pt") for n in ("detector", "bf0det", "c0", "c0_localonly")] + [B.rel(C0.OUT / "bf0det_template.npz")]
    files += [B.rel(ART / f) for f in ("model_configs.json", "parameter_match.json", "claim_matrix_val.json", "val_bootstrap.json")]
    files += ["src/ppg2ecg/evaluation/paper_metrics.py", "src/ppg2ecg/evaluation/rpeaks.py", "src/ppg2ecg/beatfirst/render.py",
              "scripts/bf0_run.py", "artifacts/c0_coherentbeat/split_manifest.json"]
    write_json("holdout_ablation_freeze.json", {"sha256": {f: B.sha256_file(ROOT / f) for f in dict.fromkeys(files)},
                                                "status": "frozen before C0-A evaluation of the previously opened ARCH-HOLDOUT"})


def stage_evaluate_holdout(ex, dev):
    check_c0a_freeze()
    r = evaluate("holdout", ex, dev)
    val = json.loads((ART / "claim_matrix_val.json").read_text())["decisions"]
    vb = json.loads((ART / "val_bootstrap.json").read_text())["comparisons"]
    rep = {k: replication(val[k], r["decisions"][k], vb[k], r["comparisons"][k], PRIMARY_EFFECTS[k]) for k in PRIMARY_EFFECTS}
    write_json("holdout_metrics.json", {k: r[k] for k in ("summary", "coherence", "raw", "missingness", "c0_reproduction_max_abs_dev")})
    write_json("holdout_bootstrap.json", {"unit": "patient", "replicates": C0.BOOT_N, "seed": C0.BOOT_SEED, "comparisons": r["comparisons"]})
    write_json("claim_matrix_holdout.json", {"stage": "frozen secondary replication on the previously opened C0 holdout",
                                             "decisions": r["decisions"]})
    write_json("replication_summary.json", {"replication": rep, "final_matrix": final_matrix(val, rep), "c1": c1_decision(val, rep),
                                            "val_decisions": val, "holdout_decisions": r["decisions"]})
    print(json.dumps(B.clean({"holdout": r["decisions"], "replication": rep, "final": final_matrix(val, rep)}), indent=1))


# ----------------------------------------------------------------------------------------------- compute / figure / table
@torch.no_grad()
def stage_compute(ex, dev):
    from torch.utils.flop_counter import FlopCounterMode
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, _, _ = C0.load_arch("val")
    rr_edge = C0.train_rr_median()
    pick = B.salted_rank("c0a-latency-v1", range(len(X)))[:50]
    tz = np.load(C0.OUT / "bf0det_template.npz")
    torch.set_num_threads(4)
    lat, flops = {}, {}
    for dname in ("cpu", "cuda"):
        d = torch.device(dname)
        detn = C0.load_net("detector", d)
        nets = {"PM": load_new("pm_bf0", d), "CONST": load_new("const", d), "WW": load_new("ww", d),
                "BF0": C0.load_net("bf0det", d), "C0": C0.load_net("c0", d), "LOCAL": C0.load_net("c0_localonly", d)}

        def run(kind, i):
            p = torch.sigmoid(detn(torch.from_numpy(X[i:i + 1]).to(d)[:, None])[:, 0]).float().cpu().numpy()[0]
            ev = C0._extract(p)
            xi = torch.from_numpy(X[i:i + 1]).to(d)
            if kind in ("C0", "LOCAL", "CONST"):
                nets[kind](xi, [ev], rr_edge)
            elif kind == "WW":
                nets["WW"](xi, torch.from_numpy(AB.event_raster([ev])).to(d))
            elif len(ev):
                ppg, rr, _ = B.conditions(X[i:i + 1], [ev], float(tz["rr_median"]))
                BR.assemble(B.predict_deterministic(nets[kind], ppg, rr, d), ev, T, 0.0)
            return len(ev)

        for kind in ("BF0", "PM", "LOCAL", "CONST", "WW", "C0"):
            ts = []
            for rep, i in enumerate(list(pick[:3]) + list(pick)):
                if dname == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                run(kind, i)
                if dname == "cuda":
                    torch.cuda.synchronize()
                if rep >= 3:
                    ts.append((time.perf_counter() - t0) * 1000)
            lat[f"{dname}_{kind}"] = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}
            if dname == "cpu":
                fc = FlopCounterMode(display=False)
                with fc:
                    nb = run(kind, int(pick[0]))
                flops[kind] = {"flops_window_incl_detector": int(fc.get_total_flops()), "events_in_window": nb}
    ck = {n: json.loads((ART / f"checkpoint_{n}.json").read_text()) for n in ("pm_bf0", "const", "ww")}
    c0c = json.loads((C0.ART / "compute_accounting.json").read_text())
    write_json("compute_accounting.json", {
        "params": {"PM": ck["pm_bf0"]["n_params"], "CONST": ck["const"]["n_params"], "WW": ck["ww"]["n_params"]} | {
            "BF0": c0c["params"]["bf0det"], "LOCAL": c0c["params"]["c0_localonly"], "C0": c0c["params"]["c0"]},
        "train_seconds": {"PM": ck["pm_bf0"]["train_seconds"], "CONST": ck["const"]["train_seconds"], "WW": ck["ww"]["train_seconds"]} | {
            "BF0": c0c["train_seconds"]["bf0det"], "LOCAL": c0c["train_seconds"]["c0_localonly"], "C0": c0c["train_seconds"]["c0"]},
        "peak_gpu_mem_mib": {"PM": ck["pm_bf0"]["peak_gpu_mem_mib"], "CONST": ck["const"]["peak_gpu_mem_mib"], "WW": ck["ww"]["peak_gpu_mem_mib"]},
        "nan_steps": {n: ck[n]["nan_steps"] for n in ck}, "batch1_latency_ms": lat, "flops_one_window": flops,
        "notes": "parameter-matched, not compute-matched; FLOPs from torch.utils.flop_counter (one salted window, CPU), a lower bound",
        "software": B.software()})
    write_json("checkpoint_hashes.json", {n: ck[n]["sha256"] for n in ck})


def stage_figure(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vm, vb = json.loads((ART / "val_metrics.json").read_text()), json.loads((ART / "val_bootstrap.json").read_text())
    hold = (ART / "holdout_metrics.json").exists()
    hm = json.loads((ART / "holdout_metrics.json").read_text()) if hold else None
    hb = json.loads((ART / "holdout_bootstrap.json").read_text()) if hold else None
    cc = json.loads((ART / "compute_accounting.json").read_text())
    pmj = json.loads((ART / "parameter_match.json").read_text())
    rows = ["PLACED", "BF0", "PM", "LOCAL", "CONST", "WW", "C0"]
    with open(ART / "table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "method", "params", "F1", "FP_per_window", "RR_MAE_ms", "FD", "morph_corr", "MAE", "GPU_latency_ms"])
        for split, m in (("ARCH-VAL", vm), ("ARCH-HOLDOUT (previously opened)", hm)):
            if m is None:
                continue
            S = m["summary"]
            for k in rows:
                s = S[k]
                w.writerow([split, LABEL[k], cc["params"].get(k, ""), f"{s['f1'][0]:.4f}", f"{s['fp'][0]:.4f}", f"{s['rr_mae_ms'][0]:.3f}",
                            f"{s['fd']:.2f}" if "fd" in s else "", f"{s['beat_corr'][0]:.4f}" if "beat_corr" in s else "",
                            f"{s['mae'][0]:.4f}" if "mae" in s else "", f"{cc['batch1_latency_ms'][f'cuda_{k}']['median_ms']:.2f}" if k != "PLACED" else ""])
    C = {"BF0": "#4A3AA7", "PM": "#7B6FD0", "LOCAL": "#B0621B", "CONST": "#D69A57", "WW": "#A8434F", "C0": "#0E7A86"}
    ks = ["BF0", "PM", "LOCAL", "CONST", "WW", "C0"]
    fig, ax = plt.subplots(2, 3, figsize=(19, 9.5))
    a = ax[0, 0]
    a.axis("off")
    a.set_title("A  model schematics", loc="left", fontsize=10)
    for j, s in enumerate(["PM-BF0-DET: BF0 beat (166-sample, absolute level), width 73 -> Hann OLA",
                           "CONST+LOCAL: scalar level a + sum_i bump_i * q_i (C0 local branch)",
                           "WW-DET: [C0 encoder features, event raster] -> TCN -> 512 samples",
                           "CoherentBeat-C0: spline g(t) (17 coef) + sum_i bump_i * q_i",
                           f"param match: PM {pmj['PM']['mismatch']:+.2%}, WW {pmj['WW']['mismatch']:+.2%}, CONST {pmj['CONST']['mismatch']:+.2%}"]):
        a.text(0.0, 0.85 - 0.17 * j, s, fontsize=9)
    a = ax[0, 1]
    a.bar(range(6), [cc["params"][k] / 1e3 for k in ks], color=[C[k] for k in ks], width=0.6)
    a.axhline(AB.C0_PARAMS / 1e3, color="k", ls=":", lw=0.8)
    a.set_xticks(range(6), [LABEL[k] for k in ks], rotation=30, fontsize=7)
    a.set_ylabel("parameters (k)")
    a.set_title("B  parameter counts (dotted: C0)", loc="left", fontsize=10)
    for (r, c), (m, lab) in zip(((0, 2), (1, 0), (1, 1)), (("fp", "FP per window"), ("fd", "FD"), ("beat_corr", "beat-aligned correlation"))):
        a = ax[r, c]
        for j, (S, off, alpha) in enumerate(((vm["summary"], -0.2, 1.0), ((hm or {}).get("summary"), 0.2, 0.5))):
            if S is None:
                continue
            vals = [S[k][m] if m == "fd" else S[k][m][0] for k in ks]
            a.bar(np.arange(6) + off, vals, width=0.38, color=[C[k] for k in ks], alpha=alpha, label=("ARCH-VAL" if j == 0 else "ARCH-HOLDOUT"))
        a.set_xticks(range(6), [LABEL[k] for k in ks], rotation=30, fontsize=7)
        a.set_ylabel(lab)
        a.legend(fontsize=7)
        a.set_title({"fp": "C  false R detections (lower = better)", "fd": "D  FD (lower = better)",
                     "beat_corr": "E  morphology correlation (higher = better)"}[m], loc="left", fontsize=10)
    a = ax[1, 2]
    labels, x = [], 0
    for comp, (a_arm, b_arm) in (("M1", ("C0", "PM")), ("M2", ("C0", "CONST")), ("M3", ("C0", "WW")), ("CARRIER", ("CONST", "LOCAL"))):
        for m in ("fp", "fd"):
            for j, (Bt, Mt, col) in enumerate(((vb, vm, "k"), (hb, hm, "0.55"))):
                if Bt is None:
                    continue
                v = Bt["comparisons"][comp][m]
                base = Mt["summary"][b_arm][m] if m == "fd" else Mt["summary"][b_arm][m][0]
                rel = [100.0 * t / base for t in v]
                a.vlines(x + 0.15 * j, min(rel[1], rel[2]), max(rel[1], rel[2]), color=col, lw=1.5)   # percentile CI
                a.plot(x + 0.15 * j, rel[0], "o", color=col, ms=5)
            labels.append(f"{comp} {m.upper()}\n{a_arm}-{b_arm}")
            x += 1
    a.axhline(0, color="k", lw=0.6)
    a.set_xticks(range(len(labels)), labels, fontsize=6.5)
    a.set_ylabel("relative effect (%) = diff / second arm; < 0 favours the first arm")
    a.set_title("F  ARCH-VAL (black) vs ARCH-HOLDOUT (grey) effects", loc="left", fontsize=10)
    dv = json.loads((ART / "claim_matrix_val.json").read_text())["decisions"]
    fig.suptitle(f"C0-A attribution — ARCH-VAL: M1 {dv['M1']}, M2 {dv['M2']}, M3 {dv['M3']}; carrier {dv['CARRIER']}"
                 + ("; holdout = frozen secondary replication on the previously opened C0 holdout" if hold else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(ART / "figure.png", dpi=150)


STAGES = {"params": stage_params, "audit": stage_audit, "manifest": stage_manifest, "train_pm": stage_train_pm,
          "train_const": stage_train_const, "train_ww": stage_train_ww, "evaluate_val": stage_evaluate_val,
          "freeze": stage_freeze, "evaluate_holdout": stage_evaluate_holdout, "compute": stage_compute, "figure": stage_figure}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(10) as ex:
        STAGES[args.stage](ex, dev)


if __name__ == "__main__":
    main()
