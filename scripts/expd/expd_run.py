"""EXP-D — functional generalisation of fixed-budget allocation (docs/EXP_D_FUNCTIONAL_GENERALIZATION_PREREGISTRATION.md,
frozen at eddbe60). U1 upstream PENGUIN checkpoints as shipped, no training, DW1 generation protocol.

Stages:
  gen <BIDMC|WESAD|MIMIC-BP>   draws per S (S=1: 32; S=2,4,8: 16; S=16: 2; S=32: 1) + Heun-50 reference, common noise
                               z_d = randn(Generator(d)) over the whole test stream; per-sample block functionals, reference
                               functionals, K = 16 block-waveform pairwise RMS (S <= 8), draw-0 waveform metrics, FD, latency.
  analyze <respiration|abp>    every preregistered cell, contrast, bootstrap, mechanism statistic and the verdict.

Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/expd/expd_run.py gen BIDMC
     PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/expd/expd_run.py analyze respiration
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sps
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "external/PENGUIN"
sys.path.insert(0, str(UP / "src"))
from utils.help_func import fid_features_to_statistics, fid_statistics_to_metric, initialize_model  # noqa: E402  (upstream)

from ppg2ecg.evaluation import penguin_metrics as PM  # noqa: E402
from ppg2ecg.flow.samplers import euler_sample, heun_sample  # noqa: E402

U1 = ROOT / "outputs/u1_upstream"
PROC = ROOT / "data/processed/upstream_u1"
RAW = ROOT / "outputs/exp_d_functional_generalization"
ART = ROOT / "artifacts/exp_d_functional_generalization"
PREREG = "docs/EXP_D_FUNCTIONAL_GENERALIZATION_PREREGISTRATION.md (eddbe60)"
TASK = {"BIDMC": "respiration", "WESAD": "respiration", "MIMIC-BP": "abp"}
FUNCS = {"respiration": ("RR",), "abp": ("SBP", "DBP", "MAP")}
UNIT = {"respiration": "breaths/min", "abp": "mmHg"}
WPB = {"respiration": 15, "abp": 2}
DRAWS = {1: 32, 2: 16, 4: 16, 8: 16, 16: 2, 32: 1}
SS = (1, 2, 4, 8, 16, 32)
MECH_S = (1, 2, 4, 8)
BS, FS, NB, BOOT_SEED = 512, 128, 5000, 20260924
B32 = [(32, 1), (16, 2), (8, 4), (4, 8), (2, 16), (1, 32)]
B16 = [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)]
HEAD = {"B32_width_(32,1)-(1,32)": ((32, 1), (1, 32)), "B32_intermediate_(8,4)-(1,32)": ((8, 4), (1, 32)),
        "B16_width_(16,1)-(1,16)": ((16, 1), (1, 16)), "B16_intermediate_(4,4)-(1,16)": ((4, 4), (1, 16))}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_stream(ds, part="test", limit=None):
    sp = json.loads((U1 / f"split_{ds}.json").read_text())
    X, Y, sub = [], [], []
    for f in sp[part][:limit]:
        with open(PROC / ds / f, "rb") as fh:
            d = pickle.load(fh)
        X.append(np.asarray(d["x_data"], np.float32)); Y.append(np.asarray(d["y_data"], np.float32))
        sub += [int(f.replace("subject", "").replace(".pkl", ""))] * len(d["x_data"])
    return np.concatenate(X), np.concatenate(Y), np.array(sub)


def functionals(task, blocks, reference=False):
    """blocks [n, w*512] -> [n, n_func]. Upstream definitions (penguin_metrics port of help_func.compute_metrics)."""
    b = np.asarray(blocks, np.float64)
    if task == "respiration":
        if not reference:                                    # upstream low-passes the prediction only
            bb, aa = sps.butter(PM.RESP_ORDER, PM.RESP_LOWPASS_HZ / (0.5 * FS), btype="low")
            b = sps.filtfilt(bb, aa, b, axis=-1)
        return PM.dominant_bpm(b, FS)[:, None]
    return np.stack([b.max(-1), b.min(-1), b.mean(-1)], 1)


def to_blocks(x, w):
    return x.reshape(len(x) // w, w * x.shape[-1])


# ================================================================================================ gen
@torch.no_grad()
def gen(ds):
    task = TASK[ds]; w = WPB[task]
    dev = torch.device("cuda")
    out = RAW / f"{ds}.npz"
    if out.exists():
        print(f"[gen] {ds} cached", flush=True); return
    RAW.mkdir(parents=True, exist_ok=True)
    ck_path = U1 / f"PENGUIN_{ds}_u1/ckpt/pretrain_ckpt.pth"
    ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    model = initialize_model(ck["cfg"], device=dev)
    r = model.load_state_dict(ck["state_dict"], strict=True)
    assert not r.missing_keys and not r.unexpected_keys
    model.eval()
    X, Y, sub = load_stream(ds)
    N = len(X); assert N % w == 0
    Xt = torch.from_numpy(X)
    blocks_sub = sub[::w]
    assert (sub.reshape(-1, w) == blocks_sub[:, None]).all()                      # no block crosses a subject
    T = functionals(task, to_blocks(Y, w), reference=True)

    def run(z, S, heun=False):
        outs = []
        for i in range(0, N, BS):
            ppg = Xt[i:i + BS].to(dev).unsqueeze(1)
            v = lambda xt, t, _p=ppg: model.forward_step(xt, _p, t)  # noqa: E731
            x, k = (heun_sample(v, z[i:i + BS].to(dev), 25) if heun else euler_sample(v, z[i:i + BS].to(dev), S))
            assert k == (50 if heun else S)
            outs.append(x[:, 0].float().cpu().numpy())
        return np.concatenate(outs)

    def noise(d):
        return torch.randn(N, 1, X.shape[1], generator=torch.Generator().manual_seed(d))

    res = {"T": T, "block_subject": blocks_sub, "window_subject": sub}
    ref = Y.astype(np.float64); rc = ref - ref.mean(1, keepdims=True); rn = np.sqrt((rc ** 2).sum(1))
    nonfin, fd = {}, {}
    t0 = time.time()

    def wave_metrics(g):
        g = g.astype(np.float64); e = g - ref; gc = g - g.mean(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            r_ = (gc * rc).sum(1) / (np.sqrt((gc ** 2).sum(1)) * rn)
        return np.stack([np.abs(e).mean(1), np.sqrt((e ** 2).mean(1)), r_], 1)

    fd_ref = fid_features_to_statistics(torch.from_numpy(Y))
    for S in SS:
        F, bad, keep = [], 0, []
        for d in range(DRAWS[S]):
            g = run(noise(d), S)
            bad += int((~np.isfinite(g).all(1)).sum())
            gb = to_blocks(g, w)
            fv = np.full((len(gb), len(FUNCS[task])), np.nan)
            ok = np.isfinite(gb).all(1)
            fv[ok] = functionals(task, gb[ok])
            F.append(fv)
            if d == 0:
                res[f"wave_S{S}"] = wave_metrics(g)
                fd[S] = fid_statistics_to_metric(fid_features_to_statistics(torch.from_numpy(g)), fd_ref)
            if S in MECH_S and d < 16:
                keep.append(gb.astype(np.float32))
        res[f"F_S{S}"] = np.stack(F)
        if S in MECH_S:
            Wk = np.stack(keep); acc = np.zeros(len(blocks_sub))
            for i in range(16):
                for j in range(i + 1, 16):
                    acc += np.sqrt(((Wk[i] - Wk[j]) ** 2).mean(1))
            res[f"wRMS_S{S}"] = acc / 120; del Wk, keep
        nonfin[S] = bad
        print(f"[gen] {ds} S={S} draws={DRAWS[S]} FD(draw0)={fd[S]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    g = run(noise(0), 25, heun=True)
    gb = to_blocks(g, w)
    res["F_heun"] = functionals(task, gb)[None]
    res["wave_heun"] = wave_metrics(g); fd["heun50"] = fid_statistics_to_metric(fid_features_to_statistics(torch.from_numpy(g)), fd_ref)
    nonfin["heun50"] = int((~np.isfinite(g).all(1)).sum())
    print(f"[gen] {ds} Heun-50 ({time.time() - t0:.0f}s)", flush=True)
    # latency per cell (content-independent): K sequential batch-1 calls vs one call with K samples batched
    lat = {}
    p1 = Xt[:1].to(dev).unsqueeze(1)

    def timed(K, S, heun=False, batched=False, reps=3):
        pp = p1.repeat(K, 1, 1) if batched else p1
        z = torch.randn(K if batched else 1, 1, X.shape[1], device=dev)
        v = lambda xt, t: model.forward_step(xt, pp, t)  # noqa: E731
        ts = []
        for i in range(reps + 1):
            torch.cuda.synchronize(); a = time.perf_counter()
            for _ in range(1 if batched else K):
                heun_sample(v, z, 25) if heun else euler_sample(v, z, S)
            torch.cuda.synchronize()
            if i:
                ts.append(1000 * (time.perf_counter() - a))
        return float(np.median(ts))
    for K, S in B32 + B16:
        lat[f"K{K}_S{S}"] = {"nfe": K * S, "sequential_batch1_ms": timed(K, S), "batched_K_ms": timed(K, S, batched=True)}
    lat["heun50_K1"] = {"nfe": 50, "sequential_batch1_ms": timed(1, 25, heun=True), "batched_K_ms": None}
    meta = {"dataset": ds, "checkpoint": str(ck_path.relative_to(ROOT)), "sha256": sha256(ck_path), "epoch": int(ck["epoch"]),
            "n_windows": N, "n_blocks": len(blocks_sub), "n_subjects": int(len(np.unique(sub))), "windows_per_block": w,
            "draws": {str(k): v for k, v in DRAWS.items()}, "noise": "z_d = torch.randn(N,1,512, generator=Generator().manual_seed(d)), same for every S",
            "nonfinite_samples": {str(k): v for k, v in nonfin.items()}, "FD_draw0": {str(k): float(v) for k, v in fd.items()},
            "latency_ms": lat, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    np.savez(out, **res)
    (RAW / f"{ds}.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[gen] {ds} done ({time.time() - t0:.0f}s)", flush=True)


# ================================================================================================ analyze helpers
class Boot:
    def __init__(self, cluster):
        self.up, self.inv = np.unique(cluster, return_inverse=True)
        self.P = len(self.up)
        rng = np.random.default_rng(BOOT_SEED)
        self.C = rng.multinomial(self.P, np.full(self.P, 1 / self.P), size=NB).astype(np.float64)

    def pmean(self, v, mask=None):
        ok = np.isfinite(v) if mask is None else (np.isfinite(v) & mask)
        s = np.bincount(self.inv[ok], v[ok], self.P); n = np.bincount(self.inv[ok], None, self.P)
        with np.errstate(invalid="ignore"):
            return s / n

    def ci(self, pv):
        m = np.isfinite(pv)
        b = (self.C[:, m] @ pv[m]) / self.C[:, m].sum(1)
        return {"point": float(pv[m].mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                "n_clusters": int(m.sum()), "share_boot_below_0": float((b < 0).mean())}


def contrast(bt, ea, eb):
    m = np.isfinite(ea) & np.isfinite(eb)
    pa, pb = bt.pmean(ea, m), bt.pmean(eb, m); d = pa - pb; fin = np.isfinite(d)
    r = {"a": bt.ci(pa), "b": bt.ci(pb), "diff": bt.ci(d), "n_blocks": int(m.sum()),
         "win_rate_a_better": float((d[fin] < 0).mean()), "tie_rate": float((d[fin] == 0).mean()),
         "block_win_rate_a_better": float((ea[m] < eb[m]).mean()), "block_tie_rate": float((ea[m] == eb[m]).mean())}
    r["significant"] = "a_better" if r["diff"]["ci"][1] < 0 else ("b_better" if r["diff"]["ci"][0] > 0 else "ns")
    return r


def cell_err(F, T, K, S, j):
    """nested draws 0..K-1 at depth S, median consensus; returns block error [n_blocks]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.abs(np.nanmedian(F[S][:K, :, j], 0) - T[:, j])


def rank(x):
    return np.argsort(np.argsort(x, axis=-1), axis=-1).astype(float)


def spearman_rows(a, b):
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(-1, keepdims=True); rb -= rb.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def rho_from_stats(n, s1, s2):
    mu = s1 / n[..., None]
    cov = s2 / n[..., None, None] - mu[..., :, None] * mu[..., None, :]
    sd = np.sqrt(np.clip(np.diagonal(cov, axis1=-2, axis2=-1), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / (sd[..., :, None] * sd[..., None, :])
    iu = np.triu_indices(16, 1)
    return corr[..., iu[0], iu[1]].mean(-1)


def fmt(r):
    return f"{r['point']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"


def train_constant(ds, task):
    """median reference functional over training-subject blocks (blocks within subject; trailing windows dropped)."""
    X, Y, sub = load_stream(ds, "train")
    w = WPB[task]
    blocks = [to_blocks(Y[sub == s][: (int((sub == s).sum()) // w) * w], w) for s in np.unique(sub)]
    return np.median(functionals(task, np.concatenate(blocks), reference=True), 0)


# ================================================================================================ analyze one dataset
def analyze_ds(ds, cluster_blocks=False):
    task = TASK[ds]; z = np.load(RAW / f"{ds}.npz"); meta = json.loads((RAW / f"{ds}.meta.json").read_text())
    F = {S: z[f"F_S{S}"] for S in SS}; T = z["T"]; bsub = z["block_subject"]
    cl = np.arange(len(bsub)) if cluster_blocks else bsub
    bt = Boot(cl)
    const = train_constant(ds, task)
    out = {"dataset": ds, "cluster": "block (single subject; exploratory)" if cluster_blocks else "subject",
           "n_blocks": int(len(bsub)), "n_subjects": int(len(np.unique(bsub))), "meta": meta, "functionals": {}}
    for j, fn in enumerate(FUNCS[task]):
        E = {(K, S): cell_err(F, T, K, S, j) for K, S in B32 + B16 + [(1, S) for S in SS] + [(16, S) for S in MECH_S]}
        E[("heun", 50)] = np.abs(z["F_heun"][0, :, j] - T[:, j])
        r = {"cells": {f"K{K}_S{S}": bt.ci(bt.pmean(E[(K, S)])) for (K, S) in E if K != "heun"},
             "heun50_K1": bt.ci(bt.pmean(E[("heun", 50)])),
             "constant_train_median": {"value": float(const[j]), "error": bt.ci(bt.pmean(np.abs(const[j] - T[:, j])))}}
        r["contrasts"] = {t: contrast(bt, E[a], E[b]) for t, (a, b) in HEAD.items()}
        r["descriptive"] = {f"B32_(K{K},S{S})-(1,32)": contrast(bt, E[(K, S)], E[(1, 32)]) for K, S in B32 if (K, S) != (1, 32)}
        r["descriptive"].update({f"B16_(K{K},S{S})-(1,16)": contrast(bt, E[(K, S)], E[(1, 16)]) for K, S in B16 if (K, S) != (1, 16)})
        r["descriptive"].update({"(32,1)-heun50": contrast(bt, E[(32, 1)], E[("heun", 50)]),
                                 "(8,4)-heun50": contrast(bt, E[(8, 4)], E[("heun", 50)]),
                                 "(1,32)-heun50": contrast(bt, E[(1, 32)], E[("heun", 50)])})
        r["consensus_checks"] = {f"(16,{S})-(1,{S})": contrast(bt, E[(16, S)], E[(1, S)]) for S in MECH_S}
        b32 = {c: r["cells"][f"K{c[0]}_S{c[1]}"]["point"] for c in B32}
        b16 = {c: r["cells"][f"K{c[0]}_S{c[1]}"]["point"] for c in B16}
        r["test_best_exploratory"] = {"B32": list(min(b32, key=b32.get)), "B16": list(min(b16, key=b16.get))}
        # ---------------- mechanism (K = 16, S in 1,2,4,8)
        om = np.all([np.isfinite(F[S][:16, :, j]).all(0) for S in MECH_S], 0) & np.isfinite(T[:, j])
        mech, stats = {}, {}
        for S in MECH_S:
            Yk = F[S][:16, om, j]; e = Yk - T[om, j]
            I = np.abs(e).mean(0); C = np.abs(np.median(Yk, 0) - T[om, j]); G = I - C
            SD = Yk.std(0, ddof=1); MAD = np.median(np.abs(Yk - np.median(Yk, 0)), 0); wr = z[f"wRMS_S{S}"][om]
            vals = {}
            for k, v in (("I", I), ("C", C), ("G", G), ("SD", SD), ("MAD", MAD), ("wRMS", wr)):
                full = np.full(len(bsub), np.nan); full[om] = v; vals[k] = bt.pmean(full)
            with np.errstate(invalid="ignore", divide="ignore"):
                cm = np.corrcoef(e)
            rho = float(np.nanmean(cm[np.triu_indices(16, 1)])) if np.isfinite(cm).any() else float("nan")
            io = bt.inv[om]
            n_p = np.bincount(io, None, bt.P).astype(float)
            s1 = np.stack([np.bincount(io, e[k], bt.P) for k in range(16)], 1)
            s2 = np.zeros((bt.P, 16, 16))
            for k in range(16):
                for l in range(k, 16):
                    s2[:, k, l] = s2[:, l, k] = np.bincount(io, e[k] * e[l], bt.P)
            stats[S] = (n_p, s1, s2, vals)
            constant_series = int(sum(np.ptp(e[k]) == 0 for k in range(16)))
            mech[S] = {"n_blocks": int(om.sum()), **{k: bt.ci(v) for k, v in vals.items()},
                       "G_over_I": float(np.nanmean(vals["G"]) / np.nanmean(vals["I"])), "rho_bar": rho,
                       "K_eff": 16 / (1 + 15 * rho) if np.isfinite(rho) else None, "constant_error_series": constant_series}
            if not cluster_blocks and task == "abp":
                rp, gp = [], []
                for p in np.flatnonzero(n_p >= 8):
                    sel = io == p
                    with np.errstate(invalid="ignore", divide="ignore"):
                        c = np.corrcoef(e[:, sel])
                    if np.isfinite(c).all():
                        rp.append(c[np.triu_indices(16, 1)].mean()); gp.append(vals["G"][p])
                rp, gp = np.array(rp), np.array(gp); rng = np.random.default_rng(BOOT_SEED)
                bs = [spearmanr(rp[ix], gp[ix])[0] for ix in (rng.integers(0, len(rp), len(rp)) for _ in range(NB))]
                mech[S]["subject_level_spearman_rho_G"] = {"n_subjects": int(len(rp)), "point": float(spearmanr(rp, gp)[0]),
                                                           "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                                                           "share_negative": float(np.mean(np.array(bs) < 0))}
        Cb = bt.C
        rb = np.stack([rho_from_stats(Cb @ stats[S][0], Cb @ stats[S][1], (Cb @ stats[S][2].reshape(bt.P, -1)).reshape(NB, 16, 16))
                       for S in MECH_S], 1)

        def bmean(S, k):
            v = stats[S][3][k]; m = np.isfinite(v)
            return (Cb[:, m] @ v[m]) / Cb[:, m].sum(1)
        pts = {k: np.array([mech[S][k]["point"] for S in MECH_S]) for k in ("G", "I", "SD", "MAD", "wRMS")}
        pts["rho_bar"] = np.array([mech[S]["rho_bar"] for S in MECH_S])
        Gb = np.stack([bmean(S, "G") for S in MECH_S], 1); Ib = np.stack([bmean(S, "I") for S in MECH_S], 1)
        sp = {}
        for name, (xa, xb, ba, bb) in {"rho_bar_vs_G": (pts["rho_bar"], pts["G"], rb, Gb),
                                       "rho_bar_vs_G_over_I": (pts["rho_bar"], pts["G"] / pts["I"], rb, Gb / Ib),
                                       "wRMS_vs_G": (pts["wRMS"], pts["G"], np.stack([bmean(S, "wRMS") for S in MECH_S], 1), Gb),
                                       "SD_vs_G": (pts["SD"], pts["G"], np.stack([bmean(S, "SD") for S in MECH_S], 1), Gb),
                                       "MAD_vs_G": (pts["MAD"], pts["G"], np.stack([bmean(S, "MAD") for S in MECH_S], 1), Gb)}.items():
            b = spearman_rows(ba, bb)
            sp[name] = {"point": float(spearmanr(xa, xb)[0]) if np.isfinite(xa).all() else float("nan"),
                        "ci": [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))], "share_negative": float(np.nanmean(b < 0))}
        r["mechanism"] = {"omega_blocks": int(om.sum()), "per_S": mech, "across_S_spearman": sp,
                          "ordering": {k: [MECH_S[i] for i in np.argsort(v)] for k, v in pts.items() if np.isfinite(v).all()},
                          "rho_bar_boot_ci": {S: [float(np.nanpercentile(rb[:, i], 2.5)), float(np.nanpercentile(rb[:, i], 97.5))]
                                              for i, S in enumerate(MECH_S)},
                          "directionally_consistent": bool(sp["rho_bar_vs_G"]["point"] < 0)}
        r["_E"] = E
        out["functionals"][fn] = r
    # ---------------- waveform metrics (draw 0) + collapse check
    wsub = z["window_subject"]; wb = Boot(np.arange(len(wsub)) if cluster_blocks else wsub)
    wv = {}
    for key in [f"wave_S{S}" for S in SS] + ["wave_heun"]:
        a = z[key]
        wv[key.replace("wave_", "")] = {m: wb.ci(wb.pmean(a[:, i])) for i, m in enumerate(("MAE", "RMSE", "pearson_r"))}
    for S in SS:
        wv[f"S{S}"]["FD"] = meta["FD_draw0"][str(S)]
    wv["heun"]["FD"] = meta["FD_draw0"]["heun50"]
    col = {}
    for c, S in (("(32,1)", 1), ("(8,4)", 4)):
        rr, rm, fd = wv[f"S{S}"]["pearson_r"]["point"], wv[f"S{S}"]["RMSE"]["point"], wv[f"S{S}"]["FD"]
        r32, rm32, fd32 = wv["S32"]["pearson_r"]["point"], wv["S32"]["RMSE"]["point"], wv["S32"]["FD"]
        col[c] = {"collapsed": bool(rr < 0.5 * r32 or rm > 1.5 * rm32 or fd > 3 * fd32 or meta["nonfinite_samples"][str(S)] / meta["n_windows"] / DRAWS[S] > 0.01),
                  "r": rr, "r_S32": r32, "RMSE": rm, "RMSE_S32": rm32, "FD": fd, "FD_S32": fd32}
    out["waveform"] = wv; out["collapse"] = col
    return out


def verdict_resp(r):
    f = r["functionals"]["RR"]; c = f["contrasts"]; col = r["collapse"]
    ok = {"B32_width_(32,1)-(1,32)": not col["(32,1)"]["collapsed"], "B32_intermediate_(8,4)-(1,32)": not col["(8,4)"]["collapsed"],
          "B16_width_(16,1)-(1,16)": not col["(32,1)"]["collapsed"], "B16_intermediate_(4,4)-(1,16)": not col["(8,4)"]["collapsed"]}
    succ = lambda keys: any(c[k]["significant"] == "a_better" and ok[k] for k in keys)  # noqa: E731
    b32, b16 = succ(list(HEAD)[:2]), succ(list(HEAD)[2:])
    depth_clear = all(c[k]["significant"] == "b_better" for k in list(HEAD)[:2])
    no_gain = all(v["diff"]["ci"][1] >= 0 for v in f["consensus_checks"].values())
    b32_dir = any(c[k]["diff"]["point"] < 0 for k in list(HEAD)[:2])
    mech = f["mechanism"]["directionally_consistent"]
    if depth_clear or no_gain or (not b32_dir and not (b32 or b16)):
        v = "FAILED"
    elif b32 and b16 and mech:
        v = "STRONG"
    else:
        v = "PARTIAL"
    return {"category": v, "B32_succeeds": b32, "B16_succeeds": b16, "depth_clear_B32": depth_clear, "no_useful_consensus_gain": no_gain,
            "B32_direction_width": b32_dir, "mechanism_consistent": mech, "collapse": {k: v["collapsed"] for k, v in col.items()}}


def verdict_abp(r):
    per = {}
    for fn, f in r["functionals"].items():
        c = [f["contrasts"][k] for k in list(HEAD)[:2]]
        per[fn] = {"success": any(x["significant"] == "a_better" for x in c), "depth_clear": all(x["significant"] == "b_better" for x in c),
                   "directional": any(x["diff"]["point"] < 0 for x in c)}
    ns = sum(p["success"] for p in per.values()); nd = sum(p["depth_clear"] for p in per.values()); ndir = sum(p["directional"] for p in per.values())
    invalid = any(v / r["meta"]["n_windows"] / DRAWS.get(int(k), 1) > 0.01 for k, v in r["meta"]["nonfinite_samples"].items() if k != "heun50")
    if nd >= 2 or invalid or (ns == 0 and ndir <= 1):
        v = "FAILED"
    elif ns >= 2:
        v = "STRONG"
    else:
        v = "PARTIAL"
    return {"category": v, "per_functional": per, "n_success": ns, "n_depth_clear": nd, "n_directional": ndir, "extraction_invalid": invalid}


def analyze(task):
    ART.joinpath(task).mkdir(parents=True, exist_ok=True)
    dsets = ("BIDMC", "WESAD") if task == "respiration" else ("MIMIC-BP",)
    res = {"prereg": PREREG, "task": task, "unit": UNIT[task], "datasets": {}}
    for ds in dsets:
        res["datasets"][ds] = analyze_ds(ds, cluster_blocks=(ds == "WESAD"))
    prim = res["datasets"][dsets[0]]
    res["verdict"] = verdict_resp(prim) if task == "respiration" else verdict_abp(prim)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (ART / task / "prereg_manifest.json").write_text(json.dumps({
        "prereg": PREREG, "code_head_at_analysis": head, "datasets": {ds: res["datasets"][ds]["meta"] for ds in dsets},
        "sampler": "Euler (ppg2ecg.flow.samplers.euler_sample) on upstream grid; reference Heun-25 = 50 NFE",
        "estimator": "nested draws 0..K-1, median consensus", "bootstrap": {"replicates": NB, "seed": BOOT_SEED},
        "cells_B32": B32, "cells_B16": B16, "headline": {k: [list(a), list(b)] for k, (a, b) in HEAD.items()}}, indent=1))
    (ART / task / "latency.json").write_text(json.dumps({ds: res["datasets"][ds]["meta"]["latency_ms"] for ds in dsets}, indent=1))
    # fixed_budget.csv
    with open(ART / task / "fixed_budget.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["dataset", "functional", "budget", "K", "S", "NFE", "error", "ci_lo", "ci_hi", "n_clusters", "latency_seq_batch1_ms", "latency_batched_ms"])
        for ds in dsets:
            d = res["datasets"][ds]; lat = d["meta"]["latency_ms"]
            for fn, f in d["functionals"].items():
                for bud, cells in (("B32", B32), ("B16", B16)):
                    for K, S in cells:
                        c = f["cells"][f"K{K}_S{S}"]; L = lat[f"K{K}_S{S}"]
                        wr.writerow([ds, fn, bud, K, S, K * S, c["point"], *c["ci"], c["n_clusters"], L["sequential_batch1_ms"], L["batched_K_ms"]])
                c = f["heun50_K1"]
                wr.writerow([ds, fn, "shipped_ref", 1, "Heun25", 50, c["point"], *c["ci"], c["n_clusters"], lat["heun50_K1"]["sequential_batch1_ms"], ""])
    # mechanism.csv
    with open(ART / task / "mechanism.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["dataset", "functional", "S", "n_blocks", "I", "C", "G", "G_lo", "G_hi", "G_over_I", "SD", "MAD", "waveform_pairwise_RMS",
                     "rho_bar", "rho_lo", "rho_hi", "K_eff"])
        for ds in dsets:
            for fn, f in res["datasets"][ds]["functionals"].items():
                m = f["mechanism"]
                for S in MECH_S:
                    x = m["per_S"][S]
                    wr.writerow([ds, fn, S, x["n_blocks"], x["I"]["point"], x["C"]["point"], x["G"]["point"], *x["G"]["ci"], x["G_over_I"],
                                 x["SD"]["point"], x["MAD"]["point"], x["wRMS"]["point"], x["rho_bar"], *m["rho_bar_boot_ci"][S], x["K_eff"]])
    # per_patient.csv (primary dataset)
    d = res["datasets"][dsets[0]]; z = np.load(RAW / f"{dsets[0]}.npz"); bt = Boot(z["block_subject"])
    with open(ART / task / "per_patient.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        keys = [k for k in d["functionals"][FUNCS[task][0]]["_E"]]
        wr.writerow(["subject", "n_blocks"] + [f"{fn}_{'heun50' if k[0] == 'heun' else f'K{k[0]}_S{k[1]}'}" for fn in FUNCS[task] for k in keys])
        cols = [np.bincount(bt.inv, None, bt.P)] + [bt.pmean(d["functionals"][fn]["_E"][k]) for fn in FUNCS[task] for k in keys]
        for p in range(bt.P):
            wr.writerow([int(bt.up[p])] + [round(float(c[p]), 6) for c in cols])
    for ds in dsets:
        for f in res["datasets"][ds]["functionals"].values():
            f.pop("_E")
    jd = lambda o: json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else (list(x) if isinstance(x, tuple) else str(x))))  # noqa: E731
    (ART / task / "bootstrap.json").write_text(json.dumps(jd(res), indent=1))
    # console
    for ds in dsets:
        d = res["datasets"][ds]
        print(f"== {ds} ({d['cluster']}, {d['n_subjects']} subjects, {d['n_blocks']} blocks)")
        for fn, f in d["functionals"].items():
            print(f"  {fn}: B32 " + " ".join(f"({K},{S}) {f['cells'][f'K{K}_S{S}']['point']:.3f}" for K, S in B32) +
                  " | B16 " + " ".join(f"({K},{S}) {f['cells'][f'K{K}_S{S}']['point']:.3f}" for K, S in B16) +
                  f" | heun50 {fmt(f['heun50_K1'])} | const {fmt(f['constant_train_median']['error'])}")
            for t, c in f["contrasts"].items():
                print(f"    {t:34s} {fmt(c['diff'])} win {c['win_rate_a_better']:.2f} tie {c['tie_rate']:.2f} {c['significant']}")
            for t, c in f["consensus_checks"].items():
                print(f"    consensus {t:14s} {fmt(c['diff'])}")
            m = f["mechanism"]
            print("    mech " + " | ".join(f"S{S}: G {m['per_S'][S]['G']['point']:.3f} rho {m['per_S'][S]['rho_bar']:.3f} wRMS {m['per_S'][S]['wRMS']['point']:.3f}"
                                         for S in MECH_S) + f" | spearman(rho,G) {m['across_S_spearman']['rho_bar_vs_G']}")
        print("  waveform", {k: (round(v["pearson_r"]["point"], 3), round(v["RMSE"]["point"], 3), round(v["FD"], 3)) for k, v in d["waveform"].items()})
        print("  collapse", d["collapse"])
    print("VERDICT", json.dumps(res["verdict"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["gen", "analyze"]); ap.add_argument("target")
    a = ap.parse_args()
    gen(a.target) if a.stage == "gen" else analyze(a.target)
