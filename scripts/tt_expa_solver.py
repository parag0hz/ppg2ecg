"""EXP-A — PENGUIN solver fairness: Euler vs upstream-exact Heun depth/width grids at exact NFE budgets
(docs/TOP_TIER_COMPLETION_PREREGISTRATION.md, EXP-A). No training.

Budget = network function evaluations: Euler step = 1 NFE, Heun step = 2 NFE (`samplers.nfe_of`). Seeds 42 / 1 / 2 PENGUIN
checkpoints, VitalDB V1 test, HR median over K samples, noise seeds 0 … K−1 (DW1 convention; Euler cells reuse the DW1 / DW2
caches, which used the same generator, sampler and checkpoints).

Run: .venv/bin/python scripts/tt_expa_solver.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation.rpeaks import detect_rpeaks  # noqa: E402
from ppg2ecg.flow.samplers import euler_sample, heun_sample, nfe_of  # noqa: E402

OUT, RAW = ROOT / "artifacts/tt_expa_solver", ROOT / "outputs/tt_expa_raw"
CK = {42: "outputs/v1_vitaldb_armC_seed42", 1: "outputs/sr1_C_seed1", 2: "outputs/sr1_C_seed2"}
SEEDS, BS, FS = (42, 1, 2), 256, 128
GRID = {32: {"heun": [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1)], "euler": [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]},
        50: {"heun": [(1, 25), (5, 5), (25, 1)], "euler": [(1, 50), (5, 10), (10, 5), (25, 2), (50, 1)]}}
REF_WIDTH = ("euler", 8, 4)                                    # preregistered width-heavy arm (DW1 / DW2-A), not re-selected


def _peaks(sig):
    return np.asarray(detect_rpeaks(sig, FS), int)


def make_sampler(seed, dev):
    net = U2.build(ROOT / CK[seed] / "checkpoint_last.pt", dev)[0]

    @torch.no_grad()
    def sample(X, noise_seed, solver, steps):
        g = torch.Generator().manual_seed(noise_seed)
        z0 = torch.randn(len(X), 1, X.shape[1], generator=g)        # identical to dw1_depth_width.make_sampler (arm C)
        out, t0 = [], time.perf_counter()
        for i in range(0, len(X), BS):
            ppg = torch.from_numpy(X[i:i + BS]).to(dev).unsqueeze(1)
            v = lambda a, t, _p=ppg: net.forward_step(a, _p, t)  # noqa: E731
            x, k = (heun_sample if solver == "heun" else euler_sample)(v, z0[i:i + BS].to(dev), steps)
            assert int(k) == nfe_of(solver, steps), (solver, steps, k)
            out.append(x[:, 0].float().cpu().numpy())
        torch.cuda.synchronize()
        return np.concatenate(out).astype(np.float64), 1000 * (time.perf_counter() - t0) / len(X)
    return sample


def seeded_cache(seed, solver, steps):
    """Existing Euler HR rows (noise seeds 0 … n−1) from DW1 (seed 42) / DW2-A (seeds 1, 2); None otherwise."""
    if solver != "euler":
        return None
    cands = [ROOT / f"outputs/dw1_raw/hr_C{steps}.npy"] if seed == 42 else []
    cands += [p for p in (ROOT / "outputs/dw2_raw").glob(f"hr_C_seed{seed}_K*_S{steps}.npy")]
    rows = [np.load(p) for p in cands if p.exists()]
    return max(rows, key=len) if rows else None


def hr_rows(sampler, ex, X, seed, solver, steps, K, lat):
    f = RAW / f"hr_seed{seed}_{solver}{steps}.npy"
    H = np.load(f) if f.exists() else seeded_cache(seed, solver, steps)
    H = np.zeros((0, len(X))) if H is None else H
    while H.shape[0] < K:
        w, ms = sampler(X, H.shape[0], solver, steps)
        lat.setdefault((seed, solver, steps), []).append(ms)
        H = np.vstack([H, V.hr_batch(ex, w)[None]])
        np.save(f, H)
    return H[:K]


def main():
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); ex = ProcessPoolExecutor(8)
    X, Y, pid = VM.load("test")
    ref = V.hr_batch(ex, Y)
    ref_peaks = list(ex.map(_peaks, list(Y), chunksize=256))
    subs = np.unique(pid)
    pp = lambda v: np.array([np.nanmean(v[pid == s]) for s in subs])  # noqa: E731
    ci = lambda v: V.cluster_ci(v, pid)  # noqa: E731
    err, rows, lat, quality = {}, [], {}, []
    for B in GRID:                                              # primary budget (32) for every seed first, then 50
        for seed in SEEDS:
            sampler = make_sampler(seed, dev)
            for solver, cells in GRID[B].items():
                for K, steps in cells:
                    H = hr_rows(sampler, ex, X, seed, solver, steps, K, lat)
                    with np.errstate(all="ignore"):
                        e = np.abs(np.nanmedian(H, 0) - ref)
                    err[(seed, B, solver, K, steps)] = e
                    mu, lo, hi = ci(e)
                    ms = float(np.mean(lat[(seed, solver, steps)])) if (seed, solver, steps) in lat else np.nan
                    rows.append(dict(seed=seed, B=B, solver=solver, K=K, steps=steps, nfe_per_sample=nfe_of(solver, steps),
                                     total_nfe=K * nfe_of(solver, steps), hr_err=mu, ci_lo=lo, ci_hi=hi, ms_per_sample_gpu=ms,
                                     ms_per_window_gpu=K * ms if np.isfinite(ms) else np.nan))
                    print(f"[expA] seed {seed:2d} B {B} {solver:5s} (K {K:2d}, steps {steps:2d}, NFE/sample {nfe_of(solver, steps):2d})  HR {mu:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
            del sampler; torch.cuda.empty_cache()
    sampler = make_sampler(42, dev)                             # single-sample quality by depth, noise seed 0 (as DW1)
    for solver, steps_list in (("heun", (1, 2, 4, 5, 8, 16, 25)), ("euler", (1, 2, 4, 5, 8, 10, 16, 32, 50))):
        for steps in steps_list:
            w, _ = sampler(X, 0, solver, steps)
            hyp_peaks = list(ex.map(_peaks, list(w), chunksize=256))
            prf = PMX.rpeak_prf_at(w, Y, FS, 50.0, peaks=(ref_peaks, hyp_peaks))
            beat = PMX.beat_level_metrics(w, Y, FS, 50.0, peaks=(ref_peaks, hyp_peaks))
            i = U2.fd_subset(len(w))
            quality.append(dict(solver=solver, steps=steps, nfe=nfe_of(solver, steps), f1=ci(prf["rpeak_f1"])[0],
                                hr_single=ci(beat["hr_abs_err"])[0], fd=float(PMX.kanflow_fd(w[i], Y[i]))))
            print(f"[expA] quality seed 42 {solver:5s} steps {steps:2d} NFE {quality[-1]['nfe']:2d}  F1 {quality[-1]['f1']:.4f}  FD {quality[-1]['fd']:.2f}  HR(1 draw) {quality[-1]['hr_single']:.3f}", flush=True)
    del sampler; torch.cuda.empty_cache()

    def contrast(seed, a, b):
        d = err[(seed,) + a] - err[(seed,) + b]
        return {"diff": ci(d), "win_rate": float(np.mean(pp(err[(seed,) + a]) < pp(err[(seed,) + b])))}
    hyp = {}
    for seed in SEEDS:
        h = {"A-H1": {"(16,H1)-(1,H16)": contrast(seed, (32, "heun", 16, 1), (32, "heun", 1, 16)),
                      "(8,H2)-(1,H16)": contrast(seed, (32, "heun", 8, 2), (32, "heun", 1, 16))},
             "A-H2": {"(8,E4)-(1,H16)": contrast(seed, (32, "euler", 8, 4), (32, "heun", 1, 16)),
                      "(8,E4)-(1,E32)": contrast(seed, (32, "euler", 8, 4), (32, "euler", 1, 32))},
             "A-H3": {"(5,H5)-(1,H25)": contrast(seed, (50, "heun", 5, 5), (50, "heun", 1, 25)),
                      "(25,H1)-(1,H25)": contrast(seed, (50, "heun", 25, 1), (50, "heun", 1, 25))},
             "solver_recovery": {"(1,H16)-(1,E32)": contrast(seed, (32, "heun", 1, 16), (32, "euler", 1, 32)),
                                 "(1,H25)-(1,E50)": contrast(seed, (50, "heun", 1, 25), (50, "euler", 1, 50)),
                                 "(16,H1)-(16,E2)": contrast(seed, (32, "heun", 16, 1), (32, "euler", 16, 2)),
                                 "(8,H2)-(8,E4)": contrast(seed, (32, "heun", 8, 2), (32, "euler", 8, 4))},
             "shipped_vs_ref_width": {"(8,E4)-(1,H25)": contrast(seed, (32, "euler", 8, 4), (50, "heun", 1, 25))}}
        for name in ("A-H1", "A-H2", "A-H3"):
            h[name]["holds"] = all(v["diff"][2] < 0 for k, v in h[name].items() if k != "holds")
        hyp[str(seed)] = h
    verdict = {name: {"seeds_holding": sum(hyp[str(s)][name]["holds"] for s in SEEDS), "success": all(hyp[str(s)][name]["holds"] for s in SEEDS)}
               for name in ("A-H1", "A-H2", "A-H3")}
    best = {}
    for seed in SEEDS:
        for B in GRID:
            cells = {f"{sv}({K},{st})": ci(err[(seed, B, sv, K, st)])[0] for sv in GRID[B] for K, st in GRID[B][sv]}
            best[f"seed{seed}|B={B}"] = {"exploratory_best_cell": min(cells, key=cells.get), "cells": cells}
    with open(OUT / "grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "result.json").write_text(json.dumps({"verdict": verdict, "hypotheses": hyp, "exploratory_grid": best,
                                                  "single_sample_quality_seed42": quality}, indent=1))
    print(json.dumps(verdict, indent=1))
    for seed in SEEDS:
        for name, d in hyp[str(seed)].items():
            print(seed, name, {k: ([round(x, 3) for x in v["diff"]], round(v["win_rate"], 3)) if isinstance(v, dict) else v for k, v in d.items()})


if __name__ == "__main__":
    main()
