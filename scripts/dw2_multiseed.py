"""DW2 part A — three fixed allocations at B = 32, three seeds, three models (docs/DW2_…_PREREGISTRATION.md)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dw1_depth_width as D  # noqa: E402
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402

OUT, RAW = ROOT / "artifacts/dw2_multiseed", ROOT / "outputs/dw2_raw"
CK = {("I", 42): "outputs/v1_vitaldb_armI_seed42", ("I", 1): "outputs/sr1_I_seed1", ("I", 2): "outputs/sr1_I_seed2",
      ("C", 42): "outputs/v1_vitaldb_armC_seed42", ("C", 1): "outputs/sr1_C_seed1", ("C", 2): "outputs/sr1_C_seed2",
      ("D", 42): "outputs/cd1_vitaldb_armD_seed42", ("D", 1): "outputs/dw2_D_seed1", ("D", 2): "outputs/dw2_D_seed2"}
CONDS = {"I": [("width", 32, 1), ("best", 16, 2), ("depth", 1, 32)],
         "D": [("width", 32, 1), ("balanced", 16, 2), ("depth", 1, 32)],
         "C": [("width", 32, 1), ("best", 8, 4), ("depth", 1, 32)]}
SEEDS = (42, 1, 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); ex = ProcessPoolExecutor(8)
    X, Y, pid = VM.load("test")
    ref = V.hr_batch(ex, Y)
    subs = np.unique(pid)
    pp = lambda v: np.array([np.nanmean(v[pid == s]) for s in subs])  # noqa: E731
    err, rows = {}, []
    for arm in ("I", "D", "C"):
        for seed in SEEDS:
            D.CK[arm] = CK[(arm, seed)]                       # point the DW1 sampler at this seed's checkpoint
            sampler = D.make_sampler(arm, dev)
            for label, K, S in CONDS[arm]:
                f = RAW / f"hr_{arm}_seed{seed}_K{K}_S{S}.npy"
                if f.exists():
                    H = np.load(f)
                elif seed == 42 and (ROOT / f"outputs/dw1_raw/hr_{arm}{S}.npy").exists() and np.load(ROOT / f"outputs/dw1_raw/hr_{arm}{S}.npy").shape[0] >= K:
                    H = np.load(ROOT / f"outputs/dw1_raw/hr_{arm}{S}.npy")[:K]; np.save(f, H)
                else:
                    H = np.stack([V.hr_batch(ex, sampler(X, s, S)) for s in range(K)]); np.save(f, H)
                with np.errstate(all="ignore"):
                    e = np.abs(np.nanmedian(H, 0) - ref)
                err[(arm, seed, label)] = e
                mu, lo, hi = V.cluster_ci(e, pid)
                rows.append(dict(model=D.NAME[arm], seed=seed, allocation=label, K=K, S=S, hr_err=mu, ci_lo=lo, ci_hi=hi))
                print(f"[dw2-A] {D.NAME[arm]:26s} seed {seed:2d} {label:8s} (K {K:2d}, S {S:2d})  HR {mu:.3f}", flush=True)
            del sampler; torch.cuda.empty_cache()
    res = {}
    for arm in ("I", "D", "C"):
        labels = [c[0] for c in CONDS[arm]]
        r = {"per_seed": {}, "mean_sd": {}, "diff_vs_depth": {}, "win_rate_vs_depth": {}}
        for label in labels:
            vals = [V.cluster_ci(err[(arm, s, label)], pid)[0] for s in SEEDS]
            r["per_seed"][label] = dict(zip(map(str, SEEDS), vals)); r["mean_sd"][label] = [float(np.mean(vals)), float(np.std(vals, ddof=1))]
            if label != "depth":
                r["diff_vs_depth"][label] = {str(s): V.cluster_ci(err[(arm, s, label)] - err[(arm, s, "depth")], pid) for s in SEEDS}
                r["win_rate_vs_depth"][label] = {str(s): float(np.mean(pp(err[(arm, s, label)]) < pp(err[(arm, s, "depth")]))) for s in SEEDS}
        r["width_beats_depth_seeds"] = sum(r["diff_vs_depth"]["width"][str(s)][2] < 0 for s in SEEDS)
        mid = labels[1]
        r["mid_beats_depth_seeds"] = sum(r["diff_vs_depth"][mid][str(s)][2] < 0 for s in SEEDS)
        r["best_allocation_per_seed"] = {str(s): min(labels, key=lambda l: V.cluster_ci(err[(arm, s, l)], pid)[0]) for s in SEEDS}
        res[D.NAME[arm]] = r
    with open(OUT / "grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    for m, r in res.items():
        print(m, "| width>depth in", r["width_beats_depth_seeds"], "/3 seeds | mid>depth", r["mid_beats_depth_seeds"], "/3 | best per seed", r["best_allocation_per_seed"],
              "| mean±sd", {k: [round(x, 3) for x in v] for k, v in r["mean_sd"].items()})


if __name__ == "__main__":
    main()
