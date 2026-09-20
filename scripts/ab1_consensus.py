"""AB1 — consensus ablation: per-sample HR matrices, pooling rules, cost-matched curve (no training)."""
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
import cd1_evaluate as CDE  # noqa: E402
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402

OUT, RAW = ROOT / "artifacts/ab1_consensus", ROOT / "outputs/ab1_raw"
PLAN = [("iMF", "I", 1, 32), ("iMF", "I", 2, 16), ("CD", "D", 1, 32), ("PENGUIN", "C", 1, 32), ("PENGUIN", "C", 50, 4)]
KS, RULES = (1, 2, 4, 8, 16, 32), ("median", "mean", "trim20")
RUN = {"C": "outputs/v1_vitaldb_armC_seed42", "I": "outputs/v1_vitaldb_armI_seed42", "D": "outputs/cd1_vitaldb_armD_seed42"}


def pool(h, rule):
    with np.errstate(all="ignore"):
        if rule == "median":
            return np.nanmedian(h, 0)
        if rule == "mean":
            return np.nanmean(h, 0)
        k = h.shape[0]
        if k < 5:
            return np.nanmedian(h, 0)
        lo, hi = int(np.floor(0.2 * k)), int(np.ceil(0.8 * k))
        return np.nanmean(np.sort(h, 0)[lo:hi], 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    X, Y, pid = VM.load("test")
    ex = ProcessPoolExecutor(8)
    ref_hr = V.hr_batch(ex, Y)
    rows = []
    for name, arm, nfe, kmax in PLAN:
        f = RAW / f"hr_{arm}{nfe}.npy"
        if f.exists():
            H = np.load(f)
        else:
            if arm == "D":
                ck = torch.load(ROOT / RUN[arm] / "checkpoint_last.pt", map_location="cpu", weights_only=False)
                net = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval(); net.load_state_dict(ck["state_dict"])
                gen = lambda s: CDE.sample(net, X, s, nfe, dev)[0]  # noqa: E731
            else:
                net = U2.build(ROOT / RUN[arm] / "checkpoint_last.pt", dev)[0]
                def gen(s, _net=net, _arm=arm, _nfe=nfe):
                    e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(s))
                    return U2.generate(_net, X, e0, _nfe, dev, _arm)[0]
            H = np.full((kmax, len(X)), np.nan, np.float32)
            for s in range(kmax):
                H[s] = V.hr_batch(ex, gen(s))
                print(f"[ab1] {name} NFE {nfe} draw {s + 1}/{kmax}", flush=True)
            np.save(f, H)
            del net; torch.cuda.empty_cache()
        for K in [k for k in KS if k <= kmax]:
            for rule in RULES:
                err = np.abs(pool(H[:K], rule) - ref_hr)
                mu, lo, hi = V.cluster_ci(err, pid)
                rows.append(dict(arm=name, nfe=nfe, K=K, rule=rule, total_nfe=K * nfe, hr_err=mu, ci_lo=lo, ci_hi=hi))
                print(f"[ab1] {name} NFE {nfe} K {K:2d} {rule:7s} HR {mu:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    sat = {}
    for name, arm, nfe, kmax in PLAN:
        med = {r["K"]: r["hr_err"] for r in rows if r["arm"] == name and r["nfe"] == nfe and r["rule"] == "median"}
        sat[f"{name}-{nfe}"] = next((K for K in KS if 2 * K in med and abs(med[2 * K] - med[K]) < 0.1), None)
    with open(OUT / "curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps({"saturation_K_median_rule": sat, "n_patients": int(len(np.unique(pid))),
                                                  "n_windows": int(len(X))}, indent=1))
    print(json.dumps(sat, indent=1))


if __name__ == "__main__":
    main()
