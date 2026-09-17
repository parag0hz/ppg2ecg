"""V1 evaluation on VitalDB test patients (docs/V1_VITALDB_PAIRED_PREREGISTRATION.md). Reuses U2 helpers."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import u2_evaluate as U2  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402

SLUG, FS = "v1_vitaldb", 128
OUT = ROOT / "artifacts/v1_vitaldb"
NFES = {"C": (1, 2, 4, 50), "I": (1, 2, 4)}
DRAWS, K = 4, 16
MARGIN = {"HR": 1.0, "Rpeak_F1": -0.02}


def _hr(row):
    return R.hr_bpm(R.detect_rpeaks(row, FS, "neurokit"), FS)


def hr_batch(ex, Z):
    return np.array(list(ex.map(_hr, list(Z), chunksize=256)))


def ppg_hr(x):
    return R.hr_bpm(find_peaks(x, distance=42, prominence=0.3)[0], FS)


def cluster_ci(d, pid):
    """Patient-clustered bootstrap of the mean of per-window values d (equal patient weight)."""
    subs = np.unique(pid)
    per = np.array([np.nanmean(d[pid == s]) for s in subs])
    rng = np.random.default_rng(U2.BOOT_SEED)
    draws = np.array([np.nanmean(per[rng.integers(0, len(subs), len(subs))]) for _ in range(U2.BOOT_N)])
    return float(np.nanmean(per)), float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def main():
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((ROOT / f"data/manifests/split_{SLUG}_seed42.json").read_text())
    split = man["splits"][0]
    X, Y, P = [], [], []
    for c in split["test"]:
        d = np.load(ROOT / f"data/processed/{SLUG}/{c}.npz")
        X.append(d["x"]); Y.append(d["y"].astype(np.float64)); P.append(np.full(len(d["x"]), int(d["subjectid"])))
    X, Y, pid = np.concatenate(X), np.concatenate(Y), np.concatenate(P)
    print(f"[v1] test cases={len(split['test'])} patients={len(np.unique(pid))} windows={len(X)}", flush=True)
    nets, cks = {}, {}
    for arm in ("C", "I"):
        p = ROOT / f"outputs/{SLUG}_arm{arm}_seed42/checkpoint_last.pt"
        net, ck, kind = U2.build(p, dev)
        assert kind == arm and int(ck["train_state"]["opt_steps"]) == 14000, (p, kind)
        nets[arm], cks[arm] = net, {"path": str(p.relative_to(ROOT)), "sha256": U2.sha256(p)}

    ex = ProcessPoolExecutor(16)
    ref_hr = hr_batch(ex, Y)
    acc, extras, gen_s, hr_samples = {}, {}, {}, {}
    plan = [(a, n, DRAWS) for a in ("C", "I") for n in NFES[a]] + [("I", 1, K), ("I", 2, K)]
    for arm, nfe, n_seeds in plan:
        for seed in range(n_seeds):
            if (arm, nfe, seed) in hr_samples:
                continue
            e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(seed))
            pred, dt = U2.generate(nets[arm], X, e0, nfe, dev, arm)
            gen_s.setdefault((arm, nfe), []).append(dt)
            hr_samples[(arm, nfe, seed)] = hr_batch(ex, pred)
            if seed < DRAWS:
                mt = U2.task_metrics("ECG", 8, pred, Y, pid)
                tab = mt.pop("_counts")[0]
                for m, (v, _) in mt.items():
                    acc.setdefault((arm, nfe, m), []).append(np.asarray(v, float))
                for m, v in U2.pooled_extras("ECG", pred, Y, tab).items():
                    extras.setdefault((arm, nfe, m), []).append(v)
            print(f"[v1] {arm}{nfe} seed {seed} gen {dt:.1f}s", flush=True)

    rows = []
    for (arm, nfe, m), draws in sorted(acc.items()):
        mu, lo, hi = cluster_ci(np.nanmean(np.vstack(draws), 0), pid)
        rows.append(dict(arm=arm, nfe=nfe, K=1, metric=m, value=mu, ci_lo=lo, ci_hi=hi))
    for (arm, nfe, m), v in sorted(extras.items()):
        if m != "_fd_n":
            rows.append(dict(arm=arm, nfe=nfe, K=1, metric=m, value=float(np.mean(v)), ci_lo=np.nan, ci_hi=np.nan))
    cons = {}
    for arm, nfe, k in (("I", 1, K), ("I", 2, K), ("C", 50, DRAWS)):
        with np.errstate(all="ignore"):
            e = np.abs(np.nanmedian(np.vstack([hr_samples[(arm, nfe, s)] for s in range(k)]), 0) - ref_hr)
        cons[(arm, nfe, k)] = e
        rows.append(dict(arm=arm, nfe=nfe, K=k, metric="HR_consensus", value=cluster_ci(e, pid)[0],
                         ci_lo=cluster_ci(e, pid)[1], ci_hi=cluster_ci(e, pid)[2]))
    e_ppg = np.abs(np.array([ppg_hr(x) for x in X]) - ref_hr)
    rows.append(dict(arm="PPG-only", nfe=0, K=1, metric="HR", value=cluster_ci(e_ppg, pid)[0],
                     ci_lo=cluster_ci(e_ppg, pid)[1], ci_hi=cluster_ci(e_ppg, pid)[2]))
    for (arm, nfe), t in gen_s.items():
        rows.append(dict(arm=arm, nfe=nfe, K=1, metric="ms_per_window", value=1000 * np.mean(t) / len(X), ci_lo=np.nan, ci_hi=np.nan))

    # H1: non-inferiority with one-sided gate
    c50 = {m: np.nanmean(np.vstack(acc[("C", 50, m)]), 0) for m in MARGIN}
    c1 = {m: np.nanmean(np.vstack(acc[("C", 1, m)]), 0) for m in MARGIN}
    h1 = []
    for k in NFES["I"]:
        cell = {"k": k}
        for m, dl in MARGIN.items():
            ik = np.nanmean(np.vstack(acc[("I", k, m)]), 0)
            pt, lo, hi = cluster_ci(ik - c50[m], pid)
            span = abs(cluster_ci(c1[m], pid)[0] - cluster_ci(c50[m], pid)[0])
            ok = hi < dl if dl > 0 else lo > dl
            cell[m] = dict(diff_I_minus_C50=pt, ci_lo=lo, ci_hi=hi, margin=dl, armC_span=span,
                           pass_=bool(ok), gate_withheld=bool(ok and span < abs(dl)))
        cell["non_inferior"] = all(cell[m]["pass_"] and not cell[m]["gate_withheld"] for m in MARGIN)
        h1.append(cell)
    first = next((c["k"] for c in h1 if c["non_inferior"]), None)
    c50_single_hr = c50["HR"]
    pt, lo, hi = cluster_ci(cons[("I", 1, K)] - c50_single_hr, pid)
    verdict = {"H1": f"NON-INFERIOR at k={first}" if first else "NOT NON-INFERIOR", "H1_cells": h1,
               "H2": "SUPERIOR" if hi < 0 else "NOT SUPERIOR",
               "H2_diff_consensusI1K16_minus_C50single": dict(point=pt, ci_lo=lo, ci_hi=hi),
               "n_patients": int(len(np.unique(pid))), "n_windows": int(len(X)), "checkpoints": cks,
               "ref_hr_nan": int(np.isnan(ref_hr).sum())}
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: verdict[k] for k in ("H1", "H2", "H2_diff_consensusI1K16_minus_C50single")}, indent=1))
    for r in rows:
        print(f"{r['arm']:>8} {r['nfe']:>2} K={r['K']:<2} {r['metric']:<16} {r['value']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")


if __name__ == "__main__":
    main()
