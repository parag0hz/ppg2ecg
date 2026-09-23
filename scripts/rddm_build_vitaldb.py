"""RDDM-EXT input builder for VitalDB (docs/RDDM_EXTERNAL_CONSENSUS_PREREGISTRATION.md §2): the exact V1 test windows
(same case, window_index, order as `vm1_evaluate.load("test")`), rebuilt from the raw 500 Hz records with RDDM's paper
preprocessing — finite context of up to ±10 s around each window (start aligned to the 128-Hz grid), polyphase resampling
to 128 Hz, neurokit `ecg_clean(method="neurokit")` / `ppg_clean(method="elgendi")` on the context, then the window's 512
samples. No z-score: RDDM's official per-window min-max (applied later by its `get_datasets`) is invariant to it.
Written under RDDM's file-name convention so the official loader reads it:
outputs/rddm_run/datasets/VITALDB/{ppg,ecg}_test_4sec.npy (+ one-window *_train_* placeholders, + meta_4sec.npz).

Run: .venv/bin/python scripts/rddm_build_vitaldb.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import neurokit2 as nk
import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/VitalDB/cases"
V1 = ROOT / "data/processed/v1_vitaldb"
OUT = ROOT / "outputs/rddm_run/datasets/VITALDB"
FS_RAW, FS, SEG = 500, 128, 4
WIN_RAW, PAD, ALIGN = FS_RAW * SEG, 10 * FS_RAW, 125      # 125 raw samples = 32 samples at 128 Hz


def one(case):
    v = np.load(V1 / f"{case}.npz")
    widx, pid = v["window_index"], int(v["subjectid"])
    with np.load(RAW / f"{case}.npz") as z:
        P, E = z["PLETH"].astype(np.float64), z["ECG_II"].astype(np.float64)
    n = min(len(P), len(E))
    fin = np.isfinite(P[:n]) & np.isfinite(E[:n])
    ppg, ecg, stats = [], [], {"ctx_seconds": []}
    for k in widx:
        s, e = int(k) * WIN_RAW, int(k) * WIN_RAW + WIN_RAW
        assert fin[s:e].all(), (case, k)
        a = max(0, s - PAD)
        bad = np.flatnonzero(~fin[a:s])
        lo = a + bad[-1] + 1 if bad.size else a                # first sample of the finite run ending at s
        lo = s - ((s - lo) // ALIGN) * ALIGN                  # align start to the 128-Hz grid
        b = min(n, e + PAD)
        bad = np.flatnonzero(~fin[e:b])
        hi = e + bad[0] if bad.size else b                    # end of the finite run starting at e
        hi = e + ((hi - e) // ALIGN) * ALIGN
        p = resample_poly(P[lo:hi], 32, 125)
        q = resample_poly(E[lo:hi], 32, 125)
        p = nk.ppg_clean(p, sampling_rate=FS, method="elgendi")
        q = nk.ecg_clean(q, sampling_rate=FS, method="neurokit")
        off = (s - lo) // ALIGN * 32
        ppg.append(p[off:off + FS * SEG]); ecg.append(q[off:off + FS * SEG])
        stats["ctx_seconds"].append((hi - lo) / FS_RAW)
    return case, pid, widx, np.array(ppg, np.float32), np.array(ecg, np.float32), float(np.mean(stats["ctx_seconds"])), v["y"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())["splits"][0]["test"]
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(one, cases, chunksize=4))
    P = np.concatenate([r[3] for r in res]); E = np.concatenate([r[4] for r in res])
    subj = np.concatenate([np.full(len(r[2]), r[1]) for r in res])
    caseid = np.concatenate([np.full(len(r[2]), int(r[0].split("_")[1])) for r in res])
    widx = np.concatenate([r[2] for r in res])
    Y1 = np.concatenate([r[6] for r in res])
    assert np.isfinite(P).all() and np.isfinite(E).all() and len(P) == 19543, (len(P), np.isfinite(P).all(), np.isfinite(E).all())
    # sanity: the rebuilt (high-passed) ECG window should track V1's target ECG of the same window
    zc = lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)  # noqa: E731
    corr = (zc(E.astype(np.float64)) * zc(Y1.astype(np.float64))).mean(1)
    np.save(OUT / "ppg_test_4sec.npy", P); np.save(OUT / "ecg_test_4sec.npy", E)
    np.save(OUT / "ppg_train_4sec.npy", P[:1]); np.save(OUT / "ecg_train_4sec.npy", E[:1])
    np.savez(OUT / "meta_4sec.npz", subject=subj, caseid=caseid, window_index=widx, window_start_s=widx * SEG)
    info = {"n_windows": int(len(P)), "n_patients": int(len(np.unique(subj))), "n_cases": len(cases),
            "mean_context_seconds": float(np.mean([r[5] for r in res])),
            "corr_rebuilt_ecg_vs_v1_target": {"median": float(np.median(corr)), "p5": float(np.percentile(corr, 5)), "min": float(corr.min())}}
    (OUT / "BUILD.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
