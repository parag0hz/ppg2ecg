"""ED2 part B data: one contiguous 2-minute block per V1 test case (docs/ED2_…_PREREGISTRATION.md)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw/VitalDB/cases", ROOT / "outputs/ed2_hrv_blocks.npz"
FS_RAW, FS, SEG, N_WIN, TRIES = 500, 128, 4, 30, 10


def one(case):
    with np.load(RAW / f"{case}.npz") as z:
        ppg, ecg = z["PLETH"], z["ECG_II"]
        win = FS_RAW * SEG
        n = min(ppg.size, ecg.size) // win
        start = n // 2
        for _ in range(TRIES + 1):
            if start + N_WIN > n:
                return case, None
            P = ppg[start * win:(start + N_WIN) * win].reshape(N_WIN, win).astype(np.float64)
            E = ecg[start * win:(start + N_WIN) * win].reshape(N_WIN, win).astype(np.float64)
            if np.isfinite(P).all() and np.isfinite(E).all() and (P.std(1) > 0).all() and (E.std(1) > 0).all():
                x = preprocess_windows(P, FS, SEG, **PPG_KW); y = preprocess_windows(E, FS, SEG, **ECG_KW)
                if np.isfinite(x).all() and np.isfinite(y).all():
                    return case, (x.astype(np.float32), y.astype(np.float32), start)
            start += N_WIN
    return case, None


def main():
    man = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())
    cases, pat = man["splits"][0]["test"], man["extra"]["patient_of_case"]
    X, Y, C, PID, skipped = [], [], [], [], []
    with ProcessPoolExecutor(12) as ex:
        for i, (case, r) in enumerate(ex.map(one, cases, chunksize=4), 1):
            if r is None:
                skipped.append(case); continue
            X.append(r[0]); Y.append(r[1]); C += [case] * N_WIN; PID += [int(pat[case])] * N_WIN
            if i % 300 == 0:
                print(f"[ed2-build] {i}/{len(cases)}", flush=True)
    np.savez(OUT, x=np.concatenate(X), y=np.concatenate(Y), case=np.array(C), pid=np.array(PID), skipped=np.array(skipped))
    print(f"[ed2-build] blocks {len(X)} skipped {len(skipped)} windows {len(C)}", flush=True)


if __name__ == "__main__":
    main()
