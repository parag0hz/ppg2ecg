"""Build data/processed/mimicbp_{L}s/<pid>.npz from raw MIMIC-BP with PENGUIN's MIMIC-BP preprocessing (upstream preprocess.yaml):
PPG band-pass 0.5-4 Hz + per-window z-score + min-max to [-1, 1]; ABP: NO band-pass / z-score / min-max (raw mmHg), both FFT-resampled
125 -> 128 Hz per window. Non-finite or constant windows are DROPPED and counted. Run: .venv/bin/python scripts/build_processed_mimicbp.py"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ppg2ecg.data.mimicbp import FS_RAW, subject_ids, windows_for_subject
from ppg2ecg.data.preprocess import PPG_KW, preprocess_windows

ROOT = Path(__file__).resolve().parents[1]
ABP_KW = dict(bandpass=False, freq_range=(-1, -1), zscore=False, normalize=False)  # upstream MIMIC-BP label settings


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/MIMIC-BP"))
    ap.add_argument("--segment-len", type=int, default=8)
    ap.add_argument("--resample-rate", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    raw = Path(args.raw)
    out = Path(args.out) if args.out else ROOT / "data" / "processed" / f"mimicbp_{args.segment_len}s"
    out.mkdir(parents=True, exist_ok=True)
    files, total, dropped_total = {}, 0, 0
    for pid in subject_ids(raw):
        w = windows_for_subject(raw, pid, args.segment_len)
        finite = np.isfinite(w.ppg).all(axis=1) & np.isfinite(w.abp).all(axis=1)
        std_ok = (w.ppg.std(axis=1) > 0) & (w.abp.std(axis=1) > 0)
        keep = finite & std_ok
        x = preprocess_windows(w.ppg[keep], args.resample_rate, args.segment_len, **PPG_KW).astype(np.float32)
        y = preprocess_windows(w.abp[keep], args.resample_rate, args.segment_len, **ABP_KW).astype(np.float32)
        ok2 = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x, y = x[ok2], y[ok2]
        idx = np.flatnonzero(keep)[ok2]
        p = out / f"{pid}.npz"
        np.savez_compressed(p, x=x, y=y, segment_idx=w.segment_idx[idx], window_start_s=w.window_start_s[idx], label_sbp=w.label_sbp[idx].astype(np.float32), label_dbp=w.label_dbp[idx].astype(np.float32), pid=pid)
        n_drop = int(len(keep) - len(x))
        files[pid] = {"path": str(p.relative_to(ROOT)), "raw_ppg_sha256": sha256(raw / "ppg" / f"{pid}_ppg.npy"), "raw_abp_sha256": sha256(raw / "abp" / f"{pid}_abp.npy"), "n_windows": int(len(x)), "n_dropped_nonfinite_or_constant": n_drop, "fs_raw": FS_RAW, "sha256": sha256(p)}
        total += len(x)
        dropped_total += n_drop
    man = {"built": datetime.now().isoformat(timespec="seconds"), "dataset": "MIMIC-BP", "source": "Harvard Dataverse doi:10.7910/DVN/DBM1NF v2.2", "segment_len_s": args.segment_len, "resample_rate": args.resample_rate, "samples_per_window": args.resample_rate * args.segment_len, "windows_per_30s_segment": 30 // args.segment_len, "ppg_preprocess": PPG_KW, "abp_preprocess": {**ABP_KW, "unit": "mmHg (raw, resampled only)"}, "dtype": "float32", "total_windows": total, "total_dropped": dropped_total, "n_subjects": len(files), "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {out}: {len(files)} subjects, {total} windows ({dropped_total} dropped)")


if __name__ == "__main__":
    main()
