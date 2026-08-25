"""Build our processed dataset (PENGUIN-faithful preprocessing, ppg2ecg.data) for a given window length.
Output: data/processed/v0_{L}s/S{n}.npz (x, y float32 [n, 128*L]; window_start_s) + MANIFEST.json with sha256 per file.
Run: .venv/bin/python scripts/build_processed_v0.py --segment-len 8"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ppg2ecg.data.dalia import BVP_FS, ECG_FS, SUBJECTS, load_subject_raw, windows_for_subject
from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment-len", type=int, default=8)
    ap.add_argument("--resample-rate", type=int, default=128)
    ap.add_argument("--raw", default=str(ROOT / "data/raw"))
    args = ap.parse_args()
    out = ROOT / "data" / "processed" / f"v0_{args.segment_len}s"
    out.mkdir(parents=True, exist_ok=True)
    files, total = {}, 0
    for s in SUBJECTS:
        raw = load_subject_raw(args.raw, s)
        w = windows_for_subject(raw, args.segment_len, align="strict")
        x = preprocess_windows(w.ppg, args.resample_rate, args.segment_len, **PPG_KW).astype(np.float32)
        y = preprocess_windows(w.ecg, args.resample_rate, args.segment_len, **ECG_KW).astype(np.float32)
        assert np.isfinite(x).all() and np.isfinite(y).all(), s
        p = out / f"{s}.npz"
        np.savez(p, x=x, y=y, window_start_s=w.window_start_s.astype(np.int32), subject=s)
        files[s] = {"path": str(p.relative_to(ROOT)), "n_windows": int(len(x)), "sha256": sha256(p), "raw_sha256": sha256(Path(raw.path))}
        total += len(x)
        print(s, x.shape, y.shape)
    manifest = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "segment_len_s": args.segment_len, "resample_rate": args.resample_rate, "samples_per_window": args.resample_rate * args.segment_len,
        "native_fs": {"ppg": BVP_FS, "ecg": ECG_FS}, "ppg_preprocess": PPG_KW, "ecg_preprocess": ECG_KW,
        "dtype": "float32", "total_windows": total, "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"total windows: {total} -> {out}/MANIFEST.json")


if __name__ == "__main__":
    main()
