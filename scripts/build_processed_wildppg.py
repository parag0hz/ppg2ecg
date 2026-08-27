"""Build data/processed/wildppg_{L}s/<subject>.npz from raw WildPPG .mat files with PENGUIN's WildPPG preprocessing
(PPG band-pass 0.5-4 Hz, ECG high-pass 0.5 Hz, per-window z-score + min-max; resample to 128*L samples). Records NaN /
constant-window handling explicitly (windows containing non-finite samples in either signal are DROPPED and counted).
Run: .venv/bin/python scripts/build_processed_wildppg.py --segment-len 8 --locations sternum,head,wrist,ankle"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows
from ppg2ecg.data.wildppg import participant_files, windows_for_participant

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/WildPPG"))
    ap.add_argument("--segment-len", type=int, default=8)
    ap.add_argument("--resample-rate", type=int, default=128)
    ap.add_argument("--locations", default="sternum,head,wrist,ankle")
    ap.add_argument("--color", default="g")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ecg-normalization", choices=["window", "none"], default="window", help="A9: 'none' keeps the ECG at the pre-window-normalisation stage (same resample + 0.5 Hz high-pass, NO z-score, NO min-max); PPG is unaffected")
    args = ap.parse_args()
    locs = tuple(args.locations.split(","))
    ecg_kw = dict(ECG_KW) if args.ecg_normalization == "window" else dict(ECG_KW, zscore=False, normalize=False)
    out = Path(args.out) if args.out else ROOT / "data" / "processed" / (f"wildppg_{args.segment_len}s" if args.ecg_normalization == "window" else f"wildppg_{args.segment_len}s_prenorm")
    out.mkdir(parents=True, exist_ok=True)
    files, total, dropped_total = {}, 0, 0
    for f in participant_files(args.raw):
        w = windows_for_participant(f, args.segment_len, locs, args.color)
        finite = np.isfinite(w.ppg).all(axis=1) & np.isfinite(w.ecg).all(axis=1)
        std_ok = (w.ppg.std(axis=1) > 0) & (w.ecg.std(axis=1) > 0)
        keep = finite & std_ok
        x = preprocess_windows(w.ppg[keep], args.resample_rate, args.segment_len, **PPG_KW).astype(np.float32)
        y = preprocess_windows(w.ecg[keep], args.resample_rate, args.segment_len, **ecg_kw).astype(np.float32)
        ok2 = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x, y = x[ok2], y[ok2]
        site, widx = w.site[keep][ok2], w.window_index[keep][ok2]
        p = out / f"{w.subject}.npz"
        np.savez(p, x=x, y=y, site=site, window_index=widx.astype(np.int32), window_start_s=(widx * args.segment_len).astype(np.int32), subject=w.subject)
        n_drop = int((~keep).sum() + (~ok2).sum())
        files[w.subject] = {"path": str(p.relative_to(ROOT)), "raw_file": f.name, "raw_sha256": sha256(f), "n_windows": int(len(x)), "n_dropped_nonfinite_or_constant": n_drop, "fs_ppg": w.fs_ppg, "fs_ecg": w.fs_ecg, "sites": {s: int((site == s).sum()) for s in locs}, "sha256": sha256(p), "notes": w.notes}
        total += len(x)
        dropped_total += n_drop
        print(w.subject, f.name, x.shape, "dropped", n_drop, "fs", w.fs_ppg, w.fs_ecg, flush=True)
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "dataset": "WildPPG", "segment_len_s": args.segment_len, "resample_rate": args.resample_rate, "samples_per_window": args.resample_rate * args.segment_len, "locations": locs, "color": args.color, "ppg_preprocess": PPG_KW, "ecg_preprocess": ecg_kw, "ecg_normalization": args.ecg_normalization, "dtype": "float32", "total_windows": total, "total_dropped": dropped_total, "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"total windows {total} (dropped {dropped_total}) -> {out}/MANIFEST.json")


if __name__ == "__main__":
    main()
