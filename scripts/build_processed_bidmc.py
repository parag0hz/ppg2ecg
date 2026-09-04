"""Build data/processed/bidmc_{L}s/<subject>.npz from the raw BIDMC WFDB records with PENGUIN's WildPPG/DaLiA
preprocessing (PPG band-pass 0.5-4 Hz, ECG high-pass 0.5 Hz, per-window z-score + min-max; FFT-resampled 125 ->
128 Hz, i.e. 128*L samples). PPG = PLETH, ECG = lead II. Records whose .dat is not on disk are skipped and listed in
the manifest. Non-finite / constant windows are DROPPED and counted (same rule as scripts/build_processed_wildppg.py).
Run: .venv/bin/python scripts/build_processed_bidmc.py --segment-len 8"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ppg2ecg.data.bidmc import participant_files, windows_for_participant
from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/BIDMC"))
    ap.add_argument("--segment-len", type=int, default=8)
    ap.add_argument("--resample-rate", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ecg-normalization", choices=["window", "none"], default="window", help="A9: 'none' keeps the ECG at the pre-window-normalisation stage (same resample + 0.5 Hz high-pass, NO z-score, NO min-max); PPG is unaffected")
    args = ap.parse_args()
    raw = Path(args.raw)
    ecg_kw = dict(ECG_KW) if args.ecg_normalization == "window" else dict(ECG_KW, zscore=False, normalize=False)
    out = Path(args.out) if args.out else ROOT / "data" / "processed" / (f"bidmc_{args.segment_len}s" if args.ecg_normalization == "window" else f"bidmc_{args.segment_len}s_prenorm")
    out.mkdir(parents=True, exist_ok=True)
    recs = participant_files(raw)
    no_dat = sorted(p.stem for p in raw.glob("bidmc[0-9][0-9].hea") if not p.with_suffix(".dat").exists())
    files, total, dropped_total = {}, 0, 0
    for f in recs:
        w = windows_for_participant(f, args.segment_len)
        finite = np.isfinite(w.ppg).all(axis=1) & np.isfinite(w.ecg).all(axis=1)
        std_ok = (w.ppg.std(axis=1) > 0) & (w.ecg.std(axis=1) > 0)
        keep = finite & std_ok
        x = preprocess_windows(w.ppg[keep], args.resample_rate, args.segment_len, **PPG_KW).astype(np.float32)
        y = preprocess_windows(w.ecg[keep], args.resample_rate, args.segment_len, **ecg_kw).astype(np.float32)
        ok2 = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x, y = x[ok2], y[ok2]
        widx = w.window_index[keep][ok2]
        p = out / f"{w.subject}.npz"
        np.savez(p, x=x, y=y, window_index=widx.astype(np.int32), window_start_s=(widx * args.segment_len).astype(np.int32), subject=w.subject)
        n_drop = int((~keep).sum() + (~ok2).sum())
        files[w.subject] = {"path": str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p), "raw_file": f"{f.name}.dat", "raw_sha256": sha256(f.with_suffix(".dat")), "raw_header_sha256": sha256(f.with_suffix(".hea")), "n_windows": int(len(x)), "n_dropped_nonfinite_or_constant": n_drop, "fs_ppg": w.fs_ppg, "fs_ecg": w.fs_ecg, "sha256": sha256(p), "notes": w.notes}
        total += len(x)
        dropped_total += n_drop
        print(w.subject, f"{f.name}.dat", x.shape, "dropped", n_drop, "fs", w.fs_ppg, w.fs_ecg, flush=True)
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "dataset": "BIDMC", "source": "PhysioNet BIDMC PPG and Respiration Dataset doi:10.13026/C2208R", "segment_len_s": args.segment_len, "resample_rate": args.resample_rate, "samples_per_window": args.resample_rate * args.segment_len, "ppg_signal": "PLETH", "ecg_signal": "II", "ppg_preprocess": PPG_KW, "ecg_preprocess": ecg_kw, "ecg_normalization": args.ecg_normalization, "dtype": "float32", "n_subjects": len(files), "total_windows": total, "total_dropped": dropped_total, "records_without_signal_file": no_dat, "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"total windows {total} (dropped {dropped_total}) over {len(files)} subjects -> {out}/MANIFEST.json")


if __name__ == "__main__":
    main()
