"""Build data/processed/capnobase_{L}s/<subject>.npz from the raw CapnoBase TBME-RR .mat files with the frozen project
preprocessing (PPG band-pass 0.5-4 Hz, ECG high-pass 0.5 Hz, per-window z-score + min-max, FFT-resample 300 -> 128 Hz),
mirroring scripts/build_processed_wildppg.py. Non-finite / constant windows are DROPPED and counted, as everywhere else.

--artifact-screen selects what to do with the expert artifact intervals; the default `none` matches every other corpus in
this project (which applies no screening). The shipped peak labels are themselves unscreened -- 177 pleth / 7 ECG expert
peaks fall strictly inside an artifact interval -- so the audit counts are written into the MANIFEST either way.

Artifact-label coverage of the 42 shipped files, re-measured 2026-09-04 over every file: 9 carry ECG intervals, 19 carry
pleth intervals, 7 carry both, hence 21 carry at least one and 21 carry NEITHER (an earlier build report said 14; that was
wrong). The same counts are recomputed at build time and emitted as MANIFEST["artifact_label_coverage"].
Run: .venv/bin/python scripts/build_processed_capnobase.py            # -> data/processed/capnobase_8s
     .venv/bin/python scripts/build_processed_capnobase.py --artifact-screen drop_any   # -> .../capnobase_8s_clean"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from ppg2ecg.data.capnobase import participant_files, windows_for_participant
from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = {"none": "", "drop_ecg": "_ecgclean", "drop_any": "_clean"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/CapnoBase/files"))
    ap.add_argument("--segment-len", type=int, default=8)
    ap.add_argument("--resample-rate", type=int, default=128)
    ap.add_argument("--artifact-screen", choices=["none", "drop_ecg", "drop_any"], default="none", help="none: keep every window (project default); drop_ecg: drop windows overlapping an ECG artifact interval; drop_any: drop windows overlapping an ECG OR pleth artifact interval")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "data" / "processed" / f"capnobase_{args.segment_len}s{SUFFIX[args.artifact_screen]}"
    out.mkdir(parents=True, exist_ok=True)
    files, total, dropped_total, screened_total, audit_ecg, audit_ppg = {}, 0, 0, 0, 0, 0
    has_iv = {"ecg": 0, "pleth": 0, "both": 0, "any": 0, "neither": 0}  # per-FILE artifact-label coverage
    for f in participant_files(args.raw):
        w = windows_for_participant(f, args.segment_len)
        e, p_ = len(w.ecg_artifact_intervals) > 0, len(w.ppg_artifact_intervals) > 0
        has_iv["ecg"] += e
        has_iv["pleth"] += p_
        has_iv["both"] += e and p_
        has_iv["any"] += e or p_
        has_iv["neither"] += not (e or p_)
        screen = {"none": np.zeros(len(w.ecg), bool), "drop_ecg": w.ecg_artifact_mask, "drop_any": w.ecg_artifact_mask | w.ppg_artifact_mask}[args.artifact_screen]
        finite = np.isfinite(w.ppg).all(axis=1) & np.isfinite(w.ecg).all(axis=1)
        std_ok = (w.ppg.std(axis=1) > 0) & (w.ecg.std(axis=1) > 0)
        keep = ~screen & finite & std_ok
        x = preprocess_windows(w.ppg[keep], args.resample_rate, args.segment_len, **PPG_KW).astype(np.float32)
        y = preprocess_windows(w.ecg[keep], args.resample_rate, args.segment_len, **ECG_KW).astype(np.float32)
        ok2 = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x, y = x[ok2], y[ok2]
        widx = w.window_index[keep][ok2]
        n_rpeaks = np.array([len(p) for p in w.ecg_peaks_in_window], dtype=np.int32)[keep][ok2]
        p = out / f"{w.subject}.npz"
        np.savez(p, x=x, y=y, window_index=widx.astype(np.int32), window_start_s=(widx * args.segment_len).astype(np.int32), subject=w.subject, n_expert_rpeaks=n_rpeaks)
        n_screened = int(screen.sum())
        n_drop = int((~screen & ~(finite & std_ok)).sum() + (~ok2).sum())  # drop rule counted on the windows screening kept
        # path is os.path.relpath, not Path.relative_to: --out may sit outside the repo
        files[w.subject] = {"path": os.path.relpath(p, ROOT), "raw_file": f.name, "raw_sha256": sha256(f), "n_windows": int(len(x)), "n_dropped_nonfinite_or_constant": n_drop, "n_screened_artifact": n_screened, "n_ecg_artifact_windows": int(w.ecg_artifact_mask.sum()), "n_ppg_artifact_windows": int(w.ppg_artifact_mask.sum()), "n_expert_rpeaks": int(n_rpeaks.sum()), "peaks_inside_artifacts": {"ecg": w.n_ecg_peaks_in_artifacts, "pleth": w.n_ppg_peaks_in_artifacts}, "fs_ppg": w.fs_ppg, "fs_ecg": w.fs_ecg, "sha256": sha256(p), "notes": w.notes}
        total += len(x)
        dropped_total += n_drop
        screened_total += n_screened
        audit_ecg += w.n_ecg_peaks_in_artifacts
        audit_ppg += w.n_ppg_peaks_in_artifacts
        print(w.subject, f.name, x.shape, "screened", n_screened, "dropped", n_drop, "fs", w.fs_ppg, w.fs_ecg, flush=True)
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "dataset": "CapnoBase TBME-RR (doi:10.5683/SP2/NLB8IT, 8 min cases)", "segment_len_s": args.segment_len, "resample_rate": args.resample_rate, "samples_per_window": args.resample_rate * args.segment_len, "fs_raw": 300, "artifact_screen": args.artifact_screen, "ppg_preprocess": PPG_KW, "ecg_preprocess": ECG_KW, "dtype": "float32", "total_windows": total, "total_dropped": dropped_total, "total_screened_artifact": screened_total, "peaks_inside_artifacts_audit": {"ecg": audit_ecg, "pleth": audit_ppg, "note": "shipped expert peaks lying strictly inside a shipped expert artifact interval; the labels are NOT screened upstream"}, "artifact_label_coverage": {**has_iv, "note": "number of FILES carrying >= 1 shipped expert artifact interval, per channel; 'neither' = files with an empty interval list on both channels"}, "extra_keys": {"n_expert_rpeaks": "int32 [n], expert ECG R-peaks per window (0-based raw 300 Hz domain)"}, "n_subjects": len(files), "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"total windows {total} (dropped {dropped_total}, screened {screened_total}) -> {out}/MANIFEST.json")


if __name__ == "__main__":
    main()
