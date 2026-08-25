"""Inspect the raw PPG-DaLiA pickles: subjects, lengths, sampling rates, window counts, PPG/ECG alignment.
Run after scripts/download_dalia.sh:  .venv/bin/python scripts/verify_dalia.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ppg2ecg.data.dalia import BVP_FS, ECG_FS, SEGMENT_LEN_S, SUBJECTS, load_subject_raw, windows_for_subject

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main():
    rows, total_windows, mismatched = [], 0, []
    for s in SUBJECTS:
        try:
            raw = load_subject_raw(RAW, s)
        except FileNotFoundError as e:
            print(f"{s}: MISSING ({e})")
            continue
        w = windows_for_subject(raw, SEGMENT_LEN_S)
        dur_diff = raw.ecg_seconds - raw.bvp_seconds
        row = {
            "subject": s, "ecg_len": int(len(raw.ecg)), "bvp_len": int(len(raw.bvp)),
            "ecg_s": round(raw.ecg_seconds, 2), "bvp_s": round(raw.bvp_seconds, 2), "ecg_minus_bvp_s": round(dur_diff, 3),
            "n_ppg_win": w.n_ppg_windows, "n_ecg_win": w.n_ecg_windows, "n_kept": w.truncated_to,
            "n_rpeaks": int(len(raw.rpeaks)) if raw.rpeaks is not None else None,
            "hr_label_n": int(len(raw.hr_label)) if raw.hr_label is not None else None,
            "hr_label_mean": round(float(np.mean(raw.hr_label)), 1) if raw.hr_label is not None else None,
            "ecg_dtype": str(raw.ecg.dtype), "bvp_dtype": str(raw.bvp.dtype),
        }
        rows.append(row)
        total_windows += w.truncated_to
        if w.n_ppg_windows != w.n_ecg_windows:
            mismatched.append((s, w.n_ppg_windows, w.n_ecg_windows))
        print(json.dumps(row))
    print(f"\nsubjects found: {len(rows)}/15 | total 4 s windows kept: {total_windows} (upstream preprocess.yaml says sample_num=16181)")
    print(f"fs assumed: ECG {ECG_FS} Hz, BVP {BVP_FS} Hz; window mismatches (subject, n_ppg, n_ecg): {mismatched or 'none'}")
    out = ROOT / "data" / "manifests" / "dalia_raw_inventory.json"
    out.write_text(json.dumps({"rows": rows, "total_windows": total_windows, "mismatched": mismatched}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
