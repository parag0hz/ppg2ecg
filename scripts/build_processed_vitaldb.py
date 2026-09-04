"""Build data/processed/vitaldb_{L}s/<case>.npz from raw VitalDB cases with the frozen PENGUIN preprocessing
(PPG band-pass 0.5-4 Hz, ECG high-pass 0.5 Hz, per-window z-score + min-max; FFT-resample 500 -> 128 Hz, L*128 samples).
Windows containing a non-finite sample (VitalDB NaN dropouts) or a constant signal are DROPPED and counted, before and
after preprocessing. The case subset is the deterministic caseid-ascending prefix of ppg2ecg.data.vitaldb.SELECTION_RULE
(see the scale-policy note in that module and in the MANIFEST).
Run: .venv/bin/python scripts/build_processed_vitaldb.py --segment-len 8 --target-windows 390000"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows
from ppg2ecg.data.vitaldb import MIN_DURATION_S, MIN_FINITE_FRAC, SELECTION_RULE, TARGET_WINDOWS, participant_files, select_cases, windows_for_participant

ROOT = Path(__file__).resolve().parents[1]
SUBJECT_IDENTITY_NOTE = (
    "Subject identity = caseid ('case_%05d'); VitalDB cases are one operation each. The full release has 6388 cases mapping "
    "to 6090 subjectids (data/raw/VitalDB/cases.csv), so caseid-level splitting is very slightly weaker than subject-level "
    "splitting (a few patients contribute more than one case). We accept this and state it."
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/VitalDB"))
    ap.add_argument("--segment-len", type=int, default=8)
    ap.add_argument("--resample-rate", type=int, default=128)
    ap.add_argument("--target-windows", type=int, default=TARGET_WINDOWS, help="cumulative ESTIMATED window budget; the case that first reaches it is included")
    ap.add_argument("--max-cases", type=int, default=None, help="hard cap on the number of selected cases (applied on top of --target-windows)")
    ap.add_argument("--min-duration-s", type=float, default=MIN_DURATION_S)
    ap.add_argument("--min-finite-frac", type=float, default=MIN_FINITE_FRAC)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "data" / "processed" / f"vitaldb_{args.segment_len}s"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sel = select_cases(args.raw, args.target_windows, args.max_cases, args.min_duration_s, args.min_finite_frac, args.segment_len)
    print(f"selected {len(sel.cases)} cases ({sel.cum_est_windows} estimated windows) after scanning {sel.n_scanned} of {len(participant_files(args.raw))} available ({len(sel.ineligible)} ineligible) in {time.perf_counter() - t0:.0f} s", flush=True)
    files, total, dropped_total = {}, 0, 0
    for i, c in enumerate(sel.cases, 1):
        w = windows_for_participant(c.path, args.segment_len)
        finite = np.isfinite(w.ppg).all(axis=1) & np.isfinite(w.ecg).all(axis=1)
        std_ok = (w.ppg.std(axis=1) > 0) & (w.ecg.std(axis=1) > 0)
        keep = finite & std_ok
        x = preprocess_windows(w.ppg[keep], args.resample_rate, args.segment_len, **PPG_KW).astype(np.float32)
        y = preprocess_windows(w.ecg[keep], args.resample_rate, args.segment_len, **ECG_KW).astype(np.float32)
        ok2 = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x, y = x[ok2], y[ok2]
        widx = w.window_index[keep][ok2]
        p = out / f"{w.subject}.npz"
        np.savez(p, x=x, y=y, window_index=widx.astype(np.int32), window_start_s=(widx * args.segment_len).astype(np.int32), subject=w.subject)
        n_drop = int((~keep).sum() + (~ok2).sum())
        files[w.subject] = {"path": str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p), "raw_file": c.path.name, "raw_sha256": sha256(c.path), "n_windows": int(len(x)), "n_dropped_nonfinite_or_constant": n_drop, "fs_ppg": w.fs_ppg, "fs_ecg": w.fs_ecg, "est_windows": c.est_windows, "sha256": sha256(p), "notes": w.notes}
        total += len(x)
        dropped_total += n_drop
        if i % 10 == 0 or i == len(sel.cases):
            print(f"[{i}/{len(sel.cases)}] {w.subject} kept {len(x)} dropped {n_drop} | running total {total} windows ({dropped_total} dropped) {time.perf_counter() - t0:.0f} s", flush=True)
    elapsed = time.perf_counter() - t0
    manifest = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "dataset": "VitalDB",
        "source": "api.vitaldb.net SNUADC/PLETH + SNUADC/ECG_II @ 500 Hz, downloaded by scripts/dl_vitaldb.py",
        "segment_len_s": args.segment_len,
        "resample_rate": args.resample_rate,
        "samples_per_window": args.resample_rate * args.segment_len,
        "fs_raw": 500,
        "ppg_preprocess": PPG_KW,
        "ecg_preprocess": ECG_KW,
        "dtype": "float32",
        "total_windows": total,
        "total_dropped": dropped_total,
        "n_cases": len(files),
        "elapsed_s": round(elapsed, 1),
        "scale_policy": {
            "reason": "training on all 6156 downloaded cases (~9M 8 s windows) is not comparable to the other corpora in this benchmark and is infeasible; the budget targets the largest natural corpus (WildPPG 8 s = 389,355 windows)",
            "rule": SELECTION_RULE,
            "target_windows": args.target_windows,
            "max_cases": args.max_cases,
            "min_duration_s": args.min_duration_s,
            "min_finite_frac": args.min_finite_frac,
            "n_case_files_available": len(participant_files(args.raw)),
            "n_cases_scanned": sel.n_scanned,
            "n_cases_ineligible": len(sel.ineligible),
            "ineligible_cases": [{"caseid": c.caseid, "duration_s": round(c.duration_s, 1), "finite_frac_ppg": round(c.finite_frac_ppg, 4), "finite_frac_ecg": round(c.finite_frac_ecg, 4), "has_both_tracks": c.has_both} for c in sel.ineligible],
            "n_cases_selected": len(sel.cases),
            "cumulative_estimated_windows": sel.cum_est_windows,
            "selected_caseids": [c.caseid for c in sel.cases],
            "deterministic": "order-deterministic caseid-ascending prefix; no randomness, no sampling",
        },
        "subject_identity": SUBJECT_IDENTITY_NOTE,
        "drop_rule": "a window is dropped if either signal has any non-finite sample OR zero std, BEFORE preprocessing; dropped again after preprocessing if any non-finite survives; both counted in n_dropped_nonfinite_or_constant",
        "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"total windows {total} (dropped {dropped_total}) from {len(files)} cases in {elapsed / 60:.1f} min -> {out}/MANIFEST.json")


if __name__ == "__main__":
    main()
