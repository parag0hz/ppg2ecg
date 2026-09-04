"""Convert the ALREADY-PREPROCESSED upstream PPG-DaLiA 8 s pickles into the D1 processed-corpus contract
(docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md §4, "convert from upstream_8s pickles").

Source: data/processed/upstream_8s/PPG-DaLiA/subject{i}.pkl  (i = S{i+1}; mapping fixed by
scripts/check_processed_parity.py L23-27), written by the UNMODIFIED upstream preprocess.py through
configs/upstream/preprocess.yaml with segment_len 8 -> PPG_KW / ECG_KW applied at 128 Hz. Parity with our own
pipeline is bit-exact for all 15 subjects (data/manifests/processed_parity_upstream_8s.json).

NOTHING IS RE-FILTERED HERE. Re-running preprocess_windows on an already band-passed, z-scored, min-max'd
signal would be a second filtering pass; this script only re-packages the arrays into the npz schema of
data/processed/wildppg_8s and verifies that the upstream normalisation is intact (per-window min == -1,
per-window max == +1 up to the 1e-8 denominator guard in preprocess_windows L38).

Run: .venv/bin/python scripts/build_processed_dalia.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from ppg2ecg.data.dalia import BVP_FS, ECG_FS, SUBJECTS
from ppg2ecg.data.preprocess import ECG_KW, PPG_KW

ROOT = Path(__file__).resolve().parents[1]
MINMAX_ATOL = 1e-6  # upstream min-max maps the window max to 1 - 1e-8/(range+1e-8), never exactly 1


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def assert_upstream_normalised(a: np.ndarray, name: str, subject: str) -> None:
    """The upstream arrays must already be per-window min-max'd to [-1, 1]; if not, the source is not what we think."""
    lo, hi = a.min(axis=1), a.max(axis=1)
    assert np.all(a >= -1.0) and np.all(a <= 1.0), f"{subject}/{name}: outside [-1, 1] ({a.min()}, {a.max()})"
    assert np.allclose(lo, -1.0, atol=MINMAX_ATOL), f"{subject}/{name}: per-window min != -1 (worst {np.abs(lo + 1).max()})"
    assert np.allclose(hi, 1.0, atol=MINMAX_ATOL), f"{subject}/{name}: per-window max != +1 (worst {np.abs(hi - 1).max()})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/processed/upstream_8s/PPG-DaLiA"))
    ap.add_argument("--segment-len", type=int, default=8)
    ap.add_argument("--resample-rate", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    src = Path(args.src)
    out = Path(args.out) if args.out else ROOT / "data" / "processed" / f"dalia_{args.segment_len}s"
    out.mkdir(parents=True, exist_ok=True)
    files, total, dropped_total = {}, 0, 0
    for subject in SUBJECTS:
        raw = src / f"subject{int(subject[1:]) - 1}.pkl"  # upstream index is 0-based over S1..S15
        with open(raw, "rb") as f:
            d = pickle.load(f)
        x_all, y_all = np.asarray(d["x_data"], np.float64), np.asarray(d["y_data"], np.float64)
        assert x_all.shape == y_all.shape and x_all.shape[1] == args.resample_rate * args.segment_len, (x_all.shape, y_all.shape)
        # the pre-preprocessing half of the frozen drop rule is not applicable (the source is already preprocessed);
        # the post-preprocessing half still is, and must find nothing.
        ok = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all).all(axis=1) & (x_all.std(axis=1) > 0) & (y_all.std(axis=1) > 0)
        x, y = x_all[ok], y_all[ok]
        assert_upstream_normalised(x, "x", subject)
        assert_upstream_normalised(y, "y", subject)
        # window_index is the position in the RECORDING, so it is numbered over x_all and then masked: numbering
        # over the survivors instead would renumber a dropped subject's windows 0..n-1 and lose the true offset.
        widx = np.arange(len(x_all), dtype=np.int32)[ok]  # non-overlapping windows from t = 0, upstream keeps them in order
        p = out / f"{subject}.npz"
        np.savez(p, x=x.astype(np.float32), y=y.astype(np.float32), window_index=widx, window_start_s=(widx * args.segment_len).astype(np.int32), subject=subject)
        n_drop = int((~ok).sum())
        files[subject] = {"path": str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p), "raw_file": raw.name, "raw_sha256": sha256(raw), "n_windows": int(len(x)), "n_dropped_nonfinite_or_constant": n_drop, "fs_ppg": BVP_FS, "fs_ecg": ECG_FS, "sha256": sha256(p), "notes": "repackaged from the upstream-preprocessed 8 s pickle; no filtering applied here"}
        total += len(x)
        dropped_total += n_drop
        print(subject, raw.name, x.shape, "dropped", n_drop, "fs", BVP_FS, ECG_FS, flush=True)
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "dataset": "PPG-DaLiA", "source": "data/processed/upstream_8s/PPG-DaLiA (UNMODIFIED external/PENGUIN preprocess.py, configs/upstream/preprocess.yaml, segment_len 8); bit-exact with ppg2ecg.data.preprocess per data/manifests/processed_parity_upstream_8s.json", "segment_len_s": args.segment_len, "resample_rate": args.resample_rate, "samples_per_window": args.resample_rate * args.segment_len, "ppg_preprocess": PPG_KW, "ecg_preprocess": ECG_KW, "preprocess_applied_by": "upstream (this script re-packages only)", "dtype": "float32", "total_windows": total, "total_dropped": dropped_total, "n_subjects": len(files), "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"total windows {total} (dropped {dropped_total}) -> {out}/MANIFEST.json")


if __name__ == "__main__":
    main()
