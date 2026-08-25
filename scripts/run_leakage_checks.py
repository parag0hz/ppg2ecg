"""Run all leakage checks against a manifest (+ processed arrays when available). Exit code 1 on failure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ppg2ecg.data.dalia import SUBJECTS
from ppg2ecg.data.leakage import check_subject_disjoint, check_window_disjoint, check_windowwise_normalization
from ppg2ecg.data.preprocess import PPG_KW, preprocess_windows
from ppg2ecg.data.splits import read_manifest

ROOT = Path(__file__).resolve().parents[1]


def load_processed(processed_dir: Path, subjects: list[str]) -> np.ndarray | None:
    xs = []
    for s in subjects:
        f = processed_dir / f"{s}.npz"
        if not f.exists():
            return None
        xs.append(np.load(f)["x"])
    return np.concatenate(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/split_p0_holdout_seed42.json"))
    ap.add_argument("--processed", default=str(ROOT / "data/processed/v0"))
    ap.add_argument("--expected", default="dalia", help="dalia = S1..S15; auto = union of the manifest split")
    args = ap.parse_args()
    ok_all = True
    for split in read_manifest(args.manifest):
        rep = {"split": {k: split[k] for k in ("protocol", "seed", "train", "val", "test") if k in split}}
        expected = SUBJECTS if args.expected == "dalia" else sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))
        rep["subject_disjoint"] = check_subject_disjoint(split, expected)
        arrays = {k: load_processed(Path(args.processed), split[k]) for k in ("train", "val", "test")}
        if all(v is not None for v in arrays.values()):
            rep["window_disjoint"] = check_window_disjoint(arrays)
            sample = arrays["test"][:64]
        else:
            rep["window_disjoint"] = {"ok": None, "note": f"processed arrays not found under {args.processed}; skipped"}
            sample = np.random.default_rng(0).standard_normal((64, 256))
        seg = sample.shape[1] // 128
        rep["windowwise_normalization"] = check_windowwise_normalization(lambda a: preprocess_windows(a, 128, seg, **PPG_KW), sample)
        ok = all(v.get("ok") is not False for v in rep.values() if isinstance(v, dict) and "ok" in v)
        ok_all &= ok
        print(json.dumps(rep, indent=1, default=str))
        print("=> PASS" if ok else "=> FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
