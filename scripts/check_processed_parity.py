"""End-to-end parity on REAL data: our loader+windowing+preprocess (ppg2ecg.data) vs the files written by the
UNMODIFIED upstream preprocess.py (data/processed/upstream/PPG-DaLiA/subject{i}.pkl, i = S{i+1}).
Also runs the window-level leakage check across the P0 manifest on the upstream-processed arrays."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from ppg2ecg.data.dalia import SUBJECTS, load_subject_raw, windows_for_subject
from ppg2ecg.data.leakage import check_window_disjoint
from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows
from ppg2ecg.data.splits import read_manifest

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed/upstream/PPG-DaLiA"


def upstream_arrays(subject: str):
    with open(PROC / f"subject{int(subject[1:]) - 1}.pkl", "rb") as f:
        d = pickle.load(f)
    return d["x_data"], d["y_data"]


def main(subjects=("S1", "S6", "S15")):
    report = {}
    for s in subjects:
        raw = load_subject_raw(RAW, s)
        w = windows_for_subject(raw, 4)
        x = preprocess_windows(w.ppg, 128, 4, **PPG_KW)
        y = preprocess_windows(w.ecg, 128, 4, **ECG_KW)
        ux, uy = upstream_arrays(s)
        report[s] = {"shape_ours": [x.shape, y.shape], "shape_upstream": [ux.shape, uy.shape], "max_abs_diff_x": float(np.max(np.abs(x - ux))) if x.shape == ux.shape else None, "max_abs_diff_y": float(np.max(np.abs(y - uy))) if y.shape == uy.shape else None, "bit_exact": bool(x.shape == ux.shape and np.array_equal(x, ux) and np.array_equal(y, uy))}
        print(s, report[s])
    ok = all(r["bit_exact"] for r in report.values())
    split = read_manifest(ROOT / "data/manifests/split_p0_holdout_seed42.json")[0]
    arrays = {k: np.concatenate([upstream_arrays(s)[0] for s in split[k]]) for k in ("train", "val", "test")}
    wd = check_window_disjoint(arrays)
    print("window-disjoint (P0, upstream-processed PPG windows):", wd)
    total = sum(len(upstream_arrays(s)[0]) for s in SUBJECTS)
    print("total processed windows:", total)
    out = ROOT / "data/manifests/processed_parity_upstream.json"
    out.write_text(json.dumps({"parity": report, "window_disjoint_p0": wd, "total_windows": total}, indent=1, default=str))
    print("=> PARITY", "PASS" if ok else "FAIL", "| WINDOW-DISJOINT", "PASS" if wd["ok"] else "FAIL")
    return 0 if ok and wd["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
