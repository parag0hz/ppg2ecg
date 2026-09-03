"""Recompute the Q1 support-fidelity coupling table from the frozen per-window artifacts.

Pure post-processing of `r1_support_metrics.csv` and `generator_fidelity_metrics.csv`: no model is loaded,
no window is regenerated. Used to bring the coupling table to the preregistered 2,000 bootstrap replicates
and to add the preregistered per-level rows; point estimates are unaffected by the replicate count.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ppg2ecg.evaluation import q1_corruption as Q

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/q1_conditional_support"
if "q1_evaluate" in sys.modules:
    Q1 = sys.modules["q1_evaluate"]
else:
    _q1 = importlib.util.spec_from_file_location("q1_evaluate", ROOT / "scripts/q1_evaluate.py")
    Q1 = importlib.util.module_from_spec(_q1); sys.modules[_q1.name] = Q1; _q1.loader.exec_module(Q1)


def _load(name, keys):
    by = defaultdict(list)
    sub = defaultdict(list)
    with open(ART / name, newline="") as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k in keys:
                try:
                    row[k] = float(r[k])
                except ValueError:
                    row[k] = np.nan
            by[r["condition"]].append(row)
            sub[r["condition"]].append(r["subject"])
    return by, sub


def main() -> int:
    sup, s1 = _load("r1_support_metrics.csv", ("r1_f1@150", "r1_rr_mae_ms"))
    fid, s2 = _load("generator_fidelity_metrics.csv", ("f1_excess", "beats_ratio_dev", "raw_qrs_rmse"))
    SUB = np.asarray(s1[Q.CLEAN])
    assert np.array_equal(SUB, np.asarray(s2[Q.CLEAN])) and len(SUB) == 2048
    rows = Q1.coupling_rows(sup, fid, SUB)
    Q1.wcsv(ART / "support_fidelity_correlations.csv", rows)
    print(f"[coupling] {len(rows)} rows at n_boot={Q.BOOT_N}, seed={Q.BOOT_SEED}", flush=True)
    for r in rows:
        if r["level"] == "pooled":
            print(f"  {r['family']:<9} {r['x']:<12} -> {r['y']:<16} rho {r['spearman_rho']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] n {r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
