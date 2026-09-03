"""O1 — recompute the generator-utilization crosswalk with the PREREGISTERED clustered bootstrap.

Preregistration section 14 asks for a paired *clustered* bootstrap where the per-window artifacts permit, and
section 2 binds the cluster to the underlying ECG window. The first pass used the unclustered
`paired_subject_bootstrap`; this script rebuilds the table with the ECG-window cluster bootstrap. Pure
post-processing of the frozen Q1 per-window artifacts: no model is loaded and no window is regenerated.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o1_component_extractability"
Q1ART = ROOT / "artifacts/q1_conditional_support"
if "o1_evaluate" in sys.modules:
    O1 = sys.modules["o1_evaluate"]
else:
    _s = importlib.util.spec_from_file_location("o1_evaluate", ROOT / "scripts/o1_evaluate.py")
    O1 = importlib.util.module_from_spec(_s); sys.modules[_s.name] = O1; _s.loader.exec_module(O1)
VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024


def q1_clusters():
    """(subject, window_index) of every row of the frozen Q1 2,048-window cohort, in population order."""
    ER.assert_no_test_subjects(VAL)
    sub, wi = [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        WI = d["window_index"].astype(np.int64)
        idx = ER.select_subset(SALT, s, len(WI), TAKE)
        sub += [s] * len(idx); wi += WI[idx].tolist()
    return np.asarray(sub), np.asarray(wi)


def main() -> int:
    sub, wi = q1_clusters()
    cluster = np.array([f"{a}|{b}" for a, b in zip(sub, wi)])
    q1 = defaultdict(list)
    for r in csv.DictReader(open(Q1ART / "generator_fidelity_metrics.csv")):
        q1[r["condition"]].append(r)
    cl, sh = q1["CLEAN"], q1["SHUFFLED"]
    assert len(cl) == len(sh) == len(sub) == 2048
    assert [r["subject"] for r in cl] == sub.tolist(), "Q1 row order does not match the rebuilt cohort"
    rows_out = []
    for r in csv.DictReader(open(ART / "generator_utilization_crosswalk.csv")):
        if r["generator_metric"] == "N/A":
            rows_out.append(r); continue
        m, orient = r["generator_metric"], r["orientation"]
        a = np.array([float(x[m]) for x in sh], float)      # SHUFFLED
        b = np.array([float(x[m]) for x in cl], float)      # CLEAN
        d = (b - a) if orient == "higher_better" else (a - b)     # positive = correct PPG is better
        res = O1.cluster_bootstrap(d, sub, cluster)
        rows_out.append({**r, "utilization_effect": res["point"], "lo": res["lo"], "hi": res["hi"],
                         "verdict": res["verdict"], "n_boot": res["n_boot"], "seed": res["seed"],
                         "bootstrap": "ECG-window clustered, subject-stratified (preregistration section 14)",
                         "n_clusters": int(len(set(cluster.tolist())))})
        print(f"  {r['component']:<30} {m:<18} effect {res['point']:+.4f} [{res['lo']:+.4f},{res['hi']:+.4f}] {res['verdict']}", flush=True)
    O1.wcsv(ART / "generator_utilization_crosswalk.csv", rows_out)
    print(f"[crosswalk] rebuilt with {len(set(cluster.tolist()))} ECG-window clusters over 2,048 rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
