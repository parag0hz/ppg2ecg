#!/usr/bin/env python
"""U2 verdicts — applies §8 (margins, informativeness gate) and §9 (decision rule) to the
paired CSVs. No metric is chosen here; the rule was frozen in the preregistration.

Sign convention, stated because the frozen §8 wording is ambiguous for score metrics.
§8 says "non-inferiority is declared when the upper bound of the 95 % CI of (I_k − C_50)
lies below δ", with δ = +1.0 bpm for HR error but δ = −0.02 for R-peak F1. Read literally,
the second would demand that arm I be SIGNIFICANTLY WORSE, which is plainly not the
hypothesis. The margin signs fix the only coherent reading:

    loss metric  (δ > 0, lower is better):  non-inferior iff  upper(I − C) <  δ
    score metric (δ < 0, higher is better): non-inferior iff  lower(I − C) > δ

Both are the same statement -- "arm I is not worse than arm C@50 by more than |δ|" -- and
the ambiguity is disclosed in the report rather than resolved silently.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/u2_paired"

# prereg §8, frozen
MARGIN = {"HR": +1.0, "Rpeak_F1": -0.02, "RR": +0.5, "Resp_corr": -0.05,
          "SBP": +1.0, "DBP": +1.0}
PRIMARY = {"ECG": ("HR", "Rpeak_F1"), "Resp": ("RR", "Resp_corr"), "ABP": ("SBP", "DBP")}
TASK = {"u2_dalia": "ECG", "u2_wildppg": "ECG", "u2_bidmc": "Resp",
        "u2_wesad": "Resp", "u2_ucibp": "ABP", "u2_mimicbp": "ABP"}
EXCLUDED_MIN_SUBJECTS = 2          # §5/§9: fewer than two test subjects -> point estimate only


def cell(r: dict) -> dict:
    """One (metric, k) cell: orientation-corrected CI of (I − C), gate, and verdict."""
    m = r["metric"]
    d = MARGIN[m]
    # the bootstrap reports `diff` as positive = arm I better, whatever the orientation
    lo_b, hi_b = float(r["ci_lo"]), float(r["ci_hi"])
    if d > 0:                                   # loss metric: I − C = −diff
        lo, hi = -hi_b, -lo_b
        non_inferior = hi < d
    else:                                       # score metric: I − C = +diff
        lo, hi = lo_b, hi_b
        non_inferior = lo > d
    span = float(r["armC_span_nfe1_to_50"])
    informative = span >= abs(d)
    return {"metric": m, "k": int(r["k"]), "armC50": float(r["armC_nfe50"]),
            "armI": float(r["armI_nfek"]), "diff_I_minus_C": (lo + hi) / 2,
            "ci_lo": lo, "ci_hi": hi, "margin": d, "armC_span": span,
            "informative": informative, "non_inferior": bool(non_inferior and informative),
            "n_subjects": int(r["n_subjects"])}


def dataset_verdict(slug: str) -> dict | None:
    f = ART / f"paired_{slug}.csv"
    if not f.exists():
        return None
    rows = [r for r in csv.DictReader(f.open()) if r["metric"] in MARGIN]
    task = TASK[slug]
    prim = PRIMARY[task]
    cells = [cell(r) for r in rows if r["metric"] in prim]
    n_subj = max((c["n_subjects"] for c in cells), default=0)
    out = {"corpus": slug, "task": task, "n_test_subjects": n_subj, "cells": cells}
    if n_subj < EXCLUDED_MIN_SUBJECTS:
        out["verdict"] = "EXCLUDED (fewer than two test subjects; point estimate only)"
        out["counts"] = False
        return out
    if not any(c["informative"] for c in cells):
        out["verdict"] = "EXCLUDED (every primary cell UNINFORMATIVE)"
        out["counts"] = False
        return out
    # §9: non-inferior at the smallest k passing on BOTH primary and co-primary,
    # counting only informative cells
    passing = []
    for k in sorted({c["k"] for c in cells}):
        at_k = [c for c in cells if c["k"] == k and c["informative"]]
        needed = {c["metric"] for c in cells if c["informative"]}
        if needed and all(c["non_inferior"] for c in at_k) and {c["metric"] for c in at_k} == needed:
            passing.append(k)
    out["counts"] = True
    out["smallest_k"] = passing[0] if passing else None
    out["verdict"] = f"NON-INFERIOR at k={passing[0]}" if passing else "NOT non-inferior at any k <= 4"
    return out


def main() -> None:
    results = [v for v in (dataset_verdict(s) for s in TASK) if v]
    print(f"{'corpus':11s} {'task':5s} {'subj':>5s}  {'verdict':46s}")
    print("-" * 72)
    for v in results:
        print(f"{v['corpus']:11s} {v['task']:5s} {v['n_test_subjects']:5d}  {v['verdict']:46s}")
    print()
    for v in results:
        print(f"[{v['corpus']}]")
        for c in v["cells"]:
            flag = "PASS" if c["non_inferior"] else ("gate" if not c["informative"] else "fail")
            print(f"   {c['metric']:10s} k={c['k']}  C@50 {c['armC50']:8.4f}  I {c['armI']:8.4f}  "
                  f"CI(I-C) [{c['ci_lo']:+8.4f},{c['ci_hi']:+8.4f}]  margin {c['margin']:+.3f}  "
                  f"span {c['armC_span']:.4f}  {flag}")
        print()
    counted = [v for v in results if v["counts"]]
    wins = [v for v in counted if v.get("smallest_k")]
    print(f"stage tally: {len(wins)} of {len(counted)} countable datasets non-inferior at some k <= 4")
    if len(counted) < len(TASK):
        print(f"  excluded: {[v['corpus'] for v in results if not v['counts']]}")
    if len(counted) == 5:
        rule = "SUPPORTED" if len(wins) >= 4 else ("PARTIAL" if len(wins) >= 2 else "NOT SUPPORTED")
        print(f"  §9 stage verdict (5 countable): {rule}")
    else:
        print("  (stage verdict withheld until all six datasets are evaluated)")
    (ART / "verdicts.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
