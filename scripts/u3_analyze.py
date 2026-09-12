#!/usr/bin/env python
"""U3-B analysis (docs/U3_GATE_AND_SELECTION_SENSITIVITY_PREREGISTRATION.md §5).

Does arm I's deficit shrink when BOTH arms are read at exactly 14,000 optimizer steps
instead of at each arm's own best checkpoint? The §5 threshold -- half a margin -- was
fixed before any of these numbers existed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts/u2_paired"
MARGIN = {"HR": +1.0, "Rpeak_F1": -0.02, "RR": +0.5, "Resp_corr": -0.05, "SBP": +1.0, "DBP": +1.0}
TASKMETRIC = {"ECG": ["HR"], "Resp": ["RR"], "ABP": ["SBP", "DBP"]}   # PENGUIN's own, = "the primary cell"
CO = {"ECG": ["Rpeak_F1"], "Resp": ["Resp_corr"], "ABP": []}
TASK = {"u2_dalia": "ECG", "u2_wildppg": "ECG", "u2_bidmc": "Resp",
        "u2_wesad": "Resp", "u2_ucibp": "ABP", "u2_mimicbp": "ABP"}
COUNTABLE = [c for c in TASK if c != "u2_ucibp"]     # UCI-BP excluded by U2 §5/§9 (one test subject)


def load(slug: str, tag: str) -> dict:
    f = ART / f"paired_{slug}{tag}.csv"
    return {(r["metric"], int(r["k"])): r for r in csv.DictReader(f.open())}


def deficit(r: dict, m: str) -> float:
    """(I - C) in metric units, positive = arm I WORSE, in units of the margin."""
    d = MARGIN[m]
    lo, hi = float(r["ci_lo"]), float(r["ci_hi"])          # bootstrap reports positive = arm I BETTER,
    point = (lo + hi) / 2                                  # for loss AND score metrics alike
    return -point / abs(d)                                 # >0 means arm I is worse by that many margins


def main() -> None:
    rows, artefact = [], []
    for slug in TASK:
        best, last = load(slug, ""), load(slug, "_last")
        task = TASK[slug]
        for m in TASKMETRIC[task] + CO[task]:
            for k in (1, 2, 4):
                b, l = deficit(best[(m, k)], m), deficit(last[(m, k)], m)
                rows.append(dict(corpus=slug, task=task, metric=m, k=k, primary=m in TASKMETRIC[task],
                                 deficit_best=b, deficit_last=l, shift=b - l,
                                 armC50_best=float(best[(m, k)]["armC_nfe50"]),
                                 armC50_last=float(last[(m, k)]["armC_nfe50"]),
                                 armI_best=float(best[(m, k)]["armI_nfek"]),
                                 armI_last=float(last[(m, k)]["armI_nfek"])))
        if slug in COUNTABLE:
            # §5: generous to the artefact hypothesis -- the most favourable primary cell counts
            shifts = [r["shift"] for r in rows if r["corpus"] == slug and r["primary"]]
            artefact.append((slug, max(shifts), max(shifts) >= 0.5))

    print("deficit of arm I vs arm C@50, in units of that metric's margin (>0 = arm I worse)\n")
    print(f"{'corpus':11s} {'metric':10s} {'k':>1s} {'best-ckpt':>10s} {'last-ckpt':>10s} {'shift':>8s}  {'':3s}")
    cur = None
    for r in rows:
        if r["corpus"] != cur:
            print(); cur = r["corpus"]
        tag = "PRIM" if r["primary"] else "co"
        arrow = "shrinks" if r["shift"] >= 0.5 else ("grows" if r["shift"] <= -0.5 else "")
        print(f"{r['corpus']:11s} {r['metric']:10s} {r['k']:1d} {r['deficit_best']:10.3f} "
              f"{r['deficit_last']:10.3f} {r['shift']:+8.3f}  {tag:4s} {arrow}")

    print("\n§5 verdict input (countable datasets, most favourable primary cell):")
    for slug, s, ok in artefact:
        print(f"   {slug:11s} best shift {s:+.3f} margins  -> {'SHRINKS >= 0.5' if ok else 'no'}")
    n = sum(1 for _, _, ok in artefact if ok)
    v = "SELECTION ARTEFACT" if n >= 3 else ("PARTIAL ARTEFACT" if n >= 1 else "NOT AN ARTEFACT")
    print(f"\n   {n} of {len(artefact)} countable datasets shift by >= half a margin  ->  §5 VERDICT: {v}")
    (ART / "u3b_shift.json").write_text(json.dumps({"rows": rows, "n_shifted": n, "verdict": v}, indent=1))


if __name__ == "__main__":
    main()
