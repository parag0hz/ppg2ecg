"""BB1 verdict from outputs/bb1_eval/arm_{I,A,B}.npz (rules: docs/BB1_BACKBONE_SENSITIVITY_PREREGISTRATION.md)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v1_evaluate import cluster_ci  # noqa: E402

IN, OUT = ROOT / "outputs/bb1_eval", ROOT / "artifacts/bb1_backbone"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = {a: dict(np.load(IN / f"arm_{a}.npz")) for a in ("I", "A", "B")}
    pid = D["I"]["pid"]
    assert all(np.array_equal(D[a]["pid"], pid) for a in D)
    rows = []
    for a, d in D.items():
        for nfe in (1, 2, 4):
            for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE"):
                v, lo, hi = cluster_ci(d[f"nfe{nfe}_{m}"], pid)
                row = dict(arm=a, nfe=nfe, metric=m, value=v, ci_lo=lo, ci_hi=hi)
                if a != "I":
                    dv, dlo, dhi = cluster_ci(d[f"nfe{nfe}_{m}"] - D["I"][f"nfe{nfe}_{m}"], pid)
                    row.update(diff_vs_I=dv, diff_ci_lo=dlo, diff_ci_hi=dhi)
                rows.append(row)
            for m in ("FD_kanflow", "Micro_F1", "Macro_F1"):
                rows.append(dict(arm=a, nfe=nfe, metric=m, value=float(d[f"nfe{nfe}_pooled_{m}"])))
            rows.append(dict(arm=a, nfe=nfe, metric="ms_per_window", value=float(d[f"nfe{nfe}_ms_per_window"])))
        v, lo, hi = cluster_ci(d["cons16_nfe1_hr_err"], pid)
        row = dict(arm=a, nfe=1, metric="HR_consensus_K16", value=v, ci_lo=lo, ci_hi=hi)
        if a != "I":
            dv, dlo, dhi = cluster_ci(d["cons16_nfe1_hr_err"] - D["I"]["cons16_nfe1_hr_err"], pid)
            row.update(diff_vs_I=dv, diff_ci_lo=dlo, diff_ci_hi=dhi)
        rows.append(row)
    verdict = {}
    for a in ("A", "B"):
        g = {r["metric"]: r for r in rows if r["arm"] == a and r["nfe"] == 1 and "diff_vs_I" in r}
        hr, f1 = g["HR"], g["Rpeak_F1"]
        imp = (hr["diff_vs_I"] <= -1.0 and hr["diff_ci_hi"] < 0) or (f1["diff_vs_I"] >= 0.02 and f1["diff_ci_lo"] > 0)
        wor = (hr["diff_vs_I"] >= 1.0 and hr["diff_ci_lo"] > 0) or (f1["diff_vs_I"] <= -0.02 and f1["diff_ci_hi"] < 0)
        verdict[a] = "MIXED" if imp and wor else "IMPROVES" if imp else "WORSE" if wor else "NO MEANINGFUL CHANGE"
    verdict["stage"] = "BACKBONE IS A BOTTLENECK" if "IMPROVES" in (verdict["A"], verdict["B"]) else "NOT A BOTTLENECK AT THIS SCALE"
    keys = ["arm", "nfe", "metric", "value", "ci_lo", "ci_hi", "diff_vs_I", "diff_ci_lo", "diff_ci_hi"]
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    for r in rows:
        extra = f"  diff {r['diff_vs_I']:+.4f} [{r['diff_ci_lo']:+.4f}, {r['diff_ci_hi']:+.4f}]" if "diff_vs_I" in r else ""
        print(f"{r['arm']} nfe{r['nfe']} {r['metric']:<18} {r['value']:.4f}{extra}")


if __name__ == "__main__":
    main()
