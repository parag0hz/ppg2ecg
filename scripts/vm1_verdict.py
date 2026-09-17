"""VM1 verdicts (docs/VM1_VANILLA_IMF_PREREGISTRATION.md): each vanilla arm vs iMF arm I (BB1 rule), P vs B."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v1_evaluate import cluster_ci  # noqa: E402

OUT = ROOT / "artifacts/vm1_vanilla_imf"
M = ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE")


def rule(hr, f1):
    imp = (hr[0] <= -1.0 and hr[2] < 0) or (f1[0] >= 0.02 and f1[1] > 0)
    wor = (hr[0] >= 1.0 and hr[1] > 0) or (f1[0] <= -0.02 and f1[2] < 0)
    return "MIXED" if imp and wor else "IMPROVES" if imp else "WORSE" if wor else "NO MEANINGFUL CHANGE"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = {"I": dict(np.load(ROOT / "outputs/bb1_eval/arm_I.npz"))}
    for a in ("S", "P", "B"):
        D[a] = dict(np.load(ROOT / f"outputs/vm1_eval/arm_{a}.npz"))
    pid = D["I"]["pid"]
    assert all(np.array_equal(D[a]["pid"], pid) and np.allclose(D[a]["ref_hr"], D["I"]["ref_hr"], equal_nan=True) for a in D)
    rows, verdict = [], {}
    for a, d in D.items():
        for nfe in (1, 2, 4):
            for m in M:
                v, lo, hi = cluster_ci(d[f"nfe{nfe}_{m}"], pid)
                row = dict(arm=a, nfe=nfe, metric=m, value=v, ci_lo=lo, ci_hi=hi)
                if a != "I":
                    row.update(zip(("diff_vs_I", "diff_ci_lo", "diff_ci_hi"), cluster_ci(d[f"nfe{nfe}_{m}"] - D["I"][f"nfe{nfe}_{m}"], pid)))
                rows.append(row)
            for m in ("FD_kanflow", "Micro_F1"):
                rows.append(dict(arm=a, nfe=nfe, metric=m, value=float(d[f"nfe{nfe}_pooled_{m}"])))
            rows.append(dict(arm=a, nfe=nfe, metric="ms_per_window", value=float(d[f"nfe{nfe}_ms_per_window"])))
        rows.append(dict(arm=a, nfe=1, metric="HR_consensus_K16", **dict(zip(("value", "ci_lo", "ci_hi"), cluster_ci(d["cons16_nfe1_hr_err"], pid)))))
        if a != "I":
            for m in ("HR", "Rpeak_F1"):
                rows.append(dict(arm=a, nfe=1, metric=f"{m}_no_guidance", **dict(zip(("value", "ci_lo", "ci_hi"), cluster_ci(d[f"noguid_nfe1_{m}"], pid)))))
            rows.append(dict(arm=a, nfe=1, metric="FD_kanflow_no_guidance", value=float(d["noguid_nfe1_pooled_FD_kanflow"])))
            verdict[f"{a}_vs_I"] = rule(cluster_ci(d["nfe1_HR"] - D["I"]["nfe1_HR"], pid), cluster_ci(d["nfe1_Rpeak_F1"] - D["I"]["nfe1_Rpeak_F1"], pid))
            verdict[f"{a}_cfg"] = [float(x) for x in d["cfg"]]
    verdict["P_vs_B_pretraining"] = rule(cluster_ci(D["P"]["nfe1_HR"] - D["B"]["nfe1_HR"], pid), cluster_ci(D["P"]["nfe1_Rpeak_F1"] - D["B"]["nfe1_Rpeak_F1"], pid))
    verdict["P_vs_B_HR_diff"] = cluster_ci(D["P"]["nfe1_HR"] - D["B"]["nfe1_HR"], pid)
    verdict["P_vs_B_F1_diff"] = cluster_ci(D["P"]["nfe1_Rpeak_F1"] - D["B"]["nfe1_Rpeak_F1"], pid)
    keys = ["arm", "nfe", "metric", "value", "ci_lo", "ci_hi", "diff_vs_I", "diff_ci_lo", "diff_ci_hi"]
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    sel = {a: json.loads(str(D[a]["selection"])) for a in ("S", "P", "B")}
    (OUT / "cfg_selection.json").write_text(json.dumps(sel, indent=1))
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    for r in rows:
        extra = f"  diff {r['diff_vs_I']:+.4f} [{r['diff_ci_lo']:+.4f}, {r['diff_ci_hi']:+.4f}]" if "diff_vs_I" in r else ""
        print(f"{r['arm']} nfe{r['nfe']} {r['metric']:<24} {r['value']:.4f}{extra}")


if __name__ == "__main__":
    main()
