"""RF1 verdicts (docs/RF1_REFLOW_PREREGISTRATION.md): arm R vs iMF arm I, and vs PENGUIN-50."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v1_evaluate import cluster_ci  # noqa: E402

IN = ROOT / "outputs/sr1_eval"
ARM = {"R": ("artifacts/rf1_reflow", "outputs/rf1_vitaldb_armR_seed42"),
       "D": ("artifacts/cd1_consistency", "outputs/cd1_vitaldb_armD_seed42")}
MET = ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE")


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=list(ARM), default="R")
    X = ap.parse_args().arm                       # the few-step competitor under test (R = reflow, D = consistency)
    OUT, RUN = ROOT / ARM[X][0], ROOT / ARM[X][1]
    OUT.mkdir(parents=True, exist_ok=True)
    D = {a: dict(np.load(IN / f"arm_{a}_seed42.npz")) for a in ("C", "I", "S", X)}
    pid = D["C"]["pid"]
    assert all(np.array_equal(d["pid"], pid) for d in D.values())
    rows = []
    for a, d in D.items():
        for nfe in (1, 2, 4, 50):
            if f"nfe{nfe}_HR" not in d:
                continue
            for m in MET:
                v, lo, hi = cluster_ci(d[f"nfe{nfe}_{m}"], pid)
                row = dict(arm=a, nfe=nfe, metric=m, value=v, ci_lo=lo, ci_hi=hi)
                if a == X:
                    row.update(zip(("diff_vs_I", "d_lo", "d_hi"), cluster_ci(d[f"nfe{nfe}_{m}"] - D["I"][f"nfe{nfe}_{m}"], pid)))
                rows.append(row)
            for m in ("FD_kanflow", "Micro_F1"):
                rows.append(dict(arm=a, nfe=nfe, metric=m, value=float(d[f"nfe{nfe}_pooled_{m}"])))
            rows.append(dict(arm=a, nfe=nfe, metric="ms_per_window", value=float(d[f"nfe{nfe}_ms_per_window"])))
        v, lo, hi = cluster_ci(d["cons16_nfe1_hr_err"], pid)
        rows.append(dict(arm=a, nfe=1, metric="HR_consensus_K16", value=v, ci_lo=lo, ci_hi=hi))

    R, I, C = D[X], D["I"], D["C"]
    hr_i = cluster_ci(R["nfe1_HR"] - I["nfe1_HR"], pid)
    f1_i = cluster_ci(R["nfe1_Rpeak_F1"] - I["nfe1_Rpeak_F1"], pid)
    imp = (hr_i[0] <= -1.0 and hr_i[2] < 0) or (f1_i[0] >= 0.02 and f1_i[1] > 0)
    wor = (hr_i[0] >= 1.0 and hr_i[1] > 0) or (f1_i[0] <= -0.02 and f1_i[2] < 0)
    hr_c = cluster_ci(R["nfe1_HR"] - C["nfe50_HR"], pid)
    f1_c = cluster_ci(R["nfe1_Rpeak_F1"] - C["nfe50_Rpeak_F1"], pid)
    subs = np.unique(pid)
    pp = lambda v: np.array([np.nanmean(v[pid == s]) for s in subs])  # noqa: E731
    win = float(np.mean(pp(R["nfe1_HR"]) < pp(C["nfe50_HR"])))
    verdict = {"RF_vs_iMF": "MIXED" if imp and wor else "IMPROVES" if imp else "WORSE" if wor else "NO MEANINGFUL CHANGE",
               "RF_vs_iMF_HR": hr_i, "RF_vs_iMF_F1": f1_i,
               "RF_vs_PENGUIN50_non_inferior": bool(hr_c[2] < 1.0 and f1_c[1] > -0.02),
               "RF_vs_PENGUIN50_HR": hr_c, "RF_vs_PENGUIN50_F1": f1_c,
               "RF_win_rate_HR_vs_PENGUIN50": win,
               "teacher_seconds": json.load(open(RUN / "training_summary.json")).get("teacher_seconds"),
               "train_seconds": json.load(open(RUN / "training_summary.json"))["total_train_time_s"]}
    with open(OUT / "metrics.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["arm", "nfe", "metric", "value", "ci_lo", "ci_hi", "diff_vs_I", "d_lo", "d_hi"])
        wcsv.writeheader(); wcsv.writerows(rows)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    for r in rows:
        if r["arm"] == X or r["metric"] in ("HR", "Rpeak_F1", "HR_consensus_K16", "FD_kanflow"):
            extra = f"  diff_vs_I {r['diff_vs_I']:+.4f} [{r['d_lo']:+.4f}, {r['d_hi']:+.4f}]" if "diff_vs_I" in r else ""
            print(f"{r['arm']} nfe{r['nfe']} {r['metric']:<18} {r['value']:.4f}{extra}")


if __name__ == "__main__":
    main()
