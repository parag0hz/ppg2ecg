"""LW1 verdicts against arm I (docs/LW1_WEIGHTED_AUX_LOSS_PREREGISTRATION.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v1_evaluate import cluster_ci  # noqa: E402

OUT = ROOT / "artifacts/lw1_aux_loss"
KEYS = ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE", "PCC")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = {a: dict(np.load(ROOT / f"outputs/lw1_eval/arm_{a}.npz")) for a in ("I", "W0", "W5", "W10", "SP")}
    pid = D["I"]["pid"]
    res = {}
    for a, d in D.items():
        row = {m: cluster_ci(d[f"nfe1_{m}"], pid)[0] for m in KEYS}
        row["FD"] = float(d["nfe1_pooled_FD_kanflow"])
        row["consensus_K16_HR"] = cluster_ci(d["cons16_nfe1_hr_err"], pid)[0]
        row["consensus_gain"] = row["HR"] - row["consensus_K16_HR"]
        row["hr_sd_across_samples"] = cluster_ci(d["hr_sd_across_samples"], pid)[0]
        if a != "I":
            diff = {m: cluster_ci(d[f"nfe1_{m}"] - D["I"][f"nfe1_{m}"], pid) for m in KEYS}
            diff["consensus_K16_HR"] = cluster_ci(d["cons16_nfe1_hr_err"] - D["I"]["cons16_nfe1_hr_err"], pid)
            hr, f1 = diff["HR"], diff["Rpeak_F1"]
            imp = (hr[0] <= -1.0 and hr[2] < 0) or (f1[0] >= 0.02 and f1[1] > 0)
            wor = (hr[0] >= 1.0 and hr[1] > 0) or (f1[0] <= -0.02 and f1[2] < 0)
            row["diff_vs_I"] = diff
            row["verdict"] = "MIXED" if imp and wor else "IMPROVES" if imp else "WORSE" if wor else "NO MEANINGFUL CHANGE"
        res[a] = row
    (OUT / "verdict.json").write_text(json.dumps(res, indent=1))
    for a, r in res.items():
        print(a, r.get("verdict", "reference"), {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
