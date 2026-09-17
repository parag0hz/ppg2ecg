"""SR1 verdicts: the three frozen claims, per seed, then the replication tally (docs/SR1_SEED_REPLICATION_PREREGISTRATION.md)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v1_evaluate import cluster_ci  # noqa: E402

IN, OUT = ROOT / "outputs/sr1_eval", ROOT / "artifacts/sr1_seeds"
SEEDS = (42, 1, 2)
MET = ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE")


def tally(flags):
    n = sum(flags)
    return f"{'REPLICATED' if n == 3 else 'PARTIAL' if n == 2 else 'NOT REPLICATED'} ({n}/3)"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = {(a, s): dict(np.load(IN / f"arm_{a}_seed{s}.npz")) for a in ("C", "I", "S") for s in SEEDS}
    pid = D[("C", 42)]["pid"]
    assert all(np.array_equal(d["pid"], pid) for d in D.values())
    rows, claims = [], {"H1_iMF_non_inferior": [], "H2_consensus_superior": [], "H3_smallDiT_HR_superior": []}
    for s in SEEDS:
        C, I, S = D[("C", s)], D[("I", s)], D[("S", s)]
        for a, d in (("C", C), ("I", I), ("S", S)):
            for nfe in (1, 2, 4, 50):
                if f"nfe{nfe}_HR" not in d:
                    continue
                for m in MET:
                    v, lo, hi = cluster_ci(d[f"nfe{nfe}_{m}"], pid)
                    rows.append(dict(seed=s, arm=a, nfe=nfe, metric=m, value=v, ci_lo=lo, ci_hi=hi))
                for m in ("FD_kanflow", "Micro_F1"):
                    rows.append(dict(seed=s, arm=a, nfe=nfe, metric=m, value=float(d[f"nfe{nfe}_pooled_{m}"])))
                rows.append(dict(seed=s, arm=a, nfe=nfe, metric="ms_per_window", value=float(d[f"nfe{nfe}_ms_per_window"])))
            v, lo, hi = cluster_ci(d["cons16_nfe1_hr_err"], pid)
            rows.append(dict(seed=s, arm=a, nfe=1, metric="HR_consensus_K16", value=v, ci_lo=lo, ci_hi=hi))
        h1_hr = cluster_ci(I["nfe1_HR"] - C["nfe50_HR"], pid)
        h1_f1 = cluster_ci(I["nfe1_Rpeak_F1"] - C["nfe50_Rpeak_F1"], pid)
        h2 = cluster_ci(I["cons16_nfe1_hr_err"] - C["nfe50_HR"], pid)
        h3_hr = cluster_ci(S["nfe1_HR"] - C["nfe50_HR"], pid)
        h3_f1 = cluster_ci(S["nfe1_Rpeak_F1"] - C["nfe50_Rpeak_F1"], pid)
        claims["H1_iMF_non_inferior"].append(bool(h1_hr[2] < 1.0 and h1_f1[1] > -0.02))
        claims["H2_consensus_superior"].append(bool(h2[2] < 0))
        claims["H3_smallDiT_HR_superior"].append(bool(h3_hr[2] < 0 and h3_f1[1] > -0.02))
        for name, (pt, lo, hi) in (("H1_HR", h1_hr), ("H1_F1", h1_f1), ("H2_HR", h2), ("H3_HR", h3_hr), ("H3_F1", h3_f1)):
            rows.append(dict(seed=s, arm="contrast", nfe=1, metric=name, value=pt, ci_lo=lo, ci_hi=hi))
    verdict = {k: {"per_seed": {str(s): f for s, f in zip(SEEDS, v)}, "verdict": tally(v)} for k, v in claims.items()}
    spread = {}
    for a in ("C", "I", "S"):
        for key in ("nfe1_HR", "nfe1_Rpeak_F1", "cons16_nfe1_hr_err", "nfe50_HR"):
            vals = [cluster_ci(D[(a, s)][key], pid)[0] for s in SEEDS if key in D[(a, s)]]
            if vals:
                spread[f"{a}.{key}"] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)),
                                        "per_seed": {str(s): v for s, v in zip(SEEDS, vals)}}
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "nfe", "metric", "value", "ci_lo", "ci_hi"]); w.writeheader(); w.writerows(rows)
    (OUT / "verdict.json").write_text(json.dumps({"claims": verdict, "seed_spread": spread,
                                                  "cfg_per_seed": {str(s): [float(x) for x in D[("S", s)]["cfg"]] for s in SEEDS}}, indent=1))
    print(json.dumps(verdict, indent=1))
    print(json.dumps(spread, indent=1))


if __name__ == "__main__":
    main()
