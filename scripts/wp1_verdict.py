"""WP1 verdicts pooled over the four folds (every WildPPG subject tested once); cluster = subject (n = 14)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v1_evaluate import cluster_ci  # noqa: E402

IN, OUT = ROOT / "outputs/wp1_eval", ROOT / "artifacts/wp1_wildppg"


def load(arm):
    parts = [dict(np.load(IN / f"fold{k}_arm{arm}.npz")) for k in range(4)]
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if parts[0][key].ndim == 1 and key not in ("checkpoint_sha256",)} | \
           {"fold_values": parts}


def rule(hr, f1):
    imp = (hr[0] <= -1.0 and hr[2] < 0) or (f1[0] >= 0.02 and f1[1] > 0)
    wor = (hr[0] >= 1.0 and hr[1] > 0) or (f1[0] <= -0.02 and f1[2] < 0)
    return "MIXED" if imp and wor else "IMPROVES" if imp else "WORSE" if wor else "NO MEANINGFUL CHANGE"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C, I, D = load("C"), load("I"), load("D")
    subj = C["subj"]
    assert np.array_equal(I["subj"], subj) and np.array_equal(D["subj"], subj) and len(np.unique(subj)) == 14
    ci = lambda v: cluster_ci(v, subj)  # noqa: E731
    table = {}
    for name, arr, nfe in (("PENGUIN-50", C, 50), ("PENGUIN-1", C, 1), ("iMF-1", I, 1), ("CD-1", D, 1)):
        table[name] = {m: ci(arr[f"nfe{nfe}_{m}"]) for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE")}
    for name, arr in (("iMF-1 K16", I), ("CD-1 K16", D)):
        table[name] = {"HR": ci(arr["cons16_nfe1_hr_err"])}
    h1_hr, h1_f1 = ci(I["nfe1_HR"] - C["nfe50_HR"]), ci(I["nfe1_Rpeak_F1"] - C["nfe50_Rpeak_F1"])
    h2 = ci(I["cons16_nfe1_hr_err"] - C["nfe50_HR"])
    h2b = ci(D["cons16_nfe1_hr_err"] - C["nfe50_HR"])
    h4_hr, h4_f1 = ci(D["nfe1_HR"] - I["nfe1_HR"]), ci(D["nfe1_Rpeak_F1"] - I["nfe1_Rpeak_F1"])
    subs = np.unique(subj)
    pp = lambda v: np.array([np.nanmean(v[subj == s]) for s in subs])  # noqa: E731
    wins = {"iMF-1": float(np.mean(pp(I["nfe1_HR"]) < pp(C["nfe50_HR"]))),
            "iMF-1 K16": float(np.mean(pp(I["cons16_nfe1_hr_err"]) < pp(C["nfe50_HR"]))),
            "CD-1": float(np.mean(pp(D["nfe1_HR"]) < pp(C["nfe50_HR"]))),
            "CD-1 K16": float(np.mean(pp(D["cons16_nfe1_hr_err"]) < pp(C["nfe50_HR"])))}
    verdict = {"H1_iMF_non_inferior": bool(h1_hr[2] < 1.0 and h1_f1[1] > -0.02), "H1_HR": h1_hr, "H1_F1": h1_f1,
               "H2_iMF_consensus_superior": bool(h2[2] < 0), "H2_HR": h2,
               "H2b_CD_consensus_superior": bool(h2b[2] < 0), "H2b_HR": h2b,
               "H4_CD_vs_iMF": rule(h4_hr, h4_f1), "H4_HR": h4_hr, "H4_F1": h4_f1,
               "HR_win_rate_vs_PENGUIN50_over_14_subjects": wins, "table": table,
               "fd": {"PENGUIN-50": [float(p["nfe50_pooled_FD_kanflow"]) for p in C["fold_values"]],
                      "iMF-1": [float(p["nfe1_pooled_FD_kanflow"]) for p in I["fold_values"]],
                      "CD-1": [float(p["nfe1_pooled_FD_kanflow"]) for p in D["fold_values"]]}}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items() if k != "table"}, indent=1))
    for n, t in table.items():
        print(n, {m: [round(x, 3) for x in v] for m, v in t.items()})


if __name__ == "__main__":
    main()
