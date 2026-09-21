"""KN1 verdicts: each KAN arm against the MLP arm of the same objective (docs/KN1_KAN_FFN_PREREGISTRATION.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v1_evaluate import cluster_ci  # noqa: E402

IN, OUT = ROOT / "outputs/sr1_eval", ROOT / "artifacts/kn1_kan"
PAIRS = [("IKo", "I", 1), ("IKa", "I", 1), ("CKo", "C", 50)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = {a: dict(np.load(IN / f"arm_{a}_seed42.npz")) for a in ("I", "C", "IKo", "IKa", "CKo")}
    pid = D["I"]["pid"]
    assert all(np.array_equal(d["pid"], pid) for d in D.values())
    res = {}
    for kan, ref, nfe in PAIRS:
        diff = {m: cluster_ci(D[kan][f"nfe{nfe}_{m}"] - D[ref][f"nfe{nfe}_{m}"], pid) for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE")}
        hr, f1 = diff["HR"], diff["Rpeak_F1"]
        imp = (hr[0] <= -1.0 and hr[2] < 0) or (f1[0] >= 0.02 and f1[1] > 0)
        wor = (hr[0] >= 1.0 and hr[1] > 0) or (f1[0] <= -0.02 and f1[2] < 0)
        res[f"{kan}_vs_{ref}@NFE{nfe}"] = {
            "verdict": "MIXED" if imp and wor else "IMPROVES" if imp else "WORSE" if wor else "NO MEANINGFUL CHANGE",
            "diff": diff,
            "values": {a: {m: cluster_ci(D[a][f"nfe{nfe}_{m}"], pid)[0] for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE")}
                          | {"FD": float(D[a][f"nfe{nfe}_pooled_FD_kanflow"])} for a in (ref, kan)},
            "mean_seeking_only": bool(diff["MAE"][2] < 0 and diff["RMSE"][2] < 0 and not imp)}
        if "cons16_nfe1_hr_err" in D[kan]:
            res[f"{kan}_vs_{ref}@NFE{nfe}"]["consensus_K16_HR"] = {a: cluster_ci(D[a]["cons16_nfe1_hr_err"], pid)[0] for a in (ref, kan)}
    (OUT / "verdict.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
