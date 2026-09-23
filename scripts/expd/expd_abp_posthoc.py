"""EXP-D Part B — POST-HOC diagnostics (not preregistered; written after the analysis and the amendment gates).

1. Signed bias of the generated functionals by depth (mean over blocks of Y − T*, draw 0 and all draws), generated vs
   reference pulse pressure (SBP − DBP) — to read why SBP favours shallow and DBP / MAP favour deep single samples.
2. Selected cells vs the training-median constant (paired subject bootstrap): (1,1), (1,32), (32,1), Heun-50.
3. Spread of the estimates across blocks vs the reference spread (shrinkage check, as in respiration).
Writes artifacts/exp_d_functional_generalization/abp/posthoc.json.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/expd/expd_abp_posthoc.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import expd_run as X  # noqa: E402


def main():
    z = np.load(X.RAW / "MIMIC-BP.npz"); T = z["T"]; sub = z["block_subject"]
    F = {S: z[f"F_S{S}"] for S in X.SS}; H = z["F_heun"][0]
    const = X.train_constant("MIMIC-BP", "abp")
    bt = X.Boot(sub)
    out = {"note": "POST-HOC, not preregistered", "reference": {}, "bias_by_S": {}, "vs_constant": {}, "spread": {}}
    for j, fn in enumerate(X.FUNCS["abp"]):
        out["reference"][fn] = {"mean": float(T[:, j].mean()), "sd_across_blocks": float(T[:, j].std()), "train_constant": float(const[j])}
    out["reference"]["pulse_pressure_mean"] = float((T[:, 0] - T[:, 1]).mean())
    for S in X.SS:
        d0 = F[S][0]; allm = F[S].mean(0)
        out["bias_by_S"][S] = {**{f"{fn}_bias_draw0": float((d0[:, j] - T[:, j]).mean()) for j, fn in enumerate(X.FUNCS["abp"])},
                               **{f"{fn}_bias_draw_mean": float((allm[:, j] - T[:, j]).mean()) for j, fn in enumerate(X.FUNCS["abp"])},
                               "pulse_pressure_draw0": float((d0[:, 0] - d0[:, 1]).mean())}
    out["bias_by_S"]["heun50"] = {**{f"{fn}_bias_draw0": float((H[:, j] - T[:, j]).mean()) for j, fn in enumerate(X.FUNCS["abp"])},
                                  "pulse_pressure_draw0": float((H[:, 0] - H[:, 1]).mean())}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = {"(1,1)": F[1][0], "(1,32)": F[32][0], "(32,1)": np.nanmedian(F[1][:32], 0), "(8,4)": np.nanmedian(F[4][:8], 0), "heun50": H}
    for j, fn in enumerate(X.FUNCS["abp"]):
        ec = np.abs(const[j] - T[:, j])
        out["vs_constant"][fn] = {k: X.contrast(bt, np.abs(v[:, j] - T[:, j]), ec)["diff"] for k, v in est.items()}
        out["spread"][fn] = {k: float(v[:, j].std()) for k, v in est.items()}
        out["spread"][fn]["reference"] = float(T[:, j].std())
    (X.ART / "abp" / "posthoc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
