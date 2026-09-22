"""DW2 part B (HR half): diversity by depth and consensus gain from the DW1 per-sample HR matrices. No GPU."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

RAW, OUT = ROOT / "outputs/dw1_raw", ROOT / "artifacts/dw2_mechanism"
STEPS, B = (1, 2, 4, 8, 16, 32), 32
NAME = {"I": "iMF", "D": "consistency distillation", "C": "PENGUIN (Euler)"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, Y, pid = VM.load("test")
    ref = np.load(ROOT / "outputs/db1_hr_regressor/test_errors.npz")["ref_hr"] if (ROOT / "outputs/db1_hr_regressor/test_errors.npz").exists() else V.hr_batch(ProcessPoolExecutor(8), Y)
    rows = []
    for arm in ("I", "D", "C"):
        for S in STEPS:
            H = np.load(RAW / f"hr_{arm}{S}.npy").astype(float)          # [K, n], K = 32 / S
            K = H.shape[0]
            with np.errstate(all="ignore"):
                sd = np.nanstd(H, 0) if K > 1 else np.full(H.shape[1], np.nan)
                single = np.abs(H[0] - ref)
                cons = np.abs(np.nanmedian(H, 0) - ref)
                single_mean = np.nanmean(np.abs(H - ref), 0)              # single-sample error averaged over the K draws
            div = V.cluster_ci(sd, pid) if K > 1 else (np.nan, np.nan, np.nan)
            e1, ek = V.cluster_ci(single_mean, pid), V.cluster_ci(cons, pid)
            gain = V.cluster_ci(single_mean - cons, pid) if K > 1 else (0.0, 0.0, 0.0)
            rows.append(dict(model=NAME[arm], S=S, K=K, hr_sd_across_samples=div[0], sd_lo=div[1], sd_hi=div[2],
                             single_err=e1[0], consensus_err=ek[0], gain=gain[0], gain_lo=gain[1], gain_hi=gain[2],
                             frac_windows_all_samples_identical_hr=float(np.mean(sd < 1e-6)) if K > 1 else np.nan))
            print(f"[dw2-hr] {NAME[arm]:26s} S={S:2d} K={K:2d}  HR SD {div[0]:6.2f}  single {e1[0]:.3f} -> K-median {ek[0]:.3f}  gain {gain[0]:+.3f} [{gain[1]:+.3f}, {gain[2]:+.3f}]", flush=True)
    with open(OUT / "hr_diversity_by_depth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    peng = {r["S"]: r["hr_sd_across_samples"] for r in rows if r["model"].startswith("PENGUIN")}
    rule = {"penguin_sd_S1": peng[1], "penguin_sd_S4": peng[4], "ratio_S1_over_S4": peng[1] / peng[4] if peng[4] else None,
            "S1_has_no_useful_diversity": bool(peng[1] < peng[4] / 3)}
    (OUT / "hr_rule.json").write_text(json.dumps(rule, indent=1))
    print(json.dumps(rule, indent=1))


if __name__ == "__main__":
    main()
