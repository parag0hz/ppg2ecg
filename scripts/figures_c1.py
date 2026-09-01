"""C1 figures (preregistration section 21). Reads only the committed CSV artefacts; no model, no data."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppg2ecg.flow.interval_exposure import ARMS, sample_tr_c1

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts/c1_interval_exposure"
F = A / "figures"; F.mkdir(parents=True, exist_ok=True)
GAP = ["M1_qrs_e_dev", "M2_p2p_dev", "M3_qrs_rmse", "M4_rmse"]
LBL = {"M1_qrs_e_dev": "M1  |QRS-E−1|", "M2_p2p_dev": "M2  |p2p−1|",
       "M3_qrs_rmse": "M3  QRS RMSE", "M4_rmse": "M4  RMSE",
       "M5_raw_corr": "M5  raw corr", "M6_slope_dev": "M6  |slope−1|"}
rows = lambda n: list(csv.DictReader(open(A / n)))  # noqa: E731

# 1 — sampler exposure
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for a_, arm in zip(ax, ARMS):
    t, r, _ = sample_tr_c1(400_000, torch.Generator().manual_seed(20260901), arm=arm,
                           p_mean=-0.4, p_std=1.0, data_proportion=0.5)
    h = (t - r).numpy().ravel()
    a_.hist(h[h > 0], bins=120, color="tab:blue", alpha=0.85)
    a_.axvline(0.5, color="tab:red", ls="--", lw=1.2, label="h = 0.5 (uniform NFE 2)")
    a_.axvline(0.25, color="tab:orange", ls="--", lw=1.2, label="h = 0.25 (uniform NFE 4)")
    a_.set_title(f"{arm}   P(h=0) = {np.mean(h == 0):.3f}   P(h≥0.5) = {np.mean(h >= 0.5):.4f}", fontsize=10)
    a_.set_xlabel("h = t − r  (positive-h draws only)"); a_.legend(fontsize=8)
ax[0].set_ylabel("count")
fig.suptitle("C1 — training-time interval exposure. The exact-h=0 branch (50 %) is preserved in every arm "
             "and is omitted from these histograms.")
fig.tight_layout(); fig.savefig(F / "c1_sampler_exposure.png", dpi=115); plt.close(fig)

# 2 — gap closure
m = {(r_["arm"], int(r_["nfe"])): r_ for r_ in rows("stage2_metrics.csv")}
cl = {r_["metric"]: r_ for r_ in rows("gap_closure.csv")}
fig, ax = plt.subplots(1, 4, figsize=(17, 4.3))
for a_, k in zip(ax, GAP):
    vals = [float(m[("B", 2)][k]), float(m[("H25", 2)][k]), float(m[("H50", 2)][k])]
    a_.bar(["B@2", "H25@2", "H50@2"], vals, color=["gray", "tab:orange", "tab:green"])
    a_.axhline(float(m[("B", 4)][k]), color="k", ls="--", lw=1.4, label="B @ NFE 4 (target)")
    a_.axhline(float(m[("B", 2)][k]), color="gray", ls=":", lw=1.2, label="B @ NFE 2 (start)")
    a_.set_title(f"{LBL[k]}   (lower better)\nclosure  H25 {float(cl[k]['C_H25']):+.2f}   "
                 f"H50 {float(cl[k]['C_H50']):+.2f}", fontsize=9.5)
    a_.legend(fontsize=7.5); a_.grid(alpha=0.25, axis="y")
fig.suptitle("C1 — NFE-2 gap closure against the C1-B replay target. Closure > 1 means NFE 2 passed the "
             "baseline's NFE-4 value on that metric.")
fig.tight_layout(); fig.savefig(F / "c1_nfe2_gap_closure.png", dpi=115); plt.close(fig)

# 3 — paired effects vs baseline
b = rows("stage2_paired.csv")
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
for a_, arm in zip(ax, ("H25", "H50")):
    sel = [r_ for r_ in b if r_["comparison"] == f"{arm}-vs-B@NFE2"]
    y = np.arange(len(sel))
    pt = [float(r_["point"]) for r_ in sel]
    a_.errorbar(pt, y, xerr=[[p - float(r_["lo"]) for p, r_ in zip(pt, sel)],
                             [float(r_["hi"]) - p for p, r_ in zip(pt, sel)]], fmt="o", capsize=4)
    a_.axvline(0, color="k", lw=1)
    a_.set_yticks(y); a_.set_yticklabels([LBL.get(r_["metric"], r_["metric"]) for r_ in sel], fontsize=9)
    a_.set_title(f"{arm} vs C1-B at NFE 2   (positive = intervention better)", fontsize=10.5)
    a_.grid(alpha=0.3)
fig.suptitle("C1 — paired subject-stratified bootstrap, 2000 resamples, seed 20260901")
fig.tight_layout(); fig.savefig(F / "c1_paired_effects.png", dpi=115); plt.close(fig)

# 4 — specificity
sp = rows("specificity.csv")
fig, a_ = plt.subplots(figsize=(8.6, 4.4))
y = np.arange(len(sp)); pt = [float(r_["point"]) for r_ in sp]
a_.errorbar(pt, y, xerr=[[p - float(r_["lo"]) for p, r_ in zip(pt, sp)],
                         [float(r_["hi"]) - p for p, r_ in zip(pt, sp)]], fmt="o", capsize=4, color="tab:green")
a_.axvline(0, color="k", lw=1)
a_.set_yticks(y); a_.set_yticklabels([LBL[r_["metric"]] for r_ in sp])
a_.set_xlabel("H50 improvement − H25 improvement at NFE 2   (positive = H50 more beneficial)")
a_.set_title("C1 — specificity: difference-of-improvement, H50 vs H25"); a_.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(F / "c1_h50_vs_h25_specificity.png", dpi=115); plt.close(fig)
print("wrote 4 figures to", F)
