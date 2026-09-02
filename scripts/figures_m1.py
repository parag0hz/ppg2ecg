"""M1 figures and the frozen visual atlas. Reads the committed artefacts and the cached cohort predictions."""
from __future__ import annotations

import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppg2ecg.evaluation import m1_structural as M
from ppg2ecg.evaluation.c2_cohort import SITES

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts/m1_c1_structural_audit"
F = A / "figures"; F.mkdir(parents=True, exist_ok=True)
AT = A / "visual_atlas"; AT.mkdir(parents=True, exist_ok=True)
ARMS, FS = ("B", "H25", "H50"), 128
rows = lambda n: list(csv.DictReader(open(A / n)))  # noqa: E731
COL = {"B": "tab:gray", "H25": "tab:orange", "H50": "tab:green"}

prof = rows("event_error_profiles.csv")
P = {(r["arm"], int(r["nfe"])): r for r in prof}
tau = sorted({float(r["tau_ms"]) for r in prof})
def curve(arm, nfe, key):
    d = {float(r["tau_ms"]): float(r[key]) for r in prof if r["arm"] == arm and int(r["nfe"]) == nfe}
    return np.array([d[t] for t in tau])

# 1 event error profile @NFE2
fig, ax = plt.subplots(figsize=(9, 4.6))
for a in ARMS:
    ax.plot(tau, curve(a, 2, "abs_err"), lw=1.8, color=COL[a], label=a)
ax.axvspan(-80, 80, color="tab:red", alpha=0.08, label="QRS-core (|τ|≤80 ms)")
ax.set_xlabel("τ = distance to nearest GT R-peak (ms)"); ax.set_ylabel("mean |pred − GT|  (fixed coordinates)")
ax.set_title("M1 — event-centred absolute-error profile @ NFE 2 (no smoothing, no translation)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(F / "m1_event_error_profile_nfe2.png", dpi=115); plt.close(fig)

# 2 H50 minus H25 / minus B
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(tau, curve("B", 2, "abs_err") - curve("H50", 2, "abs_err"), lw=1.8, color="tab:blue", label="Δ_B = E_B − E_H50")
ax.plot(tau, curve("H25", 2, "abs_err") - curve("H50", 2, "abs_err"), lw=1.8, color="tab:red", label="Δ_25 = E_H25 − E_H50")
ax.axhline(0, color="k", lw=1); ax.axvspan(-80, 80, color="tab:red", alpha=0.08)
ax.set_xlabel("τ (ms)"); ax.set_ylabel("positive = H50 better")
ax.set_title("M1 — H50 improvement profile @ NFE 2"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(F / "m1_h50_minus_h25_event_profile.png", dpi=115); plt.close(fig)

# 4 derivative profile
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
for a in ARMS:
    ax[0].plot(tau, curve(a, 2, "deriv_err"), lw=1.8, color=COL[a], label=a)
ax[0].axvspan(-80, 80, color="tab:red", alpha=0.08); ax[0].set_title("derivative error profile @ NFE 2")
ax[0].set_xlabel("τ (ms)"); ax[0].set_ylabel("mean |D pred − D GT|"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
ax[1].plot(tau, curve("H25", 2, "deriv_err") - curve("H50", 2, "deriv_err"), lw=1.8, color="tab:red", label="E_H25 − E_H50")
ax[1].plot(tau, curve("B", 2, "deriv_err") - curve("H50", 2, "deriv_err"), lw=1.8, color="tab:blue", label="E_B − E_H50")
ax[1].axhline(0, color="k", lw=1); ax[1].axvspan(-80, 80, color="tab:red", alpha=0.08)
ax[1].set_title("derivative improvement (positive = H50 better)"); ax[1].set_xlabel("τ (ms)")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(F / "m1_derivative_event_profile.png", dpi=115); plt.close(fig)

# 3 QRS vs background improvement
loc = rows("localization.csv")
reg = {(r["arm"], int(r["nfe"])): r for r in rows("region_metrics.csv")}
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
lbl = ["QRS-core", "peri-QRS", "background"]
keys = ["qrs_core__a2_sq", "peri_qrs__a2_sq", "background__a2_sq"]
x = np.arange(3); w = 0.26
for i, a in enumerate(ARMS):
    ax[0].bar(x + (i - 1) * w, [float(reg[(a, 2)][k]) for k in keys], w, color=COL[a], label=a)
ax[0].set_xticks(x); ax[0].set_xticklabels(lbl); ax[0].set_ylabel("mean squared error"); ax[0].legend(fontsize=9)
ax[0].set_title("region-wise squared error @ NFE 2"); ax[0].grid(alpha=0.25, axis="y")
y = np.arange(len(loc))
ax[1].barh(y - 0.15, [float(r["R_core"]) for r in loc], 0.3, label="R_core", color="tab:red")
ax[1].barh(y + 0.15, [float(r["R_background"]) for r in loc], 0.3, label="R_background", color="tab:blue")
for i, r in enumerate(loc):
    ax[1].errorbar(float(r["L"]), i, xerr=[[float(r["L"]) - float(r["lo"])], [float(r["hi"]) - float(r["L"])]],
                   fmt="ko", capsize=4, label="L = R_core − R_bg" if i == 0 else None)
ax[1].axvline(0, color="k", lw=1); ax[1].set_yticks(y); ax[1].set_yticklabels([r["family"] for r in loc])
ax[1].set_xlabel("relative error reduction, H50 vs H25 (positive = H50 better)")
ax[1].set_title("localisation contrast L"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(F / "m1_qrs_vs_background_improvement.png", dpi=115); plt.close(fig)

# 5 frequency decomposition
sp = {(r["arm"], int(r["nfe"])): r for r in rows("spectral_metrics.csv")}
bands = [b[0] for b in M.BANDS]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
for i, a in enumerate(ARMS):
    ax[0].bar(np.arange(4) + (i - 1) * w, [float(sp[(a, 2)][f"{b}__err_energy"]) for b in bands], w, color=COL[a], label=a)
    ax[1].bar(np.arange(4) + (i - 1) * w, [float(sp[(a, 2)][f"{b}__ratio_dev"]) for b in bands], w, color=COL[a], label=a)
for a_, t_ in zip(ax, ("reconstruction-error spectral energy", "|pred/GT energy ratio − 1|")):
    a_.set_xticks(np.arange(4)); a_.set_xticklabels([f"{b}\n{lo}-{hi} Hz" for (b, lo, hi) in M.BANDS], fontsize=8)
    a_.set_title(t_ + "  @ NFE 2"); a_.legend(fontsize=9); a_.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(F / "m1_frequency_error_decomposition.png", dpi=115); plt.close(fig)

# 6 QRS structure metrics: direct errors vs ratio deviations
bt = rows("paired_bootstrap.csv")
def g(cmp_, m):
    r = next(x for x in bt if x["comparison"] == cmp_ and x["metric"] == m)
    return float(r["point"]), float(r["lo"]), float(r["hi"]), r["verdict"]
DIRECT = ["qrs_core__a2_sq", "qrs_core__a1_abs", "qrs_core__a3_dabs", "qrs_rmse_core",
          "qrs_deriv_rmse", "qrs_curvature_err", "raw_corr"]
RATIO = ["qrs_energy_dev", "qrs_ptp_dev", "qrs_slope_dev", "qrs_maxderiv_dev"]
fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
for a_, grp, ttl in ((ax[0], DIRECT, "DIRECT fixed-coordinate errors"), (ax[1], RATIO, "aggregate RATIO deviations")):
    y = np.arange(len(grp))
    for off, cmp_, c in ((-0.18, "H50-vs-H25@NFE2", "tab:red"), (0.18, "H50-vs-B@NFE2", "tab:blue")):
        pts = [g(cmp_, m) for m in grp]
        a_.errorbar([p[0] for p in pts], y + off,
                    xerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]], fmt="o", capsize=3,
                    color=c, label=cmp_)
    a_.axvline(0, color="k", lw=1); a_.set_yticks(y); a_.set_yticklabels(grp, fontsize=9)
    a_.set_title(ttl + "   (positive = H50 better)", fontsize=11); a_.grid(alpha=0.3); a_.legend(fontsize=8)
fig.suptitle("M1 — the split that decides the verdict: ratio deviations improve while direct QRS errors do not")
fig.tight_layout(); fig.savefig(F / "m1_qrs_structure_metrics.png", dpi=115); plt.close(fig)

# 7 site-wise
st = rows("site_metrics.csv")
SK = ["qrs_core__a2_sq", "background__a2_sq", "qrs_core__a3_dabs", "qrs_energy_dev", "qrs_ptp_dev", "raw_corr"]
fig, axes = plt.subplots(2, 3, figsize=(16, 7))
for a_, k in zip(axes.ravel(), SK):
    for i, arm in enumerate(ARMS):
        v = [float(next(r for r in st if r["site"] == s and r["arm"] == arm)[k]) for s in SITES]
        a_.bar(np.arange(4) + (i - 1) * w, v, w, color=COL[arm], label=arm)
    a_.set_xticks(np.arange(4)); a_.set_xticklabels(SITES, fontsize=9); a_.set_title(k, fontsize=10)
    a_.grid(alpha=0.25, axis="y")
axes.ravel()[0].legend(fontsize=8)
fig.suptitle("M1 — site-wise structural quantities @ NFE 2 (exploratory)")
fig.tight_layout(); fig.savefig(F / "m1_sitewise_structure_effect.png", dpi=115); plt.close(fig)

# 8 NFE interaction
ni = rows("nfe_interaction.csv")
fig, ax = plt.subplots(figsize=(9.5, 4.6))
y = np.arange(len(ni))
ax.barh(y - 0.2, [float(r["E2"]) for r in ni], 0.35, label="E2 = H50−B @ NFE2", color="tab:green")
ax.barh(y + 0.2, [float(r["E4"]) for r in ni], 0.35, label="E4 = H50−B @ NFE4", color="tab:olive")
for i, r in enumerate(ni):
    ax.errorbar(float(r["D"]), i, xerr=[[float(r["D"]) - float(r["D_lo"])], [float(r["D_hi"]) - float(r["D"])]],
                fmt="ko", capsize=4, label="D = E2 − E4" if i == 0 else None)
ax.axvline(0, color="k", lw=1); ax.set_yticks(y); ax.set_yticklabels([r["metric"] for r in ni], fontsize=9)
ax.set_title("M1 — NFE 2 vs NFE 4 localisation of the H50 effect"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(F / "m1_nfe2_vs_nfe4_effect.png", dpi=115); plt.close(fig)

# ---------------- visual atlas ----------------
z = np.load(A / "_cohort_preds.npz")
coh = json.loads((A / "cohort_manifest.json").read_text())
order = [(s, site, i) for s in ("an0", "k2s") for site in SITES for i in coh["indices_into_frozen_subset"][s][site]]
gt, ppg = z["gt"], z["ppg"]
t = np.arange(1024) / FS
ROWS = [("PPG", None), ("GT ECG", None)] + [(f"{a}@NFE{n}", (a, n)) for n in (2, 4) for a in ARMS]
made = 0
for sheet_i, (subj, site) in enumerate([(s, si) for s in ("an0", "k2s") for si in SITES]):
    idxs = [k for k, (s, si, _) in enumerate(order) if s == subj and si == site]
    fig, axes = plt.subplots(len(ROWS), len(idxs), figsize=(3.1 * len(idxs), 1.15 * len(ROWS)), sharex=True)
    for c, k in enumerate(idxs):
        pk = np.asarray(np.nonzero(np.zeros(1))[0])
        for r, (name, key) in enumerate(ROWS):
            ax_ = axes[r, c]
            y = ppg[k] if key is None and r == 0 else (gt[k] if key is None else z[f"{key[0]}_{key[1]}"][k])
            ax_.plot(t, y, lw=0.6, color="k" if key is None else COL[key[0]])
            ax_.set_ylim(-1.5, 1.5) if r else None
            ax_.grid(alpha=0.15); ax_.set_yticks([])
            if c == 0:
                ax_.set_ylabel(name, fontsize=7.5)
        axes[0, c].set_title(f"{subj}/{site} #{order[k][2]}", fontsize=8)
    for ax_ in axes[-1]:
        ax_.set_xlabel("s", fontsize=7)
    fig.suptitle(f"M1 visual atlas — {subj} / {site} (frozen metadata-only cohort, no translation, no selection on output)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(AT / f"contact_{subj}_{site}.png", dpi=100); plt.close(fig)
    made += 1
print(f"wrote 8 figures and {made} atlas contact sheets")
