"""O1 figures (preregistration section 18). Pure plotting from the frozen artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ppg2ecg.evaluation import o1_targets as OT  # noqa: E402
from ppg2ecg.probes import r1_cohort as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o1_component_extractability"
FIG = ART / "figures"
LAB = [f"{OT.TARGET_IDS[t]} {t}" for t in OT.TARGETS]


def rows(n):
    return list(csv.DictReader(open(ART / n)))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def heat(M, cols, title, fname, fmt="{:.3f}", cmap="viridis_r", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(1.5 + 1.35 * len(cols), 0.55 * len(LAB) + 2.2))
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(LAB)), LAB, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if (im.norm(M[i, j]) > 0.55) else "black")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    sk = {r["target"]: r for r in rows("extractability_skill.csv")}
    cl = {r["target"]: r for r in rows("component_classification.csv")}

    cols = ["B0", "B1", "B2 rhythm", "TRUE", "SS-SHUFFLE", "XS-SHUFFLE"]
    M = np.array([[f(sk[t]["nMAE_B0"]), f(sk[t]["nMAE_B1"]), f(sk[t]["nMAE_B2"]), f(sk[t]["nMAE_TRUE"]),
                   f(sk[t]["nMAE_SS"]), f(sk[t]["nMAE_XS"])] for t in OT.TARGETS])
    heat(np.clip(M, 0, 0.75), cols, "O1 — normalised MAE (MAE / train IQR), validation an0+k2s\nlower is better; B2 of T3 is clipped (1.189)",
         "fig1_component_arm_nmae.png")

    pm = rows("probe_metrics.csv")
    S = np.array([[np.median([f(x["spearman"]) for x in pm if x["target"] == t and x["arm"] == a])
                   for a in ("TRUE", "SS-SHUFFLE", "XS-SHUFFLE")] for t in OT.TARGETS])
    heat(S, ["TRUE", "SS-SHUFFLE", "XS-SHUFFLE"], "O1 — Spearman rho (median over seeds 40/42/44)",
         "fig2_component_spearman.png", fmt="{:+.2f}", cmap="RdBu_r", vmin=-0.9, vmax=0.9)

    sm = {(r["target"], r["site"]): r for r in rows("site_extractability.csv")}
    W = np.array([[f(sm[(t, s)]["Skill_W"]) for s in C.SITES] for t in OT.TARGETS])
    heat(W, list(C.SITES), "O1 — window-specific skill  Skill_W = 1 - nMAE(TRUE)/nMAE(SS-SHUFFLE)  per PPG site",
         "fig3_site_skill.png", fmt="{:+.2f}", cmap="RdBu_r", vmin=-0.8, vmax=0.8)

    cr = {(r["target"], r["condition"]): r for r in rows("corruption_transfer.csv")}
    conds = ("LP_1.25Hz", "SNR_0dB", "DROP_2.0s", "SHUFFLED", "NULL")
    Cm = np.array([[f(sk[t]["nMAE_TRUE"])] + [f(cr[(t, c)]["nMAE"]) for c in conds] for t in OT.TARGETS])
    heat(Cm, ["CLEAN"] + list(conds), "O1 — normalised MAE under the frozen Q1 corruptions (no retraining)",
         "fig4_corruption_nmae.png")

    q = rows("extractability_utilization_map.csv")
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for r in q:
        t = r["component"]
        x = f(sk[t]["Skill_W"])
        y = f(r["utilization_effect"])
        if not np.isfinite(y):
            ax.scatter(x, 0, marker="x", c="grey", s=70)
            ax.annotate(f"{OT.TARGET_IDS[t]} (N/A)", (x, 0), fontsize=8, xytext=(4, 4), textcoords="offset points", color="grey")
            continue
        col = {"Q-A": "tab:green", "Q-B": "tab:red", "Q-C": "tab:orange", "Q-D": "tab:grey"}[r["quadrant"][:3]]
        ax.scatter(x, y, c=col, s=90)
        ax.annotate(f"{OT.TARGET_IDS[t]} {t.replace('median_QRS_', '').replace('_', ' ')}", (x, y), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("direct window-specific extractability   Skill_W = 1 - nMAE(TRUE)/nMAE(SS-SHUFFLE)")
    ax.set_ylabel("generator utilization effect   error(SHUFFLED) - error(CLEAN)")
    ax.set_title("O1 — extractability x generator utilization\ngreen Q-A, red Q-B candidate underutilization, orange Q-C, grey Q-D / N/A", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIG / "fig5_extractability_utilization.png", dpi=130); plt.close(fig)

    var = {r["target"]: r for r in rows("target_variability.csv")}
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for t in OT.TARGETS:
        b, w = f(var[t]["between_subject_variance"]), f(var[t]["within_subject_variance"])
        ax.scatter(b, w, s=80, c="tab:blue")
        ax.annotate(f"{OT.TARGET_IDS[t]}", (b, w), fontsize=9, xytext=(5, 4), textcoords="offset points")
    lim = [min(f(var[t]["between_subject_variance"]) for t in OT.TARGETS) * 0.5,
           max(f(var[t]["within_subject_variance"]) for t in OT.TARGETS) * 2]
    ax.plot(lim, lim, "k--", lw=0.8, label="within = between")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_xlabel("between-subject variance"); ax.set_ylabel("within-subject variance")
    ax.set_title("O1 — target variability (once per unique ECG window)", fontsize=9)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout(); fig.savefig(FIG / "fig6_target_variability.png", dpi=130); plt.close(fig)
    print(f"[figures] 6 written to {FIG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
