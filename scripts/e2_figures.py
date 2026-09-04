"""E2 figures (preregistration section 25) — measurement-contract figures only, from frozen artifacts.

Nothing is recomputed and no generator is run. No waveform cherry-picking.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/e2_evaluation_contract"
OUT = ART / "figures"
COL = {"AXIS_A": "#2166ac", "AXIS_B": "#b2182b", "AXIS_C": "#1a9850", "JOINT": "#762a83",
       "JOINT_EVENT": "#8c6d31", "ADHERENCE": "#666666"}
ARMS = ("ORACLE", "JITTER_8", "MISS1", "EXTRA1", "R1-SCHEDULE")


def rd(name):
    p = ART / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(row, k):
    try:
        return float(row.get(k, ""))
    except (TypeError, ValueError):
        return np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((ART / "contract_v1.json").read_text())
    M = {r["arm"]: r for r in rd("contract_validation_metrics.csv")}
    T = {r["arm"]: r for r in rd("topology_validation.csv")}
    B = {(r["contrast"], r["metric"]): r for r in rd("contract_validation_bootstrap.csv")}

    # FIG 1 — four-axis schematic straight from the contract
    fams = ["AXIS_A", "AXIS_B", "AXIS_C", "JOINT", "JOINT_EVENT", "ADHERENCE"]
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.axis("off")
    for i, fam in enumerate(fams):
        ids = [m for m, d in contract["metrics"].items() if d["family"] == fam]
        x = 0.02 + (i % 3) * 0.33
        y = 0.94 - (i // 3) * 0.48
        ax.add_patch(plt.Rectangle((x, y - 0.40), 0.30, 0.40, transform=ax.transAxes,
                                   facecolor=COL[fam], alpha=0.10, edgecolor=COL[fam], lw=1.6))
        ax.text(x + 0.015, y - 0.035, contract["families"][fam]["label"], transform=ax.transAxes,
                fontsize=10, weight="bold", color=COL[fam])
        ax.text(x + 0.015, y - 0.075, contract["families"][fam]["question"], transform=ax.transAxes,
                fontsize=7.5, style="italic", wrap=True)
        for j, mid in enumerate(sorted(ids)):
            d = contract["metrics"][mid]
            ax.text(x + 0.02, y - 0.105 - j * 0.026,
                    f"{mid}  {d['name']}  ({'↓' if d['direction'] == 'lower_better' else ('↑' if d['direction'] == 'higher_better' else '·')})",
                    transform=ax.transAxes, fontsize=6.6)
    ax.text(0.02, 0.02, "Never merged. GT-ANCHORED JOINT STRUCTURE is never called 'pure morphology'; the F1 "
            "family is never called 'pure timing accuracy'; T4/T6/T7/T8 are never placement evidence.\n"
            "Prohibited comparison: own-centre T6 vs GT-anchored derivative RMSE (cross-functional). Use the "
            "same-functional alignment sensitivity D1/D2/D3 instead.",
            transform=ax.transAxes, fontsize=7.5, color="0.25")
    ax.set_title(f"FIG 1 — frozen event-geometry evaluation contract ({contract['contract_version']})", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig1_four_axis_schematic.png", dpi=140); plt.close(fig)

    # FIG 2 — ORACLE vs JITTER_8: the placement axis must respond, own-centre must not
    if M:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
        for a, (k, lab, col) in zip(axs, (("B2_mae_ms", "schedule timing MAE (ms)", COL["AXIS_B"]),
                                          ("C6_own_local_deriv_rmse", "own-centre local deriv RMSE", COL["AXIS_C"]),
                                          ("J2_gt_local_deriv_rmse", "GT-anchored local deriv RMSE", COL["JOINT"]))):
            v = [f(M[x], k) for x in ("ORACLE", "JITTER_8")]
            a.bar(["ORACLE", "JITTER_8"], v, color=col, alpha=0.9)
            for i, val in enumerate(v):
                a.annotate(f"{val:.4f}", (i, val), ha="center", va="bottom", fontsize=8)
            a.set_ylabel(lab, fontsize=8); a.grid(alpha=0.2, axis="y")
        fig.suptitle("FIG 2 — placement degradation is visible in the schedule and GT-anchored axes, "
                     "not in own-centre shape", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "fig2_oracle_vs_jitter8.png", dpi=140); plt.close(fig)

        # FIG 3 — own-centre T6/T7 with 95 % CIs of the damage vs ORACLE
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
        arms4 = ("ORACLE", "JITTER_8", "MISS1", "EXTRA1")
        for ax_, key, lab in zip(axs, ("C2_own_T6", "C3_own_T7"), ("own-centre T6 nAE", "own-centre T7 nAE")):
            vals = [f(M[a], key) for a in arms4]
            lo, hi = [], []
            for a in arms4:
                r = B.get((f"{a}_vs_ORACLE", key))
                if r:
                    lo.append(f(r, "point") - f(r, "lo")); hi.append(f(r, "hi") - f(r, "point"))
                else:
                    lo.append(np.nan); hi.append(np.nan)   # a missing contrast must be visible, not zero-width
            ax_.bar(range(4), vals, yerr=[lo, hi], capsize=4,
                    color=[COL["AXIS_C"], COL["AXIS_B"], COL["AXIS_A"], COL["AXIS_A"]], alpha=0.9)
            ax_.set_xticks(range(4)); ax_.set_xticklabels(arms4, rotation=12, fontsize=8)
            ax_.set_ylabel(lab + " (lower better)", fontsize=8); ax_.grid(alpha=0.2, axis="y")
        fig.suptitle("FIG 3 — the topology axis responds where the placement axis does not "
                     "(whiskers: 95 % CI of the damage vs ORACLE)", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "fig3_topology_response.png", dpi=140); plt.close(fig)

    # FIG 4 — R1 error decomposition
    if T and "R1-SCHEDULE" in T:
        r = T["R1-SCHEDULE"]
        keys = [k for k in r if k.startswith("T") and "_" in k and k[1].isdigit()]
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
        n = [int(float(r[k])) for k in keys]
        axs[0].bar(range(len(keys)), n, color=COL["AXIS_A"], alpha=0.9)
        axs[0].set_xticks(range(len(keys)))
        axs[0].set_xticklabels([k.split("_")[0] for k in keys], fontsize=8)
        for i, v in enumerate(n):
            axs[0].annotate(f"{v}\n{v / sum(n):.1%}", (i, v), ha="center", va="bottom", fontsize=7)
        axs[0].set_ylabel("windows"); axs[0].set_title("AXIS A — topology", fontsize=9)
        for ax_, k, lab, col in ((axs[1], "B2_mae_ms", "matched timing MAE (ms)", COL["AXIS_B"]),
                                 (axs[2], "AD_F1_50", "P→S adherence F1@50", COL["ADHERENCE"]),
                                 (axs[3], "C2_own_T6", "own-centre T6 nAE", COL["AXIS_C"])):
            vals = [f(M[a], k) if a in M else np.nan for a in ARMS]
            ax_.bar(range(len(ARMS)), vals, color=col, alpha=0.9)
            ax_.set_xticks(range(len(ARMS)))
            ax_.set_xticklabels([a.replace("-SCHEDULE", "") for a in ARMS], rotation=20, fontsize=7)
            ax_.set_ylabel(lab, fontsize=8); ax_.grid(alpha=0.2, axis="y")
        if "ORACLE" in M and "AD_F1_50" in M["ORACLE"]:
            axs[2].axhline(0.90, color="0.35", ls="--", lw=1.0)
        fig.suptitle("FIG 4 — the natural R1 schedule decomposed on every axis at once "
                     "(no composite score)", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "fig4_r1_decomposition.png", dpi=140); plt.close(fig)

    (OUT / "figures_manifest.json").write_text(json.dumps(
        {"figures": sorted(p.name for p in OUT.glob("*.png")), "source": "frozen E2 artifacts only",
         "recomputed": False, "generator_run": False, "waveform_cherry_picking": False}, indent=2))
    print(f"[fig] {len(sorted(OUT.glob('*.png')))} figures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
