"""E3 figures — drawn only from frozen E3 artifacts. Figures for stages that never ran are not produced."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/e3_beat_set_first"
OUT = ART / "figures"
COL = {"R1-0.35": "#2166ac", "R1-TRAIN-THRESH": "#8c6d31", "ORACLE-COUNT-R1": "#b2182b",
       "E3-RIDGE-COUNT": "#1a9850"}


def rd(name):
    p = ART / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(r, k):
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    M = {r["arm"]: r for r in rd("oracle_count_schedule_metrics.csv")}
    for r in rd("schedule_metrics.csv"):
        M[r["arm"]] = r
    arms = [a for a in ("R1-0.35", "R1-TRAIN-THRESH", "ORACLE-COUNT-R1", "E3-RIDGE-COUNT") if a in M]
    boot = {(r["contrast"], r["metric"]): r for r in rd("oracle_count_bootstrap.csv") + rd("schedule_bootstrap.csv")}

    # FIG 1 — event-set comparison
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, k, lab in zip(axs, ("A5_exact_set_fraction", "T2_frac", "T3_frac"),
                          ("exact-set fraction (higher better)", "T2 undercount fraction",
                           "T3 overcount fraction")):
        v = [f(M[a], k) for a in arms]
        ax.bar(range(len(arms)), v, color=[COL[a] for a in arms], alpha=0.9)
        for i, x in enumerate(v):
            ax.annotate(f"{x:.4f}", (i, x), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=15, fontsize=7)
        ax.set_ylabel(lab, fontsize=8); ax.grid(alpha=0.2, axis="y")
    fig.suptitle("FIG 1 — event set: the correct count removes T2/T3 entirely but does not make the set exact",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig1_event_set.png", dpi=140); plt.close(fig)

    # FIG 2 — missing vs spurious trade-off
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for a in arms:
        ax.scatter([f(M[a], "A4_spurious_fraction")], [f(M[a], "A3_missing_fraction")], s=140,
                   color=COL[a], label=a, zorder=3)
        ax.annotate(a, (f(M[a], "A4_spurious_fraction"), f(M[a], "A3_missing_fraction")), fontsize=8,
                    xytext=(7, 5), textcoords="offset points")
    lim = max(max(f(M[a], "A4_spurious_fraction") for a in arms),
              max(f(M[a], "A3_missing_fraction") for a in arms)) * 1.25
    ax.plot([0, lim], [0, lim], color="0.6", ls=":", lw=1.0)
    ax.annotate("count-constrained arms sit on this line:\nM == K forces missing == spurious",
                (lim * 0.52, lim * 0.62), fontsize=7.5, color="0.35")
    ax.set_xlabel("A4 spurious fraction (lower better)"); ax.set_ylabel("A3 missing fraction (lower better)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.grid(alpha=0.2)
    ax.set_title("FIG 2 — the missing/spurious trade-off the count constraint enforces", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig2_missing_vs_spurious.png", dpi=140); plt.close(fig)

    # FIG 3 — exact-set fraction vs T0-only timing MAE
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for a in arms:
        ax.scatter([f(M[a], "B5_exact_set_mae_ms")], [f(M[a], "A5_exact_set_fraction")], s=140,
                   color=COL[a], zorder=3)
        ax.annotate(f"{a}\n(n T0 = {int(f(M[a], 'B5_n_T0_windows'))})",
                    (f(M[a], "B5_exact_set_mae_ms"), f(M[a], "A5_exact_set_fraction")), fontsize=7.5,
                    xytext=(7, -6), textcoords="offset points")
    ax.set_xlabel("B5 T0-only timing MAE (ms, lower better)")
    ax.set_ylabel("A5 exact-set fraction (higher better)")
    ax.grid(alpha=0.2)
    ax.set_title("FIG 3 — event-set correctness against placement on exactly-correct windows", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig3_exactset_vs_timing.png", dpi=140); plt.close(fig)

    # FIG 4 / 5 / 6 exist only if the corresponding stage ran
    preds = rd("validation_count_predictions.csv")
    if preds:
        fig, ax = plt.subplots(figsize=(5.6, 5.4))
        k = np.array([f(r, "K_gt") for r in preds]); kh = np.array([f(r, "K_hat") for r in preds])
        ax.scatter(k + np.random.default_rng(0).uniform(-.2, .2, k.size),
                   kh + np.random.default_rng(1).uniform(-.2, .2, k.size), s=6, alpha=0.25)
        lo, hi = min(k.min(), kh.min()) - 1, max(k.max(), kh.max()) + 1
        ax.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=1.0)
        ax.set_xlabel("GT beat count K"); ax.set_ylabel("predicted K_hat"); ax.grid(alpha=0.2)
        ax.set_title("FIG 4 — count prediction", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "fig4_count_prediction.png", dpi=140); plt.close(fig)
    gen = rd("generator_metrics.csv") or rd("oracle_count_generator_metrics.csv")
    if gen:
        G = {r["arm"]: r for r in gen}
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
        ga = [a for a in arms if a in G]
        for ax, ks, lab in ((axs[0], ("C2_own_T6", "C3_own_T7"), "own-centre nAE"),
                            (axs[1], ("J2_gt_local_deriv_rmse", "J3_gt_local_curvature_err"),
                             "GT-anchored joint structure")):
            w = 0.36
            for i, kk in enumerate(ks):
                ax.bar(np.arange(len(ga)) + (i - 0.5) * w, [f(G[a], kk) for a in ga], width=w, label=kk)
            ax.set_xticks(range(len(ga))); ax.set_xticklabels(ga, rotation=15, fontsize=7)
            ax.set_ylabel(lab, fontsize=8); ax.legend(fontsize=7); ax.grid(alpha=0.2, axis="y")
        fig.suptitle("FIG 5/6 — downstream own-centre morphology and GT-anchored joint structure", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "fig56_downstream.png", dpi=140); plt.close(fig)

    (OUT / "figures_manifest.json").write_text(json.dumps(
        {"figures": sorted(p.name for p in OUT.glob("*.png")), "arms_present": arms,
         "count_prediction_figure": bool(preds), "generator_figure": bool(gen),
         "note": "figures for stages that never ran are not produced; the report states why",
         "source": "frozen E3 artifacts only", "recomputed": False}, indent=2))
    print(f"[fig] {len(sorted(OUT.glob('*.png')))} figures | arms {arms}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
