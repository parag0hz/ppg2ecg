"""E1 figures (preregistration section 21) — FIG 1-7, drawn only from already-frozen E1/O3 artifacts.

Nothing is recomputed here and no generator is run.
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
ART = ROOT / "artifacts/e1_event_morphology_decomposition"
O3ART = ROOT / "artifacts/o3_schedule_tolerance"
OUT = ART / "figures"
JIT = ("ORACLE", "JITTER_2", "JITTER_4", "JITTER_8")
TOPO_ARMS = ("ORACLE", "JITTER_8", "MISS1", "EXTRA1")
COL = {"own": "#1a9850", "gt": "#b2182b", "R1": "#762a83", "n": "#2166ac"}


def rd(name, art=ART):
    p = art / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(row, k):
    try:
        return float(row.get(k, ""))
    except (TypeError, ValueError):
        return np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sc = {r["arm"]: r for r in rd("synthetic_contrasts.csv")}
    cov = {r["arm"]: r for r in rd("coverage_metrics.csv")}
    boot = {r["contrast"]: r for r in rd("paired_bootstrap.csv")}
    strata = rd("r1_topology_strata.csv")
    tbins = rd("r1_timing_bins.csv")
    r1 = cov.get("R1-SCHEDULE")

    # FIG 1 / FIG 2 — three-axis decomposition
    for fig_id, (gtk, ownk, lab) in enumerate((("gt_local_deriv_rmse", "own_T6", "derivative / T6"),
                                               ("gt_local_curvature_err", "own_T7", "curvature / T7")), start=1):
        x = [f(sc[a], "schedule_mae_ms") for a in JIT if a in sc]
        fig, ax1 = plt.subplots(figsize=(7.4, 4.6))
        ax2 = ax1.twinx()
        ax1.plot(x, [f(sc[a], gtk) for a in JIT if a in sc], marker="o", color=COL["gt"], lw=1.6,
                 label=f"GT-anchored local {lab.split(' / ')[0]} error")
        ax2.plot(x, [f(sc[a], ownk) for a in JIT if a in sc], marker="s", color=COL["own"], lw=1.6,
                 label=f"own-centre {lab.split(' / ')[1]} nAE")
        for a, xv in zip([a for a in JIT if a in sc], x):
            ax1.annotate(a.replace("JITTER_", "J"), (xv, f(sc[a], gtk)), fontsize=8,
                         xytext=(4, 4), textcoords="offset points")
        ax1.set_xlabel("supplied-schedule matched timing MAE (ms)")
        ax1.set_ylabel(f"GT-anchored local {lab.split(' / ')[0]} error", color=COL["gt"])
        ax2.set_ylabel(f"own-centre {lab.split(' / ')[1]} nAE", color=COL["own"])
        ax1.tick_params(axis="y", colors=COL["gt"]); ax2.tick_params(axis="y", colors=COL["own"])
        ax1.grid(alpha=0.2)
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        ax1.set_title(f"FIG {fig_id} — placement-sensitive vs own-centre {lab} under timing jitter", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / f"fig{fig_id}_decomposition_{lab.split(' / ')[0]}.png", dpi=140)
        plt.close(fig)

    # FIG 3 — topology comparison with CIs
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, lab in zip(axs, ("T6", "T7")):
        arms, vals, los, his = [], [], [], []
        for a in TOPO_ARMS:
            if a not in sc:
                continue
            arms.append(a); vals.append(f(sc[a], f"own_{lab}"))
            tag = {"JITTER_8": "JITTER8", "MISS1": "MISS1", "EXTRA1": "EXTRA1"}.get(a)
            if tag and f"{tag}_damage_{lab}" in boot:
                b = boot[f"{tag}_damage_{lab}"]
                los.append(f(b, "point") - f(b, "lo")); his.append(f(b, "hi") - f(b, "point"))
            else:
                los.append(0.0); his.append(0.0)
        ax.bar(range(len(arms)), vals, yerr=[los, his], capsize=4,
               color=[COL["own"] if a == "ORACLE" else ("#b2182b" if a == "JITTER_8" else "#2166ac")
                      for a in arms], alpha=0.9)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=15, fontsize=8)
        ax.set_ylabel(f"own-centre {lab} nAE (lower better)"); ax.grid(alpha=0.2, axis="y")
    fig.suptitle("FIG 3 — own-centre morphology: severe timing jitter vs a single count error "
                 "(whiskers: 95 % CI of the damage vs ORACLE)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig3_topology_vs_timing.png", dpi=140); plt.close(fig)

    # FIG 4 — coverage
    arms = [a for a in ("ORACLE", "JITTER_2", "JITTER_4", "JITTER_8", "MISS1", "EXTRA1", "R1-SCHEDULE")
            if a in cov]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    w = 0.26
    for i, (k, lab) in enumerate((("C1_schedule_to_gt_identity", "S→G identity"),
                                  ("C2_generated_to_supplied_adherence", "P→S adherence"),
                                  ("C3_full_chain_P_S_G", "P→S→G full chain"))):
        ax.bar(np.arange(len(arms)) + (i - 1) * w, [f(cov[a], k) for a in arms], width=w, label=lab)
    ax.axhline(0.80, color="0.4", ls="--", lw=1.0, label="frozen adequacy floor 0.80")
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=15, fontsize=8)
    ax.set_ylabel("coverage"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.2, axis="y"); ax.legend(fontsize=7)
    ax.set_title("FIG 4 — chaining coverage (every morphology number carries these)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig4_coverage.png", dpi=140); plt.close(fig)

    # FIG 5 — R1 topology distribution
    if strata:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        names = [r["stratum"] for r in strata]
        n = [int(r["n_rows"]) for r in strata]
        ax.bar(range(len(names)), n, color=COL["R1"], alpha=0.9)
        for i, r in enumerate(strata):
            ax.annotate(f"{n[i]}\n({n[i] / max(sum(n), 1):.1%})", (i, n[i]), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([s.split("_", 1)[0] + "\n" + s.split("_", 1)[1].replace("_", " ").lower()
                            for s in names], fontsize=7)
        ax.set_ylabel("windows"); ax.grid(alpha=0.2, axis="y")
        ax.set_title("FIG 5 — frozen R1 schedule: window topology categories (2,048 rows)", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "fig5_r1_topology.png", dpi=140); plt.close(fig)

    # FIG 6 — R1 set-correct timing bins
    if tbins:
        fig, ax1 = plt.subplots(figsize=(8, 4.4))
        ax2 = ax1.twinx()
        xs = range(len(tbins))
        ax1.bar([x - 0.18 for x in xs], [f(r, "own_nAE_T6") for r in tbins], width=0.36, color=COL["own"],
                label="own-centre T6")
        ax1.bar([x + 0.18 for x in xs], [f(r, "own_nAE_T7") for r in tbins], width=0.36, color="#66bd63",
                label="own-centre T7")
        ax2.plot(list(xs), [f(r, "gt_local_deriv_rmse") for r in tbins], marker="o", color=COL["gt"],
                 label="GT-anchored local derivative RMSE")
        ax1.set_xticks(list(xs))
        ax1.set_xticklabels([f"{r['bin']}  {r['range_ms']} ms\nn={r['n_rows']} ({r['n_unique_ecg_windows']} win)"
                             for r in tbins], fontsize=7)
        ax1.set_ylabel("own-centre nAE"); ax2.set_ylabel("GT-anchored local derivative RMSE", color=COL["gt"])
        ax2.tick_params(axis="y", colors=COL["gt"]); ax1.grid(alpha=0.2, axis="y")
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=7)
        ax1.set_title("FIG 6 — R1 set-correct windows by schedule timing MAE", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "fig6_r1_timing_bins.png", dpi=140); plt.close(fig)

    # FIG 7 — site secondary: topology-error fraction vs own-centre morphology
    site = rd("r1_site_topology.csv")
    if site:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        x = [f(r, "topology_error_fraction") for r in site]
        for k, m, lab in (("own_nAE_T6", "o", "own-centre T6"), ("own_nAE_T7", "s", "own-centre T7")):
            ax.scatter(x, [f(r, k) for r in site], s=60, marker=m, label=lab)
        for r, xv in zip(site, x):
            ax.annotate(r["site"], (xv, f(r, "own_nAE_T6")), fontsize=8, xytext=(5, 4),
                        textcoords="offset points")
        ax.set_xlabel("fraction of windows whose R1 schedule topology is wrong")
        ax.set_ylabel("own-centre nAE"); ax.grid(alpha=0.2); ax.legend(fontsize=7)
        ax.set_title("FIG 7 — site secondary (no site causality claim)", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "fig7_site_topology.png", dpi=140); plt.close(fig)

    (OUT / "figures_manifest.json").write_text(json.dumps(
        {"figures": sorted(p.name for p in OUT.glob("*.png")), "source": "frozen E1/O3 artifacts only",
         "recomputed": False, "generator_run": False}, indent=2))
    print(f"[fig] {len(sorted(OUT.glob('*.png')))} figures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
