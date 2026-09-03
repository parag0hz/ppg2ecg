"""O3 figures (preregistration section 26) — FIG 1-7, drawn only from already-frozen O3 artifacts.

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
ART = ROOT / "artifacts/o3_schedule_tolerance"
OUT = ART / "figures"
JIT = (1, 2, 4, 6, 8)
COL = {"JITTER": "#b2182b", "MISS": "#2166ac", "EXTRA": "#1a9850", "R1": "#762a83", "ORACLE": "#111111"}


def rd(name):
    return list(csv.DictReader(open(ART / name))) if (ART / name).exists() else []


def f(row, k):
    v = row.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sq = rd("schedule_quality_metrics.csv")
    gm = rd("synthetic_generator_metrics.csv")
    ad = rd("schedule_adherence.csv")
    rg = rd("retained_gain.csv")
    fl = rd("operator_floor_metrics.csv")
    r1q = [r for r in rd("r1_schedule_quality.csv") if r.get("site", "") == ""]
    r1g = rd("r1_generator_metrics.csv") if (ART / "r1_generator_metrics.csv").exists() else []
    Q = {(r["condition"], r["rep"]): r for r in sq}
    G = {(r["condition"], r["rep"]): r for r in gm}
    A = {(r["condition"], r["rep"]): r for r in ad}
    b_f1x = f([r for r in r1g if r["arm"] == "B"][0], "f1_excess") if r1g else np.nan
    r1_pt = None
    if r1q and r1g:
        arm = [r for r in r1g if r["arm"].startswith("O2C-R1")]
        if arm:
            r1_pt = {"mae": f(r1q[0], "timing_mae_ms"), "f1_50": f(r1q[0], "f1_at_50"),
                     "f1_150": f(r1q[0], "f1_at_150"), "bdev": f(r1q[0], "beats_ratio_dev"),
                     "f1x": f(arm[0], "f1_excess"), "T6": f(arm[0], "nAE_T6"), "T7": f(arm[0], "nAE_T7"),
                     "adh": f(arm[0], "adherence_f1_at_50")}

    def jitter_xy(xk, yk, ysrc=G):
        xs, ys = [], []
        for j in (0,) + JIT:
            c = "ORACLE" if j == 0 else f"JITTER_{j}"
            for rep in ("0", "1", "2"):
                if (c, rep) in Q and (c, rep) in ysrc:
                    xs.append(f(Q[(c, rep)], xk)); ys.append(f(ysrc[(c, rep)], yk))
        return np.asarray(xs), np.asarray(ys)

    # FIG 1 — timing MAE vs generator F1 excess
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x, y = jitter_xy("timing_mae_ms", "f1_excess")
    ax.scatter(x, y, s=26, color=COL["JITTER"], label="JITTER levels (3 replicates each)", zorder=3)
    for j in (0,) + JIT:
        c = "ORACLE" if j == 0 else f"JITTER_{j}"
        xs = [f(Q[(c, r)], "timing_mae_ms") for r in ("0", "1", "2") if (c, r) in Q]
        ys = [f(G[(c, r)], "f1_excess") for r in ("0", "1", "2") if (c, r) in G]
        if xs:
            ax.annotate(f"J{j}", (np.mean(xs), np.mean(ys)), fontsize=8, xytext=(4, 4), textcoords="offset points")
    xr = np.round(x, 6)
    vs = sorted(set(xr.tolist()))
    ax.plot(vs, [float(y[xr == v].mean()) for v in vs], lw=1.0, color=COL["JITTER"], alpha=0.5)
    if np.isfinite(b_f1x):
        ax.axhline(b_f1x, color="0.4", ls="--", lw=1.0, label=f"B baseline ({b_f1x:.4f})")
    if r1_pt:
        ax.scatter([r1_pt["mae"]], [r1_pt["f1x"]], s=90, marker="*", color=COL["R1"], zorder=4,
                   label="frozen R1 schedule")
    ax.set_xlabel("supplied-schedule matched timing MAE (ms)"); ax.set_ylabel("generator F1 excess")
    ax.set_title("FIG 1 — schedule timing error vs event fidelity (ORACLE DIAGNOSTIC)", fontsize=9)
    ax.grid(alpha=0.2); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig1_timing_vs_f1_excess.png", dpi=140); plt.close(fig)

    # FIG 2 — timing vs T6 / T7
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, k in zip(axs, ("nAE_T6", "nAE_T7")):
        x, y = jitter_xy("timing_mae_ms", k)
        a.scatter(x, y, s=26, color=COL["JITTER"], zorder=3, label="JITTER")
        if r1g:
            bb = f([r for r in r1g if r["arm"] == "B"][0], k)
            a.axhline(bb, color="0.4", ls="--", lw=1.0, label=f"B ({bb:.4f})")
        if r1_pt:
            a.scatter([r1_pt["mae"]], [r1_pt[k.split("_")[1]]], s=90, marker="*", color=COL["R1"],
                      zorder=4, label="frozen R1")
        a.set_xlabel("schedule timing MAE (ms)"); a.set_ylabel(f"{k} (lower is better)")
        a.grid(alpha=0.2); a.legend(fontsize=7)
    fig.suptitle("FIG 2 — schedule timing error vs O1-aligned morphology", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig2_timing_vs_morphology.png", dpi=140); plt.close(fig)

    # FIG 3 — retained oracle gain vs jitter level
    RG = {(r["condition"], r["rep"]): r for r in rg}
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    keys = [("event_gain_retention", "event F1 excess"), ("morph_gain_retention_nAE_T4", "T4"),
            ("morph_gain_retention_nAE_T6", "T6"), ("morph_gain_retention_nAE_T7", "T7"),
            ("morph_gain_retention_nAE_T8", "T8")]
    for k, lab in keys:
        xs, ys = [], []
        for j in (0,) + JIT:
            c = "ORACLE" if j == 0 else f"JITTER_{j}"
            v = [f(RG[(c, r)], k) for r in ("0", "1", "2") if (c, r) in RG]
            if v:
                xs.append(j); ys.append(float(np.mean(v)))
        ax.plot(xs, ys, marker="o", ms=4, lw=1.3, label=lab)
    ax.axhline(1.0, color="0.4", ls=":", lw=0.9); ax.axhline(0.0, color="0.4", ls="--", lw=0.9)
    ax.set_xlabel("jitter level J (samples; 1 sample = 7.8125 ms)")
    ax.set_ylabel("retained oracle gain (normalized retention ratio, unclipped)")
    ax.set_title("FIG 3 — retained oracle gain vs jitter level", fontsize=9)
    ax.grid(alpha=0.2); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig3_retained_gain.png", dpi=140); plt.close(fig)

    # FIG 4 — count-error sensitivity
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2))
    conds = ["ORACLE", "MISS_1", "MISS_2", "EXTRA_1", "EXTRA_2"]
    for a, k, lab in zip(axs, ("f1_excess", "nAE_T6", "nAE_T7"), ("F1 excess", "T6 nAE", "T7 nAE")):
        m = [np.mean([f(G[(c, r)], k) for r in ("0", "1", "2") if (c, r) in G]) for c in conds]
        e = [np.std([f(G[(c, r)], k) for r in ("0", "1", "2") if (c, r) in G]) for c in conds]
        a.bar(range(len(conds)), m, yerr=e, color=[COL["ORACLE"]] + [COL["MISS"]] * 2 + [COL["EXTRA"]] * 2,
              alpha=0.85, capsize=3)
        if r1g:
            a.axhline(f([r for r in r1g if r["arm"] == "B"][0], k), color="0.35", ls="--", lw=1.0)
        a.set_xticks(range(len(conds))); a.set_xticklabels(conds, rotation=20, fontsize=7)
        a.set_ylabel(lab); a.grid(alpha=0.2, axis="y")
    fig.suptitle("FIG 4 — beat-count error sensitivity (bars: mean over 3 replicates, whiskers SD; dashed line B)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig4_count_error.png", dpi=140); plt.close(fig)

    # FIG 5 — schedule quality vs adherence
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for fam in ("JITTER", "MISS", "EXTRA"):
        xs, ys = [], []
        for (c, r) in A:
            if A[(c, r)]["family"] != fam or (c, r) not in Q:
                continue
            xs.append(f(Q[(c, r)], "f1_at_50")); ys.append(f(A[(c, r)], "adherence_f1_at_50"))
        ax.scatter(xs, ys, s=26, color=COL[fam], label=fam, alpha=0.85, zorder=3)
    if r1_pt:
        ax.scatter([r1_pt["f1_50"]], [r1_pt["adh"]], s=110, marker="*", color=COL["R1"], zorder=4, label="frozen R1")
    ax.plot([0, 1], [0, 1], color="0.6", ls=":", lw=0.9)
    ax.set_xlabel("supplied schedule F1@50 vs GT R"); ax.set_ylabel("generated-event F1@50 vs supplied schedule")
    ax.set_title("FIG 5 — did the generator follow the geometry it was given?", fontsize=9)
    ax.grid(alpha=0.2); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig5_adherence.png", dpi=140); plt.close(fig)

    # FIG 6 — tolerance region with the R1 point overlaid
    gates = {(r["condition"], r["rep"]): r for r in rd("joint_benefit_gates.csv") if r.get("stage") == "synthetic"}
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for a, xk, xl in ((axs[0], "f1_at_50", "supplied schedule F1@50"),
                      (axs[1], "f1_at_150", "supplied schedule F1@150"),
                      (axs[2], "beats_ratio_dev", "supplied schedule beat-count deviation")):
        for (c, r), g in gates.items():
            if (c, r) not in Q or (c, r) not in G:
                continue
            surv = str(g["survives"]).lower() == "true"
            fam = g["family"]
            a.scatter([f(Q[(c, r)], xk)], [f(G[(c, r)], "f1_excess")], s=44,
                      marker={"JITTER": "o", "MISS": "s", "EXTRA": "^"}.get(fam, "o"),
                      facecolor=(COL[fam] if surv else "none"), edgecolor=COL[fam], linewidths=1.3, zorder=3)
        if np.isfinite(b_f1x):
            a.axhline(b_f1x, color="0.4", ls="--", lw=1.0)
        if r1_pt:
            a.scatter([r1_pt[{"f1_at_50": "f1_50", "f1_at_150": "f1_150", "beats_ratio_dev": "bdev"}[xk]]],
                      [r1_pt["f1x"]], s=130, marker="*", color=COL["R1"], zorder=5)
        a.set_xlabel(xl); a.set_ylabel("generator F1 excess"); a.grid(alpha=0.2)
    fig.suptitle("FIG 6 — tolerance region (filled = all G1-G6 pass) with the frozen R1 point (star)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig6_tolerance_region.png", dpi=140); plt.close(fig)

    # FIG 7 — site-wise R1 bridge
    if (ART / "r1_site_metrics.csv").exists():
        sm = rd("r1_site_metrics.csv")
        sites = [r["site"] for r in sm]
        fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0))
        axs[0].bar(sites, [f(r, "R1_schedule_f1_at_50") for r in sm], color=COL["R1"], alpha=0.85)
        axs[0].set_ylabel("R1 schedule F1@50")
        w = 0.27
        for i, (arm, col) in enumerate((("B", "0.5"), ("O2C-ORACLE", COL["ORACLE"]), ("O2C-R1-SCHEDULE", COL["R1"]))):
            axs[1].bar(np.arange(len(sites)) + (i - 1) * w, [f(r, f"{arm}_f1_excess") for r in sm], width=w,
                       color=col, label=arm)
            axs[2].bar(np.arange(len(sites)) + (i - 1) * w, [f(r, f"{arm}_nAE_T6") for r in sm], width=w,
                       color=col, label=arm)
        for a, lab in ((axs[1], "F1 excess"), (axs[2], "T6 nAE (lower better)")):
            a.set_xticks(range(len(sites))); a.set_xticklabels(sites); a.set_ylabel(lab)
            a.legend(fontsize=7); a.grid(alpha=0.2, axis="y")
        axs[0].grid(alpha=0.2, axis="y")
        fig.suptitle("FIG 7 — site-wise frozen R1 bridge (secondary; no site causality claim)", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "fig7_site_r1_bridge.png", dpi=140); plt.close(fig)

    (OUT / "figures_manifest.json").write_text(json.dumps(
        {"figures": sorted(p.name for p in OUT.glob("*.png")), "source": "frozen O3 artifacts only",
         "recomputed": False, "generator_run": False,
         "operator_floor_rows": len(fl)}, indent=2))
    print(f"[fig] {len(sorted(OUT.glob('*.png')))} figures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
