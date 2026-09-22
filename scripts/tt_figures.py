"""Figures for EXP-A (solver fairness) and EXP-B (error decomposition). Reads only the result files; computes nothing.

Run: .venv/bin/python scripts/tt_figures.py [expa|expb|all]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A, B = ROOT / "artifacts/tt_expa_solver", ROOT / "artifacts/tt_expb_mechanism"
COL = {"euler": "#1f4e79", "heun": "#a33"}
MK = {"42": "o", "1": "s", "2": "^", 42: "o", 1: "s", 2: "^"}


def read_csv(p):
    with open(p) as f:
        return [{k: (float(v) if _num(v) else v) for k, v in r.items()} for r in csv.DictReader(f)]


def _num(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def fig_expa():
    rows = read_csv(A / "grid.csv")
    res = json.loads((A / "result.json").read_text())
    budgets = sorted({int(r["B"]) for r in rows})
    fig, axes = plt.subplots(1, len(budgets) + 1, figsize=(4.2 * (len(budgets) + 1), 3.4))
    for ax, Bud in zip(axes, budgets):
        for solver in ("euler", "heun"):
            for seed in (42, 1, 2):
                pts = sorted([r for r in rows if int(r["B"]) == Bud and r["solver"] == solver and int(r["seed"]) == seed],
                             key=lambda r: r["nfe_per_sample"])
                if not pts:
                    continue
                ax.plot([p["nfe_per_sample"] for p in pts], [p["hr_err"] for p in pts], "-", color=COL[solver], alpha=0.35, lw=1)
                ax.scatter([p["nfe_per_sample"] for p in pts], [p["hr_err"] for p in pts], marker=MK[seed], s=26,
                           color=COL[solver], label=f"{solver} seed {seed}" if Bud == budgets[0] else None, zorder=3)
        ax.set_xscale("log", base=2); ax.set_xlabel("NFE per sample  (K = B / NFE)"); ax.set_title(f"B = {Bud} NFE")
        ax.grid(lw=0.4, alpha=0.5)
    axes[0].set_ylabel("HR error (bpm)")
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    q = res.get("single_sample_quality_seed42", [])
    ax = axes[-1]
    for solver in ("euler", "heun"):
        pts = sorted([r for r in q if r["solver"] == solver], key=lambda r: r["nfe"])
        ax.plot([p["nfe"] for p in pts], [p["fd"] for p in pts], "-o", ms=4, color=COL[solver], label=f"{solver} FD")
        ax.plot([p["nfe"] for p in pts], [p["hr_single"] for p in pts], "--s", ms=4, color=COL[solver], alpha=0.6, label=f"{solver} HR (1 draw)")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("NFE per sample"); ax.set_title("single sample, seed 42")
    ax.legend(fontsize=6.5, frameon=False); ax.grid(lw=0.4, alpha=0.5)
    fig.suptitle("EXP-A — PENGUIN: Euler vs upstream-exact Heun at matched NFE (VitalDB test)", fontsize=10)
    fig.tight_layout(); fig.savefig(A / "expa_solver.png", dpi=160)
    print("wrote", A / "expa_solver.png")


def fig_expb():
    b1, b2, b3 = (read_csv(B / n) for n in ("b1_fixed_K16_by_S.csv", "b2_fixed_S_by_K.csv", "b3_error_correlation.csv"))
    models = sorted({r["model"] for r in b1})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    for m, c in zip(models, ("#1f4e79", "#2e7d32", "#a33")):
        rs = sorted([r for r in b1 if r["model"] == m], key=lambda r: r["S"])
        S = [r["S"] for r in rs]
        axes[0].plot(S, [r["center_err"] for r in rs], "-o", ms=4, color=c, label=f"{m} — centre |m_S − T*|")
        axes[0].plot(S, [r["individual_err"] for r in rs], "--s", ms=4, color=c, alpha=0.55, label=f"{m} — individual")
        axes[1].plot(S, [r["median_gain"] for r in rs], "-o", ms=4, color=c, label=m)
        axes[1].plot(S, [r["disp_mad"] for r in rs], ":^", ms=4, color=c, alpha=0.6)
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("S (steps per sample), K = 16 fixed"); axes[0].set_ylabel("HR error (bpm)")
    axes[0].set_title("B1 — centre vs individual error"); axes[0].legend(fontsize=6, frameon=False); axes[0].grid(lw=0.4, alpha=0.5)
    axes[1].set_xscale("log", base=2); axes[1].set_xlabel("S, K = 16 fixed"); axes[1].set_ylabel("bpm")
    axes[1].set_title("B1 — median gain (solid) and dispersion MAD (dotted)"); axes[1].legend(fontsize=6.5, frameon=False); axes[1].grid(lw=0.4, alpha=0.5)
    for m, c in zip(models, ("#1f4e79", "#2e7d32", "#a33")):
        for seed, mk in (("42", "o"), ("1", "s"), ("2", "^")):
            rs = sorted([r for r in b2 if r["model"] == m and str(int(r["seed"])) == seed], key=lambda r: r["K"])
            if rs:
                axes[2].plot([r["K"] for r in rs], [r["R_partition"] for r in rs], "-", color=c, alpha=0.5, lw=1)
                axes[2].scatter([r["K"] for r in rs], [r["R_partition"] for r in rs], marker=mk, s=22, color=c,
                                label=f"{m} seed {seed}" if seed == "42" else None)
    axes[2].set_xscale("log", base=2); axes[2].set_xlabel("K (S fixed per model)"); axes[2].set_ylabel("HR error (bpm)")
    axes[2].set_title("B2 — width sweep at fixed depth"); axes[2].legend(fontsize=6.5, frameon=False); axes[2].grid(lw=0.4, alpha=0.5)
    fig.suptitle("EXP-B — functional error decomposition (VitalDB test)", fontsize=10)
    fig.tight_layout(); fig.savefig(B / "expb_decomposition.png", dpi=160)

    fig2, ax = plt.subplots(figsize=(4.4, 3.6))
    for m, c in zip(models, ("#1f4e79", "#2e7d32", "#a33")):
        rs = [r for r in b3 if r["model"] == m]
        ax.scatter([r["pearson_mean"] for r in rs], [r["median_gain"] for r in rs], s=34, color=c, label=m)
        for r in rs:
            ax.annotate(f"S={int(r['S'])}", (r["pearson_mean"], r["median_gain"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean pairwise correlation of sample functional errors"); ax.set_ylabel("median gain (bpm)")
    ax.set_title("B3 — error correlation vs pooling gain", fontsize=9); ax.legend(fontsize=6.5, frameon=False); ax.grid(lw=0.4, alpha=0.5)
    fig2.tight_layout(); fig2.savefig(B / "expb_error_correlation.png", dpi=160)
    print("wrote", B / "expb_decomposition.png", "and", B / "expb_error_correlation.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("expa", "all"):
        fig_expa()
    if what in ("expb", "all"):
        fig_expb()
