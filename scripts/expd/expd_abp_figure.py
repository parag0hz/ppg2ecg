"""EXP-D Part B figure (reads abp/bootstrap.json and abp/usability_gates.json; computes nothing new).

Run: .venv/bin/python scripts/expd/expd_abp_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import expd_run as X  # noqa: E402

ART = X.ART / "abp"


def main():
    bj = json.loads((ART / "bootstrap.json").read_text())
    gt = json.loads((ART / "usability_gates.json").read_text())
    d = bj["datasets"]["MIMIC-BP"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.6))
    for i, fn in enumerate(X.FUNCS["abp"]):
        a = ax[0, i]; r = d["functionals"][fn]
        for cells, col, lab in ((X.B32, "#1f4e79", "B = 32"), (X.B16, "#8a5a00", "B = 16")):
            S = [c[1] for c in cells]; cc = [r["cells"][f"K{K}_S{s}"] for K, s in cells]
            a.errorbar(S, [c["point"] for c in cc], yerr=[[c["point"] - c["ci"][0] for c in cc], [c["ci"][1] - c["point"] for c in cc]],
                       marker="o", ms=4, capsize=2, color=col, label=f"{lab} (K = B / S)")
        c = r["constant_train_median"]["error"]["point"]; h = r["heun50_K1"]["point"]
        a.axhline(c, color="#2e7d32", ls="--", lw=0.9, label="training-median constant")
        a.axhline(h, color="#555", ls="-.", lw=0.8, label="shipped Heun-50 (K = 1)")
        a.set_xscale("log", base=2); a.set_xticks(X.SS); a.set_xticklabels([str(s) for s in X.SS])
        g = gt["functionals"][fn]
        a.set_title(f"{fn} (mmHg)\nusability gate: {g['usability']} · allocation: {g['allocation_verdict']}", fontsize=8.5)
        a.set_xlabel("Euler steps per sample S"); a.set_ylabel(f"{fn} MAE (mmHg)"); a.grid(lw=0.4, alpha=0.5)
        if i == 0:
            a.legend(fontsize=6.5, frameon=False)
    a = ax[1, 0]
    rows = [(fn, t) for fn in X.FUNCS["abp"] for t in list(X.HEAD)[:2]]
    for j, (fn, t) in enumerate(rows):
        q = d["functionals"][fn]["contrasts"][t]["diff"]
        a.errorbar(q["point"], j, xerr=[[q["point"] - q["ci"][0]], [q["ci"][1] - q["point"]]], fmt="o", ms=4, capsize=2, color="#1f4e79")
    a.axvline(0, color="#555", lw=0.8); a.set_yticks(range(len(rows))); a.set_yticklabels([f"{fn} {t.split('_', 2)[2]}" for fn, t in rows], fontsize=7)
    a.invert_yaxis(); a.set_xlabel("width − depth at B = 32 (mmHg; < 0 = width better)"); a.set_title("B32 preregistered contrasts", fontsize=9)
    a.grid(lw=0.4, alpha=0.5)
    a = ax[1, 1]
    for fn, col in zip(X.FUNCS["abp"], ("#1f4e79", "#a33", "#2e7d32")):
        m = d["functionals"][fn]["mechanism"]
        xs = [m["per_S"][str(S)]["rho_bar"] for S in X.MECH_S]; gs = [m["per_S"][str(S)]["G"]["point"] for S in X.MECH_S]
        a.plot(xs, gs, "o-", color=col, label=f"{fn}: Spearman {m['across_S_spearman']['rho_bar_vs_G']['point']:+.1f}")
        for S, x_, g_ in zip(X.MECH_S, xs, gs):
            a.annotate(f"{S}", (x_, g_), textcoords="offset points", xytext=(3, 3), fontsize=6.5, color=col)
    a.set_xlabel("functional-error correlation ρ̄_S (K = 16)"); a.set_ylabel("consensus gain G_S (mmHg)")
    a.set_title("mechanism, S = 1, 2, 4, 8 (labels)", fontsize=9); a.legend(fontsize=7, frameon=False); a.grid(lw=0.4, alpha=0.5)
    a = ax[1, 2]; w = d["waveform"]
    keys = [f"S{S}" for S in X.SS]
    a.plot(X.SS, [w[k]["FD"] for k in keys], "o-", color="#8a5a00", label="FD (upstream, draw 0)")
    a.set_xscale("log", base=2); a.set_xticks(X.SS); a.set_xticklabels([str(s) for s in X.SS]); a.set_xlabel("S"); a.set_ylabel("FD")
    a.set_title("waveform: FD (left) — r / RMSE in the report", fontsize=9); a.legend(fontsize=7, frameon=False); a.grid(lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(ART / "figure.png", dpi=150)
    print("wrote", ART / "figure.png")


if __name__ == "__main__":
    main()
