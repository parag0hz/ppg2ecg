"""Figure for the PPGFlowECG external fixed-budget validation (reads artifacts/ppgflowecg_external/*.json; computes nothing new).

Run: .venv/bin/python scripts/external/ppgflowecg/pfe_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ppgflowecg_external"
SS = (5, 10, 15, 20, 25)


def main():
    fb = json.loads((ART / "fixed_budget_bootstrap.json").read_text())
    mk = json.loads((ART / "fixed_k_bootstrap.json").read_text())
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    # A: depth curve (K = 1) and K = 16 consensus, Hamilton
    for K, col, lab in ((1, "#1f4e79", "K = 1 (single sample)"), (16, "#8a5a00", "K = 16 median")):
        c = [fb["cells"][f"K{K}_S{S}"]["hamilton"] for S in SS]
        m = [x["point"] for x in c]
        ax[0].errorbar(SS, m, yerr=[[x["point"] - x["ci"][0] for x in c], [x["ci"][1] - x["point"] for x in c]], marker="o", ms=4,
                       color=col, capsize=2, label=lab)
    b = fb["baselines"]["ppg_findpeaks"]["point"]
    ax[0].axhline(b, color="#555", ls="-.", lw=0.8); ax[0].text(25, b, " PPG peaks", ha="right", va="bottom", fontsize=7, color="#555")
    ax[0].set_xticks(SS); ax[0].set_xlabel("Euler steps S (vector-field NFE per sample)"); ax[0].set_ylabel("Hamilton HR error (bpm)")
    ax[0].set_title("A. step count, VitalDB 10-s (18,525 win / 1,156 pts)", fontsize=9); ax[0].legend(fontsize=7, frameon=False)
    ax[0].grid(lw=0.4, alpha=0.5)
    # B: fixed-budget contrasts
    tags = [("B10", "B10: (2,5) − (1,10)"), ("B15", "B15: (3,5) − (1,15)"), ("B20", "B20: (4,5) − (1,20)"),
            ("B25", "B25*: (5,5) − (1,25)"), ("B20_intermediate_(2,10)-(1,20)", "B20: (2,10) − (1,20)")]
    for j, (t, lab) in enumerate(tags):
        for off, ev, col in ((-0.12, "hamilton", "#1f4e79"), (0.12, "project", "#a33")):
            r = fb["contrasts"][t][ev]["diff"]
            ax[1].errorbar(r["point"], j + off, xerr=[[r["point"] - r["ci"][0]], [r["ci"][1] - r["point"]]], fmt="o", ms=4, color=col,
                           capsize=2, label=("Hamilton (primary)" if ev == "hamilton" else "project HR (secondary)") if j == 0 else None)
    ax[1].axvline(0, color="#555", lw=0.8); ax[1].set_yticks(range(len(tags))); ax[1].set_yticklabels([x[1] for x in tags], fontsize=7)
    ax[1].invert_yaxis(); ax[1].set_xlabel("width − depth, same NFE (bpm; < 0 = width better)")
    ax[1].set_title("B. fixed-budget contrasts (95 % patient bootstrap)", fontsize=9); ax[1].legend(fontsize=7, frameon=False)
    ax[1].grid(lw=0.4, alpha=0.5)
    # C: mechanism, rho_bar vs G across S
    rows = mk["per_S"]
    x = [rows[str(S)]["rho_bar"] for S in SS]; g = [rows[str(S)]["G"] for S in SS]
    xe = [mk["rho_bar_boot_ci"][str(S)] for S in SS]
    ax[2].errorbar(x, [v["point"] for v in g], xerr=[[a - lo for a, (lo, hi) in zip(x, xe)], [hi - a for a, (lo, hi) in zip(x, xe)]],
                   yerr=[[v["point"] - v["ci"][0] for v in g], [v["ci"][1] - v["point"] for v in g]], fmt="o", ms=4, color="#1f4e79", capsize=2)
    for S, a, v in zip(SS, x, g):
        ax[2].annotate(f"S={S}", (a, v["point"]), textcoords="offset points", xytext=(4, 3), fontsize=7)
    sp = mk["across_S_spearman"]["rho_bar_vs_G"]
    ax[2].set_title(f"C. K = 16: Spearman(ρ̄, G) over S = {sp['point']:+.1f} ({100 * sp['share_negative']:.0f} % boot < 0)", fontsize=9)
    ax[2].set_xlabel("functional-error correlation ρ̄_S"); ax[2].set_ylabel("consensus gain G_S (bpm)"); ax[2].grid(lw=0.4, alpha=0.5)
    # D: waveform / event quality vs S, relative to S = 10
    wv = fb["waveform"]
    for key, col in (("rpeak_F1", "#1f4e79"), ("RMSE", "#a33"), ("waveform_pairwise_RMS", "#6a3d9a")):
        base = wv["10"][key]["point"]
        ax[3].plot(SS, [wv[str(S)][key]["point"] / base for S in SS], "-o", ms=3.5, color=col, label=key)
    fd = [wv[str(S)]["FD_official_draw0"] for S in SS]
    ax[3].plot(SS, np.array(fd) / fd[1], "--o", ms=3.5, color="#8a5a00", label="FD (official, draw 0)")
    ax[3].text(0.02, 0.03, "absolute, every S: per-sample Pearson r with reference "
               f"{min(wv[str(S)]['pearson_r']['point'] for S in SS):.3f}–{max(wv[str(S)]['pearson_r']['point'] for S in SS):.3f}, "
               f"R-peak F1 {min(wv[str(S)]['rpeak_F1']['point'] for S in SS):.2f}–{max(wv[str(S)]['rpeak_F1']['point'] for S in SS):.2f}",
               transform=ax[3].transAxes, fontsize=6.5, color="#333")
    ax[3].axhline(1, color="#555", lw=0.6); ax[3].set_xticks(SS); ax[3].set_xlabel("S"); ax[3].set_ylabel("value / value at S = 10")
    ax[3].set_title("D. per-sample waveform / event metrics", fontsize=9); ax[3].legend(fontsize=6.5, frameon=False); ax[3].grid(lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(ART / "ppgflowecg_external.png", dpi=160)
    print("wrote", ART / "ppgflowecg_external.png")


if __name__ == "__main__":
    main()
