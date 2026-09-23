"""Figure for RDDM-EXT (reads artifacts/rddm_ext/result.json and the saved patient-level table; computes nothing new).

Run: .venv/bin/python scripts/rddm_ext_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, RAW = ROOT / "artifacts/rddm_ext", ROOT / "outputs/rddm_ext_raw"
KS = (1, 2, 3, 4, 8, 16)
COL = {"VITALDB": "#1f4e79", "WESAD": "#8a5a00", "CAPNO": "#2e7d32", "DALIA": "#a33", "BIDMC": "#6a3d9a"}


def main():
    r = json.loads((OUT / "result.json").read_text())
    v = r["corpora"]["VITALDB"]
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 3.8))
    for op, key, ls in (("median (nested)", "C_nested", "-"), ("mean (partition)", "mean_partition", "--"), ("20 % trimmed (partition)", "trimmed20_partition", ":")):
        m = [v["k_curve"][str(K)][key][0] for K in KS]
        lo = [v["k_curve"][str(K)][key][1] for K in KS]; hi = [v["k_curve"][str(K)][key][2] for K in KS]
        ax[0].plot(KS, m, ls, marker="o", ms=4, color="#1f4e79", label=op)
        ax[0].fill_between(KS, lo, hi, color="#1f4e79", alpha=0.10 if ls == "-" else 0.0)
    ax[0].plot(KS, [v["k_curve"][str(K)]["I_nested"][0] for K in KS], "-", marker="s", ms=3, color="#999", label="mean individual-sample error")
    a = v["anchor_ppg_peaks"]["error"][0]
    ax[0].axhline(a, color="#555", lw=0.8, ls="-."); ax[0].text(16, a, " PPG peaks", va="bottom", ha="right", fontsize=7, color="#555")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(KS); ax[0].set_xticklabels([str(K) for K in KS])
    ax[0].set_xlabel("K samples (20 NFE each; budget grows with K)"); ax[0].set_ylabel("HR error (bpm) vs V1 reference")
    ax[0].set_title("A. VitalDB V1 test (external to RDDM), 1,156 patients", fontsize=9); ax[0].legend(fontsize=7, frameon=False); ax[0].grid(lw=0.4, alpha=0.5)
    for n, c in COL.items():
        k = r["corpora"][n]["k_curve"]
        base = k["1"]["C_nested"][0]
        ax[1].plot(KS, [k[str(K)]["C_nested"][0] / base for K in KS], "-o", ms=3.5, color=c, label=n + (" (training-contaminated)" if n != "VITALDB" else " (external)"))
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(KS); ax[1].set_xticklabels([str(K) for K in KS])
    ax[1].set_xlabel("K"); ax[1].set_ylabel("median-pooled HR error / error at K = 1")
    ax[1].set_title("B. relative K curve, all corpora", fontsize=9); ax[1].legend(fontsize=6.5, frameon=False); ax[1].grid(lw=0.4, alpha=0.5)
    f = RAW / "vitaldb_patient_mechanism.npy"
    if f.exists():
        A = np.load(f)
        ax[2].scatter(A[:, 0], A[:, 1], s=5, alpha=0.35, color="#1f4e79", edgecolors="none")
        pm = v["mechanism_patient"]["err_corr"]
        ax[2].set_title(f"C. VitalDB patients (n = {v['mechanism_patient']['n_patients']}): Spearman {pm['point']:+.2f}\n"
                        f"[{pm['boot']['p2.5']:+.2f}, {pm['boot']['p97.5']:+.2f}] patient bootstrap", fontsize=9)
    ax[2].set_xlabel("patient mean functional-error correlation ρ̄_p"); ax[2].set_ylabel("patient consensus gain G_p (bpm, K = 16)")
    ax[2].grid(lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT / "rddm_ext.png", dpi=160)
    print("wrote", OUT / "rddm_ext.png")


if __name__ == "__main__":
    main()
