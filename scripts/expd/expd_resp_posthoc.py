"""EXP-D respiration — POST-HOC diagnostics (not preregistered; written after the analysis stage) + figure.

1. Does the generated RR carry information about the reference RR? Block-level Pearson / Spearman between each estimator
   and RR* (subject bootstrap for BIDMC), estimator mean / SD across blocks.
2. Monte-Carlo spread of a single draw's error (S = 1, 2, 4, 8; 16 draws) — context for the shipped Heun-50 value
   and U1's upstream-printed 3.479.
3. Each preregistered width / depth cell against the (preregistered, reported-only) training-median constant.
Writes artifacts/exp_d_functional_generalization/respiration/{posthoc.json, figure.png}.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/expd/expd_resp_posthoc.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import expd_run as X  # noqa: E402

ART = X.ART / "respiration"


def main():
    bj = json.loads((ART / "bootstrap.json").read_text())
    out = {"note": "POST-HOC, not preregistered", "datasets": {}}
    for ds in ("BIDMC", "WESAD"):
        z = np.load(X.RAW / f"{ds}.npz"); T = z["T"][:, 0]; sub = z["block_subject"]
        F = {S: z[f"F_S{S}"][:, :, 0] for S in X.SS}
        est = {"(1,32)": F[32][0], "(1,1)": F[1][0], "(32,1)": np.median(F[1][:32], 0), "(16,2)": np.median(F[2][:16], 0),
               "(8,4)": np.median(F[4][:8], 0), "(16,8)": np.median(F[8][:16], 0), "heun50": z["F_heun"][0, :, 0]}
        const = bj["datasets"][ds]["functionals"]["RR"]["constant_train_median"]["value"]
        bt = X.Boot(np.arange(len(sub)) if ds == "WESAD" else sub)
        rng = np.random.default_rng(X.BOOT_SEED)
        d = {"RR_ref": {"mean": float(T.mean()), "sd": float(T.std()), "min": float(T.min()), "max": float(T.max())},
             "train_median_constant": const, "estimators": {}, "single_draw_error_spread": {}, "vs_constant": {}}
        for k, v in est.items():
            if ds == "BIDMC":           # subject bootstrap of the block-level Spearman
                subs = np.unique(sub); bs = []
                for _ in range(X.NB):
                    pick = rng.choice(subs, len(subs)); m = np.concatenate([np.flatnonzero(sub == s) for s in pick])
                    bs.append(spearmanr(v[m], T[m])[0])
                ci = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
            else:
                bs = [spearmanr(v[ix], T[ix])[0] for ix in (rng.integers(0, len(T), len(T)) for _ in range(X.NB))]
                ci = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
            d["estimators"][k] = {"pearson_with_ref": float(pearsonr(v, T)[0]), "spearman_with_ref": float(spearmanr(v, T)[0]),
                                  "spearman_ci": ci, "mean": float(v.mean()), "sd_across_blocks": float(v.std())}
        for S in (1, 2, 4, 8):
            per = [float(np.abs(F[S][i] - T).mean()) for i in range(16)]
            d["single_draw_error_spread"][S] = {"mean": float(np.mean(per)), "min": min(per), "max": max(per)}
        ec = np.abs(const - T)
        for k in ("(32,1)", "(8,4)", "(16,1)", "(4,4)", "(1,32)", "heun50"):
            v = est.get(k, np.median(F[1][:16], 0) if k == "(16,1)" else np.median(F[4][:4], 0) if k == "(4,4)" else None)
            d["vs_constant"][k] = X.contrast(bt, np.abs(v - T), ec)["diff"]
        out["datasets"][ds] = d
    (ART / "posthoc.json").write_text(json.dumps(out, indent=1))
    # ---------------- figure
    r = bj["datasets"]["BIDMC"]["functionals"]["RR"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    for cells, col, lab in ((X.B32, "#1f4e79", "B = 32"), (X.B16, "#8a5a00", "B = 16")):
        S = [c[1] for c in cells]; cc = [r["cells"][f"K{K}_S{s}"] for K, s in cells]
        ax[0].errorbar(S, [c["point"] for c in cc], yerr=[[c["point"] - c["ci"][0] for c in cc], [c["ci"][1] - c["point"] for c in cc]],
                       marker="o", ms=4, capsize=2, color=col, label=f"{lab} (K = B / S)")
    c = r["constant_train_median"]["error"]["point"]; h = r["heun50_K1"]["point"]
    ax[0].axhline(c, color="#2e7d32", ls="--", lw=0.9); ax[0].text(32, c, " training-median constant", ha="right", va="bottom", fontsize=7, color="#2e7d32")
    ax[0].axhline(h, color="#555", ls="-.", lw=0.8); ax[0].text(32, h, " shipped Heun-50 (K = 1)", ha="right", va="bottom", fontsize=7, color="#555")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(X.SS); ax[0].set_xticklabels([str(s) for s in X.SS])
    ax[0].set_xlabel("Euler steps per sample S"); ax[0].set_ylabel("RR error (breaths/min)")
    ax[0].set_title("A. BIDMC (6 test subjects, 48 × 60-s blocks)", fontsize=9); ax[0].legend(fontsize=7, frameon=False); ax[0].grid(lw=0.4, alpha=0.5)
    tags = list(X.HEAD)
    for j, t in enumerate(tags):
        for off, dsn, col in ((-0.12, "BIDMC", "#1f4e79"), (0.12, "WESAD", "#a33")):
            q = bj["datasets"][dsn]["functionals"]["RR"]["contrasts"][t]["diff"]
            ax[1].errorbar(q["point"], j + off, xerr=[[q["point"] - q["ci"][0]], [q["ci"][1] - q["point"]]], fmt="o", ms=4, capsize=2, color=col,
                           label=(dsn + (" (subject bootstrap)" if dsn == "BIDMC" else " (1 subject, block bootstrap, exploratory)")) if j == 0 else None)
    ax[1].axvline(0, color="#555", lw=0.8); ax[1].set_yticks(range(len(tags))); ax[1].set_yticklabels([t.split("_", 1)[1] for t in tags], fontsize=7)
    ax[1].invert_yaxis(); ax[1].set_xlabel("width − depth, same NFE (breaths/min)"); ax[1].set_title("B. preregistered contrasts", fontsize=9)
    ax[1].legend(fontsize=6.5, frameon=False); ax[1].grid(lw=0.4, alpha=0.5)
    m = r["mechanism"]
    xs = [m["per_S"][str(S)]["rho_bar"] for S in X.MECH_S]; gs = [m["per_S"][str(S)]["G"]["point"] for S in X.MECH_S]
    ax[2].plot(xs, gs, "o-", color="#1f4e79")
    for S, a, g in zip(X.MECH_S, xs, gs):
        ax[2].annotate(f"S={S}", (a, g), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax[2].set_xlabel("functional-error correlation ρ̄_S"); ax[2].set_ylabel("consensus gain G_S (breaths/min, K = 16)")
    ax[2].set_title(f"C. mechanism (BIDMC): Spearman {m['across_S_spearman']['rho_bar_vs_G']['point']:+.1f}", fontsize=9); ax[2].grid(lw=0.4, alpha=0.5)
    z = np.load(X.RAW / "BIDMC.npz"); T = z["T"][:, 0]; F1 = z["F_S1"][:, :, 0]; F32 = z["F_S32"][0, :, 0]
    jit = np.random.default_rng(0).uniform(-0.2, 0.2, len(T))
    ax[3].scatter(T + jit, F32, s=12, color="#999", label="(1,32) single sample")
    ax[3].scatter(T + jit, np.median(F1[:32], 0), s=12, color="#1f4e79", label="(32,1) median of 32")
    ax[3].axhline(out["datasets"]["BIDMC"]["train_median_constant"], color="#2e7d32", ls="--", lw=0.9, label="training-median constant")
    lo, hi = 8, 30; ax[3].plot([lo, hi], [lo, hi], color="#555", lw=0.6); ax[3].set_xlim(12, 24); ax[3].set_ylim(lo, hi)
    ax[3].set_xlabel("reference RR (breaths/min)"); ax[3].set_ylabel("estimated RR")
    ax[3].set_title("D. post-hoc: estimates vs reference (BIDMC)", fontsize=9); ax[3].legend(fontsize=6.5, frameon=False); ax[3].grid(lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(ART / "figure.png", dpi=160)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
