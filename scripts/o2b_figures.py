"""O2b diagnostic figures (preregistration section 10). Plotting only; no model, no training."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ppg2ecg.evaluation import event_reliability as ER  # noqa: E402
from ppg2ecg.evaluation import o2_warp as O2  # noqa: E402
from ppg2ecg.evaluation import o2b_warp as B  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2b_integer_grid"
O2ART = ROOT / "artifacts/o2_oracle_canonicalization"
FIG = ART / "figures"
VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024


def _pk(sig):
    return R.detect_rpeaks(np.asarray(sig, dtype=np.float64), 128)


def col(rows, k):
    return np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    o2 = list(csv.DictReader(open(O2ART / "warp_roundtrip_metrics.csv")))
    o2b = list(csv.DictReader(open(ART / "warp_roundtrip_metrics.csv")))
    labs = ["T4 p2p", "T6 max |d/dt|", "T7 curvature", "T8 width"]
    keys = ["nAE_T4", "nAE_T6", "nAE_T7", "nAE_T8"]

    # Figure 1 — O2 vs O2b, per-target, against the frozen threshold
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(4)
    a = [np.nanmedian(col(o2, k)) for k in keys]
    b = [np.nanmedian(col(o2b, k)) for k in keys]
    ax.bar(x - 0.2, a, 0.38, label="O2 (fractional $q_k$)", color="tab:red")
    ax.bar(x + 0.2, np.maximum(b, 1e-7), 0.38, label="O2b (integer $q_k$)", color="tab:green")
    ax.axhline(0.020, color="k", ls="--", lw=1.2, label="frozen Stage-0 threshold 0.020")
    ax.set_yscale("log"); ax.set_xticks(x, labs); ax.set_ylabel("median round-trip normalised AE (log)")
    ax.set_title("O2b — integer-grid canonicalization: round-trip morphology error vs the frozen gate", fontsize=10)
    for i, (v0, v1) in enumerate(zip(a, b)):
        ax.text(i - 0.2, v0 * 1.15, f"{v0:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, max(v1, 1e-7) * 1.15, ("0" if v1 == 0 else f"{v1:.1e}"), ha="center", fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "fig1_o2_vs_o2b_nae.png", dpi=130); plt.close(fig)

    # Figure 2 — distribution of q_real - q_int
    pb = list(csv.DictReader(open(ART / "per_beat_grid_analysis.csv")))
    dq = col(pb, "q_real") - col(pb, "q_int")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.hist(dq, bins=101, color="tab:blue", alpha=0.85)
    ax.axvline(-0.5, color="k", ls="--", lw=1); ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("$q_{real} - q_{int}$ (samples)"); ax.set_ylabel("beats")
    ax.set_title(f"O2b — integer projection of the canonical schedule (n = {len(dq)} beats, |Δ| ≤ 0.5 by construction)", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIG / "fig2_q_rounding_distribution.png", dpi=130); plt.close(fig)

    # Figure 3 — old fractional-grid distance vs old T6/T7 error, with the O2b errors on the same axes
    Y = []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        idx = ER.select_subset(SALT, s, len(d["y"]), TAKE)
        Y.append(d["y"][idx].astype(np.float32))
    Y = np.concatenate(Y)
    with ProcessPoolExecutor(max_workers=12) as ex:
        P = list(ex.map(_pk, list(Y.astype(np.float64)), chunksize=16))
    frac = np.array([float(np.median(np.abs(np.abs(O2.canonical_positions(P[i]) - np.asarray(P[i], float))
                                            - np.round(np.abs(O2.canonical_positions(P[i]) - np.asarray(P[i], float))))))
                     for i in range(len(Y))])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a_, k, nm in ((ax[0], "nAE_T6", "T6 max |dECG/dt|"), (ax[1], "nAE_T7", "T7 curvature energy")):
        a_.scatter(frac, col(o2, k), s=4, alpha=0.25, c="tab:red", label="O2 (fractional)")
        a_.scatter(frac, np.maximum(col(o2b, k), 1e-8), s=4, alpha=0.35, c="tab:green", label="O2b (integer)")
        a_.axhline(0.020, color="k", ls="--", lw=1, label="threshold 0.020")
        a_.set_yscale("log"); a_.set_xlabel("median |distance of $q_k-r_k$ to the sample grid| (O2)")
        a_.set_ylabel(f"round-trip nAE, {nm} (log)"); a_.legend(fontsize=7); a_.grid(alpha=0.2)
    fig.suptitle("O2b — removing the fractional offset removes the morphology damage across the whole cohort", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig3_frac_offset_vs_error.png", dpi=130); plt.close(fig)

    # Figure 4 — the exact window O2 named as its worst fractional-offset example
    t6_o2 = col(o2, "nAE_T6")
    k = int(np.nanargmax(np.where(np.isfinite(t6_o2), t6_o2, -1)))
    y = torch.from_numpy(Y[k][None, None])
    rt_o2 = O2.round_trip(y, [O2.EventWarp(P[k])]).squeeze().numpy()
    rt_o2b = O2.round_trip(y, [B.IntegerEventWarp(P[k])]).squeeze().numpy()
    r0 = int(P[k][len(P[k]) // 2]); lo, hi = max(0, r0 - 40), min(1024, r0 + 40)
    t = np.arange(lo, hi) / 128.0
    fig, ax = plt.subplots(2, 1, figsize=(9.2, 6.2))
    ax[0].plot(np.arange(1024) / 128.0, Y[k], lw=0.7, c="k", label="GT ECG")
    ax[0].plot(np.arange(1024) / 128.0, rt_o2, lw=0.7, c="tab:red", alpha=0.85, label="O2 round trip (fractional)")
    ax[0].plot(np.arange(1024) / 128.0, rt_o2b, lw=0.7, c="tab:green", alpha=0.85, label="O2b round trip (integer)")
    ax[0].set_title(f"the exact window O2 identified as its worst fractional-offset example (row {k})", fontsize=9)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2); ax[0].set_ylabel("normalised ECG")
    ax[1].plot(t, Y[k][lo:hi], lw=1.5, c="k", label="GT ECG")
    ax[1].plot(t, rt_o2[lo:hi], lw=1.5, c="tab:red", label="O2 round trip")
    ax[1].plot(t, rt_o2b[lo:hi], lw=1.5, c="tab:green", ls="--", label="O2b round trip (overlays GT)")
    ax[1].set_title("one QRS complex: the integer grid restores the peak, its slope and its curvature", fontsize=9)
    ax[1].set_xlabel("time (s)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIG / "fig4_worst_case_window.png", dpi=130); plt.close(fig)
    print(f"[figures] 4 written to {FIG} (worst-case row {k})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
