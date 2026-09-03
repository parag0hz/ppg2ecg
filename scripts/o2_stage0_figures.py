"""O2 Stage-0 figures: why the canonicalization operator was rejected. Plotting only."""
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
from ppg2ecg.evaluation import o2_warp as W  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2_oracle_canonicalization"
FIG = ART / "figures"
VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024


def _pk(sig):
    return R.detect_rpeaks(np.asarray(sig, dtype=np.float64), 128)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    Y = []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        idx = ER.select_subset(SALT, s, len(d["y"]), TAKE)
        Y.append(d["y"][idx].astype(np.float32))
    Y = np.concatenate(Y)
    with ProcessPoolExecutor(max_workers=12) as ex:
        P = list(ex.map(_pk, list(Y.astype(np.float64)), chunksize=16))
    rows = list(csv.DictReader(open(ART / "warp_roundtrip_metrics.csv")))
    t6 = np.array([float(r["nAE_T6"]) if r["nAE_T6"] not in ("", "nan") else np.nan for r in rows])
    t7 = np.array([float(r["nAE_T7"]) if r["nAE_T7"] not in ("", "nan") else np.nan for r in rows])
    frac = np.array([float(np.median(np.abs(np.abs(W.canonical_positions(P[i]) - np.asarray(P[i], float))
                                            - np.round(np.abs(W.canonical_positions(P[i]) - np.asarray(P[i], float))))))
                     for i in range(len(Y))])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, v, nm in ((ax[0], t6, "T6 max |dECG/dt|"), (ax[1], v7 := t7, "T7 curvature energy")):
        a.scatter(frac, v, s=4, alpha=0.25, c="tab:blue")
        edges = np.percentile(frac, np.arange(0, 101, 10))
        cen, med = [], []
        for i in range(10):
            m = (frac >= edges[i]) & (frac <= edges[i + 1])
            if m.sum() > 5:
                cen.append(np.median(frac[m])); med.append(np.nanmedian(v[m]))
        a.plot(cen, med, "o-", c="tab:red", lw=2, label="decile median")
        a.axhline(0.020, color="k", ls="--", lw=1, label="Stage-0 threshold 0.020")
        a.set_xlabel("median |distance of the canonical offset $q_k-r_k$ to the sample grid|")
        a.set_ylabel(f"round-trip normalised AE, {nm}")
        a.legend(fontsize=8); a.grid(alpha=0.2)
    fig.suptitle("O2 Stage 0 — round-trip morphology error is driven by fractional-offset interpolation, not by the schedule change", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig1_roundtrip_vs_grid_offset.png", dpi=130); plt.close(fig)

    k = int(np.nanargmax(np.where(np.isfinite(t6), t6, -1)))
    y = torch.from_numpy(Y[k][None, None])
    w = [W.EventWarp(P[k])]
    can = W.apply_warp(y, w, "to_canonical").squeeze().numpy()
    rt = W.round_trip(y, w).squeeze().numpy()
    r0 = int(P[k][len(P[k]) // 2]); lo, hi = max(0, r0 - 40), min(1024, r0 + 40)
    t = np.arange(lo, hi) / 128.0
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    ax[0].plot(np.arange(1024) / 128.0, Y[k], lw=0.7, label="GT ECG (raw coordinate)")
    ax[0].plot(np.arange(1024) / 128.0, can, lw=0.7, alpha=0.8, label="canonical coordinate")
    ax[0].plot(np.arange(1024) / 128.0, rt, lw=0.7, alpha=0.8, label="round trip $W^{-1}(W(x))$")
    for rk in P[k]:
        ax[0].axvline(rk / 128.0, color="tab:red", lw=0.4, alpha=0.4)
    ax[0].legend(fontsize=8); ax[0].set_ylabel("normalised ECG"); ax[0].grid(alpha=0.2)
    ax[0].set_title(f"worst-case window (row {k}): whole window", fontsize=9)
    ax[1].plot(t, Y[k][lo:hi], lw=1.4, label="GT ECG")
    ax[1].plot(t, rt[lo:hi], lw=1.4, label="round trip")
    ax[1].set_title("same window, one QRS complex: bilinear resampling at a fractional offset blunts the peak, its slope and its curvature", fontsize=9)
    ax[1].set_xlabel("time (s)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIG / "fig2_qrs_blunting_example.png", dpi=130); plt.close(fig)
    print(f"[figures] 2 written to {FIG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
