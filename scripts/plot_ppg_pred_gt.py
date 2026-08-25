"""Visualise input PPG, ground-truth ECG and A0 predictions (50 / 4 / 1 NFE) on one time axis for deterministic test windows.
Writes figures/ppg_pred_gt_overview.png (3 windows side by side) and figures/ppg_pred_gt_<idx>.png (one window, all rows)."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import neurokit2 as nk
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.data.dalia import load_subject_raw  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS = 128
ACT = {0: "transient", 1: "sitting", 2: "stairs", 3: "soccer", 4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working"}
ARMS = [("heun25", "Heun 25 steps · 50 NFE"), ("heun2", "Heun 2 steps · 4 NFE"), ("euler1", "Euler 1 step · 1 NFE")]


def ppg_peaks(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return np.asarray(nk.ppg_findpeaks(nk.ppg_clean(x.astype(float), sampling_rate=FS), sampling_rate=FS)["PPG_Peaks"], dtype=int)
        except Exception:  # noqa: BLE001
            return np.zeros(0, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/a0_penguin_otcfm_ppgdalia_8s_seed42")
    ap.add_argument("--subject", default="S2")
    ap.add_argument("--windows", default=None, help="comma-separated window indices; default = 10/50/90 % HR-error quantile windows of the 50-NFE arm")
    args = ap.parse_args()
    out = ROOT / args.out_dir
    d = np.load(ROOT / f"data/processed/v0_8s/{args.subject}.npz")
    x, y, starts = d["x"], d["y"], d["window_start_s"]
    preds = {k: np.load(out / "predictions" / f"{k}.npz") for k, _ in ARMS}
    met = json.loads((out / "metrics.json").read_text())
    idxs = [int(i) for i in args.windows.split(",")] if args.windows else list(met["examples"]["ref_arm_hr_err_quantiles_10_50_90"])
    raw = load_subject_raw(ROOT / "data/raw", args.subject)
    act_fs = len(raw.activity) / raw.ecg_seconds
    t = np.arange(x.shape[1]) / FS

    def activity(i):
        seg = raw.activity[int(round(starts[i] * act_fs)) : int(round((starts[i] + 8) * act_fs))].astype(int)
        return ACT.get(int(np.bincount(seg).argmax()), "?") if len(seg) else "?"

    def draw_rows(axes, i, compact=False):
        pp = ppg_peaks(x[i])
        axes[0].plot(t, x[i], color="tab:green", lw=0.9)
        axes[0].plot(pp / FS, x[i][pp], "v", color="darkgreen", ms=5)
        axes[0].set_ylabel("PPG (input)\nwrist, 128 Hz")
        rg = R.detect_rpeaks(y[i], FS)
        axes[1].plot(t, y[i], "k", lw=0.9)
        axes[1].plot(rg / FS, y[i][rg], "r.", ms=6)
        axes[1].set_ylabel(f"GT ECG\nHR {R.hr_bpm(rg, FS):.0f} bpm")
        for ax, (k, lab) in zip(axes[2:], ARMS):
            p = preds[k]["pred"][i]
            rp = R.detect_rpeaks(p, FS)
            ax.plot(t, y[i], color="0.8", lw=0.6)
            ax.plot(t, p, color="tab:blue", lw=0.9)
            ax.plot(rp / FS, p[rp], "r.", ms=6)
            hr = R.hr_bpm(rp, FS)
            ax.set_ylabel(f"pred {lab.split(' · ')[1]}\nHR {hr:.0f} bpm" if np.isfinite(hr) else f"pred {lab.split(' · ')[1]}\nno beats")
        for ax in axes:
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("time (s)")

    # overview: 3 windows side by side, rows PPG / GT / 50 NFE / 4 NFE / 1 NFE
    nrow = 2 + len(ARMS)
    fig, axes = plt.subplots(nrow, len(idxs), figsize=(6.2 * len(idxs), 2.0 * nrow), sharex=True)
    axes = np.atleast_2d(axes).reshape(nrow, len(idxs))
    for c, i in enumerate(idxs):
        draw_rows(axes[:, c], i)
        axes[0, c].set_title(f"{args.subject} window {i} · t = {starts[i]} s · {activity(i)}", fontsize=10)
        if c > 0:
            for r in range(nrow):
                axes[r, c].set_ylabel("")
    fig.suptitle("PPG → ECG (PENGUIN A0, seed 42, 8 s windows): input PPG, ground-truth ECG, predictions at 50 / 4 / 1 NFE (grey = GT)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ppg_pred_gt_overview.png", dpi=110)
    plt.close(fig)
    # per-window figures
    for i in idxs:
        fig, axes = plt.subplots(nrow, 1, figsize=(14, 1.9 * nrow), sharex=True)
        draw_rows(axes, i)
        axes[0].set_title(f"{args.subject} window {i} · start {starts[i]} s · activity {activity(i)} — PPG input, GT ECG, A0 predictions (R-peaks: red; PPG pulse peaks: green)")
        fig.tight_layout()
        fig.savefig(out / "figures" / f"ppg_pred_gt_{i:05d}.png", dpi=110)
        plt.close(fig)
    print("wrote", out / "figures" / "ppg_pred_gt_overview.png", "+", len(idxs), "per-window figures", idxs)


if __name__ == "__main__":
    main()
