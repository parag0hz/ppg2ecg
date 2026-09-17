"""Qualitative comparison: every model (PENGUIN = OT-CFM at NFE 50 / 1, iMF at NFE 1 / 2 / 4) on the SAME test
windows of every dataset. No metric, no tuning. Window choice is fixed before looking at any output: test windows
are concatenated in manifest order and the 3 samples are the linspace points 1..3 of 5 over the eligible block
starts (never the first or last). ECG / ABP: one 4 s window. Respiration: 4 consecutive 4 s windows (16 s) of one
subject. Noise seed 0 for every model. Checkpoints: U2 / V1 checkpoint_last.pt (both arms at 14,000 steps)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import u2_evaluate as U2  # noqa: E402

OUT = ROOT / "artifacts/qual_model_dataset"
DATASETS = [("u2_dalia", "PPG-DaLiA", "ECG"), ("u2_wildppg", "WildPPG", "ECG"), ("v1_vitaldb", "VitalDB", "ECG"),
            ("u2_bidmc", "BIDMC", "Resp"), ("u2_wesad", "WESAD", "Resp"),
            ("u2_ucibp", "UCI-BP", "ABP"), ("u2_mimicbp", "MIMIC-BP", "ABP")]
MODELS = [("C", 50, "PENGUIN · 50 NFE", "#1f5fa8"), ("C", 1, "PENGUIN · 1 NFE", "#7aa6d6"),
          ("I", 1, "iMF · 1 NFE", "#c0392b"), ("I", 2, "iMF · 2 NFE", "#d9663f"), ("I", 4, "iMF · 4 NFE", "#e39b3a")]
UNIT = {"ECG": "a.u.", "Resp": "a.u.", "ABP": "mmHg"}
FS = 128


def load(slug):
    split = json.loads((ROOT / f"data/manifests/split_{slug}_seed42.json").read_text())["splits"][0]
    if slug == "u2_wildppg":
        real = json.loads((ROOT / f"data/manifests/split_{slug}_seed42.json").read_text())["extra"]["wildppg_subject_map"]
        assert not any(b in real[s] for s in split["test"] for b in ("kjd", "ssx"))
    X, Y, S = [], [], []
    for s in split["test"]:
        d = np.load(ROOT / f"data/processed/{slug}/{s}.npz")
        X.append(d["x"]); Y.append(d["y"].astype(np.float64)); S.append(np.full(len(d["x"]), s))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S)


def pick(S, k):
    starts = np.array([i for i in range(len(S) - k + 1) if S[i] == S[i + k - 1]])
    return [int(starts[j]) for j in np.linspace(0, len(starts) - 1, 5).round().astype(int)[1:4]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    overview = []
    for slug, name, task in DATASETS:
        X, Y, S = load(slug)
        k = 4 if task == "Resp" else 1
        starts = pick(S, k)
        idx = np.concatenate([np.arange(s, s + k) for s in starts])
        e0 = torch.randn(len(idx), 1, X.shape[1], generator=torch.Generator().manual_seed(0))
        nets = {a: U2.build(ROOT / f"outputs/{slug}_arm{a}_seed42/checkpoint_last.pt", dev)[0] for a in ("C", "I")}
        preds = {(a, n): U2.generate(nets[a], X[idx], e0, n, dev, a)[0] for a, n, *_ in MODELS}
        del nets; torch.cuda.empty_cache()
        cat = lambda A, j: A[j * k:(j + 1) * k].reshape(-1)  # noqa: E731
        t = np.arange(k * X.shape[1]) / FS
        rows = [("PPG (input)", None)] + [("Ground truth", None)] + [(m[2], m) for m in MODELS]
        fig, ax = plt.subplots(len(rows), 3, figsize=(13, 1.35 * len(rows)), sharex=True, squeeze=False)
        for j, s in enumerate(starts):
            gt = cat(Y, j)
            lo = min([gt.min()] + [cat(preds[m[:2]], j).min() for m in MODELS])
            hi = max([gt.max()] + [cat(preds[m[:2]], j).max() for m in MODELS])
            pad = 0.05 * (hi - lo)
            for r, (lab, m) in enumerate(rows):
                a = ax[r, j]
                if r == 0:
                    a.plot(t, cat(X[idx], j), color="#2e7d32", lw=0.9)
                elif r == 1:
                    a.plot(t, gt, color="black", lw=0.9); a.set_ylim(lo - pad, hi + pad)
                else:
                    a.plot(t, gt, color="#bbbbbb", lw=0.9)
                    a.plot(t, cat(preds[m[:2]], j), color=m[3], lw=0.9); a.set_ylim(lo - pad, hi + pad)
                a.tick_params(labelsize=7)
                if j == 0:
                    a.set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center")
                if r == 0:
                    a.set_title(f"test window {s} ({S[s]})", fontsize=8)
            ax[-1, j].set_xlabel("time (s)", fontsize=8)
        fig.suptitle(f"{name} ({task}, target unit {UNIT[task]}) — same test windows, noise seed 0, grey = ground truth", fontsize=10)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"qual_{slug}.{ext}", dpi=140)
        plt.close(fig)
        overview.append((name, task, t, cat(X[idx], 0), cat(Y, 0), {m[:2]: cat(preds[m[:2]], 0) for m in MODELS}))
        print(f"[qual] {name}: windows {starts}", flush=True)

    cols = [("PPG (input)", None)] + [(m[2], m) for m in MODELS]
    fig, ax = plt.subplots(len(overview), len(cols), figsize=(2.9 * len(cols), 1.45 * len(overview)), squeeze=False)
    for r, (name, task, t, x, gt, P) in enumerate(overview):
        lo = min([gt.min()] + [p.min() for p in P.values()]); hi = max([gt.max()] + [p.max() for p in P.values()])
        pad = 0.05 * (hi - lo)
        for c, (lab, m) in enumerate(cols):
            a = ax[r, c]
            if m is None:
                a.plot(t, x, color="#2e7d32", lw=0.8)
                a.set_ylabel(f"{name}\n{task}", fontsize=8, rotation=0, ha="right", va="center")
            else:
                a.plot(t, gt, color="#bbbbbb", lw=0.8); a.plot(t, P[m[:2]], color=m[3], lw=0.8)
                a.set_ylim(lo - pad, hi + pad)
            a.set_xticks([]); a.tick_params(labelsize=6)
            if r == 0:
                a.set_title(lab, fontsize=9)
    fig.suptitle("Every model on every dataset — first selected test window; grey = ground truth", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"qual_overview.{ext}", dpi=140)
    print("[qual] done", flush=True)


if __name__ == "__main__":
    main()
