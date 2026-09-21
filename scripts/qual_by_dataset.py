"""Qualitative figures per dataset for the current arms (no metric, no tuning).

Windows are fixed by rule before any output is seen: test windows in manifest order, the linspace points 1..3 of 5.
Rows: PPG / target (GT R marked) / PENGUIN 50 NFE / PENGUIN 1 NFE / iMF 1 NFE / consistency distillation 1 NFE /
the 16 iMF samples overlaid with their vote positions / iMF consensus-decoded (ED1 frozen parameters).
Noise seed 0 for single samples, seeds 0-15 for the K = 16 set. Seed-42 checkpoints (WildPPG: WP1 fold 0)."""
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
import ed1_consensus_decode as E1  # noqa: E402
import ed2_run as E2  # noqa: E402

FS, K = 128, 16
OUT = ROOT / "artifacts/qual_by_dataset"
DATASETS = [
    ("VitalDB", "data/manifests/split_v1_vitaldb_seed42.json", "data/processed/v1_vitaldb",
     {"C": "outputs/v1_vitaldb_armC_seed42", "I": "outputs/v1_vitaldb_armI_seed42", "D": "outputs/cd1_vitaldb_armD_seed42"}),
    ("WildPPG", "data/manifests/split_wp1_fold0.json", "data/processed/u2_wildppg",
     {a: f"outputs/wp1_f0_arm{a}" for a in "CID"}),
    ("BIDMC", "data/manifests/split_mc1_bidmc.json", "data/processed/mc1_bidmc_4s", {a: f"outputs/mc1_bidmc_arm{a}" for a in "CID"}),
    ("CapnoBase", "data/manifests/split_mc1_capnobase.json", "data/processed/mc1_capnobase_4s", {a: f"outputs/mc1_capnobase_arm{a}" for a in "CID"}),
    ("PPG-DaLiA", "data/manifests/split_mc1_dalia.json", "data/processed/u2_dalia", {a: f"outputs/mc1_dalia_arm{a}" for a in "CID"}),
]
COL = {"P50": "#1f5fa8", "P1": "#7aa6d6", "I": "#c0392b", "D": "#1b8f66", "DEC": "#7a3e9d"}


def load(manifest, processed):
    split = json.loads((ROOT / manifest).read_text())["splits"][0]
    X, Y, S = [], [], []
    for s in split["test"]:
        d = np.load(ROOT / processed / f"{s}.npz")
        X.append(d["x"]); Y.append(d["y"]); S += [s] * len(d["x"])
    X, Y = np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float64)
    idx = np.linspace(0, len(X) - 1, 5).round().astype(int)[1:4]
    return X[idx], Y[idx], [S[i] for i in idx], idx


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    par = json.loads((ROOT / "artifacts/ed1_consensus_decoding/params.json").read_text())["I"]["chosen"]
    w = int(round(par["w_ms"] / 1000 * FS))
    overview = []
    for name, man, proc, ck in DATASETS:
        X, Y, subj, idx = load(man, proc)
        t = np.arange(X.shape[1]) / FS
        gC = E2.make_gen(ROOT / ck["C"] / "checkpoint_last.pt", "C", dev)
        gI = E2.make_gen(ROOT / ck["I"] / "checkpoint_last.pt", "I", dev)
        gD = E2.make_gen(ROOT / ck["D"] / "checkpoint_last.pt", "D", dev)
        p50, p1, d1 = gC(X, 0, 50), gC(X, 0, 1), gD(X, 0)
        S = np.stack([gI(X, s) for s in range(K)])                                  # [K, 3, T]
        rows = ["PPG (input)", "Target ECG", "PENGUIN · 50 NFE", "PENGUIN · 1 NFE", "iMF · 1 NFE",
                "Consistency dist. · 1 NFE", "16 iMF samples", "iMF consensus-decoded"]
        fig, ax = plt.subplots(len(rows), 3, figsize=(13.5, 1.32 * len(rows)), sharex=True, squeeze=False)
        for j in range(3):
            gt = E1._peaks(Y[j])
            P = [E1._peaks(S[k, j].astype(np.float64)) for k in range(K)]
            wave, pos, vf, _ = E1.decode_window((S[:, j, :].astype(np.float64), P, w, par["theta"], par["b"]))
            lo = min(Y[j].min(), p50[j].min(), S[:, j].min(), d1[j].min()); hi = max(Y[j].max(), p50[j].max(), S[:, j].max(), d1[j].max())
            pad = 0.06 * (hi - lo)
            series = [None, None, (p50[j], COL["P50"]), (p1[j], COL["P1"]), (S[0, j], COL["I"]), (d1[j], COL["D"]), None, (wave, COL["DEC"])]
            for r in range(len(rows)):
                a = ax[r, j]
                if r == 0:
                    a.plot(t, X[j], color="#2e7d32", lw=0.9)
                elif r == 1:
                    a.plot(t, Y[j], color="black", lw=0.9)
                    a.plot(gt / FS, np.full(len(gt), hi + pad * 0.4), "v", color="black", ms=4)
                elif r == 6:
                    for k in range(K):
                        a.plot(t, S[k, j], color=COL["I"], lw=0.5, alpha=0.28)
                    allp = np.concatenate([p for p in P if len(p)] or [np.zeros(0)])
                    a.plot(allp / FS, np.full(len(allp), lo - pad * 0.2), "|", color=COL["I"], ms=6, alpha=0.6)
                    a.plot(gt / FS, np.full(len(gt), hi + pad * 0.4), "v", color="#888888", ms=4)
                else:
                    y, c = series[r]
                    a.plot(t, Y[j], color="#c4c4c4", lw=0.9)
                    a.plot(t, y, color=c, lw=0.9)
                    if r == 7:
                        a.plot(pos / FS, np.full(len(pos), lo - pad * 0.2), "^", color=COL["DEC"], ms=4)
                        a.plot(gt / FS, np.full(len(gt), hi + pad * 0.4), "v", color="#888888", ms=4)
                if r > 0:
                    a.set_ylim(lo - pad, hi + pad)
                a.tick_params(labelsize=7)
                if j == 0:
                    a.set_ylabel(rows[r], fontsize=8, rotation=0, ha="right", va="center")
                if r == 0:
                    a.set_title(f"test window {idx[j]} ({subj[j]})", fontsize=8)
            ax[-1, j].set_xlabel("time (s)", fontsize=8)
            if j == 0:
                overview.append((name, t, X[j], Y[j], p50[j], S[:, j].copy(), wave, gt, pos))
        fig.suptitle(f"{name} — same test windows for every model; grey = target, ▼ = target R peaks, ▲ = consensus events, | = sample votes", fontsize=9.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"qual_{name.lower().replace('-', '')}.{ext}", dpi=140)
        plt.close(fig)
        del gC, gI, gD; torch.cuda.empty_cache()
        print(f"[qual] {name}: windows {list(idx)}", flush=True)

    cols = ["PPG (input)", "PENGUIN · 50 NFE", "iMF · 1 NFE (one sample)", "16 iMF samples", "iMF consensus-decoded"]
    fig, ax = plt.subplots(len(overview), len(cols), figsize=(3.0 * len(cols), 1.5 * len(overview)), squeeze=False)
    for r, (name, t, x, y, p50, S, wave, gt, pos) in enumerate(overview):
        lo, hi = min(y.min(), S.min(), p50.min()), max(y.max(), S.max(), p50.max()); pad = 0.06 * (hi - lo)
        for c in range(len(cols)):
            a = ax[r, c]
            if c == 0:
                a.plot(t, x, color="#2e7d32", lw=0.8); a.set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")
            else:
                a.plot(t, y, color="#c4c4c4", lw=0.8)
                if c == 1:
                    a.plot(t, p50, color=COL["P50"], lw=0.8)
                elif c == 2:
                    a.plot(t, S[0], color=COL["I"], lw=0.8)
                elif c == 3:
                    for k in range(K):
                        a.plot(t, S[k], color=COL["I"], lw=0.45, alpha=0.28)
                else:
                    a.plot(t, wave, color=COL["DEC"], lw=0.8)
                    a.plot(pos / FS, np.full(len(pos), lo - pad * 0.2), "^", color=COL["DEC"], ms=3.5)
                a.plot(gt / FS, np.full(len(gt), hi + pad * 0.4), "v", color="#777777", ms=3.5)
                a.set_ylim(lo - pad, hi + pad)
            a.set_xticks([]); a.tick_params(labelsize=6)
            if r == 0:
                a.set_title(cols[c], fontsize=9)
    fig.suptitle("Five ECG corpora — first selected test window; grey = target, ▼ = target R peaks, ▲ = consensus events", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"qual_overview.{ext}", dpi=140)
    print("[qual] done", flush=True)


if __name__ == "__main__":
    main()
