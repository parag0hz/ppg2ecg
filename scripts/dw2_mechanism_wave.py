"""DW2 part B (waveform half): pairwise waveform / feature-space diversity by depth on a fixed 2,000-window subset."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dw1_depth_width as D  # noqa: E402
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402

OUT, STEPS, N_SUB = ROOT / "artifacts/dw2_mechanism", (1, 2, 4, 8, 16, 32), 2000


def pairwise_mean_dist(A):
    """A [K, n, d] -> per-window mean over pairs of RMS distance, [n]."""
    K = A.shape[0]
    tot, cnt = np.zeros(A.shape[1]), 0
    for i in range(K):
        for j in range(i + 1, K):
            tot += np.sqrt(np.mean((A[i] - A[j]) ** 2, axis=-1)); cnt += 1
    return tot / max(cnt, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    X, Y, pid = VM.load("test")
    idx = np.unique(np.linspace(0, len(X) - 1, N_SUB).round().astype(int))
    X, Y, pid = X[idx], Y[idx], pid[idx]
    ref_feat = PMX.default_feature_map(Y, 128)
    rows = []
    for arm in ("I", "D", "C"):
        sampler = D.make_sampler(arm, dev)
        for S in STEPS:
            K = min(8, 32 // S)
            W = np.stack([sampler(X, s, S) for s in range(K)])                       # [K, n, T]
            wave_div = pairwise_mean_dist(W) if K > 1 else np.full(len(X), np.nan)
            F = np.stack([PMX.default_feature_map(W[k], 128) for k in range(K)])     # [K, n, d]
            feat_div = pairwise_mean_dist(F) if K > 1 else np.full(len(X), np.nan)
            feat_gap = np.mean([np.sqrt(np.mean((F[k] - ref_feat) ** 2, axis=-1)) for k in range(K)], axis=0)
            wd, fd = V.cluster_ci(wave_div, pid), V.cluster_ci(feat_div, pid)
            rows.append(dict(model=D.NAME[arm], S=S, K=K, wave_pairwise_rms=wd[0], wave_lo=wd[1], wave_hi=wd[2],
                             feat_pairwise_rms=fd[0], feat_lo=fd[1], feat_hi=fd[2], feat_dist_to_target=float(np.nanmean(feat_gap)),
                             sample_std_of_waveform=float(np.mean(W.std(axis=0))) if K > 1 else np.nan))
            print(f"[dw2-wave] {D.NAME[arm]:26s} S={S:2d} K={K}  wave pairwise RMS {wd[0]:.4f}  feature pairwise {fd[0]:.4f}  feature→target {rows[-1]['feat_dist_to_target']:.4f}", flush=True)
        del sampler; torch.cuda.empty_cache()
    with open(OUT / "waveform_diversity_by_depth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("[dw2-wave] done", flush=True)


if __name__ == "__main__":
    main()
