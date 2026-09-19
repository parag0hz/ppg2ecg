"""RF1 step 1: teacher pairs (z0, x1_hat) from arm C at 50 NFE (docs/RF1_REFLOW_PREREGISTRATION.md)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import u2_evaluate as U2  # noqa: E402

N_PAIRS, PAIR_SEED, NFE = 64_000, 20260919, 50
OUT = ROOT / "outputs/rf1_pairs"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    split = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())["splits"][0]
    X, Y = [], []
    for c in split["train"]:
        d = np.load(ROOT / f"data/processed/v1_vitaldb/{c}.npz")
        X.append(d["x"]); Y.append(d["y"])
    X = np.concatenate(X)
    idx = np.unique(np.linspace(0, len(X) - 1, N_PAIRS).round().astype(int))
    X = X[idx].astype(np.float32)
    net, meta, kind = U2.build(ROOT / "outputs/v1_vitaldb_armC_seed42/checkpoint_last.pt", dev)
    assert kind == "C" and int(meta["train_state"]["opt_steps"]) == 14000
    z0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(PAIR_SEED))
    t0 = time.perf_counter()
    x1, _ = U2.generate(net, X, z0, NFE, dev, "C")
    secs = time.perf_counter() - t0
    np.savez(OUT / "pairs.npz", ppg=X, z0=z0[:, 0].numpy().astype(np.float16), x1=x1.astype(np.float16),
             window_index=idx, teacher_seconds=np.array(secs), nfe=np.array(NFE), pair_seed=np.array(PAIR_SEED))
    print(f"[rf1] {len(X)} pairs, teacher {secs/60:.1f} min, x1 std {x1.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
