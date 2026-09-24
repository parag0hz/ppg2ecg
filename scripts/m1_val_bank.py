"""M1 — generate the VitalDB VALIDATION sample bank (docs/M1_PILOT_GAIN_PREDICTION_PREREGISTRATION.md §2).

No training, no new model, no new functional. Frozen components only:
  windows    vm1_evaluate.load("val")            (already preprocessed on disk)
  samplers   dw1_depth_width.make_sampler        (unchanged iMF / CD / PENGUIN samplers, seed-42 checkpoints)
  functional v1_evaluate.hr_batch / _hr          (neurokit, 128 Hz), reference = same functional on the target ECG
  draws      noise seeds 0 .. 15, S in {1, 2, 4, 8}

No test window is read. Writes outputs/m1_pilot_gain_prediction/val_hr_{arm}_S{S}.npy (16, n_val) and val_ref.npz.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from dw1_depth_width import make_sampler  # noqa: E402

ARMS, STEPS, K = ("I", "D", "C"), (1, 2, 4, 8), 16
OUT = ROOT / "outputs/m1_pilot_gain_prediction"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    X, Y, pid = VM.load("val")
    print(f"[m1] val windows {len(X)} patients {len(np.unique(pid))}", flush=True)
    ex = ProcessPoolExecutor(8)

    ref_path = OUT / "val_ref.npz"
    if ref_path.exists():
        ref = np.load(ref_path)["ref_hr"]
    else:
        t0 = time.time()
        ref = V.hr_batch(ex, Y)
        np.savez(ref_path, ref_hr=ref, pid=pid)
        print(f"[m1] reference HR {np.isfinite(ref).sum()}/{len(ref)} finite ({time.time() - t0:.0f}s)", flush=True)

    for arm in ARMS:
        todo = [S for S in STEPS if not (OUT / f"val_hr_{arm}_S{S}.npy").exists()]
        if not todo:
            print(f"[m1] {arm}: all depths present, skipping", flush=True)
            continue
        sampler = make_sampler(arm, dev)
        for S in todo:
            t0 = time.time()
            hr = np.full((K, len(X)), np.nan, np.float64)
            for k in range(K):
                hr[k] = V.hr_batch(ex, sampler(X, k, S))
            np.save(OUT / f"val_hr_{arm}_S{S}.npy", hr)
            finite = int((np.isfinite(hr).all(0) & np.isfinite(ref)).sum())
            print(f"[m1] {arm} S={S} K={K} saved; windows all-finite {finite}/{len(X)} ({time.time() - t0:.0f}s)", flush=True)
        del sampler
        torch.cuda.empty_cache()
    print("[m1] validation bank complete", flush=True)


if __name__ == "__main__":
    main()
