"""FBC1 — generate the MISSING VitalDB VALIDATION draws for the B = 32 / B = 16 allocation grids
(docs/FBC1_CALIBRATION_ALLOCATION_AUDIT.md).

No training, no new model, no new functional. Frozen components only:
  windows    vm1_evaluate.load("val")            (already preprocessed on disk)
  samplers   dw1_depth_width.make_sampler        (unchanged iMF / CD / PENGUIN-Euler samplers, seed-42 checkpoints)
  functional v1_evaluate.hr_batch / _hr          (neurokit, 128 Hz)
  draws      noise seed k = row k (torch.Generator().manual_seed(k) over the whole split, as DW1 / M1)

Missing on validation (the M1 bank has seeds 0-15 at S in {1, 2, 4, 8}):
  S = 1  seeds 16..31   (K = 32 at B = 32)
  S = 16 seeds 0, 1     (K = 2 at B = 32, K = 1 at B = 16)
  S = 32 seed 0         (K = 1 at B = 32)
Determinism check: seed 15 at S = 1 is regenerated and compared with row 15 of the M1 bank.
No test window and no reference HR is read. Writes outputs/fbc1_calibration_allocation/val_hr_{arm}_S{S}_seeds{a}_{b}.npy.
"""
from __future__ import annotations

import json
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

ARMS = ("I", "D", "C")
TODO = {1: range(16, 32), 16: range(0, 2), 32: range(0, 1)}
M1 = ROOT / "outputs/m1_pilot_gain_prediction"
OUT = ROOT / "outputs/fbc1_calibration_allocation"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    X, _, pid = VM.load("val")
    print(f"[fbc1-bank] val windows {len(X)} patients {len(np.unique(pid))}", flush=True)
    ex = ProcessPoolExecutor(8)
    check = {}
    for arm in ARMS:
        sampler = make_sampler(arm, dev)
        old = np.load(M1 / f"val_hr_{arm}_S1.npy")[15]
        new = V.hr_batch(ex, sampler(X, 15, 1))
        same_nan = bool(np.array_equal(np.isnan(old), np.isnan(new)))
        dmax = float(np.nanmax(np.abs(old - new)))
        check[arm] = {"seed": 15, "S": 1, "nan_pattern_identical": same_nan, "max_abs_diff_bpm": dmax}
        print(f"[fbc1-bank] {arm} determinism S=1 seed 15: nan-identical {same_nan} max|diff| {dmax:.2e}", flush=True)
        assert same_nan and dmax < 1e-6, "validation sampler does not reproduce the M1 bank"
        for S, seeds in TODO.items():
            f = OUT / f"val_hr_{arm}_S{S}_seeds{seeds[0]}_{seeds[-1]}.npy"
            if f.exists():
                print(f"[fbc1-bank] {arm} S={S} present, skipping", flush=True)
                continue
            t0 = time.time()
            hr = np.full((len(seeds), len(X)), np.nan, np.float64)
            for i, k in enumerate(seeds):
                hr[i] = V.hr_batch(ex, sampler(X, k, S))
            np.save(f, hr)
            print(f"[fbc1-bank] {arm} S={S} seeds {seeds[0]}..{seeds[-1]} saved; draw-level non-finite "
                  f"{100 * np.isnan(hr).mean():.2f} % ({time.time() - t0:.0f}s)", flush=True)
        del sampler
        torch.cuda.empty_cache()
    (OUT / "val_bank_check.json").write_text(json.dumps(check, indent=1))
    print("[fbc1-bank] validation bank complete", flush=True)


if __name__ == "__main__":
    main()
