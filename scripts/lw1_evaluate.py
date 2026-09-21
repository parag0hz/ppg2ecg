"""LW1: evaluate one auxiliary-loss arm (or the reference arm I) with PCC and diversity added."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402

ARMS = {"I": "outputs/v1_vitaldb_armI_seed42", "W0": "outputs/lw1_W0", "W5": "outputs/lw1_W5", "W10": "outputs/lw1_W10", "SP": "outputs/lw1_SP"}


def pcc_rows(a, b):
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (ac * bc).sum(1) / (np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1) + 1e-12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=list(ARMS), required=True)
    name = ap.parse_args().arm
    dev = torch.device("cuda")
    ck = ROOT / ARMS[name] / "checkpoint_last.pt"
    net, meta, kind = U2.build(ck, dev)
    assert kind == "I" and int(meta["train_state"]["opt_steps"]) == 14000
    X, Y, pid = VM.load("test")
    ex = ProcessPoolExecutor(6)
    ref_hr = V.hr_batch(ex, Y)
    acc, pooled, hrs = {}, {}, []
    for s in range(16):
        e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(s))
        pred, dt = U2.generate(net, X, e0, 1, dev, "I")
        hrs.append(V.hr_batch(ex, pred))
        if s < 4:
            mt = U2.task_metrics("ECG", 8, pred, Y, pid); tab = mt.pop("_counts")[0]
            for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE"):
                acc.setdefault(m, []).append(np.asarray(mt[m][0], float))
            acc.setdefault("PCC", []).append(pcc_rows(pred, Y))
            for m, v in U2.pooled_extras("ECG", pred, Y, tab).items():
                pooled.setdefault(m, []).append(v)
        print(f"[lw1] {name} draw {s} gen {dt:.1f}s", flush=True)
    H = np.vstack(hrs)
    save = {"pid": pid, "ref_hr": ref_hr, **{f"nfe1_{m}": np.nanmean(np.vstack(v), 0) for m, v in acc.items()},
            **{f"nfe1_pooled_{m}": np.array(np.mean(v)) for m, v in pooled.items()}}
    with np.errstate(all="ignore"):
        save["cons16_nfe1_hr_err"] = np.abs(np.nanmedian(H, 0) - ref_hr)
        save["hr_sd_across_samples"] = np.nanstd(H, 0)
    (ROOT / "outputs/lw1_eval").mkdir(exist_ok=True)
    np.savez(ROOT / f"outputs/lw1_eval/arm_{name}.npz", **save)
    print(f"[lw1] {name} saved", flush=True)


if __name__ == "__main__":
    main()
