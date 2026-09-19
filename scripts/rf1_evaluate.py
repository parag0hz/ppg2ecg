"""RF1 step 3: evaluate arm R (2-rectified flow) with the SR1 protocol; saves outputs/sr1_eval/arm_R_seed42.npz."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

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

CK = ROOT / "outputs/rf1_vitaldb_armR_seed42/checkpoint_last.pt"
OUT = ROOT / "outputs/sr1_eval"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    net, meta, kind = U2.build(CK, dev)
    assert kind == "C" and meta["train_state"]["opt_steps"] == 14000    # arm R stores a bare backbone, like arm C
    X, Y, pid = VM.load("test")
    ex = ProcessPoolExecutor(6)
    ref_hr = V.hr_batch(ex, Y)
    save = {"pid": pid, "ref_hr": ref_hr, "seed": np.array(42), "arm": np.array("R"),
            "checkpoint_sha256": np.array(U2.sha256(CK))}
    hr16 = []
    for nfe in (1, 2, 4):
        acc, pooled, secs = {}, {}, []
        for s in range(16 if nfe == 1 else 4):
            e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(s))
            pred, dt = U2.generate(net, X, e0, nfe, dev, "C"); secs.append(dt)
            if nfe == 1:
                hr16.append(V.hr_batch(ex, pred))
            if s < 4:
                mt = U2.task_metrics("ECG", 8, pred, Y, pid); tab = mt.pop("_counts")[0]
                for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE"):
                    acc.setdefault(m, []).append(np.asarray(mt[m][0], float))
                for m, v in U2.pooled_extras("ECG", pred, Y, tab).items():
                    pooled.setdefault(m, []).append(v)
            print(f"[rf1] R NFE {nfe} draw {s} gen {dt:.1f}s", flush=True)
        for m, v in acc.items():
            save[f"nfe{nfe}_{m}"] = np.nanmean(np.vstack(v), 0)
        for m, v in pooled.items():
            save[f"nfe{nfe}_pooled_{m}"] = np.array(np.mean(v))
        save[f"nfe{nfe}_ms_per_window"] = np.array(1000 * np.mean(secs) / len(X))
    with np.errstate(all="ignore"):
        save["cons16_nfe1_hr_err"] = np.abs(np.nanmedian(np.vstack(hr16), 0) - ref_hr)
    np.savez(OUT / "arm_R_seed42.npz", **save)
    print("[rf1] saved", flush=True)


if __name__ == "__main__":
    main()
