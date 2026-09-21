"""KN1: evaluate one KAN arm with the SR1 protocol -> outputs/sr1_eval/arm_<name>_seed42.npz."""
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

ARMS = {"IKo": ("outputs/kn1_I_kan_outer", "I", (1, 2, 4)), "IKa": ("outputs/kn1_I_kan_all", "I", (1, 2, 4)),
        "CKo": ("outputs/kn1_C_kan_outer", "C", (50, 1))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=list(ARMS), required=True)
    name = ap.parse_args().arm
    path, kind, nfes = ARMS[name]
    dev = torch.device("cuda")
    ck = ROOT / path / "checkpoint_last.pt"
    net, meta, k = U2.build(ck, dev)
    assert k == kind and int(meta["train_state"]["opt_steps"]) == 14000 and meta["model_cfg"].get("arch") == "kan"
    X, Y, pid = VM.load("test")
    ex = ProcessPoolExecutor(6)
    ref_hr = V.hr_batch(ex, Y)
    save, hr16 = {"pid": pid, "ref_hr": ref_hr, "checkpoint_sha256": np.array(U2.sha256(ck))}, []
    for nfe in nfes:
        acc, pooled = {}, {}
        for s in range(16 if (nfe == 1 and kind == "I") else 4):
            e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(s))
            pred, dt = U2.generate(net, X, e0, nfe, dev, kind)
            if nfe == 1 and kind == "I":
                hr16.append(V.hr_batch(ex, pred))
            if s < 4:
                mt = U2.task_metrics("ECG", 8, pred, Y, pid); tab = mt.pop("_counts")[0]
                for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE"):
                    acc.setdefault(m, []).append(np.asarray(mt[m][0], float))
                for m, v in U2.pooled_extras("ECG", pred, Y, tab).items():
                    pooled.setdefault(m, []).append(v)
            print(f"[kn1] {name} NFE {nfe} draw {s} gen {dt:.1f}s", flush=True)
        for m, v in acc.items():
            save[f"nfe{nfe}_{m}"] = np.nanmean(np.vstack(v), 0)
        for m, v in pooled.items():
            save[f"nfe{nfe}_pooled_{m}"] = np.array(np.mean(v))
    if hr16:
        with np.errstate(all="ignore"):
            save["cons16_nfe1_hr_err"] = np.abs(np.nanmedian(np.vstack(hr16), 0) - ref_hr)
    np.savez(ROOT / f"outputs/sr1_eval/arm_{name}_seed42.npz", **save)
    print(f"[kn1] {name} saved", flush=True)


if __name__ == "__main__":
    main()
