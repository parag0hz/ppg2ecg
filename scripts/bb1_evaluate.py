"""BB1: per-arm evaluation on V1 test; saves per-window arrays for the paired verdict (bb1_verdict.py)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402

RUN = {"I": "outputs/v1_vitaldb_armI_seed42", "A": "outputs/bb1_vitaldb_A_seed42", "B": "outputs/bb1_vitaldb_B_seed42"}
OUT = ROOT / "outputs/bb1_eval"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=list(RUN), required=True)
    arm = ap.parse_args().arm
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    import json
    split = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())["splits"][0]
    X, Y, P = [], [], []
    for c in split["test"]:
        d = np.load(ROOT / f"data/processed/v1_vitaldb/{c}.npz")
        X.append(d["x"]); Y.append(d["y"].astype(np.float64)); P.append(np.full(len(d["x"]), int(d["subjectid"])))
    X, Y, pid = np.concatenate(X), np.concatenate(Y), np.concatenate(P)
    ck = ROOT / RUN[arm] / "checkpoint_last.pt"
    net, meta, kind = U2.build(ck, dev)
    assert kind == "I" and int(meta["train_state"]["opt_steps"]) == 14000
    from concurrent.futures import ProcessPoolExecutor
    ex = ProcessPoolExecutor(6)
    ref_hr = V.hr_batch(ex, Y)
    save = {"pid": pid, "ref_hr": ref_hr}
    hr16 = []
    for nfe in (1, 2, 4):
        acc, pooled, secs = {}, {}, []
        for seed in range(16 if nfe == 1 else 4):
            e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(seed))
            pred, dt = U2.generate(net, X, e0, nfe, dev, "I"); secs.append(dt)
            if nfe == 1:
                hr16.append(V.hr_batch(ex, pred))
            if seed < 4:
                mt = U2.task_metrics("ECG", 8, pred, Y, pid); tab = mt.pop("_counts")[0]
                for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE"):
                    acc.setdefault(m, []).append(np.asarray(mt[m][0], float))
                for m, v in U2.pooled_extras("ECG", pred, Y, tab).items():
                    pooled.setdefault(m, []).append(v)
            print(f"[bb1] {arm} NFE {nfe} seed {seed} gen {dt:.1f}s", flush=True)
        for m, v in acc.items():
            save[f"nfe{nfe}_{m}"] = np.nanmean(np.vstack(v), 0)
        for m, v in pooled.items():
            save[f"nfe{nfe}_pooled_{m}"] = np.array(np.mean(v))
        save[f"nfe{nfe}_ms_per_window"] = np.array(1000 * np.mean(secs) / len(X))
    with np.errstate(all="ignore"):
        save["cons16_nfe1_hr_err"] = np.abs(np.nanmedian(np.vstack(hr16), 0) - ref_hr)
    save["checkpoint_sha256"] = np.array(U2.sha256(ck))
    np.savez(OUT / f"arm_{arm}.npz", **save)
    print(f"[bb1] {arm} saved", flush=True)


if __name__ == "__main__":
    main()
