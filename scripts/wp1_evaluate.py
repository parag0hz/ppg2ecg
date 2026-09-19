"""WP1: evaluate one (fold, arm) on that fold's test subjects; saves outputs/wp1_eval/fold<k>_arm<A>.npz."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cd1_evaluate as CDE  # noqa: E402
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402

PER_SUBJECT = 3000                       # exact linspace per test subject, fixed before any WP1 number
NFES = {"C": (1, 50), "I": (1,), "D": (1,)}
OUT = ROOT / "outputs/wp1_eval"


def load_test(k):
    split = json.loads((ROOT / f"data/manifests/split_wp1_fold{k}.json").read_text())["splits"][0]
    X, Y, S = [], [], []
    for s in split["test"]:
        d = np.load(ROOT / f"data/processed/u2_wildppg/{s}.npz")
        idx = np.unique(np.linspace(0, len(d["x"]) - 1, PER_SUBJECT).round().astype(int))
        X.append(d["x"][idx].astype(np.float32)); Y.append(d["y"][idx].astype(np.float64)); S.append(np.full(len(idx), s))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--arm", choices=list(NFES), required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ck_path = ROOT / f"outputs/wp1_f{a.fold}_arm{a.arm}/checkpoint_last.pt"
    if a.arm == "D":
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        net = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval(); net.load_state_dict(ck["state_dict"])
        gen = lambda X, s, nfe: CDE.sample(net, X, s, nfe, dev)  # noqa: E731
    else:
        net, meta, kind = U2.build(ck_path, dev)
        assert kind == a.arm and int(meta["train_state"]["opt_steps"]) == 14000
        def gen(X, s, nfe, _net=net, _arm=a.arm):
            e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(s))
            return U2.generate(_net, X, e0, nfe, dev, _arm)
    X, Y, subj = load_test(a.fold)
    ex = ProcessPoolExecutor(6)
    ref_hr = V.hr_batch(ex, Y)
    save = {"subj": subj, "ref_hr": ref_hr, "fold": np.array(a.fold), "arm": np.array(a.arm), "checkpoint_sha256": np.array(U2.sha256(ck_path))}
    for nfe in NFES[a.arm]:
        acc, pooled, hr16 = {}, {}, []
        n_draws = 16 if (nfe == 1 and a.arm in ("I", "D")) else 4
        for s in range(n_draws):
            pred, dt = gen(X, s, nfe)
            if nfe == 1 and a.arm in ("I", "D"):
                hr16.append(V.hr_batch(ex, pred))
            if s < 4:
                mt = U2.task_metrics("ECG", 8, pred, Y, subj); tab = mt.pop("_counts")[0]
                for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE"):
                    acc.setdefault(m, []).append(np.asarray(mt[m][0], float))
                for m, v in U2.pooled_extras("ECG", pred, Y, tab).items():
                    pooled.setdefault(m, []).append(v)
            print(f"[wp1] fold {a.fold} {a.arm} NFE {nfe} draw {s} gen {dt:.1f}s", flush=True)
        for m, v in acc.items():
            save[f"nfe{nfe}_{m}"] = np.nanmean(np.vstack(v), 0)
        for m, v in pooled.items():
            save[f"nfe{nfe}_pooled_{m}"] = np.array(np.mean(v))
        if hr16:
            with np.errstate(all="ignore"):
                save["cons16_nfe1_hr_err"] = np.abs(np.nanmedian(np.vstack(hr16), 0) - ref_hr)
    np.savez(OUT / f"fold{a.fold}_arm{a.arm}.npz", **save)
    print("[wp1] saved", flush=True)


if __name__ == "__main__":
    main()
