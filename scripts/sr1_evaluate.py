"""SR1 per-(arm, seed) evaluation on the V1 test patients; saves per-window arrays for sr1_verdict.py."""
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
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from ppg2ecg.models.imf_dit import ImfDiT1d  # noqa: E402

OUT = ROOT / "outputs/sr1_eval"
NFES = {"C": (1, 2, 4, 50), "I": (1, 2, 4), "S": (1, 2, 4)}


def run_dir(arm: str, seed: int) -> Path:
    if seed == 42:
        return ROOT / {"C": "outputs/v1_vitaldb_armC_seed42", "I": "outputs/v1_vitaldb_armI_seed42", "S": "outputs/vm1_S_seed42"}[arm]
    return ROOT / f"outputs/sr1_{arm}_seed{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(NFES), required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ck_path = run_dir(a.arm, a.seed) / "checkpoint_last.pt"
    cfg = None
    if a.arm == "S":
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        assert ck["train_state"]["opt_steps"] == 14000
        net = ImfDiT1d(**ck["model_cfg"]).to(dev).eval()
        net.load_state_dict(ck["state_dict"])
        Xv, Yv, pv = VM.load("val")                       # VM1 rule: CFG chosen on validation patients, per seed
        sel = []
        for c in VM.GRID:
            hr, f1 = [], []
            for s in (0, 1):
                pred, _ = VM.generate(net, Xv, s, 1, c, dev)
                mt = U2.task_metrics("ECG", 8, pred, Yv, pv)
                hr.append(mt["HR"][0]); f1.append(mt["Rpeak_F1"][0])
            sel.append(dict(omega=c[0], t_min=c[1], t_max=c[2], hr=V.cluster_ci(np.nanmean(np.vstack(hr), 0), pv)[0],
                            f1=V.cluster_ci(np.nanmean(np.vstack(f1), 0), pv)[0]))
        best = min(s["hr"] for s in sel)
        ch = max((s for s in sel if s["hr"] <= best + 0.01), key=lambda s: s["f1"])
        cfg = (ch["omega"], ch["t_min"], ch["t_max"])
        print(f"[sr1] S seed {a.seed} CFG {cfg}", flush=True)
        gen = lambda X, seed, nfe: VM.generate(net, X, seed, nfe, cfg, dev)  # noqa: E731
    else:
        net, meta, kind = U2.build(ck_path, dev)
        assert kind == a.arm and int(meta["train_state"]["opt_steps"]) == 14000
        def gen(X, seed, nfe, _net=net, _arm=a.arm):
            e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(seed))
            return U2.generate(_net, X, e0, nfe, dev, _arm)

    X, Y, pid = VM.load("test")
    ex = ProcessPoolExecutor(6)
    ref_hr = V.hr_batch(ex, Y)
    save = {"pid": pid, "ref_hr": ref_hr, "seed": np.array(a.seed), "arm": np.array(a.arm),
            "cfg": np.array(cfg if cfg else (0.0, 0.0, 0.0)), "checkpoint_sha256": np.array(U2.sha256(ck_path))}
    if cfg:
        save["selection"] = np.array(json.dumps(sel))
    hr16 = []
    for nfe in NFES[a.arm]:
        acc, pooled, secs = {}, {}, []
        n_seeds = 16 if nfe == 1 else 4
        for s in range(n_seeds):
            pred, dt = gen(X, s, nfe); secs.append(dt)
            if nfe == 1:
                hr16.append(V.hr_batch(ex, pred))
            if s < 4:
                mt = U2.task_metrics("ECG", 8, pred, Y, pid); tab = mt.pop("_counts")[0]
                for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE"):
                    acc.setdefault(m, []).append(np.asarray(mt[m][0], float))
                for m, v in U2.pooled_extras("ECG", pred, Y, tab).items():
                    pooled.setdefault(m, []).append(v)
            print(f"[sr1] {a.arm} seed {a.seed} NFE {nfe} draw {s} gen {dt:.1f}s", flush=True)
        for m, v in acc.items():
            save[f"nfe{nfe}_{m}"] = np.nanmean(np.vstack(v), 0)
        for m, v in pooled.items():
            save[f"nfe{nfe}_pooled_{m}"] = np.array(np.mean(v))
        save[f"nfe{nfe}_ms_per_window"] = np.array(1000 * np.mean(secs) / len(X))
    with np.errstate(all="ignore"):
        save["cons16_nfe1_hr_err"] = np.abs(np.nanmedian(np.vstack(hr16), 0) - ref_hr)
    np.savez(OUT / f"arm_{a.arm}_seed{a.seed}.npz", **save)
    print(f"[sr1] {a.arm} seed {a.seed} saved", flush=True)


if __name__ == "__main__":
    main()
