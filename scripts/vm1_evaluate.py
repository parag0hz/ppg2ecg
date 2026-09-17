"""VM1 per-arm evaluation: CFG setting chosen on V1 validation patients, then the V1 test protocol.
Saves outputs/vm1_eval/arm_<arm>.npz with the keys bb1_evaluate.py writes (pairing with arm I is by window order)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
from ppg2ecg.flow.imf_vanilla import sample_vimf  # noqa: E402
from ppg2ecg.models.imf_dit import ImfDiT1d  # noqa: E402

RUN = {"S": "outputs/vm1_S_seed42", "P": "outputs/vm1_P_seed42", "B": "outputs/vm1_B_seed42"}
GRID = [(1.0, 0.0, 1.0)] + [(w, a, b) for w in (1.5, 2.0, 3.0, 5.0) for (a, b) in ((0.0, 1.0), (0.4, 0.65))]
OUT = ROOT / "outputs/vm1_eval"


def load(role):
    split = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())["splits"][0]
    X, Y, P = [], [], []
    for c in split[role]:
        d = np.load(ROOT / f"data/processed/v1_vitaldb/{c}.npz")
        X.append(d["x"]); Y.append(d["y"].astype(np.float64)); P.append(np.full(len(d["x"]), int(d["subjectid"])))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(P)


@torch.no_grad()
def generate(net, X, seed, nfe, cfg, dev, bs=256):
    e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(seed))
    out, t0 = [], time.perf_counter()
    for i in range(0, len(X), bs):
        z, k = sample_vimf(net, torch.from_numpy(X[i:i + bs]).to(dev)[:, None], e0[i:i + bs].to(dev), nfe, *cfg)
        assert k == nfe
        out.append(z[:, 0].float().cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(out).astype(np.float64), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=list(RUN), required=True)
    arm = ap.parse_args().arm
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ck_path = ROOT / RUN[arm] / "checkpoint_last.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert ck["train_state"]["opt_steps"] == 14000
    net = ImfDiT1d(**ck["model_cfg"]).to(dev).eval()
    net.load_state_dict(ck["state_dict"])
    ex = ProcessPoolExecutor(6)

    # 1) CFG setting on validation patients (NFE 1, noise seeds 0-1): min patient-macro HR error, tie (< 0.01) -> max F1
    Xv, Yv, pv = load("val")
    sel = []
    for cfg in GRID:
        hr, f1 = [], []
        for seed in (0, 1):
            pred, _ = generate(net, Xv, seed, 1, cfg, dev)
            mt = U2.task_metrics("ECG", 8, pred, Yv, pv)
            hr.append(mt["HR"][0]); f1.append(mt["Rpeak_F1"][0])
        sel.append(dict(omega=cfg[0], t_min=cfg[1], t_max=cfg[2],
                        hr=V.cluster_ci(np.nanmean(np.vstack(hr), 0), pv)[0], f1=V.cluster_ci(np.nanmean(np.vstack(f1), 0), pv)[0]))
        print(f"[vm1] {arm} val {cfg} HR {sel[-1]['hr']:.3f} F1 {sel[-1]['f1']:.4f}", flush=True)
    best_hr = min(s["hr"] for s in sel)
    chosen = max((s for s in sel if s["hr"] <= best_hr + 0.01), key=lambda s: s["f1"])
    cfg = (chosen["omega"], chosen["t_min"], chosen["t_max"])
    print(f"[vm1] {arm} chosen CFG {cfg}", flush=True)

    # 2) test protocol (= bb1_evaluate.py) at the chosen setting
    X, Y, pid = load("test")
    ref_hr = V.hr_batch(ex, Y)
    save = {"pid": pid, "ref_hr": ref_hr, "cfg": np.array(cfg), "selection": np.array(json.dumps(sel))}
    hr16 = []
    for nfe in (1, 2, 4):
        acc, pooled, secs = {}, {}, []
        for seed in range(16 if nfe == 1 else 4):
            pred, dt = generate(net, X, seed, nfe, cfg, dev); secs.append(dt)
            if nfe == 1:
                hr16.append(V.hr_batch(ex, pred))
            if seed < 4:
                mt = U2.task_metrics("ECG", 8, pred, Y, pid); tab = mt.pop("_counts")[0]
                for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE"):
                    acc.setdefault(m, []).append(np.asarray(mt[m][0], float))
                for m, v in U2.pooled_extras("ECG", pred, Y, tab).items():
                    pooled.setdefault(m, []).append(v)
            print(f"[vm1] {arm} test NFE {nfe} seed {seed} gen {dt:.1f}s", flush=True)
        for m, v in acc.items():
            save[f"nfe{nfe}_{m}"] = np.nanmean(np.vstack(v), 0)
        for m, v in pooled.items():
            save[f"nfe{nfe}_pooled_{m}"] = np.array(np.mean(v))
        save[f"nfe{nfe}_ms_per_window"] = np.array(1000 * np.mean(secs) / len(X))
    with np.errstate(all="ignore"):
        save["cons16_nfe1_hr_err"] = np.abs(np.nanmedian(np.vstack(hr16), 0) - ref_hr)
    # 3) descriptive: no guidance (omega = 1) at NFE 1
    acc = {}
    for seed in range(4):
        pred, _ = generate(net, X, seed, 1, (1.0, 0.0, 1.0), dev)
        mt = U2.task_metrics("ECG", 8, pred, Y, pid); tab = mt.pop("_counts")[0]
        for m in ("HR", "Rpeak_F1"):
            acc.setdefault(m, []).append(np.asarray(mt[m][0], float))
        acc.setdefault("FD", []).append(U2.pooled_extras("ECG", pred, Y, tab)["FD_kanflow"])
    save["noguid_nfe1_HR"] = np.nanmean(np.vstack(acc["HR"]), 0)
    save["noguid_nfe1_Rpeak_F1"] = np.nanmean(np.vstack(acc["Rpeak_F1"]), 0)
    save["noguid_nfe1_pooled_FD_kanflow"] = np.array(np.mean(acc["FD"]))
    save["checkpoint_sha256"] = np.array(U2.sha256(ck_path))
    np.savez(OUT / f"arm_{arm}.npz", **save)
    print(f"[vm1] {arm} saved", flush=True)


if __name__ == "__main__":
    main()
