"""CD1: evaluate the consistency model (arm D) with the SR1 protocol; saves outputs/sr1_eval/arm_D_seed42.npz."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

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
import vm1_evaluate as VM  # noqa: E402
from cd1_train import f_consistency  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402

CK = ROOT / "outputs/cd1_vitaldb_armD_seed42/checkpoint_last.pt"
SCHED = {1: [], 2: [0.5], 4: [0.25, 0.5, 0.75]}          # re-noise times for multistep consistency sampling


@torch.no_grad()
def sample(net, X, seed, nfe, dev, bs=512):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(len(X), 1, X.shape[1], generator=g)
    extra = [torch.randn(len(X), 1, X.shape[1], generator=g) for _ in SCHED[nfe]]
    out, t0 = [], time.perf_counter()
    for i in range(0, len(X), bs):
        ppg = torch.from_numpy(X[i:i + bs]).to(dev).unsqueeze(1)
        z = z0[i:i + bs].to(dev)
        B = z.shape[0]
        x = f_consistency(net, z, ppg, torch.zeros(B, 1, device=dev))
        for t_val, e in zip(SCHED[nfe], extra):
            t = torch.full((B, 1), t_val, device=dev)
            x_t = (1 - t_val) * e[i:i + bs].to(dev) + t_val * x
            x = f_consistency(net, x_t, ppg, t)
        out.append(x[:, 0].float().cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(out).astype(np.float64), time.perf_counter() - t0


def main():
    dev = torch.device("cuda")
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    assert ck["train_state"]["opt_steps"] == 14000
    net = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval(); net.load_state_dict(ck["state_dict"])
    X, Y, pid = VM.load("test")
    ex = ProcessPoolExecutor(6)
    ref_hr = V.hr_batch(ex, Y)
    save = {"pid": pid, "ref_hr": ref_hr, "seed": np.array(42), "arm": np.array("D"), "checkpoint_sha256": np.array(U2.sha256(CK))}
    hr16 = []
    for nfe in (1, 2, 4):
        acc, pooled, secs = {}, {}, []
        for s in range(16 if nfe == 1 else 4):
            pred, dt = sample(net, X, s, nfe, dev); secs.append(dt)
            if nfe == 1:
                hr16.append(V.hr_batch(ex, pred))
            if s < 4:
                mt = U2.task_metrics("ECG", 8, pred, Y, pid); tab = mt.pop("_counts")[0]
                for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE"):
                    acc.setdefault(m, []).append(np.asarray(mt[m][0], float))
                for m, v in U2.pooled_extras("ECG", pred, Y, tab).items():
                    pooled.setdefault(m, []).append(v)
            print(f"[cd1] D NFE {nfe} draw {s} gen {dt:.1f}s", flush=True)
        for m, v in acc.items():
            save[f"nfe{nfe}_{m}"] = np.nanmean(np.vstack(v), 0)
        for m, v in pooled.items():
            save[f"nfe{nfe}_pooled_{m}"] = np.array(np.mean(v))
        save[f"nfe{nfe}_ms_per_window"] = np.array(1000 * np.mean(secs) / len(X))
    with np.errstate(all="ignore"):
        save["cons16_nfe1_hr_err"] = np.abs(np.nanmedian(np.vstack(hr16), 0) - ref_hr)
    np.savez(ROOT / "outputs/sr1_eval/arm_D_seed42.npz", **save)
    print("[cd1] saved", flush=True)


if __name__ == "__main__":
    main()
