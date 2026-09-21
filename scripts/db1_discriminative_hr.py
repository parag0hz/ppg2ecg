"""DB1 — direct discriminative baseline PPG -> HR (docs/DB1_DISCRIMINATIVE_HR_BASELINE_PREREGISTRATION.md)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from ppg2ecg.utils.seed import seed_everything  # noqa: E402

OUT, RUN = ROOT / "artifacts/db1_discriminative_hr", ROOT / "outputs/db1_hr_regressor"
STEPS, BATCH, FS = 14000, 64, 128


class HRNet(nn.Module):
    def __init__(self):
        super().__init__()
        ch, layers, c_in = (32, 64, 128, 128, 256, 256), [], 1
        for c in ch:
            layers += [nn.Conv1d(c_in, c, 7, stride=2, padding=3), nn.BatchNorm1d(c), nn.GELU()]; c_in = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(c_in, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):                       # [B, T] -> HR / 100
        return self.head(self.body(x.unsqueeze(1)).mean(-1)).squeeze(-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    seed_everything(42, deterministic=False)
    dev = torch.device("cuda")
    ex = ProcessPoolExecutor(10)
    lab = RUN / "train_hr_labels.npz"
    Xtr, Ytr, _ = VM.load("train")
    if lab.exists():
        hr_tr = np.load(lab)["hr"]
    else:
        hr_tr = V.hr_batch(ex, Ytr); np.savez(lab, hr=hr_tr)
    ok = np.isfinite(hr_tr)
    Xt, Ht = torch.from_numpy(Xtr[ok].astype(np.float32)).to(dev), torch.from_numpy((hr_tr[ok] / 100).astype(np.float32)).to(dev)
    net = HRNet().to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(42)
    perm, pos, t0, acc = torch.randperm(len(Xt), generator=g), 0, time.time(), []
    net.train()
    for step in range(1, STEPS + 1):
        if pos + BATCH > len(perm):
            perm, pos = torch.randperm(len(Xt), generator=g), 0
        b = perm[pos:pos + BATCH].to(dev); pos += BATCH
        loss = (net(Xt[b]) - Ht[b]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); acc.append(loss.item())
        if step % 1000 == 0:
            print(f"[db1] step {step} L1 {100 * np.mean(acc):.2f} bpm", flush=True); acc = []
    torch.save({"state_dict": net.state_dict(), "opt_steps": STEPS}, RUN / "checkpoint_last.pt")
    train_s = time.time() - t0

    X, Y, pid = VM.load("test")
    ref_hr = V.hr_batch(ex, Y)
    net.eval()
    with torch.no_grad():
        t1 = time.time()
        pred = np.concatenate([100 * net(torch.from_numpy(X[i:i + 1024]).to(dev)).float().cpu().numpy() for i in range(0, len(X), 1024)])
        torch.cuda.synchronize(); infer_ms = 1000 * (time.time() - t1) / len(X)
    e_reg = np.abs(pred - ref_hr)
    e_ppg = np.abs(np.array([R.hr_bpm(find_peaks(x, distance=42, prominence=0.3)[0], FS) for x in X.astype(np.float64)]) - ref_hr)
    sr1 = {a: dict(np.load(ROOT / f"outputs/sr1_eval/arm_{a}_seed42.npz")) for a in ("C", "I", "D")}
    assert all(np.array_equal(d["pid"], pid) for d in sr1.values())
    others = {"iMF K=16 consensus": sr1["I"]["cons16_nfe1_hr_err"], "CD K=16 consensus": sr1["D"]["cons16_nfe1_hr_err"],
              "PENGUIN 50 NFE (one sample)": sr1["C"]["nfe50_HR"], "iMF 1 NFE (one sample)": sr1["I"]["nfe1_HR"], "PPG peak counting": e_ppg}
    ci = lambda v: V.cluster_ci(v, pid)  # noqa: E731
    res = {"regressor": {"hr_err": ci(e_reg), "params": n_par, "train_seconds": round(train_s, 1), "ms_per_window_gpu_batch1024": infer_ms},
           "others": {k: ci(v) for k, v in others.items()},
           "regressor_minus_other": {k: ci(e_reg - v) for k, v in others.items()}}
    res["regressor_beats_iMF_consensus"] = bool(res["regressor_minus_other"]["iMF K=16 consensus"][2] < 0)
    np.savez(RUN / "test_errors.npz", pid=pid, e_reg=e_reg, pred=pred, ref_hr=ref_hr)
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
