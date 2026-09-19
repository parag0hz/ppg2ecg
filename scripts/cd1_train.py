"""CD1: consistency distillation from arm C on the PENGUIN backbone (docs/CD1_CONSISTENCY_DISTILLATION_PREREGISTRATION.md)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import copy
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.models import build_penguin_backbone, count_params
from ppg2ecg.training.train_a0 import git_sha, load_arrays
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import UPSTREAM_COMMIT, assert_upstream_pinned

ROOT = Path(__file__).resolve().parents[1]


def f_consistency(net, x, ppg, t):
    """f(x, t) = x + (1 - t) v(x, t); t is [B, 1]."""
    return x + (1 - t).reshape(-1, 1, 1) * net.forward_step(x, ppg, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/cd1_vitaldb_armD_seed42"))
    ap.add_argument("--teacher", default=str(ROOT / "outputs/v1_vitaldb_armC_seed42/checkpoint_last.pt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=14000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--n-disc", type=int, default=25)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--log-every", type=int, default=220)
    ap.add_argument("--manifest", default="data/manifests/split_v1_vitaldb_seed42.json")
    ap.add_argument("--processed", default="data/processed/v1_vitaldb")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, deterministic=False)
    up = assert_upstream_pinned()
    dev = torch.device("cuda")
    split = read_manifest(ROOT / args.manifest)[0]
    x_tr, y_tr, _ = load_arrays(ROOT / args.processed, split["train"])
    X, Y = torch.from_numpy(x_tr).to(dev), torch.from_numpy(y_tr).to(dev)
    ck = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if "model_cfg" not in ck:
        ck["model_cfg"] = torch.load(Path(args.teacher).with_name("checkpoint_best.pt"), map_location="cpu", weights_only=False)["model_cfg"]
    model_cfg = ck["model_cfg"]
    teacher = build_penguin_backbone(**model_cfg).to(dev).eval(); teacher.load_state_dict(ck["state_dict"]); teacher.requires_grad_(False)
    student = build_penguin_backbone(**model_cfg).to(dev); student.load_state_dict(ck["state_dict"])
    target = copy.deepcopy(student).eval(); target.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    T = X.shape[1]
    c = 0.00054 * math.sqrt(T)
    grid = torch.linspace(0, 1, args.n_disc + 1, device=dev)
    g = torch.Generator().manual_seed(args.seed)
    meta = {"exp": out.name, "objective": "consistency_distillation", "model_cfg": model_cfg, "args": vars(args),
            "params": count_params(student, exclude_prefixes=("cross_attn", "revin")), "pseudo_huber_c": c,
            "teacher": args.teacher, "git": git_sha(ROOT), "upstream": up, "started": datetime.now().isoformat(timespec="seconds")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("exp", "objective", "params", "pseudo_huber_c")}), flush=True)
    log = open(out / "training_log.csv", "w", newline="")
    w = csv.DictWriter(log, fieldnames=["step", "loss", "sec", "peak_GiB"]); w.writeheader()
    step, perm, pos, t0, acc = 0, torch.randperm(len(X), generator=g), 0, time.time(), []
    torch.cuda.reset_peak_memory_stats()
    while step < args.max_steps:
        if pos + args.batch_size > len(perm):
            perm, pos = torch.randperm(len(X), generator=g), 0
        b = perm[pos:pos + args.batch_size].to(dev); pos += args.batch_size
        ppg, x1 = X[b].unsqueeze(1), Y[b].unsqueeze(1)
        z0 = torch.randn(x1.shape, generator=g).to(dev)
        n = torch.randint(0, args.n_disc - 1, (len(b),), generator=g).to(dev)
        t_n, t_n1 = grid[n].reshape(-1, 1), grid[n + 1].reshape(-1, 1)
        x_n = (1 - t_n).reshape(-1, 1, 1) * z0 + t_n.reshape(-1, 1, 1) * x1
        with torch.no_grad():                          # one teacher Heun step t_n -> t_{n+1}
            dt = (t_n1 - t_n).reshape(-1, 1, 1)
            d1 = teacher.forward_step(x_n, ppg, t_n)
            x_e = x_n + dt * d1
            x_hat = x_n + dt / 2 * (d1 + teacher.forward_step(x_e, ppg, t_n1))
            y_tgt = f_consistency(target, x_hat, ppg, t_n1)
        y = f_consistency(student, x_n, ppg, t_n)
        loss = (torch.sqrt(((y - y_tgt) ** 2).flatten(1).sum(1) + c * c) - c).mean()
        if not torch.isfinite(loss):
            (out / "TRAINING_FAILED").write_text(f"non-finite loss at step {step}\n"); raise SystemExit(3)
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        with torch.no_grad():
            for pt, ps in zip(target.parameters(), student.parameters()):
                pt.mul_(args.ema).add_(ps.detach(), alpha=1 - args.ema)
        acc.append(loss.item())
        if step % args.log_every == 0 or step == args.max_steps:
            row = {"step": step, "loss": float(np.mean(acc)), "sec": round(time.time() - t0, 1),
                   "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            w.writerow(row); log.flush(); print(row, flush=True); acc = []
    torch.save({"state_dict": student.state_dict(), "model_cfg": model_cfg, "args": vars(args), "seed": args.seed,
                "objective": "consistency_distillation", "train_state": {"opt_steps": step}, "epoch": -1,
                "upstream_commit": UPSTREAM_COMMIT, "git": meta["git"]}, out / "checkpoint_last.pt")
    summary = {"exp": out.name, "opt_steps": step, "total_train_time_s": round(time.time() - t0, 1), "lr": args.lr,
               "finished": datetime.now().isoformat(timespec="seconds")}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=1))
    (out / "TRAINING_DONE").write_text(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
