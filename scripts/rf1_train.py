"""RF1 step 2: 2-rectified flow on the teacher pairs, PENGUIN backbone initialised from arm C."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ppg2ecg.models import build_penguin_backbone, count_params
from ppg2ecg.training.train_a0 import git_sha
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import UPSTREAM_COMMIT, assert_upstream_pinned

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/rf1_vitaldb_armR_seed42"))
    ap.add_argument("--pairs", default=str(ROOT / "outputs/rf1_pairs/pairs.npz"))
    ap.add_argument("--init", default=str(ROOT / "outputs/v1_vitaldb_armC_seed42/checkpoint_last.pt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=14000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--h-dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--ssm-ratio", type=float, default=2.0)
    ap.add_argument("--mlp-ratio", type=float, default=2.0)
    ap.add_argument("--sample-rate", type=int, default=128)
    ap.add_argument("--n-step", type=int, default=25)
    ap.add_argument("--log-every", type=int, default=220)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, deterministic=False)
    up = assert_upstream_pinned()
    dev = torch.device("cuda")
    d = np.load(args.pairs)
    ppg = torch.from_numpy(d["ppg"].astype(np.float32)).to(dev)
    z0 = torch.from_numpy(d["z0"].astype(np.float32)).to(dev)
    x1 = torch.from_numpy(d["x1"].astype(np.float32)).to(dev)
    model_cfg = dict(n_step=args.n_step, sample_rate=args.sample_rate, h_dim=args.h_dim,
                     ssm_block_num=args.blocks, ssm_ratio=args.ssm_ratio, mlp_ratio=args.mlp_ratio)
    model = build_penguin_backbone(**model_cfg).to(dev)
    init = torch.load(args.init, map_location="cpu", weights_only=False)
    model.load_state_dict(init["state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    g = torch.Generator().manual_seed(args.seed)
    meta = {"exp": out.name, "objective": "rectified_flow_reflow_2RF", "model_cfg": model_cfg, "args": vars(args),
            "params": count_params(model, exclude_prefixes=("cross_attn", "revin")), "n_pairs": int(len(ppg)),
            "teacher_seconds": float(d["teacher_seconds"]), "teacher_nfe": int(d["nfe"]), "pair_seed": int(d["pair_seed"]),
            "init_from": args.init, "git": git_sha(ROOT), "upstream": up, "started": datetime.now().isoformat(timespec="seconds")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("exp", "objective", "params", "n_pairs", "teacher_seconds")}), flush=True)
    log = open(out / "training_log.csv", "w", newline="")
    w = csv.DictWriter(log, fieldnames=["step", "loss", "sec", "peak_GiB"]); w.writeheader()
    step, perm, pos, t0, acc = 0, torch.randperm(len(ppg), generator=g), 0, time.time(), []
    torch.cuda.reset_peak_memory_stats()
    model.train()
    while step < args.max_steps:
        if pos + args.batch_size > len(perm):
            perm, pos = torch.randperm(len(ppg), generator=g), 0
        b = perm[pos:pos + args.batch_size].to(dev); pos += args.batch_size
        t = torch.rand(len(b), 1, 1, device=dev)
        p, a, c = ppg[b].unsqueeze(1), z0[b].unsqueeze(1), x1[b].unsqueeze(1)
        z_t = (1 - t) * a + t * c                     # upstream convention: t = 0 noise, t = 1 data
        v_pred = model.forward_step(z_t, p, t.reshape(-1, 1))
        loss = F.mse_loss(v_pred, c - a)              # the rectified-flow target on the TEACHER's own coupling
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        acc.append(loss.item())
        if step % args.log_every == 0 or step == args.max_steps:
            row = {"step": step, "loss": float(np.mean(acc)), "sec": round(time.time() - t0, 1),
                   "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            w.writerow(row); log.flush(); print(row, flush=True); acc = []
    torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg, "args": vars(args), "seed": args.seed,
                "objective": "rectified_flow_reflow_2RF", "train_state": {"opt_steps": step}, "epoch": -1,
                "upstream_commit": UPSTREAM_COMMIT, "git": meta["git"]}, out / "checkpoint_last.pt")
    summary = {"exp": out.name, "opt_steps": step, "total_train_time_s": round(time.time() - t0, 1),
               "teacher_seconds": float(d["teacher_seconds"]), "finished": datetime.now().isoformat(timespec="seconds")}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=1))
    (out / "TRAINING_DONE").write_text(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
