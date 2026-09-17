"""VM1 trainer: official improved MeanFlow (DiT, auxiliary v head, in-model CFG) on PPG -> ECG.

Data, split, seed, batch 64 and the 14,000-step budget are those of every VitalDB arm. Optimiser = the official
recipe (AdamW, weight decay 0, betas (0.9, 0.95), lr 1e-4, linear warmup then constant). No EMA and no checkpoint
selection: checkpoint_last.pt at exactly 14,000 steps is the evaluated model (docs/VM1_VANILLA_IMF_PREREGISTRATION.md).
Run: .venv/bin/python -m ppg2ecg.training.train_vimf --arch S --init scratch --out-dir outputs/vm1_S_seed42
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.flow.imf_vanilla import vimf_loss
from ppg2ecg.models.imf_dit import ARCHS, ImfDiT1d, count_params, load_official_weights
from ppg2ecg.training.train_a0 import git_sha, load_arrays
from ppg2ecg.utils.seed import seed_everything


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--init", choices=["scratch", "official"], default="scratch")
    ap.add_argument("--pretrained", default="data/pretrained/imf/iMF-B-2.pth")
    ap.add_argument("--processed", default="data/processed/v1_vitaldb")
    ap.add_argument("--manifest", default="data/manifests/split_v1_vitaldb_seed42.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=14000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=220)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=220)
    ap.add_argument("--val-subsample", type=int, default=1024)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, deterministic=False)
    dev = torch.device("cuda")
    split = read_manifest(root / args.manifest)[0]
    x_tr, y_tr, _ = load_arrays(root / args.processed, split["train"])
    x_va, y_va, _ = load_arrays(root / args.processed, split["val"])
    idx = np.unique(np.linspace(0, len(x_va) - 1, args.val_subsample).round().astype(int))
    x_va, y_va = x_va[idx], y_va[idx]
    T = x_tr.shape[1]
    X, Y = torch.from_numpy(x_tr).to(dev), torch.from_numpy(y_tr).to(dev)
    XV, YV = torch.from_numpy(x_va).to(dev)[:, None], torch.from_numpy(y_va).to(dev)[:, None]
    model_cfg = dict(seq_len=T, patch=args.patch, **ARCHS[args.arch])
    net = ImfDiT1d(**model_cfg).to(dev)
    init_report = load_official_weights(net, str(root / args.pretrained)) if args.init == "official" else {}
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps))
    shuffle = torch.Generator().manual_seed(args.seed)
    draws = torch.Generator().manual_seed(args.seed + 1)
    meta = {"exp": out.name, "objective": "imf_vanilla_official", "model_cfg": model_cfg, "args": vars(args),
            "params_train": count_params(net), "params_inference": count_params(net, inference=True),
            "init_report": init_report, "n_train": int(len(x_tr)), "n_val_monitor": int(len(x_va)),
            "git": git_sha(root), "torch": torch.__version__, "started": datetime.now().isoformat(timespec="seconds")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("exp", "model_cfg", "params_train", "params_inference")}), flush=True)
    if init_report:
        print("init:", {k: v for k, v in init_report.items() if not isinstance(v, list)}, flush=True)
    log = open(out / "training_log.csv", "w", newline="")
    writer = csv.DictWriter(log, fieldnames=["step", "loss", "loss_u", "loss_v", "val_loss_u", "val_loss_v", "lr", "sec", "peak_GiB"])
    writer.writeheader()
    step, perm, pos, t0 = 0, torch.randperm(len(X), generator=shuffle), 0, time.time()
    acc = {"loss": [], "loss_u": [], "loss_v": []}
    torch.cuda.reset_peak_memory_stats()
    while step < args.max_steps:
        if pos + args.batch_size > len(perm):
            perm, pos = torch.randperm(len(X), generator=shuffle), 0
        b = perm[pos:pos + args.batch_size].to(dev); pos += args.batch_size
        net.train(); opt.zero_grad(set_to_none=True)
        for i in range(0, args.batch_size, args.micro_batch):
            bb = b[i:i + args.micro_batch]
            loss, info = vimf_loss(net, Y[bb][:, None], X[bb][:, None], draws)
            (loss * len(bb) / args.batch_size).backward()
            acc["loss"].append(float(loss)); acc["loss_u"].append(float(info["loss_u"])); acc["loss_v"].append(float(info["loss_v"]))
        opt.step(); sched.step(); step += 1
        if not np.isfinite(acc["loss"][-1]):
            (out / "TRAINING_FAILED").write_text(f"non-finite loss at step {step}\n"); raise SystemExit(1)
        if step % args.log_every == 0 or step == args.max_steps:
            net.eval(); vu, vv = [], []
            g = torch.Generator().manual_seed(1000)
            for i in range(0, len(XV), 64):
                _, inf = vimf_loss(net, YV[i:i + 64], XV[i:i + 64], g)
                vu.append(float(inf["loss_u"])); vv.append(float(inf["loss_v"]))
            row = {"step": step, "loss": np.mean(acc["loss"]), "loss_u": np.mean(acc["loss_u"]), "loss_v": np.mean(acc["loss_v"]),
                   "val_loss_u": np.mean(vu), "val_loss_v": np.mean(vv), "lr": sched.get_last_lr()[0],
                   "sec": round(time.time() - t0, 1), "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            writer.writerow(row); log.flush()
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
            acc = {"loss": [], "loss_u": [], "loss_v": []}
    torch.save({"state_dict": net.state_dict(), "model_cfg": model_cfg, "args": vars(args), "objective": "imf_vanilla_official",
                "train_state": {"opt_steps": step}, "init_report": init_report, "seed": args.seed}, out / "checkpoint_last.pt")
    summary = {"exp": out.name, "opt_steps": step, "total_train_time_s": round(time.time() - t0, 1),
               "peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2), "finished": datetime.now().isoformat(timespec="seconds")}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=1))
    (out / "TRAINING_DONE").write_text(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
