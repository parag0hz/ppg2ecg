"""Arm A0: PENGUIN (OT-CFM + Flow-SSM/S5, upstream class unmodified) on PPG-DaLiA with OUR deterministic subject split.

The loop mirrors external/PENGUIN/src/train.py step for step (AdamW lr 1e-3 wd 0.01, batch 64, <=300 epochs, early stopping
on validation waveform MAE of full n_step-Heun samples with patience 10, best-val checkpoint), with these documented differences:
  * subjects come from a manifest (upstream: glob order + random.sample)         -> reproducible split
  * val/test sampling under torch.no_grad()                                      -> identical numbers, less memory
  * DataLoader shuffling uses an explicit torch.Generator(seed)                  -> reproducible order
  * extra logging: window-weighted val MAE, val CFM loss (fixed noise), LR, time, peak memory, resume support
Run: .venv/bin/python -m ppg2ecg.training.train_a0 --out-dir outputs/<exp> --manifest ... --processed data/processed/v0_8s
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.flow.cfm import cfm_loss, cfm_targets
from ppg2ecg.models import build_penguin_backbone, count_params
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import UPSTREAM_COMMIT, assert_upstream_pinned

LOG_FIELDS = ["epoch", "train_loss", "train_mae_monitor", "val_mae_batchmean", "val_mae_window", "val_cfm_loss", "lr", "epoch_time_s", "elapsed_s", "peak_mem_MiB", "is_best", "best_epoch", "no_improve", "event"]


def load_arrays(processed: Path, subjects: list[str], limit: int | None = None):
    xs, ys, sid = [], [], []
    for s in subjects:
        d = np.load(processed / f"{s}.npz")
        x, y = d["x"], d["y"]
        if limit:
            x, y = x[:limit], y[:limit]
        xs.append(x)
        ys.append(y)
        sid += [s] * len(x)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), np.array(sid)


def git_sha(root: Path) -> dict:
    import subprocess

    def g(*a):
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()

    return {"commit": g("rev-parse", "HEAD"), "dirty_files": len([ln for ln in g("status", "--porcelain").splitlines() if ln.strip() and not ln.startswith("?? outputs")])}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", default="a0_penguin_otcfm_ppgdalia_8s_seed42")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--processed", default="data/processed/v0_8s")
    ap.add_argument("--manifest", default="data/manifests/split_p0_holdout_seed42.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--n-step", type=int, default=25)
    ap.add_argument("--h-dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--ssm-ratio", type=float, default=2.0)
    ap.add_argument("--mlp-ratio", type=float, default=2.0)
    ap.add_argument("--sample-rate", type=int, default=128)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit-windows", type=int, default=None, help="smoke: cap windows per subject")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    up = assert_upstream_pinned()
    split = read_manifest(root / args.manifest if not Path(args.manifest).is_absolute() else args.manifest)[0]
    processed = root / args.processed if not Path(args.processed).is_absolute() else Path(args.processed)

    x_tr, y_tr, _ = load_arrays(processed, split["train"], args.limit_windows)
    x_va, y_va, _ = load_arrays(processed, split["val"], args.limit_windows)
    T = x_tr.shape[1]
    assert T == args.sample_rate * 8 or args.limit_windows, f"expected 8 s windows ({args.sample_rate*8}), got T={T}"
    x_tr_t, y_tr_t = torch.from_numpy(x_tr).to(device), torch.from_numpy(y_tr).to(device)
    x_va_t, y_va_t = torch.from_numpy(x_va).to(device), torch.from_numpy(y_va).to(device)

    model = build_penguin_backbone(n_step=args.n_step, sample_rate=args.sample_rate, h_dim=args.h_dim, ssm_block_num=args.blocks, ssm_ratio=args.ssm_ratio, mlp_ratio=args.mlp_ratio).to(device)
    params = count_params(model, exclude_prefixes=("cross_attn", "revin"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(TensorDataset(x_va_t, y_va_t), batch_size=args.batch_size, shuffle=False)

    state = {"epoch": 0, "best": float("inf"), "best_epoch": -1, "no_improve": 0, "elapsed": 0.0, "peak_mem": 0.0}
    last_ckpt, best_ckpt = out / "checkpoint_last.pt", out / "checkpoint_best.pt"
    log_path = out / "training_log.csv"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)  # RNG states must stay CPU ByteTensors
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["loader_generator"].cpu())
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["rng_cuda"]])
        state = ck["train_state"]
        print(f"[resume] from epoch {state['epoch']} best {state['best']:.4f} @ {state['best_epoch']}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    meta = {"exp_name": args.exp_name, "args": vars(args), "params": params, "n_train_windows": int(len(x_tr)), "n_val_windows": int(len(x_va)), "T": int(T), "split": split, "upstream": up, "git": git_sha(root), "device": str(device), "torch": torch.__version__, "started": datetime.now().isoformat(timespec="seconds")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("exp_name", "params", "n_train_windows", "n_val_windows", "T")}))

    def run_val(epoch: int):
        model.eval()
        batch_maes, sum_abs, n_win, cfm = [], 0.0, 0, []
        gval = torch.Generator(device="cpu")
        gval.manual_seed(1234)  # fixed noise/t for the val CFM loss => comparable across epochs
        with torch.no_grad():
            for ppg, ecg in val_loader:
                pred = model(ppg)  # upstream .sample(): n_step Heun steps = 2*n_step NFE
                err = (pred - ecg).abs()
                batch_maes.append(err.mean().item())
                sum_abs += err.mean(dim=1).sum().item()
                n_win += len(ppg)
                t = torch.rand(len(ppg), 1, generator=gval).to(device)
                x0 = torch.randn(len(ppg), 1, ecg.shape[1], generator=gval).to(device)
                x_t, v_star, t, x0 = cfm_targets(ecg.unsqueeze(1), t, x0)
                cfm.append(cfm_loss(model.forward_step(x_t, ppg.unsqueeze(1), t), v_star).item())
        return float(np.mean(batch_maes)), sum_abs / n_win, float(np.mean(cfm))

    try:
        for epoch in range(state["epoch"], args.epochs):
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            model.train()
            losses, maes = [], []
            for ppg, ecg in train_loader:
                pred = model(ppg, target_signal=ecg)  # upstream train_flow: one-step Euler x1 estimate (monitor only)
                loss = model.optimize(pred, ecg, opt)  # MSE(v_pred, x1 - x0); AdamW step
                losses.append(loss.item())
                maes.append((pred - ecg).abs().mean().item())
            val_bm, val_win, val_cfm = run_val(epoch)
            ep_time = time.perf_counter() - t0
            state["elapsed"] += ep_time
            peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
            state["peak_mem"] = max(state["peak_mem"], peak)
            is_best = val_bm < state["best"]
            event = ""
            if is_best:
                state.update(best=val_bm, best_epoch=epoch, no_improve=0)
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_mae_batchmean": val_bm, "val_mae_window": val_win, "model_cfg": dict(n_step=args.n_step, sample_rate=args.sample_rate, h_dim=args.h_dim, ssm_block_num=args.blocks, ssm_ratio=args.ssm_ratio, mlp_ratio=args.mlp_ratio), "args": vars(args), "seed": args.seed, "git": meta["git"], "upstream_commit": UPSTREAM_COMMIT}, best_ckpt)
                event = "best"
            else:
                state["no_improve"] += 1
            state["epoch"] = epoch + 1
            torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "loader_generator": gen.get_state(), "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(), "train_state": state, "epoch": epoch}, last_ckpt)
            stop = state["no_improve"] >= args.patience
            if stop:
                event = (event + ";" if event else "") + f"early_stop(patience={args.patience})"
            row = dict(epoch=epoch, train_loss=np.mean(losses), train_mae_monitor=np.mean(maes), val_mae_batchmean=val_bm, val_mae_window=val_win, val_cfm_loss=val_cfm, lr=opt.param_groups[0]["lr"], epoch_time_s=ep_time, elapsed_s=state["elapsed"], peak_mem_MiB=peak, is_best=int(is_best), best_epoch=state["best_epoch"], no_improve=state["no_improve"], event=event)
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
            print(f"epoch {epoch+1:3d}/{args.epochs} loss {row['train_loss']:.4f} trainMAE(monitor) {row['train_mae_monitor']:.4f} valMAE {val_bm:.4f} (win {val_win:.4f}) valCFM {val_cfm:.4f} {ep_time:.0f}s peak {peak:.0f}MiB best@{state['best_epoch']+1} {event}", flush=True)
            if stop:
                break
        summary = {"exp_name": args.exp_name, "epochs_run": state["epoch"], "best_epoch": state["best_epoch"], "best_val_mae_batchmean": state["best"], "early_stopped": state["no_improve"] >= args.patience, "total_train_time_s": state["elapsed"], "peak_mem_MiB": state["peak_mem"], "finished": datetime.now().isoformat(timespec="seconds"), "checkpoint_best": str(best_ckpt)}
        (out / "training_summary.json").write_text(json.dumps(summary, indent=1))
        (out / "TRAINING_DONE").write_text(json.dumps(summary))
        print("TRAINING_DONE", json.dumps(summary), flush=True)
    except Exception:
        (out / "TRAINING_FAILED").write_text(traceback.format_exc())
        print("TRAINING_FAILED\n" + traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
