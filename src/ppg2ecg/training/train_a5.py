"""Arm A5: deterministic MSE regression control (S5ConditionalMeanRegressor) — same data/split/optimiser as A0-b/A2.
Loss = MSE(f(PPG), ECG). Checkpoint selection = deterministic validation MSE (full validation set or the pre-registered subset),
min_delta 1e-4, patience 20, max 300 epochs/rounds; --val-every-steps for the WildPPG round rule (A4 parity)."""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401  (MKL warm-up must precede `import torch`, docs/ENVIRONMENT.md)
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.data.target_norm import TargetNorm
from ppg2ecg.models.regressor import REGRESSOR_MODELS
from ppg2ecg.training.train_a0 import batch_rounds, git_sha, load_arrays
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import UPSTREAM_COMMIT, assert_upstream_pinned

LOG_FIELDS = ["epoch", "train_mse", "val_mse", "selection_metric", "diag_hr_abs_err", "diag_morph_corr", "diag_amp_ratio", "diag_beats_ratio", "lr", "epoch_time_s", "elapsed_s", "peak_mem_MiB", "is_best", "best_epoch", "no_improve", "event"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--processed", default="data/processed/v0_8s")
    ap.add_argument("--manifest", default="data/manifests/split_p0_holdout_seed42.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--h-dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--ssm-ratio", type=float, default=2.0)
    ap.add_argument("--mlp-ratio", type=float, default=2.0)
    ap.add_argument("--sample-rate", type=int, default=128)
    ap.add_argument("--val-every-steps", type=int, default=None)
    ap.add_argument("--val-subsample", type=int, default=None)
    ap.add_argument("--gen-diag-every", type=int, default=1)
    ap.add_argument("--gen-diag-windows", type=int, default=128)
    ap.add_argument("--model", choices=list(REGRESSOR_MODELS), default="state_token", help="state_token = A5 regressor; full_backbone = A6 capacity-matched control")
    ap.add_argument("--x-const", default=None, help="full_backbone only: constant state input (float) or fixed_normal:<seed>")
    ap.add_argument("--t-const", type=float, default=None, help="full_backbone only: fixed auxiliary time constant")
    ap.add_argument("--cond-scale", type=float, default=1.0, help="full_backbone only (hard-test diagnostic): cond = cond_scale * E(t_const)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--target-norm", default=None, help="A8: path to normalization.json (global train-only affine applied to the TARGET only)")
    ap.add_argument("--limit-windows", type=int, default=None)
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
    tnorm = TargetNorm.load(args.target_norm) if args.target_norm else TargetNorm.identity()
    if not tnorm.is_identity:  # A8: TARGET ONLY; the PPG inputs are untouched (bit-exact with the raw-scale run)
        y_tr, y_va = tnorm.forward(y_tr), tnorm.forward(y_va)
        print(f"target normalisation: y_norm = (y - {tnorm.mu:.6f}) / {tnorm.sigma:.6f}  [{tnorm.source}]", flush=True)
    if args.val_subsample and len(x_va) > args.val_subsample:
        stride = -(-len(x_va) // args.val_subsample)
        x_va, y_va = x_va[::stride], y_va[::stride]
    T = x_tr.shape[1]
    x_tr_t, y_tr_t = torch.from_numpy(x_tr).to(device), torch.from_numpy(y_tr).to(device)
    x_va_t, y_va_t = torch.from_numpy(x_va).to(device), torch.from_numpy(y_va).to(device)
    model_cls, count_fn = REGRESSOR_MODELS[args.model]
    model_cfg = dict(sample_rate=args.sample_rate, h_dim=args.h_dim, ssm_block_num=args.blocks, ssm_ratio=args.ssm_ratio, mlp_ratio=args.mlp_ratio)
    if args.model == "full_backbone":
        xc = args.x_const
        model_cfg.update(x_const=(float(xc) if xc is not None and ":" not in xc else xc), t_const=args.t_const, cond_scale=args.cond_scale)
    model = model_cls(**model_cfg).to(device)
    if args.model == "full_backbone":
        model_cfg.update(x_const=model.x_const, t_const=model.t_const)  # resolved defaults, saved with the checkpoint
    params = count_fn(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=args.batch_size, shuffle=True, generator=gen)
    state = {"epoch": 0, "best": float("inf"), "best_epoch": -1, "no_improve": 0, "elapsed": 0.0, "peak_mem": 0.0}
    last_ckpt, best_ckpt, log_path = out / "checkpoint_last.pt", out / "checkpoint_best.pt", out / "training_log.csv"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["loader_generator"].cpu())
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
        state = ck["train_state"]
    else:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
    meta = {"exp_name": args.exp_name, "target_norm": {"mu": tnorm.mu, "sigma": tnorm.sigma, "source": tnorm.source}, "objective": "mse_regression", "model": model_cls.__name__, "model_cfg": model_cfg, "target_norm": {"mu": tnorm.mu, "sigma": tnorm.sigma, "source": tnorm.source}, "args": vars(args), "params": params, "n_train_windows": int(len(x_tr)), "n_val_windows": int(len(x_va)), "T": int(T), "split": split, "upstream": up, "git": git_sha(root), "device": str(device), "torch": torch.__version__, "selection": {"criterion": "val_mse (deterministic)", "min_delta": args.min_delta, "patience": args.patience, "val_subsample": args.val_subsample, "val_every_steps": args.val_every_steps}, "started": datetime.now().isoformat(timespec="seconds")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("exp_name", "model", "params", "n_train_windows", "n_val_windows", "T")}), flush=True)

    @torch.no_grad()
    def predict(x):
        model.eval()
        return torch.cat([model(x[i : i + args.batch_size].unsqueeze(1)).squeeze(1) for i in range(0, len(x), args.batch_size)])

    def val_mse():
        return float(((predict(x_va_t) - y_va_t) ** 2).mean())

    def gen_diag():
        from ppg2ecg.evaluation.metrics import rhythm_morphology_metrics

        m = min(args.gen_diag_windows, len(x_va))
        pred, tgt = predict(x_va_t[:m]).cpu().numpy(), y_va[:m]
        rm = rhythm_morphology_metrics(pred, tgt, args.sample_rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmean(rm["hr_abs_err"])), float(np.nanmean(rm["morph_corr"])), float(np.mean(pred.std(axis=1) / (tgt.std(axis=1) + 1e-8))), float(rm["n_pred_beats"].mean() / max(rm["n_ref_beats"].mean(), 1e-9))

    try:
        round_iter = batch_rounds(train_loader, args.val_every_steps)
        for epoch in range(state["epoch"], args.epochs):
            t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model.train()
            losses = []
            for ppg, ecg in next(round_iter):
                pred = model(ppg.unsqueeze(1)).squeeze(1)
                loss = ((pred - ecg) ** 2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch}")
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            vm = val_mse()
            do_diag = args.gen_diag_every > 0 and ((epoch + 1) % args.gen_diag_every == 0 or epoch == 0)
            d_hr, d_morph, d_amp, d_beats = gen_diag() if do_diag else (float("nan"),) * 4
            ep_time = time.perf_counter() - t0
            state["elapsed"] += ep_time
            peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
            state["peak_mem"] = max(state["peak_mem"], peak)
            is_best = vm < state["best"] - args.min_delta
            event = ""
            if is_best:
                state.update(best=vm, best_epoch=epoch, no_improve=0)
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "objective": "mse_regression", "model": model_cls.__name__, "model_key": args.model, "val_mse": vm, "model_cfg": model_cfg, "target_norm": {"mu": tnorm.mu, "sigma": tnorm.sigma, "source": tnorm.source}, "args": vars(args), "seed": args.seed, "git": meta["git"], "upstream_commit": UPSTREAM_COMMIT}, best_ckpt)
                event = "best"
            else:
                state["no_improve"] += 1
            state["epoch"] = epoch + 1
            torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "loader_generator": gen.get_state(), "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [], "train_state": state, "epoch": epoch}, last_ckpt)
            stop = state["no_improve"] >= args.patience
            if stop:
                event = (event + ";" if event else "") + f"early_stop(patience={args.patience})"
            row = dict(epoch=epoch, train_mse=np.mean(losses), val_mse=vm, selection_metric=vm, diag_hr_abs_err=d_hr, diag_morph_corr=d_morph, diag_amp_ratio=d_amp, diag_beats_ratio=d_beats, lr=opt.param_groups[0]["lr"], epoch_time_s=ep_time, elapsed_s=state["elapsed"], peak_mem_MiB=peak, is_best=int(is_best), best_epoch=state["best_epoch"], no_improve=state["no_improve"], event=event)
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
            print(f"epoch {epoch+1:3d}/{args.epochs} trainMSE {row['train_mse']:.4f} valMSE {vm:.5f} diag(HR {d_hr:.1f} morph {d_morph:.3f} amp {d_amp:.2f} beats {d_beats:.2f}) {ep_time:.0f}s peak {peak:.0f}MiB best@{state['best_epoch']+1} {event}", flush=True)
            if stop:
                break
        summary = {"exp_name": args.exp_name, "objective": "mse_regression", "epochs_run": state["epoch"], "best_epoch": state["best_epoch"], "selection_criterion": "val_mse", "best_selection_metric": state["best"], "early_stopped": state["no_improve"] >= args.patience, "total_train_time_s": state["elapsed"], "peak_mem_MiB": state["peak_mem"], "params": params, "finished": datetime.now().isoformat(timespec="seconds"), "checkpoint_best": str(best_ckpt)}
        (out / "training_summary.json").write_text(json.dumps(summary, indent=1))
        (out / "TRAINING_DONE").write_text(json.dumps(summary))
        print("TRAINING_DONE", json.dumps(summary), flush=True)
    except Exception:
        (out / "TRAINING_FAILED").write_text(traceback.format_exc())
        print("TRAINING_FAILED\n" + traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
