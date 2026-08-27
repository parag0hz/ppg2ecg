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
import ppg2ecg.utils.mkl_warmup  # noqa: F401  (MKL warm-up must precede `import torch`, docs/ENVIRONMENT.md)
import torch
from torch.utils.data import DataLoader, TensorDataset

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.data.target_norm import TargetNorm
from ppg2ecg.flow.cfm import cfm_loss, cfm_targets
from ppg2ecg.flow.samplers import heun_sample
from ppg2ecg.training.valbank import bank_hash, fixed_cfm_loss, make_banks
from ppg2ecg.models import build_penguin_backbone, count_params
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import UPSTREAM_COMMIT, assert_upstream_pinned

LOG_FIELDS = ["epoch", "train_loss", "train_mae_monitor", "val_mae_batchmean", "val_mae_window", "val_cfm_loss", "val_cfm_fixed", "selection_metric", "diag_hr_abs_err", "diag_morph_corr", "diag_amp_ratio", "lr", "epoch_time_s", "elapsed_s", "peak_mem_MiB", "is_best", "best_epoch", "no_improve", "event"]


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


def batch_rounds(loader, steps_per_round):
    """Yield one 'validation round' of batches at a time. steps_per_round=None: one round == one epoch (A0-b/A2 behaviour).
    Otherwise a round ends after steps_per_round optimizer steps OR at the end of the epoch, whichever comes first
    (A3/A4 pre-registration Part II §7: round = min(epoch, N steps)); the underlying epoch order/shuffle is unchanged."""
    if not steps_per_round:
        while True:
            yield iter(loader)
        return
    it = iter(loader)
    while True:
        chunk = []
        while len(chunk) < steps_per_round:
            try:
                chunk.append(next(it))
            except StopIteration:
                it = iter(loader)
                if chunk:
                    break
        yield chunk


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
    ap.add_argument("--val-every-steps", type=int, default=None, help="validation round = min(epoch, N optimizer steps); default: one epoch (A0-b/A2)")
    ap.add_argument("--val-subsample", type=int, default=None, help="deterministic uniform stride subsample of the validation windows to at most N (A4 rule)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--target-norm", default=None, help="A8: path to normalization.json (global train-only affine applied to the TARGET only)")
    ap.add_argument("--limit-windows", type=int, default=None, help="smoke: cap windows per subject")
    ap.add_argument("--select", choices=["val_mae", "fixed_cfm"], default="val_mae", help="checkpoint/early-stopping criterion (A0: val_mae; A0-b: fixed_cfm)")
    ap.add_argument("--min-delta", type=float, default=0.0, help="required improvement of the selection metric")
    ap.add_argument("--n-val-banks", type=int, default=4)
    ap.add_argument("--bank-seed", type=int, default=1000)
    ap.add_argument("--val-mae-every", type=int, default=1, help="epochs between stochastic 50-NFE val MAE passes (0 = never)")
    ap.add_argument("--gen-diag-every", type=int, default=0, help="epochs between fixed-noise generation diagnostics on a val subset (0 = never)")
    ap.add_argument("--gen-diag-windows", type=int, default=128)
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
    assert T == args.sample_rate * 8 or args.limit_windows, f"expected 8 s windows ({args.sample_rate*8}), got T={T}"
    x_tr_t, y_tr_t = torch.from_numpy(x_tr).to(device), torch.from_numpy(y_tr).to(device)
    x_va_t, y_va_t = torch.from_numpy(x_va).to(device), torch.from_numpy(y_va).to(device)

    banks = make_banks(len(x_va), T, args.n_val_banks, args.bank_seed) if args.n_val_banks > 0 else []
    banks_hash = bank_hash(banks) if banks else None

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

    meta = {"exp_name": args.exp_name, "target_norm": {"mu": tnorm.mu, "sigma": tnorm.sigma, "source": tnorm.source}, "args": vars(args), "params": params, "n_train_windows": int(len(x_tr)), "n_val_windows": int(len(x_va)), "T": int(T), "split": split, "upstream": up, "git": git_sha(root), "device": str(device), "selection": {"criterion": args.select, "min_delta": args.min_delta, "patience": args.patience, "n_val_banks": args.n_val_banks, "bank_seed": args.bank_seed, "bank_hash": banks_hash, "val_mae_every": args.val_mae_every, "gen_diag_every": args.gen_diag_every, "gen_diag_windows": args.gen_diag_windows}, "torch": torch.__version__, "started": datetime.now().isoformat(timespec="seconds")}
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

    def gen_diag():
        """50-NFE Heun generation on the first gen_diag_windows val windows with fixed noise (bank 0) -> HR err, morph corr, amplitude ratio."""
        from ppg2ecg.evaluation.metrics import rhythm_morphology_metrics

        m = min(args.gen_diag_windows, len(x_va))
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, m, args.batch_size):
                ppg = x_va_t[i : min(i + args.batch_size, m)]
                z = banks[0][1][i : min(i + args.batch_size, m)].to(device)
                x1, _ = heun_sample(lambda x, t: model.forward_step(x, ppg.unsqueeze(1), t), z, args.n_step)  # noqa: B023
                preds.append(x1.squeeze(1).cpu().numpy())
        pred = np.concatenate(preds)
        tgt = y_va[:m]
        rm = rhythm_morphology_metrics(pred, tgt, args.sample_rate)
        amp = float(np.mean(pred.std(axis=1) / (tgt.std(axis=1) + 1e-8)))
        return float(np.nanmean(rm["hr_abs_err"])), float(np.nanmean(rm["morph_corr"])), amp

    try:
        round_iter = batch_rounds(train_loader, args.val_every_steps)
        for epoch in range(state["epoch"], args.epochs):  # "epoch" == validation round (== true epoch unless --val-every-steps)
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            model.train()
            losses, maes = [], []
            for ppg, ecg in next(round_iter):
                pred = model(ppg, target_signal=ecg)  # upstream train_flow: one-step Euler x1 estimate (monitor only)
                loss = model.optimize(pred, ecg, opt)  # MSE(v_pred, x1 - x0); AdamW step
                losses.append(loss.item())
                maes.append((pred - ecg).abs().mean().item())
            do_val_mae = args.val_mae_every > 0 and (epoch + 1) % args.val_mae_every == 0
            val_bm, val_win, val_cfm = run_val(epoch) if do_val_mae else (float("nan"), float("nan"), float("nan"))
            val_fixed = fixed_cfm_loss(model.eval(), x_va_t, y_va_t, banks, args.batch_size)[0] if banks else float("nan")
            do_diag = args.gen_diag_every > 0 and ((epoch + 1) % args.gen_diag_every == 0 or epoch == 0)
            d_hr, d_morph, d_amp = gen_diag() if do_diag else (float("nan"),) * 3
            sel = val_bm if args.select == "val_mae" else val_fixed
            ep_time = time.perf_counter() - t0
            state["elapsed"] += ep_time
            peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
            state["peak_mem"] = max(state["peak_mem"], peak)
            is_best = sel < state["best"] - args.min_delta
            event = ""
            if is_best:
                state.update(best=sel, best_epoch=epoch, no_improve=0)
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "selection": {"criterion": args.select, "value": sel, "min_delta": args.min_delta}, "val_cfm_fixed": val_fixed, "val_mae_batchmean": val_bm, "val_mae_window": val_win, "model_cfg": dict(n_step=args.n_step, sample_rate=args.sample_rate, h_dim=args.h_dim, ssm_block_num=args.blocks, ssm_ratio=args.ssm_ratio, mlp_ratio=args.mlp_ratio), "target_norm": {"mu": tnorm.mu, "sigma": tnorm.sigma, "source": tnorm.source}, "args": vars(args), "seed": args.seed, "git": meta["git"], "upstream_commit": UPSTREAM_COMMIT}, best_ckpt)
                event = "best"
            else:
                state["no_improve"] += 1
            state["epoch"] = epoch + 1
            torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "loader_generator": gen.get_state(), "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(), "train_state": state, "epoch": epoch}, last_ckpt)
            stop = state["no_improve"] >= args.patience
            if stop:
                event = (event + ";" if event else "") + f"early_stop(patience={args.patience})"
            row = dict(epoch=epoch, train_loss=np.mean(losses), train_mae_monitor=np.mean(maes), val_mae_batchmean=val_bm, val_mae_window=val_win, val_cfm_loss=val_cfm, val_cfm_fixed=val_fixed, selection_metric=sel, diag_hr_abs_err=d_hr, diag_morph_corr=d_morph, diag_amp_ratio=d_amp, lr=opt.param_groups[0]["lr"], epoch_time_s=ep_time, elapsed_s=state["elapsed"], peak_mem_MiB=peak, is_best=int(is_best), best_epoch=state["best_epoch"], no_improve=state["no_improve"], event=event)
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
            print(f"epoch {epoch+1:3d}/{args.epochs} loss {row['train_loss']:.4f} trainMAE(monitor) {row['train_mae_monitor']:.4f} valMAE {val_bm:.4f} valCFMfixed {val_fixed:.5f} sel {sel:.5f} diag(HR {d_hr:.1f} morph {d_morph:.3f} amp {d_amp:.2f}) {ep_time:.0f}s peak {peak:.0f}MiB best@{state['best_epoch']+1} {event}", flush=True)
            if stop:
                break
        summary = {"exp_name": args.exp_name, "epochs_run": state["epoch"], "best_epoch": state["best_epoch"], "selection_criterion": args.select, "best_selection_metric": state["best"], "best_val_mae_batchmean": state["best"] if args.select == "val_mae" else None, "early_stopped": state["no_improve"] >= args.patience, "total_train_time_s": state["elapsed"], "peak_mem_MiB": state["peak_mem"], "finished": datetime.now().isoformat(timespec="seconds"), "checkpoint_best": str(best_ckpt)}
        (out / "training_summary.json").write_text(json.dumps(summary, indent=1))
        (out / "TRAINING_DONE").write_text(json.dumps(summary))
        print("TRAINING_DONE", json.dumps(summary), flush=True)
    except Exception:
        (out / "TRAINING_FAILED").write_text(traceback.format_exc())
        print("TRAINING_FAILED\n" + traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
