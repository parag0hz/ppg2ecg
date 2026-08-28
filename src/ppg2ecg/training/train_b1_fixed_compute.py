"""B1-v2 fixed-compute paired driver — the frozen A2 iMF trainer with (1) NO early-stop termination (diagnostic only),
(2) an optional progressive temporal-gap factor beta(h, s) on the per-sample adaptive-weighted loss (curriculum arm; the vanilla
arm uses the same driver with beta=None), (3) fraction/final checkpoints, per-round h-bin diagnostics, schedule-state logging and a
paired-randomness probe. Derived from Arm A2 (docs/A2_IMEANFLOW_PREREGISTRATION.md).

Mirrors train_a0.py (data, split, seed, optimiser, batch, logging, resume); only the objective/parameterisation differs:
  * model  : MeanFlowS5(upstream PENGUIN backbone)  — u_theta(z, ppg, t, h)
  * loss   : imeanflow_loss (V = u + (t-r) sg(du/dt), adaptive-weighted v-loss), (t, r) ~ sample_tr, e ~ N(0, I)
  * select : deterministic fixed-bank iMF validation MSE (4 banks), min_delta, patience — as A0-b
  * diag   : 1-NFE generation on the first N validation windows with fixed e every K epochs (never used for selection)
Run: .venv/bin/python -m ppg2ecg.training.train_a2 --out-dir outputs/<exp> [--resume]
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
from ppg2ecg.flow.imeanflow import MeanFlowS5, fixed_imf_mse, imf_bank_hash, make_imf_banks, sample_meanflow, sample_tr
from ppg2ecg.flow.imeanflow_curriculum import LAMBDA_DEFAULT, curriculum_beta, imeanflow_loss_b1, progress_s
from ppg2ecg.models import build_penguin_backbone, count_params
from ppg2ecg.training.train_a0 import git_sha, load_arrays
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import UPSTREAM_COMMIT, assert_upstream_pinned

WSTAT_KEYS = ("w_mean", "w_median", "w_p01", "w_p10", "w_p25", "w_p75", "w_p90", "w_p99", "w_min", "w_max", "w_std", "w_saturation_frac", "w_near_lower_frac", "delta2_mean")
LOG_FIELDS = ["epoch", "global_step", "schedule_s", "beta_mean_train", "train_loss_weighted", "train_mse", "train_u_abs", "train_dudt_abs", "w_mean", "w_median", "w_p01", "w_p10", "w_p25", "w_p75", "w_p90", "w_p99", "w_min", "w_max", "w_std", "w_saturation_frac", "w_near_lower_frac", "delta2_mean", "val_imf_mse_fixed", "selection_metric", "diag_hr_abs_err", "diag_morph_corr", "diag_amp_ratio", "diag_beats_ratio", "lr", "epoch_time_s", "elapsed_s", "peak_mem_MiB", "is_best", "best_epoch", "no_improve", "event"]


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
    ap.add_argument("--exp-name", default="a2_imeanflow_s5_ppgdalia_8s_seed42")
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
    # iMF objective (official defaults; docs/IMEANFLOW_AUDIT.md)
    ap.add_argument("--p-mean", type=float, default=-0.4)
    ap.add_argument("--p-std", type=float, default=1.0)
    ap.add_argument("--data-proportion", type=float, default=0.5)
    ap.add_argument("--norm-p", type=float, default=1.0)
    ap.add_argument("--norm-eps", type=float, default=0.01)
    ap.add_argument("--jvp-mode", choices=["forward", "double_vjp"], default="forward")
    ap.add_argument("--cond-mode", choices=["h_only", "t_plus_h"], default="h_only", help="time conditioning: h_only = official iMF design (E(h)); t_plus_h = E(t)+E(h_scale*h)")
    ap.add_argument("--h-scale", type=float, default=1.0, help="scale applied to h before the shared sinusoidal embedder (1 = official; 1000 diverged)")
    ap.add_argument("--micro-batch", type=int, default=32, help="gradient-accumulation chunk; effective batch stays --batch-size (forward-mode JVP needs ~0.51 GiB/sample at T=1024)")
    ap.add_argument("--val-batch", type=int, default=32, help="batch for the fixed-bank validation metric (implementation detail, no effect on values)")
    ap.add_argument("--n-val-banks", type=int, default=4)
    ap.add_argument("--bank-seed", type=int, default=1000)
    ap.add_argument("--gen-diag-every", type=int, default=1)
    ap.add_argument("--gen-diag-windows", type=int, default=128)
    ap.add_argument("--val-every-steps", type=int, default=None, help="validation round = min(epoch, N optimizer steps); default: one epoch (A0-b/A2)")
    ap.add_argument("--val-subsample", type=int, default=None, help="deterministic uniform stride subsample of the validation windows to at most N (A4 rule)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--target-norm", default=None, help="A8: path to normalization.json (global train-only affine applied to the TARGET only)")
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--arm", choices=["vanilla", "curriculum"], required=True)
    ap.add_argument("--t-schedule", type=int, required=True, help="frozen budget in optimizer steps; also the curriculum horizon (T_schedule = T_train)")
    ap.add_argument("--curriculum-lambda", type=float, default=LAMBDA_DEFAULT)
    ap.add_argument("--frac-checkpoints", default="0,0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--probe-batches", type=int, default=64, help="initial micro-batches hashed for the paired-randomness audit")
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
    tr_kw = dict(p_mean=args.p_mean, p_std=args.p_std, data_proportion=args.data_proportion)
    banks = make_imf_banks(len(x_va), T, args.n_val_banks, args.bank_seed, **tr_kw)
    banks_hash = imf_bank_hash(banks)

    backbone = build_penguin_backbone(n_step=1, sample_rate=args.sample_rate, h_dim=args.h_dim, ssm_block_num=args.blocks, ssm_ratio=args.ssm_ratio, mlp_ratio=args.mlp_ratio)
    net = MeanFlowS5(backbone, cond_mode=args.cond_mode, h_scale=args.h_scale).to(device)
    params = count_params(backbone, exclude_prefixes=("cross_attn", "revin"))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    tr_gen = torch.Generator()
    tr_gen.manual_seed(args.seed + 1)  # (t, r) sampling stream (CPU), independent of the shuffle stream
    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=args.batch_size, shuffle=True, generator=gen)

    state = {"epoch": 0, "best": float("inf"), "best_epoch": -1, "no_improve": 0, "elapsed": 0.0, "peak_mem": 0.0, "global_step": 0, "historical_early_stop_round": None, "probe_hash": None, "probe_batches_done": 0}
    import hashlib as _hashlib

    probe = _hashlib.sha256()
    frac_targets = sorted({max(0, round(float(f) * args.t_schedule)) for f in args.frac_checkpoints.split(",")})
    BINS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0001)]

    def small_ckpt(kind):
        return {"state_dict": net.state_dict(), "checkpoint_kind": kind, "global_step": state["global_step"], "schedule_s": progress_s(state["global_step"], args.t_schedule), "arm": args.arm, "objective": "improved_meanflow", "model_cfg": dict(n_step=1, sample_rate=args.sample_rate, h_dim=args.h_dim, ssm_block_num=args.blocks, ssm_ratio=args.ssm_ratio, mlp_ratio=args.mlp_ratio), "target_norm": {"mu": tnorm.mu, "sigma": tnorm.sigma, "source": tnorm.source}, "imf_cfg": dict(tr_kw, norm_p=args.norm_p, norm_eps=args.norm_eps, jvp_mode=args.jvp_mode, cond_mode=args.cond_mode, h_scale=args.h_scale), "curriculum": {"arm": args.arm, "lambda": args.curriculum_lambda, "t_schedule": args.t_schedule}, "seed": args.seed}
    last_ckpt, best_ckpt, log_path = out / "checkpoint_last.pt", out / "checkpoint_best.pt", out / "training_log.csv"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["loader_generator"].cpu())
        tr_gen.set_state(ck["tr_generator"].cpu())
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
        state = ck["train_state"]
        print(f"[resume] from epoch {state['epoch']} best {state['best']:.5f} @ {state['best_epoch']}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    meta = {"exp_name": args.exp_name, "b1_arm": args.arm, "t_schedule": args.t_schedule, "curriculum_lambda": (args.curriculum_lambda if args.arm == "curriculum" else None), "target_norm": {"mu": tnorm.mu, "sigma": tnorm.sigma, "source": tnorm.source}, "objective": "improved_meanflow", "args": vars(args), "params": params, "n_train_windows": int(len(x_tr)), "n_val_windows": int(len(x_va)), "T": int(T), "split": split, "upstream": up, "git": git_sha(root), "device": str(device), "torch": torch.__version__, "selection": {"criterion": "fixed_imf_mse", "min_delta": args.min_delta, "patience": args.patience, "n_val_banks": args.n_val_banks, "bank_seed": args.bank_seed, "bank_hash": banks_hash, "cond_mode": args.cond_mode, "h_scale": args.h_scale, "micro_batch": args.micro_batch, "effective_batch": args.batch_size, "val_batch": args.val_batch, "gen_diag_every": args.gen_diag_every, "gen_diag_windows": args.gen_diag_windows}, "started": datetime.now().isoformat(timespec="seconds")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("exp_name", "objective", "params", "n_train_windows", "n_val_windows", "T")}))

    def gen_diag():
        from ppg2ecg.evaluation.metrics import rhythm_morphology_metrics

        m = min(args.gen_diag_windows, len(x_va))
        net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, m, args.batch_size):
                ppg = x_va_t[i : min(i + args.batch_size, m)].unsqueeze(1)
                e = banks[0][2][i : min(i + args.batch_size, m)].to(device)
                x1, _ = sample_meanflow(net, ppg, e, n_steps=1)
                preds.append(x1.squeeze(1).cpu().numpy())
        pred, tgt = np.concatenate(preds), y_va[:m]
        rm = rhythm_morphology_metrics(pred, tgt, args.sample_rate)
        amp = float(np.mean(pred.std(axis=1) / (tgt.std(axis=1) + 1e-8)))
        beats = float(rm["n_pred_beats"].mean() / max(rm["n_ref_beats"].mean(), 1e-9))
        return float(np.nanmean(rm["hr_abs_err"])), float(np.nanmean(rm["morph_corr"])), amp, beats

    if frac_targets and frac_targets[0] == 0 and state["global_step"] == 0:
        frac_targets.pop(0)
        torch.save(small_ckpt("frac000"), out / "checkpoint_frac000.pt")
    try:
        round_iter = batch_rounds(train_loader, args.val_every_steps)
        for epoch in range(state["epoch"], args.epochs):  # "epoch" == validation round (== true epoch unless --val-every-steps)
            t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            net.train()
            lw, lm, ua, da, ws = [], [], [], [], []
            bin_acc = [dict() for _ in range(len(BINS) + 1)]
            beta_acc = []
            for ppg, ecg in next(round_iter):
                B = len(ppg)
                opt.zero_grad()
                acc = {"loss": 0.0, "mse": 0.0, "u": 0.0, "d": 0.0}
                for i0 in range(0, B, args.micro_batch):  # gradient accumulation: identical objective/optimiser step, lower peak memory
                    ppg_c, ecg_c = ppg[i0 : i0 + args.micro_batch], ecg[i0 : i0 + args.micro_batch]
                    Bc = len(ppg_c)
                    t, r, fm = sample_tr(Bc, tr_gen, **tr_kw)
                    t, r, fm = t.to(device), r.to(device), fm.to(device)
                    e = torch.randn(Bc, 1, ecg_c.shape[1], device=device)
                    if state["probe_batches_done"] < args.probe_batches:  # paired-randomness audit: identical streams in both arms
                        for tt_ in (ppg_c, ecg_c, t, r, fm, e):
                            probe.update(tt_.detach().cpu().numpy().tobytes())
                        state["probe_batches_done"] += 1
                        if state["probe_batches_done"] == args.probe_batches:
                            state["probe_hash"] = probe.hexdigest()
                            (out / "paired_randomness_probe.json").write_text(json.dumps({"arm": args.arm, "n_micro_batches": args.probe_batches, "sha256": state["probe_hash"], "seed": args.seed}))
                    s_now = progress_s(state["global_step"], args.t_schedule)
                    beta = curriculum_beta(t, r, fm, s_now, args.curriculum_lambda) if args.arm == "curriculum" else None
                    loss, info = imeanflow_loss_b1(net, ecg_c.unsqueeze(1), ppg_c.unsqueeze(1), e, t, r, fm, beta, norm_p=args.norm_p, norm_eps=args.norm_eps, jvp_mode=args.jvp_mode)
                    ps = info["per_sample"]
                    hb, fmb = ps["h"].cpu().numpy(), ps["fm_mask"].cpu().numpy()
                    psn = {k_: ps[k_].cpu().numpy() for k_ in ("beta", "w", "delta2", "u_norm", "dudt_norm", "hdudt_norm")}
                    for bi, (lo, hi) in enumerate(BINS):
                        m_ = (~fmb) & (hb >= lo) & (hb < hi)
                        if m_.any():
                            a_ = bin_acc[bi]
                            for key_, val_ in (("n", float(m_.sum())), ("h", float(hb[m_].sum())), ("beta", float(psn["beta"][m_].sum())), ("w", float(psn["w"][m_].sum())), ("d2", float(psn["delta2"][m_].sum())), ("un", float(psn["u_norm"][m_].sum())), ("dn", float(psn["dudt_norm"][m_].sum())), ("hdn", float(psn["hdudt_norm"][m_].sum()))):
                                a_[key_] = a_.get(key_, 0.0) + val_
                    if fmb.any():
                        a_ = bin_acc[len(BINS)]
                        for key_, val_ in (("n", float(fmb.sum())), ("beta", float(psn["beta"][fmb].sum())), ("w", float(psn["w"][fmb].sum())), ("d2", float(psn["delta2"][fmb].sum()))):
                            a_[key_] = a_.get(key_, 0.0) + val_
                    beta_acc.append(float(info["beta_mean"]))
                    if not (torch.isfinite(loss) and torch.isfinite(info["mse"]) and torch.isfinite(info["dudt_abs_mean"])):
                        raise RuntimeError(f"non-finite loss at epoch {epoch}: {loss.item()} info={ {k: float(v) for k, v in info.items() if not isinstance(v, dict)} }")
                    (loss * (Bc / B)).backward()  # mean over the full batch of 64
                    acc["loss"] += loss.item() * Bc / B
                    acc["mse"] += float(info["mse"]) * Bc / B
                    acc["u"] += float(info["u_abs_mean"]) * Bc / B
                    acc["d"] += float(info["dudt_abs_mean"]) * Bc / B
                    for _k in WSTAT_KEYS:  # A8 §11 diagnostics (no effect on the objective)
                        acc[_k] = acc.get(_k, 0.0) + float(info[_k]) * Bc / B
                opt.step()
                state["global_step"] += 1
                while frac_targets and state["global_step"] >= frac_targets[0]:
                    ft = frac_targets.pop(0)
                    torch.save(small_ckpt(f"frac{round(100 * ft / max(args.t_schedule, 1)):03d}"), out / f"checkpoint_frac{round(100 * ft / max(args.t_schedule, 1)):03d}.pt")
                lw.append(acc["loss"])
                lm.append(acc["mse"])
                ua.append(acc["u"])
                da.append(acc["d"])
                ws.append({_k: acc[_k] for _k in WSTAT_KEYS})
            net.eval()
            val_fixed = fixed_imf_mse(net, x_va_t, y_va_t, banks, args.val_batch, args.jvp_mode)[0]
            do_diag = args.gen_diag_every > 0 and ((epoch + 1) % args.gen_diag_every == 0 or epoch == 0)
            d_hr, d_morph, d_amp, d_beats = gen_diag() if do_diag else (float("nan"),) * 4
            sel = val_fixed
            ep_time = time.perf_counter() - t0
            state["elapsed"] += ep_time
            peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
            state["peak_mem"] = max(state["peak_mem"], peak)
            is_best = sel < state["best"] - args.min_delta
            event = ""
            if is_best:
                state.update(best=sel, best_epoch=epoch, no_improve=0)
                torch.save({"state_dict": net.state_dict(), "epoch": epoch, "objective": "improved_meanflow", "selection": {"criterion": "fixed_imf_mse", "value": sel, "min_delta": args.min_delta}, "model_cfg": dict(n_step=1, sample_rate=args.sample_rate, h_dim=args.h_dim, ssm_block_num=args.blocks, ssm_ratio=args.ssm_ratio, mlp_ratio=args.mlp_ratio), "target_norm": {"mu": tnorm.mu, "sigma": tnorm.sigma, "source": tnorm.source}, "imf_cfg": dict(tr_kw, norm_p=args.norm_p, norm_eps=args.norm_eps, jvp_mode=args.jvp_mode, cond_mode=args.cond_mode, h_scale=args.h_scale), "args": vars(args), "seed": args.seed, "git": meta["git"], "upstream_commit": UPSTREAM_COMMIT, "arm": args.arm, "checkpoint_kind": "best_validation", "global_step": state["global_step"], "schedule_s": progress_s(state["global_step"], args.t_schedule), "curriculum": {"arm": args.arm, "lambda": args.curriculum_lambda, "t_schedule": args.t_schedule}}, best_ckpt)
                event = "best"
            else:
                state["no_improve"] += 1
            state["epoch"] = epoch + 1
            torch.save({"state_dict": net.state_dict(), "optimizer": opt.state_dict(), "loader_generator": gen.get_state(), "tr_generator": tr_gen.get_state(), "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [], "train_state": state, "epoch": epoch}, last_ckpt)
            if state["no_improve"] >= args.patience and state["historical_early_stop_round"] is None:
                state["historical_early_stop_round"] = epoch + 1
                event = (event + ";" if event else "") + f"historical_early_stop(patience={args.patience})"
            stop = False  # B1-v2: fixed budget — early stopping is diagnostic only
            s_round = progress_s(state["global_step"], args.t_schedule)
            with open(out / "schedule_state.csv", "a", newline="") as f_:
                w_ = csv.writer(f_)
                if f_.tell() == 0:
                    w_.writerow(["round", "global_step", "frac_of_T", "s", "beta_h005", "beta_h025", "beta_h050", "beta_h075", "beta_h095", "is_best", "no_improve"])
                betas_ = [1 - s_round + args.curriculum_lambda * s_round * (1 - hh) for hh in (0.05, 0.25, 0.50, 0.75, 0.95)] if args.arm == "curriculum" else [1.0] * 5
                w_.writerow([epoch + 1, state["global_step"], round(state["global_step"] / args.t_schedule, 6), round(s_round, 6)] + [round(b_, 6) for b_ in betas_] + [int(is_best), state["no_improve"]])
            with open(out / "gap_bins_train.csv", "a", newline="") as f_:
                w_ = csv.writer(f_)
                if f_.tell() == 0:
                    w_.writerow(["round", "global_step", "s", "bin", "n", "h_mean", "beta_mean", "w_mean", "effective_w_mean", "delta2_mean", "u_norm_mean", "dudt_norm_mean", "hdudt_norm_mean"])
                for bi in range(len(BINS) + 1):
                    a_ = bin_acc[bi]
                    n_ = a_.get("n", 0.0)
                    if n_ == 0:
                        continue
                    name_ = f"[{BINS[bi][0]},{min(BINS[bi][1], 1.0)})" if bi < len(BINS) else "boundary(r=t)"
                    w_.writerow([epoch + 1, state["global_step"], round(s_round, 6), name_, int(n_), round(a_.get("h", 0.0) / n_, 6), round(a_.get("beta", 0.0) / n_, 6), round(a_.get("w", 0.0) / n_, 8), round((a_.get("beta", 0.0) / n_) * (a_.get("w", 0.0) / n_), 8), round(a_.get("d2", 0.0) / n_, 4), round(a_.get("un", 0.0) / n_, 4), round(a_.get("dn", 0.0) / n_, 4), round(a_.get("hdn", 0.0) / n_, 4)])
            wrow = {k: float(np.mean([w[k] for w in ws])) for k in WSTAT_KEYS} if ws else {k: float("nan") for k in WSTAT_KEYS}
            row = dict(epoch=epoch, global_step=state["global_step"], schedule_s=round(s_round, 6), beta_mean_train=float(np.mean(beta_acc)) if beta_acc else 1.0, train_loss_weighted=np.mean(lw), train_mse=np.mean(lm), train_u_abs=np.mean(ua), train_dudt_abs=np.mean(da), **wrow, val_imf_mse_fixed=val_fixed, selection_metric=sel, diag_hr_abs_err=d_hr, diag_morph_corr=d_morph, diag_amp_ratio=d_amp, diag_beats_ratio=d_beats, lr=opt.param_groups[0]["lr"], epoch_time_s=ep_time, elapsed_s=state["elapsed"], peak_mem_MiB=peak, is_best=int(is_best), best_epoch=state["best_epoch"], no_improve=state["no_improve"], event=event)
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
            print(f"epoch {epoch+1:3d}/{args.epochs} lossW {row['train_loss_weighted']:.4f} mse {row['train_mse']:.4f} |u| {row['train_u_abs']:.3f} |dudt| {row['train_dudt_abs']:.3f} valMSEfixed {val_fixed:.5f} w(med {row['w_median']:.2e} p10 {row['w_p10']:.2e} p90 {row['w_p90']:.2e} sat {row['w_saturation_frac']:.3f}) diag1NFE(HR {d_hr:.1f} morph {d_morph:.3f} amp {d_amp:.2f} beats {d_beats:.2f}) {ep_time:.0f}s peak {peak:.0f}MiB best@{state['best_epoch']+1} {event}", flush=True)
            if stop:
                break
        final_ckpt = out / "checkpoint_final.pt"
        fc = small_ckpt("final_fixed_budget")
        fc.update({"epoch": state["epoch"] - 1, "selection": {"criterion": "fixed_budget_final", "value": None}, "args": vars(args), "git": meta["git"], "upstream_commit": UPSTREAM_COMMIT})
        torch.save(fc, final_ckpt)
        summary = {"exp_name": args.exp_name, "arm": args.arm, "objective": "improved_meanflow", "epochs_run": state["epoch"], "best_epoch": state["best_epoch"], "selection_criterion": "fixed_imf_mse (diagnostic)", "best_selection_metric": state["best"], "historical_early_stop_round": state["historical_early_stop_round"], "total_optimizer_steps": state["global_step"], "t_schedule": args.t_schedule, "schedule_s_final": progress_s(state["global_step"], args.t_schedule), "probe_sha256": state["probe_hash"], "total_train_time_s": state["elapsed"], "peak_mem_MiB": state["peak_mem"], "finished": datetime.now().isoformat(timespec="seconds"), "checkpoint_best": str(best_ckpt), "checkpoint_final": str(final_ckpt)}
        (out / "training_summary.json").write_text(json.dumps(summary, indent=1))
        (out / "TRAINING_DONE").write_text(json.dumps(summary))
        print("TRAINING_DONE", json.dumps(summary), flush=True)
    except Exception:
        (out / "TRAINING_FAILED").write_text(traceback.format_exc())
        print("TRAINING_FAILED\n" + traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
