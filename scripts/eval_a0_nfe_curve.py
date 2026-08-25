"""Arm A1 evaluation on ONE trained checkpoint: sampling-step / NFE curve with paired noise, full metric suite,
conditional-fidelity (PPG-shuffle) and seed-diversity diagnostics, efficiency, deterministic example figures.

Solver steps vs NFE are always reported separately: Heun = 2 evaluations per step, Euler = 1.
Run: .venv/bin/python scripts/eval_a0_nfe_curve.py --out-dir outputs/<exp> [--checkpoint ...] [--limit-windows N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import ppg2ecg.utils.mkl_warmup  # noqa: F401  (MKL warm-up must precede `import torch`, docs/ENVIRONMENT.md)
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.data.splits import read_manifest  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from ppg2ecg.evaluation.efficiency import benchmark  # noqa: E402
from ppg2ecg.evaluation.metrics import evaluate_windows, hf_energy_ratio, penguin_hr_error, rhythm_morphology_metrics, summarize  # noqa: E402
from ppg2ecg.flow.samplers import SAMPLERS, nfe_of  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402
from ppg2ecg.utils.seed import seed_everything  # noqa: E402
from ppg2ecg.utils.upstream import assert_upstream_pinned  # noqa: E402

ROOT = Path(os.environ.get("PPG2ECG_ROOT") or Path(__file__).resolve().parents[1])
FS = 128
DEFAULT_ARMS = "heun:25,heun:10,heun:5,heun:2,heun:1,euler:1"
CSV_FIELDS = ["solver", "solver_steps", "actual_NFE", "hr_abs_err_bpm", "rpeak_f1", "rpeak_precision", "rpeak_recall", "rr_mae_ms", "mae", "rmse", "pcc", "qrs_width_err_ms", "morph_corr", "hr_err_penguin_corrected", "hr_err_penguin_as_shipped", "amp_ratio", "amp_ratio_median", "hf_ratio_pred", "hf_ratio_target", "cond_gain_bpm", "hr_err_shuffled_right_target", "hr_err_shuffled_wrong_target", "seed_std_mean", "seed_pairwise_corr", "latency_ms_batch64", "samples_per_s", "peak_mem_MiB", "n_windows", "frac_windows_no_pred_beats"]


def load_test(processed: Path, subjects, limit=None):
    xs, ys, ids, starts = [], [], [], []
    for s in subjects:
        d = np.load(processed / f"{s}.npz")
        n = len(d["x"]) if not limit else min(limit, len(d["x"]))
        xs.append(d["x"][:n])
        ys.append(d["y"][:n])
        ids += [s] * n
        starts.append(d["window_start_s"][:n])
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), np.array(ids), np.concatenate(starts)


@torch.no_grad()
def sample_all(model, ppg_np, x0_all, solver, steps, batch, device):
    fn = SAMPLERS[solver]
    out, nfe_seen = [], set()
    for i in range(0, len(ppg_np), batch):
        ppg = torch.from_numpy(ppg_np[i : i + batch]).to(device)
        x0 = x0_all[i : i + batch].to(device)
        v = lambda x, t: model.forward_step(x, ppg.unsqueeze(1), t)  # noqa: E731
        x1, nfe = fn(v, x0, steps)
        nfe_seen.add(nfe)
        out.append(x1.squeeze(1).float().cpu().numpy())
    assert len(nfe_seen) == 1 and nfe_seen.pop() == nfe_of(solver, steps)
    return np.concatenate(out)


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not np.any(p == np.arange(n)):
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--processed", default="data/processed/v0_8s")
    ap.add_argument("--manifest", default="data/manifests/split_p0_holdout_seed42.json")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=None, help="deterministic uniform stride subsample of the test windows to at most N (applied identically to all arms)")
    ap.add_argument("--diversity-seeds", type=int, default=4)
    ap.add_argument("--diversity-windows", type=int, default=256)
    ap.add_argument("--bench-repeats", type=int, default=10)
    ap.add_argument("--detector", default="neurokit")
    args = ap.parse_args()
    out = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out / "checkpoint_best.pt"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    seed_everything(args.noise_seed, deterministic=True)
    device = torch.device("cuda")
    up = assert_upstream_pinned()

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_penguin_backbone(**ck["model_cfg"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    split = read_manifest(ROOT / args.manifest)[0]
    x_te, y_te, sid, starts = load_test(ROOT / args.processed, split["test"], args.limit_windows)
    if args.subsample and len(x_te) > args.subsample:
        stride = -(-len(x_te) // args.subsample)
        x_te, y_te, sid, starts = x_te[::stride], y_te[::stride], sid[::stride], starts[::stride]
    n, T = x_te.shape
    np.savez_compressed(out / "predictions" / "test_inputs.npz", x=x_te, y=y_te, sid=sid, starts=starts)
    g = torch.Generator().manual_seed(args.noise_seed)
    x0_all = torch.randn(n, 1, T, generator=g)  # paired noise shared by every arm (CPU draw, like upstream)
    perm = derangement(n, args.noise_seed + 1)
    arms = [(a.split(":")[0], int(a.split(":")[1])) for a in args.arms.split(",")]
    xb = torch.from_numpy(x_te[: args.batch_size]).to(device)
    x0b = x0_all[: args.batch_size].to(device)

    rows, full = [], {"checkpoint": str(ckpt_path), "checkpoint_epoch": ck.get("epoch"), "checkpoint_val_mae": ck.get("val_mae_batchmean"), "test_subjects": split["test"], "n_windows": int(n), "T": int(T), "noise_seed": args.noise_seed, "upstream": up, "created": datetime.now().isoformat(timespec="seconds"), "arms": {}}
    hf_t = float(hf_energy_ratio(y_te).mean())
    preds = {}
    for solver, steps in arms:
        key = f"{solver}{steps}"
        nfe = nfe_of(solver, steps)
        pred = sample_all(model, x_te, x0_all, solver, steps, args.batch_size, device)
        preds[key] = pred
        ev = evaluate_windows(pred, y_te, fs=FS, hr_window_segments=1, detector=args.detector)
        summ = summarize(ev, n_boot=1000, seed=0)
        amp = pred.std(axis=1) / (y_te.std(axis=1) + 1e-8)
        hr_c = penguin_hr_error(pred, y_te, window_s=8, mode="corrected")
        hr_s = penguin_hr_error(pred, y_te, window_s=8, mode="as_shipped")
        # efficiency (fixed batch of 64, median of repeats)
        vb = lambda x, t: model.forward_step(x, xb.unsqueeze(1), t)  # noqa: E731
        eff = benchmark(lambda: SAMPLERS[solver](vb, x0b, steps), n_warmup=3, n_repeats=args.bench_repeats, batch_size=len(xb))
        # conditional fidelity: PPG-shuffle (derangement) with the same noise
        pred_sh = sample_all(model, x_te[perm], x0_all, solver, steps, args.batch_size, device)
        rm_right = rhythm_morphology_metrics(pred_sh, y_te[perm], FS, detector=args.detector)  # target that belongs to the given PPG
        rm_wrong = rhythm_morphology_metrics(pred_sh, y_te, FS, detector=args.detector)  # original (wrong) target
        hr_right, hr_wrong = float(np.nanmean(rm_right["hr_abs_err"])), float(np.nanmean(rm_wrong["hr_abs_err"]))
        # seed diversity on a subset
        m = min(args.diversity_windows, n)
        seeds_pred = [pred[:m]]
        for k in range(1, args.diversity_seeds):
            gk = torch.Generator().manual_seed(args.noise_seed + 100 + k)
            seeds_pred.append(sample_all(model, x_te[:m], torch.randn(m, 1, T, generator=gk), solver, steps, args.batch_size, device))
        sp = np.stack(seeds_pred)  # [S, m, T]
        seed_std = float(sp.std(axis=0).mean())
        corrs = [float(np.mean([np.corrcoef(sp[a, i], sp[b, i])[0, 1] for i in range(m)])) for a in range(len(sp)) for b in range(a + 1, len(sp))]
        np.savez_compressed(out / "predictions" / f"{key}.npz", pred=pred.astype(np.float32), pred_shuffled=pred_sh.astype(np.float32), perm=perm, solver=solver, steps=steps, nfe=nfe, **{f"pw_{k}": v for k, v in ev["signal"].items()}, **{f"pw_{k}": v for k, v in ev["rhythm"].items()})
        row = dict(solver=solver, solver_steps=steps, actual_NFE=nfe, hr_abs_err_bpm=summ["hr_abs_err"]["mean"], rpeak_f1=summ["rpeak_f1"]["mean"], rpeak_precision=summ["rpeak_precision"]["mean"], rpeak_recall=summ["rpeak_recall"]["mean"], rr_mae_ms=summ["rr_mae_ms"]["mean"], mae=summ["mae"]["mean"], rmse=summ["rmse"]["mean"], pcc=summ["pcc"]["mean"], qrs_width_err_ms=summ["qrs_width_err_ms"]["mean"], morph_corr=summ["morph_corr"]["mean"], hr_err_penguin_corrected=hr_c, hr_err_penguin_as_shipped=hr_s, amp_ratio=float(amp.mean()), amp_ratio_median=float(np.median(amp)), hf_ratio_pred=float(hf_energy_ratio(pred).mean()), hf_ratio_target=hf_t, cond_gain_bpm=hr_wrong - hr_right, hr_err_shuffled_right_target=hr_right, hr_err_shuffled_wrong_target=hr_wrong, seed_std_mean=seed_std, seed_pairwise_corr=float(np.mean(corrs)) if corrs else np.nan, latency_ms_batch64=eff["latency_ms_median"], samples_per_s=eff["samples_per_s"], peak_mem_MiB=eff.get("peak_mem_MiB"), n_windows=int(n), frac_windows_no_pred_beats=float(np.mean(ev["rhythm"]["n_pred_beats"] == 0)))
        rows.append(row)
        full["arms"][key] = {"row": row, "summary": summ, "efficiency": eff, "shuffle": {"hr_right": hr_right, "hr_wrong": hr_wrong, "morph_corr_right": float(np.nanmean(rm_right["morph_corr"])), "rpeak_f1_right": float(np.nanmean(rm_right["rpeak_f1"]))}, "diversity": {"n_seeds": len(sp), "n_windows": m, "std_mean": seed_std, "pairwise_corr": corrs}}
        print(f"{solver:5s} steps={steps:2d} NFE={nfe:2d} | HRerr {row['hr_abs_err_bpm']:.2f} amp {row['amp_ratio']:.2f} F1 {row['rpeak_f1']:.3f} RR {row['rr_mae_ms']:.1f}ms RMSE {row['rmse']:.3f} PCC {row['pcc']:.3f} QRS {row['qrs_width_err_ms']:.1f}ms morph {row['morph_corr']:.3f} | up-corr {hr_c:.2f} up-ship {hr_s:.2f} | gain {row['cond_gain_bpm']:.2f} seedstd {seed_std:.3f} | {eff['latency_ms_median']:.0f} ms", flush=True)

    with open(out / "nfe_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out / "metrics.json").write_text(json.dumps(full, indent=1, default=str))

    # ---- figures: deterministic example selection (fixed positions + HR-error quantiles of the reference arm)
    ref_key = f"{arms[0][0]}{arms[0][1]}"
    ref_err = np.load(out / "predictions" / f"{ref_key}.npz")["pw_hr_abs_err"]
    fixed = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    finite = np.where(np.isfinite(ref_err))[0]
    quant = [int(finite[np.argsort(ref_err[finite])[int(q * (len(finite) - 1))]]) for q in (0.1, 0.5, 0.9)] if len(finite) else []
    examples = {"fixed_positions": fixed, "ref_arm_hr_err_quantiles_10_50_90": quant}
    t_axis = np.arange(T) / FS
    for tag, idxs in (("fixed", fixed), ("quantile", quant)):
        for i in idxs:
            fig, axes = plt.subplots(len(arms) + 1, 1, figsize=(14, 2.0 * (len(arms) + 1)), sharex=True)
            gt = y_te[i]
            rp_gt = R.detect_rpeaks(gt, FS, args.detector)
            axes[0].plot(t_axis, gt, "k", lw=0.8)
            axes[0].plot(rp_gt / FS, gt[rp_gt], "r.", ms=6)
            axes[0].set_ylabel("GT ECG")
            axes[0].set_title(f"test window {i} ({sid[i]}, start {starts[i]} s) — R-peaks marked (same detector for all rows)")
            for ax, (solver, steps) in zip(axes[1:], arms):
                key = f"{solver}{steps}"
                p = preds[key][i]
                rp = R.detect_rpeaks(p, FS, args.detector)
                ax.plot(t_axis, gt, color="0.75", lw=0.6)
                ax.plot(t_axis, p, "b", lw=0.8)
                ax.plot(rp / FS, p[rp], "r.", ms=6)
                ax.set_ylabel(f"{solver} {steps} st\n{nfe_of(solver, steps)} NFE")
            axes[-1].set_xlabel("time (s)")
            fig.tight_layout()
            fig.savefig(out / "figures" / f"example_{tag}_{i:05d}.png", dpi=110)
            plt.close(fig)
    # NFE curve plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (k, lab) in zip(axes.ravel(), [("hr_abs_err_bpm", "HR abs err (bpm)"), ("rpeak_f1", "R-peak F1"), ("rr_mae_ms", "RR MAE (ms)"), ("qrs_width_err_ms", "QRS width err (ms)"), ("morph_corr", "beat morph corr"), ("rmse", "RMSE"), ("pcc", "PCC"), ("latency_ms_batch64", "latency ms / batch 64")]):
        for solver, mk in (("heun", "o-"), ("euler", "s--")):
            rr = [r for r in rows if r["solver"] == solver]
            if rr:
                ax.plot([r["actual_NFE"] for r in rr], [r[k] for r in rr], mk, label=solver)
        ax.set_xscale("log")
        ax.set_xlabel("actual NFE")
        ax.set_title(lab)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "nfe_curve.png", dpi=110)
    plt.close(fig)
    full["examples"] = examples
    (out / "metrics.json").write_text(json.dumps(full, indent=1, default=str))
    print("wrote", out / "nfe_curve.csv", out / "metrics.json", "figures:", len(list((out / "figures").glob("*.png"))))


if __name__ == "__main__":
    main()
