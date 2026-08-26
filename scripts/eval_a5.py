"""A5 regressor evaluation on the same test windows / subset as the paired generative arms; same metrics, same shuffle derangement.
Writes nfe_curve.csv (single row, solver 'regressor'), metrics.json, predictions/regressor.npz, predictions/test_inputs.npz."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401  (MKL warm-up must precede `import torch`, docs/ENVIRONMENT.md)
import numpy as np
import torch

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.evaluation.efficiency import benchmark
from ppg2ecg.evaluation.metrics import evaluate_windows, hf_energy_ratio, penguin_hr_error, rhythm_morphology_metrics, summarize
from ppg2ecg.models.regressor import REGRESSOR_MODELS
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import assert_upstream_pinned

ROOT = Path(__file__).resolve().parents[1]
FS = 128
CSV_FIELDS = ["solver", "solver_steps", "actual_NFE", "hr_abs_err_bpm", "rpeak_f1", "rpeak_precision", "rpeak_recall", "rr_mae_ms", "mae", "rmse", "pcc", "qrs_width_err_ms", "morph_corr", "hr_err_penguin_corrected", "hr_err_penguin_as_shipped", "amp_ratio", "amp_ratio_median", "hf_ratio_pred", "hf_ratio_target", "cond_gain_bpm", "hr_err_shuffled_right_target", "hr_err_shuffled_wrong_target", "seed_std_mean", "seed_pairwise_corr", "latency_ms_batch64", "samples_per_s", "peak_mem_MiB", "n_windows", "frac_windows_no_pred_beats", "beats_ratio", "params_total", "params_effective"]


def load_test(processed, subjects, limit=None):
    xs, ys, ids, starts = [], [], [], []
    for s in subjects:
        d = np.load(processed / f"{s}.npz")
        n = len(d["x"]) if not limit else min(limit, len(d["x"]))
        xs.append(d["x"][:n])
        ys.append(d["y"][:n])
        ids += [s] * n
        starts.append(d["window_start_s"][:n])
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), np.array(ids), np.concatenate(starts)


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not np.any(p == np.arange(n)):
            return p


@torch.no_grad()
def predict(model, x_np, batch, device):
    return np.concatenate([model(torch.from_numpy(x_np[i : i + batch]).to(device).unsqueeze(1)).squeeze(1).float().cpu().numpy() for i in range(0, len(x_np), batch)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--processed", default="data/processed/v0_8s")
    ap.add_argument("--manifest", default="data/manifests/split_p0_holdout_seed42.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--noise-seed", type=int, default=0, help="only fixes the shuffle derangement seed (+1) for parity with the generative arms")
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=None)
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
    model_cls, count_fn = REGRESSOR_MODELS[ck.get("model_key", "state_token")]
    model = model_cls(**ck["model_cfg"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    params = count_fn(model)
    split = read_manifest(ROOT / args.manifest)[0]
    x_te, y_te, sid, starts = load_test(ROOT / args.processed, split["test"], args.limit_windows)
    if args.subsample and len(x_te) > args.subsample:
        stride = -(-len(x_te) // args.subsample)
        x_te, y_te, sid, starts = x_te[::stride], y_te[::stride], sid[::stride], starts[::stride]
    n, T = x_te.shape
    np.savez_compressed(out / "predictions" / "test_inputs.npz", x=x_te, y=y_te, sid=sid, starts=starts)
    perm = derangement(n, args.noise_seed + 1)
    pred = predict(model, x_te, args.batch_size, device)
    ev = evaluate_windows(pred, y_te, fs=FS, hr_window_segments=1, detector=args.detector)
    summ = summarize(ev, n_boot=1000, seed=0)
    amp = pred.std(axis=1) / (y_te.std(axis=1) + 1e-8)
    hr_c, hr_s = penguin_hr_error(pred, y_te, window_s=8, mode="corrected"), penguin_hr_error(pred, y_te, window_s=8, mode="as_shipped")
    xb = torch.from_numpy(x_te[: args.batch_size]).to(device).unsqueeze(1)
    eff = benchmark(lambda: model(xb), n_warmup=3, n_repeats=args.bench_repeats, batch_size=len(xb))
    pred_sh = predict(model, x_te[perm], args.batch_size, device)
    rm_right, rm_wrong = rhythm_morphology_metrics(pred_sh, y_te[perm], FS, detector=args.detector), rhythm_morphology_metrics(pred_sh, y_te, FS, detector=args.detector)
    hr_right, hr_wrong = float(np.nanmean(rm_right["hr_abs_err"])), float(np.nanmean(rm_wrong["hr_abs_err"]))
    np.savez_compressed(out / "predictions" / "regressor.npz", pred=pred.astype(np.float32), pred_shuffled=pred_sh.astype(np.float32), perm=perm, solver="regressor", steps=1, nfe=1, **{f"pw_{k}": v for k, v in ev["signal"].items()}, **{f"pw_{k}": v for k, v in ev["rhythm"].items()})
    row = dict(solver="regressor", solver_steps=1, actual_NFE=1, hr_abs_err_bpm=summ["hr_abs_err"]["mean"], rpeak_f1=summ["rpeak_f1"]["mean"], rpeak_precision=summ["rpeak_precision"]["mean"], rpeak_recall=summ["rpeak_recall"]["mean"], rr_mae_ms=summ["rr_mae_ms"]["mean"], mae=summ["mae"]["mean"], rmse=summ["rmse"]["mean"], pcc=summ["pcc"]["mean"], qrs_width_err_ms=summ["qrs_width_err_ms"]["mean"], morph_corr=summ["morph_corr"]["mean"], hr_err_penguin_corrected=hr_c, hr_err_penguin_as_shipped=hr_s, amp_ratio=float(amp.mean()), amp_ratio_median=float(np.median(amp)), hf_ratio_pred=float(hf_energy_ratio(pred).mean()), hf_ratio_target=float(hf_energy_ratio(y_te).mean()), cond_gain_bpm=hr_wrong - hr_right, hr_err_shuffled_right_target=hr_right, hr_err_shuffled_wrong_target=hr_wrong, seed_std_mean=0.0, seed_pairwise_corr=1.0, latency_ms_batch64=eff["latency_ms_median"], samples_per_s=eff["samples_per_s"], peak_mem_MiB=eff.get("peak_mem_MiB"), n_windows=int(n), frac_windows_no_pred_beats=float(np.mean(ev["rhythm"]["n_pred_beats"] == 0)), beats_ratio=float(ev["rhythm"]["n_pred_beats"].mean() / max(ev["rhythm"]["n_ref_beats"].mean(), 1e-9)), params_total=params["total"], params_effective=params["effective"])
    with open(out / "nfe_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerow(row)
    (out / "metrics.json").write_text(json.dumps({"checkpoint": str(ckpt_path), "checkpoint_epoch": ck.get("epoch"), "objective": "mse_regression", "model": model_cls.__name__, "params": params, "test_subjects": split["test"], "n_windows": int(n), "T": int(T), "upstream": up, "created": datetime.now().isoformat(timespec="seconds"), "arms": {"regressor": {"row": row, "summary": summ, "efficiency": eff, "shuffle": {"hr_right": hr_right, "hr_wrong": hr_wrong}}}}, indent=1, default=str))
    print(f"regressor 1 fwd | HRerr {row['hr_abs_err_bpm']:.2f} amp {row['amp_ratio']:.2f} beats {row['beats_ratio']:.2f} F1 {row['rpeak_f1']:.3f} RMSE {row['rmse']:.3f} MAE {row['mae']:.3f} morph {row['morph_corr']:.3f} HF {row['hf_ratio_pred']:.3f} | gain {row['cond_gain_bpm']:.2f} | {eff['latency_ms_median']:.0f} ms | params {params['total']}", flush=True)
    print("wrote", out / "nfe_curve.csv")


if __name__ == "__main__":
    main()
