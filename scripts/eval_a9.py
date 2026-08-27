"""A9 evaluation: any of the three frozen objectives on the WildPPG test subset, in the trained (global-z) target space, with the
UNCHANGED frozen ECG metric code and the A4 protocol (same test windows, paired noise seed 0, derangement seed 1, ≤4096 subset).
Writes nfe_curve.csv (same schema as A4/A6), metrics.json and predictions/*.npz."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401
import numpy as np
import torch

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.data.target_norm import TargetNorm
from ppg2ecg.evaluation.efficiency import benchmark
from ppg2ecg.evaluation.metrics import evaluate_windows, hf_energy_ratio, penguin_hr_error, rhythm_morphology_metrics, summarize
from ppg2ecg.flow.imeanflow import MeanFlowS5, sample_meanflow
from ppg2ecg.flow.samplers import SAMPLERS, nfe_of
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.models.regressor import REGRESSOR_MODELS
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import assert_upstream_pinned

ROOT = Path(__file__).resolve().parents[1]
FS = 128
OTCFM_ARMS = [("heun", 25), ("heun", 10), ("heun", 5), ("heun", 2), ("heun", 1), ("euler", 1)]
IMF_ARMS = [1, 2, 4]
CSV_FIELDS = ["solver", "solver_steps", "actual_NFE", "hr_abs_err_bpm", "rpeak_f1", "rpeak_precision", "rpeak_recall", "rr_mae_ms", "mae", "rmse", "pcc", "qrs_width_err_ms", "morph_corr", "hr_err_penguin_corrected", "hr_err_penguin_as_shipped", "amp_ratio", "amp_ratio_median", "hf_ratio_pred", "hf_ratio_target", "cond_gain_bpm", "hr_err_shuffled_right_target", "hr_err_shuffled_wrong_target", "seed_std_mean", "seed_pairwise_corr", "latency_ms_batch64", "samples_per_s", "peak_mem_MiB", "n_windows", "frac_windows_no_pred_beats", "beats_ratio"]


def load_test(processed: Path, subjects, subsample=None):
    xs, ys, ids, starts = [], [], [], []
    for s in subjects:
        d = np.load(processed / f"{s}.npz")
        xs.append(d["x"])
        ys.append(d["y"])
        ids += [s] * len(d["x"])
        starts.append(d["window_start_s"])
    x, y, ids, starts = np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), np.array(ids), np.concatenate(starts)
    if subsample and len(x) > subsample:
        st = -(-len(x) // subsample)
        x, y, ids, starts = x[::st], y[::st], ids[::st], starts[::st]
    return x, y, ids, starts


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not np.any(p == np.arange(n)):
            return p


@torch.no_grad()
def run_otcfm(model, x_np, x0_all, solver, steps, batch, device):
    fn, out, nfes = SAMPLERS[solver], [], set()
    for i in range(0, len(x_np), batch):
        ppg = torch.from_numpy(x_np[i : i + batch]).to(device).unsqueeze(1)
        x1, nfe = fn(lambda x, t: model.forward_step(x, ppg, t), x0_all[i : i + batch].to(device), steps)
        nfes.add(nfe)
        out.append(x1.squeeze(1).float().cpu().numpy())
    assert nfes == {nfe_of(solver, steps)}
    return np.concatenate(out)


@torch.no_grad()
def run_imf(net, x_np, e_all, steps, batch, device):
    out = []
    for i in range(0, len(x_np), batch):
        ppg = torch.from_numpy(x_np[i : i + batch]).to(device).unsqueeze(1)
        x1, nfe = sample_meanflow(net, ppg, e_all[i : i + batch].to(device), steps)
        assert nfe == steps
        out.append(x1.squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def run_mse(model, x_np, batch, device):
    return np.concatenate([model(torch.from_numpy(x_np[i : i + batch]).to(device).unsqueeze(1)).squeeze(1).float().cpu().numpy() for i in range(0, len(x_np), batch)])


def row_for(pred, pred_sh, y, perm, key, solver, steps, nfe, eff, detector):
    ev = evaluate_windows(pred, y, fs=FS, hr_window_segments=1, detector=detector)
    summ = summarize(ev, n_boot=1000, seed=0)
    amp = pred.std(axis=1) / (y.std(axis=1) + 1e-8)
    rm_right = rhythm_morphology_metrics(pred_sh, y[perm], FS, detector=detector)
    rm_wrong = rhythm_morphology_metrics(pred_sh, y, FS, detector=detector)
    hr_right, hr_wrong = float(np.nanmean(rm_right["hr_abs_err"])), float(np.nanmean(rm_wrong["hr_abs_err"]))
    row = dict(solver=solver, solver_steps=steps, actual_NFE=nfe, hr_abs_err_bpm=summ["hr_abs_err"]["mean"], rpeak_f1=summ["rpeak_f1"]["mean"], rpeak_precision=summ["rpeak_precision"]["mean"], rpeak_recall=summ["rpeak_recall"]["mean"], rr_mae_ms=summ["rr_mae_ms"]["mean"], mae=summ["mae"]["mean"], rmse=summ["rmse"]["mean"], pcc=summ["pcc"]["mean"], qrs_width_err_ms=summ["qrs_width_err_ms"]["mean"], morph_corr=summ["morph_corr"]["mean"], hr_err_penguin_corrected=penguin_hr_error(pred, y, window_s=8, mode="corrected"), hr_err_penguin_as_shipped=penguin_hr_error(pred, y, window_s=8, mode="as_shipped"), amp_ratio=float(amp.mean()), amp_ratio_median=float(np.median(amp)), hf_ratio_pred=float(hf_energy_ratio(pred).mean()), hf_ratio_target=float(hf_energy_ratio(y).mean()), cond_gain_bpm=hr_wrong - hr_right, hr_err_shuffled_right_target=hr_right, hr_err_shuffled_wrong_target=hr_wrong, seed_std_mean=0.0, seed_pairwise_corr=1.0, latency_ms_batch64=eff["latency_ms_median"], samples_per_s=eff["samples_per_s"], peak_mem_MiB=eff.get("peak_mem_MiB"), n_windows=int(len(y)), frac_windows_no_pred_beats=float(np.mean(ev["rhythm"]["n_pred_beats"] == 0)), beats_ratio=float(ev["rhythm"]["n_pred_beats"].mean() / max(ev["rhythm"]["n_ref_beats"].mean(), 1e-9)))
    return row, ev, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["otcfm", "imf", "mse"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--processed", default="data/processed/wildppg_8s_prenorm")
    ap.add_argument("--manifest", default="data/manifests/split_a4_wildppg_seed42.json")
    ap.add_argument("--subsample", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--bench-repeats", type=int, default=10)
    ap.add_argument("--detector", default="neurokit")
    args = ap.parse_args()
    out = ROOT / args.out_dir
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    seed_everything(args.noise_seed, deterministic=True)
    device = torch.device("cuda")
    up = assert_upstream_pinned()
    ck = torch.load(Path(args.checkpoint) if args.checkpoint else out / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    tn = ck.get("target_norm") or {"mu": 0.0, "sigma": 1.0, "source": "identity"}
    tnorm = TargetNorm(float(tn["mu"]), float(tn["sigma"]), str(tn.get("source", "")))
    split = read_manifest(ROOT / args.manifest)[0]
    x, y_native, sid, starts = load_test(ROOT / args.processed, split["test"], args.subsample)
    y = tnorm.forward(y_native) if not tnorm.is_identity else y_native  # evaluate in the trained space (prereg §9)
    n, T = x.shape
    np.savez_compressed(out / "predictions" / "test_inputs.npz", x=x, y=y, y_native=y_native, sid=sid, starts=starts, target_norm=json.dumps(tn))
    g = torch.Generator().manual_seed(args.noise_seed)
    noise = torch.randn(n, 1, T, generator=g)
    perm = derangement(n, args.noise_seed + 1)
    xb, nb = x[: args.batch_size], noise[: args.batch_size]
    meta = {"arm": args.arm, "checkpoint_epoch": ck.get("epoch"), "target_norm": tn, "evaluation_space": "global-z (trained space); y_native and the inverse transform are stored alongside", "test_subjects": split["test"], "n_windows": int(n), "T": int(T), "noise_seed": args.noise_seed, "upstream": up, "created": datetime.now().isoformat(timespec="seconds"), "arms": {}}
    rows = []
    if args.arm == "otcfm":
        model = build_penguin_backbone(**ck["model_cfg"]).to(device).eval()
        model.load_state_dict(ck["state_dict"])
        arms = [(f"{s}{k}", s, k, nfe_of(s, k), lambda a, b, s=s, k=k: run_otcfm(model, a, b, s, k, args.batch_size, device)) for s, k in OTCFM_ARMS]
    elif args.arm == "imf":
        imf_cfg = ck.get("imf_cfg", {})
        net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=imf_cfg.get("cond_mode", "h_only"), h_scale=imf_cfg.get("h_scale", 1.0)).to(device).eval()
        net.load_state_dict(ck["state_dict"])
        arms = [(f"meanflow{k}", "meanflow", k, k, lambda a, b, k=k: run_imf(net, a, b, k, args.batch_size, device)) for k in IMF_ARMS]
    else:
        cls, count = REGRESSOR_MODELS[ck.get("model_key", "state_token" if "state_token" in ck["state_dict"] else "full_backbone")]
        model = cls(**ck["model_cfg"]).to(device).eval()
        model.load_state_dict(ck["state_dict"])
        meta["params"] = count(model)
        arms = [("regressor", "regressor", 1, 1, lambda a, b: run_mse(model, a, args.batch_size, device))]
    for key, solver, steps, nfe, fn in arms:
        pred, pred_sh = fn(x, noise), fn(x[perm], noise)
        eff = benchmark(lambda: fn(xb, nb), n_warmup=2, n_repeats=args.bench_repeats, batch_size=len(xb))
        row, ev, summ = row_for(pred, pred_sh, y, perm, key, solver, steps, nfe, eff, args.detector)
        rows.append(row)
        meta["arms"][key] = {"row": row, "summary": summ}
        np.savez_compressed(out / "predictions" / f"{key}.npz", pred=pred.astype(np.float32), pred_shuffled=pred_sh.astype(np.float32), perm=perm, solver=solver, steps=steps, nfe=nfe, **{f"pw_{k}": v for k, v in ev["signal"].items()}, **{f"pw_{k}": v for k, v in ev["rhythm"].items()})
        print(f"{key}: NFE {nfe} | HR {row['hr_abs_err_bpm']:.2f} morph {row['morph_corr']:.3f} amp {row['amp_ratio']:.2f} (med {row['amp_ratio_median']:.2f}) beats {row['beats_ratio']:.2f} F1 {row['rpeak_f1']:.3f} RR {row['rr_mae_ms']:.1f} HF {row['hf_ratio_pred']:.3f} (GT {row['hf_ratio_target']:.3f}) gain {row['cond_gain_bpm']:.2f} RMSE {row['rmse']:.3f} | {eff['latency_ms_median']:.0f} ms", flush=True)
    with open(out / "nfe_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out / "metrics.json").write_text(json.dumps(meta, indent=1, default=str))
    print("wrote", out / "nfe_curve.csv")


if __name__ == "__main__":
    main()
