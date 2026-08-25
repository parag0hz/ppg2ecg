"""A2 evaluation: iMeanFlow checkpoint on the test subject with the SAME paired noise / shuffle / metrics as the OT-CFM arms.
Arms: meanflow 1 step (1 NFE, primary), 2 and 4 steps (diagnostic). Writes nfe_curve.csv, metrics.json, predictions/, figures/."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401  (MKL warm-up must precede `import torch`, docs/ENVIRONMENT.md)
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from ppg2ecg.data.splits import read_manifest  # noqa: E402
from ppg2ecg.evaluation.efficiency import benchmark  # noqa: E402
from ppg2ecg.evaluation.metrics import evaluate_windows, hf_energy_ratio, penguin_hr_error, rhythm_morphology_metrics, summarize  # noqa: E402
from ppg2ecg.flow.imeanflow import MeanFlowS5, sample_meanflow  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402
from ppg2ecg.utils.seed import seed_everything  # noqa: E402
from ppg2ecg.utils.upstream import assert_upstream_pinned  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS = 128
CSV_FIELDS = ["solver", "solver_steps", "actual_NFE", "hr_abs_err_bpm", "rpeak_f1", "rpeak_precision", "rpeak_recall", "rr_mae_ms", "mae", "rmse", "pcc", "qrs_width_err_ms", "morph_corr", "hr_err_penguin_corrected", "hr_err_penguin_as_shipped", "amp_ratio", "amp_ratio_median", "hf_ratio_pred", "hf_ratio_target", "cond_gain_bpm", "hr_err_shuffled_right_target", "hr_err_shuffled_wrong_target", "seed_std_mean", "seed_pairwise_corr", "latency_ms_batch64", "samples_per_s", "peak_mem_MiB", "n_windows", "frac_windows_no_pred_beats", "beats_ratio"]


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


@torch.no_grad()
def sample_all(net, ppg_np, e_all, steps, batch, device):
    out, nfes = [], set()
    for i in range(0, len(ppg_np), batch):
        ppg = torch.from_numpy(ppg_np[i : i + batch]).to(device).unsqueeze(1)
        e = e_all[i : i + batch].to(device)
        x1, nfe = sample_meanflow(net, ppg, e, steps)
        nfes.add(nfe)
        out.append(x1.squeeze(1).float().cpu().numpy())
    assert nfes == {steps}
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
    ap.add_argument("--steps", default="1,2,4")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--limit-windows", type=int, default=None)
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
    imf_cfg = ck.get("imf_cfg", {})
    net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=imf_cfg.get("cond_mode", "t_plus_h"), h_scale=imf_cfg.get("h_scale", 1.0)).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    split = read_manifest(ROOT / args.manifest)[0]
    x_te, y_te, sid, starts = load_test(ROOT / args.processed, split["test"], args.limit_windows)
    n, T = x_te.shape
    g = torch.Generator().manual_seed(args.noise_seed)
    e_all = torch.randn(n, 1, T, generator=g)  # IDENTICAL tensor to the OT-CFM arms' paired x0 (same seed, same draw)
    perm = derangement(n, args.noise_seed + 1)
    xb = torch.from_numpy(x_te[: args.batch_size]).to(device).unsqueeze(1)
    eb = e_all[: args.batch_size].to(device)
    rows, full, preds = [], {"checkpoint": str(ckpt_path), "checkpoint_epoch": ck.get("epoch"), "objective": "improved_meanflow", "test_subjects": split["test"], "n_windows": int(n), "T": int(T), "noise_seed": args.noise_seed, "upstream": up, "created": datetime.now().isoformat(timespec="seconds"), "arms": {}}, {}
    hf_t = float(hf_energy_ratio(y_te).mean())
    for steps in [int(s) for s in args.steps.split(",")]:
        key = f"meanflow{steps}"
        pred = sample_all(net, x_te, e_all, steps, args.batch_size, device)
        preds[key] = pred
        ev = evaluate_windows(pred, y_te, fs=FS, hr_window_segments=1, detector=args.detector)
        summ = summarize(ev, n_boot=1000, seed=0)
        amp = pred.std(axis=1) / (y_te.std(axis=1) + 1e-8)
        hr_c = penguin_hr_error(pred, y_te, window_s=8, mode="corrected")
        hr_s = penguin_hr_error(pred, y_te, window_s=8, mode="as_shipped")
        eff = benchmark(lambda: sample_meanflow(net, xb, eb, steps), n_warmup=3, n_repeats=args.bench_repeats, batch_size=len(xb))
        pred_sh = sample_all(net, x_te[perm], e_all, steps, args.batch_size, device)
        rm_right = rhythm_morphology_metrics(pred_sh, y_te[perm], FS, detector=args.detector)
        rm_wrong = rhythm_morphology_metrics(pred_sh, y_te, FS, detector=args.detector)
        hr_right, hr_wrong = float(np.nanmean(rm_right["hr_abs_err"])), float(np.nanmean(rm_wrong["hr_abs_err"]))
        m = min(args.diversity_windows, n)
        sp = [pred[:m]]
        for k in range(1, args.diversity_seeds):
            gk = torch.Generator().manual_seed(args.noise_seed + 100 + k)
            sp.append(sample_all(net, x_te[:m], torch.randn(m, 1, T, generator=gk), steps, args.batch_size, device))
        sp = np.stack(sp)
        seed_std = float(sp.std(axis=0).mean())
        corrs = [float(np.mean([np.corrcoef(sp[a, i], sp[b, i])[0, 1] for i in range(m)])) for a in range(len(sp)) for b in range(a + 1, len(sp))]
        np.savez_compressed(out / "predictions" / f"{key}.npz", pred=pred.astype(np.float32), pred_shuffled=pred_sh.astype(np.float32), perm=perm, solver="meanflow", steps=steps, nfe=steps, **{f"pw_{k}": v for k, v in ev["signal"].items()}, **{f"pw_{k}": v for k, v in ev["rhythm"].items()})
        row = dict(solver="meanflow", solver_steps=steps, actual_NFE=steps, hr_abs_err_bpm=summ["hr_abs_err"]["mean"], rpeak_f1=summ["rpeak_f1"]["mean"], rpeak_precision=summ["rpeak_precision"]["mean"], rpeak_recall=summ["rpeak_recall"]["mean"], rr_mae_ms=summ["rr_mae_ms"]["mean"], mae=summ["mae"]["mean"], rmse=summ["rmse"]["mean"], pcc=summ["pcc"]["mean"], qrs_width_err_ms=summ["qrs_width_err_ms"]["mean"], morph_corr=summ["morph_corr"]["mean"], hr_err_penguin_corrected=hr_c, hr_err_penguin_as_shipped=hr_s, amp_ratio=float(amp.mean()), amp_ratio_median=float(np.median(amp)), hf_ratio_pred=float(hf_energy_ratio(pred).mean()), hf_ratio_target=hf_t, cond_gain_bpm=hr_wrong - hr_right, hr_err_shuffled_right_target=hr_right, hr_err_shuffled_wrong_target=hr_wrong, seed_std_mean=seed_std, seed_pairwise_corr=float(np.mean(corrs)) if corrs else np.nan, latency_ms_batch64=eff["latency_ms_median"], samples_per_s=eff["samples_per_s"], peak_mem_MiB=eff.get("peak_mem_MiB"), n_windows=int(n), frac_windows_no_pred_beats=float(np.mean(ev["rhythm"]["n_pred_beats"] == 0)), beats_ratio=float(ev["rhythm"]["n_pred_beats"].mean() / max(ev["rhythm"]["n_ref_beats"].mean(), 1e-9)))
        rows.append(row)
        full["arms"][key] = {"row": row, "summary": summ, "efficiency": eff, "shuffle": {"hr_right": hr_right, "hr_wrong": hr_wrong, "morph_corr_right": float(np.nanmean(rm_right["morph_corr"]))}, "diversity": {"n_seeds": len(sp), "n_windows": m, "std_mean": seed_std, "pairwise_corr": corrs}}
        print(f"meanflow steps={steps} NFE={steps} | HRerr {row['hr_abs_err_bpm']:.2f} amp {row['amp_ratio']:.2f} beats {row['beats_ratio']:.2f} F1 {row['rpeak_f1']:.3f} RMSE {row['rmse']:.3f} morph {row['morph_corr']:.3f} | gain {row['cond_gain_bpm']:.2f} seedstd {seed_std:.3f} | {eff['latency_ms_median']:.0f} ms", flush=True)
    with open(out / "nfe_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out / "metrics.json").write_text(json.dumps(full, indent=1, default=str))
    print("wrote", out / "nfe_curve.csv", out / "metrics.json")


if __name__ == "__main__":
    main()
