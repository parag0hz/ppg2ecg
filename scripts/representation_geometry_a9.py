"""A9 §8: representation geometry audit — window-local-normalised vs global-z ECG targets against the standard-normal prior,
on a fixed deterministic train subset. Definitions frozen before any A9 training."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401
import numpy as np
import torch

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.data.target_norm import TargetNorm
from ppg2ecg.training.train_a0 import load_arrays

ROOT = Path(__file__).resolve().parents[1]
TS = (0.0, 0.25, 0.5, 0.75, 1.0)
FS = 128


def stats(y: np.ndarray, e: np.ndarray) -> dict:
    ny, ne = np.linalg.norm(y, axis=1), np.linalg.norm(e, axis=1)
    d = np.diff(y, axis=1) * FS
    spec = np.abs(np.fft.rfft(y - y.mean(1, keepdims=True), axis=1)) ** 2
    f = np.fft.rfftfreq(y.shape[1], 1 / FS)
    out = {"target_mean": float(y.mean()), "target_std": float(y.std()), "target_min": float(y.min()), "target_max": float(y.max()),
           "per_window_mean_mean": float(y.mean(1).mean()), "per_window_mean_std": float(y.mean(1).std()),
           "per_window_std_mean": float(y.std(1).mean()), "per_window_std_std": float(y.std(1).std()),
           "target_l2_norm_mean": float(ny.mean()), "target_l2_norm_std": float(ny.std()),
           "prior_l2_norm_mean": float(ne.mean()), "norm_ratio_target_over_prior": float((ny / ne).mean()),
           "distance_to_prior_mean": float(np.linalg.norm(y - e, axis=1).mean()),
           "total_energy_mean": float((y**2).sum(1).mean()), "hf_ratio_gt15hz": float((spec[:, f > 15].sum(1) / (spec.sum(1) + 1e-12)).mean()),
           "derivative_rms": float(np.sqrt((d**2).mean())), "max_slope_median": float(np.median(np.abs(d).max(1))), "max_slope_p90": float(np.percentile(np.abs(d).max(1), 90))}
    for t in TS:
        zt = (1 - t) * y + t * e
        out[f"interp_norm_t{t}"] = float(np.linalg.norm(zt, axis=1).mean())
        out[f"prior_share_of_interp_energy_t{t}"] = float(((t * e) ** 2).sum(1).mean() / (((1 - t) * y) ** 2 + (t * e) ** 2).sum(1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/split_a4_wildppg_seed42.json")
    ap.add_argument("--window-norm", default="data/processed/wildppg_8s")
    ap.add_argument("--prenorm", default="data/processed/wildppg_8s_prenorm")
    ap.add_argument("--norm", default="artifacts/a9_ecg_representation_control/normalization.json")
    ap.add_argument("--n-windows", type=int, default=4096)
    ap.add_argument("--n-subjects", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/a9_ecg_representation_control/representation_geometry.json")
    args = ap.parse_args()
    split = read_manifest(ROOT / args.manifest)[0]
    subs = sorted(split["train"])[: args.n_subjects]
    xw, yw, _ = load_arrays(ROOT / args.window_norm, subs)
    xp, yp, _ = load_arrays(ROOT / args.prenorm, subs)
    assert xw.shape == xp.shape and np.array_equal(xw, xp), "PPG differs between the two processed dirs"
    stride = max(1, len(yw) // args.n_windows)
    yw, yp = yw[::stride][: args.n_windows].astype(np.float64), yp[::stride][: args.n_windows].astype(np.float64)
    g = torch.Generator().manual_seed(args.seed)
    e = torch.randn(yw.shape, generator=g).numpy().astype(np.float64)
    tn = TargetNorm.load(ROOT / args.norm)
    rec = {"created": datetime.now().isoformat(timespec="seconds"), "n_windows": int(len(yw)), "n_train_subjects_used": len(subs), "subjects": subs,
           "normalization": {"mu": tn.mu, "sigma": tn.sigma}, "ppg_bit_exact_between_dirs": True,
           "window_norm": stats(yw, e), "global_z": stats((yp - tn.mu) / tn.sigma, e), "prenorm_raw": stats(yp, e)}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    for k in ("target_mean", "target_std", "per_window_mean_std", "per_window_std_mean", "target_l2_norm_mean", "prior_l2_norm_mean", "norm_ratio_target_over_prior", "distance_to_prior_mean", "hf_ratio_gt15hz", "derivative_rms", "max_slope_median", "prior_share_of_interp_energy_t0.5"):
        print(f"{k:34s} window-norm {rec['window_norm'][k]:12.4f} | global-z {rec['global_z'][k]:12.4f} | prenorm-raw {rec['prenorm_raw'][k]:12.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
