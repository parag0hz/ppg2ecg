"""A8 §6: pre-training transport-scale audit — raw-mmHg vs global-z target geometry against the standard-normal prior,
on a fixed deterministic subset of TRAIN windows. Metric definitions are frozen before any A8 result."""
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


def stats(y: np.ndarray, e: np.ndarray) -> dict:
    ny, ne = np.linalg.norm(y, axis=1), np.linalg.norm(e, axis=1)
    d = {"target_mean": float(y.mean()), "target_std": float(y.std()), "target_min": float(y.min()), "target_max": float(y.max()),
         "target_l2_norm_mean": float(ny.mean()), "target_l2_norm_std": float(ny.std()),
         "prior_mean": float(e.mean()), "prior_std": float(e.std()), "prior_l2_norm_mean": float(ne.mean()),
         "norm_ratio_target_over_prior": float((ny / ne).mean()), "distance_to_prior_mean": float(np.linalg.norm(y - e, axis=1).mean()),
         "velocity_norm_mean_e_minus_y": float(np.linalg.norm(e - y, axis=1).mean()),
         "cos_target_prior_mean": float(np.mean((y * e).sum(1) / (ny * ne)))}
    for t in TS:  # interpolant of the OT-CFM/iMF path (t = 1 is the prior in the iMF convention used here)
        zt = (1 - t) * y + t * e
        d[f"interp_norm_t{t}"] = float(np.linalg.norm(zt, axis=1).mean())
        d[f"interp_std_t{t}"] = float(zt.std())
        d[f"prior_share_of_interp_energy_t{t}"] = float(((t * e) ** 2).sum(1).mean() / (((1 - t) * y) ** 2 + (t * e) ** 2).sum(1).mean())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/split_a7_mimicbp_official.json")
    ap.add_argument("--processed", default="data/processed/mimicbp_8s")
    ap.add_argument("--norm", default="artifacts/a8_abp_scale_control/normalization.json")
    ap.add_argument("--n-windows", type=int, default=4096)
    ap.add_argument("--n-subjects", type=int, default=64, help="first N train subjects (sorted) — deterministic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/a8_abp_scale_control/transport_geometry.json")
    args = ap.parse_args()
    split = read_manifest(ROOT / args.manifest)[0]
    subs = sorted(split["train"])[: args.n_subjects]
    _, y, _ = load_arrays(ROOT / args.processed, subs)
    stride = max(1, len(y) // args.n_windows)
    y = y[::stride][: args.n_windows].astype(np.float64)
    g = torch.Generator().manual_seed(args.seed)
    e = torch.randn(y.shape, generator=g).numpy().astype(np.float64)
    tn = TargetNorm.load(ROOT / args.norm)
    rec = {"created": datetime.now().isoformat(timespec="seconds"), "n_windows": int(len(y)), "n_train_subjects_used": len(subs), "T": int(y.shape[1]), "prior": "standard normal, seed 0", "normalization": {"mu": tn.mu, "sigma": tn.sigma},
           "raw": stats(y, e), "global_z": stats((y - tn.mu) / tn.sigma, e)}
    rec["ratios_raw_over_norm"] = {k: (rec["raw"][k] / rec["global_z"][k] if rec["global_z"][k] not in (0.0,) else None) for k in ("target_std", "target_l2_norm_mean", "norm_ratio_target_over_prior", "distance_to_prior_mean", "velocity_norm_mean_e_minus_y")}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    for k in ("target_mean", "target_std", "target_l2_norm_mean", "prior_l2_norm_mean", "norm_ratio_target_over_prior", "distance_to_prior_mean", "velocity_norm_mean_e_minus_y", "prior_share_of_interp_energy_t0.5", "interp_std_t0.5"):
        print(f"{k:38s} raw {rec['raw'][k]:12.4f} | global-z {rec['global_z'][k]:10.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
