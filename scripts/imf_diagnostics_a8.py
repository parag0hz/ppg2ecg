"""A8 §11-§12: objective/JVP/gradient diagnostics for an iMeanFlow checkpoint on a FIXED validation subset, in the space the
model was trained in (raw mmHg or global-z). Same (t, r, e) draw for every checkpoint (seed 1000) so raw and normalised runs are
compared on identical stochastic inputs. Writes/append rows to artifacts/a8_abp_scale_control/imeanflow_diagnostics.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401
import numpy as np
import torch

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.data.target_norm import TargetNorm
from ppg2ecg.flow.imeanflow import MeanFlowS5, compound_V, imeanflow_loss, sample_tr
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.training.train_a0 import load_arrays

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["run", "label", "scale", "checkpoint", "epoch", "n_windows", "delta2_mean", "mse_V", "u_norm", "v_theta_norm", "V_norm", "dudt_norm", "interval_dudt_norm", "jvp_norm", "residual_norm", "grad_norm", "grad_max_abs", "w_mean", "w_median", "w_p01", "w_p10", "w_p25", "w_p75", "w_p90", "w_p99", "w_min", "w_max", "w_std", "w_saturation_frac", "w_near_lower_frac", "loss_before_weighting", "loss_after_weighting", "target_std", "target_mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/<run dir>")
    ap.add_argument("--label", default="best")
    ap.add_argument("--checkpoint", default="checkpoint_best.pt")
    ap.add_argument("--manifest", default="data/manifests/split_a7_mimicbp_official.json")
    ap.add_argument("--processed", default="data/processed/mimicbp_8s")
    ap.add_argument("--n-windows", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="artifacts/a8_abp_scale_control/imeanflow_diagnostics.csv")
    args = ap.parse_args()
    device = torch.device("cuda")
    ck = torch.load(ROOT / args.run / args.checkpoint, map_location="cpu", weights_only=False)
    tn = ck.get("target_norm") or {"mu": 0.0, "sigma": 1.0}
    tnorm = TargetNorm(float(tn["mu"]), float(tn["sigma"]))
    imf_cfg = ck.get("imf_cfg", {})
    net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=imf_cfg.get("cond_mode", "h_only"), h_scale=imf_cfg.get("h_scale", 1.0)).to(device)
    net.load_state_dict(ck["state_dict"])
    split = read_manifest(ROOT / args.manifest)[0]
    x, y, _ = load_arrays(ROOT / args.processed, split["val"])
    stride = max(1, len(x) // args.n_windows)
    x, y = x[::stride][: args.n_windows], y[::stride][: args.n_windows]
    if not tnorm.is_identity:
        y = tnorm.forward(y)
    gen = torch.Generator().manual_seed(args.seed)
    acc, n = {k: 0.0 for k in FIELDS if k not in ("run", "label", "scale", "checkpoint", "epoch", "n_windows", "target_std", "target_mean", "grad_max_abs")}, 0
    net.zero_grad(set_to_none=True)
    for i in range(0, len(x), args.micro_batch):
        xb = torch.from_numpy(x[i : i + args.micro_batch]).to(device).unsqueeze(1)
        yb = torch.from_numpy(y[i : i + args.micro_batch]).to(device).unsqueeze(1)
        B = len(xb)
        t, r, _ = sample_tr(B, gen, p_mean=-0.4, p_std=1.0, data_proportion=0.5)
        t, r = t.to(device), r.to(device)
        e = torch.randn(B, 1, xb.shape[-1], generator=gen).to(device)
        loss, info = imeanflow_loss(net, yb, xb, e, t, r)
        (loss * B / len(x)).backward()
        with torch.no_grad():
            tt = t.reshape(-1, 1, 1)
            z_t = (1 - tt) * yb + tt * e
            v_theta = net.u(z_t, xb, t, torch.zeros_like(t))
            u, dudt, V = compound_V(lambda z, t_, r_: net.u(z, xb, t_, t_ - r_), z_t, t, r, v_theta, "forward")
            w = B / len(x)
            acc["u_norm"] += float(u.flatten(1).norm(dim=1).mean()) * w
            acc["v_theta_norm"] += float(v_theta.flatten(1).norm(dim=1).mean()) * w
            acc["V_norm"] += float(V.flatten(1).norm(dim=1).mean()) * w
            acc["dudt_norm"] += float(dudt.flatten(1).norm(dim=1).mean()) * w
            acc["interval_dudt_norm"] += float(((t - r).reshape(-1, 1, 1) * dudt).flatten(1).norm(dim=1).mean()) * w
            acc["jvp_norm"] += float(dudt.flatten(1).norm(dim=1).mean()) * w
            acc["residual_norm"] += float((V - (e - yb)).flatten(1).norm(dim=1).mean()) * w
            for k in ("delta2_mean", "w_mean", "w_median", "w_p01", "w_p10", "w_p25", "w_p75", "w_p90", "w_p99", "w_min", "w_max", "w_std", "w_saturation_frac", "w_near_lower_frac", "loss_before_weighting", "loss_after_weighting"):
                acc[k] += float(info[k]) * w
            acc["mse_V"] += float(info["mse"]) * w
        n += B
    gsq = sum(float((p.grad.detach().double() ** 2).sum()) for p in net.parameters() if p.grad is not None)  # float64: the raw-mmHg run overflows fp32
    gn = float(np.sqrt(gsq))
    gmax = max(float(p.grad.detach().abs().max()) for p in net.parameters() if p.grad is not None)
    row = {"run": args.run, "label": args.label, "scale": "global-z" if not tnorm.is_identity else "native (no target transform)", "checkpoint": args.checkpoint, "epoch": ck.get("epoch"), "n_windows": n, "grad_norm": gn, "grad_max_abs": gmax, "target_std": float(y.std()), "target_mean": float(y.mean()), **{k: v for k, v in acc.items() if k != "grad_norm"}}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})
    print(f"{args.run} [{row['scale']}] {args.label}: delta2 {row['delta2_mean']:.4g} | |u| {row['u_norm']:.4g} |v| {row['v_theta_norm']:.4g} |V| {row['V_norm']:.4g} |du/dt| {row['dudt_norm']:.4g} |(t-r)du/dt| {row['interval_dudt_norm']:.4g} | residual {row['residual_norm']:.4g} | grad {gn:.4g} (max |g| {gmax:.3g}) | w med {row['w_median']:.3e} p10 {row['w_p10']:.3e} p90 {row['w_p90']:.3e} sat {row['w_saturation_frac']:.4f}")


if __name__ == "__main__":
    main()
