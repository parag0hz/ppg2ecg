"""A7: dump paired predictions of one arm on the (subsampled) MIMIC-BP test set — no ECG metrics (ABP metrics are computed in
scripts/analyze_a7.py from the saved arrays). Arms: otcfm (heun 25/10/5/2/1 + euler 1 = 50/20/10/4/2/1 NFE), imf (meanflow 1/2/4),
mse (one deterministic forward). Paired noise seed 0 (identical x0/e across arms), PPG derangement seed 1 for the shuffle control,
latency at batch 64. Writes predictions/test_inputs.npz (+ label_sbp/dbp, pid, segment_idx) and predictions/<arm>.npz."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401
import numpy as np
import torch

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.evaluation.efficiency import benchmark
from ppg2ecg.flow.imeanflow import MeanFlowS5, sample_meanflow
from ppg2ecg.flow.samplers import SAMPLERS, nfe_of
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.models.regressor import REGRESSOR_MODELS
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import assert_upstream_pinned

ROOT = Path(__file__).resolve().parents[1]
OTCFM_ARMS = [("heun", 25), ("heun", 10), ("heun", 5), ("heun", 2), ("heun", 1), ("euler", 1)]
IMF_ARMS = [1, 2, 4]


def load_test(processed, subjects, subsample=None):
    xs, ys, ids, seg, starts, sbp, dbp = [], [], [], [], [], [], []
    for s in subjects:
        d = np.load(processed / f"{s}.npz")
        xs.append(d["x"])
        ys.append(d["y"])
        ids += [s] * len(d["x"])
        seg.append(d["segment_idx"])
        starts.append(d["window_start_s"])
        sbp.append(d["label_sbp"])
        dbp.append(d["label_dbp"])
    x, y, ids, seg, starts, sbp, dbp = (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), np.array(ids), np.concatenate(seg), np.concatenate(starts), np.concatenate(sbp), np.concatenate(dbp))
    if subsample and len(x) > subsample:  # uniform stride, as A4/A5
        st = -(-len(x) // subsample)
        x, y, ids, seg, starts, sbp, dbp = x[::st], y[::st], ids[::st], seg[::st], starts[::st], sbp[::st], dbp[::st]
    return x, y, ids, seg, starts, sbp, dbp


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
        v = lambda x, t: model.forward_step(x, ppg, t)  # noqa: E731
        x1, nfe = fn(v, x0_all[i : i + batch].to(device), steps)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["otcfm", "imf", "mse"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--processed", default="data/processed/mimicbp_8s")
    ap.add_argument("--manifest", default="data/manifests/split_a7_mimicbp_official.json")
    ap.add_argument("--subsample", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--bench-repeats", type=int, default=10)
    args = ap.parse_args()
    out = ROOT / args.out_dir
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    seed_everything(args.noise_seed, deterministic=True)
    device = torch.device("cuda")
    up = assert_upstream_pinned()
    ck = torch.load(Path(args.checkpoint) if args.checkpoint else out / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    split = read_manifest(ROOT / args.manifest)[0]
    x, y, ids, seg, starts, sbp, dbp = load_test(ROOT / args.processed, split["test"], args.subsample)
    n, T = x.shape
    np.savez_compressed(out / "predictions" / "test_inputs.npz", x=x, y=y, sid=ids, segment_idx=seg, starts=starts, label_sbp=sbp, label_dbp=dbp)
    g = torch.Generator().manual_seed(args.noise_seed)
    noise = torch.randn(n, 1, T, generator=g)  # identical paired noise for OT-CFM x0 and iMF e
    perm = derangement(n, args.noise_seed + 1)
    xb = x[: args.batch_size]
    meta = {"arm": args.arm, "checkpoint_epoch": ck.get("epoch"), "n_windows": int(n), "T": int(T), "noise_seed": args.noise_seed, "upstream": up, "created": datetime.now().isoformat(timespec="seconds"), "arms": {}}
    if args.arm == "otcfm":
        model = build_penguin_backbone(**ck["model_cfg"]).to(device).eval()
        model.load_state_dict(ck["state_dict"])
        for solver, steps in OTCFM_ARMS:
            key, nfe = f"{solver}{steps}", nfe_of(solver, steps)
            pred = run_otcfm(model, x, noise, solver, steps, args.batch_size, device)
            pred_sh = run_otcfm(model, x[perm], noise, solver, steps, args.batch_size, device)
            eff = benchmark(lambda: run_otcfm(model, xb, noise[: len(xb)], solver, steps, args.batch_size, device), n_warmup=2, n_repeats=args.bench_repeats, batch_size=len(xb))
            np.savez_compressed(out / "predictions" / f"{key}.npz", pred=pred, pred_shuffled=pred_sh, perm=perm, solver=solver, steps=steps, nfe=nfe)
            meta["arms"][key] = {"solver": solver, "steps": steps, "nfe": nfe, "latency_ms_batch64": eff["latency_ms_median"], "peak_mem_MiB": eff.get("peak_mem_MiB")}
            print(f"{key}: NFE {nfe} latency {eff['latency_ms_median']:.0f} ms | pred mean {pred.mean():.1f} std {pred.std(axis=1).mean():.2f} (GT std {y.std(axis=1).mean():.2f})", flush=True)
    elif args.arm == "imf":
        imf_cfg = ck.get("imf_cfg", {})
        net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=imf_cfg.get("cond_mode", "h_only"), h_scale=imf_cfg.get("h_scale", 1.0)).to(device).eval()
        net.load_state_dict(ck["state_dict"])
        for steps in IMF_ARMS:
            key = f"meanflow{steps}"
            pred = run_imf(net, x, noise, steps, args.batch_size, device)
            pred_sh = run_imf(net, x[perm], noise, steps, args.batch_size, device)
            eff = benchmark(lambda: run_imf(net, xb, noise[: len(xb)], steps, args.batch_size, device), n_warmup=2, n_repeats=args.bench_repeats, batch_size=len(xb))
            np.savez_compressed(out / "predictions" / f"{key}.npz", pred=pred, pred_shuffled=pred_sh, perm=perm, solver="meanflow", steps=steps, nfe=steps)
            meta["arms"][key] = {"solver": "meanflow", "steps": steps, "nfe": steps, "latency_ms_batch64": eff["latency_ms_median"], "peak_mem_MiB": eff.get("peak_mem_MiB")}
            print(f"{key}: NFE {steps} latency {eff['latency_ms_median']:.0f} ms | pred mean {pred.mean():.1f} std {pred.std(axis=1).mean():.2f}", flush=True)
    else:
        cls, count = REGRESSOR_MODELS[ck.get("model_key", "state_token" if "state_token" in ck["state_dict"] else "full_backbone")]
        model = cls(**ck["model_cfg"]).to(device).eval()
        model.load_state_dict(ck["state_dict"])
        pred, pred_sh = run_mse(model, x, args.batch_size, device), run_mse(model, x[perm], args.batch_size, device)
        eff = benchmark(lambda: run_mse(model, xb, args.batch_size, device), n_warmup=2, n_repeats=args.bench_repeats, batch_size=len(xb))
        np.savez_compressed(out / "predictions" / "regressor.npz", pred=pred, pred_shuffled=pred_sh, perm=perm, solver="regressor", steps=1, nfe=1)
        meta["arms"]["regressor"] = {"solver": "regressor", "steps": 1, "nfe": 1, "latency_ms_batch64": eff["latency_ms_median"], "peak_mem_MiB": eff.get("peak_mem_MiB"), "params": count(model), "model": cls.__name__}
        print(f"regressor: 1 fwd latency {eff['latency_ms_median']:.0f} ms | pred mean {pred.mean():.1f} std {pred.std(axis=1).mean():.2f}", flush=True)
    (out / "predictions_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("wrote", out / "predictions")


if __name__ == "__main__":
    main()
