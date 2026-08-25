"""Smoke test of the UNMODIFIED upstream PENGUIN model on random data (no dataset, no training run).

Checks: (1) upstream imports on this torch; (2) one CFM train step (loss finite, grads flow);
(3) sampling runs; (4) our heun_sample == upstream .sample() bit-for-bit; (5) parameter count incl. dead cross_attn;
(6) latency / peak memory vs. solver steps at batch 64 — the cost axis of the pre-registered NFE curve.
Run: .venv/bin/python scripts/smoke_test_penguin.py [--full] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401  (MKL warm-up must precede `import torch`, docs/ENVIRONMENT.md)
import torch

from ppg2ecg.evaluation.efficiency import benchmark
from ppg2ecg.flow.samplers import euler_sample, heun_sample, nfe_of
from ppg2ecg.models import PENGUIN_DALIA_CFG, build_penguin_backbone, count_params
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import assert_upstream_pinned

ROOT = Path(__file__).resolve().parents[1]


def run(cfg: dict, device: str, batch: int, steps: list[int], repeats: int) -> dict:
    seed_everything(0, deterministic=False)
    model = build_penguin_backbone(**cfg).to(device)
    T = cfg["sample_rate"] * 4
    ppg = torch.randn(batch, T, device=device)
    ecg = torch.randn(batch, T, device=device)
    rep = {"cfg": cfg, "batch": batch, "T": T, "params": count_params(model, exclude_prefixes=("cross_attn",))}

    # (2) one training step exactly like upstream train.py L61+L68
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred = model(ppg, target_signal=ecg)
    loss = model.optimize(pred, ecg, opt)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    rep["train_step"] = {"loss": float(loss.detach()), "finite": bool(torch.isfinite(loss)), "time_ms": (time.perf_counter() - t0) * 1000, "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None}
    # adaLN-Zero + zero-init final linear => output == 0 at init, so step 1 only updates final_layer.linear (2 tensors).
    n_grad1 = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    pred = model(ppg, target_signal=ecg)
    loss2 = model.optimize(pred, ecg, opt)
    n_grad2 = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in model.parameters())
    rep["train_step"].update({"loss_step2": float(loss2.detach()), "n_tensors_with_nonzero_grad_step1": n_grad1, "n_tensors_with_nonzero_grad_step2": n_grad2, "n_param_tensors": n_total})
    rep["train_step"]["cross_attn_grad_is_none"] = all(p.grad is None for n, p in model.named_parameters() if "cross_attn" in n)

    # (3)+(4) sampling parity: upstream draws x0 first, then no further RNG => reseed and replay
    model.eval()
    with torch.no_grad():
        torch.manual_seed(123)
        up = model(ppg)  # upstream .sample(): n_step Heun steps
        torch.manual_seed(123)
        x0 = torch.randn(batch, 1, T).to(device)  # upstream L243 draws on CPU, then .to(device)
        v = lambda x, t: model.forward_step(x, ppg.unsqueeze(1), t)  # noqa: E731
        ours, nfe = heun_sample(v, x0, cfg["n_step"])
    rep["sampling_parity"] = {"max_abs_diff": float((up - ours.squeeze(1)).abs().max()), "bit_exact": bool(torch.equal(up, ours.squeeze(1))), "nfe": nfe, "expected_nfe": nfe_of("heun", cfg["n_step"])}

    # (6) cost curve on random weights (latency only — quality needs a trained model)
    curve = []
    for solver, fn in (("heun", heun_sample), ("euler", euler_sample)):
        for s in steps:
            res = benchmark(lambda: fn(v, x0, s), n_warmup=2, n_repeats=repeats, batch_size=batch, device=device)
            res.update({"solver": solver, "steps": s, "nfe": nfe_of(solver, s)})
            curve.append(res)
    rep["cost_curve"] = curve
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--full", action="store_true", help="also run the shipped DaLiA config (h_dim=128) at batch 64")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    out = {"timestamp": datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__, "device": args.device, "upstream": assert_upstream_pinned()}
    tiny = dict(PENGUIN_DALIA_CFG, h_dim=16, ssm_block_num=2, n_step=2)
    print("[tiny]", tiny)
    out["tiny"] = run(tiny, args.device, batch=8, steps=[1, 2], repeats=2)
    print(json.dumps({k: out["tiny"][k] for k in ("params", "train_step", "sampling_parity")}, indent=1))
    if args.full:
        print("[full]", PENGUIN_DALIA_CFG)
        out["full"] = run(dict(PENGUIN_DALIA_CFG), args.device, batch=64, steps=[1, 2, 5, 10, 25], repeats=args.repeats)
        print(json.dumps({k: out["full"][k] for k in ("params", "train_step", "sampling_parity")}, indent=1))
        for r in out["full"]["cost_curve"]:
            print(f"  {r['solver']:5s} steps={r['steps']:2d} NFE={r['nfe']:2d}  {r['latency_ms_median']:8.1f} ms/batch64  {r['samples_per_s']:8.1f} samp/s  peak {r.get('peak_mem_MiB', 0):7.1f} MiB")
    dst = ROOT / "outputs" / "smoke"
    dst.mkdir(parents=True, exist_ok=True)
    p = dst / f"penguin_smoke_{out['timestamp'].replace(':', '')}.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
