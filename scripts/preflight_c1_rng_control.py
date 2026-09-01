"""C1 RNG-control preflight (preregistration sections 6 and 15). Runs BEFORE any training.

Replicates the trainer's RNG setup exactly for each arm and asserts that the three arms share the model
initialisation, the dataloader window order, the Gaussian noise stream and the validation banks, and
differ ONLY in the (t, r, h) stream. A shared global seed is not accepted as evidence of pairing.

No weight update. No optimiser step. Test subjects are never loaded.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.evaluation.event_reliability import assert_no_test_subjects
from ppg2ecg.flow.imeanflow import MeanFlowS5, imf_bank_hash, make_imf_banks
from ppg2ecg.flow.interval_exposure import ARMS, exposure_stats, sample_tr_c1
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.training.train_a0 import load_arrays
from ppg2ecg.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/c1_interval_exposure"
SEED, BATCH, MICRO, N_ROUNDS = 42, 64, 32, 40
TR_KW = dict(p_mean=-0.4, p_std=1.0, data_proportion=0.5)
MC_DRAWS, MC_SEED = 2_000_000, 20260901


def h_of(*arrays) -> str:
    m = hashlib.sha256()
    for a in arrays:
        x = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
        m.update(np.ascontiguousarray(x.astype("float64")).tobytes())
    return m.hexdigest()


def simulate(arm: str, x_tr, y_tr, n_val: int, T: int) -> dict:
    """Reproduce the trainer's stream setup for one arm, without a single weight update."""
    seed_everything(SEED, deterministic=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = build_penguin_backbone(n_step=1, sample_rate=128, h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0)
    net = MeanFlowS5(bb, cond_mode="h_only", h_scale=1.0).to(dev)
    init_hash = h_of(*[p for _, p in sorted(net.state_dict().items())])
    banks_hash = imf_bank_hash(make_imf_banks(n_val, T, 4, 1000, **TR_KW))

    gen = torch.Generator(); gen.manual_seed(SEED)
    tr_gen = torch.Generator(); tr_gen.manual_seed(SEED + 1)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                        batch_size=BATCH, shuffle=True, generator=gen)

    order, noise, tr_vals, hs = [], [], [], []
    it = iter(loader)
    for _ in range(N_ROUNDS):
        ppg, ecg = next(it)
        order.append(ppg[:, :4].clone())                       # window identity proxy, order-sensitive
        for i0 in range(0, len(ppg), MICRO):
            bc = len(ppg[i0:i0 + MICRO])
            t, r, _ = sample_tr_c1(bc, tr_gen, arm=arm, **TR_KW)
            e = torch.randn(bc, 1, ecg.shape[1], device=dev)    # global CUDA stream, as in the trainer
            tr_vals.append(torch.cat([t, r], 1).clone())
            hs.append((t - r).clone())
            noise.append(e[:, :, :4].detach().cpu().clone())
    return {"arm": arm, "init_hash": init_hash, "banks_hash": banks_hash,
            "order_hash": h_of(*order), "noise_hash": h_of(*noise), "tr_hash": h_of(*tr_vals),
            "mean_h": float(torch.cat(hs).mean()), "p_h0": float((torch.cat(hs) == 0).double().mean())}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    split = read_manifest(ROOT / "data/manifests/split_a4_wildppg_seed42.json")[0]
    assert_no_test_subjects(split["train"] + split["val"])
    print(f"[P] train {len(split['train'])} subjects, val {split['val']}", flush=True)
    x_tr, y_tr, _ = load_arrays(ROOT / "data/processed/wildppg_8s", split["train"], None)
    x_va, _, _ = load_arrays(ROOT / "data/processed/wildppg_8s", split["val"], None)
    stride = -(-len(x_va) // 4096)
    n_val, T = len(x_va[::stride]), x_tr.shape[1]
    print(f"[P] train windows {len(x_tr)} | val windows {n_val} | T {T}", flush=True)

    rows = [simulate(a, x_tr, y_tr, n_val, T) for a in ARMS]
    for r in rows:
        print(f"[R] {r['arm']:4s} init {r['init_hash'][:16]} order {r['order_hash'][:16]} "
              f"noise {r['noise_hash'][:16]} banks {r['banks_hash'][:16]} tr {r['tr_hash'][:16]} "
              f"| mean h {r['mean_h']:.4f} P(h=0) {r['p_h0']:.4f}", flush=True)

    ref = rows[0]
    checks = {}
    for k in ("init_hash", "order_hash", "noise_hash", "banks_hash"):
        same = all(r[k] == ref[k] for r in rows)
        checks[f"identical_{k}"] = same
        print(f"[C] {k:12s} identical across arms: {same}", flush=True)
    checks["tr_hash_B_differs_from_H25"] = rows[0]["tr_hash"] != rows[1]["tr_hash"]
    checks["tr_hash_B_differs_from_H50"] = rows[0]["tr_hash"] != rows[2]["tr_hash"]
    checks["tr_hash_H25_differs_from_H50"] = rows[1]["tr_hash"] != rows[2]["tr_hash"]
    for k in ("tr_hash_B_differs_from_H25", "tr_hash_B_differs_from_H50", "tr_hash_H25_differs_from_H50"):
        print(f"[C] {k}: {checks[k]}", flush=True)

    exp_rows = []
    for a in ARMS:
        g = torch.Generator().manual_seed(MC_SEED)
        t, r, _ = sample_tr_c1(MC_DRAWS, g, arm=a, **TR_KW)
        s = exposure_stats(t - r)
        exp_rows.append({"arm": a, **s})
        print(f"[E] {a:4s} " + "  ".join(f"{k}={v:.4f}" for k, v in s.items()), flush=True)
    import csv
    with open(OUT / "sampler_exposure.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(exp_rows[0])); w.writeheader(); w.writerows(exp_rows)

    # preregistered exposure expectations (section 3 / 14)
    e = {r["arm"]: r for r in exp_rows}
    checks["p_h0_preserved"] = all(abs(e[a]["p_h_eq_0"] - 0.5) < 1e-6 for a in ARMS)
    checks["h25_forced_mass"] = abs(e["H25"]["p_h_eq_0.25"] - 0.25) < 1e-6
    checks["h50_forced_mass"] = abs(e["H50"]["p_h_eq_0.50"] - 0.25) < 1e-6
    checks["h50_raises_p_ge_0.5"] = e["H50"]["p_h_ge_0.5"] > e["B"]["p_h_ge_0.5"] + 0.15
    checks["baseline_matches_record"] = (abs(e["B"]["p_h_eq_0"] - 0.50) < 0.01 and
                                         abs(e["B"]["positive_median"] - 0.201) < 0.01 and
                                         abs(e["B"]["p_h_ge_0.5"] - 0.042) < 0.005)
    ok = all(checks.values())
    (OUT / "rng_control.json").write_text(json.dumps({"rows": rows, "checks": checks, "pass": ok,
                                                      "n_rounds_simulated": N_ROUNDS}, indent=2))
    print(f"\n[GATE] RNG control + exposure: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("[GATE] failing checks: " + ", ".join(k for k, v in checks.items() if not v), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
