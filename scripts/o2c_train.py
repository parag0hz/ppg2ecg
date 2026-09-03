"""O2c — train O2C-CANON-ORACLE for exactly the compute-matched step count (preregistration sections 7-11).

The ONLY difference from the C1 arm-B replay is that the paired PPG and ECG are event-canonicalized with the
ACCEPTED O2b integer-grid operator before entering the unchanged iMeanFlow objective. Validation subjects are
never loaded here; there is no validation scoring, no checkpoint selection and no early stopping.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o2_warp as O2W
from ppg2ecg.evaluation import o2b_warp as BW
from ppg2ecg.flow.imeanflow import MeanFlowS5, imeanflow_loss
from ppg2ecg.flow.interval_exposure import sample_tr_c1
from ppg2ecg.models import build_penguin_backbone, count_params
from ppg2ecg.training.train_a0 import git_sha, load_arrays, read_manifest
from ppg2ecg.training.train_a2 import batch_rounds
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import assert_upstream_pinned

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2c_oracle_integer_grid"
OUT = ROOT / "outputs/o2c_canon_oracle_seed42"
CACHE = ART / "_cache_train_rpeaks.npz"
MANIFEST = "data/manifests/split_a4_wildppg_seed42.json"
PROCESSED = "data/processed/wildppg_8s"
# resolved from the C1 arm-B replay provenance (train_meta.json args) — never guessed
CFG = dict(seed=42, batch_size=64, micro_batch=32, lr=1e-3, weight_decay=0.01, val_every_steps=220,
           p_mean=-0.4, p_std=1.0, data_proportion=0.5, norm_p=1.0, norm_eps=0.01, jvp_mode="forward",
           cond_mode="h_only", h_scale=1.0, sample_rate=128, h_dim=128, blocks=4, ssm_ratio=2.0, mlp_ratio=2.0,
           c1_arm="B")


def state_sha(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def canonicalize(x, y, pks, dev, batch=512):
    """Warp BOTH modalities with the SAME accepted integer-grid map; identity rows stay bit-exact."""
    xo = torch.empty_like(torch.from_numpy(x)); yo = torch.empty_like(torch.from_numpy(y))
    n_ident = 0
    for i in range(0, len(x), batch):
        wl = [BW.IntegerEventWarp(p) for p in pks[i:i + batch]]
        n_ident += sum(w.identity for w in wl)
        xb = torch.from_numpy(x[i:i + batch]).to(dev).unsqueeze(1)
        yb = torch.from_numpy(y[i:i + batch]).to(dev).unsqueeze(1)
        xo[i:i + batch] = O2W.apply_warp(xb, wl, "to_canonical").squeeze(1).cpu()
        yo[i:i + batch] = O2W.apply_warp(yb, wl, "to_canonical").squeeze(1).cpu()
    return xo.numpy(), yo.numpy(), n_ident


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="resolved from the frozen artifact when omitted")
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    up = assert_upstream_pinned()
    steps_target = args.steps or int(json.loads((ART / "baseline_step_resolution.json").read_text())["exact_optimizer_steps_at_best_round"])
    if not args.preflight and steps_target != 10046:
        raise RuntimeError(f"compute-matched step count must be 10046, got {steps_target}")
    dev = torch.device("cuda")

    split = read_manifest(ROOT / MANIFEST)[0]
    ER.assert_no_test_subjects(split["train"])
    assert not (set(split["train"]) & {"an0", "k2s"})
    x, y, _ = load_arrays(ROOT / PROCESSED, split["train"], None)
    z = np.load(CACHE)
    flat, off = z["flat"], z["offsets"]
    pks = [flat[off[i]:off[i + 1]].astype(np.int64) for i in range(len(off) - 1)]
    assert len(pks) == len(x), "cached R schedule does not match the corpus"
    t0 = time.perf_counter()
    xc, yc, n_ident = canonicalize(x, y, pks, dev)
    print(f"[canon] {len(xc)} windows canonicalized in {time.perf_counter()-t0:.0f} s | identity rows {n_ident}", flush=True)

    seed_everything(CFG["seed"], deterministic=True)                    # exactly the C1 seeding path
    backbone = build_penguin_backbone(n_step=1, sample_rate=CFG["sample_rate"], h_dim=CFG["h_dim"],
                                      ssm_block_num=CFG["blocks"], ssm_ratio=CFG["ssm_ratio"], mlp_ratio=CFG["mlp_ratio"])
    net = MeanFlowS5(backbone, cond_mode=CFG["cond_mode"], h_scale=CFG["h_scale"]).to(dev)
    params = count_params(backbone, exclude_prefixes=("cross_attn", "revin"))
    init_sha = state_sha(net.state_dict())
    if params["total"] != 4_568_707:
        raise RuntimeError(f"parameter count {params['total']} != frozen B 4568707")
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    gen = torch.Generator(); gen.manual_seed(CFG["seed"])
    tr_gen = torch.Generator(); tr_gen.manual_seed(CFG["seed"] + 1)
    xt, yt = torch.from_numpy(xc).to(dev), torch.from_numpy(yc).to(dev)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=CFG["batch_size"], shuffle=True, generator=gen)
    tr_kw = dict(p_mean=CFG["p_mean"], p_std=CFG["p_std"], data_proportion=CFG["data_proportion"])
    if not args.preflight:
        (ART / "initialization_hash.json").write_text(json.dumps(
            {"seed": CFG["seed"], "constructor": "build_penguin_backbone(n_step=1, ...) + MeanFlowS5(h_only, 1.0)",
             "state_dict_sha256": init_sha, "params": params,
             "historical_c1_init_hash_available": False,
             "note": "no historical C1 initialization hash exists, so exact historical initialization identity "
                     "could not be independently verified; the constructor, seed and initial hash are recorded instead"},
            indent=2))

    torch.cuda.reset_peak_memory_stats()
    budget = 100 if args.preflight else steps_target
    rounds = batch_rounds(loader, CFG["val_every_steps"])
    log, step, t_train = [], 0, time.perf_counter()
    net.train()
    while step < budget:
        for ppg, ecg in next(rounds):
            B = len(ppg)
            opt.zero_grad()
            acc = {"loss": 0.0, "mse": 0.0, "u": 0.0, "d": 0.0}
            for i0 in range(0, B, CFG["micro_batch"]):
                ppg_c, ecg_c = ppg[i0:i0 + CFG["micro_batch"]], ecg[i0:i0 + CFG["micro_batch"]]
                Bc = len(ppg_c)
                t, r, _ = sample_tr_c1(Bc, tr_gen, arm=CFG["c1_arm"], **tr_kw)
                t, r = t.to(dev), r.to(dev)
                e = torch.randn(Bc, 1, ecg_c.shape[1], device=dev)      # source drawn IN canonical coordinates
                loss, info = imeanflow_loss(net, ecg_c.unsqueeze(1), ppg_c.unsqueeze(1), e, t, r,
                                            norm_p=CFG["norm_p"], norm_eps=CFG["norm_eps"], jvp_mode=CFG["jvp_mode"])
                if not (torch.isfinite(loss) and torch.isfinite(info["mse"]) and torch.isfinite(info["dudt_abs_mean"])):
                    raise RuntimeError(f"non-finite loss at step {step}")
                (loss * (Bc / B)).backward()
                acc["loss"] += loss.item() * Bc / B; acc["mse"] += float(info["mse"]) * Bc / B
                acc["u"] += float(info["u_abs_mean"]) * Bc / B; acc["d"] += float(info["dudt_abs_mean"]) * Bc / B
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                log.append({"step": step, "loss_weighted": acc["loss"], "mse": acc["mse"], "u_abs": acc["u"],
                            "dudt_abs": acc["d"], "elapsed_s": time.perf_counter() - t_train})
            if step >= budget:
                break
    wall = time.perf_counter() - t_train
    peak = float(torch.cuda.max_memory_allocated() / 2 ** 20)

    if args.preflight:
        per = wall / step
        proj_h = per * steps_target / 3600.0
        out = {"utc": datetime.now(timezone.utc).isoformat(), "git": git_sha(ROOT), "steps": step, "wall_s": wall,
               "s_per_step": per, "ms_per_step": per * 1000, "peak_mem_MiB": peak,
               "target_steps": steps_target, "projected_gpu_hours": proj_h, "budget_gpu_hours": 3.0,
               "stop": bool(proj_h > 3.0), "note": "preflight state discarded"}
        (ART / "gpu_preflight.json").write_text(json.dumps(out, indent=2, default=str))
        print(json.dumps({k: out[k] for k in ("ms_per_step", "peak_mem_MiB", "projected_gpu_hours", "stop")}, indent=2))
        if out["stop"]:
            raise RuntimeError(f"projected {proj_h:.2f} GPU-h exceeds the frozen 3.0 h budget (STOP)")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    final_sd = net.state_dict()
    final_sha = state_sha(final_sd)
    torch.save({"step": step, "state_dict": final_sd, "state_dict_sha256": final_sha, "model_cfg": dict(n_step=1, sample_rate=CFG["sample_rate"],
                h_dim=CFG["h_dim"], ssm_block_num=CFG["blocks"], ssm_ratio=CFG["ssm_ratio"], mlp_ratio=CFG["mlp_ratio"]),
                "imf_cfg": {"cond_mode": CFG["cond_mode"], "h_scale": CFG["h_scale"]}, "cfg": CFG,
                "coordinate": "o2b integer-grid canonical", "git": git_sha(ROOT)}, OUT / "checkpoint_final.pt")
    with open(ART / "training_log.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(log[0])); w_.writeheader(); w_.writerows(log)
    man = {"git": git_sha(ROOT), "prereg": "d458895", "utc_start": datetime.now(timezone.utc).isoformat(),
           "upstream": up, "steps": step, "target_steps": steps_target, "config": CFG,
           "params": params, "init_state_dict_sha256": init_sha,
           "final_state_dict_sha256": final_sha,
           "checkpoint_file_sha256": hashlib.sha256((OUT / "checkpoint_final.pt").read_bytes()).hexdigest(),
           "train_windows": int(len(xc)), "identity_rows": int(n_ident), "subjects": list(split["train"]),
           "validation_loaded": False, "validation_selection": False, "early_stopping": False,
           "coordinate": "o2b integer-grid canonical (operator imported unchanged)",
           "wall_s": wall, "peak_mem_MiB": peak, "gpu": torch.cuda.get_device_name(0),
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
           "total_wall_s": time.perf_counter() - t_all}
    (ART / "training_manifest.json").write_text(json.dumps(man, indent=2, default=str))
    (ART / "checkpoint_manifest.json").write_text(json.dumps(
        {"final": str(OUT / "checkpoint_final.pt"), "state_dict_sha256": man["final_state_dict_sha256"],
         "file_sha256": man["checkpoint_file_sha256"], "steps": step, "frozen_before_validation": True}, indent=2))
    print(f"[done] {step} steps in {wall/60:.1f} min ({wall/3600:.3f} GPU-h), peak {peak:.0f} MiB", flush=True)
    print(f"[ckpt] final state sha {man['final_state_dict_sha256'][:16]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
