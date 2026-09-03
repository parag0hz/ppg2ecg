"""O1 — train the component probes (preregistration sections 5, 7, 8).

PPG-only regression probes, one per (target, seed). ECG never enters the probe input; ECG-derived scalars are
training labels only. THIS SCRIPT NEVER READS A VALIDATION ROW: it builds the cohort for the probe_train and
internal_dev roles only, and asserts that an0/k2s (and the test subjects) are absent.
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
import torch.nn as nn

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.training.train_a0 import git_sha
from ppg2ecg.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o1_component_extractability"
LOGS = ART / "probe_training_logs"
SPLIT = C.internal_dev_split()
PROBE_TRAIN, INTERNAL_DEV = SPLIT["probe_train"], SPLIT["internal_dev"]
TRAIN_ROLES = ("probe_train", "internal_dev")


def load_targets():
    z = np.load(ART / "_cache_targets.npz", allow_pickle=False)
    key = {(str(s), int(w)): i for i, (s, w) in enumerate(zip(z["subjects"], z["window_index"]))}
    return z["targets"], key, [str(n) for n in z["names"]]


def load_role(role: str, T, key):
    """PPG rows + per-row targets for one role. Validation subjects are never requested here."""
    subs = PROBE_TRAIN if role == "probe_train" else INTERNAL_DEV
    ER.assert_no_test_subjects(subs)
    assert not (set(subs) & set(C.VAL)), "training must not touch validation subjects"
    X, Y, SUB, SITE, WI = [], [], [], [], []
    for s in subs:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs, SITEs, WIs = d["x"], np.asarray(d["site"]).astype(str), d["window_index"].astype(np.int64)
        pos = C.cohort_positions(s, SITEs, WIs, C.n_per_for(s))
        for site in C.SITES:
            idx = pos[site]
            X.append(Xs[idx].astype(np.float32))
            Y.append(np.stack([T[key[(s, int(WIs[p]))]] for p in idx]))
            SUB.append(np.full(len(idx), s)); SITE.append(np.full(len(idx), site)); WI.append(WIs[idx])
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(SUB), np.concatenate(SITE), np.concatenate(WI))


def train_one(target: str, seed: int, Xtr, ytr, Xdv, ydv, dev, scaling, max_steps: int | None = None):
    j = OT.TARGETS.index(target)
    c, s_ = scaling[target]["center_train_median"], scaling[target]["scale_train_IQR"]
    ytr_z = torch.from_numpy(((ytr[:, j] - c) / s_).astype(np.float32)).to(dev)
    ydv_z = torch.from_numpy(((ydv[:, j] - c) / s_).astype(np.float32)).to(dev)
    ok_tr = torch.isfinite(ytr_z); ok_dv = torch.isfinite(ydv_z)
    Xtr_t = torch.from_numpy(Xtr).to(dev).unsqueeze(1)[ok_tr]
    ytr_z = ytr_z[ok_tr]
    Xdv_t = torch.from_numpy(Xdv).to(dev).unsqueeze(1)[ok_dv]
    ydv_z = ydv_z[ok_dv]
    seed_everything(int(seed), deterministic=True)
    torch.manual_seed(int(seed))
    net = OT.build_probe(seed).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=OT.LR, weight_decay=OT.WEIGHT_DECAY)
    lossf = nn.SmoothL1Loss(beta=OT.HUBER_BETA)
    g = torch.Generator().manual_seed(int(seed))
    n, steps_per_epoch = len(Xtr_t), int(np.ceil(len(Xtr_t) / OT.BATCH))
    best, best_ep, bad, log, step = float("inf"), -1, 0, [], 0
    best_state = None
    t0 = time.perf_counter()
    for ep in range(OT.MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for i in range(0, n, OT.BATCH):
            b = perm[i:i + OT.BATCH]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xtr_t[b]), ytr_z[b])
            loss.backward(); opt.step()
            tot += float(loss) * len(b); step += 1
            if max_steps is not None and step >= max_steps:
                return {"steps": step, "wall_s": time.perf_counter() - t0,
                        "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20)}
        net.eval()
        with torch.no_grad():
            pr = torch.cat([net(Xdv_t[i:i + 512]) for i in range(0, len(Xdv_t), 512)])
            dev_mae = float((pr - ydv_z).abs().mean())
        log.append({"target": target, "seed": seed, "epoch": ep, "train_loss": tot / n, "dev_std_mae": dev_mae,
                    "steps": step, "wall_s": time.perf_counter() - t0})
        if dev_mae < best - 1e-6:
            best, best_ep, bad = dev_mae, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= OT.PATIENCE:
                break
    return {"target": target, "seed": seed, "best_dev_std_mae": best, "best_epoch": best_ep,
            "epochs_run": len(log), "steps": step, "wall_s": time.perf_counter() - t0,
            "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20), "log": log, "state": best_state}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="100 optimizer steps on T2, discard the state")
    ap.add_argument("--seeds", default=",".join(str(s) for s in OT.SEEDS))
    ap.add_argument("--targets", default=",".join(OT.TARGETS))
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    T, key, names = load_targets()
    assert names == list(OT.TARGETS)
    scaling = json.loads((ART / "target_scaling.json").read_text())["targets"]
    Xtr, ytr, SUBtr, SITEtr, WItr = load_role("probe_train", T, key)
    Xdv, ydv, SUBdv, SITEdv, WIdv = load_role("internal_dev", T, key)
    print(f"[data] train {Xtr.shape} dev {Xdv.shape} | subjects train {sorted(set(SUBtr))} dev {sorted(set(SUBdv))}", flush=True)

    if args.preflight:
        torch.cuda.reset_peak_memory_stats()
        r = train_one("median_RR_ms", 42, Xtr, ytr, Xdv, ydv, dev, scaling, max_steps=OT.PREFLIGHT_STEPS)
        per_step = r["wall_s"] / r["steps"]
        steps_per_epoch = int(np.ceil(len(Xtr) / OT.BATCH))
        worst = per_step * steps_per_epoch * OT.MAX_EPOCHS * len(OT.TARGETS) * len(OT.SEEDS)
        out = {"utc": datetime.now(timezone.utc).isoformat(), "git": git_sha(ROOT), "steps": r["steps"],
               "wall_s": r["wall_s"], "s_per_step": per_step, "peak_mem_MiB": r["peak_mem_MiB"],
               "steps_per_epoch": steps_per_epoch, "n_targets": len(OT.TARGETS), "n_seeds": len(OT.SEEDS),
               "projected_worst_case_gpu_hours": worst / 3600.0, "budget_gpu_hours": OT.BUDGET_GPU_HOURS,
               "stop": bool(worst / 3600.0 > OT.BUDGET_GPU_HOURS), "note": "preflight state discarded"}
        (ART / "runtime_preflight.json").write_text(json.dumps(out, indent=2, default=str))
        print(json.dumps({k: out[k] for k in ("s_per_step", "peak_mem_MiB", "projected_worst_case_gpu_hours", "stop")}, indent=2))
        if out["stop"]:
            raise RuntimeError(f"projected {out['projected_worst_case_gpu_hours']:.2f} GPU-h exceeds the frozen budget (STOP)")
        return 0

    seeds = [int(s) for s in args.seeds.split(",")]
    targets = [t for t in args.targets.split(",")]
    man, t_all = [], time.perf_counter()
    for seed in seeds:
        for target in targets:
            torch.cuda.reset_peak_memory_stats()
            r = train_one(target, seed, Xtr, ytr, Xdv, ydv, dev, scaling)
            out_dir = ROOT / f"outputs/o1_{target}_seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            ck = {"target": target, "seed": seed, "state_dict": r["state"], "best_epoch": r["best_epoch"],
                  "best_dev_std_mae": r["best_dev_std_mae"], "scaling": scaling[target],
                  "arch": {"dilations": list(OT.DILATIONS), "ch": OT.CH, "k": OT.K, "params": OT.n_params(OT.build_probe(seed))},
                  "git": git_sha(ROOT), "selection": "internal_dev standardized MAE"}
            torch.save(ck, out_dir / "checkpoint_best.pt")
            sd_sha = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for _k, v in sorted(r["state"].items()))).hexdigest()
            with open(LOGS / f"{target}_seed{seed}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(r["log"][0])); w.writeheader(); w.writerows(r["log"])
            man.append({"target": target, "id": OT.TARGET_IDS[target], "seed": seed, "epochs_run": r["epochs_run"],
                        "best_epoch": r["best_epoch"], "best_dev_std_mae": r["best_dev_std_mae"], "steps": r["steps"],
                        "wall_s": r["wall_s"], "peak_mem_MiB": r["peak_mem_MiB"], "params": ck["arch"]["params"],
                        "state_dict_sha256": sd_sha, "checkpoint": str(out_dir / "checkpoint_best.pt")})
            print(f"[{target:<30} s{seed}] ep {r['epochs_run']:2d} best {r['best_epoch']:2d} "
                  f"dev {r['best_dev_std_mae']:.4f} | {r['wall_s']:.0f}s", flush=True)
    with open(ART / "probe_training_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(man[0])); w.writeheader(); w.writerows(man)
    (ART / "probe_training_provenance.json").write_text(json.dumps(
        {"git": git_sha(ROOT), "utc": datetime.now(timezone.utc).isoformat(), "test_subjects_loaded": [],
         "validation_subjects_loaded": [], "seeds": seeds, "targets": targets,
         "hyperparameters": {"lr": OT.LR, "weight_decay": OT.WEIGHT_DECAY, "batch": OT.BATCH, "loss": "SmoothL1",
                             "huber_beta": OT.HUBER_BETA, "max_epochs": OT.MAX_EPOCHS, "patience": OT.PATIENCE,
                             "selection": "internal_dev standardized MAE"},
         "n_train_rows": int(len(Xtr)), "n_dev_rows": int(len(Xdv)),
         "wall_s": time.perf_counter() - t_all, "gpu": torch.cuda.get_device_name(0),
         "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()}}, indent=2, default=str))
    print(f"[done] {len(man)} probes in {(time.perf_counter()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
