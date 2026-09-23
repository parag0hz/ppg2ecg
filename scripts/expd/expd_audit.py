"""EXP-D functional generalisation — PHASE 0 audit (no result numbers).

For every upstream-as-shipped PENGUIN checkpoint with a respiration or ABP target (U1: BIDMC, WESAD, MIMIC-BP, UCI-BP):
checkpoint hash / epoch / embedded training cfg; strict re-load (upstream's own `load_checkpoint` uses strict=False, so the
incompatible-key lists are recorded here); split sizes, subject disjointness, windows per test subject and metric-block
structure; target units (from TRAIN subjects only); upstream-sampler equivalence (our `heun_sample` vs upstream
`PENGUIN.sample`, same noise); vector-field NFE by forward hook; latency; sampler smoke test (finite, same seed ->
identical, different seed -> different). Inputs for every model call are TRAIN-subject PPG windows; no generated sample
is compared with any target.

Writes artifacts/exp_d_functional_generalization/{respiration,abp}/audit.json.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/expd/expd_audit.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "external/PENGUIN"
sys.path.insert(0, str(UP / "src"))
from utils.help_func import initialize_model  # noqa: E402  (upstream, unmodified)

from ppg2ecg.flow.samplers import euler_sample, heun_sample  # noqa: E402

U1 = ROOT / "outputs/u1_upstream"
PROC = ROOT / "data/processed/upstream_u1"
OUT = ROOT / "artifacts/exp_d_functional_generalization"
TASK = {"BIDMC": "respiration", "WESAD": "respiration", "MIMIC-BP": "abp", "UCI-BP": "abp"}
BLOCK_S = {"respiration": 60, "abp": 8}
SS = (1, 2, 4, 8, 16, 32)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_split(ds):
    return json.loads((U1 / f"split_{ds}.json").read_text())


def windows(ds, files):
    x, y = [], []
    for f in files:
        with open(PROC / ds / f, "rb") as fh:
            d = pickle.load(fh)
        x.append(np.asarray(d["x_data"], np.float32)); y.append(np.asarray(d["y_data"], np.float32))
    return x, y


def audit(ds, dev):
    task = TASK[ds]
    ck_path = U1 / f"PENGUIN_{ds}_u1/ckpt/pretrain_ckpt.pth"
    ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    dcfg = getattr(cfg.preprocess, ds)
    model = initialize_model(cfg, device=dev)
    r = model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    npar = int(sum(p.numel() for p in model.parameters()))
    sp = load_split(ds)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    seg = int(cfg.preprocess.segment_len); fs = int(cfg.preprocess.resample_rate)
    per_block = BLOCK_S[task] // seg
    wc = {f: sp["window_counts"][f][0] for f in te}
    res = {
        "dataset": ds, "task": task, "checkpoint": str(ck_path.relative_to(ROOT)), "sha256": sha256(ck_path),
        "size_bytes": ck_path.stat().st_size, "epoch_saved": int(ck["epoch"]), "checkpoint_keys": sorted(ck.keys()),
        "cfg": {"train.dataset": str(cfg.train.dataset), "train.model": str(cfg.train.model), "seed": int(cfg.seed),
                "segment_len_s": seg, "resample_rate_hz": fs, "ppg_fs_raw": float(dcfg.ppg_fs), "label": str(dcfg.label),
                "label_fs_raw": float(dcfg.label_fs), "label_bandpass": bool(dcfg.label_bandpass),
                "label_freq_range": list(dcfg.label_freq_range), "label_zscore": bool(dcfg.label_zscore),
                "label_normalize": bool(dcfg.label_normalize), "ppg_bandpass": bool(cfg.preprocess.ppg_bandpass),
                "ppg_freq_range": list(cfg.preprocess.ppg_freq_range), "ppg_zscore": bool(cfg.preprocess.ppg_zscore),
                "ppg_normalize": bool(cfg.preprocess.ppg_normalize), "n_step": int(cfg.models.PENGUIN.n_step),
                "h_dim": int(cfg.models.PENGUIN.h_dim), "fold_num": int(cfg.train.training.fold_num),
                "epoch_num_cap": int(cfg.train.training.epoch_num), "ema_decay_in_cfg": float(cfg.train.training.ema_decay),
                "metric_window_s": {k: int(v) for k, v in cfg.train.task_specific_metrics.window_size.items()}},
        "strict_load": {"missing": list(r.missing_keys), "unexpected": list(r.unexpected_keys)},
        "parameters": npar,
        "split": {"train_subjects": len(tr), "val_subjects": len(va), "test_subjects": len(te), "n_windows": sp["n_windows"],
                  "subject_disjoint": not (set(tr) & set(te) or set(va) & set(te) or set(tr) & set(va)),
                  "duplicate_groups": sp["duplicate_groups"]},
        "test_windows_per_subject": {"min": min(wc.values()), "max": max(wc.values()), "total": sum(wc.values())},
        "metric_block": {"seconds": BLOCK_S[task], "windows_per_block": per_block,
                         "all_test_subjects_divisible": all(v % per_block == 0 for v in wc.values()),
                         "test_blocks": int(sum(v // per_block for v in wc.values())),
                         "blocks_straddling_subjects_in_upstream_stream": int(sum(v % per_block != 0 for v in wc.values()))},
    }
    # target units from TRAIN subjects only
    _, ytr = windows(ds, tr[:30])
    y = np.concatenate(ytr)
    res["train_target_units_check"] = {"n_train_subjects_read": min(30, len(tr)), "min": float(y.min()), "max": float(y.max()),
                                       "mean": float(y.mean()), "per_window_max_median": float(np.median(y.max(1))),
                                       "per_window_min_median": float(np.median(y.min(1))),
                                       "per_window_minmax_is_pm1_share": float(np.mean((np.abs(y.max(1) - 1) < 1e-6) & (np.abs(y.min(1) + 1) < 1e-6)))}
    # sampler checks on TRAIN-subject PPG
    xtr, _ = windows(ds, tr[:1])
    ppg = torch.from_numpy(xtr[0][:16]).to(dev)
    calls = {"n": 0}
    model.final_layer.register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
    v = lambda xt, t: model.forward_step(xt, ppg.unsqueeze(1), t)  # noqa: E731
    with torch.no_grad():
        torch.manual_seed(123)
        up = model.sample(ppg).float().cpu().numpy()                       # upstream sampler, n_step from cfg
        n_up = calls["n"]
        torch.manual_seed(123)
        z = torch.randn(len(ppg), 1, ppg.shape[1]).to(dev)
        calls["n"] = 0
        ours, k = heun_sample(v, z, int(cfg.models.PENGUIN.n_step))
        res["upstream_sampler"] = {"n_step": int(cfg.models.PENGUIN.n_step), "vector_field_calls_upstream": n_up,
                                   "vector_field_calls_ours": calls["n"], "our_heun_equals_upstream_max_abs_diff":
                                   float(np.abs(ours[:, 0].float().cpu().numpy() - up).max())}
        nfe, lat, smoke = {}, {"gpu_batch1_ms": {}, "gpu_batch512_per_window_ms": {}}, {}
        g = torch.Generator().manual_seed(0)
        zb = torch.randn(512, 1, ppg.shape[1], generator=g).to(dev)
        pb = torch.from_numpy(np.concatenate(xtr)[:512] if len(xtr[0]) >= 512 else np.resize(xtr[0], (512, xtr[0].shape[1]))).to(dev)
        vb = lambda xt, t: model.forward_step(xt, pb.unsqueeze(1), t)  # noqa: E731
        v1 = lambda xt, t: model.forward_step(xt, ppg[:1].unsqueeze(1), t)  # noqa: E731
        for S in SS:
            calls["n"] = 0
            a, _ = euler_sample(v, z, S)
            nfe[str(S)] = calls["n"]
            a2, _ = euler_sample(v, z, S)
            b, _ = euler_sample(v, torch.randn(z.shape, generator=torch.Generator().manual_seed(7)).to(dev), S)
            a, a2, b = (q[:, 0].float().cpu().numpy() for q in (a, a2, b))
            smoke[str(S)] = {"finite": bool(np.isfinite(a).all()), "same_noise_max_abs_diff": float(np.abs(a - a2).max()),
                             "different_noise_pairwise_rms_median": float(np.median(np.sqrt(((a - b) ** 2).mean(1)))),
                             "output_range": [float(a.min()), float(a.max())]}
            for key, fn, zz, reps in (("gpu_batch1_ms", v1, z[:1], 20), ("gpu_batch512_per_window_ms", vb, zb, 5)):
                ts = []
                for i in range(reps + 1):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    euler_sample(fn, zz, S)
                    torch.cuda.synchronize()
                    if i:
                        ts.append(1000 * (time.perf_counter() - t0))
                lat[key][str(S)] = float(np.median(ts)) / (512 if "512" in key else 1)
        ts = []
        for i in range(11):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            heun_sample(v1, z[:1], 25)
            torch.cuda.synchronize()
            if i:
                ts.append(1000 * (time.perf_counter() - t0))
        lat["gpu_batch1_ms"]["heun25_50nfe"] = float(np.median(ts))
    res["nfe_euler_by_S"] = nfe
    res["nfe_heun25"] = res["upstream_sampler"]["vector_field_calls_ours"]
    res["latency"] = lat
    res["smoke_train_inputs"] = smoke
    res["fixed_overhead"] = "none: PENGUIN has no encoder / decoder outside the vector field; the PPG pre-convolution runs inside every vector-field call"
    del model; torch.cuda.empty_cache()
    return res


def main():
    dev = torch.device("cuda")
    up_commit = subprocess.run(["git", "-C", str(UP), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    up_clean = subprocess.run(["git", "-C", str(UP), "status", "--porcelain"], capture_output=True, text=True).stdout.strip() == ""
    out = {"respiration": {}, "abp": {}}
    for ds in ("BIDMC", "WESAD", "MIMIC-BP", "UCI-BP"):
        r = audit(ds, dev)
        out[TASK[ds]][ds] = r
        print(ds, json.dumps({k: r[k] for k in ("epoch_saved", "strict_load", "split", "test_windows_per_subject", "metric_block",
                                                "train_target_units_check", "upstream_sampler", "nfe_euler_by_S")}), flush=True)
    cached = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "outputs").glob("*expd*"))
    for task, d in out.items():
        (OUT / task).mkdir(parents=True, exist_ok=True)
        (OUT / task / "audit.json").write_text(json.dumps({"upstream_commit": up_commit, "upstream_clean": up_clean,
                                                           "cached_generated_samples_found": cached, "datasets": d}, indent=1))


if __name__ == "__main__":
    main()
