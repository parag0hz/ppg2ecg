"""Final preflight before A0 training: gathers provenance and HARD-FAILS on any leakage / pinning / protocol violation.
Writes <out>/provenance.json, <out>/split_manifest.json, <out>/config.yaml. Exit 1 on failure => training must not start."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from ppg2ecg.data.dalia import SUBJECTS
from ppg2ecg.data.leakage import check_subject_disjoint, check_window_disjoint, check_windowwise_normalization
from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows
from ppg2ecg.data.splits import read_manifest
from ppg2ecg.models import build_penguin_backbone, count_params
from ppg2ecg.training.valbank import bank_hash, make_banks
from ppg2ecg.utils.upstream import UPSTREAM_COMMIT, upstream_git_state

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exp-name", default="a0_penguin_otcfm_ppgdalia_8s_seed42")
    ap.add_argument("--processed", default="data/processed/v0_8s")
    ap.add_argument("--manifest", default="data/manifests/split_p0_holdout_seed42.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window-s", type=int, default=8)
    ap.add_argument("--sample-rate", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--n-step", type=int, default=25)
    ap.add_argument("--select", choices=["val_mae", "fixed_cfm"], default="val_mae")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--n-val-banks", type=int, default=4)
    ap.add_argument("--bank-seed", type=int, default=1000)
    ap.add_argument("--val-mae-every", type=int, default=1)
    ap.add_argument("--gen-diag-every", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures = []

    # git / upstream
    git_state = {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "dirty_files": [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]}
    up = upstream_git_state()
    if up["commit"] != UPSTREAM_COMMIT or up["dirty_files"]:
        failures.append(f"upstream not pinned/clean: {up}")
    # dataset
    processed = ROOT / args.processed
    man = json.loads((processed / "MANIFEST.json").read_text())
    if man["samples_per_window"] != args.sample_rate * args.window_s:
        failures.append(f"window length mismatch: manifest {man['samples_per_window']} vs {args.sample_rate*args.window_s}")
    for s, info in man["files"].items():
        if sha256_file(ROOT / info["path"]) != info["sha256"]:
            failures.append(f"processed file hash changed: {s}")
    raw_checksums = (ROOT / "data/raw/CHECKSUMS.sha256").read_text().splitlines()
    # split
    manifest_path = ROOT / args.manifest
    split = read_manifest(manifest_path)[0]
    subj = check_subject_disjoint(split, SUBJECTS)
    if not subj["ok"]:
        failures.append(f"subject overlap: {subj}")
    arrays = {k: np.concatenate([np.load(processed / f"{s}.npz")["x"] for s in split[k]]) for k in ("train", "val", "test")}
    win = check_window_disjoint(arrays)
    if not win["ok"]:
        failures.append(f"window overlap: {win}")
    norm = check_windowwise_normalization(lambda a: preprocess_windows(a, args.sample_rate, args.window_s, **PPG_KW), arrays["test"][:32])
    norm_y = check_windowwise_normalization(lambda a: preprocess_windows(a, args.sample_rate, args.window_s, **ECG_KW), arrays["test"][:32], seed=1)
    if not (norm["ok"] and norm_y["ok"]):
        failures.append(f"normalisation not window-local: {norm} {norm_y}")
    # fixed validation banks (deterministic selection metric)
    T = arrays["val"].shape[1]
    banks = make_banks(len(arrays["val"]), T, args.n_val_banks, args.bank_seed) if args.n_val_banks > 0 else []
    banks_h = bank_hash(banks) if banks else None
    # model
    model = build_penguin_backbone(n_step=args.n_step, sample_rate=args.sample_rate)
    params = count_params(model, exclude_prefixes=("cross_attn", "revin"))
    # hardware
    gpu = {"name": torch.cuda.get_device_name(0), "total_MiB": torch.cuda.get_device_properties(0).total_memory / 2**20, "capability": torch.cuda.get_device_capability(0)} if torch.cuda.is_available() else None
    cfg = {
        "exp_name": args.exp_name, "arm": "A0", "model": "PENGUIN (upstream class, unmodified) Flow-SSM/S5 + OT-CFM",
        "model_cfg": dict(n_step=args.n_step, sample_rate=args.sample_rate, h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0),
        "dataset": "PPG-DaLiA", "window_s": args.window_s, "sample_rate": args.sample_rate, "samples_per_window": args.sample_rate * args.window_s,
        "preprocess": {"ppg": PPG_KW, "ecg": ECG_KW, "per_window_stats": True, "global_stats": None},
        "split": {"manifest": args.manifest, "sha256": sha256_file(manifest_path), **{k: split[k] for k in ("protocol", "seed", "train", "val", "test")}},
        "seed": args.seed, "optimizer": "AdamW(betas=(0.9,0.999))", "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "epochs_max": args.epochs, "early_stopping": {"metric": "val_mae_batchmean (full n_step Heun samples)" if args.select == "val_mae" else f"val_cfm_fixed (deterministic CFM loss on {args.n_val_banks} fixed (t,z) banks, seed {args.bank_seed})", "patience": args.patience, "min_delta": args.min_delta}, "checkpoint": "best " + ("val MAE" if args.select == "val_mae" else "val_cfm_fixed"),
        "selection": {"criterion": args.select, "min_delta": args.min_delta, "n_val_banks": args.n_val_banks, "bank_seed": args.bank_seed, "bank_hash": banks_h, "val_mae_every": args.val_mae_every, "gen_diag_every": args.gen_diag_every},
        "sampler_train_val": {"solver": "heun", "steps": args.n_step, "nfe": 2 * args.n_step},
        "precision": {"dtype": "float32", "amp": False, "bf16": False, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32},
        "deterministic": {"cudnn_deterministic": True, "use_deterministic_algorithms": "True (warn_only)"},
    }
    prov = {
        "created": datetime.now().isoformat(timespec="seconds"), "git": git_state, "upstream": {**up, "expected": UPSTREAM_COMMIT, "url": "https://github.com/Neurogica/PENGUIN.git"},
        "dataset": {"raw_checksums_sha256": raw_checksums, "processed_manifest": {k: man[k] for k in ("built", "segment_len_s", "resample_rate", "samples_per_window", "total_windows")}, "processed_files": man["files"]},
        "leakage_checks": {"subject_disjoint": subj, "window_disjoint": win, "windowwise_normalization_ppg": norm, "windowwise_normalization_ecg": norm_y},
        "n_windows": {k: int(len(v)) for k, v in arrays.items()},
        "model_params": params, "hardware": {"gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "python": sys.version.split()[0]},
        "config": cfg, "failures": failures, "ok": not failures,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    (out / "split_manifest.json").write_text(json.dumps({"split": split, "manifest_sha256": cfg["split"]["sha256"], "n_windows": prov["n_windows"]}, indent=1))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("train subjects:", split["train"])
    print("val subjects  :", split["val"])
    print("test subjects :", split["test"])
    print("subject disjoint:", subj["ok"], subj["overlaps"], "| window disjoint:", win["ok"], win["overlaps"], "| window-local norm:", norm["ok"], norm_y["ok"])
    print("selection:", cfg["selection"])
    print("window:", args.window_s, "s @", args.sample_rate, "Hz | seed", args.seed, "| params", params, "| git", git_state["commit"][:8], f"({len(git_state['dirty_files'])} dirty)", "| upstream", up["commit"][:8])
    print("GPU:", gpu, "| torch", torch.__version__, "cuda", torch.version.cuda)
    print("PREFLIGHT", "OK" if not failures else "FAILED: " + "; ".join(failures))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
