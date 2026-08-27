"""A8 §4: compute the single global train-only ABP normalisation and write artifacts/a8_abp_scale_control/normalization.json.
Hard assertion: only TRAIN subjects are read (val/test files are never opened)."""
from __future__ import annotations

import argparse
import builtins
import json
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.data.target_norm import compute_train_stats, sha256_file
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/split_a7_mimicbp_official.json")
    ap.add_argument("--processed", default="data/processed/mimicbp_8s")
    ap.add_argument("--out", default="artifacts/a8_abp_scale_control/normalization.json")
    args = ap.parse_args()
    split = read_manifest(ROOT / args.manifest)[0]
    processed = ROOT / args.processed
    forbidden = {f"{s}.npz" for s in split["val"] + split["test"]}
    opened: list[str] = []
    real_open = builtins.open

    def guard(file, *a, **kw):  # hard leakage assertion: no val/test file may be opened while computing the statistics
        name = Path(str(file)).name
        opened.append(name)
        assert name not in forbidden, f"LEAKAGE: attempted to read {name} while computing train-only statistics"
        return real_open(file, *a, **kw)

    builtins.open = guard
    try:
        stats = compute_train_stats(processed, split["train"])
    finally:
        builtins.open = real_open
    man = json.loads((processed / "MANIFEST.json").read_text())
    rec = {"created": datetime.now().isoformat(timespec="seconds"), "definition": "y_norm = (y_mmHg - mu_train) / sigma_train; ONE global scalar pair from all ABP samples of the TRAIN subjects only", "mu_train": stats["mu_train"], "sigma_train": stats["sigma_train"], "n_train_samples": stats["n_train_samples"], "n_train_subjects": stats["n_train_subjects"], "split_manifest": args.manifest, "split_manifest_sha256": sha256_file(ROOT / args.manifest), "processed_dir": args.processed, "processed_manifest_sha256": sha256_file(processed / "MANIFEST.json"), "processed_total_windows": man["total_windows"], "git": git_sha(ROOT), "leakage_check": {"n_files_opened": len(opened), "val_test_files_opened": sorted(set(opened) & forbidden), "ok": not (set(opened) & forbidden)}, "per_subject_stats_excluded_from_use": "recorded for audit only; the model uses the single global pair"}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**rec, "per_subject": stats["per_subject"]}, indent=1))
    print(f"mu_train {rec['mu_train']:.6f} mmHg | sigma_train {rec['sigma_train']:.6f} mmHg | n {rec['n_train_samples']} samples from {rec['n_train_subjects']} subjects | leakage ok {rec['leakage_check']['ok']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
