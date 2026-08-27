"""A9 §5-6: global TRAIN-ONLY affine ECG normalisation from the pre-window-normalisation WildPPG targets.
Hard guard: opening any val/test subject file while computing the statistics raises."""
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
    ap.add_argument("--manifest", default="data/manifests/split_a4_wildppg_seed42.json")
    ap.add_argument("--processed", default="data/processed/wildppg_8s_prenorm")
    ap.add_argument("--out", default="artifacts/a9_ecg_representation_control/normalization.json")
    args = ap.parse_args()
    split = read_manifest(ROOT / args.manifest)[0]
    processed = ROOT / args.processed
    forbidden = {f"{s}.npz" for s in split["val"] + split["test"]}
    opened: list[str] = []
    real_open = builtins.open

    def guard(file, *a, **kw):
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
    rec = {"created": datetime.now().isoformat(timespec="seconds"),
           "definition": "y_global = (y_ecg - mu_train) / sigma_train; ONE global scalar pair from all pre-window-normalisation ECG samples of the TRAIN subjects only",
           "source_stage": "WildPPG ECG after FFT resample to 128 Hz + 4th-order Butterworth 0.5 Hz high-pass (zero-phase), BEFORE any per-window z-score/min-max — i.e. exactly the signal the frozen pipeline normalises per window",
           "mu_train": stats["mu_train"], "sigma_train": stats["sigma_train"], "n_train_samples": stats["n_train_samples"], "n_train_subjects": stats["n_train_subjects"],
           "train_subjects": sorted(split["train"]), "split_manifest": args.manifest, "split_manifest_sha256": sha256_file(ROOT / args.manifest),
           "processed_dir": args.processed, "processed_manifest_sha256": sha256_file(processed / "MANIFEST.json"), "processed_total_windows": man["total_windows"],
           "ecg_preprocess": man["ecg_preprocess"], "git": git_sha(ROOT),
           "leakage_check": {"n_files_opened": len(opened), "val_test_files_opened": sorted(set(opened) & forbidden), "ok": not (set(opened) & forbidden)}}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**rec, "per_subject": stats["per_subject"]}, indent=1))
    print(f"mu_train {rec['mu_train']:.6f} | sigma_train {rec['sigma_train']:.6f} | n {rec['n_train_samples']} samples from {rec['n_train_subjects']} subjects | leakage ok {rec['leakage_check']['ok']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
