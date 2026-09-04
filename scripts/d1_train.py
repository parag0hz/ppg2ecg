"""D1 per-corpus training driver (docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md §3).

Builds the frozen arm-B `train_a2` argv from scripts/d1_common.py and runs it — one corpus per invocation, into
outputs/d1_<corpus>_seed42/. The hyperparameters are identical for every corpus and are NOT adjusted for corpus
size: a train split below --min-train-windows is recorded as `small_corpus` in d1_train_meta.json (§7) and
trained anyway. Nothing here may differ between corpora except the two corpus paths.

Run: .venv/bin/python scripts/d1_train.py --corpus bidmc [--resume] [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_common import (  # noqa: E402
    MIN_TRAIN_WINDOWS, ROOT, TRAIN_HP, TRAIN_KEYS, assert_no_forbidden_subjects, corpus, ensure_split_manifest,
    git_sha, sha256_file, split_sizes, train_argv,
)


def stream(cmd: list[str], log: Path) -> int:
    """Run `cmd`, mirroring its merged stdout/stderr to `log` and to this process's stdout, line by line."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    with open(log, "a", buffering=1) as fh, subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
        return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(TRAIN_KEYS))
    ap.add_argument("--resume", action="store_true", help="pass train_a2's own --resume through")
    ap.add_argument("--force", action="store_true", help="train even though TRAINING_DONE already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved train_a2 argv and exit without training")
    ap.add_argument("--min-train-windows", type=int, default=MIN_TRAIN_WINDOWS, help="below this the run is FLAGGED small_corpus; hyperparameters are never changed (§7)")
    args = ap.parse_args()

    c = corpus(args.corpus)
    split = ensure_split_manifest(c)
    assert_no_forbidden_subjects(split, f"d1_train[{c.key}]")
    sizes = split_sizes(c, split)
    small = sizes["train"]["n_windows"] < args.min_train_windows
    argv = [sys.executable, *train_argv(c, resume=args.resume)]
    done, failed = c.out_dir / "TRAINING_DONE", c.out_dir / "TRAINING_FAILED"

    print(f"[d1-train] corpus {c.key} ({c.name}) -> {c.out_dir.relative_to(ROOT)}")
    for k in ("train", "val", "test"):
        print(f"[d1-train]   {k:5s} {sizes[k]['n_subjects']:4d} subjects {sizes[k]['n_windows']:8d} windows")
    if small:
        print(f"[d1-train] WARNING small_corpus: {sizes['train']['n_windows']} train windows < {args.min_train_windows}; hyperparameters UNCHANGED, flag recorded")
    print(f"[d1-train] argv: {' '.join(argv)}")
    if args.dry_run:
        print("[d1-train] --dry-run: nothing executed")
        return 0
    if done.exists() and not args.force:
        print(f"[d1-train] {done.relative_to(ROOT)} exists -> skipping (use --force to retrain)")
        return 0

    c.out_dir.mkdir(parents=True, exist_ok=True)
    failed.unlink(missing_ok=True)  # a previous failure must not shadow this attempt's outcome
    meta = {"corpus": c.key, "dataset": c.name, "citation": c.citation, "target": c.target,
            "processed": c.processed, "processed_manifest_sha256": sha256_file(c.processed_dir / "MANIFEST.json"),
            "manifest": c.manifest, "manifest_sha256": sha256_file(c.manifest_path), "split": split, "split_sizes": sizes,
            "small_corpus": bool(small), "min_train_windows": int(args.min_train_windows),
            "hyperparameters": TRAIN_HP, "argv": argv, "resume": bool(args.resume), "force": bool(args.force),
            "git": git_sha(), "started": datetime.now().isoformat(timespec="seconds")}
    (c.out_dir / "d1_train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    rc = stream(argv, c.out_dir / "train.log")
    print(f"[d1-train] {c.key} exit {rc} | TRAINING_DONE={done.exists()} TRAINING_FAILED={failed.exists()}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
