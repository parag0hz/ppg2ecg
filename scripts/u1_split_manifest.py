#!/usr/bin/env python
"""U1 — reproduce and record upstream PENGUIN's realised train/val/test split.

Upstream `train.py` never logs which subject files it drew. Its split is produced by
`fix_seed(cfg.seed)` (which seeds the stdlib `random` module) followed by
`load_dataset_path` -> `random.sample(glob.glob(...), n)`. Nothing between those two
points consumes the stdlib RNG (`summarize` and `initialize_model` use torch only), so
the split is reproduced exactly here without touching upstream code.

Also enforces the project firewall: WildPPG subjects `kjd` / `ssx` may never appear in
any split list.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

FORBIDDEN = ("kjd", "ssx")


def realised_split(procdata_path: str, dataset: str, subject_num: int, fold_num: int, seed: int) -> dict:
    file_list = sorted(glob.glob(f"{procdata_path}/{dataset}/*"), key=lambda p: p)
    unsorted = glob.glob(f"{procdata_path}/{dataset}/*")
    random.seed(seed)
    shuffled = random.sample(unsorted, len(unsorted))
    val_size = subject_num // fold_num
    val = shuffled[:val_size]
    test = shuffled[val_size : 2 * val_size]
    train = [f for f in shuffled if f not in (val + test)]
    return {
        "glob_order": [Path(p).name for p in unsorted],
        "sorted_order": [Path(p).name for p in file_list],
        "val": [Path(p).name for p in val],
        "test": [Path(p).name for p in test],
        "train": [Path(p).name for p in train],
        "val_size": val_size,
        "n_files": len(unsorted),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procdata", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subject-num", type=int, required=True)
    ap.add_argument("--fold-num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subject-map", default="", help="optional JSON: subject index -> source identifier")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split = realised_split(args.procdata, args.dataset, args.subject_num, args.fold_num, args.seed)
    if split["n_files"] != args.subject_num:
        raise SystemExit(
            f"expected {args.subject_num} processed subject files for {args.dataset}, found {split['n_files']}"
        )
    if args.subject_map:
        split["subject_map"] = json.loads(Path(args.subject_map).read_text())
        for role in ("train", "val", "test"):
            for name in split[role]:
                src = split["subject_map"].get(name.replace(".pkl", "").replace("subject", ""), "")
                if any(bad in str(src) for bad in FORBIDDEN):
                    raise SystemExit(f"FIREWALL VIOLATION: {src} reached the {role} split of {args.dataset}")
    # window counts + per-file hash (U1-P1 duplication check)
    import pickle

    counts, digests = {}, {}
    for name in split["sorted_order"]:
        p = Path(args.procdata) / args.dataset / name
        with p.open("rb") as fh:
            d = pickle.load(fh)
        counts[name] = [int(d["x_data"].shape[0]), int(d["x_data"].shape[1])]
        digests[name] = hashlib.sha256(d["y_data"].tobytes()).hexdigest()[:16]
    split["window_counts"] = counts
    split["y_sha256_16"] = digests
    split["n_windows_total"] = sum(v[0] for v in counts.values())
    split["n_windows"] = {
        role: sum(counts[n][0] for n in split[role]) for role in ("train", "val", "test")
    }
    dup = {}
    for name, dg in digests.items():
        dup.setdefault(dg, []).append(name)
    split["duplicate_groups"] = {k: v for k, v in dup.items() if len(v) > 1}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(split, indent=2))
    print(json.dumps({k: split[k] for k in ("val", "test", "n_windows", "duplicate_groups")}, indent=2))


if __name__ == "__main__":
    main()
