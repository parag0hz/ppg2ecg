#!/usr/bin/env python
"""U2 steps (a)+(b): materialise the U2 corpora and splits.

Source is U1's data, unchanged: `data/processed/upstream_u1/<DS>/subject*.pkl`, produced by
upstream PENGUIN's own `preprocess.py` at the shipped 4 s. This script only re-containers it
as the `<subject>.npz` the trainers read, and writes the split manifest declared in
docs/U2_PAIRED_IMEANFLOW_VS_PENGUIN_PREREGISTRATION.md §5. It changes no sample value.

Two preregistered interventions, both applied here and asserted:
  * UCI-BP is deduplicated to subjects {0,1,4,5} (U1-P1: 0=2, 1=3, 4=6, 5=7 byte-identically);
  * WildPPG never contains `kjd` / `ssx` (the 14-subject symlink view U1 preprocessed through).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "upstream_u1"
OUTROOT = ROOT / "data" / "processed"
MANIFESTS = ROOT / "data" / "manifests"

SPLIT_SEED = 20260911
FORBIDDEN = ("kjd", "ssx")

# dataset -> (slug, task, keep_subjects or None)
DATASETS = {
    "PPG-DaLiA": ("u2_dalia", "ECG", None),
    "WildPPG":   ("u2_wildppg", "ECG", None),
    "BIDMC":     ("u2_bidmc", "Resp", None),
    "WESAD":     ("u2_wesad", "Resp", None),
    "MIMIC-BP":  ("u2_mimicbp", "ABP", None),
    "UCI-BP":    ("u2_ucibp", "ABP", [0, 1, 4, 5]),  # U1-P1 deduplication
}
RULE = ("subjects sorted by numeric index -> numpy.random.default_rng(20260911).permutation; "
        "n_test = n_val = max(2, round(0.15 n)); test = perm[:n_test], val = perm[n_test:2*n_test], "
        "train = remainder. UCI-BP (n=4 after U1-P1 deduplication) is fixed at 2/1/1.")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def subject_index(name: str) -> int:
    return int(name.replace("subject", "").replace(".pkl", ""))


def plan_split(n: int, dataset: str) -> tuple[list[int], list[int], list[int]]:
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    if dataset == "UCI-BP":
        n_test = n_val = 1
    else:
        n_test = n_val = max(2, round(0.15 * n))
    test = sorted(int(i) for i in perm[:n_test])
    val = sorted(int(i) for i in perm[n_test:n_test + n_val])
    train = sorted(int(i) for i in perm[n_test + n_val:])
    assert len(train) and not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    return train, val, test


def build(dataset: str) -> dict:
    slug, task, keep = DATASETS[dataset]
    src_files = sorted((SRC / dataset).glob("subject*.pkl"), key=lambda p: subject_index(p.name))
    if keep is not None:
        src_files = [p for p in src_files if subject_index(p.name) in keep]
    assert src_files, f"no source pkl for {dataset}"

    out = OUTROOT / slug
    out.mkdir(parents=True, exist_ok=True)

    ids, src_sha, y_sha, counts = [], {}, {}, {}
    for p in src_files:
        sid = p.stem  # "subject7"
        with p.open("rb") as fh:
            d = pickle.load(fh)
        x = np.asarray(d["x_data"], dtype=np.float32)
        y = np.asarray(d["y_data"], dtype=np.float32)
        assert x.shape == y.shape and x.shape[1] == 512, f"{dataset}/{sid}: unexpected shape {x.shape}"
        np.savez(out / f"{sid}.npz", x=x, y=y,
                 window_index=np.arange(len(x), dtype=np.int64), subject=np.array(sid))
        ids.append(sid)
        src_sha[sid] = sha256_file(p)
        y_sha[sid] = hashlib.sha256(y.tobytes()).hexdigest()[:16]
        counts[sid] = int(len(x))

    # deduplication must have worked: no two kept subjects may share a target hash
    dup = {}
    for sid, h in y_sha.items():
        dup.setdefault(h, []).append(sid)
    dup = {h: v for h, v in dup.items() if len(v) > 1}
    assert not dup, f"{dataset}: duplicate subjects survived deduplication: {dup}"

    order = sorted(ids, key=subject_index)
    tr_i, va_i, te_i = plan_split(len(order), dataset)
    split = {"train": [order[i] for i in tr_i], "val": [order[i] for i in va_i], "test": [order[i] for i in te_i]}

    if dataset == "WildPPG":
        view = sorted((ROOT / "data/raw_u1_wildppg_view/WildPPG").glob("*.mat"))
        real = {f"subject{i}": p.name for i, p in enumerate(view)}
        assert len(real) == 14, f"WildPPG view has {len(real)} subjects"
        for role in ("train", "val", "test"):
            for sid in split[role]:
                assert not any(b in real[sid] for b in FORBIDDEN), f"FIREWALL: {real[sid]} in {role}"
        split["subject_map"] = real

    manifest = {
        "splits": [{
            "protocol": "U2-paired-subject-holdout",
            "seed": SPLIT_SEED, "corpus": slug, "dataset": dataset, "task": task,
            "rule": RULE,
            "deduplicated": keep, "n_subjects": len(order),
            **{k: v for k, v in split.items() if k != "subject_map"},
        }],
        "extra": {
            "source": f"data/processed/upstream_u1/{dataset} (upstream preprocess.py, shipped 4 s)",
            "source_sha256": src_sha, "target_sha256_16": y_sha, "windows_per_subject": counts,
            "n_windows": {r: sum(counts[s] for s in split[r]) for r in ("train", "val", "test")},
            **({"wildppg_subject_map": split["subject_map"]} if dataset == "WildPPG" else {}),
        },
    }
    mpath = MANIFESTS / f"split_{slug}_seed42.json"
    mpath.write_text(json.dumps(manifest, indent=1))
    nw = manifest["extra"]["n_windows"]
    print(f"{dataset:10s} -> {slug:11s} subj {len(order):5d} "
          f"({len(split['train'])}/{len(split['val'])}/{len(split['test'])})  "
          f"windows {nw['train']:>7,}/{nw['val']:>7,}/{nw['test']:>7,}  test={split['test']}")
    return manifest


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        build(ds)


if __name__ == "__main__":
    main()
