"""Deterministic subject-level splits + JSON manifests.

Upstream (load_data.py L13-24) shuffles `glob.glob(...)` output with `random.sample` — the result depends on
directory-listing order, so the split is NOT reproducible across machines even with seed 42.
Here the subject list is SORTED before shuffling with an explicit `random.Random(seed)`.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .dalia import SUBJECTS


def make_holdout_split(subjects=SUBJECTS, n_val: int = 1, n_test: int = 1, seed: int = 42) -> dict:
    """Protocol P0 (upstream-shaped): 15 // fold_num(8) = 1 val subject, 1 test subject, 13 train."""
    subs = sorted(subjects, key=lambda s: int(s[1:]))
    order = random.Random(seed).sample(subs, len(subs))
    val, test = order[:n_val], order[n_val : n_val + n_test]
    train = [s for s in subs if s not in val and s not in test]
    return {"protocol": "P0-holdout", "seed": seed, "train": train, "val": val, "test": test}


def make_kfold_splits(subjects=SUBJECTS, n_folds: int = 5, n_val: int = 2, seed: int = 42) -> list[dict]:
    """Protocol P1: subject-wise K-fold. Each subject is in exactly one test fold; val subjects drawn from the rest."""
    subs = sorted(subjects, key=lambda s: int(s[1:]))
    rng = random.Random(seed)
    order = rng.sample(subs, len(subs))
    folds = [order[i::n_folds] for i in range(n_folds)]
    out = []
    for k in range(n_folds):
        test = sorted(folds[k], key=lambda s: int(s[1:]))
        nxt = folds[(k + 1) % n_folds]  # validation subjects rotate with the fold (never overlap the test fold)
        val = sorted(nxt[:n_val], key=lambda s: int(s[1:]))
        train = sorted([s for s in order if s not in test and s not in val], key=lambda s: int(s[1:]))
        out.append({"protocol": "P1-kfold", "seed": seed, "fold": k, "n_folds": n_folds, "train": train, "val": val, "test": test})
    return out


def write_manifest(path: Path, split: dict | list, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"splits": split if isinstance(split, list) else [split], "extra": extra or {}}
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_manifest(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())["splits"]
