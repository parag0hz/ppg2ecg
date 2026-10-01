"""C0 architecture split of the V1 VitalDB TRAIN patient pool (prereg §3). Built from the V1 manifest only."""
from __future__ import annotations

import hashlib
import json

import numpy as np

SPLIT_SEED = 20261001
N_HOLDOUT, N_VAL, N_TRAIN = 434, 433, 3470


def arch_split(manifest: dict) -> dict:
    """Patients of V1 `train`, sorted numerically, permuted with `default_rng(20261001)`: the first 434 are ARCH-HOLDOUT,
    the next 433 ARCH-VAL, the remaining 3,470 ARCH-TRAIN (the V1 convention: held-out roles first). All cases of a
    patient follow the patient; cases keep their V1 manifest order."""
    sp = manifest["splits"][0]
    poc = manifest["extra"]["patient_of_case"]
    pats = np.array(sorted({int(poc[c]) for c in sp["train"]}))
    if pats.size != N_HOLDOUT + N_VAL + N_TRAIN:
        raise ValueError(f"expected {N_HOLDOUT + N_VAL + N_TRAIN} V1 train patients, found {pats.size}")
    perm = np.random.default_rng(SPLIT_SEED).permutation(pats)
    roles = {"holdout": set(perm[:N_HOLDOUT].tolist()), "val": set(perm[N_HOLDOUT:N_HOLDOUT + N_VAL].tolist()),
             "train": set(perm[N_HOLDOUT + N_VAL:].tolist())}
    out = {}
    for role, ps in roles.items():
        out[role] = {"patients": sorted(int(p) for p in ps),
                     "cases": [c for c in sp["train"] if int(poc[c]) in ps]}
    return out


def old_heldout_patients(manifest: dict) -> dict:
    """Patients of the old V1 validation and test splits (from the manifest; no data file is opened)."""
    sp = manifest["splits"][0]
    poc = manifest["extra"]["patient_of_case"]
    return {role: sorted({int(poc[c]) for c in sp[role]}) for role in ("val", "test")}


def window_offsets(manifest: dict) -> dict:
    """First window index of every V1 train case in the concatenated V1 train order (`vm1_evaluate.load('train')`)."""
    sp = manifest["splits"][0]
    wpc = manifest["extra"]["windows_per_case"]
    off, out = 0, {}
    for c in sp["train"]:
        out[c] = (off, off + int(wpc[c]))
        off += int(wpc[c])
    return out


def ids_sha256(ids) -> str:
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()
