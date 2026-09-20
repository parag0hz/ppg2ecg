"""MC1: build the 4 s BIDMC / CapnoBase corpora and the three subject-level holdout manifests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260920
CORPORA = {"bidmc": ("data/processed/mc1_bidmc_4s", ["scripts/build_processed_bidmc.py", "--segment-len", "4"]),
           "capnobase": ("data/processed/mc1_capnobase_4s", ["scripts/build_processed_capnobase.py", "--segment-len", "4"]),
           "dalia": ("data/processed/u2_dalia", None)}


def main():
    for name, (proc, builder) in CORPORA.items():
        p = ROOT / proc
        if builder and not any(p.glob("*.npz")):
            subprocess.run([sys.executable, *builder, "--out", str(p)], cwd=ROOT, check=True)
        subs = sorted(f.stem for f in p.glob("*.npz"))
        assert all(np.load(p / f"{s}.npz")["x"].shape[1] == 512 for s in subs[:3]), name
        perm = [subs[i] for i in np.random.default_rng(SEED).permutation(len(subs))]
        n_te, n_va = round(0.30 * len(subs)), max(2, round(0.10 * len(subs)))
        test, val, train = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
        man = {"splits": [{"protocol": "MC1-subject-holdout", "seed": SEED, "corpus": name, "task": "ECG",
                           "train": sorted(train), "val": sorted(val), "test": sorted(test)}],
               "extra": {"processed": proc, "n_subjects": len(subs),
                         "n_windows": {r: int(sum(len(np.load(p / f"{s}.npz")["x"]) for s in v)) for r, v in (("train", train), ("val", val), ("test", test))}}}
        (ROOT / f"data/manifests/split_mc1_{name}.json").write_text(json.dumps(man, indent=1))
        print(name, len(subs), "subjects", man["extra"]["n_windows"], "test", len(test), flush=True)


if __name__ == "__main__":
    main()
