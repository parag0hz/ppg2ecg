"""WP1: subject-level 4-fold manifests over the 14 WildPPG subjects of the U2 corpus (kjd/ssx are not in it)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260919


def main():
    m = json.loads((ROOT / "data/manifests/split_u2_wildppg_seed42.json").read_text())
    s, real = m["splits"][0], m["extra"]["wildppg_subject_map"]
    subs = sorted(s["train"] + s["val"] + s["test"], key=lambda x: int(x.replace("subject", "")))
    assert len(subs) == 14 and not any(b in real[x] for x in subs for b in ("kjd", "ssx"))
    perm = [subs[i] for i in np.random.default_rng(SEED).permutation(len(subs))]
    tests = [perm[0:4], perm[4:8], perm[8:11], perm[11:14]]
    for k, test in enumerate(tests):
        rest = [x for x in perm if x not in test]
        val, train = rest[:2], rest[2:]                      # first two of the remaining permutation order
        man = {"splits": [{"protocol": "WP1-subject-4fold", "seed": SEED, "fold": k, "corpus": "u2_wildppg",
                           "dataset": "WildPPG", "task": "ECG", "train": sorted(train), "val": sorted(val), "test": sorted(test)}],
               "extra": {"wildppg_subject_map": real, "source": "data/processed/u2_wildppg (U2 corpus, 14 subjects)"}}
        (ROOT / f"data/manifests/split_wp1_fold{k}.json").write_text(json.dumps(man, indent=1))
        print(k, "test", [real[x][13:16] for x in test], "val", [real[x][13:16] for x in val], "train", len(train))
    allt = sorted(x for t in tests for x in t)
    assert allt == sorted(subs), "every subject must be tested exactly once"


if __name__ == "__main__":
    main()
