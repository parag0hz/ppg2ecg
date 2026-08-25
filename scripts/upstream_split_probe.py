"""Reproduce the subject split that upstream train.py WOULD use (glob order + random.seed(42) + random.sample),
without training, so the exact val/test subjects of an upstream run are recorded. Must be run on the same
filesystem/directory as the upstream run (glob order is filesystem-dependent)."""
from __future__ import annotations

import glob
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(procdata_dir: str = str(ROOT / "data/processed/upstream/PPG-DaLiA"), seed: int = 42, subject_num: int = 15, fold_num: int = 8):
    random.seed(seed)  # upstream: fix_seed(42) in main(); python `random` untouched until load_dataset_path
    file_list = glob.glob(f"{procdata_dir}/*")  # upstream load_data.py L15 (unsorted!)
    if not file_list:
        print(f"no processed files under {procdata_dir} (run scripts/run_upstream_preprocess.sh first)")
        return 1
    file_list = random.sample(file_list, len(file_list))
    val_size = subject_num // fold_num
    val, test = file_list[:val_size], file_list[val_size : 2 * val_size]
    train = [f for f in file_list if f not in val + test]
    name = lambda p: Path(p).stem  # noqa: E731  (subject{idx} -> S{idx+1})
    print("glob order :", [name(p) for p in glob.glob(f"{procdata_dir}/*")])
    print("val        :", [name(p) for p in val])
    print("test       :", [name(p) for p in test])
    print("train      :", [name(p) for p in train])
    print("NOTE: upstream file 'subject{i}.pkl' corresponds to DaLiA S{i+1}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
