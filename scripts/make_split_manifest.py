"""Write the pre-registered subject splits to data/manifests/ (P0 hold-out, P1 5-fold). Deterministic."""
from __future__ import annotations

from pathlib import Path

from ppg2ecg.data.leakage import check_subject_disjoint
from ppg2ecg.data.dalia import SUBJECTS
from ppg2ecg.data.splits import make_holdout_split, make_kfold_splits, write_manifest

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests"


def main(seed: int = 42):
    p0 = make_holdout_split(seed=seed)
    assert check_subject_disjoint(p0, SUBJECTS)["ok"]
    print("P0:", p0)
    write_manifest(MAN / f"split_p0_holdout_seed{seed}.json", p0, {"note": "13/1/1 subjects; shape matches upstream fold_num=8 (15//8=1) but subjects chosen from a SORTED list with random.Random(seed)"})
    p1 = make_kfold_splits(seed=seed)
    for f in p1:
        assert check_subject_disjoint(f, SUBJECTS)["ok"]
        print(f"P1 fold {f['fold']}: test={f['test']} val={f['val']}")
    tests = [s for f in p1 for s in f["test"]]
    assert sorted(tests, key=lambda s: int(s[1:])) == sorted(SUBJECTS, key=lambda s: int(s[1:])), "every subject must be tested exactly once"
    write_manifest(MAN / f"split_p1_kfold5_seed{seed}.json", p1, {"note": "5 folds x 3 test subjects, 2 val subjects, 10 train subjects"})
    print("wrote manifests to", MAN)


if __name__ == "__main__":
    main()
