"""D1 multi-dataset benchmark configuration — single source of truth for the drivers
(docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md §§3-6).

Everything that must be identical across corpora lives here: the frozen arm-B hyperparameters (§3), the corpus
records (§4), the 70/15/15 subject-level split rule (§5), the NFE grid and the bootstrap constants (§6).
`scripts/d1_train.py` and `scripts/d1_evaluate.py` import from this module and add nothing of their own.

Self-check (read-only): .venv/bin/python scripts/d1_common.py
Write the split manifests:  .venv/bin/python scripts/d1_common.py --write-manifests
Re-emit them against the current corpora (partition must not move): scripts/d1_common.py --refresh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ppg2ecg.data.splits import write_manifest

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
#: WildPPG test hold-out. Standing rule for the whole benchmark: never in a train or val list, anywhere.
FORBIDDEN_SUBJECTS: tuple[str, ...] = ("kjd", "ssx")
#: §6 sampling budget. 1 is the headline (one-step), 50 is PENGUIN's published operating point.
NFE_GRID: tuple[int, ...] = (1, 2, 4, 10, 25, 50)
#: §6 uncertainty: subject-clustered bootstrap.
BOOTSTRAP_N, BOOTSTRAP_SEED = 2000, 20260904
#: deterministic figure material: the first N SCORED test windows of every test subject (a subset of the
#: evaluated population, so every figure panel is described by a per_window_metrics.csv row).
WAVEFORM_WINDOWS_PER_SUBJECT = 16
#: §7 disclosure threshold; below this a run is flagged, never re-tuned.
MIN_TRAIN_WINDOWS = 2000
FS = 128
SAMPLES_PER_WINDOW = 1024
SPLIT_RULE = ("natural-sorted subject ids -> numpy.random.Generator(PCG64(42)).permutation -> "
              "train = ceil(0.70*n) (capped at n-2), val = max(1, floor(rest/2)), test = the remainder")
#: what a split manifest pins the corpus to. NOT the MANIFEST.json sha256: that file carries a 'built' timestamp
#: (and other run-local fields) that moves on every rebuild even when every stored array byte is identical.
CORPUS_IDENTITY_RULE = ("sha256 of json.dumps of the natural-sorted list of [subject, n_windows, per-file sha256] "
                        "triples taken from the corpus MANIFEST.json 'files' block")

#: §3 frozen arm-B recipe, replayed byte-for-byte from outputs/c1_imf_baseline_replay_seed42/train_meta.json.
#: Identical for every corpus — that identity is the benchmark. Every key is a real `train_a2.py` flag.
TRAIN_HP: dict[str, object] = {
    "seed": SEED, "epochs": 300, "patience": 20, "c1-arm": "B", "min-delta": 1e-4,
    "batch-size": 64, "lr": 1e-3, "weight-decay": 0.01,
    "h-dim": 128, "blocks": 4, "ssm-ratio": 2.0, "mlp-ratio": 2.0, "sample-rate": FS,
    "p-mean": -0.4, "p-std": 1.0, "data-proportion": 0.5, "norm-p": 1.0, "norm-eps": 0.01,
    "jvp-mode": "forward", "cond-mode": "h_only", "h-scale": 1.0,
    "micro-batch": 32, "val-batch": 32, "n-val-banks": 4, "bank-seed": 1000,
    "gen-diag-every": 1, "gen-diag-windows": 128, "val-every-steps": 220, "val-subsample": 4096,
}


class ForbiddenSubjectError(RuntimeError):
    """Raised when a WildPPG test-held-out subject reaches a train or val list."""


class CorpusIdentityError(RuntimeError):
    """Raised when a split manifest no longer matches the corpus it was written against."""


class SplitPartitionChanged(RuntimeError):
    """Raised when re-emitting a split manifest would move a subject between train/val/test."""


@dataclass(frozen=True)
class Corpus:
    key: str
    name: str
    processed: str              # relative to ROOT
    manifest: str               # relative to ROOT
    fs_ppg: int                 # native PPG sampling rate (Hz), before the frozen 128 Hz resample
    fs_ecg: int                 # native target sampling rate (Hz)
    citation: str
    target: str = "ecg"         # 'abp' corpora are recorded but excluded from the ECG benchmark
    trains: bool = True         # False = reuse an existing frozen checkpoint instead of training
    checkpoint: str | None = None
    notes: str = ""

    @property
    def included(self) -> bool:
        return self.target == "ecg"

    @property
    def out_dir(self) -> Path:
        return ROOT / "outputs" / f"d1_{self.key}_seed{SEED}"

    @property
    def exp_name(self) -> str:
        return f"d1_{self.key}_seed{SEED}"

    @property
    def processed_dir(self) -> Path:
        return ROOT / self.processed

    @property
    def manifest_path(self) -> Path:
        return ROOT / self.manifest

    @property
    def checkpoint_path(self) -> Path:
        return ROOT / self.checkpoint if self.checkpoint else self.out_dir / "checkpoint_best.pt"

    @property
    def eval_dir(self) -> Path:
        return self.out_dir / "eval"


CORPORA: tuple[Corpus, ...] = (
    Corpus("wildppg", "WildPPG", "data/processed/wildppg_8s", f"data/manifests/split_d1_wildppg_seed{SEED}.json",
           128, 128, "Meier et al. 2024, WildPPG (ETH Zurich), doi:10.3929/ethz-b-000691074",
           notes="prereg D1 §4: split over the 14 subjects that remain after the never-loaded kjd/ssx, so the D1 "
                 "split differs from A4 and this is a NEW run; the frozen C1 arm-B checkpoint is untouched and is "
                 "reported only as historical context, never as a D1 row"),
    Corpus("dalia", "PPG-DaLiA", "data/processed/dalia_8s", f"data/manifests/split_d1_dalia_seed{SEED}.json",
           64, 700, "Reiss et al. 2019, PPG-DaLiA, UCI ML Repository #495, CC BY 4.0",
           notes="converted from the upstream-preprocessed 8 s pickles by scripts/build_processed_dalia.py"),
    Corpus("bidmc", "BIDMC", "data/processed/bidmc_8s", f"data/manifests/split_d1_bidmc_seed{SEED}.json",
           125, 125, "Pimentel et al. 2017, BIDMC PPG and Respiration Dataset, PhysioNet doi:10.13026/C2208R",
           notes="small corpus (~3k windows); flagged, never re-tuned"),
    Corpus("capnobase", "CapnoBase", "data/processed/capnobase_8s", f"data/manifests/split_d1_capnobase_seed{SEED}.json",
           300, 300, "Karlen et al. 2021, CapnoBase TBME-RR, Borealis doi:10.5683/SP2/NLB8IT",
           notes="unscreened corpus is the primary D1 row; capnobase_8s_clean is a secondary disclosure"),
    Corpus("vitaldb", "VitalDB", "data/processed/vitaldb_8s", f"data/manifests/split_d1_vitaldb_seed{SEED}.json",
           500, 500, "Lee et al. 2022, VitalDB, Sci Data 9:279, doi:10.1038/s41597-022-01411-5",
           notes="caseid-level subset frozen in the corpus MANIFEST; caseid is a marginally weaker cluster than subjectid"),
    Corpus("mimicbp", "MIMIC-BP", "data/processed/mimicbp_8s", f"data/manifests/split_d1_mimicbp_seed{SEED}.json",
           125, 125, "Harvard Dataverse doi:10.7910/DVN/DBM1NF v2.2", target="abp",
           notes="EXCLUDED from the ECG benchmark: the target is arterial blood pressure, not ECG"),
)
BY_KEY: dict[str, Corpus] = {c.key: c for c in CORPORA}
#: the corpora the benchmark actually reports, in execution order
BENCH_KEYS: tuple[str, ...] = tuple(c.key for c in CORPORA if c.included)
TRAIN_KEYS: tuple[str, ...] = tuple(c.key for c in CORPORA if c.included and c.trains)


def corpus(key: str) -> Corpus:
    if key not in BY_KEY:
        raise KeyError(f"unknown corpus {key!r}; known: {sorted(BY_KEY)}")
    return BY_KEY[key]


# ----------------------------------------------------------------------------------------------------------------------
# Hashes, git
# ----------------------------------------------------------------------------------------------------------------------
def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_sha(root: Path = ROOT) -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()

    return {"commit": g("rev-parse", "HEAD"), "dirty_files": len([ln for ln in g("status", "--porcelain").splitlines() if ln.strip() and not ln.startswith("?? outputs")])}


# ----------------------------------------------------------------------------------------------------------------------
# Corpus inventory (read from the corpus MANIFEST.json written by the builders)
# ----------------------------------------------------------------------------------------------------------------------
def natural_key(s: str) -> tuple:
    """Digit-aware order so S2 < S10 (the DaLiA/WildPPG convention in ppg2ecg.data.dalia). Fixes the shuffle input."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(s)))


def corpus_manifest(c: Corpus) -> dict:
    return json.loads((c.processed_dir / "MANIFEST.json").read_text())


def corpus_subjects(c: Corpus) -> list[str]:
    """Splittable subjects of a corpus: every stored subject EXCEPT the standing never-loaded WildPPG hold-outs.

    prereg D1 §4: kjd and ssx are excluded from the eligible pool itself, so no D1 split can place them in train,
    val or test — they are not merely kept out of training, they are never loaded at all.
    """
    return [s for s in sorted(corpus_manifest(c)["files"], key=natural_key) if s not in FORBIDDEN_SUBJECTS]


def window_counts(c: Corpus) -> dict[str, int]:
    return {s: int(v["n_windows"]) for s, v in corpus_manifest(c)["files"].items()}


def corpus_identity_sha256(c: Corpus) -> str:
    """Stable identity of a processed corpus: sha256 over the sorted (subject, n_windows, per-file sha256) triples.

    This is what a split manifest pins (`CORPUS_IDENTITY_RULE`). It changes only when a subject appears/disappears,
    a window count moves, or a stored .npz changes byte-for-byte — never merely because the corpus was rebuilt.
    """
    files = corpus_manifest(c)["files"]
    payload = [[str(s), int(files[s]["n_windows"]), str(files[s]["sha256"])] for s in sorted(files, key=natural_key)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def capped_indices(n: int, cap: int) -> np.ndarray:
    """Exactly min(n, cap) unique ascending indices out of range(n); `cap` <= 0 means no cap (all n).

    Replaces the integer-stride rule `x[::-(-n//cap)]`, which is NOT a cap: for cap = 1024 and n = 1070 the stride
    is 2 and the population is HALVED to 535 instead of trimmed to 1024. Deterministic, outcome-blind, no shuffle.
    """
    n, cap = int(n), int(cap)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    idx = np.unique(np.round(np.linspace(0, n - 1, cap)).astype(np.int64))
    if len(idx) != cap:  # unreachable while cap < n; kept so a collapse can never silently shrink the population
        idx = np.arange(cap, dtype=np.int64)
    assert len(idx) == min(n, cap), (n, cap, len(idx))
    return idx


# ----------------------------------------------------------------------------------------------------------------------
# Splits (§5): one rule, every corpus, seed 42
# ----------------------------------------------------------------------------------------------------------------------
def make_d1_split(subjects: list[str], key: str, seed: int = SEED) -> dict:
    """70/15/15 subject-level partition; train takes the ceiling, val and test are guaranteed non-empty."""
    subs = sorted(subjects, key=natural_key)
    n = len(subs)
    if n < 3:
        raise ValueError(f"{key}: {n} subjects cannot give a non-empty train/val/test partition")
    order = list(np.random.Generator(np.random.PCG64(seed)).permutation(np.asarray(subs, dtype=object)))
    n_train = min(math.ceil(0.70 * n), n - 2)
    n_val = max(1, (n - n_train) // 2)
    train, val, test = order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]
    assert len(val) >= 1 and len(test) >= 1, (key, n, n_train, n_val)
    assert len(train) + len(val) + len(test) == n and not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    split = {"protocol": "D1-subject-holdout", "seed": seed, "corpus": key, "rule": SPLIT_RULE,
             "n_subjects": n, "train": sorted(map(str, train), key=natural_key),
             "val": sorted(map(str, val), key=natural_key), "test": sorted(map(str, test), key=natural_key)}
    assert_no_forbidden_subjects(split, f"make_d1_split({key})")
    return split


def assert_no_forbidden_subjects(split: dict, where: str = "") -> None:
    """Standing rule: kjd and ssx are never LOADED — not for training, not for validation, not for test.

    prereg D1 §4 states the stronger form of the rule than the original §5 wording: the subjects are excluded from
    the eligible pool by `corpus_subjects`, so their appearance in ANY of the three lists is a violation.
    """
    seen = {str(s) for s in list(split.get("train", [])) + list(split.get("val", [])) + list(split.get("test", []))}
    bad = sorted(seen & set(FORBIDDEN_SUBJECTS))
    if bad:
        raise ForbiddenSubjectError(f"{where or 'split'}: never-loaded WildPPG subjects {bad} appear in the split")


def read_split_manifest(path: Path) -> tuple[dict, dict]:
    """(split, extra) from a manifest written by `ppg2ecg.data.splits.write_manifest`."""
    payload = json.loads(Path(path).read_text())
    return payload["splits"][0], dict(payload.get("extra") or {})


def split_provenance(c: Corpus) -> dict:
    """The `extra` block of a D1 split manifest: the pin that is asserted, plus the informational MANIFEST hash."""
    return {"note": f"D1 §5 split for {c.name}; written before any {c.key} weight update",
            "processed": c.processed,
            "corpus_identity_sha256": corpus_identity_sha256(c), "corpus_identity_rule": CORPUS_IDENTITY_RULE,
            "corpus_manifest_sha256": sha256_file(c.processed_dir / "MANIFEST.json")}


def assert_corpus_identity(c: Corpus, split: dict, extra: dict, where: str = "") -> None:
    """Hard check of a split against the corpus on disk. Never warns: a stale pin is a failed run, not a note.

    1. the three id lists must cover EXACTLY the corpus subject set;
    2. the recorded stable corpus-identity hash must equal the current one.
    A D1-protocol manifest without a pin is itself a failure. The one manifest allowed to carry no pin is the frozen
    pre-D1 WildPPG A4 split, which must not be regenerated (Corpus.trains = False) and so can never acquire one.
    """
    where = where or str(c.manifest_path)
    subs = set(corpus_subjects(c))
    cover = {str(v) for k in ("train", "val", "test") for v in split.get(k, [])}
    if cover != subs:
        raise CorpusIdentityError(
            f"{where}: split covers {len(cover)} subjects but corpus {c.processed} has {len(subs)} — "
            f"missing from split {sorted(subs - cover)[:8]}, unknown to corpus {sorted(cover - subs)[:8]}")
    rec = extra.get("corpus_identity_sha256")
    if rec is None:
        if str(split.get("protocol", "")) == "D1-subject-holdout":
            raise CorpusIdentityError(f"{where}: no corpus_identity_sha256 pin; re-emit with "
                                      f"`.venv/bin/python scripts/d1_common.py --refresh`")
        return
    cur = corpus_identity_sha256(c)
    if str(rec) != cur:
        raise CorpusIdentityError(
            f"{where}: corpus identity changed — recorded {rec}, {c.processed} on disk {cur}. The split was written "
            f"against a different corpus build; re-emit with `scripts/d1_common.py --refresh` (which refuses to move "
            f"any subject between train/val/test) and re-verify every result that used it.")


def ensure_split_manifest(c: Corpus, write: bool = True, refresh: bool = False) -> dict:
    """Return the corpus split, writing data/manifests/split_d1_<key>_seed42.json on first use. WildPPG reuses A4.

    Every read is validated against the corpus on disk by `assert_corpus_identity`. `refresh=True` re-derives the
    split from the current corpus and re-emits the manifest, refusing to write if the partition would move.
    """
    if c.manifest_path.exists() and not refresh:
        split, extra = read_split_manifest(c.manifest_path)
        assert_no_forbidden_subjects(split, str(c.manifest_path))
        assert_corpus_identity(c, split, extra)
        return split
    if not c.trains:
        raise FileNotFoundError(f"{c.key}: the frozen manifest {c.manifest_path} is missing and must not be regenerated")
    split = make_d1_split(corpus_subjects(c), c.key)
    if c.manifest_path.exists():
        old = read_split_manifest(c.manifest_path)[0]
        moved = {k: (list(old.get(k, [])), list(split[k])) for k in ("train", "val", "test") if list(old.get(k, [])) != list(split[k])}
        if moved:
            raise SplitPartitionChanged(f"{c.manifest_path}: re-emitting would MOVE subjects — {moved}. The §5 rule "
                                        f"is frozen; stop and report this rather than overwriting the partition.")
    if write:
        write_manifest(c.manifest_path, split, split_provenance(c))
        assert_corpus_identity(c, *read_split_manifest(c.manifest_path))
    return split


def split_sizes(c: Corpus, split: dict) -> dict:
    """Subjects and windows per partition, from the corpus MANIFEST (no array is loaded)."""
    wc = window_counts(c)
    return {k: {"n_subjects": len(split[k]), "n_windows": int(sum(wc[s] for s in split[k]))} for k in ("train", "val", "test")}


# ----------------------------------------------------------------------------------------------------------------------
# Training argv (§3): identical hyperparameters, only the corpus paths differ
# ----------------------------------------------------------------------------------------------------------------------
def train_argv(c: Corpus, resume: bool = False) -> list[str]:
    if not c.included:
        raise ValueError(f"{c.key} is excluded from the ECG benchmark (target={c.target})")
    if not c.trains:
        raise ValueError(f"{c.key} is not trained by D1; it reuses {c.checkpoint}")
    argv = ["-m", "ppg2ecg.training.train_a2", "--exp-name", c.exp_name, "--out-dir", str(c.out_dir),
            "--processed", c.processed, "--manifest", c.manifest]
    for k, v in TRAIN_HP.items():
        argv += [f"--{k}", str(v)]
    if resume:
        argv.append("--resume")
    return argv


# ----------------------------------------------------------------------------------------------------------------------
# Uncertainty (§6): subject-clustered bootstrap
# ----------------------------------------------------------------------------------------------------------------------
def subject_cluster_bootstrap(values, subjects, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    """Cluster bootstrap over SUBJECTS for one metric on one corpus.

    Unlike `paired_stats.paired_subject_bootstrap` (which holds the subject set fixed because arms are paired
    window-by-window), D1 makes no arm-vs-arm comparison, so the resampling unit is the subject itself: subjects
    are drawn with replacement and each contributes its own window mean.

      primary   `subject_macro_mean` = mean over subjects of that subject's window mean (equal subject weight)
      secondary `pooled_window_mean` = mean over all windows (window weight)
    The two are returned side by side and are never merged.
    """
    v = np.asarray(values, dtype=np.float64)
    s = np.asarray(subjects)
    uniq = sorted(set(s.tolist()), key=natural_key)
    with np.errstate(invalid="ignore"):
        means = np.array([np.nanmean(v[s == u]) if np.isfinite(v[s == u]).any() else np.nan for u in uniq])
        counts = np.array([int(np.isfinite(v[s == u]).sum()) for u in uniq], dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(int(n_boot), len(uniq)))
    with np.errstate(invalid="ignore"):
        macro = np.nanmean(means[idx], axis=1)
        w = counts[idx]
        pooled = np.nansum(means[idx] * w, axis=1) / np.where(np.nansum(w, axis=1) > 0, np.nansum(w, axis=1), np.nan)
        point_macro = float(np.nanmean(means))
        point_pooled = float(np.nansum(means * counts) / counts.sum()) if counts.sum() > 0 else float("nan")
        mlo, mhi = (float(x) for x in np.nanpercentile(macro, [2.5, 97.5]))
        plo, phi = (float(x) for x in np.nanpercentile(pooled, [2.5, 97.5]))
    return {"subject_macro_mean": point_macro, "macro_ci_lo": mlo, "macro_ci_hi": mhi,
            "pooled_window_mean": point_pooled, "pooled_ci_lo": plo, "pooled_ci_hi": phi,
            "n_subjects": len(uniq), "n_windows": int(np.isfinite(v).sum()), "n_boot": int(n_boot), "seed": int(seed)}


# ----------------------------------------------------------------------------------------------------------------------
# Self-check
# ----------------------------------------------------------------------------------------------------------------------
def refresh_manifests() -> int:
    """Re-emit every D1 split manifest against the current corpora, printing the before/after partition."""
    rc = 0
    for c in CORPORA:
        if not c.included or not c.trains:
            print(f"[{c.key}] skipped (excluded={not c.included}, frozen manifest={not c.trains})")
            continue
        before = read_split_manifest(c.manifest_path)[0] if c.manifest_path.exists() else {}
        try:
            after = ensure_split_manifest(c, write=True, refresh=True)
        except SplitPartitionChanged as e:
            print(f"[{c.key}] PARTITION WOULD CHANGE — nothing written: {e}")
            rc = 1
            continue
        same = all(list(before.get(k, [])) == list(after[k]) for k in ("train", "val", "test"))
        print(f"[{c.key}] partition identical: {same} | corpus_identity_sha256 {corpus_identity_sha256(c)}")
        for k in ("train", "val", "test"):
            print(f"    before {k:5s} ({len(before.get(k, []))}): {list(before.get(k, []))}")
            print(f"    after  {k:5s} ({len(after[k])}): {list(after[k])}")
        rc |= 0 if same else 1
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="print (and optionally write) the resolved D1 benchmark configuration")
    ap.add_argument("--write-manifests", action="store_true", help="write any missing data/manifests/split_d1_<key>_seed42.json")
    ap.add_argument("--refresh", action="store_true",
                    help="re-emit every D1 split manifest against the CURRENT corpora (refreshes the provenance pins); "
                         "prints the before/after id lists and refuses to write if any partition would move")
    args = ap.parse_args()
    if args.refresh:
        return refresh_manifests()
    print(f"D1 benchmark | root {ROOT} | seed {SEED} | NFE {list(NFE_GRID)} | bootstrap {BOOTSTRAP_N}@{BOOTSTRAP_SEED}")
    print(f"forbidden subjects (never train/val): {list(FORBIDDEN_SUBJECTS)}")
    print(f"split rule: {SPLIT_RULE}\n")
    print(f"{'key':10s} {'target':6s} {'fsPPG/fsECG':12s} {'trains':6s} {'processed':32s} {'built':5s} {'manifest':6s}")
    for c in CORPORA:
        print(f"{c.key:10s} {c.target:6s} {f'{c.fs_ppg}/{c.fs_ecg} Hz':12s} {str(c.trains):6s} {c.processed:32s} "
              f"{str((c.processed_dir / 'MANIFEST.json').exists()):5s} {str(c.manifest_path.exists()):6s}")
    print()
    for c in CORPORA:
        head = f"[{c.key}] {c.name} — {c.citation}"
        if not c.included:
            print(f"{head}\n    EXCLUDED (target={c.target}): {c.notes}\n")
            continue
        if not (c.processed_dir / "MANIFEST.json").exists():
            print(f"{head}\n    processed corpus not built yet: {c.processed_dir} — split and sizes pending\n")
            continue
        split = ensure_split_manifest(c, write=args.write_manifests) if (args.write_manifests or c.manifest_path.exists()) else make_d1_split(corpus_subjects(c), c.key)
        assert_no_forbidden_subjects(split, f"d1_common self-check [{c.key}]")
        sz = split_sizes(c, split)
        small = sz["train"]["n_windows"] < MIN_TRAIN_WINDOWS
        print(head)
        print(f"    subjects {len(corpus_subjects(c))} | windows {sum(window_counts(c).values())} | manifest "
              f"{c.manifest} ({'on disk' if c.manifest_path.exists() else 'NOT WRITTEN'})")
        for k in ("train", "val", "test"):
            print(f"    {k:5s} {sz[k]['n_subjects']:4d} subjects {sz[k]['n_windows']:8d} windows  {split[k] if len(split[k]) <= 12 else str(split[k][:12]) + ' ...'}")
        print(f"    small_corpus={small} (threshold {MIN_TRAIN_WINDOWS} train windows) | checkpoint {c.checkpoint_path.relative_to(ROOT)} "
              f"({'present' if c.checkpoint_path.exists() else 'absent'})")
        print(f"    train argv: {'(not trained by D1) ' + (c.notes or '') if not c.trains else ' '.join(train_argv(c))}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
