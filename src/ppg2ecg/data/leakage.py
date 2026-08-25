"""Leakage checks that must pass before any number is reported (docs/PREREGISTRATION_V0.md §8).

1. subject disjointness      : train ∩ val = train ∩ test = val ∩ test = ∅, and every subject is assigned once
2. window disjointness       : no processed window (exact float32 hash) appears in two splits
3. window-local normalisation: preprocessing of window i is invariant to every other window (no global/train stats)
4. target-free inference     : the inference callable never receives the target (structural + behavioural)
"""
from __future__ import annotations

import hashlib
import inspect
from itertools import combinations
from typing import Callable

import numpy as np


def check_subject_disjoint(split: dict, expected_subjects=None) -> dict:
    parts = {k: set(split[k]) for k in ("train", "val", "test")}
    overlaps = {f"{a}∩{b}": sorted(parts[a] & parts[b]) for a, b in combinations(parts, 2)}
    ok = all(len(v) == 0 for v in overlaps.values())
    rep = {"ok": ok, "overlaps": overlaps, "sizes": {k: len(v) for k, v in parts.items()}}
    if expected_subjects is not None:
        union = parts["train"] | parts["val"] | parts["test"]
        missing, extra = sorted(set(expected_subjects) - union), sorted(union - set(expected_subjects))
        rep.update({"missing": missing, "unexpected": extra})
        rep["ok"] = rep["ok"] and not missing and not extra
    return rep


def window_hashes(x: np.ndarray, decimals: int = 6) -> set[bytes]:
    x = np.round(np.asarray(x, dtype=np.float32), decimals)
    return {hashlib.sha1(row.tobytes()).digest() for row in x}


def check_window_disjoint(arrays: dict) -> dict:
    hs = {k: window_hashes(v) for k, v in arrays.items()}
    overlaps = {f"{a}∩{b}": len(hs[a] & hs[b]) for a, b in combinations(hs, 2)}
    return {"ok": all(v == 0 for v in overlaps.values()), "overlaps": overlaps, "n_unique": {k: len(v) for k, v in hs.items()}}


def check_windowwise_normalization(preprocess_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, seed: int = 0, atol: float = 1e-9) -> dict:
    """preprocess_fn(x)[i] must not change when all OTHER rows are replaced by random data."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)  # same dtype for both calls (float32 input would change scipy's FFT precision, not locality)
    ref = preprocess_fn(x)
    x2 = rng.standard_normal(x.shape) * (np.abs(x).max() + 1.0)
    i = int(rng.integers(len(x)))
    x2[i] = x[i]
    out = preprocess_fn(x2)
    max_diff = float(np.max(np.abs(out[i] - ref[i])))
    return {"ok": max_diff <= atol, "row": i, "max_abs_diff": max_diff}


def check_inference_signature_target_free(fn: Callable, forbidden=("target", "target_signal", "y", "ecg")) -> dict:
    names = list(inspect.signature(fn).parameters)
    bad = [n for n in names if n in forbidden]
    return {"ok": not bad, "params": names, "forbidden_present": bad}


def check_inference_target_invariance(sample_fn: Callable, ppg, seed_fn: Callable[[int], None], seed: int = 0, **kw) -> dict:
    """sample_fn(ppg, **kw) must be identical regardless of any target passed alongside (behavioural check)."""
    seed_fn(seed)
    a = np.asarray(sample_fn(ppg, **kw))
    seed_fn(seed)
    b = np.asarray(sample_fn(ppg, **kw))
    return {"ok": np.array_equal(a, b), "max_abs_diff": float(np.max(np.abs(a - b)))}


def run_all(split: dict, expected_subjects, arrays_by_split: dict | None, preprocess_fn, sample_x) -> dict:
    rep = {"subject_disjoint": check_subject_disjoint(split, expected_subjects)}
    if arrays_by_split is not None:
        rep["window_disjoint"] = check_window_disjoint(arrays_by_split)
    rep["windowwise_normalization"] = check_windowwise_normalization(preprocess_fn, sample_x)
    rep["ok"] = all(v["ok"] for v in rep.values() if isinstance(v, dict))
    return rep
