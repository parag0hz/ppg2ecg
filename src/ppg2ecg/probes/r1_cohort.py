"""R1 cohorts and subject split. METADATA ONLY (docs/R1_PPG_GLOBAL_RHYTHM_OBSERVABILITY_PREREGISTRATION.md, c7481f9)."""
from __future__ import annotations

import hashlib

import numpy as np

FS = 128
TRAIN12 = ("e61", "fex", "l38", "n31", "ngh", "p5d", "p9p", "qm9", "trh", "tz8", "u7y", "w4p")
VAL = ("an0", "k2s")
SITES = ("sternum", "head", "wrist", "ankle")
COHORT_SALT = "r1-global-rhythm-observability-v1"
DEV_SALT = "r1-internal-dev-v1"
VISUAL_SALT = "r1-visual-v1"
N_TRAIN_PER, N_VAL_PER, N_VISUAL_PER = 2048, 1024, 8


def internal_dev_split(train: tuple = TRAIN12) -> dict:
    """Two internal-dev subjects by SHA256 rank; the rest are probe-train."""
    r = sorted(train, key=lambda s: hashlib.sha256(f"{DEV_SALT}|{s}".encode()).hexdigest())
    return {"internal_dev": tuple(r[:2]), "probe_train": tuple(sorted(r[2:]))}


def _rank(salt: str, subject: str, sites: np.ndarray, window_index: np.ndarray, site: str) -> np.ndarray:
    m = np.flatnonzero(np.asarray(sites) == site)
    if m.size == 0:
        return m
    keys = [hashlib.sha256(f"{salt}|{subject}|{site}|{int(window_index[i])}".encode()).hexdigest() for i in m]
    return m[np.argsort(keys, kind="stable")]


def cohort_positions(subject: str, sites: np.ndarray, window_index: np.ndarray, n_per: int,
                     salt: str = COHORT_SALT) -> dict[str, np.ndarray]:
    """Array positions per site, smallest-hash-first, capped at n_per (all if fewer)."""
    return {s: np.sort(_rank(salt, subject, sites, window_index, s)[:n_per]) for s in SITES}


def n_per_for(subject: str) -> int:
    return N_VAL_PER if subject in VAL else N_TRAIN_PER


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of range(n) with no fixed points (n >= 2)."""
    if n < 2:
        raise ValueError("derangement needs n >= 2")
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def circular_offsets(n: int, rng: np.random.Generator, fs: int = FS) -> np.ndarray:
    """Per-window circular-shift offsets, uniform on [1.0, 4.0] s = [128, 512] samples inclusive."""
    lo, hi = int(round(1.0 * fs)), int(round(4.0 * fs))
    return rng.integers(lo, hi + 1, size=n)
