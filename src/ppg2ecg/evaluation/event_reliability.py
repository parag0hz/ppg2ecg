"""X4-0 event-reliability primitives (docs/X4_0_EVENT_RELIABILITY_PREREGISTRATION.md).

FROZEN-INFERENCE / ANALYSIS ONLY. Nothing here trains or modifies a model.

Every event/oracle semantic is REUSED from X0 (`ppg2ecg.evaluation.rpeaks`, `ppg2ecg.evaluation.alignment_diagnostics`);
this module only adds (a) deterministic subset hashing, (b) predicted-vs-predicted event matching for source-seed
consistency, (c) GT-anchored detection/timing statistics across sources, (d) non-uniform MeanFlow interval schedules, and
(e) a subject-stratified window bootstrap.

Terminology fixed by the pre-registration: exact h = 1 has ZERO training probability and is an extreme boundary query;
"outside the support" is not used. Oracle translation is diagnosis only, never deployable performance.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.alignment_diagnostics import FS, HF_CUT_HZ, LOCAL_MAX_SHIFT_MS, QRS_HALF_MS  # noqa: F401  (parity constants)

MATCH_TOL_MS = 50.0          # X0's frozen one-to-one event tolerance
GT_ANCHOR_MS = 150.0         # X0's oracle local-translation window, reused for GT-anchored presence
TEST_SUBJECTS = ("kjd", "ssx")
PREVIEWED_WINDOWS = (("an0", 9066), ("an0", 18138), ("k2s", 5852), ("k2s", 16436))


class WildPPGTestFirewallError(RuntimeError):
    """Raised when X4-0 code is asked to load a WildPPG test subject."""


def assert_no_test_subjects(subjects) -> None:
    bad = sorted(set(map(str, subjects)) & set(TEST_SUBJECTS))
    if bad:
        raise WildPPGTestFirewallError(f"X4-0 must never load WildPPG test subjects; got {bad}")


# ----------------------------------------------------------------------------------------------------------------------
# Deterministic, outcome-independent subset selection
# ----------------------------------------------------------------------------------------------------------------------
def window_hash(salt: str, subject: str, window_index: int) -> str:
    return hashlib.sha256(f"{salt}|{subject}|{int(window_index)}".encode()).hexdigest()


def select_subset(salt: str, subject: str, n_windows_total: int, n_take: int, exclude=PREVIEWED_WINDOWS) -> np.ndarray:
    """Smallest-`n_take` SHA256 hashes over (subject, original window index), excluding pre-viewed windows."""
    excl = {int(w) for s, w in exclude if s == subject}
    idx = np.array([i for i in range(int(n_windows_total)) if i not in excl], dtype=np.int64)
    order = np.argsort([window_hash(salt, subject, int(i)) for i in idx], kind="stable")
    return np.sort(idx[order[:int(n_take)]])


# ----------------------------------------------------------------------------------------------------------------------
# MeanFlow sampling with an explicit (possibly non-uniform) interval schedule
# ----------------------------------------------------------------------------------------------------------------------
def schedule_times(h_list) -> np.ndarray:
    """Chronological sampler times t_0 = 1 (noise) -> t_N = 0 (data) implied by the interval list."""
    h = np.asarray(h_list, dtype=np.float64)
    if h.ndim != 1 or len(h) == 0:
        raise ValueError("h_list must be a non-empty 1-D sequence")
    if not np.isclose(h.sum(), 1.0, atol=1e-9):
        raise ValueError(f"interval schedule must sum to 1, got {h.sum()!r}")
    return np.concatenate([[1.0], 1.0 - np.cumsum(h)])


@torch.no_grad()
def sample_meanflow_schedule(net, ppg: torch.Tensor, e: torch.Tensor, h_list):
    """z_r = z_t - (t-r) u(z_t, ppg, t, t-r) along an explicit interval schedule. NFE = len(h_list).

    h_list = [1/N]*N reproduces the standard uniform N-step sampler bit-exactly.
    """
    ts = schedule_times(h_list)
    if not np.isclose(ts[-1], 0.0, atol=1e-9):
        raise ValueError(f"schedule must terminate at t = 0, got {ts[-1]!r}")
    b = e.shape[0]
    z, nfe = e, 0
    for i in range(len(ts) - 1):
        t = torch.full((b, 1), float(ts[i]), device=e.device)
        r = torch.full((b, 1), float(ts[i + 1]), device=e.device)
        z = z - (t - r).reshape(-1, 1, 1) * net.u(z, ppg, t, t - r)
        nfe += 1
    return z, nfe


UNIFORM = {n: [1.0 / n] * n for n in (1, 2, 4, 8, 16, 25, 50)}
SCHEDULES = {
    "U4": [0.25] * 4,
    "LN4": [0.70, 0.10, 0.10, 0.10],
    "LD4": [0.10, 0.10, 0.10, 0.70],
    "U8": [0.125] * 8,
    "LN8": [0.50] + [0.50 / 7] * 7,
    "LD8": [0.50 / 7] * 7 + [0.50],
}


# ----------------------------------------------------------------------------------------------------------------------
# Event matching between two PREDICTED peak trains (no ground truth involved)
# ----------------------------------------------------------------------------------------------------------------------
def peak_train_agreement(a, b, fs: int = FS, tol_ms: float = MATCH_TOL_MS) -> dict:
    """One-to-one match of two predicted peak trains with the frozen X0 matcher. Returns precision/recall/F1.

    Symmetric by construction in F1; `a` is treated as the reference for precision/recall naming.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if len(a) == 0 and len(b) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_matched": 0, "n_a": 0, "n_b": 0}
    if len(a) == 0 or len(b) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_matched": 0, "n_a": len(a), "n_b": len(b)}
    pairs, _n_missing, _n_spurious = R.match_rpeaks(a, b, fs, tol_ms=tol_ms)  # X0 contract: (pairs, missing, spurious)
    matched = len(pairs)
    prec = matched / len(b)
    rec = matched / len(a)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "n_matched": int(matched),
            "n_a": int(len(a)), "n_b": int(len(b))}


def gt_anchored_presence(gt_peaks, pred_peaks_per_source, fs: int = FS, window_ms: float = GT_ANCHOR_MS) -> dict:
    """For each GT beat: fraction of sources with a predicted peak within +/- window_ms, and the timing SD of those matches.

    Timing SD is only meaningful where enough sources detect the beat; the caller applies the frozen >= 16/32 filter.
    """
    half = window_ms / 1000.0 * fs
    gt = np.asarray(gt_peaks, dtype=float)
    n_src = len(pred_peaks_per_source)
    det = np.zeros(len(gt))
    offs: list[list[float]] = [[] for _ in gt]
    for pk in pred_peaks_per_source:
        pk = np.asarray(pk, dtype=float)
        for i, g in enumerate(gt):
            if len(pk) == 0:
                continue
            d = pk - g
            j = int(np.argmin(np.abs(d)))
            if abs(d[j]) <= half:
                det[i] += 1
                offs[i].append(float(d[j]) / fs * 1000.0)
    return {"detection_probability": det / max(n_src, 1), "n_detected": det.astype(int), "n_sources": int(n_src),
            "timing_sd_ms": np.array([np.std(o) if len(o) > 1 else np.nan for o in offs]),
            "timing_mean_ms": np.array([np.mean(o) if len(o) else np.nan for o in offs])}


# ----------------------------------------------------------------------------------------------------------------------
# Uncertainty: subject-stratified window bootstrap (descriptive, NOT population-confirmatory)
# ----------------------------------------------------------------------------------------------------------------------
def subject_stratified_bootstrap(values: np.ndarray, subjects: np.ndarray, n_boot: int = 2000, seed: int = 20260830):
    """Resample windows WITHIN each subject, then average the subject means with EQUAL weight."""
    values = np.asarray(values, dtype=np.float64)
    subjects = np.asarray(subjects)
    ok = np.isfinite(values)
    values, subjects = values[ok], subjects[ok]
    labs = np.unique(subjects)
    if len(values) == 0 or len(labs) == 0:
        return float("nan"), float("nan"), float("nan")
    groups = [values[subjects == s] for s in labs]
    point = float(np.mean([g.mean() for g in groups]))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = np.mean([g[rng.integers(0, len(g), len(g))].mean() for g in groups])
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ----------------------------------------------------------------------------------------------------------------------
# Event-matching tolerance calibration (metric calibration, NOT model evaluation, NOT an upper bound)
# ----------------------------------------------------------------------------------------------------------------------
def jitter_peaks(peaks, sd_ms: float, fs: int = FS, rng: np.random.Generator | None = None, shift_ms: float = 0.0,
                 n_time: int | None = None) -> np.ndarray:
    """Perturb a GT peak train with zero-mean Gaussian timing jitter and/or a fixed shift. Peak COUNT is preserved."""
    pk = np.asarray(peaks, dtype=float)
    if len(pk) == 0:
        return pk.astype(int)
    d = (rng.normal(0.0, sd_ms, len(pk)) if (rng is not None and sd_ms > 0) else np.zeros(len(pk))) + shift_ms
    out = np.round(pk + d / 1000.0 * fs).astype(int)
    if n_time is not None:
        out = np.clip(out, 0, n_time - 1)
    return np.sort(out)
