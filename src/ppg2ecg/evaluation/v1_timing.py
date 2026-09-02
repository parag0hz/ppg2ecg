"""V1 cohorts and the ECG-R -> PPG-pulse delay audit.

Frozen by docs/V1_STEPWISE_VISUALIZATION_PREREGISTRATION.md (a73cafa).

Cohort selection is METADATA ONLY. The delay audit uses ground-truth R-peaks and independently detected
PPG systolic peaks; nothing is shifted, and R-peaks serve as coordinate references only.
"""
from __future__ import annotations

import hashlib

import numpy as np

FS = 128
SALT = "v1-all-subject-stepwise-visualization"
VIZ_N, METRICS_N, DELAY_N = 8, 32, 128
TRAIN = ("e61", "fex", "l38", "n31", "ngh", "p5d", "p9p", "qm9", "trh", "tz8", "u7y", "w4p")
VAL = ("an0", "k2s")
SITES = ("sternum", "head", "wrist", "ankle")
#: delay search window, one-to-one, forward in time
DELAY_LO_MS, DELAY_HI_MS = 80.0, 800.0
#: PPG foot proxy: backward search span and its abandonment threshold
FOOT_BACK_MS, FOOT_FAIL_ABORT = 400.0, 0.20


def _key(subject: str, site: str, widx: int) -> str:
    return hashlib.sha256(f"{SALT}|{subject}|{site}|{int(widx)}".encode()).hexdigest()


def rank_within_stratum(subject: str, sites: np.ndarray, window_index: np.ndarray, site: str) -> np.ndarray:
    """Positions into the subject's arrays for `site`, ordered by SHA256 rank (metadata only)."""
    m = np.flatnonzero(np.asarray(sites) == site)
    if m.size == 0:
        return m
    keys = [_key(subject, site, int(window_index[i])) for i in m]
    return m[np.argsort(keys, kind="stable")]


def cohorts(subject: str, sites: np.ndarray, window_index: np.ndarray) -> dict:
    """Nested VIZ subset METRICS subset DELAY, per site, as prefixes of one ranking."""
    out = {}
    for site in SITES:
        r = rank_within_stratum(subject, sites, window_index, site)
        out[site] = {"viz": np.sort(r[:VIZ_N]), "metrics": np.sort(r[:METRICS_N]),
                     "delay": np.sort(r[:DELAY_N]), "available": int(r.size)}
    return out


def match_r_to_ppg(r_peaks: np.ndarray, ppg_peaks: np.ndarray, n_time: int, fs: int = FS):
    """First subsequent PPG systolic peak in [80, 800] ms, one-to-one, forward in time.

    Returns (pairs[(r_idx, p_idx)], n_excluded_boundary, n_unmatched).
    A beat whose search window would leave the record is excluded, not counted as unmatched.
    """
    r = np.sort(np.asarray(r_peaks, dtype=np.int64))
    p = np.sort(np.asarray(ppg_peaks, dtype=np.int64))
    lo = int(round(DELAY_LO_MS / 1000.0 * fs))
    hi = int(round(DELAY_HI_MS / 1000.0 * fs))
    pairs, used, bnd, unm = [], np.zeros(p.size, dtype=bool), 0, 0
    for i, rr in enumerate(r):
        if rr + hi >= n_time:                      # search window leaves the record
            bnd += 1
            continue
        cand = np.flatnonzero((p >= rr + lo) & (p <= rr + hi) & (~used))
        if cand.size == 0:
            unm += 1
            continue
        j = int(cand[0])                           # FIRST subsequent, and never reused
        used[j] = True
        pairs.append((i, j))
    return pairs, bnd, unm


def ppg_foot(ppg: np.ndarray, peak: int, prev_peak: int | None, fs: int = FS) -> int | None:
    """Onset proxy: argmin of the PPG in a fixed 400 ms backward window, strictly before `peak`
    and after `prev_peak`. Returns None when no valid region exists."""
    back = int(round(FOOT_BACK_MS / 1000.0 * fs))
    lo = max(0, peak - back)
    if prev_peak is not None:
        lo = max(lo, int(prev_peak) + 1)
    if peak - lo < 2:
        return None
    return int(lo + np.argmin(np.asarray(ppg, dtype=np.float64)[lo:peak]))


def delay_summary(delay_ms: np.ndarray) -> dict:
    d = np.asarray(delay_ms, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {k: np.nan for k in ("n", "median", "mean", "sd", "iqr", "mad", "p5", "p95", "cv")} | {"n": 0}
    q1, q3 = np.percentile(d, [25, 75])
    med = float(np.median(d))
    return {"n": int(d.size), "median": med, "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if d.size > 1 else np.nan,
            "iqr": float(q3 - q1), "mad": float(np.median(np.abs(d - med))),
            "p5": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
            "cv": float(d.std(ddof=1) / d.mean()) if d.size > 1 and d.mean() != 0 else np.nan}


def nearest_abs_error_ms(true_r: np.ndarray, pred_r: np.ndarray, fs: int = FS) -> np.ndarray:
    """For every TRUE R-peak, the distance to the nearest PREDICTED R location, in ms."""
    t = np.asarray(true_r, dtype=np.float64)
    p = np.asarray(pred_r, dtype=np.float64)
    if t.size == 0 or p.size == 0:
        return np.full(t.size, np.inf)
    return np.min(np.abs(t[:, None] - p[None, :]), axis=1) / fs * 1000.0
