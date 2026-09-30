"""BF0 renderer: weighted overlap-add of R-aligned beats at given positions (prereg §2.5–2.6)."""
from __future__ import annotations

import numpy as np

from .beats import R_INDEX, SEG_LEN

WEIGHTS = np.hanning(SEG_LEN + 2)[1:-1]      # Hann of length 168 without its two zero end points: all > 0
EPS_DEN = 1e-6


def assemble(beats, positions, n_time: int, empty_fill: float) -> tuple[np.ndarray, dict]:
    """y(t) = sum_i w_i(t) b_i(t) / sum_i w_i(t); uncovered samples take the nearest covered value (ties -> left).

    `beats`: iterable of [166] arrays, `positions`: R indices. A window without beats is filled with `empty_fill`.
    """
    num = np.zeros(int(n_time))
    den = np.zeros(int(n_time))
    for b, p in zip(beats, positions):
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        assert b.size == SEG_LEN, b.size
        idx = np.arange(SEG_LEN) + int(p) - R_INDEX
        ok = (idx >= 0) & (idx < n_time)
        num[idx[ok]] += WEIGHTS[ok] * b[ok]
        den[idx[ok]] += WEIGHTS[ok]
    covered = den > EPS_DEN
    if not covered.any():
        return np.full(int(n_time), float(empty_fill)), {"empty": True, "n_uncovered": int(n_time)}
    y = np.zeros(int(n_time))
    y[covered] = num[covered] / den[covered]
    cov_idx = np.flatnonzero(covered)
    unc = np.flatnonzero(~covered)
    if unc.size:
        pos = np.searchsorted(cov_idx, unc)
        left = cov_idx[np.clip(pos - 1, 0, cov_idx.size - 1)]
        right = cov_idx[np.clip(pos, 0, cov_idx.size - 1)]
        pick = np.where(np.abs(unc - left) <= np.abs(right - unc), left, right)
        y[unc] = y[pick]
    return y, {"empty": False, "n_uncovered": int(unc.size)}


def render_template(template, positions, n_time: int, empty_fill: float) -> np.ndarray:
    """The N1-matched baseline: the same assembly with one fixed template at every position."""
    t = np.asarray(template, dtype=np.float64)
    return assemble([t] * len(positions), positions, n_time, empty_fill)[0]


BW_BEFORE, BW_AFTER = 32, 51                 # rpeaks.beat_window(0.25 s, 0.40 s) at 128 Hz: 83 samples, R at 32
BW_LEN = BW_BEFORE + BW_AFTER


def beat_windows(sig, positions) -> np.ndarray:
    """The 83-sample windows sig[r-32 : r+51] of every position that lies fully inside the signal."""
    x = np.asarray(sig, dtype=np.float64).reshape(-1)
    rows = [x[int(p) - BW_BEFORE:int(p) + BW_AFTER] for p in np.asarray(positions, dtype=int).reshape(-1)
            if int(p) - BW_BEFORE >= 0 and int(p) + BW_AFTER <= x.size]
    return np.asarray(rows, dtype=np.float64).reshape(-1, BW_LEN)


def mean_pairwise_rms(W) -> float:
    """Mean RMS difference over all pairs of rows (nan with fewer than 2 rows)."""
    W = np.asarray(W, dtype=np.float64)
    if W.shape[0] < 2:
        return float("nan")
    d = W[:, None, :] - W[None, :, :]
    rms = np.sqrt((d ** 2).mean(axis=2))
    iu = np.triu_indices(W.shape[0], k=1)
    return float(rms[iu].mean())


def within_window_diversity(sig, positions) -> float:
    """D: mean pairwise RMS between the 83-sample beat windows of one waveform."""
    return mean_pairwise_rms(beat_windows(sig, positions))


def matched_pairs(ref_peaks, positions, n_time: int, fs: int = 128, tol_ms: float = 50.0) -> list[tuple[int, int]]:
    """(reference R, placed position) pairs, greedy one-to-one within ±tol_ms (`rpeaks.match_rpeaks`), whose 83-sample
    windows both lie inside the signal. Depends only on (reference, positions): every arm that shares the positions
    shares the pairs."""
    from ppg2ecg.evaluation.rpeaks import match_rpeaks
    ref = np.asarray(ref_peaks, dtype=int).reshape(-1)
    pos = np.asarray(positions, dtype=int).reshape(-1)
    out = []
    for i, j in match_rpeaks(ref, pos, fs, tol_ms)[0]:
        r, p = int(ref[i]), int(pos[j])
        if r - BW_BEFORE >= 0 and r + BW_AFTER <= n_time and p - BW_BEFORE >= 0 and p + BW_AFTER <= n_time:
            out.append((r, p))
    return out


def pair_correlations(gt_sig, arm_sig, pairs) -> np.ndarray:
    """Pearson correlation between the reference window at r and the arm window at p, per pair (nan if either is flat)."""
    g = np.asarray(gt_sig, dtype=np.float64).reshape(-1)
    a = np.asarray(arm_sig, dtype=np.float64).reshape(-1)
    out = np.full(len(pairs), np.nan)
    for k, (r, p) in enumerate(pairs):
        u, v = g[r - BW_BEFORE:r + BW_AFTER], a[p - BW_BEFORE:p + BW_AFTER]
        if u.std() > 1e-8 and v.std() > 1e-8:
            out[k] = float(np.corrcoef(u, v)[0, 1])
    return out


def boundary_statistic(sig, peaks, exclude: int = 12) -> float:
    """Max |first difference| outside ±`exclude` samples of every peak (nan if nothing is left)."""
    x = np.asarray(sig, dtype=np.float64).reshape(-1)
    d = np.abs(np.diff(x))
    keep = np.ones(d.size, bool)
    for p in np.asarray(peaks, dtype=int).reshape(-1):
        keep[max(0, p - exclude): min(d.size, p + exclude + 1)] = False
    return float(d[keep].max()) if keep.any() else float("nan")
