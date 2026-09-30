"""BF0 beat geometry and training-beat extraction (prereg §2.4).

A segment is R-aligned: 64 samples before R and 101 after, length 166, R at index 64 (128 Hz).
"""
from __future__ import annotations

import numpy as np

FS = 128
N_BEFORE, N_AFTER = 64, 101
SEG_LEN = N_BEFORE + 1 + N_AFTER            # 166
R_INDEX = N_BEFORE                          # 64


def segment_bounds(r: int) -> tuple[int, int]:
    """[start, stop) of the segment around R-peak index `r`."""
    return int(r) - N_BEFORE, int(r) + N_AFTER + 1


def inside(r: int, n_time: int) -> bool:
    a, b = segment_bounds(r)
    return a >= 0 and b <= int(n_time)


def padded_segment(sig: np.ndarray, r: int) -> np.ndarray:
    """sig[r-64 : r+102], edge-padded with the edge value where it leaves the window."""
    sig = np.asarray(sig)
    idx = np.clip(np.arange(int(r) - N_BEFORE, int(r) + N_AFTER + 1), 0, sig.size - 1)
    return sig[idx]


def neighbour_rr(peaks, fallback_s: float, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """RR_prev, RR_next in seconds for each (sorted) peak.

    A neighbour outside the window is replaced by the window's median RR (>= 2 peaks); a single peak gets
    `fallback_s` for both.
    """
    p = np.sort(np.asarray(peaks, dtype=np.float64).reshape(-1))
    n = p.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    if n == 1:
        return np.array([float(fallback_s)]), np.array([float(fallback_s)])
    d = np.diff(p) / fs
    med = float(np.median(d))
    prev = np.concatenate([[med], d])
    nxt = np.concatenate([d, [med]])
    return prev, nxt


def extract_training_beats(X, Y, peaks_list, fs: int = FS):
    """Every reference R whose segment lies fully inside its window, from windows with >= 2 reference R.

    Returns ecg [N, 166], ppg [N, 166] (float32), rr [N, 2] (RR_prev, RR_next in s, float32), window index [N].
    """
    E, P, RR, W = [], [], [], []
    for n, (x, y, pk) in enumerate(zip(X, Y, peaks_list)):
        pk = np.sort(np.asarray(pk, dtype=int).reshape(-1))
        if pk.size < 2:
            continue
        prev, nxt = neighbour_rr(pk, np.nan, fs)
        for j, r in enumerate(pk):
            if not inside(r, len(y)):
                continue
            a, b = segment_bounds(r)
            E.append(np.asarray(y[a:b], dtype=np.float32))
            P.append(np.asarray(x[a:b], dtype=np.float32))
            RR.append((prev[j], nxt[j]))
            W.append(n)
    if not E:
        z = np.zeros((0, SEG_LEN), np.float32)
        return z, z.copy(), np.zeros((0, 2), np.float32), np.zeros(0, int)
    return np.stack(E), np.stack(P), np.asarray(RR, np.float32), np.asarray(W, int)
