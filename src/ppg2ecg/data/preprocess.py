"""Window-level preprocessing — a line-for-line re-statement of upstream `preprocess()` (external/PENGUIN/src/preprocess.py L14-62).

Every statistic is computed PER WINDOW from that window alone (z-score over axis=1, min-max over axis=1).
Nothing is fitted on the training set; nothing from the target is used to normalise the input.
tests/test_preprocess_equivalence.py checks numerical equality against the upstream function.
"""
from __future__ import annotations

import numpy as np
import scipy.stats
from scipy import signal


def preprocess_windows(
    x: np.ndarray,
    resample_rate: int = 128,
    segment_len: int = 4,
    bandpass: bool = True,
    freq_range=(0.5, 4),
    zscore: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """x: [n_windows, native_len] -> [n_windows, resample_rate*segment_len] (float64, same as upstream)."""
    x = signal.resample(x, resample_rate * segment_len, axis=1)  # FFT resampling, per window
    if bandpass:
        nyq = 0.5 * resample_rate
        low, high = freq_range[0] / nyq, freq_range[1] / nyq
        if freq_range[0] < 0:
            b, a = signal.butter(4, high, btype="low")
        elif freq_range[1] < 0:
            b, a = signal.butter(4, low, btype="high")
        else:
            b, a = signal.butter(4, [low, high], btype="band")
        x = signal.filtfilt(b, a, x)  # zero-phase, default padlen, per window
    if zscore:
        x = scipy.stats.zscore(x, axis=1)
    if normalize:
        mn = x.min(axis=1, keepdims=True)
        mx = x.max(axis=1, keepdims=True)
        x = (x - mn) / (mx - mn + 1e-8) * 2 - 1
    return x


# Shipped PPG-DaLiA settings (configs/upstream/preprocess.yaml)
PPG_KW = dict(bandpass=True, freq_range=(0.5, 4), zscore=True, normalize=True)
ECG_KW = dict(bandpass=True, freq_range=(0.5, -1), zscore=True, normalize=True)  # (0.5,-1) => 0.5 Hz high-pass
