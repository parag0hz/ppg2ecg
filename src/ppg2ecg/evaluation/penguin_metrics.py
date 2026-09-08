"""D3 metric ports — external/PENGUIN/src/utils/help_func.py::compute_metrics, reimplemented exactly.

Frozen by docs/D3_PENGUIN_SIX_DATASET_PREREGISTRATION.md §5. These are PORTS, not redesigns; each is
unit-tested against a hand-computed reference. `ppg2ecg.evaluation.metrics.rr_mae_ms` is a DIFFERENT
quantity (R-R interval error in ms) and is never reported as RR error.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

RESAMPLE_RATE = 128           # cfg.preprocess.resample_rate
RESP_LOWPASS_HZ = 1.0         # help_func.py:191  signal.butter(8, 1/nyq, btype="low")
RESP_ORDER = 8
RESP_WINDOW_S = 60            # train.yaml  window_size.RespRateError


def sbp_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """|max(pred) - max(target)| per window, in the label's own units (mmHg). help_func.py:164-167."""
    p, t = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    return np.abs(p.max(axis=-1) - t.max(axis=-1))


def dbp_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """|min(pred) - min(target)| per window (mmHg). help_func.py:169-172."""
    p, t = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    return np.abs(p.min(axis=-1) - t.min(axis=-1))


def dominant_bpm(x: np.ndarray, fs: int = RESAMPLE_RATE) -> np.ndarray:
    """60 x (positive frequency bin of maximum FFT magnitude). help_func.py:196-206."""
    x = np.atleast_2d(np.asarray(x, np.float64))
    T = x.shape[-1]
    mag = np.abs(np.fft.fft(x, axis=-1))
    fb = np.fft.fftfreq(T, d=1 / fs)
    pos = fb > 0
    return 60.0 * fb[pos][np.argmax(mag[..., pos], axis=-1)]


def resp_rate_error(pred: np.ndarray, target: np.ndarray, fs: int = RESAMPLE_RATE) -> np.ndarray:
    """RespRateError over 60 s windows, in breaths per minute.

    Upstream low-passes the PREDICTION ONLY at 1 Hz with an order-8 Butterworth (`filtfilt`), leaves the
    target untouched, then compares the dominant positive FFT frequency of each. The asymmetry is theirs
    and is preserved deliberately (help_func.py:190-208).
    """
    p, t = np.atleast_2d(np.asarray(pred, np.float64)), np.atleast_2d(np.asarray(target, np.float64))
    b, a = sps.butter(RESP_ORDER, RESP_LOWPASS_HZ / (0.5 * fs), btype="low")
    p = sps.filtfilt(b, a, p, axis=-1)
    return np.abs(dominant_bpm(p, fs) - dominant_bpm(t, fs))


def concat_windows(x: np.ndarray, k: int) -> np.ndarray:
    """[n, T] -> [n//k, k*T] by concatenating k consecutive windows, as train.py:41-47 buffers them."""
    n = (len(x) // k) * k
    return np.asarray(x)[:n].reshape(n // k, -1) if n else np.zeros((0, k * x.shape[-1]))


def segments_per_metric_window(window_s: int, segment_len_s: int) -> int:
    """train.py:43 `assert window_size % segment_len == 0` — a non-integer ratio is a hard error upstream."""
    if window_s % segment_len_s:
        raise ValueError(f"PENGUIN requires window_size % segment_len == 0; {window_s} % {segment_len_s} != 0")
    return window_s // segment_len_s
