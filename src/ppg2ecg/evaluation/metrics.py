"""Evaluation suite v0 (docs/PREREGISTRATION_V0.md §4). All functions are pure numpy; batching is the caller's job.

Per-window signal metrics : MAE, RMSE, PCC                    (on the 4 s, [-1,1]-normalised windows)
Rhythm / morphology       : R-peak P/R/F1 (50 ms), HR error (bpm), RR MAE (ms), QRS-width error (ms),
                            beat-aligned morphology correlation   (on 8 s windows = 2 consecutive segments, like upstream HR)
Upstream parity           : `penguin_hr_error` calls the unmodified upstream HeartRateError code.
Efficiency                : see efficiency.py (NFE, latency, samples/s, peak memory).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import rpeaks as R


def signal_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """pred/target: [n, T]. Returns per-window arrays."""
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    err = pred - target
    mae = np.abs(err).mean(axis=1)
    rmse = np.sqrt((err**2).mean(axis=1))
    pc = pred - pred.mean(axis=1, keepdims=True)
    tc = target - target.mean(axis=1, keepdims=True)
    denom = np.sqrt((pc**2).sum(axis=1) * (tc**2).sum(axis=1)) + 1e-12
    pcc = (pc * tc).sum(axis=1) / denom
    return {"mae": mae, "rmse": rmse, "pcc": pcc}


def concat_consecutive(x: np.ndarray, k: int) -> np.ndarray:
    """[n, T] -> [n//k, k*T] by concatenating consecutive windows (upstream: window_size 8 s / segment 4 s => k=2)."""
    n = (len(x) // k) * k
    return np.asarray(x[:n]).reshape(n // k, -1)


def rhythm_morphology_metrics(pred: np.ndarray, target: np.ndarray, fs: int, tol_ms: float = 50.0, detector: str = "neurokit") -> dict:
    """pred/target: [m, L] (already 8 s). Returns per-window arrays (nan where undefined)."""
    keys = ["rpeak_precision", "rpeak_recall", "rpeak_f1", "hr_ref", "hr_pred", "hr_abs_err", "rr_mae_ms", "qrs_width_err_ms", "morph_corr", "n_ref_beats", "n_pred_beats"]
    out = {k: [] for k in keys}
    for p, t in zip(pred, target):
        rr, rp = R.detect_rpeaks(t, fs, detector), R.detect_rpeaks(p, fs, detector)
        m, fp, fn = R.match_rpeaks(rr, rp, fs, tol_ms)
        pr, rc, f1 = R.prf(len(m), fp, fn)
        hr_t, hr_p = R.hr_bpm(rr, fs), R.hr_bpm(rp, fs)
        out["rpeak_precision"].append(pr)
        out["rpeak_recall"].append(rc)
        out["rpeak_f1"].append(f1)
        out["hr_ref"].append(hr_t)
        out["hr_pred"].append(hr_p)
        out["hr_abs_err"].append(abs(hr_p - hr_t) if np.isfinite(hr_p) and np.isfinite(hr_t) else np.nan)
        out["rr_mae_ms"].append(R.rr_mae_ms(rr, rp, m, fs))
        out["qrs_width_err_ms"].append(R.qrs_width_error_ms(t, p, rr, rp, m, fs))
        out["morph_corr"].append(R.morphology_corr(t, p, rr, rp, m, fs))
        out["n_ref_beats"].append(len(rr))
        out["n_pred_beats"].append(len(rp))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def evaluate_windows(pred: np.ndarray, target: np.ndarray, fs: int = 128, hr_window_segments: int = 2, tol_ms: float = 50.0, detector: str = "neurokit") -> dict:
    """Full per-window evaluation. pred/target: [n, T] 4 s windows in temporal order within a subject."""
    sig = signal_metrics(pred, target)
    rm = rhythm_morphology_metrics(concat_consecutive(pred, hr_window_segments), concat_consecutive(target, hr_window_segments), fs, tol_ms, detector)
    return {"signal": sig, "rhythm": rm}


def summarize(per_window: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """mean / std / bootstrap 95% CI over windows for every metric (nan-aware)."""
    rng = np.random.default_rng(seed)
    summ = {}
    for group in per_window.values():
        for k, v in group.items():
            v = np.asarray(v, np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                summ[k] = {"mean": np.nan, "std": np.nan, "ci95": (np.nan, np.nan), "n": 0}
                continue
            boots = np.array([rng.choice(v, size=v.size, replace=True).mean() for _ in range(n_boot)])
            summ[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))), "n": int(v.size)}
    return summ


def penguin_hr_error(pred: np.ndarray, target: np.ndarray, window_s: int = 8, mode: str = "corrected", shipped_segment_len_s: int = 4, fs: int = 128) -> float:
    """HeartRateError through the UNMODIFIED upstream `compute_metrics` (help_func.py L174-188); pred/target are [m, fs*window_s]
    windows of `window_s` seconds (upstream's HR window is 8 s). Returns the mean over windows.

    Upstream resamples each window to `128 * cfg.preprocess.segment_len` samples (L177-180) and treats the result as 128 Hz:
      mode="corrected" : cfg.segment_len = window_s            -> no resampling pathology (what upstream does when segment_len == 8)
      mode="as_shipped": cfg.segment_len = shipped_segment_len_s (4) -> the shipped 4 s config path: 2x time compression, doubled
                         HR estimates, high-HR windows masked, 0.0 fallback. DIAGNOSTIC ONLY - never a primary result.
    """
    import torch

    from ppg2ecg.utils.upstream import import_upstream_compute_metrics

    assert pred.shape[1] == fs * window_s, (pred.shape, fs * window_s)
    cm = import_upstream_compute_metrics()
    seg = window_s if mode == "corrected" else shipped_segment_len_s
    cfg = SimpleNamespace(preprocess=SimpleNamespace(segment_len=seg))
    vals = [cm(torch.as_tensor(p, dtype=torch.float32), torch.as_tensor(t, dtype=torch.float32), "HeartRateError", cfg) for p, t in zip(pred, target)]
    return float(np.mean(vals))


def hf_energy_ratio(x: np.ndarray, fs: int = 128, cutoff_hz: float = 15.0) -> np.ndarray:
    """Fraction of spectral power above cutoff_hz per window [n, T] - a smoothing/averaging indicator (pred vs target)."""
    x = np.asarray(x, np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / fs)
    tot = spec.sum(axis=1) + 1e-12
    return spec[:, freqs >= cutoff_hz].sum(axis=1) / tot
