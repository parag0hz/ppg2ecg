"""Frozen local features of a rendered R detection (E0 prereg §6). Simple scalars only; no learned representation.

Numbering follows the E0 specification §16-§18:
  A amplitude / baseline   1 amp_rel  2 amp_abs  3 prominence
  B shape                  4 width_ms  5 max_pos_slope  6 max_neg_slope  7 curvature  8 log_symmetry
  C high-frequency         9 diff_rms  10 hf_frac
  D event context          11 dist_nearest_ms  12 phase  13 prev_rr_ms  14 next_rr_ms
  E PPG context            15 ppg_slope  16 ppg_peak_lag_ms
  reference-QRS template   17 qrs_corr  18 qrs_l2
  C0 contrast (§18)        c0_prominence  c0_amp_rel  c0_max_pos_slope  c0_qrs_corr  and the WW - C0 differences
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import peak_prominences, peak_widths

FS = 128
H_LOCAL = 32          # +-250 ms context segment (median, prominence window, first-difference RMS, spectrum)
S_SLOPE = 10          # +-78 ms (the specified +-80 ms rounded to whole samples) for slopes and symmetry
REFINE = 3            # the detection is moved to the waveform maximum within +-3 samples (+-23 ms)
Q_HALF = 10           # QRS template support +-78 ms (21 samples) about the R peak
PPG_HALF = 5          # PPG slope averaged over +-39 ms
HF_BAND = (5.0, 20.0)

WW_FEATURES = ("amp_rel", "amp_abs", "prominence", "width_ms", "max_pos_slope", "max_neg_slope", "curvature", "log_symmetry",
               "diff_rms", "hf_frac", "qrs_corr", "qrs_l2")
CONTEXT_FEATURES = ("dist_nearest_ms", "phase", "prev_rr_ms", "next_rr_ms")
PPG_FEATURES = ("ppg_slope", "ppg_peak_lag_ms")
C0_FEATURES = ("c0_prominence", "c0_amp_rel", "c0_max_pos_slope", "c0_qrs_corr")
DIFF_FEATURES = ("d_prominence", "d_amp_rel", "d_max_pos_slope", "d_qrs_corr")
ALL_FEATURES = CONTEXT_FEATURES + WW_FEATURES + PPG_FEATURES + C0_FEATURES + DIFF_FEATURES
GROUPS = {"P1": CONTEXT_FEATURES, "P2": WW_FEATURES, "P3": ALL_FEATURES}


def refine(x: np.ndarray, t: int) -> int:
    lo, hi = max(0, t - REFINE), min(x.size, t + REFINE + 1)
    return int(lo + np.argmax(x[lo:hi]))


def qrs_template(Y, peaks) -> np.ndarray:
    """Mean reference ECG over +-Q_HALF samples about every reference R peak whose support lies inside its window.
    Called with ARCH-TRAIN data only (the stage passes role 'train')."""
    acc, n = np.zeros(2 * Q_HALF + 1), 0
    for y, pk in zip(Y, peaks):
        for r in np.asarray(pk, int):
            if r - Q_HALF >= 0 and r + Q_HALF + 1 <= y.size:
                acc += y[r - Q_HALF:r + Q_HALF + 1]
                n += 1
    if n == 0:
        raise ValueError("no complete QRS segment")
    return acc / n


def _z(v):
    s = np.std(v)
    return (v - np.mean(v)) / s if s > 1e-12 else np.full_like(v, np.nan)


def template_similarity(x: np.ndarray, ts: int, template: np.ndarray) -> tuple[float, float]:
    """(Pearson correlation, normalized L2 distance of z-scored segments) to the TRAIN QRS template; nan at window edges."""
    if ts - Q_HALF < 0 or ts + Q_HALF + 1 > x.size:
        return float("nan"), float("nan")
    q = x[ts - Q_HALF:ts + Q_HALF + 1]
    zq, zt = _z(q), _z(template)
    if not np.isfinite(zq).all():
        return float("nan"), float("nan")
    return float(np.mean(zq * zt)), float(np.linalg.norm(zq - zt) / np.sqrt(q.size))


def waveform_features(x: np.ndarray, t: int, template: np.ndarray) -> dict:
    """Features 1-10, 17, 18 of the detection at sample t of waveform x."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    ts = refine(x, t)
    lo, hi = max(0, ts - H_LOCAL), min(n, ts + H_LOCAL + 1)
    seg = x[lo:hi]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prom, lb, rb = peak_prominences(x, [ts], wlen=2 * H_LOCAL + 1)
        width = peak_widths(x, [ts], rel_height=0.5, prominence_data=(prom, lb, rb))[0][0]
    a, b = max(0, ts - S_SLOPE), min(n, ts + S_SLOPE + 1)
    dx = np.diff(x[a:b]) * FS
    left = np.diff(x[a:ts + 1]) * FS
    right = np.diff(x[ts:b]) * FS
    lmax = float(left.max()) if left.size else float("nan")
    rmax = float(-right.min()) if right.size else float("nan")
    sym = float(np.log(max(lmax, 1e-6) / max(rmax, 1e-6))) if np.isfinite(lmax) and np.isfinite(rmax) else float("nan")
    curv = float(abs(x[ts - 1] - 2 * x[ts] + x[ts + 1]) * FS ** 2) if 0 < ts < n - 1 else float("nan")
    if seg.size >= H_LOCAL + 1:
        s = (seg - seg.mean()) * np.hanning(seg.size)
        P = np.abs(np.fft.rfft(s)) ** 2
        f = np.fft.rfftfreq(seg.size, 1.0 / FS)
        tot = P[f > 0].sum()
        hf = float(P[(f >= HF_BAND[0]) & (f <= HF_BAND[1])].sum() / tot) if tot > 0 else float("nan")
    else:
        hf = float("nan")
    corr, l2 = template_similarity(x, ts, template)
    return {"amp_rel": float(x[ts] - np.median(seg)), "amp_abs": float(x[ts]), "prominence": float(prom[0]),
            "width_ms": float(width * 1000.0 / FS), "max_pos_slope": float(dx.max()) if dx.size else float("nan"),
            "max_neg_slope": float(dx.min()) if dx.size else float("nan"), "curvature": curv, "log_symmetry": sym,
            "diff_rms": float(np.sqrt(np.mean(np.diff(seg) ** 2)) * FS) if seg.size > 1 else float("nan"), "hf_frac": hf,
            "qrs_corr": corr, "qrs_l2": l2}


def context_features(loc: dict) -> dict:
    """Features 11-14 from `taxonomy.localize`: 13 / 14 are the RR intervals the candidate would form with the previous /
    next placed event (t - r_prev, r_next - t)."""
    return {"dist_nearest_ms": loc["dist_nearest_ms"], "phase": loc["phase"], "prev_rr_ms": loc["dist_prev_ms"],
            "next_rr_ms": loc["dist_next_ms"]}


def ppg_features(ppg: np.ndarray, t: int, ppg_peaks) -> dict:
    """15: mean PPG first difference over +-PPG_HALF samples (per s); 16: signed lag (ms) from the detection to the
    nearest PPG peak of the frozen project PPG detector (`s1_audit.dsp_ppg_peaks`), nan if the window has none."""
    ppg = np.asarray(ppg, dtype=np.float64)
    a, b = max(0, t - PPG_HALF), min(ppg.size, t + PPG_HALF + 1)
    d = np.diff(ppg[a:b])
    pk = np.asarray(ppg_peaks, int)
    lag = float((pk[np.argmin(np.abs(pk - t))] - t) * 1000.0 / FS) if pk.size else float("nan")
    return {"ppg_slope": float(d.mean() * FS) if d.size else float("nan"), "ppg_peak_lag_ms": lag}


def c0_contrast(ww: np.ndarray, c0: np.ndarray, t: int, template: np.ndarray, ww_feats: dict | None = None) -> dict:
    """§18: the frozen C0 waveform at the same timestamp (its own refined peak) and the WW - C0 differences."""
    w = ww_feats if ww_feats is not None else waveform_features(ww, t, template)
    c = waveform_features(c0, t, template)
    out = {"c0_prominence": c["prominence"], "c0_amp_rel": c["amp_rel"], "c0_max_pos_slope": c["max_pos_slope"],
           "c0_qrs_corr": c["qrs_corr"]}
    for k in ("prominence", "amp_rel", "max_pos_slope", "qrs_corr"):
        out[f"d_{k}"] = w[k] - c[k]
    return out


def segment(x: np.ndarray, t: int, half: int = H_LOCAL) -> np.ndarray:
    """x[t - half : t + half + 1] without warping, nan-padded outside the window."""
    out = np.full(2 * half + 1, np.nan)
    lo, hi = max(0, t - half), min(x.size, t + half + 1)
    out[lo - (t - half):hi - (t - half)] = x[lo:hi]
    return out
