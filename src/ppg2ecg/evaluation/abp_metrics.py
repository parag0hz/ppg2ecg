"""ABP-specific evaluation (A7; docs/A7_ABP_PREREGISTRATION.md). All pressures in mmHg (targets are raw, resampled only).

Per window (8 s @ 128 Hz):
- SBP_win / DBP_win: max / min of the waveform (PENGUIN's definition, paper §4: "maximum and minimum values of the ABP waveform");
- systolic peaks: scipy.signal.find_peaks on the waveform with min distance 0.3 s and prominence >= 0.25*(p95-p5) of the GT window
  (the same absolute prominence threshold, derived from the GT window, is applied to the prediction so that a flat prediction yields no
  peaks rather than spurious ones); diastolic troughs = minimum between consecutive systolic peaks;
- SBP_beat / DBP_beat: median of peak / trough values (MIMIC-BP label definition); PP = SBP - DBP;
- morphology: mean Pearson correlation of matched pulse templates (-0.25 s .. +0.55 s around matched systolic peaks; match tolerance
  100 ms), as the ECG template correlation but anchored on systolic peaks;
- timing: systolic-peak precision/recall/F1 (100 ms), matched-peak timing MAE (ms), pulse-interval MAE (ms), pulse count ratio;
- sharpness: max upstroke slope dP/dt (mmHg/s, median over beats) ratio pred/GT; HF-energy ratio (> 5 Hz) of the waveform;
- pointwise RMSE / MAE (mmHg); peak-region (+/-150 ms around GT systolic peaks) vs non-peak RMSE, peak-region energy ratio, peak
  amplitude error (mmHg, matched peaks).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

FS = 128
PEAK_TOL_MS = 100.0
PEAK_REGION_MS = 150.0
MIN_PULSE_S = 0.3
HF_CUT_HZ = 5.0
TEMPLATE = (-0.25, 0.55)


def systolic_peaks(x: np.ndarray, prominence: float, fs: int = FS) -> np.ndarray:
    pk, _ = find_peaks(x, distance=int(MIN_PULSE_S * fs), prominence=prominence)
    return pk


def gt_prominence(y: np.ndarray) -> float:
    p95, p5 = np.percentile(y, [95, 5])
    return max(0.25 * (p95 - p5), 1e-3)


def troughs_between(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    return np.array([p0 + int(np.argmin(x[p0:p1])) for p0, p1 in zip(peaks[:-1], peaks[1:])], dtype=int)


def match_peaks(ref: np.ndarray, pred: np.ndarray, fs: int = FS, tol_ms: float = PEAK_TOL_MS):
    tol = tol_ms / 1000 * fs
    used, matches = set(), []
    for r in ref:
        if len(pred) == 0:
            break
        j = int(np.argmin(np.abs(pred - r)))
        if abs(pred[j] - r) <= tol and j not in used:
            used.add(j)
            matches.append((int(r), int(pred[j])))
    return matches


def upstroke_slope(x: np.ndarray, peaks: np.ndarray, troughs: np.ndarray, fs: int = FS) -> float:
    if len(peaks) < 2:
        return np.nan
    d = np.diff(x) * fs
    slopes = [d[t:p].max() for t, p in zip(troughs, peaks[1:]) if p > t + 1]
    return float(np.median(slopes)) if slopes else np.nan


def template(x: np.ndarray, p: int, fs: int = FS):
    a, b = p + int(TEMPLATE[0] * fs), p + int(TEMPLATE[1] * fs)
    return x[a:b] if a >= 0 and b <= len(x) else None


def hf_ratio(x: np.ndarray, fs: int = FS, cut: float = HF_CUT_HZ) -> float:
    xc = x - x.mean()
    spec = np.abs(np.fft.rfft(xc)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return float(spec[f > cut].sum() / (spec.sum() + 1e-12))


def window_metrics(pred: np.ndarray, y: np.ndarray, fs: int = FS) -> dict:
    prom = gt_prominence(y)
    pr, pp = systolic_peaks(y, prom, fs), systolic_peaks(pred, prom, fs)
    tr, tp = troughs_between(y, pr), troughs_between(pred, pp)
    m = match_peaks(pr, pp, fs)
    n_match, n_fp, n_fn = len(m), len(pp) - len(m), len(pr) - len(m)
    prec = n_match / (n_match + n_fp) if n_match + n_fp else 0.0
    rec = n_match / (n_match + n_fn) if n_match + n_fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    sbp_b_gt = float(np.median(y[pr])) if len(pr) else np.nan
    dbp_b_gt = float(np.median(y[tr])) if len(tr) else np.nan
    sbp_b_pr = float(np.median(pred[pp])) if len(pp) else np.nan
    dbp_b_pr = float(np.median(pred[tp])) if len(tp) else np.nan
    corr = [np.corrcoef(template(y, r, fs), template(pred, q, fs))[0, 1] for r, q in m if template(y, r, fs) is not None and template(pred, q, fs) is not None and template(pred, q, fs).std() > 0]
    timing = [abs(q - r) / fs * 1000 for r, q in m]
    ri_gt, ri_pr = np.diff(pr) / fs * 1000, np.diff(pp) / fs * 1000
    pi_mae = float(abs(np.median(ri_pr) - np.median(ri_gt))) if len(ri_gt) and len(ri_pr) else np.nan
    slope_gt, slope_pr = upstroke_slope(y, pr, tr, fs), upstroke_slope(pred, pp, tp, fs)
    half = int(PEAK_REGION_MS / 1000 * fs)
    mask = np.zeros(len(y), bool)
    for r in pr:
        mask[max(0, r - half) : r + half + 1] = True
    err2 = (pred - y) ** 2
    yc, pc = y - y.mean(), pred - pred.mean()
    return dict(
        sbp_win_gt=float(y.max()), dbp_win_gt=float(y.min()), sbp_win_pr=float(pred.max()), dbp_win_pr=float(pred.min()),
        sbp_win_ae=abs(float(pred.max() - y.max())), dbp_win_ae=abs(float(pred.min() - y.min())),
        sbp_beat_gt=sbp_b_gt, dbp_beat_gt=dbp_b_gt, sbp_beat_pr=sbp_b_pr, dbp_beat_pr=dbp_b_pr,
        sbp_beat_ae=abs(sbp_b_pr - sbp_b_gt), dbp_beat_ae=abs(dbp_b_pr - dbp_b_gt),
        pp_gt=sbp_b_gt - dbp_b_gt, pp_pr=sbp_b_pr - dbp_b_pr, pp_ae=abs((sbp_b_pr - dbp_b_pr) - (sbp_b_gt - dbp_b_gt)), pp_ratio=(sbp_b_pr - dbp_b_pr) / max(sbp_b_gt - dbp_b_gt, 1e-6),
        amp_ratio=float(pred.std() / (y.std() + 1e-8)), morph_corr=float(np.mean(corr)) if corr else np.nan,
        peak_precision=prec, peak_recall=rec, peak_f1=f1, peak_timing_mae_ms=float(np.mean(timing)) if timing else np.nan, pulse_interval_mae_ms=pi_mae,
        n_peaks_gt=len(pr), n_peaks_pr=len(pp), pulse_count_ratio=len(pp) / max(len(pr), 1),
        slope_gt=slope_gt, slope_pr=slope_pr, slope_ratio=slope_pr / slope_gt if np.isfinite(slope_gt) and slope_gt > 0 and np.isfinite(slope_pr) else np.nan,
        hf_ratio_gt=hf_ratio(y, fs), hf_ratio_pr=hf_ratio(pred, fs),
        rmse=float(np.sqrt(err2.mean())), mae=float(np.abs(pred - y).mean()), pcc=float((yc * pc).sum() / (np.sqrt((yc**2).sum() * (pc**2).sum()) + 1e-12)),
        rmse_peak=float(np.sqrt(err2[mask].mean())) if mask.any() else np.nan, rmse_nonpeak=float(np.sqrt(err2[~mask].mean())) if (~mask).any() else np.nan,
        peak_region_energy_ratio=float(pc[mask].var() / (yc[mask].var() + 1e-8)) if mask.any() else np.nan, peak_region_frac=float(mask.mean()),
        peak_amp_ae=float(np.mean([abs(pred[q] - y[r]) for r, q in m])) if m else np.nan,
    )


def evaluate_abp(pred: np.ndarray, y: np.ndarray, fs: int = FS) -> dict:
    rows = [window_metrics(p, t, fs) for p, t in zip(pred, y)]
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def summarize_abp(pw: dict) -> dict:
    out = {}
    for k, v in pw.items():
        out[k] = {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"), "median": float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan"), "n_valid": int(np.isfinite(v).sum())}
    return out
