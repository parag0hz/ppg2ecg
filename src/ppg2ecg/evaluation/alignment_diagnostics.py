"""X0 ORACLE DIAGNOSTICS — temporal-alignment decomposition of one-step ECG error (docs/X0_ERROR_DECOMPOSITION_PREREGISTRATION.md).

Everything here USES THE GROUND TRUTH to find the best integer time translation of a prediction. These are diagnostics of what a
prediction contains, NOT deployable metrics and NOT "timing-corrected model performance". Rules: integer translation only (no
warping, no amplitude scaling, no width change); explicit crop of the non-overlapping edge (no circular wrap); deterministic.
Frozen primitives are reused: `rpeaks.detect_rpeaks` (neurokit), `rpeaks.match_rpeaks` (50 ms), `rpeaks.beat_window` (-0.25/+0.40 s).
"""
from __future__ import annotations

import numpy as np

from ppg2ecg.evaluation import rpeaks as R

FS = 128
GLOBAL_MAX_LAG_MS = 250.0  # +/-32 samples @128 Hz
LOCAL_MAX_SHIFT_MS = 150.0  # +/-19 samples @128 Hz
QRS_HALF_MS = 100.0  # frozen A5 definition
HF_CUT_HZ = 15.0


def _zn(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def global_lag(pred: np.ndarray, gt: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Oracle integer lag L in [-max_lag, max_lag] maximising the normalised cross-correlation of the OVERLAPPING samples of
    pred shifted by L against gt (positive L = prediction is late: pred[i] corresponds to gt[i - L]). No wrap-around."""
    n = len(gt)
    best_l, best_c = 0, -np.inf
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            a, b = pred[L:], gt[: n - L]
        else:
            a, b = pred[: n + L], gt[-L:]
        c = float(np.mean(_zn(a) * _zn(b)))
        if c > best_c + 1e-12 or (abs(c - best_c) <= 1e-12 and abs(L) < abs(best_l)):
            best_l, best_c = L, c
    return best_l, best_c


def shift_crop(pred: np.ndarray, gt: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (pred_aligned, gt_cropped, offset) — the overlapping region only (length n - |lag|); gt_cropped starts at gt[offset]."""
    n = len(gt)
    if lag >= 0:
        return pred[lag:].copy(), gt[: n - lag].copy(), 0
    return pred[: n + lag].copy(), gt[-lag:].copy(), -lag


def segment_stats(pseg: np.ndarray, gseg: np.ndarray, fs: int = FS, r_index: int | None = None) -> dict:
    """Shape/amplitude/sharpness statistics of a prediction beat segment vs the GT segment (same length, same coordinates)."""
    pseg, gseg = np.asarray(pseg, float), np.asarray(gseg, float)
    corr = float(np.corrcoef(pseg, gseg)[0, 1]) if pseg.std() > 1e-8 and gseg.std() > 1e-8 else (0.0 if gseg.std() > 1e-8 else np.nan)
    p2p = float((pseg.max() - pseg.min()) / (gseg.max() - gseg.min() + 1e-12))
    dp, dg = np.diff(pseg) * fs, np.diff(gseg) * fs
    slope = float(np.abs(dp).max() / (np.abs(dg).max() + 1e-12))
    out = {"corr": corr, "p2p_ratio": p2p, "slope_ratio": slope, "rmse": float(np.sqrt(np.mean((pseg - gseg) ** 2)))}
    if r_index is not None:
        h = int(round(QRS_HALF_MS / 1000 * fs))
        a, b = max(0, r_index - h), min(len(gseg), r_index + h + 1)
        out["qrs_energy_ratio"] = float(pseg[a:b].var() / (gseg[a:b].var() + 1e-12))
        out["qrs_rmse"] = float(np.sqrt(np.mean((pseg[a:b] - gseg[a:b]) ** 2)))
        out["r_amp_ratio"] = float(abs(pseg[r_index]) / (abs(gseg[r_index]) + 1e-12))
    f = np.fft.rfftfreq(len(gseg), 1 / fs)
    sp, sg = np.abs(np.fft.rfft(pseg - pseg.mean())) ** 2, np.abs(np.fft.rfft(gseg - gseg.mean())) ** 2
    out["hf_ratio_pred"] = float(sp[f > HF_CUT_HZ].sum() / (sp.sum() + 1e-12))
    out["hf_ratio_gt"] = float(sg[f > HF_CUT_HZ].sum() / (sg.sum() + 1e-12))
    return out


def beat_segments_gt(gt: np.ndarray, rpeaks: np.ndarray, fs: int = FS, before_s: float = 0.25, after_s: float = 0.40, margin: int = 0):
    """Indices (a, b, r_local) of GT beat windows (frozen -0.25/+0.40 s) that fit entirely inside the window with `margin` extra
    samples on both sides (so that local shifts up to `margin` stay inside the signal). Beats near the edges are skipped and counted."""
    bef, aft = int(round(before_s * fs)), int(round(after_s * fs))
    segs, skipped = [], 0
    for r in rpeaks:
        a, b = r - bef, r + aft
        if a - margin < 0 or b + margin > len(gt):
            skipped += 1
            continue
        segs.append((a, b, bef))
    return segs, skipped


def oracle_local_shift(pred: np.ndarray, gt: np.ndarray, a: int, b: int, max_shift: int) -> tuple[int, float]:
    """Best integer translation d in [-max_shift, max_shift] of the prediction segment pred[a+d:b+d] against gt[a:b] by Pearson
    correlation (translation only; the caller guarantees a-max_shift >= 0 and b+max_shift <= len). Ties -> smallest |d|."""
    g = _zn(gt[a:b])
    best_d, best_c = 0, -np.inf
    for d in range(-max_shift, max_shift + 1):
        p = pred[a + d : b + d]
        c = float(np.mean(_zn(p) * g)) if p.std() > 1e-12 else -1.0
        if c > best_c + 1e-12 or (abs(c - best_c) <= 1e-12 and abs(d) < abs(best_d)):
            best_d, best_c = d, c
    return best_d, best_c


def beat_level_analysis(pred: np.ndarray, gt: np.ndarray, gt_rpeaks: np.ndarray, fs: int = FS, max_shift: int = int(round(LOCAL_MAX_SHIFT_MS / 1000 * FS))) -> dict:
    """For every valid GT beat: stats at the same absolute coordinates (A: no shift) and after the oracle local translation (B).
    Detector-independent: uses GT R-peaks only, so flattened predictions are included. Returns per-beat arrays."""
    segs, skipped = beat_segments_gt(gt, gt_rpeaks, fs, margin=max_shift)
    keys_a, keys_b, shifts = [], [], []
    for a, b, rl in segs:
        sa = segment_stats(pred[a:b], gt[a:b], fs, rl)
        d, _ = oracle_local_shift(pred, gt, a, b, max_shift)
        sb = segment_stats(pred[a + d : b + d], gt[a:b], fs, rl)
        keys_a.append(sa)
        keys_b.append(sb)
        shifts.append(d)
    if not segs:
        return {"n_beats": 0, "n_skipped_edge": skipped}
    out = {"n_beats": len(segs), "n_skipped_edge": skipped, "shift_samples": np.asarray(shifts)}
    for k in keys_a[0]:
        out[f"raw_{k}"] = np.asarray([s[k] for s in keys_a], float)
        out[f"oracle_{k}"] = np.asarray([s[k] for s in keys_b], float)
    return out


def event_timing(gt: np.ndarray, pred: np.ndarray, fs: int = FS, tol_ms: float = 50.0) -> dict:
    """Frozen detector + frozen matching; returns counts, missing/spurious and signed matched timing errors (ms, pred - ref)."""
    rr, rp = R.detect_rpeaks(gt, fs), R.detect_rpeaks(pred, fs)
    m, fp, fn = R.match_rpeaks(rr, rp, fs, tol_ms)
    errs = np.asarray([(rp[j] - rr[i]) / fs * 1000.0 for i, j in m], float)
    return {"n_ref": int(len(rr)), "n_pred": int(len(rp)), "n_matched": int(len(m)), "n_missing": int(fn), "n_spurious": int(fp), "signed_err_ms": errs, "ref_rpeaks": rr, "pred_rpeaks": rp}
