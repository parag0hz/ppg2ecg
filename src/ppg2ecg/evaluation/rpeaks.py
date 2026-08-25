"""R-peak detection, matching and beat-level morphology primitives.

Design rules (docs/PREREGISTRATION_V0.md §4):
  * the SAME detector + cleaning is applied to prediction and reference (upstream cleans only the prediction);
  * matching is one-to-one, greedy by |dt|, tolerance 50 ms (default);
  * morphology is compared on beats aligned at their *own* R-peaks (shape), timing is scored separately.
"""
from __future__ import annotations

import warnings

import numpy as np

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover
    nk = None


def detect_rpeaks(sig: np.ndarray, fs: int, method: str = "neurokit") -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size < fs or not np.isfinite(sig).all() or np.std(sig) < 1e-8:
        return np.zeros(0, dtype=int)
    if nk is None:
        raise ImportError("neurokit2 is required for R-peak detection")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clean = nk.ecg_clean(sig, sampling_rate=fs, method=method if method != "neurokit" else "neurokit")
            _, info = nk.ecg_peaks(clean, sampling_rate=fs, method=method)
            return np.asarray(info["ECG_R_Peaks"], dtype=int)
        except Exception:  # noqa: BLE001  (detector failure => no beats)
            return np.zeros(0, dtype=int)


def match_rpeaks(ref: np.ndarray, pred: np.ndarray, fs: int, tol_ms: float = 50.0):
    """Greedy one-to-one matching within tolerance. Returns (matches[(i_ref, j_pred)], n_fp, n_fn)."""
    tol = tol_ms / 1000.0 * fs
    ref, pred = np.asarray(ref), np.asarray(pred)
    if len(ref) == 0 or len(pred) == 0:
        return [], int(len(pred)), int(len(ref))
    pairs = [(abs(r - p), i, j) for i, r in enumerate(ref) for j, p in enumerate(pred) if abs(r - p) <= tol]
    pairs.sort()
    used_r, used_p, matches = set(), set(), []
    for _, i, j in pairs:
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        matches.append((i, j))
    matches.sort()
    return matches, int(len(pred) - len(used_p)), int(len(ref) - len(used_r))


def prf(n_match: int, n_fp: int, n_fn: int) -> tuple[float, float, float]:
    p = n_match / (n_match + n_fp) if n_match + n_fp else 0.0
    r = n_match / (n_match + n_fn) if n_match + n_fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def hr_bpm(rpeaks: np.ndarray, fs: int) -> float:
    if len(rpeaks) < 2:
        return float("nan")
    return 60.0 / (np.mean(np.diff(rpeaks)) / fs)


def rr_mae_ms(ref: np.ndarray, pred: np.ndarray, matches, fs: int) -> float:
    """MAE of RR intervals over consecutive reference beats that are BOTH matched."""
    m = dict(matches)
    errs = [abs((ref[i + 1] - ref[i]) - (pred[m[i + 1]] - pred[m[i]])) / fs * 1000 for i in range(len(ref) - 1) if i in m and i + 1 in m]
    return float(np.mean(errs)) if errs else float("nan")


def beat_window(sig: np.ndarray, r: int, fs: int, before_s: float = 0.25, after_s: float = 0.40) -> np.ndarray | None:
    a, b = r - int(round(before_s * fs)), r + int(round(after_s * fs))
    if a < 0 or b > len(sig):
        return None
    return sig[a:b]


def morphology_corr(ref_sig, pred_sig, ref_r, pred_r, matches, fs: int) -> float:
    cs = []
    for i, j in matches:
        wr, wp = beat_window(ref_sig, ref_r[i], fs), beat_window(pred_sig, pred_r[j], fs)
        if wr is None or wp is None or np.std(wr) < 1e-8 or np.std(wp) < 1e-8:
            continue
        cs.append(float(np.corrcoef(wr, wp)[0, 1]))
    return float(np.mean(cs)) if cs else float("nan")


def qrs_width_ms(sig: np.ndarray, r: int, fs: int, q_win_s: float = 0.08, s_win_s: float = 0.12) -> float:
    """QS-trough proxy: Q = argmin on [r-80 ms, r], S = argmin on [r, r+120 ms]; width = (S-Q)/fs."""
    a, b = max(0, r - int(round(q_win_s * fs))), min(len(sig), r + int(round(s_win_s * fs)) + 1)
    if r - a < 2 or b - r < 3:
        return float("nan")
    q = a + int(np.argmin(sig[a:r]))
    s = r + int(np.argmin(sig[r:b]))
    return (s - q) / fs * 1000.0


def qrs_width_error_ms(ref_sig, pred_sig, ref_r, pred_r, matches, fs: int) -> float:
    errs = []
    for i, j in matches:
        wr, wp = qrs_width_ms(ref_sig, ref_r[i], fs), qrs_width_ms(pred_sig, pred_r[j], fs)
        if np.isfinite(wr) and np.isfinite(wp):
            errs.append(abs(wr - wp))
    return float(np.mean(errs)) if errs else float("nan")
