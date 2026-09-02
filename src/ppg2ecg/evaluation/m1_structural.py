"""M1 structural-mechanism primitives (docs/M1_C1_STRUCTURAL_MECHANISM_AUDIT_PREREGISTRATION.md, 959eb60).

Everything here works at ORIGINAL FIXED COORDINATES. Ground-truth R-peaks define local coordinates only;
no prediction is ever shifted, aligned, or matched to a predicted peak. No oracle statistic is computed.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, welch

FS = 128
CORE_MS, PERI_MS = 80.0, 250.0
CORE = int(round(CORE_MS / 1000.0 * FS))     # 10
PERI = int(round(PERI_MS / 1000.0 * FS))     # 32
PROFILE_TAU = np.arange(-PERI, PERI + 1)     # 65 points, -250..+250 ms
BANDS = (("F1", 0.5, 4.0), ("F2", 4.0, 8.0), ("F3", 8.0, 15.0), ("F4", 15.0, 64.0))
WELCH = dict(fs=FS, nperseg=256, noverlap=128, window="hann", detrend="constant")
QRS_BAND = (8.0, 40.0)                        # atlas view C only, frozen before viewing output
ENV_WIN = 25                                  # atlas view E, ~195 ms moving average


def tau_map(n_time: int, gt_peaks: np.ndarray) -> np.ndarray:
    """Signed distance (samples) from each sample to the NEAREST GT R-peak; +inf if there are none."""
    p = np.asarray(gt_peaks, dtype=np.int64)
    n = np.arange(int(n_time))
    if p.size == 0:
        return np.full(n_time, np.inf)
    d = n[:, None] - p[None, :]
    j = np.argmin(np.abs(d), axis=1)
    return d[np.arange(n_time), j].astype(np.float64)


def region_masks(tau: np.ndarray) -> dict[str, np.ndarray]:
    """Frozen regions. Samples with no GT peak in the window fall in `background` by construction."""
    a = np.abs(tau)
    return {"qrs_core": a <= CORE, "peri_qrs": (a > CORE) & (a <= PERI), "background": a > PERI}


def d1(x: np.ndarray) -> np.ndarray:
    return np.diff(np.asarray(x, dtype=np.float64))


def d2(x: np.ndarray) -> np.ndarray:
    """Fixed second finite difference x[n+1] - 2x[n] + x[n-1]."""
    x = np.asarray(x, dtype=np.float64)
    return x[2:] - 2.0 * x[1:-1] + x[:-2]


def region_errors(pred: np.ndarray, gt: np.ndarray, gt_peaks: np.ndarray) -> dict:
    """A1-A5 per region, at original coordinates. Returns {region: {metric: value}} flattened."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    tau = tau_map(pred.size, gt_peaks)
    m = region_masks(tau)
    dp, dg = d1(pred), d1(gt)
    mt = {k: v[:-1] for k, v in m.items()}                 # derivative lives on n-1 samples
    out = {}
    for r, mask in m.items():
        md = mt[r]
        if not mask.any():
            out |= {f"{r}__a1_abs": np.nan, f"{r}__a2_sq": np.nan, f"{r}__a3_dabs": np.nan,
                    f"{r}__a4_amp": np.nan, f"{r}__a5_energy": np.nan, f"{r}__n": 0}
            continue
        gp, gg = pred[mask], gt[mask]
        out[f"{r}__a1_abs"] = float(np.mean(np.abs(gp - gg)))
        out[f"{r}__a2_sq"] = float(np.mean((gp - gg) ** 2))
        out[f"{r}__a3_dabs"] = float(np.mean(np.abs(dp[md] - dg[md]))) if md.any() else np.nan
        out[f"{r}__a4_amp"] = float(abs(np.ptp(gp) - np.ptp(gg)))
        out[f"{r}__a5_energy"] = float(abs(np.sum(gp ** 2) - np.sum(gg ** 2)) / (np.sum(gg ** 2) + 1e-12))
        out[f"{r}__n"] = int(mask.sum())
    return out


def event_profile(pred: np.ndarray, gt: np.ndarray, gt_peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-tau mean |pred-GT| and mean |D pred - D GT| around every fully-contained GT beat.

    Returns (abs_err[65], deriv_err[65], n_beats). Fixed coordinates; no translation.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    dp, dg = d1(pred), d1(gt)
    a, b, n = [], [], 0
    for r in np.asarray(gt_peaks, dtype=int):
        lo, hi = r - PERI, r + PERI
        if lo < 0 or hi + 1 > pred.size or hi + 1 > dp.size:
            continue
        a.append(np.abs(pred[lo:hi + 1] - gt[lo:hi + 1]))
        b.append(np.abs(dp[lo:hi + 1] - dg[lo:hi + 1]))
        n += 1
    if not n:
        return np.full(PROFILE_TAU.size, np.nan), np.full(PROFILE_TAU.size, np.nan), 0
    return np.mean(a, axis=0), np.mean(b, axis=0), n


def qrs_core_morphology(pred: np.ndarray, gt: np.ndarray, gt_peaks: np.ndarray) -> dict:
    """Question C, inside QRS-core only, at GT coordinates. No alignment, no predicted-peak matching."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    rmse, ptp_p, ptp_g, e_p, e_g, s_p, s_g, drmse, dmax_p, dmax_g, curv = ([] for _ in range(11))
    for r in np.asarray(gt_peaks, dtype=int):
        lo, hi = r - CORE, r + CORE
        if lo - 1 < 0 or hi + 2 > pred.size:
            continue
        p, g = pred[lo:hi + 1], gt[lo:hi + 1]
        rmse.append(np.sqrt(np.mean((p - g) ** 2)))
        ptp_p.append(np.ptp(p)); ptp_g.append(np.ptp(g))
        e_p.append(np.sum(p ** 2)); e_g.append(np.sum(g ** 2))
        dp, dg = d1(pred[lo - 1:hi + 2]), d1(gt[lo - 1:hi + 2])
        s_p.append(np.abs(dp).max()); s_g.append(np.abs(dg).max())
        drmse.append(np.sqrt(np.mean((dp - dg) ** 2)))
        dmax_p.append(np.abs(dp).max()); dmax_g.append(np.abs(dg).max())
        curv.append(np.mean(np.abs(d2(pred[lo - 1:hi + 2]) - d2(gt[lo - 1:hi + 2]))))
    if not rmse:
        return {k: np.nan for k in ("qrs_rmse_core", "qrs_ptp_dev", "qrs_energy_dev", "qrs_slope_dev",
                                    "qrs_deriv_rmse", "qrs_maxderiv_dev", "qrs_curvature_err", "n_beats")}
    f = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    return {"qrs_rmse_core": float(np.mean(rmse)),
            "qrs_ptp_dev": float(np.median(np.abs(f(ptp_p) / (f(ptp_g) + 1e-12) - 1.0))),
            "qrs_energy_dev": float(np.median(np.abs(f(e_p) / (f(e_g) + 1e-12) - 1.0))),
            "qrs_slope_dev": float(np.median(np.abs(f(s_p) / (f(s_g) + 1e-12) - 1.0))),
            "qrs_deriv_rmse": float(np.mean(drmse)),
            "qrs_maxderiv_dev": float(np.median(np.abs(f(dmax_p) / (f(dmax_g) + 1e-12) - 1.0))),
            "qrs_curvature_err": float(np.mean(curv)), "n_beats": int(len(rmse))}


def band_energy(x: np.ndarray) -> dict[str, float]:
    """Welch band energies with the frozen settings, identical for GT, prediction and residual."""
    f, p = welch(np.asarray(x, dtype=np.float64), **WELCH)
    return {name: float(np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]))
            if np.any((f >= lo) & (f < hi)) else 0.0 for name, lo, hi in BANDS}


def spectral_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    ep, eg, er = band_energy(pred), band_energy(gt), band_energy(np.asarray(pred) - np.asarray(gt))
    out = {}
    for name, _lo, _hi in BANDS:
        out[f"{name}__err_energy"] = er[name]
        out[f"{name}__gt_energy"] = eg[name]
        out[f"{name}__pred_energy"] = ep[name]
        out[f"{name}__ratio_dev"] = float(abs(ep[name] / (eg[name] + 1e-20) - 1.0))
    return out


def qrs_band_component(x: np.ndarray) -> np.ndarray:
    """Atlas view C only: zero-phase Butterworth 8-40 Hz, order 4. Frozen before viewing output."""
    b, a = butter(4, [QRS_BAND[0] / (FS / 2), QRS_BAND[1] / (FS / 2)], btype="bandpass")
    return filtfilt(b, a, np.asarray(x, dtype=np.float64))


def energy_envelope(x: np.ndarray, win: int = ENV_WIN) -> np.ndarray:
    """Atlas view E only: squared signal, moving average of `win` samples."""
    s = np.asarray(x, dtype=np.float64) ** 2
    k = np.ones(int(win)) / float(win)
    return np.convolve(s, k, mode="same")
