"""Paper-metric suite: every column the three reference papers report, on top of the frozen v0 primitives.

Sources (metric -> equation / section):
  KANFlow      Eq. 21 MAE, Eq. 22 RMSE, Eq. 23 FD, Eqs. 24-25 Micro-F1, Eqs. 26-27 Macro-F1,
               Eqs. 28-29 RR-MAE (ms), Eq. 30 MAE_HR (bpm).
  PENGUIN      "Evaluation metrics": HR Error (bpm, Hamilton, 8 s window), RR Error (respiratory rate, bpm,
               FFT dominant frequency, 60 s window), SBP / DBP Error (mmHg, max / min of the ABP window).
  PPGFlowECG   MAE, RMSE, FD, FID (ECGFounder features), MAE_HR (bpm, Hamilton, 10 s window) -- no formulas given.
  CardioGAN / RDDM lineage (already in this repo's tables): PRD, discrete Frechet, DTW, SNR, cosine similarity,
               QRS-region vs non-QRS RMSE.

Contract for every function here:
  * inputs are [n_windows, n_samples] batches -- a single [T] window RAISES, it is never broadcast;
  * per-window functions return float64 arrays of length n; dataset-level functions return a float;
  * a window with any non-finite sample yields nan for that window -- never 0, never an exception;
  * nothing is clipped or epsilon-nudged: undefined is nan, and +-inf is returned where it is the correct
    answer (snr_db of a perfect reconstruction is +inf).

`PAPER_METRIC_SPEC` is the single source of truth for the report generator (name, orientation, aggregation
level, modality, definition, attribution, implementing callable), so no arrow or citation is hardcoded
downstream. `paper_metric_table` / `paper_metric_pooled` emit the ECG-modality columns only: PENGUIN's other
vital-sign targets (`resp_rate_abs_err_bpm`, modality "resp"; `sbp_abs_err_mmhg` / `dbp_abs_err_mmhg`, modality
"abp") are implemented here and callable, but they are NOT part of this ECG benchmark and are never assembled
into the table -- a report generator filters them on `MetricSpec.modality` instead of flagging them as missing.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np
from scipy.linalg import sqrtm

from . import metrics as M
from . import rpeaks as R

FS = 128
TOLERANCES_MS = (25.0, 50.0, 100.0)      # KANFlow / PENGUIN both score at 50 ms; 25 / 100 bracket it
QRS_HALF_MS = 50.0                       # RDDM region split: +-50 ms around each reference R peak
DTW_BAND = 64                            # Sakoe-Chiba radius in samples (0.5 s at 128 Hz)
FD_EPS = 1e-4                            # KANFlow Eq. 23 covariance regulariser
FD_PCA_DIM = 32                          # KANFlow: "at most 32" PCA dims for small test sets
FD_SMALL_SET = 3000                      # KANFlow: PCA + 5-trial averaging below this many segments
FD_TRIALS = 5


# --------------------------------------------------------------------------- batch / nan plumbing
def _batch(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D [n_windows, n_samples] batch, got shape {a.shape}; pass x[None] for a single window")
    return a


def _pair(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the pair, return float64 copies with non-finite windows ZEROED plus the `bad` mask.

    Zeroing keeps every downstream kernel (DP sweeps, FFTs, detectors) finite; the caller restores nan via
    `_mask` so a nan window can never silently score 0.
    """
    p, t = _batch(pred, "pred"), _batch(target, "target")
    if p.shape != t.shape:
        raise ValueError(f"pred {p.shape} and target {t.shape} must have the same shape")
    bad = ~(np.isfinite(p).all(axis=1) & np.isfinite(t).all(axis=1))
    p, t = p.copy(), t.copy()
    p[bad] = 0.0
    t[bad] = 0.0
    return p, t, bad


def _mask(values: np.ndarray, bad: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out[bad] = np.nan
    return out


# --------------------------------------------------------------------------- pointwise per-window metrics
def mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-window mean squared error (papers that report MSE instead of RMSE)."""
    p, t, bad = _pair(pred, target)
    return _mask(((p - t) ** 2).mean(axis=1), bad)


def prd(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """BOTH published PRD variants, never collapsed into one column.

    prd_raw     = 100 * sqrt( sum((p-t)^2) / sum(t^2) )              -- the original Percentage Root-mean-square
                                                                        Difference (energy-normalised).
    prd_meansub = 100 * sqrt( sum((p-t)^2) / sum((t-mean(t))^2) )    -- the mean-subtracted variant used by most
                                                                        compression papers; identical to prd_raw
                                                                        iff the target window is zero-mean.
    """
    p, t, bad = _pair(pred, target)
    num = ((p - t) ** 2).sum(axis=1)
    den_raw = (t ** 2).sum(axis=1)
    den_sub = ((t - t.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100.0 * np.sqrt(num / den_raw)
        sub = 100.0 * np.sqrt(num / den_sub)
    return {"prd_raw": _mask(raw, bad), "prd_meansub": _mask(sub, bad)}


def cosine_similarity(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-window cosine similarity (NOT mean-subtracted -- that would be `pcc` in metrics.signal_metrics)."""
    p, t, bad = _pair(pred, target)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = (p * t).sum(axis=1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1))
    return _mask(cs, bad)


def snr_db(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """10*log10( sum(t^2) / sum((p-t)^2) ). +inf for a bit-exact reconstruction, nan when both energies are 0."""
    p, t, bad = _pair(pred, target)
    sig, err = (t ** 2).sum(axis=1), ((p - t) ** 2).sum(axis=1)
    out = np.full(p.shape[0], np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[err > 0] = 10.0 * np.log10(sig[err > 0] / err[err > 0])
    out[(err == 0) & (sig == 0)] = np.nan
    return _mask(out, bad)


# --------------------------------------------------------------------------- pooled pointwise (KANFlow Eq. 21/22)
def pooled_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """KANFlow Eq. 21: (1/(N*L)) * sum_n sum_l |x - xhat|, pooled over ALL windows and samples."""
    p, t, bad = _pair(pred, target)
    return float(np.abs(p[~bad] - t[~bad]).mean())


def pooled_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """KANFlow Eq. 22: sqrt( (1/(N*L)) * sum_n sum_l (x - xhat)^2 ) -- the sqrt is taken ONCE, after pooling.

    This is NOT the mean of per-window RMSEs; by Jensen pooled_rmse >= mean_window_rmse always. Reproducing
    KANFlow's Table II requires this variant, so both are exported and reported side by side.
    """
    p, t, bad = _pair(pred, target)
    return float(np.sqrt(((p[~bad] - t[~bad]) ** 2).mean()))


def mean_window_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean over windows of the per-window RMSE -- the ambiguous reading KANFlow explicitly does NOT use."""
    p, t, bad = _pair(pred, target)
    return float(np.mean(np.sqrt(((p[~bad] - t[~bad]) ** 2).mean(axis=1))))


# --------------------------------------------------------------------------- anti-diagonal DP (Frechet / DTW)
def _diag_dp(pred: np.ndarray, target: np.ndarray, band: int | None, accumulate: Callable) -> np.ndarray:
    """Batched anti-diagonal sweep of ca[i,j] = accumulate( |p_i - t_j|, min(ca[i-1,j], ca[i-1,j-1], ca[i,j-1]) ).

    Every cell on anti-diagonal k = i + j depends only on diagonals k-1 and k-2, so one vectorised update per
    diagonal replaces the T^2 python iterations of the textbook double loop; the whole [n, T] batch advances
    together. `accumulate` is np.maximum for the discrete Frechet distance and np.add for DTW.

    Cost: O(n * T^2) arithmetic in 2T-1 numpy steps, O(n * T) memory -- the T x T distance matrix is never
    materialised (each diagonal reads a contiguous slice of `pred` and a reversed slice of `target`).
    A Sakoe-Chiba `band` (|i - j| <= band) shortens every diagonal to at most band+1 cells.
    """
    n, ni = pred.shape
    nj = target.shape[1]
    inf = np.inf
    buf_a = np.full((n, ni + 1), inf)     # buf[:, i+1] holds ca[i, k-1-i]; buf[:, 0] is the i-1 = -1 sentinel
    buf_b = np.full((n, ni + 1), inf)
    rng_a = rng_b = (1, 0)                # empty (lo, hi) ranges
    for k in range(ni + nj - 1):
        lo, hi = max(0, k - (nj - 1)), min(k, ni - 1)
        if band is not None:
            lo, hi = max(lo, (k - band + 1) // 2), min(hi, (k + band) // 2)
        d = np.abs(pred[:, lo:hi + 1] - target[:, k - hi:k - lo + 1][:, ::-1])   # reversed view, no gather
        if k == 0:
            cur = d
        else:
            prev = np.minimum(np.minimum(buf_a[:, lo + 1:hi + 2], buf_a[:, lo:hi + 1]), buf_b[:, lo:hi + 1])
            cur = accumulate(d, prev)
        buf_b[:, rng_b[0] + 1:rng_b[1] + 2] = inf   # the k-2 buffer is dead; its only finite cells are that range
        buf_b[:, lo + 1:hi + 2] = cur
        buf_a, buf_b = buf_b, buf_a
        rng_a, rng_b = (lo, hi), rng_a
    return buf_a[:, ni].copy()            # ca[ni-1, nj-1] sits at index (ni-1)+1 on the final diagonal


def discrete_frechet(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """CardioGAN's discrete (coupling) Frechet distance per window: the minimum over order-preserving pairings
    of the maximum |p_i - t_j| along the pairing (Eiter & Mannila 1994), on the 1-D sequences.

    Unbanded and exact. See `_diag_dp` for the O(T^2) cost -- at T = 1024 this is ~1e6 cell updates per window,
    which dominates `paper_metric_table`; pass with_quadratic=False there to skip it and DTW.
    """
    p, t, bad = _pair(pred, target)
    return _mask(_diag_dp(p, t, None, np.maximum), bad)


def dtw_distance(pred: np.ndarray, target: np.ndarray, band: int | None = DTW_BAND) -> np.ndarray:
    """Sakoe-Chiba-banded DTW per window: minimum over monotone warpings of the SUM of |p_i - t_j|.

    Unnormalised (no division by path length), so dtw >= discrete_frechet holds for every window: any path's
    sum of non-negative costs is at least its own maximum, which is at least the unbanded Frechet optimum.
    `band` is the radius in samples; None removes the constraint (T^2 cells, ~16x the default band-64 cost).
    """
    p, t, bad = _pair(pred, target)
    return _mask(_diag_dp(p, t, band, np.add), bad)


# --------------------------------------------------------------------------- Gaussian (FID-style) Frechet
def gaussian_frechet(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, imag_tol: float = 1e-3) -> float:
    """d^2 = |mu1 - mu2|^2 + Tr( S1 + S2 - 2 (S1 S2)^{1/2} ) -- the RDDM / FID / KANFlow-Eq.-23 formula.

    `scipy.linalg.sqrtm` of a product of two symmetric PSD matrices is real up to round-off; an imaginary part
    larger than `imag_tol` (relative to the real part) RAISES ValueError rather than being silently discarded.
    The check is an explicit raise, not an `assert`: asserts are stripped under `python -O` / PYTHONOPTIMIZE,
    which would silently `.real` a materially complex result.
    """
    diff = np.asarray(mu1, np.float64) - np.asarray(mu2, np.float64)
    covmean = sqrtm(np.asarray(sigma1, np.float64) @ np.asarray(sigma2, np.float64))
    if np.iscomplexobj(covmean):
        scale = max(1.0, float(np.abs(covmean.real).max()))
        imag = float(np.abs(covmean.imag).max())
        if imag > imag_tol * scale:
            raise ValueError(f"sqrtm returned a materially complex matrix: max |imaginary| = {imag:.6g} exceeds "
                             f"imag_tol {imag_tol:g} * real scale {scale:.6g} = {imag_tol * scale:.6g}")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def _moments(feats: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    sigma = np.atleast_2d(sigma) + eps * np.eye(feats.shape[1])
    return mu, sigma


def fid_frechet(pred_feats: np.ndarray, target_feats: np.ndarray, eps: float = 1e-6, imag_tol: float = 1e-3) -> float:
    """Frechet distance between two Gaussians fitted to FEATURE ACTIVATIONS (RDDM / PPGFlowECG FID form).

    `pred_feats` / `target_feats` are [n_windows, n_features]; the two sets need not have the same n. `eps` is a
    diagonal covariance regulariser (default 1e-6; KANFlow's FD uses 1e-4, see `kanflow_fd`).

    IMPORTANT: with `default_feature_map` (used when no learned extractor is available) the resulting number is
    NOT comparable to RDDM's or PPGFlowECG's published FID -- those are computed in the activation space of
    their own networks (PPGFlowECG uses ECGFounder). Only differences between models scored with the SAME
    feature map here are meaningful.
    """
    a, b = _batch(pred_feats, "pred_feats"), _batch(target_feats, "target_feats")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"feature dimensionality differs: pred {a.shape[1]} vs target {b.shape[1]}")
    a, b = a[np.isfinite(a).all(axis=1)], b[np.isfinite(b).all(axis=1)]
    mu1, s1 = _moments(a, eps)
    mu2, s2 = _moments(b, eps)
    return gaussian_frechet(mu1, s1, mu2, s2, imag_tol)


def default_feature_map(x: np.ndarray, fs: int = FS, n_bands: int = 32) -> np.ndarray:
    """Documented stand-in feature map for `fid_frechet` when no pretrained ECG encoder is available.

    36 dimensions per window: log1p power in `n_bands` equal-width rFFT bands (0 .. fs/2 Hz) of the
    mean-removed window, then [mean, std, skewness, kurtosis] of the raw window. Deterministic, numpy-only,
    and NOT the space any published FID was computed in -- see the warning in `fid_frechet`.
    """
    a = _batch(x, "x")
    spec = np.abs(np.fft.rfft(a - a.mean(axis=1, keepdims=True), axis=1)) ** 2
    edges = np.linspace(0, spec.shape[1], n_bands + 1).astype(int)
    bands = np.stack([np.log1p(spec[:, s:e].sum(axis=1)) for s, e in zip(edges[:-1], edges[1:])], axis=1)
    m, s = a.mean(axis=1, keepdims=True), a.std(axis=1, keepdims=True)
    z = (a - m) / np.where(s > 0, s, 1.0)
    return np.concatenate([bands, m, s, (z ** 3).mean(axis=1, keepdims=True), (z ** 4).mean(axis=1, keepdims=True)], axis=1)


def kanflow_fd(pred: np.ndarray, target: np.ndarray, pca_dim: int = FD_PCA_DIM, small_set: int = FD_SMALL_SET,
               n_trials: int = FD_TRIALS, eps: float = FD_EPS, seed: int = 0) -> float:
    """KANFlow Eq. 23 FD: the Gaussian Frechet distance with the RAW FLATTENED WAVEFORM as the feature vector.

    Two regimes, exactly as stated in the paper:
      * fewer than `small_set` (3000) test segments -> real and generated waveforms are concatenated, projected
        into a PCA space of at most `pca_dim` (32) dimensions, and the FD is averaged over `n_trials` (5) trials;
      * otherwise -> FD is computed directly in the flattened waveform space.
    `eps` (1e-4) regularises both covariances.

    The paper does not say what varies across the 5 trials -- PCA by SVD is deterministic, so an unresampled
    reading would make all 5 identical. Here each trial is a seeded bootstrap resample (with replacement, same
    size) of both sets; the mean over trials is returned. KANFlow itself warns FD is only interpretable WITHIN
    a dataset because of this PCA / raw split.
    """
    p, t, bad = _pair(pred, target)
    p, t = p[~bad], t[~bad]
    if len(p) >= small_set:
        return fid_frechet(p, t, eps)
    dim = min(pca_dim, p.shape[1], len(p) + len(t) - 1)
    joint = np.concatenate([p, t], axis=0)
    mu = joint.mean(axis=0)
    axes = np.linalg.svd(joint - mu, full_matrices=False)[2][:dim].T
    a, b = (p - mu) @ axes, (t - mu) @ axes
    rng = np.random.default_rng(seed)
    vals = [fid_frechet(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))], eps) for _ in range(n_trials)]
    return float(np.mean(vals))


# --------------------------------------------------------------------------- R-peak driven metrics
def detect_batch(x: np.ndarray, fs: int = FS, detector: str = "neurokit") -> list[np.ndarray]:
    """One `rpeaks.detect_rpeaks` pass per window, hoisted so every tolerance and the region split share it."""
    return [R.detect_rpeaks(row, fs, detector) for row in _batch(x, "x")]


def rpeak_prf_at(pred: np.ndarray, target: np.ndarray, fs: int = FS, tol_ms: float = 50.0, detector: str = "neurokit",
                 peaks: tuple[Sequence[np.ndarray], Sequence[np.ndarray]] | None = None) -> dict[str, np.ndarray]:
    """Per-window R-peak precision / recall / F1 and raw TP / FP / FN counts at `tol_ms`.

    Reference peaks come from the DETECTOR RUN ON THE REAL ECG (KANFlow does not use annotation files); matching
    is `rpeaks.match_rpeaks` (greedy one-to-one by |dt|). Pass `peaks=(ref_lists, pred_lists)` from
    `detect_batch` to score several tolerances without re-detecting. Counts are returned so `micro_f1` can pool
    them (KANFlow Eqs. 24-25) while `macro_f1` averages the per-window F1 over evaluable segments (Eqs. 26-27).
    """
    p, t, bad = _pair(pred, target)
    ref, hyp = peaks if peaks is not None else (detect_batch(t, fs, detector), detect_batch(p, fs, detector))
    rows = []
    for r, h in zip(ref, hyp):
        m, fp, fn = R.match_rpeaks(r, h, fs, tol_ms)
        prec, rec, f1 = R.prf(len(m), fp, fn)
        rows.append((prec, rec, f1, len(m), fp, fn))
    arr = np.asarray(rows, dtype=np.float64).reshape(len(p), 6)
    keys = ("rpeak_precision", "rpeak_recall", "rpeak_f1", "n_tp", "n_fp", "n_fn")
    return {k: _mask(arr[:, i], bad) for i, k in enumerate(keys)}


def micro_f1(n_tp: np.ndarray, n_fp: np.ndarray, n_fn: np.ndarray) -> float:
    """KANFlow Eqs. 24-25: TP/FP/FN are SUMMED over the whole dataset before the ratio (nan windows excluded)."""
    tp, fp, fn = (np.asarray(v, np.float64) for v in (n_tp, n_fp, n_fn))
    ok = np.isfinite(tp) & np.isfinite(fp) & np.isfinite(fn)
    prec, rec, f1 = R.prf(int(tp[ok].sum()), int(fp[ok].sum()), int(fn[ok].sum()))
    del prec, rec
    return float(f1)


def macro_f1(rpeak_f1: np.ndarray, n_ref_beats: np.ndarray | None = None) -> tuple[float, int]:
    """KANFlow Eqs. 26-27: mean of the per-segment F1 over the segments with a VALID peak-based evaluation.

    A segment is evaluable iff its F1 is finite AND the REFERENCE has at least one detected beat. `rpeaks.prf`
    returns 0.0 (never nan) when the reference detector finds no beat, so without this filter an unevaluable
    segment would be averaged in as F1 = 0 and drag the mean down. A segment whose reference HAS beats but whose
    prediction has none stays in and scores 0 -- that is a genuine recall failure, not an unevaluable segment.

    Returns (macro_f1, n_excluded_segments); the value is nan (and every segment counted excluded) when nothing is
    evaluable. `n_ref_beats` -- the `beat_level_metrics` column -- is optional only so a caller that has no beat
    counts can still average; omitted, ONLY non-finite F1 values are dropped and the filter above is not applied.
    """
    v = np.asarray(rpeak_f1, np.float64)
    ok = np.isfinite(v)
    if n_ref_beats is not None:
        nref = np.asarray(n_ref_beats, np.float64)
        if nref.shape != v.shape:
            raise ValueError(f"n_ref_beats {nref.shape} must have the same shape as rpeak_f1 {v.shape}")
        ok &= np.isfinite(nref) & (nref > 0)
    return (float(v[ok].mean()) if ok.any() else float("nan")), int(v.size - ok.sum())


def beat_level_metrics(pred: np.ndarray, target: np.ndarray, fs: int = FS, tol_ms: float = 50.0, detector: str = "neurokit",
                       peaks: tuple[Sequence[np.ndarray], Sequence[np.ndarray]] | None = None) -> dict[str, np.ndarray]:
    """`metrics.rhythm_morphology_metrics` with the detector pass hoisted out (identical outputs, one detection).

    Columns: hr_ref / hr_pred / hr_abs_err (bpm), rr_mae_ms (KANFlow Eq. 28, matched consecutive beats only),
    qrs_width_err_ms, morph_corr, n_ref_beats, n_pred_beats.
    """
    p, t, bad = _pair(pred, target)
    ref, hyp = peaks if peaks is not None else (detect_batch(t, fs, detector), detect_batch(p, fs, detector))
    keys = ("hr_ref", "hr_pred", "hr_abs_err", "rr_mae_ms", "qrs_width_err_ms", "morph_corr", "n_ref_beats", "n_pred_beats")
    rows = []
    for sig_t, sig_p, r, h in zip(t, p, ref, hyp):
        m = R.match_rpeaks(r, h, fs, tol_ms)[0]
        hr_t, hr_p = R.hr_bpm(r, fs), R.hr_bpm(h, fs)
        rows.append((hr_t, hr_p, abs(hr_p - hr_t) if np.isfinite(hr_p) and np.isfinite(hr_t) else np.nan,
                     R.rr_mae_ms(r, h, m, fs), R.qrs_width_error_ms(sig_t, sig_p, r, h, m, fs),
                     R.morphology_corr(sig_t, sig_p, r, h, m, fs), len(r), len(h)))
    arr = np.asarray(rows, dtype=np.float64).reshape(len(p), len(keys))
    return {k: _mask(arr[:, i], bad) for i, k in enumerate(keys)}


def hr_mae_bpm(hr_pred: np.ndarray, hr_ref: np.ndarray) -> float:
    """KANFlow Eq. 30 / PENGUIN "HR Error" / PPGFlowECG MAE_HR: mean |HR_pred - HR_ref| in bpm.

    Takes the two per-window HR ARRAYS (from `beat_level_metrics`), not waveforms. Windows where HR is undefined
    on either side are excluded, which is KANFlow's Omega_HR rule. The window length is NOT part of the metric
    and is not comparable across papers: KANFlow 4 s, PENGUIN 8 s, PPGFlowECG 10 s.
    """
    e = np.abs(np.asarray(hr_pred, np.float64) - np.asarray(hr_ref, np.float64))
    return float(np.nanmean(e)) if np.isfinite(e).any() else float("nan")


def hr_rmse_bpm(hr_pred: np.ndarray, hr_ref: np.ndarray) -> float:
    """Pooled RMSE of the same per-window HR errors (no paper reports it; included so the table is not MAE-only)."""
    e = np.asarray(hr_pred, np.float64) - np.asarray(hr_ref, np.float64)
    return float(np.sqrt(np.nanmean(e ** 2))) if np.isfinite(e).any() else float("nan")


def region_rmse(pred: np.ndarray, target: np.ndarray, fs: int = FS, half_ms: float = QRS_HALF_MS, detector: str = "neurokit",
                peaks: Sequence[np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """RDDM's region-disentangled RMSE: inside vs outside +-`half_ms` of each REFERENCE R peak.

    `half_ms` defaults to 50 ms, i.e. 13 samples (+-6) at 128 Hz per beat (~10 % of an 8 s window at 60 bpm). Reference
    peaks come from the ground-truth ECG; `qrs_region_frac` reports the covered fraction so the two RMSEs can be
    read against the amount of signal each covers.
    """
    p, t, bad = _pair(pred, target)
    half = int(round(half_ms / 1000.0 * fs))
    ref = peaks if peaks is not None else detect_batch(t, fs, detector)
    err2 = (p - t) ** 2
    rows = []
    for e, r in zip(err2, ref):
        m = np.zeros(len(e), dtype=bool)
        for idx in np.asarray(r, dtype=int):
            m[max(0, idx - half):idx + half + 1] = True
        rows.append((np.sqrt(e[m].mean()) if m.any() else np.nan, np.sqrt(e[~m].mean()) if (~m).any() else np.nan, m.mean()))
    arr = np.asarray(rows, dtype=np.float64).reshape(len(p), 3)
    return {"qrs_region_rmse": _mask(arr[:, 0], bad), "non_qrs_rmse": _mask(arr[:, 1], bad), "qrs_region_frac": _mask(arr[:, 2], bad)}


def qrs_region_rmse(pred: np.ndarray, target: np.ndarray, fs: int = FS, half_ms: float = QRS_HALF_MS, **kw) -> np.ndarray:
    """RMSE restricted to +-`half_ms` around the reference R peaks (see `region_rmse`, which returns both halves)."""
    return region_rmse(pred, target, fs, half_ms, **kw)["qrs_region_rmse"]


def non_qrs_rmse(pred: np.ndarray, target: np.ndarray, fs: int = FS, half_ms: float = QRS_HALF_MS, **kw) -> np.ndarray:
    """RMSE on the complement of the QRS regions (see `region_rmse`, which returns both halves)."""
    return region_rmse(pred, target, fs, half_ms, **kw)["non_qrs_rmse"]


# --------------------------------------------------------------------------- PENGUIN's non-ECG columns
def respiratory_rate_bpm(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """PENGUIN "RR Error" estimator: the DOMINANT NON-NEGATIVE rFFT frequency of the respiratory waveform, in
    breaths per minute. PENGUIN uses a 60 s window and feeds the 1 Hz-low-passed, z-scored, [-1,1]-scaled
    waveform, so bin 0 carries no power and is kept in the argmax exactly as the paper words it.

    NOTE the name collision: this is RESPIRATORY RATE (bpm), unrelated to KANFlow's RR-MAE (R-R interval, ms).
    """
    a = _batch(x, "x")
    bad = ~np.isfinite(a).all(axis=1)
    a = a.copy()
    a[bad] = 0.0
    freqs = np.fft.rfftfreq(a.shape[1], d=1.0 / fs)
    return _mask(freqs[np.argmax(np.abs(np.fft.rfft(a, axis=1)), axis=1)] * 60.0, bad)


def respiratory_rate_abs_err_bpm(pred: np.ndarray, target: np.ndarray, fs: int = FS) -> np.ndarray:
    """Per-window |RR_pred - RR_ref| in breaths/min; PENGUIN reports the mean over 60 s windows."""
    p, t, bad = _pair(pred, target)
    return _mask(np.abs(respiratory_rate_bpm(p, fs) - respiratory_rate_bpm(t, fs)), bad)


def sbp_abs_err_mmhg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """PENGUIN "SBP Error": |max(pred_window) - max(gt_window)| in mmHg on the UNNORMALISED ABP waveform.

    Same quantity as `abp_metrics.window_metrics`'s `sbp_win_ae`; exposed here batched so the paper table can
    be assembled from one module.
    """
    p, t, bad = _pair(pred, target)
    return _mask(np.abs(p.max(axis=1) - t.max(axis=1)), bad)


def dbp_abs_err_mmhg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """PENGUIN "DBP Error": |min(pred_window) - min(gt_window)| in mmHg (= `abp_metrics` `dbp_win_ae`)."""
    p, t, bad = _pair(pred, target)
    return _mask(np.abs(p.min(axis=1) - t.min(axis=1)), bad)


# --------------------------------------------------------------------------- entry points
def paper_metric_table(pred: np.ndarray, target: np.ndarray, fs: int = FS, tolerances: Sequence[float] = TOLERANCES_MS,
                       qrs_half_ms: float = QRS_HALF_MS, dtw_band: int | None = DTW_BAND, with_rpeaks: bool = True,
                       with_quadratic: bool = True, detector: str = "neurokit") -> dict[str, np.ndarray]:
    """Every PER-WINDOW paper column for one (pred, target) ECG batch. Returns {column: float64 [n]}, nan-safe.

    `with_quadratic=False` drops `discrete_frechet` and `dtw` (the O(T^2) sweeps); `with_rpeaks=False` drops
    everything that needs a detector. Dataset-level columns (pooled MAE/RMSE, Micro/Macro-F1, FD, FID, pooled
    HR error) are in `paper_metric_pooled`.
    """
    p, t, bad = _pair(pred, target)
    out: dict[str, np.ndarray] = dict(M.signal_metrics(p, t))
    out["mse"] = ((p - t) ** 2).mean(axis=1)
    out.update(prd(p, t))
    out["cosine_similarity"] = cosine_similarity(p, t)
    out["snr_db"] = snr_db(p, t)
    out["hf_energy_ratio_pred"] = M.hf_energy_ratio(p, fs)
    out["hf_energy_ratio_ref"] = M.hf_energy_ratio(t, fs)
    if with_quadratic:
        out["discrete_frechet"] = _diag_dp(p, t, None, np.maximum)
        out["dtw"] = _diag_dp(p, t, dtw_band, np.add)
    if with_rpeaks:
        peaks = (detect_batch(t, fs, detector), detect_batch(p, fs, detector))
        for tol in tolerances:
            for k, v in rpeak_prf_at(p, t, fs, tol, detector, peaks).items():
                out[f"{k}_{int(round(tol))}ms"] = v
        out.update(beat_level_metrics(p, t, fs, 50.0, detector, peaks))
        out.update(region_rmse(p, t, fs, qrs_half_ms, detector, peaks[0]))
    return {k: _mask(v, bad) for k, v in out.items()}


def paper_metric_pooled(pred: np.ndarray, target: np.ndarray, fs: int = FS, table: dict[str, np.ndarray] | None = None,
                        feature_map: Callable[[np.ndarray], np.ndarray] | None = None, **kw) -> dict[str, float]:
    """Every DATASET-LEVEL paper column. `table` reuses a `paper_metric_table` result instead of recomputing it.

    EXCLUSION SEMANTICS: every pooled number here is computed over the FINITE windows only -- a window with any
    non-finite sample is dropped, never scored. `_pair` zeroes such windows to keep downstream kernels finite, so
    the filtered arrays (not `p` / `t`) are what the pooled helpers receive and the raw `pred` / `target` (not the
    zeroed copies) are what `paper_metric_table` receives; otherwise a dropped window would be scored as an
    all-zero waveform. `n_excluded_windows` reports how many windows were dropped so the exclusion is never
    invisible, and the two call paths (`table` supplied vs. computed here) return identical values for every key.

    `feature_map` defaults to `default_feature_map` for the FID surrogate -- see the comparability warning in
    `fid_frechet`. `fid_ecgfounder` is deliberately absent: it needs the ECGFounder checkpoint.
    """
    p, t, bad = _pair(pred, target)
    pc, tc = p[~bad], t[~bad]                                                  # finite windows only
    tab = table if table is not None else paper_metric_table(pred, target, fs, **kw)   # RAW inputs: it masks itself
    fmap = feature_map if feature_map is not None else (lambda x: default_feature_map(x, fs))
    out = {
        "pooled_mae": pooled_mae(pc, tc),
        "pooled_rmse": pooled_rmse(pc, tc),
        "mean_window_rmse": mean_window_rmse(pc, tc),
        "kanflow_fd": kanflow_fd(pc, tc),
        "fid_default_features": fid_frechet(fmap(pc), fmap(tc)),
        "n_excluded_windows": float(int(bad.sum())),
    }
    if "n_tp_50ms" in tab:
        out["micro_f1"] = micro_f1(tab["n_tp_50ms"], tab["n_fp_50ms"], tab["n_fn_50ms"])
        out["macro_f1"], n_unevaluable = macro_f1(tab["rpeak_f1_50ms"], tab.get("n_ref_beats"))
        out["n_macro_f1_excluded_segments"] = float(n_unevaluable)
        out["rr_mae_ms_macro"] = float(np.nanmean(tab["rr_mae_ms"])) if np.isfinite(tab["rr_mae_ms"]).any() else float("nan")
        out["hr_mae_bpm"] = hr_mae_bpm(tab["hr_pred"], tab["hr_ref"])
        out["hr_rmse_bpm"] = hr_rmse_bpm(tab["hr_pred"], tab["hr_ref"])
    return out


# --------------------------------------------------------------------------- report-generator contract
class MetricSpec(NamedTuple):
    """One reportable column. `orientation` drives the arrow, `reported_by` the attribution footnote."""

    key: str
    name: str
    orientation: str          # one of ORIENTATIONS
    level: str                # "window" (per-window array) or "dataset" (single pooled number)
    modality: str             # "ecg", "resp" or "abp"
    definition: str
    reported_by: tuple[str, ...]
    impl: str | None          # dotted path of the implementing callable, None if not implementable here


ORIENTATIONS = ("lower_is_better", "higher_is_better", "neutral")
_P = "ppg2ecg.evaluation.paper_metrics."


def _rpeak_specs() -> tuple[MetricSpec, ...]:
    """The six R-peak columns replicated across `TOLERANCES_MS`, so the table is not tied to one tolerance.

    Reference peaks always come from the detector run on the REAL ECG (KANFlow uses no annotation file);
    matching is greedy one-to-one by |dt|. KANFlow scores at 50 ms; 25 / 100 ms bracket it.
    """
    rows = (
        ("rpeak_precision", "R-peak precision", "higher_is_better", "Fraction of generated R peaks matched to a reference peak within {t} ms."),
        ("rpeak_recall", "R-peak recall", "higher_is_better", "Fraction of reference R peaks matched within {t} ms."),
        ("rpeak_f1", "R-peak F1", "higher_is_better", "Harmonic mean of precision and recall at {t} ms."),
        ("n_tp", "TP", "neutral", "Matched-peak count per window at {t} ms; summed by micro_f1 (KANFlow Eq. 24)."),
        ("n_fp", "FP", "neutral", "Unmatched generated peaks per window at {t} ms; summed by micro_f1."),
        ("n_fn", "FN", "neutral", "Unmatched reference peaks per window at {t} ms; summed by micro_f1."),
    )
    return tuple(
        MetricSpec(f"{key}_{int(round(tol))}ms", f"{name} @{int(round(tol))} ms", orientation, "window", "ecg",
                   text.format(t=int(round(tol))) + " Reference peaks are the detector's output on the real ECG; matching is greedy one-to-one by |dt|.",
                   ("KANFlow", "repo") if tol == 50.0 else ("repo",), _P + "rpeak_prf_at")
        for tol in TOLERANCES_MS
        for key, name, orientation, text in rows
    )


PAPER_METRIC_SPEC: tuple[MetricSpec, ...] = (
    MetricSpec("mae", "MAE", "lower_is_better", "window", "ecg",
               "Per-window mean |pred - target| in z-score units. Equals KANFlow Eq. 21 only after averaging over windows, because every window has the same length L.",
               ("KANFlow", "PPGFlowECG"), "ppg2ecg.evaluation.metrics.signal_metrics"),
    MetricSpec("pooled_mae", "MAE (pooled)", "lower_is_better", "dataset", "ecg",
               "KANFlow Eq. 21: (1/(N L)) sum_n sum_l |x - xhat|, pooled over all segments and samples.",
               ("KANFlow", "PPGFlowECG"), _P + "pooled_mae"),
    MetricSpec("rmse", "RMSE", "lower_is_better", "window", "ecg",
               "Per-window sqrt(mean squared error), z-score units.",
               ("KANFlow", "PPGFlowECG"), "ppg2ecg.evaluation.metrics.signal_metrics"),
    MetricSpec("pooled_rmse", "RMSE (pooled)", "lower_is_better", "dataset", "ecg",
               "KANFlow Eq. 22: the square root is taken ONCE after pooling over all segments and samples. Pooled >= mean-of-per-window by Jensen; only this variant reproduces KANFlow Table II.",
               ("KANFlow", "PPGFlowECG"), _P + "pooled_rmse"),
    MetricSpec("mean_window_rmse", "RMSE (mean of windows)", "lower_is_better", "dataset", "ecg",
               "Mean over windows of the per-window RMSE -- the alternative reading KANFlow does NOT use; reported alongside pooled_rmse to expose the Jensen gap that PPGFlowECG leaves ambiguous.",
               ("PPGFlowECG",), _P + "mean_window_rmse"),
    MetricSpec("mse", "MSE", "lower_is_better", "window", "ecg",
               "Per-window mean squared error, for papers that tabulate MSE rather than RMSE.",
               ("repo",), _P + "mse"),
    MetricSpec("pcc", "PCC", "higher_is_better", "window", "ecg",
               "Per-window Pearson correlation between prediction and ground truth.",
               ("repo",), "ppg2ecg.evaluation.metrics.signal_metrics"),
    MetricSpec("prd_raw", "PRD", "lower_is_better", "window", "ecg",
               "100 sqrt( sum (p-t)^2 / sum t^2 ): the energy-normalised Percentage Root-mean-square Difference.",
               ("CardioGAN-lineage",), _P + "prd"),
    MetricSpec("prd_meansub", "PRD (mean-subtracted)", "lower_is_better", "window", "ecg",
               "100 sqrt( sum (p-t)^2 / sum (t - mean t)^2 ): the compression-literature variant; equals prd_raw iff the target window is zero-mean.",
               ("CardioGAN-lineage",), _P + "prd"),
    MetricSpec("cosine_similarity", "Cosine similarity", "higher_is_better", "window", "ecg",
               "Per-window cosine of the angle between prediction and ground truth (not mean-subtracted).",
               ("repo",), _P + "cosine_similarity"),
    MetricSpec("snr_db", "SNR", "higher_is_better", "window", "ecg",
               "10 log10( sum t^2 / sum (p-t)^2 ) in dB; +inf for an exact reconstruction.",
               ("repo",), _P + "snr_db"),
    MetricSpec("discrete_frechet", "Discrete Frechet", "lower_is_better", "window", "ecg",
               "CardioGAN's coupling Frechet distance: min over order-preserving pairings of the max |p_i - t_j|. NOT the Gaussian FD that KANFlow and PPGFlowECG report.",
               ("CardioGAN-lineage",), _P + "discrete_frechet"),
    MetricSpec("dtw", "DTW", "lower_is_better", "window", "ecg",
               "Sakoe-Chiba-banded dynamic time warping cost (unnormalised sum of |p_i - t_j| along the optimal warping); band 64 samples = 0.5 s at 128 Hz.",
               ("repo",), _P + "dtw_distance"),
    MetricSpec("kanflow_fd", "FD", "lower_is_better", "dataset", "ecg",
               "KANFlow Eq. 23: |mu_r - mu_g|^2 + Tr(S_r + S_g - 2 (S_r S_g)^{1/2}) with the RAW FLATTENED WAVEFORM as the feature vector; PCA to <=32 dims averaged over 5 trials below 3000 segments, raw space above; eps = 1e-4. Interpretable only within a dataset.",
               ("KANFlow", "PPGFlowECG"), _P + "kanflow_fd"),
    MetricSpec("fid_default_features", "FID (surrogate features)", "lower_is_better", "dataset", "ecg",
               "The same Gaussian Frechet formula on feature activations. PPGFlowECG uses ECGFounder; with no checkpoint available this uses `default_feature_map` (32 log-band powers + 4 moments), so the value is NOT comparable to any published FID.",
               ("PPGFlowECG",), _P + "fid_frechet"),
    MetricSpec("fid_ecgfounder", "FID (ECGFounder)", "lower_is_better", "dataset", "ecg",
               "PPGFlowECG's FID, computed on ECGFounder [Li et al. 2024] activations. NOT IMPLEMENTED: requires the external ECGFounder checkpoint, and the paper names no layer or feature dimensionality.",
               ("PPGFlowECG",), None),
    MetricSpec("micro_f1", "Micro-F1", "higher_is_better", "dataset", "ecg",
               "KANFlow Eqs. 24-25: TP/FP/FN summed over the whole dataset, then P, R and F1 from those sums (50 ms tolerance).",
               ("KANFlow",), _P + "micro_f1"),
    MetricSpec("macro_f1", "Macro-F1", "higher_is_better", "dataset", "ecg",
               "KANFlow Eqs. 26-27: mean of the per-segment F1 over segments with a valid peak-based evaluation -- a segment counts iff its F1 is finite AND the reference has at least one detected beat (with no reference beats the F1 is 0 by convention, not a score); a segment with reference beats and no predicted beats stays in and scores 0. Differs from Micro-F1 only in aggregation.",
               ("KANFlow",), _P + "macro_f1"),
    MetricSpec("n_macro_f1_excluded_segments", "Macro-F1 segments excluded", "neutral", "dataset", "ecg",
               "Number of segments excluded from Macro-F1 as not evaluable (no detected reference beats, or a non-finite window); reported so the Macro-F1 denominator is never invisible.",
               ("repo",), _P + "macro_f1"),
    MetricSpec("n_excluded_windows", "Windows excluded (non-finite)", "neutral", "dataset", "ecg",
               "Number of windows dropped from every dataset-level number because some sample was non-finite; the pooled metrics are computed over the finite windows only.",
               ("repo",), _P + "paper_metric_pooled"),
    MetricSpec("rr_mae_ms", "RR-MAE", "lower_is_better", "window", "ecg",
               "KANFlow Eq. 28, ms: per-segment MAE of R-R intervals over CONSECUTIVE MATCHED beats only, which makes it optimistic and coupled to the 50 ms matching step.",
               ("KANFlow",), "ppg2ecg.evaluation.rpeaks.rr_mae_ms"),
    MetricSpec("rr_mae_ms_macro", "RR-MAE (dataset)", "lower_is_better", "dataset", "ecg",
               "KANFlow Eq. 29, ms: mean of the per-segment RR-MAE over segments with at least one valid RR comparison (per-segment then averaged, not pooled over RR pairs).",
               ("KANFlow",), _P + "paper_metric_pooled"),
    MetricSpec("hr_ref", "HR (reference)", "neutral", "window", "ecg",
               "Heart rate in bpm from the ground-truth ECG (60 / mean RR of the detected peaks); a diagnostic, not a score.",
               ("repo",), "ppg2ecg.evaluation.rpeaks.hr_bpm"),
    MetricSpec("hr_pred", "HR (predicted)", "neutral", "window", "ecg",
               "Heart rate in bpm from the generated ECG; a diagnostic, not a score.", ("repo",), "ppg2ecg.evaluation.rpeaks.hr_bpm"),
    MetricSpec("hr_abs_err", "HR error", "lower_is_better", "window", "ecg",
               "Per-window |HR_pred - HR_ref| in bpm.", ("KANFlow", "PENGUIN", "PPGFlowECG"), _P + "beat_level_metrics"),
    MetricSpec("hr_mae_bpm", "MAE_HR", "lower_is_better", "dataset", "ecg",
               "KANFlow Eq. 30 / PENGUIN 'HR Error' / PPGFlowECG 'MAE_HR', bpm. Window length is NOT part of the metric and differs across papers (KANFlow 4 s, PENGUIN 8 s, PPGFlowECG 10 s), so the three columns must never share a comparison cell without that caveat.",
               ("KANFlow", "PENGUIN", "PPGFlowECG"), _P + "hr_mae_bpm"),
    MetricSpec("hr_rmse_bpm", "RMSE_HR", "lower_is_better", "dataset", "ecg",
               "Pooled RMSE of the same per-window HR errors; no paper reports it, included so HR is not summarised by MAE alone.",
               ("repo",), _P + "hr_rmse_bpm"),
    MetricSpec("qrs_region_rmse", "QRS-region RMSE", "lower_is_better", "window", "ecg",
               "RDDM's region-disentangled reporting: RMSE restricted to +-50 ms around each reference R peak.",
               ("RDDM-lineage",), _P + "qrs_region_rmse"),
    MetricSpec("non_qrs_rmse", "Non-QRS RMSE", "lower_is_better", "window", "ecg",
               "RMSE on the complement of the QRS regions.", ("RDDM-lineage",), _P + "non_qrs_rmse"),
    MetricSpec("qrs_region_frac", "QRS-region fraction", "neutral", "window", "ecg",
               "Fraction of window samples inside the QRS regions; context for the two region RMSEs.", ("repo",), _P + "region_rmse"),
    MetricSpec("qrs_width_err_ms", "QRS-width error", "lower_is_better", "window", "ecg",
               "Mean |QRS width difference| over matched beats, ms (QS-trough proxy).", ("repo",), "ppg2ecg.evaluation.rpeaks.qrs_width_error_ms"),
    MetricSpec("morph_corr", "Morphology correlation", "higher_is_better", "window", "ecg",
               "Mean Pearson correlation of matched beats aligned at their own R peaks.", ("repo",), "ppg2ecg.evaluation.rpeaks.morphology_corr"),
    MetricSpec("n_ref_beats", "Beats (reference)", "neutral", "window", "ecg",
               "Number of detected reference beats; diagnostic.", ("repo",), _P + "beat_level_metrics"),
    MetricSpec("n_pred_beats", "Beats (predicted)", "neutral", "window", "ecg",
               "Number of detected predicted beats; diagnostic.", ("repo",), _P + "beat_level_metrics"),
    MetricSpec("hf_energy_ratio_pred", "HF energy ratio (pred)", "neutral", "window", "ecg",
               "Fraction of spectral power above 15 Hz in the generated ECG; an over-smoothing indicator.", ("repo",), "ppg2ecg.evaluation.metrics.hf_energy_ratio"),
    MetricSpec("hf_energy_ratio_ref", "HF energy ratio (ref)", "neutral", "window", "ecg",
               "The same quantity on the ground-truth ECG, as the reference level.", ("repo",), "ppg2ecg.evaluation.metrics.hf_energy_ratio"),
    MetricSpec("resp_rate_abs_err_bpm", "RR Error (respiratory)", "lower_is_better", "window", "resp",
               "PENGUIN: MAE in breaths/min between the dominant non-negative FFT frequencies of the reconstructed and ground-truth respiratory waveforms, 60 s window. NOT KANFlow's RR-MAE, which is an R-R interval error in ms.",
               ("PENGUIN",), _P + "respiratory_rate_abs_err_bpm"),
    MetricSpec("sbp_abs_err_mmhg", "SBP Error", "lower_is_better", "window", "abp",
               "PENGUIN: |max(pred) - max(gt)| of the 8 s ABP window in mmHg. ABP receives no preprocessing, so this is on a real physical scale.",
               ("PENGUIN",), _P + "sbp_abs_err_mmhg"),
    MetricSpec("dbp_abs_err_mmhg", "DBP Error", "lower_is_better", "window", "abp",
               "PENGUIN: |min(pred) - min(gt)| of the 8 s ABP window in mmHg.", ("PENGUIN",), _P + "dbp_abs_err_mmhg"),
)

PAPER_METRIC_SPEC = PAPER_METRIC_SPEC + _rpeak_specs()
SPEC_BY_KEY: dict[str, MetricSpec] = {s.key: s for s in PAPER_METRIC_SPEC}
