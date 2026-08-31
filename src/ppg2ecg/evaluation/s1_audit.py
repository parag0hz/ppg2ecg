"""S1.2-S1.6 primitives, implementing docs/S1_METRIC_VALIDITY_PREREGISTRATION.md (b749339) verbatim.

Every threshold, tolerance, delay regime, null construction and RNG seed here is the one the frozen
preregistration names. Where the preregistration names a quantity without giving its formula, the
operationalisation is written here and committed BEFORE any real-data S1.2-S1.6 number is computed; those
places are marked OPERATIONALISATION and listed in the report.

NO TRAINING. These are analysis primitives only.
"""
from __future__ import annotations

import numpy as np

from . import alignment_diagnostics as AD
from . import rpeaks as R

FS = 128
MATCH_TOL_MS = 50.0                 # X0's frozen one-to-one event tolerance
LOCAL_MAX_SHIFT_MS = AD.LOCAL_MAX_SHIFT_MS      # 150 ms, the frozen oracle radius
QRS_HALF_MS = AD.QRS_HALF_MS                    # 100 ms
BOOT_N, BOOT_SEED = 2000, 20260901              # prereg section 6
NULL_DRAWS, NULL_SEED = 20, 20260901            # prereg S1.4b / S1.4c

#: S1.6 - the project's frozen detector, and a standard independent alternative. Both are reported;
#: selecting the more favourable one is prohibited by the preregistration.
DETECTOR_A = "neurokit"
DETECTOR_B = "pantompkins1985"

#: S1.2 - the GT-R -> next-PPG-peak search window. This is the window already disclosed as exploratory
#: exposure in preregistration section 3, so using it introduces no new choice.
PAT_LO_MS, PAT_HI_MS = 0.0, 500.0


# ---------------------------------------------------------------------------------------- amplitude
def amp_rel(sig: np.ndarray, r: int, fs: int = FS, half_ms: float = QRS_HALF_MS) -> float:
    """OPERATIONALISATION of the preregistration's `amp_rel`.

    Peak absolute deviation within +-`half_ms` of index `r`, in units of the window's own robust scale
    (1.4826 x MAD about the window median). Frozen `QRS_HALF_MS` is the half-width.
    """
    sig = np.asarray(sig, dtype=np.float64)
    med = float(np.median(sig))
    scale = 1.4826 * float(np.median(np.abs(sig - med)))
    h = int(round(half_ms / 1000.0 * fs))
    a, b = max(0, int(r) - h), min(sig.size, int(r) + h + 1)
    if b <= a:
        return float("nan")
    return float(np.max(np.abs(sig[a:b] - med)) / (scale + 1e-12))


# ---------------------------------------------------------------------------------------- S1.2
def dsp_ppg_peaks(ppg: np.ndarray, fs: int = FS) -> np.ndarray:
    """`nk.ppg_findpeaks(nk.ppg_clean(ppg))` at library defaults. No tuning of any kind."""
    import warnings

    import neurokit2 as nk

    x = np.asarray(ppg, dtype=np.float64)
    if x.size < fs or not np.isfinite(x).all() or x.std() < 1e-8:
        return np.zeros(0, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return np.asarray(nk.ppg_findpeaks(nk.ppg_clean(x, sampling_rate=fs), sampling_rate=fs)["PPG_Peaks"], dtype=int)
        except Exception:  # noqa: BLE001  (detector failure => no peaks)
            return np.zeros(0, dtype=int)


def pat_delays_ms(gt_peaks: np.ndarray, ppg_peaks: np.ndarray, fs: int = FS,
                  lo_ms: float = PAT_LO_MS, hi_ms: float = PAT_HI_MS) -> np.ndarray:
    """GT R-peak -> NEXT PPG peak delay, in ms, for GT beats with such a peak inside (lo, hi]."""
    g, p = np.asarray(gt_peaks, dtype=int), np.sort(np.asarray(ppg_peaks, dtype=int))
    if g.size == 0 or p.size == 0:
        return np.zeros(0)
    j = np.searchsorted(p, g, side="right")               # strictly after the R-peak
    ok = j < p.size
    d = np.full(g.size, np.nan)
    d[ok] = (p[j[ok]] - g[ok]) / fs * 1000.0
    return d[np.isfinite(d) & (d > lo_ms) & (d <= hi_ms)]


def shift_peaks(peaks: np.ndarray, delay_ms: float, fs: int = FS, n_time: int | None = None) -> np.ndarray:
    """Move a PPG peak train back by `delay_ms` to become a predicted R-peak train. Out-of-range dropped."""
    p = np.asarray(peaks, dtype=int) - int(round(delay_ms / 1000.0 * fs))
    p = p[p >= 0]
    return p if n_time is None else p[p < int(n_time)]


# ---------------------------------------------------------------------------------------- S1.4a
def classify_unmatched(gt_peaks, pred_peaks, pred_sig, matches, threshold: float,
                       fs: int = FS, tol_ms: float = MATCH_TOL_MS,
                       radius_ms: float = LOCAL_MAX_SHIFT_MS) -> dict:
    """Preregistered three-way split of UNMATCHED GT beats, searching +-150 ms (not +-100 ms).

    DISPLACED : a detected predicted peak at distance in (tol, radius] ms
    WEAK      : no such peak, but amp_rel(pred_sig, r) >= threshold
    ABSENT    : neither

    `contested` counts unmatched GT beats that do have a predicted peak within the match tolerance
    (one consumed by another GT beat under the one-to-one matcher). Reported as a bookkeeping check,
    never as a fourth class; such beats still fall into WEAK or ABSENT by the rule above.
    """
    g = np.asarray(gt_peaks, dtype=int)
    p = np.asarray(pred_peaks, dtype=int)
    matched = {int(i) for i, _ in matches}
    out = {"displaced": 0, "weak": 0, "absent": 0, "contested": 0, "n_unmatched": 0}
    for i, r in enumerate(g):
        if i in matched:
            continue
        out["n_unmatched"] += 1
        d = np.abs(p - int(r)) / fs * 1000.0 if p.size else np.zeros(0)
        if d.size and np.any(d <= tol_ms):
            out["contested"] += 1
        if d.size and np.any((d > tol_ms) & (d <= radius_ms)):
            out["displaced"] += 1
        elif amp_rel(pred_sig, int(r), fs) >= threshold:
            out["weak"] += 1
        else:
            out["absent"] += 1
    return out


# ---------------------------------------------------------------------------------------- S1.4b
def _zn_rows(m: np.ndarray) -> np.ndarray:
    """Row-wise standardisation matching `alignment_diagnostics._zn` (population std, zeros if flat)."""
    m = np.asarray(m, dtype=np.float64)
    c = m - m.mean(axis=-1, keepdims=True)
    s = np.sqrt((c ** 2).mean(axis=-1, keepdims=True))
    return np.divide(c, s, out=np.zeros_like(c), where=s > 1e-12)


def oracle_max_corr(pred: np.ndarray, gt_seg: np.ndarray, a: int, b: int, max_shift: int) -> tuple[int, float]:
    """Vectorised equivalent of `alignment_diagnostics.oracle_local_shift`, including its tie rule.

    Returns (best_d, best_corr) maximising corr(pred[a+d:b+d], gt_seg) over d in [-max_shift, max_shift];
    ties resolved to the smallest |d|; a flat prediction segment scores -1.0.
    """
    ds = np.arange(-max_shift, max_shift + 1)
    idx = (np.arange(a, b)[None, :] + ds[:, None])
    segs = np.asarray(pred, dtype=np.float64)[idx]
    flat = segs.std(axis=-1) <= 1e-12
    c = (_zn_rows(segs) * _zn_rows(gt_seg[None, :])).mean(axis=-1)
    c[flat] = -1.0
    best = -np.inf
    best_d = 0
    for d, cc in zip(ds, c):
        if cc > best + 1e-12 or (abs(cc - best) <= 1e-12 and abs(int(d)) < abs(best_d)):
            best, best_d = float(cc), int(d)
    return best_d, best


def oracle_null_gain(pred: np.ndarray, gt: np.ndarray, gt_peaks: np.ndarray, rng, fs: int = FS,
                     n_draws: int = NULL_DRAWS) -> dict:
    """S1.4b: the same +-150 ms maximisation applied to a MISMATCHED pair.

    For each valid GT beat i, draw a different GT beat j from the SAME window and correlate the
    prediction segment at j's coordinates against the reference segment at i's coordinates, both at the
    same coordinates and after the identical 39-shift maximisation. The difference is the chance level of
    the oracle's gain.
    """
    max_shift = int(round(LOCAL_MAX_SHIFT_MS / 1000.0 * fs))
    segs, _ = AD.beat_segments_gt(gt, gt_peaks, fs, margin=max_shift)
    n_b = len(segs)
    if n_b < 2:
        return {"n_beats": n_b, "null_same": np.nan, "null_oracle": np.nan, "n_pairs": 0}
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    ds = np.arange(-max_shift, max_shift + 1)

    # Precompute every (reference beat i, prediction beat j, shift d) correlation once, then let the draws
    # sample from it. Mathematically identical to evaluating each drawn pair separately, but O(n^2) instead
    # of O(draws x n) oracle searches.
    L = segs[0][1] - segs[0][0]
    P = np.empty((n_b, ds.size, L))
    G = np.empty((n_b, L))
    flat = np.empty((n_b, ds.size), dtype=bool)
    for j, (aj, bj, _) in enumerate(segs):
        idx = np.arange(aj, bj)[None, :] + ds[:, None]
        seg = pred[idx]
        flat[j] = seg.std(axis=-1) <= 1e-12
        P[j] = _zn_rows(seg)
    for i, (ai, bi, _) in enumerate(segs):
        G[i] = _zn_rows(gt[ai:bi][None, :])[0]
    corr = np.einsum("jdl,il->ijd", P, G) / L             # (ref i, pred j, shift d)
    corr[:, flat] = -1.0                                  # frozen rule: a flat prediction segment scores -1
    zero = int(np.flatnonzero(ds == 0)[0])
    same_ij = corr[:, :, zero]
    orac_ij = corr.max(axis=-1)

    same, orac, n = [], [], 0
    for _ in range(int(n_draws)):
        for i in range(n_b):
            j = int(rng.integers(n_b - 1))
            j = j + 1 if j >= i else j                     # uniform over j != i
            same.append(same_ij[i, j])
            orac.append(orac_ij[i, j])
            n += 1
    return {"n_beats": n_b, "null_same": float(np.nanmean(same)), "null_oracle": float(np.nanmean(orac)),
            "n_pairs": n}


# ---------------------------------------------------------------------------------------- S1.4c
def chance_random_phase(n_beats: int, n_time: int, rng) -> np.ndarray:
    """Count-matched, rate-matched, random-phase peak train: n evenly spaced peaks at a random offset."""
    n = int(n_beats)
    if n <= 0:
        return np.zeros(0, dtype=int)
    step = n_time / n
    off = float(rng.random()) * step
    p = np.unique(np.mod(np.round(off + step * np.arange(n)), n_time).astype(int))
    return np.sort(p)


def chance_circular_shift(peaks: np.ndarray, n_time: int, rng) -> np.ndarray:
    """Count-preserving circular shift of an existing peak train."""
    p = np.asarray(peaks, dtype=int)
    if p.size == 0:
        return p
    return np.sort(np.mod(p + int(rng.integers(n_time)), int(n_time)))


# ---------------------------------------------------------------------------------------- stats
def subject_bootstrap(values, subjects, n_boot: int = BOOT_N, seed: int = BOOT_SEED, agg=np.nanmean):
    """Subject-stratified bootstrap with equal subject weight (prereg section 6)."""
    v = np.asarray(values, dtype=np.float64)
    s = np.asarray(subjects)
    uniq = sorted(set(s.tolist()))
    idx = {u: np.flatnonzero(s == u) for u in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        draws[b] = float(np.mean([agg(v[rng.choice(idx[u], idx[u].size, replace=True)]) for u in uniq]))
    point = float(np.mean([agg(v[idx[u]]) for u in uniq]))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot": int(n_boot), "seed": int(seed)}


def macro(values, subjects, agg=np.nanmean) -> float:
    """Equal-subject-weight macro mean."""
    v, s = np.asarray(values, dtype=np.float64), np.asarray(subjects)
    return float(np.mean([agg(v[s == u]) for u in sorted(set(s.tolist()))]))


# ---------------------------------------------------------------------------------------- S1.6
def detector_agreement(sig: np.ndarray, fs: int = FS, tol_ms: float = MATCH_TOL_MS) -> dict:
    """Agreement of the two preregistered detectors on the SAME reference signal. Both reported."""
    a = R.detect_rpeaks(sig, fs, DETECTOR_A)
    b = R.detect_rpeaks(sig, fs, DETECTOR_B)
    m, fp, fn = R.match_rpeaks(a, b, fs, tol_ms)
    p, r, f = R.prf(len(m), fp, fn)
    return {"n_a": int(a.size), "n_b": int(b.size), "n_matched": int(len(m)),
            "precision_b_vs_a": p, "recall_b_vs_a": r, "f1": f,
            "beat_count_diff": int(abs(a.size - b.size)),
            "hr_a": R.hr_bpm(a, fs), "hr_b": R.hr_bpm(b, fs)}


def rr_plausibility(peaks: np.ndarray, fs: int = FS, lo_ms: float = 333.0, hi_ms: float = 1500.0) -> dict:
    """RR and beat-count plausibility census, using the bounds already used in the X4-0 line of work."""
    p = np.asarray(peaks, dtype=int)
    if p.size < 2:
        return {"n_rr": 0, "n_rr_out": 0, "frac_rr_out": np.nan, "beat_count": int(p.size),
                "count_plausible": bool(4 <= p.size <= 25)}
    rr = np.diff(p) / fs * 1000.0
    out = int(np.sum((rr < lo_ms) | (rr > hi_ms)))
    return {"n_rr": int(rr.size), "n_rr_out": out, "frac_rr_out": float(out / rr.size),
            "beat_count": int(p.size), "count_plausible": bool(4 <= p.size <= 25)}
