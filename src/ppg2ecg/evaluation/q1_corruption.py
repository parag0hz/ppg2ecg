"""Q1 — conditional support degradation audit: frozen corruption families, cohorts, plausibility proxies,
quality scores and the preregistered verdict logic.

Frozen by docs/Q1_CONDITIONAL_SUPPORT_DEGRADATION_PREREGISTRATION.md. ANALYSIS ONLY: nothing here trains,
holds an optimizer, or creates a parameter that requires grad. Every corruption is deterministic given
(subject, site, window_index, condition) and never depends on global RNG state, on the ECG, or on any
signal-derived statistic other than the RMS scaling explicitly required for the SNR definition.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
from scipy import signal as sps

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.metrics import hf_energy_ratio

FS, T_LEN = 128, 1024
NYQ = FS / 2.0

# ---------------------------------------------------------------------------------------------- frozen corruption
FILTER_ORDER = 4
LP_CUTOFFS_HZ = (3.0, 2.0, 1.25)
NOISE_BAND_HZ = (0.5, 4.0)
SNR_DB = (20.0, 10.0, 5.0, 0.0)
DROP_S = (0.5, 1.0, 2.0)
NOISE_SALT = "q1-noise-v1"
DROP_SALT = "q1-drop-v1"
UNCERTAINTY_SALT = "q1-uncertainty-v1"
SHUFFLE_SALT = "q1-condition-shuffle-v1"
REFERENCE_SALT = "q1-marginal-reference-v1"

CLEAN = "CLEAN"
LP_CONDS = tuple(f"LP_{c}Hz" for c in LP_CUTOFFS_HZ)                    # LP_3.0Hz, LP_2.0Hz, LP_1.25Hz
SNR_CONDS = tuple(f"SNR_{int(s)}dB" for s in SNR_DB)                    # SNR_20dB .. SNR_0dB
DROP_CONDS = tuple(f"DROP_{d}s" for d in DROP_S)                        # DROP_0.5s, DROP_1.0s, DROP_2.0s
SHUFFLED, NULL = "SHUFFLED", "NULL"
NATURAL_CONDITIONS = (CLEAN,) + LP_CONDS + SNR_CONDS + DROP_CONDS
CONDITIONS = NATURAL_CONDITIONS + (SHUFFLED, NULL)
FAMILIES = {"BANDLIMIT": LP_CONDS, "NOISE": SNR_CONDS, "DROPOUT": DROP_CONDS}
SEVERE = {"BANDLIMIT": LP_CONDS[-1], "NOISE": SNR_CONDS[-1], "DROPOUT": DROP_CONDS[-1]}
FAMILY_OF = {c: f for f, cs in FAMILIES.items() for c in cs}

# ---------------------------------------------------------------------------------------------- frozen protocol
NFE_PRIMARY = 4
SRC_SEED = 0
UNC_SEEDS = tuple(range(8))
N_UNCERTAINTY_PER_STRATUM = 64
N_REFERENCE_PER_STRATUM = 256
BOOT_N, BOOT_SEED = 2000, 20260903
PLAUS_PCTS = (1.0, 99.0)
PLAUS_FEATURES = ("hr_bpm", "qrs_width_ms", "qrs_p2p", "max_deriv", "hf_ratio")   # P2..P6
DET_VALID_DROP_MAX = 0.05          # P-A
MARGINAL_DROP_MAX = 0.05           # P-B
UNC_REL_INCREASE = 0.10            # U-A
PERIODICITY_BPM = (30.0, 200.0)
MIN_PULSES_TEMPLATE = 3
PREFLIGHT_WINDOWS = 100
BUDGET_GPU_HOURS = 4.0
TRAIN12 = ("e61", "fex", "l38", "n31", "ngh", "p5d", "p9p", "qm9", "trh", "tz8", "u7y", "w4p")
VAL = ("an0", "k2s")


def _int_seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------------------------- A. band-limit
def lowpass_coeffs(cutoff_hz: float):
    """Frozen zero-phase design: 4th-order Butterworth low-pass at `cutoff_hz` (Nyquist 64 Hz)."""
    return sps.butter(FILTER_ORDER, float(cutoff_hz) / NYQ, btype="low")


def bandpass_coeffs(band_hz=NOISE_BAND_HZ):
    lo, hi = float(band_hz[0]) / NYQ, float(band_hz[1]) / NYQ
    return sps.butter(FILTER_ORDER, [lo, hi], btype="band")


def apply_lowpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    b, a = lowpass_coeffs(cutoff_hz)
    return sps.filtfilt(b, a, np.asarray(x, dtype=np.float64))


# ---------------------------------------------------------------------------------------------- B. in-band noise
def noise_seed(subject: str, site: str, window_index: int, tag: str) -> int:
    return _int_seed(f"{NOISE_SALT}|{subject}|{site}|{int(window_index)}|{tag}")


def apply_noise(x: np.ndarray, subject: str, site: str, window_index: int, snr_db: float) -> np.ndarray:
    """x + band-limited Gaussian noise scaled to EXACTLY `snr_db` = 20 log10(rms(x)/rms(added))."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(noise_seed(subject, site, window_index, f"SNR_{int(snr_db)}dB"))
    b, a = bandpass_coeffs()
    n_bp = sps.filtfilt(b, a, rng.standard_normal(x.size))
    rms_x = float(np.sqrt(np.mean(x ** 2)))
    rms_n = float(np.sqrt(np.mean(n_bp ** 2)))
    if rms_n <= 0 or rms_x <= 0:
        return x.copy()
    return x + n_bp * (rms_x / rms_n) * (10.0 ** (-float(snr_db) / 20.0))


def achieved_snr_db(clean: np.ndarray, corrupted: np.ndarray) -> float:
    c = np.asarray(clean, dtype=np.float64)
    d = np.asarray(corrupted, dtype=np.float64) - c
    return float(20.0 * np.log10(np.sqrt(np.mean(c ** 2)) / max(np.sqrt(np.mean(d ** 2)), 1e-300)))


# ---------------------------------------------------------------------------------------------- C. dropout
def drop_samples(duration_s: float) -> int:
    return int(round(float(duration_s) * FS))


def drop_start(subject: str, site: str, window_index: int, duration_s: float, n_time: int = T_LEN) -> int:
    """Metadata-only placement: 1 <= start and start + L <= n_time - 1, so both boundary samples exist."""
    L = drop_samples(duration_s)
    span = n_time - L - 1
    if span < 1:
        raise ValueError(f"dropout of {L} samples does not fit in {n_time}")
    h = _int_seed(f"{DROP_SALT}|{subject}|{site}|{int(window_index)}|DROP_{duration_s}s")
    return 1 + int(h % span)


def apply_dropout(x: np.ndarray, subject: str, site: str, window_index: int, duration_s: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    L = drop_samples(duration_s)
    s = drop_start(subject, site, window_index, duration_s, x.size)
    x[s:s + L] = np.linspace(x[s - 1], x[s + L], L + 2)[1:-1]
    return x


# ---------------------------------------------------------------------------------------------- dispatch
def corrupt_row(x: np.ndarray, condition: str, subject: str, site: str, window_index: int) -> np.ndarray:
    """One frozen preprocessed PPG row -> corrupted row. NO renormalisation is applied anywhere."""
    if condition == CLEAN:
        return np.asarray(x, dtype=np.float64).copy()
    if condition in LP_CONDS:
        return apply_lowpass(x, LP_CUTOFFS_HZ[LP_CONDS.index(condition)])
    if condition in SNR_CONDS:
        return apply_noise(x, subject, site, window_index, SNR_DB[SNR_CONDS.index(condition)])
    if condition in DROP_CONDS:
        return apply_dropout(x, subject, site, window_index, DROP_S[DROP_CONDS.index(condition)])
    raise ValueError(f"{condition!r} is not a per-row corruption (SHUFFLED/NULL act on the population)")


def corrupt_block(X: np.ndarray, condition: str, subjects, sites, window_index, partner=None) -> np.ndarray:
    """Population-level corruption -> float32 block of the same shape. SHUFFLED/NULL need `partner`/nothing."""
    X = np.asarray(X)
    if condition == NULL:
        return np.zeros_like(X, dtype=np.float32)
    if condition == SHUFFLED:
        if partner is None:
            raise ValueError("SHUFFLED needs the partner permutation")
        return X[np.asarray(partner)].astype(np.float32)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(len(X)):
        out[i] = corrupt_row(X[i], condition, str(subjects[i]), str(sites[i]), int(window_index[i])).astype(np.float32)
    return out


# ---------------------------------------------------------------------------------------------- cohorts (metadata only)
def uncertainty_positions(subjects, sites, window_index, n_per: int = N_UNCERTAINTY_PER_STRATUM,
                          salt: str = UNCERTAINTY_SALT) -> np.ndarray:
    """Balanced metadata-only subset of the primary population: lowest SHA256 ranks per (subject, site)."""
    subjects, sites, window_index = np.asarray(subjects), np.asarray(sites), np.asarray(window_index)
    keep: list[int] = []
    for sub in sorted(set(subjects.tolist())):
        for site in sorted(set(sites[subjects == sub].tolist())):
            m = np.flatnonzero((subjects == sub) & (sites == site))
            keys = [hashlib.sha256(f"{salt}|{sub}|{site}|{int(window_index[i])}".encode()).hexdigest() for i in m]
            keep += m[np.argsort(keys, kind="stable")[:int(n_per)]].tolist()
    return np.sort(np.asarray(keep, dtype=np.int64))


def reference_positions(subject: str, sites: np.ndarray, window_index: np.ndarray,
                        n_per: int = N_REFERENCE_PER_STRATUM, salt: str = REFERENCE_SALT) -> np.ndarray:
    """Train-subject marginal-reference cohort positions (metadata only)."""
    if subject not in TRAIN12:
        raise RuntimeError(f"marginal reference may only use the 12 WildPPG train subjects; got {subject!r}")
    ER.assert_no_test_subjects([subject])
    sites, window_index = np.asarray(sites), np.asarray(window_index)
    keep: list[int] = []
    for site in sorted(set(sites.tolist())):
        m = np.flatnonzero(sites == site)
        keys = [hashlib.sha256(f"{salt}|{subject}|{site}|{int(window_index[i])}".encode()).hexdigest() for i in m]
        keep += m[np.argsort(keys, kind="stable")[:int(n_per)]].tolist()
    return np.sort(np.asarray(keep, dtype=np.int64))


# ---------------------------------------------------------------------------------------------- corruption sanity
def spectrum(x: np.ndarray) -> np.ndarray:
    p = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) - np.mean(x))) ** 2
    return p / max(p.sum(), 1e-300)


SANITY_BANDS = ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 15.0), (15.0, 64.0))


def band_fractions(x: np.ndarray) -> dict:
    p = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) - np.mean(x))) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0 / FS)
    tot = max(p.sum(), 1e-300)
    return {f"band_frac_{lo}-{hi}Hz": float(p[(f >= lo) & (f < hi)].sum() / tot) for lo, hi in SANITY_BANDS}


def noise_band_confinement(clean: np.ndarray, corrupted: np.ndarray, lo: float = 0.4, hi: float = 4.5) -> float:
    """Fraction of the ADDED signal's power inside [lo, hi] Hz (test: >= 0.99 for the noise family)."""
    d = np.asarray(corrupted, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    p = np.abs(np.fft.rfft(d - d.mean())) ** 2
    f = np.fft.rfftfreq(len(d), d=1.0 / FS)
    return float(p[(f >= lo) & (f <= hi)].sum() / max(p.sum(), 1e-300))


def ppg_sanity(clean: np.ndarray, corrupted: np.ndarray) -> dict:
    c, k = np.asarray(clean, dtype=np.float64), np.asarray(corrupted, dtype=np.float64)
    corr = float(np.corrcoef(c, k)[0, 1]) if c.std() > 1e-12 and k.std() > 1e-12 else np.nan
    out = {"ppg_corr": corr,
           "ppg_nrmse": float(np.linalg.norm(k - c) / (np.linalg.norm(c) + 1e-12)),
           "rms_ratio": float(np.sqrt(np.mean(k ** 2)) / (np.sqrt(np.mean(c ** 2)) + 1e-12)),
           "spec_l1": float(np.abs(spectrum(k) - spectrum(c)).sum())}
    out |= band_fractions(k)
    return out


def pulse_interval_mae_ms(clean_peaks, corrupted_peaks, tol_ms: float = 150.0) -> float:
    """MAE between corrupted and clean PPG pulse intervals over one-to-one matched consecutive pulses."""
    g, p = np.asarray(clean_peaks, dtype=int), np.asarray(corrupted_peaks, dtype=int)
    if len(g) < 2 or len(p) < 2:
        return np.nan
    m, _, _ = R.match_rpeaks(g, p, FS, tol_ms=tol_ms)
    mm = dict(m)
    errs = [abs(((g[i + 1] - g[i]) - (p[mm[i + 1]] - p[mm[i]])) / FS * 1000.0)
            for i in range(len(g) - 1) if i in mm and i + 1 in mm]
    return float(np.mean(errs)) if errs else np.nan


# ---------------------------------------------------------------------------------------------- plausibility (P1..P6)
def ecg_features(sig: np.ndarray, fs: int = FS) -> dict:
    """GT-INDEPENDENT marginal features of ONE ECG window, using the frozen detector and metric primitives.

    P1 detector_valid (>= 2 peaks), P2 hr_bpm, P3 qrs_width_ms (median over beats), P4 qrs_p2p (median 83-sample
    peak-to-peak), P5 max_deriv (median max |diff| * fs), P6 hf_ratio (>= 15 Hz power fraction).
    """
    x = np.asarray(sig, dtype=np.float64)
    pk = R.detect_rpeaks(x, fs)
    out = {"n_peaks": int(len(pk)), "detector_valid": bool(len(pk) >= 2),
           "hr_bpm": np.nan, "qrs_width_ms": np.nan, "qrs_p2p": np.nan, "max_deriv": np.nan,
           "hf_ratio": float(hf_energy_ratio(x[None])[0])}
    if len(pk) >= 2:
        out["hr_bpm"] = float(R.hr_bpm(pk, fs))
    widths, p2p, mderiv = [], [], []
    for r in pk:
        w = R.qrs_width_ms(x, int(r), fs)
        if np.isfinite(w):
            widths.append(float(w))
        seg = R.beat_window(x, int(r), fs)
        if seg is not None and len(seg) > 1:
            p2p.append(float(np.ptp(seg)))
            mderiv.append(float(np.max(np.abs(np.diff(seg))) * fs))
    if widths:
        out["qrs_width_ms"] = float(np.median(widths))
    if p2p:
        out["qrs_p2p"] = float(np.median(p2p))
        out["max_deriv"] = float(np.median(mderiv))
    return out


def reference_intervals(feature_rows, features=PLAUS_FEATURES, pcts=PLAUS_PCTS) -> dict:
    """Train-real [p1, p99] per feature over windows where the feature is defined."""
    out = {}
    for f in features:
        v = np.asarray([r[f] for r in feature_rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        lo, hi = (float(x) for x in np.percentile(v, list(pcts)))
        out[f] = {"p_lo": lo, "p_hi": hi, "n": int(v.size), "pct_lo": pcts[0], "pct_hi": pcts[1]}
    return out


def support_indicators(feat: dict, ref: dict, features=PLAUS_FEATURES) -> dict:
    """A feature is in support iff it is DEFINED and inside the frozen train-real interval."""
    ind = {}
    for f in features:
        v = feat[f]
        ind[f"in_support_{f}"] = float(np.isfinite(v) and ref[f]["p_lo"] <= v <= ref[f]["p_hi"])
    ind["marginal_support_fraction"] = float(np.mean([ind[f"in_support_{f}"] for f in features]))
    ind["detector_valid"] = float(bool(feat["detector_valid"]))
    return ind


# ---------------------------------------------------------------------------------------------- uncertainty (U1..U6)
def uncertainty_from_samples(preds: np.ndarray, peaks_per_sample, gt_peaks=None) -> dict:
    """preds [S, T] samples of ONE window from S different Gaussian sources; peaks already detected per sample."""
    P = np.asarray(preds, dtype=np.float64)
    S = P.shape[0]
    pairs = list(itertools.combinations(range(S), 2))
    counts = np.asarray([len(p) for p in peaks_per_sample], dtype=np.float64)
    out = {"u1_pointwise_sd": float(P.std(axis=0, ddof=0).mean()),
           "u2_pairwise_rmse": float(np.mean([np.sqrt(np.mean((P[i] - P[j]) ** 2)) for i, j in pairs])),
           "u3_beatcount_sd": float(counts.std(ddof=0)),
           "u4_pairwise_event_f1_50": float(np.mean([ER.peak_train_agreement(peaks_per_sample[i], peaks_per_sample[j], FS, 50.0)["f1"] for i, j in pairs])),
           "u5_pairwise_event_f1_150": float(np.mean([ER.peak_train_agreement(peaks_per_sample[i], peaks_per_sample[j], FS, 150.0)["f1"] for i, j in pairs])),
           "n_sources": int(S)}
    out["u6_gt_beat_timing_sd_ms"] = np.nan
    if gt_peaks is not None and len(gt_peaks) > 0:
        g = ER.gt_anchored_presence(gt_peaks, peaks_per_sample, FS, window_ms=250.0)
        ok = g["n_detected"] >= max(4, S // 2)
        sd = g["timing_sd_ms"][ok]
        sd = sd[np.isfinite(sd)]
        out["u6_gt_beat_timing_sd_ms"] = float(np.mean(sd)) if sd.size else np.nan
        out["u6_n_beats_used"] = int(sd.size)
    return out


# ---------------------------------------------------------------------------------------------- natural PPG quality
def periodicity_score(x: np.ndarray, bpm_range=PERIODICITY_BPM, fs: int = FS) -> float:
    """Maximum normalised autocorrelation over lags corresponding to `bpm_range` (30-200 bpm -> lags 39..256)."""
    v = np.asarray(x, dtype=np.float64)
    v = v - v.mean()
    d = float(np.dot(v, v))
    if d <= 0:
        return np.nan
    lo = int(np.ceil(60.0 / bpm_range[1] * fs))     # 200 bpm -> 38.4 -> 39
    hi = int(np.floor(60.0 / bpm_range[0] * fs))    # 30 bpm  -> 256
    hi = min(hi, v.size - 1)
    if hi < lo:
        return np.nan
    ac = np.correlate(v, v, mode="full")[v.size - 1:]
    return float(np.max(ac[lo:hi + 1]) / d)


def pulse_template_consistency(x: np.ndarray, pulse_peaks, fs: int = FS) -> float:
    """Median correlation between detected pulse snippets (frozen 83-sample beat window) and their median template."""
    v = np.asarray(x, dtype=np.float64)
    segs = [R.beat_window(v, int(p), fs) for p in np.asarray(pulse_peaks, dtype=int)]
    segs = [s for s in segs if s is not None and np.std(s) > 1e-8]
    if len(segs) < MIN_PULSES_TEMPLATE:
        return np.nan
    M = np.vstack(segs)
    tpl = np.median(M, axis=0)
    if np.std(tpl) <= 1e-8:
        return np.nan
    return float(np.median([np.corrcoef(s, tpl)[0, 1] for s in M]))


# ---------------------------------------------------------------------------------------------- preregistered verdicts
def _worse(res: dict) -> bool:
    """`res` comes from paired_subject_bootstrap with the orientation already applied (positive = clean better
    for support/fidelity/plausibility, = corrupted more uncertain for uncertainty)."""
    return bool(res.get("verdict") == "improves")


def family_flags(support: dict, fidelity: dict, plaus: dict, unc: dict) -> dict:
    """All inputs are dicts of metric -> paired bootstrap result / scalar drop, for ONE family's severe level.

    support: {"f1@150": res, "rr_mae_ms": res, "missing": res, "spurious": res}
    fidelity: {"f1_excess": res, "raw_qrs_rmse": res, "qrs_deriv_rmse": res, "qrs_curvature_err": res}
    plaus: {"detector_valid_drop": float, "marginal_support_drop": float}     (clean minus corrupted)
    unc: {"u1_rel_increase": float, "u3": res, "u4": res}
    """
    s_a = _worse(support["f1@150"])
    s_b = any(_worse(support[m]) for m in ("rr_mae_ms", "missing", "spurious"))
    f_a = _worse(fidelity["f1_excess"])
    f_b = any(_worse(fidelity[m]) for m in ("raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err"))
    p_a = float(plaus["detector_valid_drop"]) < DET_VALID_DROP_MAX
    p_b = float(plaus["marginal_support_drop"]) < MARGINAL_DROP_MAX
    u_a = not (float(unc["u1_rel_increase"]) >= UNC_REL_INCREASE)
    u_b = not (_worse(unc["u3"])) or not (_worse(unc["u4"]))
    return {"S_A": s_a, "S_B": s_b, "support_degrading": bool(s_a and s_b),
            "F_A": f_a, "F_B": f_b, "fidelity_degrading": bool(s_a and s_b and f_a and f_b),
            "P_A": p_a, "P_B": p_b, "plausibility_preserved": bool(p_a and p_b),
            "U_A": u_a, "U_B": u_b, "uncertainty_nonresponsive": bool(u_a and u_b),
            "uncertainty_clear_increase": bool(float(unc["u1_rel_increase"]) >= UNC_REL_INCREASE
                                               and (_worse(unc["u3"]) or _worse(unc["u4"]))),
            "plausibility_degrades": bool((not p_a) or (not p_b))}


VERDICT_A = "CONDITIONAL-SUPPORT / PLAUSIBILITY DECOUPLING OBSERVED"
VERDICT_B = "CONDITION LOSS IS REFLECTED IN GENERATOR UNCERTAINTY"
VERDICT_C = "OUTPUT PLAUSIBILITY COLLAPSES WITH CONDITION"
VERDICT_D = "NO CONSISTENT CONDITION-DEGRADATION PATTERN"


def decide_q1(flags_by_family: dict) -> dict:
    """Preregistration section 11, evaluated in the frozen order A -> B -> C -> D."""
    fam = flags_by_family
    a = [f for f, x in fam.items() if x["support_degrading"] and x["fidelity_degrading"]
         and x["plausibility_preserved"] and x["uncertainty_nonresponsive"]]
    b = [f for f, x in fam.items() if (x["support_degrading"] or x["fidelity_degrading"]) and x["uncertainty_clear_increase"]]
    c = [f for f, x in fam.items() if (x["support_degrading"] or x["fidelity_degrading"]) and x["plausibility_degrades"]]
    degrading = [f for f, x in fam.items() if x["support_degrading"]]
    if len(a) >= 2:
        v = VERDICT_A
    elif len(b) >= 2:
        v = VERDICT_B
    elif len(c) >= 2:
        v = VERDICT_C
    else:
        v = VERDICT_D
    if len(degrading) < 2 and v == VERDICT_D:
        pass
    return {"verdict": v, "families_A": a, "families_B": b, "families_C": c, "families_support_degrading": degrading,
            "n_families_A": len(a), "n_families_B": len(b), "n_families_C": len(c)}
