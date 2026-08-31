"""S1 G1 primitives: train-only canonical QRS template construction and event stamping.

Frozen by `docs/S1_METRIC_VALIDITY_PREREGISTRATION.md` (commit b749339) as amended by
`docs/S1_METRIC_VALIDITY_AMENDMENT_1.md` (commit dc75079).

The hard gate stamps an unmistakable QRS-like event at the exact ground-truth R-peak positions and scores
the result with the frozen, unmodified detector and matcher. A metric that cannot return ~1.0 for a signal
whose beats are in exactly the right place cannot be used to argue that a model's beats are in the wrong
place.

Template shape provenance is training-only; validation data supplies R-peak POSITIONS at stamping time and
contributes no shape information whatsoever.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .rpeaks import beat_window, detect_rpeaks

FS = 128

# --- frozen by the preregistration -------------------------------------------------------------------
TEMPLATE_SALT = "s1-template-v1"
TEMPLATE_N_TAKE = 256
#: the 12 A4 training subjects minus the two WildPPG author-flagged noisy-ECG participants (fex, p5d)
TEMPLATE_SUBJECTS = ("e61", "l38", "n31", "ngh", "p9p", "qm9", "trh", "tz8", "u7y", "w4p")
TEMPLATE_EXCLUDED_NOISY = ("fex", "p5d")
#: `morphology_corr`'s own beat window, so the template lives on the same support
BEFORE_S, AFTER_S = 0.25, 0.40
#: T-B QRS crop, frozen by Amendment 1 section 1B
QRS_BEFORE_S, QRS_AFTER_S = 0.080, 0.120
EPS = 1e-12


def template_geometry(fs: int = FS) -> dict:
    """The frozen discrete index convention, derived once and asserted by the unit tests.

    `beat_window(before_s=0.25, after_s=0.40)` returns `sig[r-32 : r+51]` at 128 Hz: length 83 with the
    R-peak at index 32. The T-B crop uses `int(round(seconds * fs))` on BOTH endpoints, inclusive.
    """
    n_before_full = int(round(BEFORE_S * fs))          # 32
    n_after_full = int(round(AFTER_S * fs))            # 51
    n_before = int(round(QRS_BEFORE_S * fs))           # 10
    n_after = int(round(QRS_AFTER_S * fs))             # 15
    return {
        "fs": fs,
        "full_len": n_before_full + n_after_full,      # 83
        "r_index_full": n_before_full,                 # 32
        "qrs_n_before": n_before,                      # 10
        "qrs_n_after": n_after,                        # 15
        "qrs_len": n_before + n_after + 1,             # 26
        "qrs_slice": (n_before_full - n_before, n_before_full + n_after + 1),   # (22, 48)
        "r_index_qrs": n_before,                       # 10
        "min_rr_samples_for_no_overlap": n_before + n_after + 1,               # 26
    }


def collect_train_beats(load_subject, subjects=TEMPLATE_SUBJECTS, fs: int = FS,
                        salt: str = TEMPLATE_SALT, n_take: int = TEMPLATE_N_TAKE) -> np.ndarray:
    """Extract the frozen train-only beat set. `load_subject(name) -> ecg array (n_windows, n_time)`.

    Returns (n_beats, full_len) float64. Raises if a validation or test subject is requested.
    """
    from .event_reliability import assert_no_test_subjects, select_subset

    subjects = tuple(subjects)
    assert_no_test_subjects(subjects)
    forbidden = {"an0", "k2s"} & set(subjects)
    if forbidden:
        raise ValueError(f"template shape must not come from validation subjects; got {sorted(forbidden)}")

    beats: list[np.ndarray] = []
    for s in subjects:
        ecg = np.asarray(load_subject(s))
        idx = select_subset(salt, s, ecg.shape[0], n_take, exclude=())
        for i in idx:
            sig = ecg[int(i)].astype(np.float64)
            for r in detect_rpeaks(sig, fs):
                w = beat_window(sig, int(r), fs, before_s=BEFORE_S, after_s=AFTER_S)
                if w is not None:
                    beats.append(np.asarray(w, dtype=np.float64))
    if not beats:
        raise RuntimeError("no training beats extracted")
    return np.stack(beats)


def build_template_a(beats: np.ndarray) -> tuple[np.ndarray, dict]:
    """Frozen Amendment 1 section 1B scaling.

        T_raw[t] = median_b x_b[t]
        A_target = median_b ( max(x_b) - min(x_b) )
        T_A      = T_raw * A_target / ( ptp(T_raw) + eps )

    No DC re-centering, no smoothing, no fitting, no per-window or per-subject scaling.
    """
    beats = np.asarray(beats, dtype=np.float64)
    t_raw = np.median(beats, axis=0)
    a_target = float(np.median(np.ptp(beats, axis=1)))
    raw_ptp = float(np.ptp(t_raw))
    t_a = t_raw * a_target / (raw_ptp + EPS)
    return t_a, {
        "n_beats": int(beats.shape[0]),
        "length": int(t_a.size),
        "raw_ptp": raw_ptp,
        "a_target_median_ptp": a_target,
        "final_ptp": float(np.ptp(t_a)),
        "eps": EPS,
    }


def crop_qrs(template_a: np.ndarray, fs: int = FS) -> np.ndarray:
    """T-B: the frozen [-80, +120] ms portion of the final Template A. 26 samples, R at index 10."""
    g = template_geometry(fs)
    lo, hi = g["qrs_slice"]
    if template_a.size != g["full_len"]:
        raise ValueError(f"expected a length-{g['full_len']} Template A, got {template_a.size}")
    return np.asarray(template_a, dtype=np.float64)[lo:hi].copy()


def analytic_template_c(peak_to_peak: float, fs: int = FS) -> np.ndarray:
    """T-C: Ricker (Mexican-hat) wavelet, sigma = 10 ms, t in [-80, +120] ms, scaled to `peak_to_peak`.

    Data-independent apart from the amplitude anchor. Reproducible from Amendment 1 section 1A alone.
    """
    g = template_geometry(fs)
    n = np.arange(-g["qrs_n_before"], g["qrs_n_after"] + 1, dtype=np.float64)
    t = n / fs
    sigma = 0.010
    w = (1.0 - (t / sigma) ** 2) * np.exp(-(t ** 2) / (2.0 * sigma ** 2))
    return w * (float(peak_to_peak) / (float(np.ptp(w)) + EPS))


def stamp(template: np.ndarray, peaks, n_time: int, r_index: int, baseline=None) -> np.ndarray:
    """Additively place `template` so its index `r_index` sits on each peak. Superposition where they overlap.

    `baseline=None` means a zero baseline. Out-of-range portions are clipped, never wrapped or shifted.
    """
    out = np.zeros(int(n_time), dtype=np.float64) if baseline is None else np.asarray(baseline, dtype=np.float64).copy()
    tmpl = np.asarray(template, dtype=np.float64)
    for p in np.asarray(peaks, dtype=int):
        a, b = int(p) - int(r_index), int(p) - int(r_index) + tmpl.size
        ta, tb = max(0, -a), tmpl.size - max(0, b - out.size)
        a, b = max(0, a), min(out.size, b)
        if b > a and tb > ta:
            out[a:b] += tmpl[ta:tb]
    return out


def stamp_supports_overlap(peaks, r_index: int, tmpl_len: int) -> bool:
    """True iff any two consecutive stamps would share a sample index."""
    p = np.sort(np.asarray(peaks, dtype=int))
    if p.size < 2:
        return False
    last = p[:-1] - r_index + tmpl_len - 1          # inclusive end of stamp k
    first = p[1:] - r_index                          # inclusive start of stamp k+1
    return bool(np.any(first <= last))


def lowfreq_baseline(sig: np.ndarray, fs: int = FS, cut_hz: float = 1.0) -> np.ndarray:
    """T-D realism arm: the < `cut_hz` component of a signal, via a real FFT brick-wall.

    Carries only low-frequency drift; at 1 Hz over an 8 s window it retains no beat shape.
    """
    x = np.asarray(sig, dtype=np.float64)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    spec[freqs >= cut_hz] = 0.0
    return np.fft.irfft(spec, n=x.size)


def sha256_array(a: np.ndarray) -> str:
    """Stable content hash of an array's float64 bytes, for artifact provenance."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()
