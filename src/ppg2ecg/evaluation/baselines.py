"""D2 baseline predictors (docs/D2_BASELINE_FLOOR_PREREGISTRATION.md §4).

Every fitted quantity comes from TRAIN subjects only; no test window and no test subject's ECG ever enters a fitted
parameter. Each predictor is deterministic and pure-numpy. Two arms deliberately consume test ground truth and are
labelled at the call site as diagnostics: B0 (wrong-window null) and B5 (GT-timing template).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import rpeaks as R

FS = 128
TEMPLATE_HALF_MS = 60.0      # B1 QRS template support, +-60 ms about the R peak (prereg §4)
BEAT_HALF_MS = 400.0         # B2 train-mean beat support, +-0.4 s
PAT_MAX_SAMPLES = 128        # PAT search grid 0..128 samples = 0..1000 ms at 128 Hz
MATCH_TOL_MS = 50.0


def ppg_peaks(x: np.ndarray, fs: int = FS) -> list[np.ndarray]:
    """Elgendi PPG peak positions per window. Frozen detector (prereg §4); failures yield an empty array."""
    import neurokit2 as nk

    out = []
    for w in np.asarray(x, dtype=np.float64):
        try:
            p = nk.ppg_findpeaks(w, sampling_rate=fs, method="elgendi")["PPG_Peaks"]
            out.append(np.asarray(p, dtype=np.int64))
        except Exception:
            out.append(np.zeros(0, dtype=np.int64))
    return out


def gt_rpeaks(y: np.ndarray, fs: int = FS) -> list[np.ndarray]:
    """Reference R peaks via the repo's frozen detector, so B5 and the PAT fit use the same peaks D1 scored against."""
    return [np.asarray(R.detect_rpeaks(w, fs, "neurokit"), dtype=np.int64) for w in np.asarray(y, dtype=np.float64)]


def match_rate(pred: list[np.ndarray], ref: list[np.ndarray], fs: int = FS, tol_ms: float = MATCH_TOL_MS) -> float:
    """Pooled fraction of reference peaks matched within tol_ms under the repo's one-to-one matcher."""
    tp = n = 0
    for p, r in zip(pred, ref):
        if len(r) == 0:
            continue
        m, _fp, _fn = R.match_rpeaks(r, p, fs, tol_ms)
        tp += len(m)
        n += len(r)
    return tp / n if n else float("nan")


def fit_pat(ppg_pk: list[np.ndarray], gt_pk: list[np.ndarray], fs: int = FS) -> tuple[int, float]:
    """One integer PAT offset per corpus: the value in [0, PAT_MAX_SAMPLES] maximising the +-50 ms train match rate.

    Ties break to the SMALLEST offset, so the choice is deterministic and does not depend on grid iteration order.
    """
    best, best_rate = 0, -1.0
    for off in range(PAT_MAX_SAMPLES + 1):
        r = match_rate([p + off for p in ppg_pk], gt_pk, fs)
        if np.isfinite(r) and r > best_rate:
            best, best_rate = off, float(r)
    return best, best_rate


@dataclass(frozen=True)
class TrainFit:
    """Everything D2 fits, all of it from train subjects. Hashable content goes into the run manifest."""
    pat_offset: int
    pat_train_match_rate: float
    qrs_template: np.ndarray     # B1, length 2*h+1, peak-normalised then amplitude-scaled
    mean_beat: np.ndarray        # B2, length 2*H+1
    mean_waveform: np.ndarray    # B3, length T
    n_train_beats: int


def _beat_stack(y: np.ndarray, pk: list[np.ndarray], half: int) -> np.ndarray:
    """[n_beats, 2*half+1] of beats fully inside their window; partial beats at the edges are dropped, not padded."""
    T = y.shape[1]
    out = []
    for w, p in zip(y, pk):
        for c in p:
            if c - half >= 0 and c + half < T:
                out.append(w[c - half : c + half + 1])
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, 2 * half + 1))


def fit_on_train(y_train: np.ndarray, x_train: np.ndarray, fs: int = FS) -> TrainFit:
    """Fit the PAT offset, the B1 QRS template, the B2 mean beat and the B3 mean waveform on TRAIN windows only."""
    gt_pk = gt_rpeaks(y_train, fs)
    pat, rate = fit_pat(ppg_peaks(x_train, fs), gt_pk, fs)

    hq, hb = int(round(TEMPLATE_HALF_MS / 1000 * fs)), int(round(BEAT_HALF_MS / 1000 * fs))
    beats_q, beats_b = _beat_stack(y_train, gt_pk, hq), _beat_stack(y_train, gt_pk, hb)
    qrs = beats_q.mean(axis=0) if len(beats_q) else np.zeros(2 * hq + 1)
    # prereg §4 calls B1 a template that is "zero outside" its +-60 ms support. The raw mean beat sits on the ECG's
    # own baseline (~-0.4 here), so placing it additively would step from 0 to that baseline at every template edge
    # and inject an edge artefact the arm is not meant to have. Subtract the mean of the two end samples so the
    # template starts and ends at 0; the QRS lobe amplitude is untouched.
    qrs = qrs - 0.5 * (qrs[0] + qrs[-1])
    beat = beats_b.mean(axis=0) if len(beats_b) else np.zeros(2 * hb + 1)
    # B1 is a QRS-only template: keep the +-60 ms lobe, zero elsewhere by construction (it has no elsewhere).
    return TrainFit(int(pat), float(rate), qrs, beat, y_train.mean(axis=0), int(len(beats_b)))


def _place(positions: np.ndarray, template: np.ndarray, T: int) -> np.ndarray:
    """Additively place `template` centred on each position; overlapping tails sum, matching a real beat train."""
    half = (len(template) - 1) // 2
    out = np.zeros(T, dtype=np.float64)
    for c in np.asarray(positions, dtype=np.int64):
        lo, hi = c - half, c + half + 1
        a, b = max(lo, 0), min(hi, T)
        if a < b:
            out[a:b] += template[a - lo : b - lo]
    return out


def predict_template(positions: list[np.ndarray], template: np.ndarray, T: int) -> np.ndarray:
    return np.stack([_place(p, template, T) for p in positions])


def b1_b2(x: np.ndarray, fit: TrainFit, fs: int = FS) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """B1 (QRS template) and B2 (train-mean beat) at PPG peaks shifted by the train-fitted PAT offset."""
    pos = [p + fit.pat_offset for p in ppg_peaks(x, fs)]
    T = x.shape[1]
    return predict_template(pos, fit.qrs_template, T), predict_template(pos, fit.mean_beat, T), pos


def b3(n: int, fit: TrainFit) -> np.ndarray:
    return np.tile(fit.mean_waveform, (n, 1))


def b5_gt_timing(y: np.ndarray, fit: TrainFit, fs: int = FS) -> np.ndarray:
    """(GT-R leakage; diagnostic only) the B1 template placed at the TRUE R peaks — an upper bound for templates."""
    return predict_template(gt_rpeaks(y, fs), fit.qrs_template, y.shape[1])


def cyclic_partner(subject: np.ndarray) -> np.ndarray:
    """Row index of the cyclically-next row OF THE SAME SUBJECT (prereg §4), for B0 and B4.

    A subject with a single evaluated row maps to itself; those rows are excluded by the caller and counted, never
    silently scored as if they were a genuine mismatch.
    """
    partner = np.arange(len(subject), dtype=np.int64)
    for s in np.unique(subject):
        idx = np.flatnonzero(subject == s)
        partner[idx] = idx[(np.arange(len(idx)) + 1) % len(idx)]
    return partner
