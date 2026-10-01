"""D0 event-cell renderer (docs/D0_EVENT_CELL_SUPPORT_PREREGISTRATION.md).

Every frozen BF0 beat output is rendered only inside its own event cell, in absolute time:
  - interior cell limits are the midpoints of adjacent placed events; the first / last cell extends half the adjacent
    interval outward; a single event uses half the TRAIN median RR on each side;
  - neighbouring cells meet in a complementary raised-cosine crossfade (weights sum to 1); a crossfade boundary sits at
    the midpoint unless the crossfade would leave either beat's own output support (R - 64 ... R + 101 samples), in which
    case it moves the least amount needed to lie inside both;
  - weights already sum to 1, so nothing is renormalized; samples no beat owns take BF0's nearest-covered-value fill.
No time warping: a beat sample keeps its absolute offset from its R.
"""
from __future__ import annotations

import numpy as np

from .beats import N_AFTER, N_BEFORE, R_INDEX, SEG_LEN

TOL = 1e-9


def nominal_cells(positions, rr_single: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell limits [L_i, U_i] in absolute samples, before any clipping."""
    P = np.asarray(positions, dtype=np.float64).reshape(-1)
    n = P.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    if n == 1:
        return np.array([P[0] - 0.5 * rr_single]), np.array([P[0] + 0.5 * rr_single])
    mids = 0.5 * (P[:-1] + P[1:])
    L = np.concatenate([[P[0] - 0.5 * (P[1] - P[0])], mids])
    U = np.concatenate([mids, [P[-1] + 0.5 * (P[-1] - P[-2])]])
    return L, U


def crossfade_up(t, centre: float, width: float) -> np.ndarray:
    """Weight of the later beat: 0 up to centre - width/2, raised cosine, 1 from centre + width/2 (a step if width <= 0)."""
    t = np.asarray(t, dtype=np.float64)
    if width <= 0:
        return (t >= centre).astype(np.float64)
    u = np.clip((t - (centre - width / 2.0)) / width, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def cell_weights(positions, n_time: int, width: int, rr_single: float) -> tuple[np.ndarray, dict]:
    """Per-beat weights [N, n_time] and a record of the boundaries (crossfade centres / widths, gaps, shifts, limits)."""
    P = np.asarray(positions, dtype=int).reshape(-1)
    n = P.size
    t = np.arange(n_time, dtype=np.float64)
    rec = {"centres": [], "widths": [], "midpoints": [], "shifted": 0, "gaps": 0, "left_limit": None, "right_limit": None}
    if n == 0:
        return np.zeros((0, n_time)), rec
    if np.any(np.diff(P) < 0):
        raise ValueError("positions must be sorted")
    L, U = nominal_cells(P, rr_single)
    lo_sup = np.maximum(P - N_BEFORE, 0).astype(np.float64)
    hi_sup = np.minimum(P + N_AFTER, n_time - 1).astype(np.float64)
    left_f, right_f = np.ones((n, n_time)), np.ones((n, n_time))
    rec["left_limit"] = float(max(L[0], lo_sup[0], 0.0))
    rec["right_limit"] = float(min(U[-1], hi_sup[-1], n_time - 1.0))
    left_f[0] = t >= rec["left_limit"] - TOL
    right_f[-1] = t <= rec["right_limit"] + TOL
    for k in range(n - 1):
        a, b = lo_sup[k + 1], hi_sup[k]                     # overlap of the two output supports
        m = U[k]
        w = float(max(min(width, P[k + 1] - P[k] - 2), 0))  # never reaches either R sample
        if b - a >= w:
            c = float(np.clip(m, a + w / 2.0, b - w / 2.0))
        elif b >= a:                                        # supports overlap by less than the crossfade width
            c, w = 0.5 * (a + b), float(b - a)
        else:                                               # no overlap: a true gap, filled like BF0
            right_f[k] = t <= b + TOL
            left_f[k + 1] = t >= a - TOL
            rec["gaps"] += 1
            continue
        up = crossfade_up(t, c, w)
        right_f[k], left_f[k + 1] = 1.0 - up, up
        rec["centres"].append(c)
        rec["widths"].append(w)
        rec["midpoints"].append(float(m))
        rec["shifted"] += int(abs(c - m) > TOL)
    W = left_f * right_f
    for i in range(n):                                      # never outside the beat's own output support
        W[i, (t < lo_sup[i] - TOL) | (t > hi_sup[i] + TOL)] = 0.0
    return W, rec


def nearest_fill(y: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """BF0's rule (`render.assemble`): an uncovered sample takes the nearest covered value, ties to the left."""
    y = y.copy()
    cov_idx = np.flatnonzero(covered)
    unc = np.flatnonzero(~covered)
    if unc.size and cov_idx.size:
        pos = np.searchsorted(cov_idx, unc)
        left = cov_idx[np.clip(pos - 1, 0, cov_idx.size - 1)]
        right = cov_idx[np.clip(pos, 0, cov_idx.size - 1)]
        y[unc] = y[np.where(np.abs(unc - left) <= np.abs(right - unc), left, right)]
    return y


def render(beats, positions, n_time: int, width: int, rr_single: float, empty_fill: float) -> tuple[np.ndarray, dict]:
    """y(t) = sum_i w_i(t) b_i(t - r_i + 64) with event-cell weights; uncovered samples take the nearest covered value."""
    P = np.asarray(positions, dtype=int).reshape(-1)
    if P.size == 0:
        return np.full(int(n_time), float(empty_fill)), {"empty": True, "flat_fill_fraction": 1.0, "pou_error": 0.0,
                                                          "centres": [], "widths": [], "midpoints": [], "shifted": 0,
                                                          "gaps": 0, "left_limit": None, "right_limit": None}
    W, rec = cell_weights(P, n_time, width, rr_single)
    num = np.zeros(int(n_time))
    for i, (b, p) in enumerate(zip(beats, P)):
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if b.size != SEG_LEN:
            raise ValueError(f"beat length {b.size} != {SEG_LEN}")
        idx = np.arange(SEG_LEN) + int(p) - R_INDEX
        ok = (idx >= 0) & (idx < n_time)
        num[idx[ok]] += W[i, idx[ok]] * b[ok]
    s = W.sum(axis=0)
    covered = s > TOL
    pou = float(np.max(np.abs(s[covered] - 1.0))) if covered.any() else 0.0
    y = nearest_fill(np.where(covered, num, 0.0), covered)
    return y, rec | {"empty": False, "flat_fill_fraction": float(1.0 - covered.mean()), "pou_error": pou}


def edge_events(positions, n_time: int, rr_single: float) -> np.ndarray:
    """EDGE = first or last placed event, or a nominal cell reaching past the waveform boundary; others INTERIOR."""
    L, U = nominal_cells(positions, rr_single)
    e = np.zeros(L.size, dtype=bool)
    if L.size:
        e[0] = e[-1] = True
        e |= (L < 0) | (U > n_time - 1)
    return e


ZONES = ("A_within_50ms", "B_after_50_150ms", "C_after_150_450ms", "D_before_300_50ms", "E_far")
FAR_TYPES = ("before_first", "after_last", "gap_missed_event", "interior")


def fp_zone(d: int, placed, fs: int = 128) -> tuple[str, str | None, float]:
    """Post-hoc zones frozen from the pre-D0 attribution: offset of a detection from the nearest placed R (ms), and for
    far detections whether they lie before the first / after the last placed event, inside a gap wider than 1.5 x the
    window's median placed RR, or between normally spaced placed events (INTERIOR)."""
    P = np.asarray(placed, dtype=int).reshape(-1)
    if P.size == 0:
        return "E_far", "no_placed_event", float("nan")
    near = P[np.argmin(np.abs(P - d))]
    dm = (int(d) - int(near)) / fs * 1000.0
    if abs(dm) <= 50:
        return "A_within_50ms", None, dm
    if 50 < dm <= 150:
        return "B_after_50_150ms", None, dm
    if 150 < dm <= 450:
        return "C_after_150_450ms", None, dm
    if -300 <= dm < -50:
        return "D_before_300_50ms", None, dm
    prev, nxt = P[P < d], P[P > d]
    if prev.size == 0:
        return "E_far", "before_first", dm
    if nxt.size == 0:
        return "E_far", "after_last", dm
    med = float(np.median(np.diff(P)))
    return "E_far", ("gap_missed_event" if (nxt[0] - prev[-1]) > 1.5 * med else "interior"), dm


def boundary_near(d: int, centres, limits, width: float) -> str | None:
    """'crossfade' if within one frozen crossfade width of a crossfade centre, 'edge_limit' if within one width of the
    owned-region limit at the waveform edge (where BF0's flat fill starts), else None."""
    c = np.asarray(centres, dtype=np.float64)
    if c.size and np.min(np.abs(c - d)) <= width:
        return "crossfade"
    lim = np.asarray([x for x in limits if x is not None], dtype=np.float64)
    if lim.size and np.min(np.abs(lim - d)) <= width:
        return "edge_limit"
    return None
