"""O2b — integer-grid oracle canonicalization operator (operator repair audit only).

Frozen by docs/O2B_INTEGER_GRID_WARP_AUDIT.md and
docs/O2B_INTEGER_GRID_CANONICALIZATION_PREREGISTRATION.md.

The ONLY mathematical change with respect to `o2_warp` is that the canonical event positions are projected to
integers. Boundaries, the monotone piecewise-linear map and its inverse, the W = 10 QRS protection, the bilinear
grid_sample resampler, the absence of post-warp normalisation and of any amplitude Jacobian are all reused
unchanged from O2. No generator is trained anywhere in O2b, and the GT ECG R schedule still builds the operator,
so the object is an ORACLE operator and is not deployable.
"""
from __future__ import annotations

import numpy as np

from ppg2ecg.evaluation import o2_warp as O2

FS, T_LEN = O2.FS, O2.T_LEN
ANCHOR_W = O2.ANCHOR_W                 # 10 samples, unchanged
MIN_BEATS = O2.MIN_BEATS               # K < 3 -> identity, unchanged
EPS = O2.EPS
MIN_INT_SPACING = 2 * ANCHOR_W + 1     # 21 samples
CORE_OFFSET_TOL = 1e-6                 # max |coordinate - round(coordinate)| inside a protected core


def round_half_to_even(x) -> np.ndarray:
    """Explicit deterministic banker's rounding to an integer array (ties go to the even integer)."""
    a = np.asarray(x, dtype=np.float64)
    fl = np.floor(a)
    diff = a - fl
    up = diff > 0.5
    tie = diff == 0.5
    out = np.where(up, fl + 1.0, fl)
    out = np.where(tie, np.where(np.mod(fl, 2.0) == 0.0, fl, fl + 1.0), out)
    return out.astype(np.int64)


def canonical_positions_int(r: np.ndarray) -> np.ndarray | None:
    """Integer canonical schedule: endpoints kept at r_1 / r_K, interior beats rounded half-to-even."""
    q = O2.canonical_positions(r)
    if q is None:
        return None
    r = np.asarray(r, dtype=np.float64)
    qi = round_half_to_even(q)
    qi[0] = int(round(float(r[0])))
    qi[-1] = int(round(float(r[-1])))
    return qi


def spacing_ok(q_int: np.ndarray, n_time: int = T_LEN) -> bool:
    q = np.asarray(q_int, dtype=np.int64)
    return bool(np.all(np.diff(q) >= MIN_INT_SPACING) and q[0] >= 0 and q[-1] <= n_time - 1)


def build_int_anchors(r: np.ndarray, n_time: int = T_LEN, w: int = ANCHOR_W):
    """O2's anchor construction with the integer canonical schedule."""
    q = canonical_positions_int(r)
    if q is None:
        return None, "K<3"
    if not spacing_ok(q, n_time):
        return None, "integer spacing violated"
    r = np.asarray(r, dtype=np.float64)
    cand: list[tuple[float, float, bool]] = [(0.0, 0.0, False)]
    for rk, qk in zip(r, q.astype(np.float64)):
        for off in (-w, 0, w):
            s, d = float(rk + off), float(qk + off)
            if 0.0 < s < n_time - 1 and 0.0 < d < n_time - 1:
                cand.append((s, d, off == 0))
    cand.append((float(n_time - 1), float(n_time - 1), False))
    cand.sort(key=lambda a: (a[0], a[1]))
    src, dst, kept = [cand[0][0]], [cand[0][1]], 0
    for s, d, is_centre in cand[1:]:
        if s > src[-1] + EPS and d > dst[-1] + EPS:
            src.append(s); dst.append(d); kept += int(is_centre)
        elif is_centre:
            return None, "centre anchor not monotone"
    n_centres = int(sum(1 for rk, qk in zip(r, q) if 0.0 < rk < n_time - 1 and 0.0 < qk < n_time - 1))
    if kept != n_centres:
        return None, "centre anchor dropped"
    if len(src) < 2 or src[-1] != float(n_time - 1) or dst[-1] != float(n_time - 1):
        return None, "boundary anchor lost"
    return (np.asarray(src), np.asarray(dst)), "ok"


class IntegerEventWarp(O2.EventWarp):
    """O2's EventWarp with an integer canonical schedule; every other behaviour is inherited unchanged."""

    def __init__(self, r: np.ndarray, n_time: int = T_LEN, w: int = ANCHOR_W):
        self.n_time = int(n_time)
        anchors, status = build_int_anchors(r, n_time, w)
        self.status = status
        self.identity = anchors is None
        self.src, self.dst = (None, None) if self.identity else anchors
        self.r = np.asarray(r, dtype=np.float64)
        self.q_real = O2.canonical_positions(r)
        self.q = canonical_positions_int(r)

    def core_offsets(self) -> np.ndarray:
        """|coordinate - nearest integer| of every inverse sampling coordinate inside a protected core."""
        if self.identity:
            return np.zeros(1)
        out = []
        for qk in np.asarray(self.q, dtype=np.int64):
            tau = np.arange(max(0, qk - ANCHOR_W), min(self.n_time, qk + ANCHOR_W + 1), dtype=np.float64)
            t = self.inverse(tau)
            out.append(np.abs(t - np.round(t)))
        return np.concatenate(out) if out else np.zeros(1)

    def integer_shift(self) -> np.ndarray:
        """q_int - r per beat (integer by construction)."""
        return np.asarray(self.q, dtype=np.float64) - self.r


# ------------------------------------------------------------------------------------------------ hard-copy diagnostic
def hard_copy_cores(x_can, x_raw, warps, w: int = ANCHOR_W):
    """DIAGNOSTIC ONLY (never the primary operator): copy protected-core samples instead of resampling them."""
    out = x_can.clone()
    for i, wp in enumerate(warps):
        if wp.identity:
            continue
        for rk, qk in zip(np.asarray(wp.r, dtype=np.int64), np.asarray(wp.q, dtype=np.int64)):
            lo_q, hi_q = max(0, qk - w), min(wp.n_time, qk + w + 1)
            lo_r, hi_r = rk - (qk - lo_q), rk + (hi_q - qk)
            if lo_r < 0 or hi_r > wp.n_time:
                continue
            out[i, :, lo_q:hi_q] = x_raw[i, :, lo_r:hi_r]
    return out


# ------------------------------------------------------------------------------------------------ verdicts
VERDICT_INVALID = "INTEGER GRID SCHEDULE INVALID"
VERDICT_A = "INTEGER-GRID CANONICALIZATION OPERATOR ACCEPTED"
VERDICT_B = "INTEGER GRID FIXES SHARPNESS BUT WIDTH REMAINS INVALID"
VERDICT_C = "INTEGER-GRID REPAIR INSUFFICIENT"


def decide_o2b(checks: dict, precheck_ok: bool = True) -> dict:
    """Preregistration section 5. `checks` maps the R0 ids to booleans (True = the check passed)."""
    if not precheck_ok:
        return {"verdict": VERDICT_INVALID, "checks": {}, "precheck_ok": False}
    core = ("R0-1", "R0-2", "R0-3", "R0-4a", "R0-5", "R0-6")
    if all(checks[k] for k in core) and checks["R0-4b"]:
        v = VERDICT_A
    elif all(checks[k] for k in core) and not checks["R0-4b"]:
        v = VERDICT_B
    else:
        v = VERDICT_C
    return {"verdict": v, "checks": {k: bool(checks[k]) for k in ("R0-1", "R0-2", "R0-3", "R0-4a", "R0-4b", "R0-5", "R0-6")},
            "precheck_ok": True}
