"""Four-way taxonomy of rendered R detections, detection-level counterfactuals and localization (E0 prereg §3-§5).

Every rendered detection is classified on two axes with the project's one-to-one matching (`rpeaks.match_rpeaks`: greedy
by |dt|, +-50 ms):

    placed?  reference?   type
    yes      yes          A  supported true event
    yes      no           B  inherited false event
    no       yes          C  spontaneous recovery
    no       no           D  spontaneous hallucination

"placed? yes" means the detection is assigned to a placed event in the PLACED <-> RENDERED matching; "reference? yes"
means it is assigned to a reference beat in the REFERENCE <-> RENDERED matching (exactly the TP set of the C0 metric).
"""
from __future__ import annotations

import numpy as np

from ppg2ecg.evaluation import rpeaks as RP

FS = 128
TOL_MS = 50.0
TYPES = ("A", "B", "C", "D")
PHASE_EDGES = (0.0, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 1.0)
ABS_EDGES_MS = (0.0, 100.0, 200.0, 350.0, 500.0, np.inf)          # |t - nearest placed R|
EDGE_EDGES_MS = (0.0, 125.0, 250.0, 500.0, np.inf)                # distance to the nearest window edge
REGIONS = ("no_placed", "before_first", "between", "after_last")


def _assigned(a, b, fs=FS, tol_ms=TOL_MS):
    """Map index in b -> index in a for the one-to-one matching of a (first) and b (second)."""
    m, _, _ = RP.match_rpeaks(np.asarray(a, int), np.asarray(b, int), fs, tol_ms)
    return {j: i for i, j in m}


def classify(ref, placed, rendered, fs: int = FS, tol_ms: float = TOL_MS) -> dict:
    """Types and assignment identities of every rendered detection of one window.

    Returns {"type": [n_rendered] str, "ref_idx": [n_rendered] (-1 if none), "placed_idx": [n_rendered] (-1 if none),
             "ref_placed": {ref index: placed index} (REFERENCE <-> PLACED matching)}."""
    rendered = np.asarray(rendered, int)
    to_ref = _assigned(ref, rendered, fs, tol_ms)
    to_placed = _assigned(placed, rendered, fs, tol_ms)
    rp = {i: j for i, j in RP.match_rpeaks(np.asarray(ref, int), np.asarray(placed, int), fs, tol_ms)[0]}
    types, ri, pi = [], [], []
    for j in range(rendered.size):
        r, p = to_ref.get(j, -1), to_placed.get(j, -1)
        types.append("A" if p >= 0 and r >= 0 else "B" if p >= 0 else "C" if r >= 0 else "D")
        ri.append(r)
        pi.append(p)
    return {"type": np.asarray(types, dtype="<U1"), "ref_idx": np.asarray(ri, int), "placed_idx": np.asarray(pi, int),
            "ref_placed": rp}


def counts(cls: dict) -> dict:
    t = cls["type"]
    return {k: int(np.sum(t == k)) for k in TYPES}


def accounting(cls: dict, n_ref: int, n_placed: int) -> dict:
    """Event-count identities of one window (prereg §4.3):
        TP_rendered - TP_placed = C - L,   L = TP_placed - A   (placed true events not carried as type A)
        FP_rendered - FP_placed = D - M,   M = FP_placed - B   (placed false events not carried as type B)"""
    c = counts(cls)
    tp_p = len(cls["ref_placed"])
    fp_p = n_placed - tp_p
    return {**c, "tp_placed": tp_p, "fp_placed": fp_p, "fn_placed": n_ref - tp_p, "L": tp_p - c["A"], "M": fp_p - c["B"],
            "tp_rendered": c["A"] + c["C"], "fp_rendered": c["B"] + c["D"], "fn_rendered": n_ref - c["A"] - c["C"]}


def hard_guard(rendered, placed, fs: int = FS, tol_ms: float = TOL_MS) -> np.ndarray:
    """Diagnostic counterfactual (not a method): keep a rendered detection iff it lies within tol of ANY placed event."""
    rendered = np.asarray(rendered, int)
    placed = np.asarray(placed, int)
    if rendered.size == 0 or placed.size == 0:
        return np.zeros(0, int)
    tol = tol_ms / 1000.0 * fs
    keep = np.min(np.abs(rendered[:, None] - placed[None, :]), axis=1) <= tol
    return rendered[keep]


def remove_types(rendered, types, drop) -> np.ndarray:
    """ORACLE counterfactuals: drop the rendered detections whose type is in `drop` (impossible-reference bounds)."""
    rendered = np.asarray(rendered, int)
    keep = ~np.isin(np.asarray(types), list(drop))
    return rendered[keep]


def localize(t: int, placed, n_time: int = 512, fs: int = FS) -> dict:
    """Position of a detection relative to the placed events and the window edges (prereg §5)."""
    placed = np.sort(np.asarray(placed, int))
    ms = 1000.0 / fs
    out = {"dist_edge_ms": float(min(t, n_time - 1 - t) * ms)}
    if placed.size == 0:
        return out | {"region": "no_placed", "dist_nearest_ms": np.nan, "signed_nearest_ms": np.nan, "dist_prev_ms": np.nan,
                      "dist_next_ms": np.nan, "phase": np.nan}
    prev = placed[placed <= t]
    nxt = placed[placed > t]
    k = int(np.argmin(np.abs(placed - t)))
    d_prev = float((t - prev[-1]) * ms) if prev.size else np.nan
    d_next = float((nxt[0] - t) * ms) if nxt.size else np.nan
    if prev.size and nxt.size:
        region, phase = "between", float((t - prev[-1]) / (nxt[0] - prev[-1]))
    else:
        region, phase = ("before_first" if not prev.size else "after_last"), np.nan
    return out | {"region": region, "dist_nearest_ms": float(abs(t - placed[k]) * ms), "signed_nearest_ms": float((t - placed[k]) * ms),
                  "dist_prev_ms": d_prev, "dist_next_ms": d_next, "phase": phase}


def bin_index(v, edges) -> int:
    """Index of the half-open bin [e_k, e_{k+1}) containing v (the last bin is closed); -1 for nan / out of range."""
    if v is None or not np.isfinite(v):
        return -1
    for k in range(len(edges) - 1):
        last = k == len(edges) - 2
        if edges[k] <= v < edges[k + 1] or (last and v == edges[k + 1]):
            return k
    return -1
