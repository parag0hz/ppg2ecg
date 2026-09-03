"""O2 — oracle event-canonical temporal warp (QRS-preserving piecewise-linear event warp).

Frozen by docs/O2_CANONICAL_WARP_AUDIT.md and docs/O2_ORACLE_EVENT_CANONICALIZATION_PREREGISTRATION.md.

ORACLE / TARGET-LEAKAGE DIAGNOSTIC: the GT ECG R schedule is used to build a temporal coordinate at training
AND at inference. Nothing here is deployable. The GT R indices are used ONLY to construct the coordinate map and
its inverse; no R field, no phase channel and no event scalar ever becomes a model input, and no GT ECG sample
value enters the model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

FS, T_LEN = 128, 1024
ANCHOR_W = 10                     # samples, 78.125 ms: the M1/O1 QRS core half-width
MIN_BEATS = 3                     # K < MIN_BEATS -> identity warp
EPS = 1e-3                        # minimum strict increase (samples) required of both anchor coordinates
FALLBACK_BUDGET = 0.005           # > 0.5 % non-(K<3) fallbacks -> STOP (preregistration section 7)


def canonical_positions(r: np.ndarray) -> np.ndarray | None:
    """Equalised interior beat schedule: q_1 = r_1, q_K = r_K, interior R peaks uniformly spaced."""
    r = np.asarray(r, dtype=np.float64)
    K = r.size
    if K < MIN_BEATS:
        return None
    return r[0] + (np.arange(K, dtype=np.float64) / (K - 1)) * (r[-1] - r[0])


def build_anchors(r: np.ndarray, n_time: int = T_LEN, w: int = ANCHOR_W):
    """(src, dst) anchor pairs of the QRS-preserving warp, or None when the window must use the identity.

    Anchors: the two boundaries, and for every GT beat the triple (r-w, r, r+w) -> (q-w, q, q+w), which pins the
    local slope to 1 across the QRS core. Anchors are kept in order only while BOTH coordinates strictly
    increase; if any beat's CENTRE anchor cannot be kept, the window is invalid and falls back to the identity.
    """
    q = canonical_positions(r)
    if q is None:
        return None, "K<3"
    r = np.asarray(r, dtype=np.float64)
    cand: list[tuple[float, float, bool]] = [(0.0, 0.0, False)]
    for rk, qk in zip(r, q):
        for off in (-w, 0, w):
            s, d = float(rk + off), float(qk + off)
            if 0.0 < s < n_time - 1 and 0.0 < d < n_time - 1:
                cand.append((s, d, off == 0))
    cand.append((float(n_time - 1), float(n_time - 1), False))
    cand.sort(key=lambda a: (a[0], a[1]))
    src, dst, kept_centres = [cand[0][0]], [cand[0][1]], 0
    for s, d, is_centre in cand[1:]:
        if s > src[-1] + EPS and d > dst[-1] + EPS:
            src.append(s); dst.append(d)
            kept_centres += int(is_centre)
        elif is_centre:
            return None, "centre anchor not monotone"
    n_centres = int(sum(1 for rk, qk in zip(r, q) if 0.0 < rk < n_time - 1 and 0.0 < qk < n_time - 1))
    if kept_centres != n_centres:
        return None, "centre anchor dropped"
    if len(src) < 2 or src[-1] != float(n_time - 1) or dst[-1] != float(n_time - 1):
        return None, "boundary anchor lost"
    return (np.asarray(src), np.asarray(dst)), "ok"


class EventWarp:
    """Monotone piecewise-linear map tau = f(t) with its inverse, built from one window's GT R schedule."""

    def __init__(self, r: np.ndarray, n_time: int = T_LEN, w: int = ANCHOR_W):
        self.n_time = int(n_time)
        anchors, status = build_anchors(r, n_time, w)
        self.status = status
        self.identity = anchors is None
        if self.identity:
            self.src = self.dst = None
        else:
            self.src, self.dst = anchors
        self.r = np.asarray(r, dtype=np.float64)
        self.q = canonical_positions(r)

    def forward(self, t):
        """t (raw coordinate) -> tau (canonical coordinate)."""
        t = np.asarray(t, dtype=np.float64)
        return t.copy() if self.identity else np.interp(t, self.src, self.dst)

    def inverse(self, tau):
        """tau (canonical coordinate) -> t (raw coordinate)."""
        tau = np.asarray(tau, dtype=np.float64)
        return tau.copy() if self.identity else np.interp(tau, self.dst, self.src)

    def slopes(self) -> np.ndarray:
        if self.identity:
            return np.ones(1)
        return np.diff(self.dst) / np.diff(self.src)

    def core_slopes(self) -> np.ndarray:
        """Slope of the segments that lie inside a +-w QRS core (should be exactly 1)."""
        if self.identity:
            return np.ones(1)
        mids = 0.5 * (self.src[:-1] + self.src[1:])
        sl = self.slopes()
        keep = np.zeros(len(sl), dtype=bool)
        for rk in self.r:
            keep |= (mids > rk - ANCHOR_W) & (mids < rk + ANCHOR_W)
        return sl[keep] if keep.any() else np.ones(1)

    def valid(self) -> bool:
        if self.identity:
            return True
        return bool(np.all(np.diff(self.src) > 0) and np.all(np.diff(self.dst) > 0)
                    and np.all(np.isfinite(self.src)) and np.all(np.isfinite(self.dst))
                    and np.all(self.slopes() > 0))


# ------------------------------------------------------------------------------------------------ resampling
def resample_at(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x [B,C,L] sampled at fractional positions pos [B,L] with bilinear grid_sample(align_corners=True).

    No amplitude Jacobian scaling: the signal is treated as a value, not a density.
    """
    if x.dim() != 3:
        raise ValueError(f"expected [B,C,L], got {tuple(x.shape)}")
    B, C, L = x.shape
    gx = (2.0 * pos.to(x.dtype) / (L - 1)) - 1.0                      # [B,L] -> [-1,1]
    grid = torch.stack([gx, torch.zeros_like(gx)], dim=-1).unsqueeze(1)   # [B,1,L,2]
    out = F.grid_sample(x.unsqueeze(2), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(2)


def warp_positions(warps, n_time: int = T_LEN, direction: str = "to_canonical") -> torch.Tensor:
    """Sampling positions for a batch of EventWarps.

    to_canonical:  x_can(tau) = x(f^-1(tau))  -> sample the raw signal at f^-1(tau)
    to_raw:        x(t)       = x_can(f(t))   -> sample the canonical signal at f(t)
    """
    grid = np.arange(n_time, dtype=np.float64)
    if direction == "to_canonical":
        pos = np.stack([w.inverse(grid) for w in warps])
    elif direction == "to_raw":
        pos = np.stack([w.forward(grid) for w in warps])
    else:
        raise ValueError(direction)
    return torch.from_numpy(pos.astype(np.float32))


def apply_warp(x: torch.Tensor, warps, direction: str = "to_canonical") -> torch.Tensor:
    """Warp a batch. Rows whose warp is the identity are returned BIT-EXACTLY (no interpolation)."""
    if x.dim() != 3:
        raise ValueError(f"expected [B,C,L], got {tuple(x.shape)}")
    ident = torch.tensor([w.identity for w in warps], device=x.device)
    pos = warp_positions(warps, x.shape[-1], direction).to(x.device)
    out = resample_at(x, pos)
    if ident.any():
        out = torch.where(ident.view(-1, 1, 1), x, out)
    return out


def round_trip(x: torch.Tensor, warps) -> torch.Tensor:
    return apply_warp(apply_warp(x, warps, "to_canonical"), warps, "to_raw")


# ------------------------------------------------------------------------------------------------ diagnostics
def center_only_anchors(r: np.ndarray, n_time: int = T_LEN):
    """Diagnostic-only warp WITHOUT the +-w morphology-preserving anchors (never trained)."""
    q = canonical_positions(r)
    if q is None:
        return None, "K<3"
    cand = [(0.0, 0.0)] + [(float(a), float(b)) for a, b in zip(r, q) if 0.0 < a < n_time - 1 and 0.0 < b < n_time - 1]
    cand.append((float(n_time - 1), float(n_time - 1)))
    cand.sort()
    src, dst = [cand[0][0]], [cand[0][1]]
    for s, d in cand[1:]:
        if s > src[-1] + EPS and d > dst[-1] + EPS:
            src.append(s); dst.append(d)
    if len(src) < 2 or src[-1] != float(n_time - 1):
        return None, "boundary anchor lost"
    return (np.asarray(src), np.asarray(dst)), "ok"


class CenterOnlyWarp(EventWarp):
    def __init__(self, r: np.ndarray, n_time: int = T_LEN):
        self.n_time = int(n_time)
        anchors, status = center_only_anchors(r, n_time)
        self.status = status
        self.identity = anchors is None
        self.src, self.dst = (None, None) if self.identity else anchors
        self.r = np.asarray(r, dtype=np.float64)
        self.q = canonical_positions(r)


# ------------------------------------------------------------------------------------------------ gate
ROUNDTRIP_GATE = {"raw_rmse": 0.020, "T6": 0.020, "T7": 0.020, "T4": 0.020, "T8": 0.020,
                  "f1_at_50": 0.98, "beat_count_diff": 0}
VERDICT_REJECT = "CANONICALIZATION OPERATOR REJECTED"
VERDICT_A = "ORACLE EVENT-CANONICALIZATION JOINTLY SUPPORTED"
VERDICT_B = "EVENT ANCHOR HELPS BUT MORPHOLOGY REMAINS UNRESOLVED"
VERDICT_C = "MORPHOLOGY IMPROVES WITHOUT MATERIAL EVENT BENEFIT"
VERDICT_D = "NO MATERIAL ORACLE CANONICALIZATION BENEFIT"
NONINF_MARGIN = 0.020
F1_EXCESS_MIN = 0.10


def roundtrip_gate(med: dict) -> dict:
    """Preregistration section 12. `med` holds the medians of the round-trip metrics."""
    checks = {
        "R0-1 raw_rmse <= 0.020": float(med["raw_rmse"]) <= ROUNDTRIP_GATE["raw_rmse"],
        "R0-2 T6 nAE <= 0.020": float(med["T6"]) <= ROUNDTRIP_GATE["T6"],
        "R0-3 T7 nAE <= 0.020": float(med["T7"]) <= ROUNDTRIP_GATE["T7"],
        "R0-4a T4 nAE <= 0.020": float(med["T4"]) <= ROUNDTRIP_GATE["T4"],
        "R0-4b T8 nAE <= 0.020": float(med["T8"]) <= ROUNDTRIP_GATE["T8"],
        "R0-5 F1@50 >= 0.98": float(med["f1_at_50"]) >= ROUNDTRIP_GATE["f1_at_50"],
        "R0-6 median beat-count difference == 0": float(med["beat_count_diff"]) == 0.0,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def decide_o2(j: dict) -> dict:
    """Preregistration sections 31-32. `j` maps J1..J7 to booleans; morphology_improves is used for verdict C."""
    if all(j[k] for k in ("J1", "J2", "J3", "J4", "J5", "J6", "J7")):
        v = VERDICT_A
    elif j["J1"]:
        v = VERDICT_B
    elif j.get("morphology_improves"):
        v = VERDICT_C
    else:
        v = VERDICT_D
    return {"verdict": v, "gates": {k: bool(j.get(k)) for k in ("J1", "J2", "J3", "J4", "J5", "J6", "J7")},
            "morphology_improves": bool(j.get("morphology_improves"))}
