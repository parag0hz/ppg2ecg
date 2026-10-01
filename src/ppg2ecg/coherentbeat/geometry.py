"""C0 fixed geometry (prereg §4): cubic B-spline global basis, compact C-infinity bump envelope, local supports."""
from __future__ import annotations

import numpy as np
import torch

FS = 128
N_TIME = 512
N_COEF, KNOT_SPACING = 17, 32                 # control points at 0, 32, ..., 512 samples: 250 ms spacing
L_MAX, R_MAX = 0.300 * FS, 0.450 * FS         # 38.4 and 57.6 samples (exactly 300 / 450 ms)
TAU_LO, TAU_HI = -38, 57                      # integer residual grid that contains every possible support
N_TAU = TAU_HI - TAU_LO + 1                   # 96
RR_FRAC = 0.45


def cubic_bspline(x) -> np.ndarray:
    """Centred cardinal cubic B-spline, support (-2, 2)."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    return np.where(a < 1, 2.0 / 3.0 - a ** 2 + 0.5 * a ** 3, np.where(a < 2, (2.0 - a) ** 3 / 6.0, 0.0))


def spline_basis(n_time: int = N_TIME, n_coef: int = N_COEF, spacing: int = KNOT_SPACING) -> np.ndarray:
    """[n_time, n_coef]: basis k is the cubic B-spline centred at sample k * spacing with knot spacing `spacing`."""
    t = np.arange(n_time, dtype=np.float64)[:, None]
    c = (np.arange(n_coef, dtype=np.float64) * spacing)[None, :]
    return cubic_bspline((t - c) / spacing)


def bump(u):
    """b(u) = exp(1 - 1 / (1 - u^2)) for 0 <= u < 1, else 0: b(0) = 1, value and every derivative vanish at u = 1."""
    if isinstance(u, torch.Tensor):
        uu = torch.clamp(u, min=0.0)
        inside = uu < 1.0
        safe = torch.where(inside, uu, torch.zeros_like(uu))
        return torch.where(inside, torch.exp(1.0 - 1.0 / (1.0 - safe ** 2)), torch.zeros_like(uu))
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, None)
    out = np.zeros_like(u)
    m = u < 1.0
    out[m] = np.exp(1.0 - 1.0 / (1.0 - u[m] ** 2))
    return out


def envelope(tau, left, right):
    """m(tau) = b(-tau / left) for tau < 0, b(tau / right) for tau >= 0 (absolute samples; no phase normalization)."""
    if isinstance(tau, torch.Tensor):
        neg = tau < 0
        u = torch.where(neg, -tau / left, tau / right)
        return bump(u)
    tau = np.asarray(tau, dtype=np.float64)
    u = np.where(tau < 0, -tau / np.asarray(left, dtype=np.float64), tau / np.asarray(right, dtype=np.float64))
    return bump(u)


def supports(positions, rr_edge: float) -> tuple[np.ndarray, np.ndarray]:
    """L_i = min(38.4, 0.45 RR_prev), R_i = min(57.6, 0.45 RR_next) in samples; a missing neighbour uses `rr_edge`
    (the ARCH-TRAIN median reference RR in samples)."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1)
    n = p.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    d = np.diff(p)
    prev = np.concatenate([[rr_edge], d])
    nxt = np.concatenate([d, [rr_edge]])
    return np.minimum(L_MAX, RR_FRAC * prev), np.minimum(R_MAX, RR_FRAC * nxt)
