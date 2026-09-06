"""GSW-iMF structure map (docs/M2_STRUCTURE_WEIGHTED_IMEANFLOW_PREREGISTRATION.md §6).

Training-only. Computed from the normalised target ECG window alone — no R peaks, no learnable part,
no validation-derived statistic. The map never appears on an inference path and carries no gradient.

  d[0] = 0 ; d[n] = x[n] - x[n-1]                                   §6.1  first difference, not central
  s    = conv_same(|d|, [1,4,6,4,1]/16) with reflect padding        §6.2  fixed binomial smoothing
  m    = clip(s / (q95(s) + 1e-8), 0, 1)                            §6.3  per-window robust normalisation
  a    = 1 + LAMBDA_STRUCT * m                                      §6.4  LAMBDA_STRUCT = 2.0, frozen
  structure_weight = a / mean(a)                                    §6.4  mean 1 per window
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LAMBDA_STRUCT: float = 2.0                         # frozen; never changed after results (prereg §6.4)
BINOMIAL_KERNEL: tuple[float, ...] = (1.0, 4.0, 6.0, 4.0, 1.0)   # normalised by 16 at use
Q95: float = 0.95
EPS_Q95: float = 1e-8


def first_difference(x: torch.Tensor) -> torch.Tensor:
    """[..., T] -> [..., T] with d[0] = 0 and d[n] = x[n] - x[n-1]. Not central; no fs rescale."""
    d = torch.zeros_like(x)
    d[..., 1:] = x[..., 1:] - x[..., :-1]
    return d


def smooth_binomial(e: torch.Tensor) -> torch.Tensor:
    """SAME-length convolution with [1,4,6,4,1]/16 under REFLECT padding (prereg §6.2)."""
    shape = e.shape
    v = e.reshape(-1, 1, shape[-1])
    k = torch.tensor(BINOMIAL_KERNEL, dtype=v.dtype, device=v.device).div(16.0).view(1, 1, -1)
    v = F.pad(v, (2, 2), mode="reflect")
    return F.conv1d(v, k).reshape(shape)


def structure_map(x: torch.Tensor) -> torch.Tensor:
    """Normalised structural importance m in [0, 1], per window independently (prereg §6.3)."""
    s = smooth_binomial(first_difference(x).abs())
    q = torch.quantile(s.flatten(start_dim=-1 if s.dim() == 1 else 1).float(), Q95,
                       dim=-1, keepdim=True) if s.dim() > 1 else torch.quantile(s.float(), Q95)
    q = q.to(s.dtype).reshape(*s.shape[:-1], 1) if s.dim() > 1 else q.to(s.dtype)
    return torch.clamp(s / (q + EPS_Q95), 0.0, 1.0)


def structure_weight(x: torch.Tensor, lam: float = LAMBDA_STRUCT) -> torch.Tensor:
    """Mean-1 spatial loss weight for the target window x. Detached: it is data, never a parameter."""
    a = 1.0 + lam * structure_map(x)
    return (a / a.mean(dim=-1, keepdim=True)).detach()


def shifted_structure_weight(x: torch.Tensor, shift: int = 256, lam: float = LAMBDA_STRUCT) -> torch.Tensor:
    """ARM X (prereg §7): S's map circularly shifted by +256 samples.

    Preserves the per-window histogram, mean, max and smoothness exactly while breaking alignment with the
    target structure, so it isolates whether any gain is tied to WHERE the weight falls.
    """
    if x.shape[-1] != 1024:
        raise ValueError(f"prereg §7 fixes the shift at +256 for T=1024; got T={x.shape[-1]} — a new preregistration is required")
    return torch.roll(structure_weight(x, lam), shifts=shift, dims=-1)


def hard_qrs_weight(x: torch.Tensor, r_peaks: list, half: int = 10, lam: float = LAMBDA_STRUCT) -> torch.Tensor:
    """ARM Q (prereg §7) — a target-derived ROI control, NOT the proposed method.

    Uses frozen GT R annotations during TRAINING ONLY: mask = R +- `half` samples, a = 1 + lam*mask, mean-1
    normalised exactly as arm S. `r_peaks` is one integer array per row of x. Never reachable at inference.
    """
    mask = torch.zeros_like(x)
    T = x.shape[-1]
    flat = mask.reshape(-1, T)
    for i, pk in enumerate(r_peaks):
        for c in pk:
            lo, hi = max(int(c) - half, 0), min(int(c) + half + 1, T)
            if lo < hi:
                flat[i, lo:hi] = 1.0
    a = 1.0 + lam * mask
    return (a / a.mean(dim=-1, keepdim=True)).detach()
