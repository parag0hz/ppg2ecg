"""M3 SEC-iMF operators (docs/M3_SEC_IMEANFLOW_PREREGISTRATION.md §4-§6).

Differentiable torch equivalents of the FROZEN evaluation finite differences in
`ppg2ecg.evaluation.m1_structural` (d1 = np.diff, d2 = x[2:] - 2x[1:-1] + x[:-2]), plus the clean-endpoint
construction the production sampler implies.

M3 reuses the frozen finite-difference OPERATORS, but not the GT-R-dependent QRS support that the evaluation
metrics apply them on: `qrs_core_morphology` slices `pred[r-CORE-1 : r+CORE+2]` around GT R peaks, whereas the M3
auxiliary is computed over the WHOLE waveform. No R peak, QRS mask, event mask, predicted peak or other
target-derived support enters this module.

Endpoint algebra, verified against the production sampler
(`imeanflow.sample_meanflow` and `event_reliability.sample_meanflow_schedule`, which both compute
`z_r = z_t - (t - r) * net.u(z_t, ppg, t, t - r)`, so h = t - r):

    r = 0  =>  h0 = t  =>  u0 = net.u(z_t, ppg, t, t)  and  x0_hat = z_t - t * u0

No padding, no smoothing, no sample-rate scaling: D1 shortens the waveform axis by 1 and D2 by 2, exactly as the
frozen numpy operators do.
"""
from __future__ import annotations

import torch

EPS_STRUCT: float = 1e-6      # frozen (prereg §6); never changed after results
LAMBDA_SEC: float = 0.10      # frozen (prereg §6)
W_D1: float = 0.5             # frozen mixture (prereg §6)
W_D2: float = 0.5


def d1(x: torch.Tensor) -> torch.Tensor:
    """x[..., n+1] - x[..., n]. Output length T-1. Matches m1_structural.d1 (np.diff) exactly."""
    return x[..., 1:] - x[..., :-1]


def d2(x: torch.Tensor) -> torch.Tensor:
    """x[..., n+2] - 2x[..., n+1] + x[..., n]. Output length T-2. Matches m1_structural.d2 exactly."""
    return x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]


def clean_endpoint(net, z_t: torch.Tensor, ppg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x0_hat = z_t - t * u(z_t, ppg, t, t) — one extra TRAINING forward; never reachable at inference.

    Gradients flow through u0 into the network (prereg §6). `t` is [B,1] and broadcasts over the waveform axis.
    """
    u0 = net.u(z_t, ppg, t, t)
    return z_t - t.reshape(-1, 1, 1) * u0


def sec_loss(x0_hat: torch.Tensor, x: torch.Tensor, eps: float = EPS_STRUCT,
             w1: float = W_D1, w2: float = W_D2) -> tuple[torch.Tensor, dict]:
    """L_SEC = mean_b[ w1*L1_b + w2*L2_b ], the dimensionless relative derivative errors of prereg §6.

    The per-sample denominators s1, s2 are DETACHED; x0_hat, d1_hat and d2_hat are NOT (prereg §6). The target x is
    data and carries no gradient. Returns (loss, per-sample diagnostics).
    """
    flat = lambda a: a.flatten(1)  # noqa: E731  — [B,1,T] -> [B,T*C]; the waveform axis is last in both layouts
    d1x, d1h = d1(x), d1(x0_hat)
    d2x, d2h = d2(x), d2(x0_hat)
    s1 = flat(d1x.pow(2)).mean(1).detach()
    s2 = flat(d2x.pow(2)).mean(1).detach()
    L1 = flat((d1h - d1x).pow(2)).mean(1) / (s1 + eps)
    L2 = flat((d2h - d2x).pow(2)).mean(1) / (s2 + eps)
    loss = (w1 * L1 + w2 * L2).mean()
    return loss, {"L1": L1.detach(), "L2": L2.detach(), "s1": s1, "s2": s2}


def value_loss(x0_hat: torch.Tensor, x: torch.Tensor, eps: float = EPS_STRUCT) -> torch.Tensor:
    """ARM V control (prereg §11): waveform-VALUE endpoint consistency, no derivative terms.

    Reached only if the primary gates pass; defined here so the two arms differ solely in the auxiliary term.
    """
    sx = x.flatten(1).pow(2).mean(1).detach()
    return (((x0_hat - x).flatten(1).pow(2).mean(1)) / (sx + eps)).mean()
