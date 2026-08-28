"""B1-v2: progressive temporal-gap weighting for iMeanFlow (docs/B1_GAP_CURRICULUM_SOURCE_AUDIT.md; arXiv:2511.19065 v2 §5, App. B).

The frozen objective in `imeanflow.py` is untouched. This module re-states the identical per-sample loss and multiplies it by an
external, RNG-free, gradient-free curriculum factor STRICTLY AFTER the frozen adaptive weighting (source: "we strictly apply these
weightings after the adaptive normalization"):

    beta(h, s) = 1 - s + lambda * s * (1 - h),   h = t - r,   s = max(0, 1 - step / T_schedule)   (k = 1, linear)

Boundary rows (r = t, the 50 % fm_mask flow-matching samples) always get beta = 1 (source Eq. 8/9: beta multiplies only the
non-boundary u-loss; the alpha(t) boundary component of the source method is intentionally NOT implemented — isolated intervention).
With beta ≡ 1 (vanilla arm, or s = 0) the loss equals the frozen `imeanflow_loss` exactly.
"""
from __future__ import annotations

import torch

from ppg2ecg.flow.imeanflow import compound_V

LAMBDA_DEFAULT = 1.304639  # = 1/E[1-h] over the frozen non-boundary sampler; artifacts/b1_gap_curriculum/curriculum_calibration.json


def progress_s(step: int, t_schedule: int) -> float:
    """Linear schedule (k = 1): s = 1 at step 0, s = 0 at step >= T_schedule."""
    return max(0.0, 1.0 - step / t_schedule)


def curriculum_beta(t: torch.Tensor, r: torch.Tensor, fm_mask: torch.Tensor, s: float, lam: float = LAMBDA_DEFAULT) -> torch.Tensor:
    """beta per sample [B,1]; deterministic function of (h, s) — consumes no RNG, carries no gradient (inputs are constants)."""
    h = (t - r).detach()
    beta = 1.0 - s + lam * s * (1.0 - h)
    return torch.where(fm_mask, torch.ones_like(beta), beta)


def imeanflow_loss_b1(net, x, ppg, e, t, r, fm_mask, beta: torch.Tensor | None, norm_p: float = 1.0, norm_eps: float = 0.01, jvp_mode: str = "forward"):
    """Identical computation to the frozen `imeanflow_loss` (same V, same delta2, same adaptive weight w — byte-for-byte the same
    formulas), then `loss = mean(beta * w * delta2)`. `beta=None` (or all-ones) reproduces the frozen loss exactly.
    Returns (loss, info) with the same info keys as the frozen implementation plus per-sample diagnostics for the h-bin logging."""
    tt = t.reshape(-1, 1, 1)
    z_t = (1 - tt) * x + tt * e
    v_tgt = e - x
    with torch.no_grad():
        v_tangent = net.u(z_t, ppg, t, torch.zeros_like(t))

    def u_fn(z, t_, r_):
        return net.u(z, ppg, t_, t_ - r_)

    u, dudt, V = compound_V(u_fn, z_t, t, r, v_tangent, jvp_mode)
    delta2 = ((V - v_tgt) ** 2).flatten(1).sum(1)  # frozen: per-sample sum over dims
    w = 1.0 / (delta2.detach() + norm_eps) ** norm_p  # frozen adaptive weight — beta must NOT enter this statistic
    base = delta2 * w  # frozen per-sample adaptive-weighted loss
    b = torch.ones_like(delta2) if beta is None else beta.reshape(-1).to(base.dtype)
    loss = (b * base).mean()
    wd = w.detach()
    info = {"mse": ((V - v_tgt) ** 2).mean().detach(), "delta2_mean": delta2.mean().detach(), "u_abs_mean": u.abs().mean().detach(),
            "dudt_abs_mean": dudt.abs().mean().detach(), "v_tangent_abs_mean": v_tangent.abs().mean().detach(),
            "base_loss_mean": base.mean().detach(), "beta_mean": b.mean().detach(),
            "w_mean": wd.mean(), "w_std": wd.std(), "w_min": wd.min(), "w_max": wd.max(), "w_median": wd.median(),
            "w_p01": torch.quantile(wd, 0.01), "w_p10": torch.quantile(wd, 0.10), "w_p25": torch.quantile(wd, 0.25), "w_p75": torch.quantile(wd, 0.75), "w_p90": torch.quantile(wd, 0.90), "w_p99": torch.quantile(wd, 0.99),
            "w_saturation_frac": (wd - 1.0).abs().lt(1e-6).float().mean(), "w_near_lower_frac": wd.lt(1e-4).float().mean(),
            "loss_before_weighting": delta2.mean().detach(), "loss_after_weighting": loss.detach(),
            "per_sample": {"h": (t - r).detach().reshape(-1), "fm_mask": fm_mask.reshape(-1), "beta": b.detach(), "w": w.detach(), "delta2": delta2.detach(),
                           "u_norm": u.detach().flatten(1).norm(dim=1), "dudt_norm": dudt.detach().flatten(1).norm(dim=1),
                           "hdudt_norm": ((t - r).reshape(-1, 1, 1) * dudt.detach()).flatten(1).norm(dim=1), "V_norm": V.detach().flatten(1).norm(dim=1)}}
    return loss, info
