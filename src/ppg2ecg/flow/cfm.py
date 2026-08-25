"""Conditional flow matching objective — a faithful re-statement of upstream PENGUIN.train_flow/optimize.

Upstream (external/PENGUIN/src/models/PENGUIN.py L220-237, L260-266):
    t   ~ U(0,1)                       # torch.rand(B,1)
    x0  ~ N(0, I)                      # torch.randn_like(x1)
    x_t = (1 - t) * x0 + t * x1        # t=0 is NOISE, t=1 is DATA
    v*  = x1 - x0                      # target velocity (independent coupling; NO minibatch-OT)
    loss = MSE(v_theta(x_t, ppg, t), v*)
This is the Lipman et al. "OT" conditional path with sigma_min = 0 (a.k.a. rectified-flow / linear interpolant),
not Tong et al. minibatch-OT coupling. MeanFlow / iMeanFlow is intentionally NOT implemented yet (prereg §3, arm A2).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cfm_targets(x1: torch.Tensor, t: torch.Tensor | None = None, x0: torch.Tensor | None = None):
    """Return (x_t, v_target, t, x0). x1: [B,1,T]. t: [B,1] in [0,1]."""
    B = x1.shape[0]
    if t is None:
        t = torch.rand(B, 1, device=x1.device)
    if x0 is None:
        x0 = torch.randn_like(x1)
    tt = t.reshape(-1, 1, 1)
    x_t = (1 - tt) * x0 + tt * x1
    return x_t, x1 - x0, t, x0


def cfm_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v_pred, v_target)


def euler_x1_estimate(x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Upstream's train-time 'pred_signal' (one Euler step to t=1) used only for the train MAE monitor."""
    return x_t + (1 - t).unsqueeze(-1) * v_pred
