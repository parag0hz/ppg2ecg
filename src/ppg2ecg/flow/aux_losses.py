"""LW1 auxiliary endpoint losses (docs/LW1_WEIGHTED_AUX_LOSS_PREREGISTRATION.md). Training-only; both terms are
dimensionless and equal ~1 for an uninformative prediction, so the mixing weight alpha is interpretable.

  mae_pcc : alpha * MAE / MAE_ref + (1 - alpha) * (1 - PCC)      MAE_ref = mean |x - mean(x)| of the target window
  stft    : multi-resolution spectral convergence  || |S(x_hat)| - |S(x)| ||_F / || |S(x)| ||_F  (phase-insensitive)
"""
from __future__ import annotations

import torch

EPS = 1e-6
STFT_SIZES = (32, 64, 128)


def mae_pcc_loss(x_hat: torch.Tensor, x: torch.Tensor, alpha: float) -> tuple[torch.Tensor, dict]:
    a, b = x_hat.flatten(1), x.flatten(1)
    mae = (a - b).abs().mean(1) / ((b - b.mean(1, keepdim=True)).abs().mean(1).detach() + EPS)
    ac, bc = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    pcc = (ac * bc).sum(1) / (ac.norm(dim=1) * bc.norm(dim=1) + EPS)
    loss = alpha * mae + (1.0 - alpha) * (1.0 - pcc)
    return loss.mean(), {"aux_mae_n": mae.mean().detach(), "aux_pcc": pcc.mean().detach()}


def stft_loss(x_hat: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    a, b = x_hat.flatten(1), x.flatten(1)
    terms = []
    for n in STFT_SIZES:
        win = torch.hann_window(n, device=a.device, dtype=a.dtype)
        A = torch.stft(a, n, hop_length=n // 4, window=win, return_complex=True).abs()
        B = torch.stft(b, n, hop_length=n // 4, window=win, return_complex=True).abs()
        terms.append((A - B).flatten(1).norm(dim=1) / (B.flatten(1).norm(dim=1) + EPS))
    sc = torch.stack(terms).mean(0)
    return sc.mean(), {"aux_stft_sc": sc.mean().detach()}
