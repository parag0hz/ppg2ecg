"""C0-A ablation architectures (docs/C0A_COHERENTBEAT_ABLATION_PREREGISTRATION.md). The frozen C0 modules are imported,
never modified.

  PM-BF0-DET          BF0's BeatFlowNet with only the channel width changed to match C0's parameter count
  CONST-GLOBAL-LOCAL  CoherentBeat with the 17-coefficient spline replaced by one scalar level per window
  WW-DET              a direct whole-window predictor: C0's encoder family + a decoder that also sees a Gaussian event raster
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ppg2ecg.beatfirst.model import BeatFlowNet, n_params
from ppg2ecg.probes.rhythm_tcn import soft_event_field

from . import model as CM

C0_PARAMS = 592_770
RASTER_SIGMA = 20.0 / 1000.0 * 128          # the RD1 / C0 detector target convention: Gaussian, sigma = 20 ms
WW_DEC_DILATIONS = (1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)
PM_CH_GRID = tuple(range(32, 129))
WW_GRID = tuple((d, n) for d in (32, 40, 48, 56, 64, 72, 80, 88, 96) for n in range(1, 13))


class ConstGlobalLocal(CM.CoherentBeat):
    """CoherentBeat-C0 with g(t) = a_global (one scalar per window from the window encoder mean). Encoder, local residual
    branch, supports, envelope and forward are inherited unchanged; only `global_field` is overridden."""

    def __init__(self, ch: int = CM.CH):
        super().__init__(use_global=True, ch=ch)
        del self.global_head                      # no spline coefficients, no unused parameters
        del self.basis
        self.const_head = nn.Sequential(nn.Linear(ch, ch), nn.GELU(), nn.Linear(ch, 1))

    def global_field(self, h):
        return self.const_head(h.mean(dim=-1)).expand(-1, h.shape[-1])


class WWDet(nn.Module):
    """Event-conditioned whole-window deterministic predictor: PPG -> C0 encoder family; [features, event raster] -> 1x1 ->
    residual dilated blocks -> 1x1 -> the 512-sample ECG window. No additive decomposition, no supports, no stitching."""

    def __init__(self, dec_ch: int, n_dec: int, ch: int = CM.CH):
        super().__init__()
        self.encoder = CM.Encoder(ch)
        self.dec_in = nn.Conv1d(ch + 1, dec_ch, 1)
        self.dec = nn.ModuleList([CM.ResBlock(dec_ch, CM.KERNEL, int(WW_DEC_DILATIONS[i])) for i in range(n_dec)])
        self.head = nn.Conv1d(dec_ch, 1, 1)

    def forward(self, x, raster):
        """x, raster: [B, T] -> [B, T]."""
        h = self.encoder(x[:, None])
        z = self.dec_in(torch.cat([h, raster[:, None]], dim=1))
        for b in self.dec:
            z = b(z)
        return self.head(z)[:, 0]


def event_raster(events, n_time: int = 512) -> np.ndarray:
    """[B, T] float32: per window, max over placed events of a Gaussian with sigma = 20 ms (`soft_event_field`)."""
    return np.stack([soft_event_field(np.asarray(e, dtype=np.float64), n_time, RASTER_SIGMA) for e in events]).astype(np.float32)


def select_closest(candidates: dict, target: int = C0_PARAMS):
    """Pick the candidate whose parameter count is closest to `target` (ties: the first in the given order).
    Returns (key, params, relative mismatch). Parameter count is the ONLY criterion."""
    best = min(candidates.items(), key=lambda kv: (abs(kv[1] - target), list(candidates).index(kv[0])))
    return best[0], int(best[1]), (best[1] - target) / target


def pm_bf0_candidates() -> dict:
    return {ch: n_params(BeatFlowNet(ch=ch)) for ch in PM_CH_GRID}


def ww_candidates() -> dict:
    return {(d, n): n_params(WWDet(d, n)) for d, n in WW_GRID}
