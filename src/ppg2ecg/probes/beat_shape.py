"""N2 beat-scale shape models (docs/N2_ORACLE_TIMING_BEAT_SHAPE_PREREGISTRATION.md §3-§4).

Both trained arms share ONE encoder architecture, so the only difference between REG and IMF is the
objective and the sampler. Everything here operates at beat scale: a 193-sample PPG context centred on
a GT R anchor, predicting the 83-sample beat window [r-32, r+51].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

FS = 128
BEAT_LEN = 83                      # stamping.template_geometry()["full_len"]
BEAT_R_INDEX = 32                  # ...["r_index_full"]
CTX_HALF = 96                      # prereg §3: [r-96, r+96]
CTX_LEN = 2 * CTX_HALF + 1         # 193 samples = 1.51 s
H_DIM = 128
SEED = 42


class BeatEncoder(nn.Module):
    """PPG context [B,1,193] -> h [B,H]. Identical in both arms."""

    def __init__(self, h_dim: int = H_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(32, 64, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(64, 128, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(128, h_dim, 9, stride=2, padding=4), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, ppg: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(ppg)).squeeze(-1)


class BeatDecoder(nn.Module):
    """h [B,H] (+ optional extra conditioning) -> beat [B,83]."""

    def __init__(self, in_dim: int, h_dim: int = H_DIM, out_len: int = BEAT_LEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 4 * h_dim), nn.GELU(),
            nn.Linear(4 * h_dim, 4 * h_dim), nn.GELU(),
            nn.Linear(4 * h_dim, out_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BeatRegressor(nn.Module):
    """arm REG: deterministic PPG context -> beat, trained with MSE."""

    def __init__(self, h_dim: int = H_DIM):
        super().__init__()
        self.enc = BeatEncoder(h_dim)
        self.dec = BeatDecoder(h_dim, h_dim)

    def forward(self, ppg: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(ppg))


def sinusoidal(t: torch.Tensor, dim: int = 64, base: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(base) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.reshape(-1, 1).float() * freqs.reshape(1, -1)
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class BeatMeanFlow(nn.Module):
    """arm IMF: u(z, ppg, t, r) at beat scale. Same encoder; a (t, r) embedding is the declared extra."""

    def __init__(self, h_dim: int = H_DIM, t_dim: int = 64):
        super().__init__()
        self.enc = BeatEncoder(h_dim)
        self.t_dim = t_dim
        self.temb = nn.Sequential(nn.Linear(2 * t_dim, h_dim), nn.GELU(), nn.Linear(h_dim, h_dim))
        self.dec = BeatDecoder(h_dim + BEAT_LEN, h_dim)

    def u(self, z: torch.Tensor, ppg: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        h = self.enc(ppg) + self.temb(torch.cat([sinusoidal(t, self.t_dim), sinusoidal(r, self.t_dim)], dim=1))
        return self.dec(torch.cat([h, z], dim=1))

    def forward(self, z, ppg, t, r):
        return self.u(z, ppg, t, r)


@torch.no_grad()
def sample_one_step(net: BeatMeanFlow, ppg: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """NFE 1: x0_hat = z_1 - 1 * u(z_1, ppg, t=1, r=0), the production endpoint form."""
    one = torch.ones(len(ppg), device=ppg.device)
    return e - net.u(e, ppg, one, torch.zeros_like(one))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
