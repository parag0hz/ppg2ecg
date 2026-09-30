"""BF0 beat generator: `BeatFlowNet`, OT-CFM loss and Euler sampler (prereg §2.4)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

DILATIONS = (1, 2, 4, 8, 16, 32, 1, 2)
CH, KERNEL, T_DIM, COND_HIDDEN = 64, 5, 32, 128
EULER_STEPS = 8


def sinusoidal(t: torch.Tensor, dim: int = T_DIM) -> torch.Tensor:
    """[B] in [0, 1] -> [B, dim] sin/cos features."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
    ang = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class FiLMBlock(nn.Module):
    """Two dilated convs with a residual connection; FiLM (zero-initialised, starts as identity) between them."""

    def __init__(self, ch: int, k: int, d: int, cond_dim: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding="same", dilation=d)
        self.c2 = nn.Conv1d(ch, ch, k, padding="same", dilation=d)
        self.film = nn.Linear(cond_dim, 2 * ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g, b = self.film(c).chunk(2, dim=-1)
        z = self.act(self.c1(h))
        z = z * (1.0 + g[:, :, None]) + b[:, :, None]
        return self.act(h + self.c2(z))


class BeatFlowNet(nn.Module):
    """Velocity field v(x_t, t | PPG segment, RR_prev, RR_next) on R-aligned 166-sample beat segments."""

    def __init__(self, ch: int = CH, k: int = KERNEL, t_dim: int = T_DIM, cond_hidden: int = COND_HIDDEN,
                 dilations=DILATIONS):
        super().__init__()
        self.t_dim = int(t_dim)
        self.stem = nn.Conv1d(2, ch, 1)
        self.cond = nn.Sequential(nn.Linear(t_dim + 2, cond_hidden), nn.GELU(),
                                  nn.Linear(cond_hidden, cond_hidden), nn.GELU())
        self.blocks = nn.ModuleList([FiLMBlock(ch, k, int(d), cond_hidden) for d in dilations])
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, x_t: torch.Tensor, ppg: torch.Tensor, t: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        """x_t, ppg: [B, L]; t: [B]; rr: [B, 2] (seconds) -> velocity [B, L]."""
        h = self.stem(torch.stack([x_t, ppg], dim=1))
        c = self.cond(torch.cat([sinusoidal(t, self.t_dim), rr.float()], dim=1))
        for blk in self.blocks:
            h = blk(h, c)
        return self.head(h)[:, 0]


def cfm_loss(net: BeatFlowNet, x1: torch.Tensor, ppg: torch.Tensor, rr: torch.Tensor,
             x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """OT-CFM: x_t = (1 - t) x0 + t x1, regress v on x1 - x0. Noise and times are passed in (seeded by the caller)."""
    xt = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    return ((net(xt, ppg, t, rr) - (x1 - x0)) ** 2).mean()


def deterministic_predict(net: BeatFlowNet, ppg: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
    """The deterministic arm (A2): the same network with x_t = 0 and t = 0, i.e. a function of (PPG, RR) only."""
    return net(torch.zeros_like(ppg), ppg, torch.zeros(ppg.shape[0], device=ppg.device), rr)


def l1_loss(net: BeatFlowNet, x1: torch.Tensor, ppg: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
    return (deterministic_predict(net, ppg, rr) - x1).abs().mean()


@torch.no_grad()
def euler_sample(net: BeatFlowNet, ppg: torch.Tensor, rr: torch.Tensor, x0: torch.Tensor,
                 steps: int = EULER_STEPS) -> torch.Tensor:
    """Integrate dx/dt = v from t = 0 (noise x0) to t = 1 with `steps` Euler steps."""
    x = x0.clone()
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((x.shape[0],), k * dt, device=x.device)
        x = x + dt * net(x, ppg, t, rr)
    return x


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
