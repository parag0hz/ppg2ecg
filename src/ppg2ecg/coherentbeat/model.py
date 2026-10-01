"""CoherentBeat-C0 network (prereg §4).

x_hat(t) = g(t) + sum_i m_i(t - r_i) q_i(t - r_i)
  g      cubic B-spline field (17 coefficients, 250 ms spacing) over the whole 4 s window, from a shared PPG encoder
  q_i    deterministic event-local residual on the integer grid -38 ... +57 samples around event r_i, conditioned on the
         encoder features around r_i, (RR_prev, RR_next) and the window's encoder summary
  m_i    compact bump: 1 at r_i, value and slope 0 at r_i - L_i and r_i + R_i
No renormalization, no fill, no time warping. LOCAL-ONLY is the same network with g = 0.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppg2ecg.beatfirst.model import FiLMBlock

from . import geometry as G

CH, KERNEL = 64, 5
ENC_DILATIONS = (1, 2, 4, 8, 16, 32, 1, 2)
LOCAL_DILATIONS = (1, 2, 4, 8)
COND_HIDDEN = 128


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int, d: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding="same", dilation=d)
        self.c2 = nn.Conv1d(ch, ch, k, padding="same", dilation=d)
        self.act = nn.GELU()

    def forward(self, h):
        return self.act(h + self.c2(self.act(self.c1(h))))


class Encoder(nn.Module):
    def __init__(self, ch: int = CH, k: int = KERNEL, dilations=ENC_DILATIONS):
        super().__init__()
        self.stem = nn.Conv1d(1, ch, 1)
        self.blocks = nn.ModuleList([ResBlock(ch, k, int(d)) for d in dilations])

    def forward(self, x):                       # [B, 1, T] -> [B, C, T]
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return h


class CoherentBeat(nn.Module):
    def __init__(self, use_global: bool = True, ch: int = CH):
        super().__init__()
        self.use_global = bool(use_global)
        self.encoder = Encoder(ch)
        if self.use_global:
            self.global_head = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.GELU(), nn.Conv1d(ch, 1, 1))
            self.register_buffer("basis", torch.from_numpy(G.spline_basis()).float(), persistent=False)
        self.cond = nn.Sequential(nn.Linear(ch + 2, COND_HIDDEN), nn.GELU(), nn.Linear(COND_HIDDEN, COND_HIDDEN), nn.GELU())
        self.local_in = nn.Conv1d(ch, ch, 1)
        self.local_blocks = nn.ModuleList([FiLMBlock(ch, KERNEL, int(d), COND_HIDDEN) for d in LOCAL_DILATIONS])
        self.local_head = nn.Conv1d(ch, 1, 1)
        self.register_buffer("tau", torch.arange(G.TAU_LO, G.TAU_HI + 1, dtype=torch.float32), persistent=False)

    def global_field(self, h):
        """Average the encoder features over +-16 samples around each control point, map to one coefficient each."""
        hp = F.pad(h, (16, 16), mode="replicate")                    # length T + 32 = 544 = 17 * 32
        pooled = F.avg_pool1d(hp, kernel_size=G.KNOT_SPACING, stride=G.KNOT_SPACING)
        coef = self.global_head(pooled)[:, 0]                        # [B, 17]
        return coef @ self.basis.T                                   # [B, T]

    def forward(self, x, events, rr_edge: float, return_parts: bool = False):
        """x: [B, T] PPG; events: list of B sorted int arrays (absolute sample positions); rr_edge in samples."""
        B, T = x.shape
        h = self.encoder(x[:, None])
        g = self.global_field(h) if self.use_global else torch.zeros(B, T, device=x.device)
        win, pos, Ls, Rs, rprev, rnext = [], [], [], [], [], []
        for b, ev in enumerate(events):
            ev = np.asarray(ev, dtype=np.int64).reshape(-1)
            if ev.size == 0:
                continue
            L, R = G.supports(ev, rr_edge)
            d = np.diff(ev.astype(np.float64))
            win.append(np.full(ev.size, b))
            pos.append(ev)
            Ls.append(L)
            Rs.append(R)
            rprev.append(np.concatenate([[rr_edge], d]))
            rnext.append(np.concatenate([d, [rr_edge]]))
        local = torch.zeros(B, T, device=x.device)
        if win:
            win_t = torch.from_numpy(np.concatenate(win)).to(x.device)
            pos_t = torch.from_numpy(np.concatenate(pos)).to(x.device)
            L_t = torch.from_numpy(np.maximum(np.concatenate(Ls), 1e-6)).float().to(x.device)
            R_t = torch.from_numpy(np.maximum(np.concatenate(Rs), 1e-6)).float().to(x.device)
            rr = torch.from_numpy(np.stack([np.concatenate(rprev), np.concatenate(rnext)], 1) / G.FS).float().to(x.device)
            hp = F.pad(h, (-G.TAU_LO, G.TAU_HI))                     # zero features outside the window
            idx = pos_t[:, None] + torch.arange(G.N_TAU, device=x.device)[None]     # padded coordinates
            feat = hp[win_t[:, None], :, idx].permute(0, 2, 1)       # [E, C, 96]
            summary = h.mean(dim=-1)[win_t]                          # [E, C]
            c = self.cond(torch.cat([summary, rr], dim=1))
            z = self.local_in(feat)
            for blk in self.local_blocks:
                z = blk(z, c)
            q = self.local_head(z)[:, 0]                             # [E, 96]
            m = G.envelope(self.tau[None, :], L_t[:, None], R_t[:, None])
            contrib = m * q
            t_abs = pos_t[:, None] + self.tau.long()[None, :]
            ok = (t_abs >= 0) & (t_abs < T)
            flat = (win_t[:, None] * T + t_abs)[ok]
            local = local.reshape(-1).index_add(0, flat, contrib[ok]).reshape(B, T)
        out = g + local
        if return_parts:
            return out, g, local
        return out


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
