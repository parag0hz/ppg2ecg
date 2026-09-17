"""BB1 arm B: the upstream PENGUIN skeleton with every S5 sequence mixer replaced by bidirectional multi-head
self-attention (docs/BB1_BACKBONE_SENSITIVITY_PREREGISTRATION.md).

The upstream classes are imported unmodified; only the two `ssm_*` submodules of each block INSTANCE are swapped, so
pre-convs, timestep embedder, adaLN modulation, MLPs, residual wiring and the final layer are identical to PENGUIN.
Attention is written out (no fused kernel) so `torch.func.jvp`, which the iMF objective needs, is supported.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe


class SelfAttnMixer(nn.Module):
    """[B, T, H] -> [B, T, H]; the same call signature as upstream `S5(..., bidir=True)` inside Flow_SSM_Layer."""

    def __init__(self, h_dim: int, n_heads: int = 4, max_len: int = 4096):
        super().__init__()
        assert h_dim % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(h_dim, 3 * h_dim)
        self.proj = nn.Linear(h_dim, h_dim)
        self.register_buffer("pe", sinusoidal(max_len, h_dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        hd = H // self.n_heads
        q, k, v = self.qkv(x + self.pe[:T]).view(B, T, 3, self.n_heads, hd).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(hd), dim=-1)
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, H))


def build_attn_backbone(PENGUIN, cfg: dict, n_heads: int = 4) -> nn.Module:
    model = PENGUIN(**cfg)
    h = cfg["h_dim"]
    for blk in model.flow_ssm_list:
        blk.ssm_ppg = SelfAttnMixer(h, n_heads)
        blk.ssm_target = SelfAttnMixer(h, n_heads)
    return model
