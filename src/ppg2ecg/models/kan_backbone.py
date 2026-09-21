"""KN1: the upstream PENGUIN skeleton with its feed-forward MLPs replaced by RBF-KAN layers
(docs/KN1_KAN_FFN_PREREGISTRATION.md).

KANLinear follows the design KANFlow describes (IEEE IoT-J, DOI 10.1109/JIOT.2026.3717960): a base branch plus a
Gaussian-RBF branch on a bounded input, grid size G = 5, tanh temperature 2.0:
    y = W_b · SiLU(x)  +  W_s · [exp(−((tanh(x/τ) − c_g) / σ)²)]_{g=1..G},   c_g = linspace(−1, 1, G), σ = 2 / (G − 1)
The upstream classes are imported unmodified; only `mlp_ppg` and `mlp_target` of the selected block INSTANCES are
swapped. Everything is elementwise / linear, so torch.func.jvp (needed by the iMF objective) applies.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, grid: int = 5, tau: float = 2.0):
        super().__init__()
        self.tau = tau
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, grid), persistent=False)
        self.inv_sigma = (grid - 1) / 2.0
        self.base = nn.Linear(d_in, d_out)
        self.spline = nn.Linear(d_in * grid, d_out, bias=False)
        nn.init.normal_(self.spline.weight, std=0.1 / (d_in * grid) ** 0.5)

    def forward(self, x):
        xb = torch.tanh(x / self.tau).unsqueeze(-1)                       # [..., d_in, 1]
        phi = torch.exp(-((xb - self.centers) * self.inv_sigma) ** 2)     # [..., d_in, G]
        return self.base(F.silu(x)) + self.spline(phi.flatten(-2))


class KANFFN(nn.Module):
    """Drop-in for the upstream two-layer MLP-FFN (h -> h): one KANLinear, 1.5x the parameters of the MLP at ratio 2."""

    def __init__(self, h_dim: int, grid: int = 5):
        super().__init__()
        self.kan = KANLinear(h_dim, h_dim, grid)

    def forward(self, x):
        return self.kan(x)


def build_kan_backbone(PENGUIN, cfg: dict, blocks: str = "outer", grid: int = 5) -> nn.Module:
    model = PENGUIN(**cfg)
    n = len(model.flow_ssm_list)
    chosen = range(n) if blocks == "all" else sorted({0, n - 1})
    assert blocks in ("all", "outer")
    for i in chosen:
        blk = model.flow_ssm_list[i]
        blk.mlp_ppg = KANFFN(cfg["h_dim"], grid)
        blk.mlp_target = KANFFN(cfg["h_dim"], grid)
    return model
