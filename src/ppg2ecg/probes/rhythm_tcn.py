"""R1 probes: Global-TCN and Local-TCN, the soft event target, and event extraction.

Frozen by docs/R1_PPG_GLOBAL_RHYTHM_OBSERVABILITY_PREREGISTRATION.md (c7481f9).
Input is PPG only. No ECG sample value ever enters these modules' forward paths.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FS = 128
CH, K, N_BLOCKS = 64, 5, 8
GLOBAL_DILATIONS = tuple(2 ** i for i in range(N_BLOCKS))     # 1..128
LOCAL_DILATIONS = tuple(1 for _ in range(N_BLOCKS))
SIGMA_MS = 100.0
SIGMA_SAMPLES = SIGMA_MS / 1000.0 * FS                          # 12.8, kept as float
REFRACTORY_MS = 250.0
REFRACTORY_SAMPLES = int(round(REFRACTORY_MS / 1000.0 * FS))    # 32
THRESH_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))   # 0.05 .. 0.95
TOL_MS = (50.0, 100.0, 150.0, 200.0, 250.0)


def receptive_field(dilations, k: int = K, convs_per_block: int = 2) -> int:
    """Theoretical RF of a stack of same-padded dilated convs: 1 + convs*(k-1)*sum(d)."""
    return 1 + convs_per_block * (k - 1) * int(sum(dilations))


class _Block(nn.Module):
    def __init__(self, ch: int, k: int, d: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding="same", dilation=d)
        self.c2 = nn.Conv1d(ch, ch, k, padding="same", dilation=d)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.c1(x))
        h = self.c2(h)
        return self.act(x + h)


class RhythmTCN(nn.Module):
    """PPG [B,1,T] -> per-sample R-event logits [B,1,T]. Optional 4-way site FiLM (secondary variant)."""

    def __init__(self, dilations=GLOBAL_DILATIONS, ch: int = CH, k: int = K, n_sites: int = 0):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self.stem = nn.Conv1d(1, ch, 1)
        self.blocks = nn.ModuleList([_Block(ch, k, d) for d in self.dilations])
        self.head = nn.Conv1d(ch, 1, 1)
        self.n_sites = int(n_sites)
        if self.n_sites:
            self.film = nn.Embedding(self.n_sites, 2 * ch)
            nn.init.zeros_(self.film.weight)                    # starts as identity

    @property
    def rf(self) -> int:
        return receptive_field(self.dilations)

    def forward(self, ppg: torch.Tensor, site: torch.Tensor | None = None) -> torch.Tensor:
        h = self.stem(ppg)
        if self.n_sites and site is not None:
            g, b = self.film(site).chunk(2, dim=-1)
            h = h * (1.0 + g[:, :, None]) + b[:, :, None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


def n_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def soft_event_field(r_peaks, n_time: int = 1024, sigma: float = SIGMA_SAMPLES) -> np.ndarray:
    """y(t) = max_j exp(-(t - r_j)^2 / (2 sigma^2)); zeros if no peaks."""
    t = np.arange(int(n_time), dtype=np.float64)
    r = np.asarray(r_peaks, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return np.zeros(n_time, dtype=np.float32)
    g = np.exp(-((t[:, None] - r[None, :]) ** 2) / (2.0 * sigma ** 2))
    return g.max(axis=1).astype(np.float32)


def extract_events(prob: np.ndarray, threshold: float, refractory: int = REFRACTORY_SAMPLES) -> np.ndarray:
    """Local maxima above `threshold`, greedy NMS by descending probability with a refractory window."""
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    n = p.size
    lm = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    if n >= 2:
        if p[0] > p[1]:
            lm = np.concatenate([[0], lm])
        if p[-1] > p[-2]:
            lm = np.concatenate([lm, [n - 1]])
    lm = lm[p[lm] >= threshold]
    if lm.size == 0:
        return np.zeros(0, dtype=int)
    order = lm[np.argsort(-p[lm], kind="stable")]
    keep = []
    for i in order:
        if all(abs(int(i) - int(j)) > refractory for j in keep):
            keep.append(int(i))
    return np.sort(np.asarray(keep, dtype=int))
