"""BF0 timing stage: event features, N5-style heteroscedastic head, conformal quantiles, coverage (prereg §2.2–2.3)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

FS = 128
FIELD_HALF, PPG_HALF = 32, 96
MATCH_MS = 150.0
MIN_SIGMA_MS, MAX_SIGMA_MS = 1.0, 500.0
ALPHAS = (0.5, 0.8, 0.9)


class TimingHead(nn.Module):
    """N5's head: features -> (mu, log sigma) of the timing residual in ms."""

    def __init__(self, n_in: int = (2 * FIELD_HALF + 1) + (2 * PPG_HALF + 1), h: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2))

    def forward(self, x):
        o = self.net(x)
        return o[:, 0], o[:, 1]


def clamp_log_sigma(log_sigma: torch.Tensor) -> torch.Tensor:
    return torch.clamp(log_sigma, math.log(MIN_SIGMA_MS), math.log(MAX_SIGMA_MS))


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    s = clamp_log_sigma(log_sigma)
    return 0.5 * math.log(2 * math.pi) + s + 0.5 * ((y - mu) / torch.exp(s)) ** 2


def _window(sig: np.ndarray, c: int, half: int) -> np.ndarray:
    idx = np.clip(np.arange(int(c) - half, int(c) + half + 1), 0, sig.size - 1)
    return sig[idx]


def event_features(field: np.ndarray, ppg: np.ndarray, events) -> np.ndarray:
    """Detector field e±32 and PPG e±96 per event, edge-padded. -> [n_events, 65 + 193] float32."""
    ev = np.asarray(events, dtype=int).reshape(-1)
    if ev.size == 0:
        return np.zeros((0, (2 * FIELD_HALF + 1) + (2 * PPG_HALF + 1)), np.float32)
    f = np.asarray(field, dtype=np.float32)
    p = np.asarray(ppg, dtype=np.float32)
    return np.stack([np.concatenate([_window(f, e, FIELD_HALF), _window(p, e, PPG_HALF)]) for e in ev]).astype(np.float32)


def matched_residuals(events, ref_peaks, fs: int = FS, match_ms: float = MATCH_MS) -> np.ndarray:
    """Residual (nearest reference R - event) in ms for each event; nan if none within ±match_ms."""
    ev = np.asarray(events, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref_peaks, dtype=np.float64).reshape(-1)
    out = np.full(ev.size, np.nan)
    if ev.size == 0 or ref.size == 0:
        return out
    half = match_ms / 1000.0 * fs
    for i, e in enumerate(ev):
        d = ref - e
        j = int(np.argmin(np.abs(d)))
        if abs(d[j]) <= half:
            out[i] = d[j] / fs * 1000.0
    return out


def conformal_quantile(scores, alpha: float) -> float:
    """Split-conformal quantile: the ceil((n+1) alpha)-th smallest score (inf if that exceeds n)."""
    s = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * float(alpha)))
    return float("inf") if k > n else float(s[k - 1])


def coverage(resid, mu, sigma, q: float) -> float:
    """Fraction of finite residuals inside mu ± q sigma."""
    r, m, s = (np.asarray(a, dtype=np.float64).reshape(-1) for a in (resid, mu, sigma))
    ok = np.isfinite(r)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(r[ok] - m[ok]) <= q * s[ok]))


def gaussian_z(alpha: float) -> float:
    """Two-sided standard normal quantile z_{(1+alpha)/2}."""
    from scipy.stats import norm
    return float(norm.ppf(0.5 + 0.5 * float(alpha)))
