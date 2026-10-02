"""AF0 models.

Anchor (B0 baseline)      SF0's WW-L1 (C0-A WWDet 72 x 5): [PPG, event raster] -> mu(c), L1.
Full-signal ScaleFlow (B1) SF0's SCALEFLOW-COUPLED (ScaleFM(50, coupled)), flow on the full ECG.
Vanilla residual FM (B2)   WW family on the waveform-domain normalized residual: C0 encoder on PPG; 1x1 on
                           [h, raster, mu, y_t] -> 5 time-conditioned residual blocks -> velocity (512).
Residual scale FM (B3/B4)  Haar bands of the normalized residual; per-scale branch input [z_s, PPG_s, event_s, mu_s];
                           coupling None (B3) or C0 additive / C1 concatenative / C2 gated (B4); anchor injection
                           B0 concat / B1 FiLM / B2 both.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ppg2ecg.coherentbeat import model as CM
from ppg2ecg.scaleflow import model as SM

PROJ = SM.PROJ
DEPTH, DIL = SM.BRANCH_DEPTH, SM.BRANCH_DIL
BANDS = (("coarse", 0, 128), ("mid", 128, 256), ("fine", 256, 512))      # positions in the concatenated [c, m, f] vector


# ----------------------------------------------------------------------------------------------- band helpers
def to_bands(x):
    """[B, 512] waveform -> [B, 512] concatenated Haar coefficients [coarse 128 | mid 128 | fine 256]."""
    c, m, f = SM.haar(x)
    return torch.cat([c, m, f], dim=-1)


def from_bands(z):
    return SM.ihaar(z[..., 0:128], z[..., 128:256], z[..., 256:512])


class Normalizer:
    """Frozen AF-TRAIN statistics. kind A0: per-band mean / std; A1: per-band median / IQR; A2: one global mean / std.
    'waveform': global mean / std of the waveform-domain residual (vanilla baseline)."""

    def __init__(self, kind: str, loc, scale):
        self.kind = kind
        self.loc = np.asarray(loc, np.float64)
        self.scale = np.asarray(scale, np.float64)

    @staticmethod
    def fit(kind: str, R: np.ndarray, floor: float = 1e-6) -> "Normalizer":
        R = np.asarray(R, np.float64)
        if kind == "waveform":
            return Normalizer(kind, [R.mean()], [max(R.std(), floor)])
        Z = to_bands(torch.from_numpy(R)).numpy()
        if kind == "A2":
            return Normalizer(kind, [Z.mean()], [max(Z.std(), floor)])
        loc, scale = [], []
        for _, a, b in BANDS:
            v = Z[:, a:b].ravel()
            if kind == "A0":
                loc.append(v.mean()); scale.append(max(v.std(), floor))
            elif kind == "A1":
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                loc.append(med); scale.append(max(q3 - q1, floor))
            else:
                raise ValueError(kind)
        return Normalizer(kind, loc, scale)

    def _vec(self, a, dev, dtype):
        if self.kind in ("waveform", "A2"):
            return torch.full((512,), float(a[0]), device=dev, dtype=dtype)
        return torch.cat([torch.full((b - s,), float(a[i]), device=dev, dtype=dtype) for i, (_, s, b) in enumerate(BANDS)])

    def encode(self, r):
        """waveform residual [B, 512] -> normalized flow variable [B, 512]."""
        z = r if self.kind == "waveform" else to_bands(r)
        return (z - self._vec(self.loc, r.device, r.dtype)) / self._vec(self.scale, r.device, r.dtype)

    def decode(self, y):
        z = y * self._vec(self.scale, y.device, y.dtype) + self._vec(self.loc, y.device, y.dtype)
        return z if self.kind == "waveform" else from_bands(z)

    def to_json(self):
        return {"kind": self.kind, "loc": self.loc.tolist(), "scale": self.scale.tolist()}

    @staticmethod
    def from_json(d):
        return Normalizer(d["kind"], d["loc"], d["scale"])


# ----------------------------------------------------------------------------------------------- vanilla residual FM (B2)
class VanillaResidualFM(nn.Module):
    def __init__(self, dec_ch: int):
        super().__init__()
        self.encoder = CM.Encoder(CM.CH)
        self.temb = SM.TimeEmbed()
        self.dec_in = nn.Conv1d(CM.CH + 3, dec_ch, 1)
        self.dec = nn.ModuleList([SM.TResBlock(dec_ch, d) for d in SM.WW_FM_DIL])
        self.head = nn.Conv1d(dec_ch, 1, 1)

    def forward(self, yt, t, ppg, raster, mu):
        h = self.encoder(ppg[:, None])
        e = self.temb(t)
        z = self.dec_in(torch.cat([h, raster[:, None], mu[:, None], yt[:, None]], dim=1))
        for b in self.dec:
            z = b(z, e)
        return self.head(z)[:, 0]


# ----------------------------------------------------------------------------------------------- residual scale FM (B3 / B4)
class FiLMResBlock(SM.TResBlock):
    """TResBlock with optional FiLM (1 + gamma, beta) from per-position anchor features after the first convolution."""

    def __init__(self, ch, d, film: bool):
        super().__init__(ch, d)
        self.film = nn.Conv1d(ch, 2 * ch, 1) if film else None

    def forward(self, h, e, a=None):
        u = self.c1(h) + self.tp(e)[:, :, None]
        if self.film is not None:
            g, b = self.film(a).chunk(2, dim=1)
            u = u * (1.0 + g) + b
        return self.act(h + self.c2(self.act(u)))


class ResBranch(nn.Module):
    def __init__(self, cin, w, film: bool):
        super().__init__()
        self.stem = nn.Conv1d(cin, w, 1)
        self.anchor_feat = nn.Conv1d(1, w, 1) if film else None
        self.blocks = nn.ModuleList([FiLMResBlock(w, d, film) for d in DIL[:DEPTH]])
        self.head = nn.Conv1d(w, 1, 1)

    def forward(self, x, e, mu_s, add=None):
        h = self.stem(x)
        if add is not None:
            h = h + add
        a = self.anchor_feat(mu_s[:, None]) if self.anchor_feat is not None else None
        for b in self.blocks:
            h = b(h, e, a)
        return h, self.head(h)[:, 0]


class ResidualScaleFM(nn.Module):
    """coupling: None (independent), 'C0' additive projected features, 'C1' concatenative (stem input), 'C2' gated additive.
    anchor: 'B0' concat mu_s, 'B1' FiLM from mu_s, 'B2' concat + FiLM. Input / output: normalized flow variable [B, 512]."""

    def __init__(self, w: int, coupling=None, anchor: str = "B0"):
        super().__init__()
        self.coupling, self.anchor = coupling, anchor
        concat_mu = anchor in ("B0", "B2")
        film = anchor in ("B1", "B2")
        base = 3 + (1 if concat_mu else 0)                                   # z, PPG, event (+ mu)
        p = PROJ if coupling == "C1" else 0
        self.temb = SM.TimeEmbed()
        self.coarse = ResBranch(base, w, film)
        self.mid = ResBranch(base + p, w, film)
        self.fine = ResBranch(base + 2 * p, w, film)
        if coupling in ("C0", "C1", "C2"):
            pc = PROJ if coupling == "C1" else w
            self.proj_cm = nn.Conv1d(w, pc, 1)
            self.proj_cf = nn.Conv1d(w, pc, 1)
            self.proj_mf = nn.Conv1d(w, pc, 1)
        if coupling == "C2":
            self.g_m = nn.Parameter(torch.zeros(w, 1))
            self.g_fm = nn.Parameter(torch.zeros(w, 1))
            self.g_fc = nn.Parameter(torch.zeros(w, 1))

    def forward(self, yt, t, ppg, raster, mu):
        zc, zm, zf = yt[:, 0:128], yt[:, 128:256], yt[:, 256:512]
        pc, pm, pf = SM.haar(ppg)
        ec, em, ef = SM.haar(raster)
        mc, mm, mf = SM.haar(mu)
        e = self.temb(t)
        cat_mu = self.anchor in ("B0", "B2")
        inp = lambda z, p, ev, m: torch.stack([z, p, ev, m] if cat_mu else [z, p, ev], dim=1)  # noqa: E731
        hc, vc = self.coarse(inp(zc, pc, ec, mc), e, mc)
        if self.coupling == "C1":
            hm, vm = self.mid(torch.cat([inp(zm, pm, em, mm), self.proj_cm(hc)], dim=1), e, mm)
            _, vf = self.fine(torch.cat([inp(zf, pf, ef, mf), SM.up2(self.proj_mf(hm)), SM.up2(self.proj_cf(hc))], dim=1), e, mf)
        elif self.coupling in ("C0", "C2"):
            gm = torch.sigmoid(self.g_m) if self.coupling == "C2" else 1.0
            hm, vm = self.mid(inp(zm, pm, em, mm), e, mm, add=gm * self.proj_cm(hc))
            gfm = torch.sigmoid(self.g_fm) if self.coupling == "C2" else 1.0
            gfc = torch.sigmoid(self.g_fc) if self.coupling == "C2" else 1.0
            _, vf = self.fine(inp(zf, pf, ef, mf), e, mf, add=gfm * SM.up2(self.proj_mf(hm)) + gfc * SM.up2(self.proj_cf(hc)))
        else:
            hm, vm = self.mid(inp(zm, pm, em, mm), e, mm)
            _, vf = self.fine(inp(zf, pf, ef, mf), e, mf)
        return torch.cat([vc, vm, vf], dim=-1)                               # normalized-coefficient velocity


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


TARGET = 600_000
GRID = tuple(range(16, 161))


def width_for(builder, target: int = TARGET) -> tuple[int, int]:
    c = {w: n_params(builder(w)) for w in GRID}
    w = min(c, key=lambda k: (abs(c[k] - target), k))
    return w, c[w]


# ----------------------------------------------------------------------------------------------- flow helpers
def fm_loss(model, y1, ppg, raster, mu, gen, ppg_dropout: float = 0.0):
    y0 = torch.randn(y1.shape, generator=gen, device=y1.device)
    t = torch.rand(y1.shape[0], generator=gen, device=y1.device)
    if ppg_dropout > 0:
        keep = (torch.rand(y1.shape[0], generator=gen, device=y1.device) >= ppg_dropout).float()[:, None]
        ppg = ppg * keep
    yt, u = SM.fm_path(y1, y0, t)
    return ((model(yt, t, ppg, raster, mu) - u) ** 2).mean()


@torch.no_grad()
def euler(model, y0, ppg, raster, mu, nfe: int = 8):
    y = y0.clone()
    dt = 1.0 / nfe
    for i in range(nfe):
        t = torch.full((y.shape[0],), i * dt, device=y.device)
        y = y + dt * model(y, t, ppg, raster, mu)
    return y
