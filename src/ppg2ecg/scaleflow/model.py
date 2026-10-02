"""SF0 models, the fixed Haar transform, the linear flow-matching path, Euler sampling and the per-window noise map.

A WW-L1                 C0-A's WWDet (decoder width 72, depth 5): [PPG features, event raster] -> ECG, L1.
B WW-FM                 the WW family + x_t input + time embedding: C0 encoder on PPG; 1x1 on [h, raster, x_t]; time-conditioned
                        residual decoder -> 512-sample velocity.
C SCALE-FM-INDEPENDENT  Haar(x_t), Haar(PPG), Haar(raster) -> three independent residual branches (coarse 128, mid 128,
                        fine 256) -> inverse Haar velocity.
D SCALEFLOW-COUPLED     as C plus hierarchical coupling: coarse features -> mid input; mid and coarse features -> fine input
                        (1x1 projections, nearest upsampling). No fine -> coarse path.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn as nn

from ppg2ecg.coherentbeat import ablation as AB
from ppg2ecg.coherentbeat import model as CM

T = 512
SQ2 = math.sqrt(2.0)
KERNEL = 5
T_SIN, T_HID = 64, 128                       # sinusoidal dim, time-embedding width
WW_FM_DEPTH = 5                              # WW-DET decoder depth
WW_FM_DIL = (1, 2, 4, 8, 16)
BRANCH_DEPTH = 6                             # residual blocks per scale branch
BRANCH_DIL = (1, 2, 4, 8, 16, 32)
PROJ = 16                                    # channels of each cross-scale projection (coupled model)
NOISE_SALT = 20261002


# ----------------------------------------------------------------------------------------------- fixed Haar transform
def haar1(x):
    a, b = x[..., 0::2], x[..., 1::2]
    return (a + b) / SQ2, (a - b) / SQ2


def ihaar1(low, high):
    return torch.stack([(low + high) / SQ2, (low - high) / SQ2], dim=-1).flatten(-2)


def haar(x):
    """[..., 512] -> (coarse = low_2 [..., 128], mid = high_2 [..., 128], fine = high_1 [..., 256]). No parameters."""
    low1, fine = haar1(x)
    coarse, mid = haar1(low1)
    return coarse, mid, fine


def ihaar(coarse, mid, fine):
    return ihaar1(ihaar1(coarse, mid), fine)


# ----------------------------------------------------------------------------------------------- blocks
class TimeEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(T_SIN, T_HID), nn.GELU(), nn.Linear(T_HID, T_HID))

    def forward(self, t):                      # t [B] in [0, 1]
        half = T_SIN // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        a = 1000.0 * t.float()[:, None] * f[None, :]
        return self.mlp(torch.cat([torch.sin(a), torch.cos(a)], dim=1))


class TResBlock(nn.Module):
    """C0's residual block with the time embedding added (additive projection) after the first convolution."""

    def __init__(self, ch, d):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, KERNEL, padding="same", dilation=d)
        self.c2 = nn.Conv1d(ch, ch, KERNEL, padding="same", dilation=d)
        self.tp = nn.Linear(T_HID, ch)
        self.act = nn.GELU()

    def forward(self, h, e):
        return self.act(h + self.c2(self.act(self.c1(h) + self.tp(e)[:, :, None])))


def ww_l1() -> AB.WWDet:
    return AB.WWDet(72, 5)


class WWFM(nn.Module):
    def __init__(self, dec_ch: int):
        super().__init__()
        self.encoder = CM.Encoder(CM.CH)
        self.temb = TimeEmbed()
        self.dec_in = nn.Conv1d(CM.CH + 2, dec_ch, 1)
        self.dec = nn.ModuleList([TResBlock(dec_ch, d) for d in WW_FM_DIL])
        self.head = nn.Conv1d(dec_ch, 1, 1)

    def forward(self, xt, t, ppg, raster):
        h = self.encoder(ppg[:, None])
        e = self.temb(t)
        z = self.dec_in(torch.cat([h, raster[:, None], xt[:, None]], dim=1))
        for b in self.dec:
            z = b(z, e)
        return self.head(z)[:, 0]


class Branch(nn.Module):
    def __init__(self, cin, w):
        super().__init__()
        self.stem = nn.Conv1d(cin, w, 1)
        self.blocks = nn.ModuleList([TResBlock(w, d) for d in BRANCH_DIL[:BRANCH_DEPTH]])
        self.head = nn.Conv1d(w, 1, 1)

    def forward(self, x, e):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h, e)
        return h, self.head(h)[:, 0]


def up2(h):
    return h.repeat_interleave(2, dim=-1)


class ScaleFM(nn.Module):
    """coupled=False: SCALE-FM-INDEPENDENT (no hidden feature crosses scales). coupled=True: SCALEFLOW-COUPLED."""

    def __init__(self, w: int, coupled: bool):
        super().__init__()
        self.coupled = bool(coupled)
        self.temb = TimeEmbed()
        p = PROJ if coupled else 0
        self.coarse = Branch(3, w)
        self.mid = Branch(3 + p, w)
        self.fine = Branch(3 + 2 * p, w)
        if coupled:
            self.proj_cm = nn.Conv1d(w, PROJ, 1)
            self.proj_cf = nn.Conv1d(w, PROJ, 1)
            self.proj_mf = nn.Conv1d(w, PROJ, 1)

    def forward(self, xt, t, ppg, raster, return_parts: bool = False):
        xc, xm, xf = haar(xt)
        pc, pm, pf = haar(ppg)
        ec, em, ef = haar(raster)
        e = self.temb(t)
        hc, vc = self.coarse(torch.stack([xc, pc, ec], dim=1), e)
        m_in = [torch.stack([xm, pm, em], dim=1)]
        if self.coupled:
            m_in.append(self.proj_cm(hc))                                     # coarse and mid share length 128
        hm, vm = self.mid(torch.cat(m_in, dim=1), e)
        f_in = [torch.stack([xf, pf, ef], dim=1)]
        if self.coupled:
            f_in += [up2(self.proj_mf(hm)), up2(self.proj_cf(hc))]
        _, vf = self.fine(torch.cat(f_in, dim=1), e)
        v = ihaar(vc, vm, vf)                                                  # ONE whole-window velocity
        return (v, (vc, vm, vf)) if return_parts else v


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------- parameter matching
TARGET = 600_000
WIDTH_GRID = tuple(range(16, 161))


def closest(cands: dict, target: int):
    k = min(cands, key=lambda c: (abs(cands[c] - target), c))
    return k, cands[k], (cands[k] - target) / target


def param_match() -> dict:
    """Widths chosen by parameter count ONLY (depths fixed structurally): SCALEFLOW closest to 600k; INDEPENDENT and
    WW-FM closest to SCALEFLOW's count."""
    sf = {w: n_params(ScaleFM(w, True)) for w in WIDTH_GRID}
    w_sf, p_sf, r_sf = closest(sf, TARGET)
    ind = {w: n_params(ScaleFM(w, False)) for w in WIDTH_GRID}
    w_ind, p_ind, _ = closest(ind, p_sf)
    ww = {w: n_params(WWFM(w)) for w in WIDTH_GRID}
    w_ww, p_ww, _ = closest(ww, p_sf)
    return {"SF": {"width": w_sf, "params": p_sf, "vs_600k": r_sf},
            "IND": {"width": w_ind, "params": p_ind, "vs_SF": (p_ind - p_sf) / p_sf},
            "WWFM": {"dec_width": w_ww, "params": p_ww, "vs_SF": (p_ww - p_sf) / p_sf},
            "WWL1": {"params": n_params(ww_l1())}}


# ----------------------------------------------------------------------------------------------- flow matching
def fm_path(x1, x0, t):
    """Linear conditional path: x_t = (1 - t) x0 + t x1, target u = x1 - x0."""
    tt = t[:, None]
    return (1.0 - tt) * x0 + tt * x1, x1 - x0


def fm_loss(model, x1, ppg, raster, gen):
    x0 = torch.randn(x1.shape, generator=gen, device=x1.device)
    t = torch.rand(x1.shape[0], generator=gen, device=x1.device)
    xt, u = fm_path(x1, x0, t)
    return ((model(xt, t, ppg, raster) - u) ** 2).mean()


@torch.no_grad()
def euler(model, x0, ppg, raster, nfe: int = 8):
    x = x0.clone()
    dt = 1.0 / nfe
    for i in range(nfe):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * model(x, t, ppg, raster)
    return x


def noise_seed(patient_id, window_id, k: int = 0) -> int:
    key = f"{int(patient_id)}:{int(window_id)}:{NOISE_SALT}" + (f":k{k}" if k else "")
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def window_noise(pids, wids, k: int = 0) -> np.ndarray:
    """One fixed N(0, I) 512-vector per window (identical for every FM arm)."""
    return np.stack([np.random.default_rng(noise_seed(p, w, k)).standard_normal(T) for p, w in zip(pids, wids)]).astype(np.float32)


# ----------------------------------------------------------------------------------------------- frozen gates
M_CORR, M_FP, M_RECALL = 0.02, 0.05, 0.01


def g1_label(d: dict) -> str:
    """WW-FM - WW-L1 (context only)."""
    better_fd = d["fd"][2] < 0
    safe = d["corr"][1] > -M_CORR and d["fp"][2] < M_FP
    if better_fd and safe:
        return "FM BETTER"
    if not better_fd and d["fd"][1] >= 0 and d["corr"][2] <= 0:
        return "NO BENEFIT"
    return "MIXED"


def gates(c: dict) -> dict:
    g = {"G2": c["fd_sf_wwfm"][2] < 0, "G3": c["fd_sf_best"][2] < 0, "G4": c["corr_sf_bestcorr"][1] > -M_CORR,
         "G5": c["fp_sf_wwl1"][2] < M_FP and c["recall_sf_wwl1"][1] > -M_RECALL,
         "G6": c["fd_sf_ind"][2] < 0 and c["corr_sf_ind"][1] > -M_CORR,
         "G7": c["fd_shuf_cond"][1] > 0 and c["corr_shuf_cond"][2] < 0}
    g = {k: bool(v) for k, v in g.items()}
    g["QUALIFIED"] = all(g.values())
    return g


def failure_categories(g: dict, g1: str) -> list[str]:
    out = []
    if g1 == "NO BENEFIT":
        out.append("FM objective itself weak")
    if not (g["G2"] and g["G3"]):
        out.append("multiresolution representation weak")
    if not g["G6"]:
        out.append("coupling unsupported")
    if not g["G4"]:
        out.append("morphology degradation")
    if not g["G5"]:
        out.append("event degradation")
    if not g["G7"]:
        out.append("PPG conditioning unused")
    return out
