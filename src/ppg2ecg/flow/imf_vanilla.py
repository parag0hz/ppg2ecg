"""VM1: the official improved-MeanFlow objective and sampler (external/iMeanFlow @ bf60cd7, imf.py iMeanFlow),
ported to PyTorch for the PPG-conditioned 1-D model in ppg2ecg.models.imf_dit.

Kept exactly: logit-normal (t, r) with half the batch at r = t; CFG scale omega ~ power law on [1, 1 + s_max]
(cfg_beta = 1), guidance interval t_min ~ U(0, .5), t_max ~ U(.5, 1) (0 / 1 on the r = t rows); the guided target
v_g = v_t + (1 - 1/w)(v_c - v_u) from the auxiliary v head (h = 0, interval (0, 1)), with w forced to 1 outside
[t_min, t_max] except on the r = t rows; condition dropout on the first round(p * B) rows (their target falls back
to v_t); JVP tangent = the interval-masked v_c; V = u + (t - r) sg(du/dt); adaptive weight 1 / sg(loss + eps)^p on
both the u loss and the auxiliary v loss; sampling z_r = z_t - (t - r) u(z_t, h = t - r, omega, t_min, t_max).
Time convention: t = 1 is noise, t = 0 is data. The class label is replaced by the PPG window (null = learned).
The teacher passes (v_c, v_u) carry no gradient: in the official code they reach the loss only through
stop-gradients (v_g) or as a JVP tangent whose output is stop-gradiented.
"""
from __future__ import annotations

import torch


def sample_tr(bz: int, gen: torch.Generator, p_mean: float = -0.4, p_std: float = 1.0, data_proportion: float = 0.5):
    t = torch.sigmoid(torch.randn(bz, 1, 1, generator=gen) * p_std + p_mean)
    r = torch.sigmoid(torch.randn(bz, 1, 1, generator=gen) * p_std + p_mean)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    fm = (torch.arange(bz) < int(bz * data_proportion)).reshape(bz, 1, 1)
    return t, torch.where(fm, t, r), fm


def sample_cfg_scale(bz: int, gen: torch.Generator, s_max: float = 7.0):
    u = torch.rand(bz, 1, 1, generator=gen)
    return torch.exp(u * torch.log1p(torch.tensor(s_max)))


def sample_cfg_interval(bz: int, gen: torch.Generator, fm: torch.Tensor):
    t_min = torch.rand(bz, 1, 1, generator=gen) * 0.5
    t_max = 0.5 + torch.rand(bz, 1, 1, generator=gen) * 0.5
    return torch.where(fm, torch.zeros_like(t_min), t_min), torch.where(fm, torch.ones_like(t_max), t_max)


def vimf_loss(net, x, ppg, gen: torch.Generator, class_dropout_prob: float = 0.1, norm_p: float = 1.0,
              norm_eps: float = 0.01, s_max: float = 7.0):
    """x, ppg: [B, 1, T] on the model device; all random draws come from the CPU generator `gen`."""
    dev, bz = x.device, x.shape[0]
    t, r, fm = (a.to(dev) for a in sample_tr(bz, gen))
    e = torch.randn(x.shape, generator=gen).to(dev)
    z_t = (1 - t) * x + t * e
    v_t = e - x
    t_min, t_max = (a.to(dev) for a in sample_cfg_interval(bz, gen, fm.cpu()))
    omega = sample_cfg_scale(bz, gen, s_max).to(dev)
    zeros, ones = torch.zeros_like(t), torch.ones_like(t)
    no_null = torch.zeros(bz, dtype=torch.bool, device=dev)

    with torch.no_grad():
        v_c_full = net(z_t, zeros, omega, zeros, ones, ppg, no_null)[1]
        v_u = net(z_t, zeros, ones, zeros, ones, ppg, ~no_null)[1]
        v_g_fm = v_t + (1 - 1 / omega) * (v_c_full - v_u)
        w = torch.where((t >= t_min) & (t <= t_max), omega, ones)
        v_c = net(z_t, zeros, w, zeros, ones, ppg, no_null)[1]
        v_g = v_t + (1 - 1 / w) * (v_c - v_u)
        v_g = torch.where(fm, v_g_fm, v_g)
        n_drop = int((torch.rand(bz, generator=gen) < class_dropout_prob).sum())
        drop = torch.arange(bz, device=dev) < n_drop
        v_g = torch.where(drop.reshape(bz, 1, 1), v_t, v_g)

    def u_fn(z, t_, r_):
        return net(z, t_ - r_, omega, t_min, t_max, ppg, drop)

    (u, v), (du_dt, _) = torch.func.jvp(u_fn, (z_t, t, r), (v_c, ones, zeros))
    V = u + (t - r) * du_dt.detach()

    def adp(loss):
        return loss / (loss.detach() + norm_eps) ** norm_p

    loss_u = adp(((V - v_g) ** 2).flatten(1).sum(1))
    loss_v = adp(((v - v_g) ** 2).flatten(1).sum(1))
    loss = (loss_u + loss_v).mean()
    info = {"loss_u": ((V - v_g) ** 2).mean().detach(), "loss_v": ((v - v_g) ** 2).mean().detach(),
            "n_drop": n_drop, "omega_mean": omega.mean().detach()}
    return loss, info


@torch.no_grad()
def sample_vimf(net, ppg, e, n_steps: int, omega: float, t_min: float, t_max: float):
    """1..n-step sampling from z_1 = e along t = linspace(1, 0, n + 1). Returns (x0, nfe)."""
    bz, dev = e.shape[0], e.device
    ts = torch.linspace(1.0, 0.0, n_steps + 1)
    full = lambda v: torch.full((bz, 1, 1), float(v), device=dev)  # noqa: E731
    z = e
    for i in range(n_steps):
        t, r = float(ts[i]), float(ts[i + 1])
        u, _ = net(z, full(t - r), full(omega), full(t_min), full(t_max), ppg, None, need_v=False)
        z = z - (t - r) * u
    return z, n_steps
