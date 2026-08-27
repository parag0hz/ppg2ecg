"""Improved MeanFlow (iMF) objective and samplers for a conditional 1-D waveform backbone.

Sources (verified in docs/IMEANFLOW_AUDIT.md): MeanFlow arXiv:2505.13447 (Eq. 3-12, 22, Alg. 1-2) and Improved Mean Flows
arXiv:2512.02012 v2 (Sec. 4.1 Eq. 9-12, Alg. 1); official code external/iMeanFlow/imf.py @ bf60cd7 (L120-139 sample_tr,
L331-401 forward, L42/L112-114 sampling).

Time convention (papers/code): t = 1 is NOISE, t = 0 is DATA.
    z_t = (1 - t) x + t e,   e ~ N(0, I),   v = e - x
    u(z_t, r, t) = 1/(t - r) * int_r^t v          (average velocity, r <= t; u(z_t, t, t) = v(z_t, t))
    MeanFlow identity:  u = v - (t - r) d/dt u,     d/dt u = v * d_z u + d_t u  (= JVP of u with tangent (v, 0, 1) on (z, r, t))
    iMF (Eq. 12 / Alg. 1):  V = u_theta(z_t, r, t) + (t - r) * sg[ JVP(u_theta; (v_theta, 0, 1)) ],   v_theta = u_theta(z_t, t, t)
                            loss = E[ w * ||V - (e - x)||^2 ],   w = sg( 1 / (||V - (e - x)||^2 + c)^p ),  p = 1, c = 0.01
    1-NFE sampling:  x_hat = z_1 - u_theta(z_1, r = 0, t = 1)  (z_1 ~ N(0, I));  multi-step: z_r = z_t - (t - r) u_theta(z_t, r, t)

The network receives (t, h = t - r) as scalars (MF Sec. 4.3: u_theta(., r, t) := net(., t, t - r)); no CFG, no labels,
no auxiliary v-head (boundary-condition variant of Alg. 1), parameter count of the backbone unchanged.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class MeanFlowS5(nn.Module):
    """u_theta(z, ppg, t, h) on top of the UNMODIFIED upstream PENGUIN backbone.

    Conditioning modes (all use the backbone's single existing `timestep_embedder`, so the parameter count is identical to the
    OT-CFM baseline):
      * "h_only"   : cond = E(h_scale * h)          — the official iMF code's design (imfDiT.py L342-344: condition on h = t - r only;
                     t is inferred from z_t). Adopted for A2 (pre-registration §9, amendment 2).
      * "t_plus_h" : cond = E(t) + E(h_scale * h)   — with h_scale = 1 almost a function of t + h alone (r decodable R^2 = 0.18);
                     with h_scale = 1000 fully decodable but the JVP term is amplified 1000x and training diverged (A2 amendment 2).
      * "t_only"   : cond = E(t)                    — reproduces `backbone.forward_step` bit-exactly (parity test only).
    """

    def __init__(self, backbone: nn.Module, cond_mode: str = "h_only", h_scale: float = 1.0):
        super().__init__()
        assert cond_mode in ("h_only", "t_plus_h", "t_only")
        self.backbone = backbone
        self.cond_mode = cond_mode
        self.h_scale = float(h_scale)

    def u(self, z: torch.Tensor, ppg: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """z, ppg: [B, 1, T]; t, h: [B, 1] -> average velocity [B, 1, T]. Mirrors upstream PENGUIN.forward_step (L197-209)."""
        bb = self.backbone
        ppg_e = bb.pre_conv_ppg(ppg)
        z_e = bb.pre_conv_target(z)
        if self.cond_mode == "h_only":
            cond = bb.timestep_embedder(h.reshape(-1) * self.h_scale)
        else:
            cond = bb.timestep_embedder(t.reshape(-1))
            if self.cond_mode == "t_plus_h":
                cond = cond + bb.timestep_embedder(h.reshape(-1) * self.h_scale)
        all_dx = torch.zeros_like(z_e)
        for blk in bb.flow_ssm_list:
            ppg_e, dx = blk(ppg_e, z_e, cond)
            all_dx = all_dx + dx
        return bb.final_layer(all_dx, cond)

    def forward(self, z, ppg, t, h):
        return self.u(z, ppg, t, h)


# ----------------------------------------------------------------------------------------------------------------------
# (t, r) sampling — official imf.py L120-139
# ----------------------------------------------------------------------------------------------------------------------
def sample_tr(batch: int, generator: torch.Generator | None = None, p_mean: float = -0.4, p_std: float = 1.0, data_proportion: float = 0.5):
    """t, r ~ logit-normal(p_mean, p_std) i.i.d.; t = max, r = min; the first int(B * data_proportion) rows get r = t
    (plain flow-matching samples). Returns (t [B,1], r [B,1], fm_mask [B,1] bool) on CPU."""
    n1 = torch.randn(batch, 1, generator=generator)
    n2 = torch.randn(batch, 1, generator=generator)
    t = torch.sigmoid(n1 * p_std + p_mean)
    r = torch.sigmoid(n2 * p_std + p_mean)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    fm_mask = (torch.arange(batch) < int(batch * data_proportion)).reshape(batch, 1)
    r = torch.where(fm_mask, t, r)
    return t, r, fm_mask


# ----------------------------------------------------------------------------------------------------------------------
# Objective — official imf.py L347-393 without CFG / label dropout / auxiliary head
# ----------------------------------------------------------------------------------------------------------------------
def compound_V(u_fn: Callable, z_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor, v_tangent: torch.Tensor, jvp_mode: str = "forward"):
    """V = u + (t - r) * sg(du/dt) with du/dt = JVP of u_fn(z, t, r) along (v_tangent, 1, 0). Returns (u, dudt_detached, V).

    jvp_mode="forward": torch.func.jvp (forward-mode AD; verified through the S5 scan, docs/IMEANFLOW_AUDIT.md Sec. 4).
    jvp_mode="double_vjp": torch.autograd.functional.jvp (reverse-over-reverse) — verified fallback, more expensive.
    """
    tangents = (v_tangent, torch.ones_like(t), torch.zeros_like(r))
    if jvp_mode == "forward":
        u, dudt = torch.func.jvp(u_fn, (z_t, t, r), tangents)
    elif jvp_mode == "double_vjp":
        u = u_fn(z_t, t, r)
        _, dudt = torch.autograd.functional.jvp(u_fn, (z_t, t, r), tangents, create_graph=False)
    else:
        raise ValueError(jvp_mode)
    dudt = dudt.detach()  # stop-gradient on the JVP outcome (imf.py L376; iMF Eq. 12 "JVP_sg")
    V = u + (t - r).reshape(-1, 1, 1) * dudt
    return u, dudt, V


def imeanflow_loss(net: MeanFlowS5, x: torch.Tensor, ppg: torch.Tensor, e: torch.Tensor, t: torch.Tensor, r: torch.Tensor,
                   norm_p: float = 1.0, norm_eps: float = 0.01, jvp_mode: str = "forward", v_tangent: torch.Tensor | None = None):
    """iMF loss for a batch. x, e, ppg: [B,1,T]; t, r: [B,1] with r <= t.
    v_theta (JVP tangent) = u_theta(z_t, t, t) evaluated WITHOUT gradient (boundary condition; the official code takes the
    aux-head prediction, also gradient-free through the stop-gradiented JVP). Returns (loss, info)."""
    tt = t.reshape(-1, 1, 1)
    z_t = (1 - tt) * x + tt * e
    v_tgt = e - x
    if v_tangent is None:
        with torch.no_grad():
            v_tangent = net.u(z_t, ppg, t, torch.zeros_like(t))

    def u_fn(z, t_, r_):
        return net.u(z, ppg, t_, t_ - r_)

    u, dudt, V = compound_V(u_fn, z_t, t, r, v_tangent, jvp_mode)
    delta2 = ((V - v_tgt) ** 2).flatten(1).sum(1)  # per-sample sum over dims (imf.py L385)
    w = 1.0 / (delta2.detach() + norm_eps) ** norm_p  # adaptive weight, stop-gradient (MF Eq. 22; imf.py L380-382)
    loss = (delta2 * w).mean()
    # diagnostics only (A8 §11) — the loss above is unchanged; w is already stop-gradiented
    wd = w.detach()
    info = {"mse": ((V - v_tgt) ** 2).mean().detach(), "delta2_mean": delta2.mean().detach(), "u_abs_mean": u.abs().mean().detach(), "dudt_abs_mean": dudt.abs().mean().detach(), "v_tangent_abs_mean": v_tangent.abs().mean().detach(),
            "w_mean": wd.mean(), "w_std": wd.std(), "w_min": wd.min(), "w_max": wd.max(), "w_median": wd.median(),
            "w_p01": torch.quantile(wd, 0.01), "w_p10": torch.quantile(wd, 0.10), "w_p25": torch.quantile(wd, 0.25), "w_p75": torch.quantile(wd, 0.75), "w_p90": torch.quantile(wd, 0.90), "w_p99": torch.quantile(wd, 0.99),
            "w_saturation_frac": (wd - 1.0).abs().lt(1e-6).float().mean(), "w_near_lower_frac": wd.lt(1e-4).float().mean(),
            "loss_before_weighting": delta2.mean().detach(), "loss_after_weighting": loss.detach()}
    return loss, info


@torch.no_grad()
def imeanflow_mse(net: MeanFlowS5, x, ppg, e, t, r, jvp_mode: str = "forward") -> torch.Tensor:
    """Unweighted per-element MSE of V vs (e - x) — the deterministic validation metric (no adaptive weight)."""
    tt = t.reshape(-1, 1, 1)
    z_t = (1 - tt) * x + tt * e
    v_tangent = net.u(z_t, ppg, t, torch.zeros_like(t))

    def u_fn(z, t_, r_):
        return net.u(z, ppg, t_, t_ - r_)

    assert t.shape[0] == r.shape[0] == e.shape[0] == x.shape[0], "bank/batch size mismatch"
    if jvp_mode == "forward":  # forward-mode JVP needs no autograd graph (reviewed: identical V to ~1e-8, far less memory)
        _, _, V = compound_V(u_fn, z_t, t, r, v_tangent, jvp_mode)
    else:
        with torch.enable_grad():
            _, _, V = compound_V(u_fn, z_t, t, r, v_tangent, jvp_mode)
    return ((V.detach() - (e - x)) ** 2).mean()


# ----------------------------------------------------------------------------------------------------------------------
# Sampling — official imf.py L42, L90-114
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_meanflow(net: MeanFlowS5, ppg: torch.Tensor, e: torch.Tensor, n_steps: int = 1):
    """z_1 = e ~ N(0,I); t_steps = linspace(1, 0, n_steps + 1); z_r = z_t - (t - r) u(z_t, r, t). NFE = n_steps. Returns (x_hat, nfe)."""
    B = e.shape[0]
    ts = torch.linspace(1.0, 0.0, n_steps + 1)
    z = e
    nfe = 0
    for i in range(n_steps):
        t = torch.full((B, 1), float(ts[i]), device=e.device)
        r = torch.full((B, 1), float(ts[i + 1]), device=e.device)
        z = z - (t - r).reshape(-1, 1, 1) * net.u(z, ppg, t, t - r)
        nfe += 1
    return z, nfe


# ----------------------------------------------------------------------------------------------------------------------
# Fixed validation banks for the deterministic selection metric (A2 pre-registration)
# ----------------------------------------------------------------------------------------------------------------------
def make_imf_banks(n: int, T: int, n_banks: int = 4, seed: int = 1000, **tr_kwargs):
    """Bank b: (t, r) via sample_tr with generator seed+b, rows then randomly permuted (so the r = t half is a random subset of
    the temporally ordered validation windows, not always the first half), e ~ N(0,I) from the same generator."""
    banks = []
    for b in range(n_banks):
        g = torch.Generator().manual_seed(seed + b)
        t, r, _ = sample_tr(n, g, **tr_kwargs)
        perm = torch.randperm(n, generator=g)
        t, r = t[perm], r[perm]
        e = torch.randn(n, 1, T, generator=g)
        banks.append((t, r, e))
    return banks


def imf_bank_hash(banks) -> str:
    import hashlib

    h = hashlib.sha256()
    for t, r, e in banks:
        for a in (t, r, e):
            h.update(a.numpy().astype("float32").tobytes())
    return h.hexdigest()


def fixed_imf_mse(net: MeanFlowS5, x_val: torch.Tensor, y_val: torch.Tensor, banks, batch_size: int = 64, jvp_mode: str = "forward"):
    """Window-weighted unweighted MSE(V, e - x) per bank; returns (mean over banks, per-bank)."""
    device = x_val.device
    assert all(len(t_b) >= len(x_val) for t_b, _, _ in banks), "validation bank shorter than the validation set"
    per = []
    for t_b, r_b, e_b in banks:
        se, n = 0.0, 0
        for i in range(0, len(x_val), batch_size):
            ppg, ecg = x_val[i : i + batch_size].unsqueeze(1), y_val[i : i + batch_size].unsqueeze(1)
            t, r, e = t_b[i : i + batch_size].to(device), r_b[i : i + batch_size].to(device), e_b[i : i + batch_size].to(device)
            se += imeanflow_mse(net, ecg, ppg, e, t, r, jvp_mode).item() * len(ppg)
            n += len(ppg)
        per.append(se / n)
    return float(sum(per) / len(per)), per
