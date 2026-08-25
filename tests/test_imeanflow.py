"""Unit tests for the Improved MeanFlow implementation (docs/A2_IMEANFLOW_PREREGISTRATION.md Sec. 3; docs/IMEANFLOW_AUDIT.md)."""
import importlib.util

import pytest
import torch

from ppg2ecg.flow.imeanflow import MeanFlowS5, compound_V, imeanflow_loss, imf_bank_hash, make_imf_banks, sample_meanflow, sample_tr
from ppg2ecg.models import build_penguin_backbone


def tiny_backbone(seed=0):
    torch.manual_seed(seed)
    bb = build_penguin_backbone(n_step=1, sample_rate=128, h_dim=16, ssm_block_num=2, ssm_ratio=2.0, mlp_ratio=2.0)
    with torch.no_grad():  # break adaLN-zero init so that outputs / derivatives are non-trivial
        for n, p in bb.named_parameters():
            if "final_layer.linear" in n or "adaLN_modulation" in n:
                p.add_(0.05 * torch.randn_like(p))
    return bb


class AnalyticLinearField(torch.nn.Module):
    """v(z, t) = a z + c (linear, time-independent). Exact average velocity over [r, t] with h = t - r:
    z_r = exp(-a h)(z_t + c/a) - c/a  =>  u(z_t, r, t) = (z_t - z_r)/h = (z_t + c/a) * (1 - exp(-a h)) / h ; u -> a z + c as h -> 0."""

    def __init__(self, a=0.7, c=0.3):
        super().__init__()
        self.a, self.c = a, c

    def u(self, z, ppg, t, h):
        hh = h.reshape(-1, 1, 1).expand_as(z)
        exact = (z + self.c / self.a) * (-torch.expm1(-self.a * hh)) / torch.where(hh == 0, torch.ones_like(hh), hh)
        return torch.where(hh == 0, self.a * z + self.c, exact)


@pytest.fixture
def analytic():
    return AnalyticLinearField()


def test_analytic_meanflow_identity(analytic):
    """V = u + (t-r) d/dt u must equal the instantaneous velocity v(z_t) = a z_t + c when the tangent is the true v."""
    torch.manual_seed(0)
    B, T = 6, 32
    z = torch.randn(B, 1, T, dtype=torch.float64)
    t = torch.rand(B, 1, dtype=torch.float64) * 0.5 + 0.5
    r = t - torch.rand(B, 1, dtype=torch.float64) * 0.4  # 0 < t - r <= 0.4
    v_true = analytic.a * z + analytic.c
    u, dudt, V = compound_V(lambda z_, t_, r_: analytic.u(z_, None, t_, t_ - r_), z, t, r, v_true)
    assert torch.allclose(V, v_true, atol=1e-9), float((V - v_true).abs().max())
    assert torch.allclose(analytic.u(z, None, t, torch.zeros_like(t)), v_true)  # boundary u(z,t,t) = v


def test_analytic_loss_is_zero_for_consistent_pairs(analytic):
    """If e - x equals the true velocity at z_t, the iMF loss of the exact average-velocity field is 0 (V == e - x)."""
    torch.manual_seed(1)
    B, T = 4, 16
    a, c = analytic.a, analytic.c
    x = torch.randn(B, 1, T, dtype=torch.float64)
    t, r, _ = sample_tr(B, torch.Generator().manual_seed(3))
    t, r = t.double(), r.double()
    tt = t.reshape(-1, 1, 1)
    e = (x * (1 + a * (1 - tt)) + c) / (1 - a * tt)  # solves e - x = a z_t + c with z_t = (1-t)x + t e
    loss, info = imeanflow_loss(analytic, x, None, e, t, r)
    assert float(loss) < 1e-10 and float(info["mse"]) < 1e-12


def test_shapes_and_conditioning():
    net = MeanFlowS5(tiny_backbone())
    B, T = 3, 256
    z, ppg = torch.randn(B, 1, T), torch.randn(B, 1, T)
    t, h = torch.rand(B, 1), torch.rand(B, 1)
    u = net.u(z, ppg, t, h)
    assert u.shape == (B, 1, T)
    ppg2 = ppg.clone()
    ppg2[1] += 1.0
    u2 = net.u(z, ppg2, t, h)
    assert not torch.allclose(u[1], u2[1])  # PPG conditions the output ...
    assert torch.allclose(u[0], u2[0]) and torch.allclose(u[2], u2[2])  # ... and batch elements are independent


def test_backbone_parity_t_only():
    """cond_mode='t_only' must reproduce the upstream forward_step bit-exactly (same modules, same arithmetic)."""
    bb = tiny_backbone()
    net = MeanFlowS5(bb, cond_mode="t_only")
    z, ppg, t = torch.randn(2, 1, 256), torch.randn(2, 1, 256), torch.rand(2, 1)
    assert torch.equal(net.u(z, ppg, t, torch.rand(2, 1)), bb.forward_step(z, ppg, t))


def test_jvp_matches_finite_differences_on_backbone():
    net = MeanFlowS5(tiny_backbone())
    torch.manual_seed(2)
    B, T = 2, 256
    z, ppg = torch.randn(B, 1, T), torch.randn(B, 1, T)
    t = torch.rand(B, 1) * 0.5 + 0.5
    r = t - torch.rand(B, 1) * 0.3
    v = torch.randn_like(z)

    def u_fn(z_, t_, r_):
        return net.u(z_, ppg, t_, t_ - r_)

    _, dudt, _ = compound_V(u_fn, z, t, r, v)
    eps = 0.01
    with torch.no_grad():
        fd = (u_fn(z + eps * v, t + eps, r) - u_fn(z - eps * v, t - eps, r)) / (2 * eps)
    assert float((dudt - fd).abs().max() / (fd.abs().max() + 1e-8)) < 5e-3
    _, dudt2, _ = compound_V(u_fn, z, t, r, v, jvp_mode="double_vjp")
    assert torch.allclose(dudt, dudt2, atol=1e-5)


def test_stop_gradient_on_jvp_term():
    net = MeanFlowS5(tiny_backbone())
    torch.manual_seed(4)
    B, T = 2, 128
    x, ppg, e = torch.randn(B, 1, T), torch.randn(B, 1, T), torch.randn(B, 1, T)
    t, r, _ = sample_tr(B, torch.Generator().manual_seed(0), data_proportion=0.0)
    loss, _ = imeanflow_loss(net, x, ppg, e, t, r)
    loss.backward()
    g_ref = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    net.zero_grad()
    # manual: same loss with the JVP term as a constant => gradients must be identical
    tt = t.reshape(-1, 1, 1)
    z_t = (1 - tt) * x + tt * e
    with torch.no_grad():
        v_th = net.u(z_t, ppg, t, torch.zeros_like(t))
    _, dudt, _ = compound_V(lambda z_, t_, r_: net.u(z_, ppg, t_, t_ - r_), z_t, t, r, v_th)
    assert not dudt.requires_grad
    u = net.u(z_t, ppg, t, t - r)
    V = u + tt * 0 + (t - r).reshape(-1, 1, 1) * dudt
    d2 = ((V - (e - x)) ** 2).flatten(1).sum(1)
    (d2 / (d2.detach() + 0.01)).mean().backward()
    g_man = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    assert len(g_ref) == len(g_man) and all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g_ref, g_man))


def test_finite_loss_and_grads():
    net = MeanFlowS5(tiny_backbone())
    B, T = 4, 256
    x, ppg, e = torch.randn(B, 1, T), torch.randn(B, 1, T), torch.randn(B, 1, T)
    t, r, fm = sample_tr(B, torch.Generator().manual_seed(1))
    assert fm[:2].all() and (r[:2] == t[:2]).all() and (r <= t).all()
    loss, info = imeanflow_loss(net, x, ppg, e, t, r)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in info.values())
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_deterministic_with_seed():
    a = sample_tr(8, torch.Generator().manual_seed(7))
    b = sample_tr(8, torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    banks1, banks2 = make_imf_banks(5, 16, 2, seed=11), make_imf_banks(5, 16, 2, seed=11)
    assert imf_bank_hash(banks1) == imf_bank_hash(banks2) != imf_bank_hash(make_imf_banks(5, 16, 2, seed=12))
    net = MeanFlowS5(tiny_backbone())
    x, ppg, e = torch.randn(2, 1, 64), torch.randn(2, 1, 64), torch.randn(2, 1, 64)
    l1, _ = imeanflow_loss(net, x, ppg, e, a[0][:2], a[1][:2])
    l2, _ = imeanflow_loss(net, x, ppg, e, b[0][:2], b[1][:2])
    assert torch.equal(l1, l2)


def test_bank_rows_are_permuted():
    """The r = t half of a validation bank must not be the first half of the (temporally ordered) validation set."""
    t, r, e = make_imf_banks(200, 8, 1, seed=1000)[0]
    fm = (t == r).reshape(-1)
    assert fm.sum() == 100 and not fm[:100].all() and e.shape == (200, 1, 8)


def test_one_nfe_inference():
    net = MeanFlowS5(tiny_backbone())
    calls = []
    orig = net.u

    def counted(z, ppg, t, h):
        calls.append((float(t[0, 0]), float(h[0, 0])))
        return orig(z, ppg, t, h)

    net.u = counted
    ppg, e = torch.randn(2, 1, 128), torch.randn(2, 1, 128)
    x_hat, nfe = sample_meanflow(net, ppg, e, n_steps=1)
    assert nfe == 1 and len(calls) == 1 and calls[0] == (1.0, 1.0)  # u(z_1, r=0, t=1), h = 1
    with torch.no_grad():
        ref = e - orig(e, ppg, torch.ones(2, 1), torch.ones(2, 1))
    assert torch.allclose(x_hat, ref)
    x2, nfe2 = sample_meanflow(net, ppg, e, n_steps=2)
    assert nfe2 == 2 and x2.shape == e.shape


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")
def test_parity_with_official_jax_objective():
    """Port of the official iMF forward (external/iMeanFlow/imf.py L347-393; no CFG, no label dropout, boundary v instead of the
    aux head) in JAX, evaluated on a tiny per-timestep MLP with the SAME weights as a torch copy. loss, V AND parameter
    gradients must agree (gradient parity pins the stop-gradient placement to the official code). float64 on both sides."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    torch.manual_seed(0)
    B, T, H = 4, 24, 8
    dt = torch.float64
    W1 = (torch.randn(4, H) * 0.5).to(dt)
    b1 = (torch.randn(H) * 0.1).to(dt)
    W2 = (torch.randn(H, 1) * 0.5).to(dt)
    b2 = (torch.randn(1) * 0.1).to(dt)

    class TorchNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W1, self.b1, self.W2, self.b2 = (torch.nn.Parameter(w.clone()) for w in (W1, b1, W2, b2))

        def u(self, z, ppg, t, h):
            feats = torch.stack([z[:, 0], ppg[:, 0], t.expand(-1, z.shape[-1]), h.expand(-1, z.shape[-1])], -1)  # [B,T,4]
            return (torch.tanh(feats @ self.W1 + self.b1) @ self.W2 + self.b2).transpose(1, 2)

    x, ppg, e = torch.randn(B, 1, T, dtype=dt), torch.randn(B, 1, T, dtype=dt), torch.randn(B, 1, T, dtype=dt)
    t, r, _ = sample_tr(B, torch.Generator().manual_seed(5))
    t, r = t.to(dt), r.to(dt)
    net = TorchNet()
    loss_t, info = imeanflow_loss(net, x, ppg, e, t, r)
    loss_t.backward()
    grads_t = [p.grad.detach().numpy().copy() for p in (net.W1, net.b1, net.W2, net.b2)]

    # --- JAX port (official semantics) ---
    jW1, jb1, jW2, jb2 = (jnp.asarray(w.numpy()) for w in (W1, b1, W2, b2))
    jx, jppg, je = (jnp.asarray(a.numpy()) for a in (x, ppg, e))
    jt, jr = jnp.asarray(t.numpy()), jnp.asarray(r.numpy())

    def loss_fn(params):
        W1_, b1_, W2_, b2_ = params

        def jnet(z, tt, hh):  # z [B,1,T], tt/hh [B,1]
            feats = jnp.stack([z[:, 0], jppg[:, 0], jnp.broadcast_to(tt, (B, T)), jnp.broadcast_to(hh, (B, T))], -1)
            return jnp.swapaxes(jnp.tanh(feats @ W1_ + b1_) @ W2_ + b2_, 1, 2)

        tt3 = jt.reshape(-1, 1, 1)
        z_t = (1 - tt3) * jx + tt3 * je  # imf.py L350
        v_t = je - jx  # L351
        v_c = jax.lax.stop_gradient(jnet(z_t, jt, jnp.zeros_like(jt)))  # boundary v_theta (Alg. 1: v = fn(z, t, t))

        def u_fn(z_, t_, r_):
            return jnet(z_, t_, t_ - r_)  # L366-367

        u, du_dt = jax.jvp(u_fn, (z_t, jt, jr), (v_c, jnp.ones_like(jt), jnp.zeros_like(jt)))  # L369-373
        V = u + (jt - jr).reshape(-1, 1, 1) * jax.lax.stop_gradient(du_dt)  # L376
        loss_u = jnp.sum((V - v_t) ** 2, axis=(1, 2))  # L385
        return jnp.mean(loss_u / jax.lax.stop_gradient((loss_u + 0.01) ** 1.0)), V  # L380-386, L393

    (loss_j, V), grads_j = jax.value_and_grad(loss_fn, has_aux=True)((jW1, jb1, jW2, jb2))
    assert abs(float(loss_j) - float(loss_t)) < 1e-9, (float(loss_j), float(loss_t))
    for gj, gt in zip(grads_j, grads_t):  # gradient parity (stop-gradient placement) against the official semantics
        assert np.abs(np.asarray(gj) - gt).max() < 1e-9
    # V parity (recompute torch V explicitly)
    tt = t.reshape(-1, 1, 1)
    zt_t = (1 - tt) * x + tt * e
    with torch.no_grad():
        vth = net.u(zt_t, ppg, t, torch.zeros_like(t))
    _, _, V_t = compound_V(lambda z_, t_, r_: net.u(z_, ppg, t_, t_ - r_), zt_t, t, r, vth)
    assert np.abs(np.asarray(V) - V_t.detach().numpy()).max() < 1e-9
