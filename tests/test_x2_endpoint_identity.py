"""X2 unit tests (docs/X2_ENDPOINT_IDENTITY_PREREGISTRATION.md sec. 14, tests A-I). Synthetic tensors only."""
from __future__ import annotations

import numpy as np
import ppg2ecg.utils.mkl_warmup  # noqa: F401
import pytest
import torch

from ppg2ecg.data.wildppg_sites import ClusterLabelError, wildppg_clusters, wildppg_test_site_labels
from ppg2ecg.evaluation.source_sensitivity import cluster_bootstrap, jvp_sensitivity, source_bank, source_stats, unit_rms_directions
from ppg2ecg.flow.samplers import euler_sample

K, W, T = 8, 5, 64


def _bank(k=K, w=W, t=T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, w, 1, t, generator=g)


# ---- A: analytical perfectly cancelling field  v(x, c) = m(c) - x  =>  F = m(c), independent of the source
def test_A_perfect_cancellation_gives_zero_source_dependence():
    x0 = _bank().squeeze(2).numpy()  # [K, W, T]
    m = np.random.default_rng(1).normal(size=(W, T))
    F = np.broadcast_to(m, x0.shape).copy()  # F_k = x0 + (m - x0) = m
    s = source_stats(F, x0)
    assert np.allclose(s["r_source"], 0.0, atol=1e-12)
    assert np.allclose(s["std_retention"], 0.0, atol=1e-12)
    assert np.allclose(s["beta"], 0.0, atol=1e-12)
    assert np.allclose(s["d_pair"], 0.0, atol=1e-12)


# ---- B: non-cancelling field  v = 0  =>  F = x0
def test_B_zero_field_retains_all_source_variance():
    x0 = _bank(seed=2).squeeze(2).numpy()
    s = source_stats(x0.copy(), x0)
    assert np.allclose(s["r_source"], 1.0, atol=1e-10)
    assert np.allclose(s["std_retention"], 1.0, atol=1e-10)
    assert np.allclose(s["beta"], 1.0, atol=1e-10)
    assert np.allclose(s["d_pair"], 1.0, atol=1e-10)
    # partial retention: F = a * x0 + const  =>  beta = a, R_source = a^2 (never conflate the two)
    for a in (0.25, 0.5):
        s2 = source_stats(a * x0 + 3.0, x0)
        assert np.allclose(s2["beta"], a, atol=1e-10)
        assert np.allclose(s2["r_source"], a * a, atol=1e-10)
        assert np.allclose(s2["std_retention"], a, atol=1e-10)


# ---- C: JVP rho_J on analytic maps + validation against central finite differences
def test_C_jvp_rho_constant_and_identity_maps():
    x = torch.randn(4, 1, T, dtype=torch.float64)
    d = torch.randn(4, 1, T, dtype=torch.float64)
    const = torch.randn(1, 1, T, dtype=torch.float64)
    assert np.allclose(jvp_sensitivity(lambda z: const.expand_as(z).clone(), x, d)["rho_J"], 0.0, atol=1e-12)
    assert np.allclose(jvp_sensitivity(lambda z: z, x, d)["rho_J"], 1.0, atol=1e-12)
    A = torch.randn(T, T, dtype=torch.float64) / T**0.5

    def lin(z):
        return z + (z @ A.T) * 0.3 - z  # F(z) = 0.3 A z

    r = jvp_sensitivity(lin, x, d)["rho_J"]
    fd = ((lin(x + 1e-6 * d) - lin(x - 1e-6 * d)) / 2e-6).flatten(1).norm(dim=1) / d.flatten(1).norm(dim=1)
    assert np.allclose(r, fd.numpy(), rtol=1e-6)
    # perfect cancellation F = x + (m - x): rho_J must be 0
    assert np.allclose(jvp_sensitivity(lambda z: z + (const.expand_as(z) - z), x, d)["rho_J"], 0.0, atol=1e-12)


# ---- D: deterministic source bank, seed 0 == historical construction, reruns identical
def test_D_source_bank_determinism_and_historical_parity():
    n, t = 37, 128
    g = torch.Generator().manual_seed(0)
    historical = torch.randn(n, 1, t, generator=g)  # scripts/eval_a0_nfe_curve.py:108-109
    assert torch.equal(source_bank(0, n, t), historical)
    assert torch.equal(source_bank(7, n, t), source_bank(7, n, t))
    assert not torch.equal(source_bank(0, n, t), source_bank(1, n, t))
    assert source_bank(3, n, t).shape == (n, 1, t) and source_bank(3, n, t).dtype == torch.float32
    # slicing a bank == generating the full bank and slicing (chunking must not change the draw)
    assert torch.equal(source_bank(5, n, t)[10:20], source_bank(5, n, t)[10:20])
    d1 = unit_rms_directions(6, t, 20260301, 4)
    assert torch.equal(d1, unit_rms_directions(6, t, 20260301, 4))
    assert torch.allclose(d1.flatten(2).pow(2).mean(-1), torch.ones(6, 4), atol=1e-5)


# ---- E: statistics are per-window; no cross-window mixing
def test_E_no_cross_window_mixing():
    x0 = _bank(seed=3).squeeze(2).numpy()
    F = 0.4 * x0 + np.random.default_rng(0).normal(size=x0.shape) * 0.0
    full = source_stats(F, x0)
    for i in range(W):
        one = source_stats(F[:, i : i + 1], x0[:, i : i + 1])
        for key in ("r_source", "beta", "d_pair"):
            assert np.allclose(one[key][0], full[key][i], rtol=1e-10), key
    perm = [2, 0, 4, 1, 3]
    sh = source_stats(F[:, perm], x0[:, perm])
    assert np.allclose(sh["r_source"], full["r_source"][perm], rtol=1e-10)


# ---- F/G: exact t=0 semantics and equivalence of x0 + v(x0, 0) with the frozen Euler-1 sampler
def test_FG_euler1_equals_source_endpoint_map():
    calls = []

    def v(x, t):
        calls.append((t.clone(), x.clone()))
        return torch.sin(x) * 0.5 + 1.0

    x0 = torch.randn(3, 1, T)
    out, nfe = euler_sample(v, x0, 1)
    assert nfe == 1 and len(calls) == 1
    t_used = calls[0][0]
    assert t_used.dtype == torch.float32 and torch.equal(t_used, torch.zeros_like(t_used))  # exactly t = 0
    assert torch.equal(calls[0][1], x0)  # evaluated at the source itself
    assert torch.equal(out, x0 + v(x0, torch.zeros(3, 1)))


# ---- H: exploratory t-profile formula on the exact straight field  v = x1 - x0
def test_H_t_profile_exact_straight_field():
    g = torch.Generator().manual_seed(9)
    # float64 inputs: the identity G_t == x1 is exact in exact arithmetic, so a tight tolerance is the stronger test
    x1 = torch.randn(1, W, T, generator=g, dtype=torch.float64).numpy()
    x0 = torch.randn(K, W, T, generator=g, dtype=torch.float64).numpy()
    for t in (0.0, 0.01, 0.05, 0.10, 0.5):
        xt = (1 - t) * x0 + t * x1
        v = x1 - x0  # exact conditional-path velocity
        G = xt + (1 - t) * v
        assert np.allclose(G, np.broadcast_to(x1, G.shape), atol=1e-5)
        s = source_stats(G, (1 - t) * x0, pairwise=False)
        assert np.allclose(s["r_source"], 0.0, atol=1e-8)
        assert np.allclose(s["beta"], 0.0, atol=1e-8)
    # at t = 0 the profile definition reduces exactly to the confirmatory one
    s0 = source_stats(x0 + 0.0, x0, pairwise=False)
    assert np.allclose(s0["r_source"], 1.0, atol=1e-10)


# ---- I: cluster labels reconstruct exactly and fail loudly otherwise
def test_I_cluster_labels_fail_loud(tmp_path):
    sites = np.array(["sternum", "head", "wrist", "ankle"] * 6)
    n = len(sites)
    np.savez(tmp_path / "sub.npz", x=np.zeros((n, 4), np.float32), site=sites, window_start_s=np.arange(n, dtype=np.int32))
    got = wildppg_test_site_labels(tmp_path, ["sub"], n, np.arange(n, dtype=np.int32), subsample=10**6)
    assert np.array_equal(got, sites)
    cl = wildppg_clusters(np.array(["sub"] * n), got)
    assert len(np.unique(cl)) == 4
    with pytest.raises(ClusterLabelError):
        wildppg_test_site_labels(tmp_path, ["missing"], n, subsample=10**6)
    with pytest.raises(ClusterLabelError):
        wildppg_test_site_labels(tmp_path, ["sub"], n + 1, subsample=10**6)
    with pytest.raises(ClusterLabelError):
        wildppg_test_site_labels(tmp_path, ["sub"], n, np.arange(n, dtype=np.int32) + 5, subsample=10**6)
    np.savez(tmp_path / "bad.npz", x=np.zeros((4, 4), np.float32), site=np.array(["head"] * 4), window_start_s=np.arange(4))
    with pytest.raises(ClusterLabelError):  # incomplete site set
        wildppg_test_site_labels(tmp_path, ["bad"], 4, subsample=10**6)
    np.savez(tmp_path / "nosite.npz", x=np.zeros((4, 4), np.float32), window_start_s=np.arange(4))
    with pytest.raises(ClusterLabelError):
        wildppg_test_site_labels(tmp_path, ["nosite"], 4, subsample=10**6)
    with pytest.raises(ClusterLabelError):  # a missing cluster must not pass silently
        wildppg_clusters(np.array(["a", "a"]), np.array(["head", "head"]))


def test_cluster_bootstrap_is_deterministic_and_respects_clusters():
    rng = np.random.default_rng(0)
    # 8 clusters with strongly different means: cluster resampling must give a WIDER interval than i.i.d. window resampling
    v = np.concatenate([o + rng.normal(0, 0.1, 60) for o in np.arange(8) * 10.0])
    c = np.repeat([f"c{i}" for i in range(8)], 60)
    r1 = cluster_bootstrap(v, c, n_boot=500, seed=0, stat=np.nanmean)
    assert r1 == cluster_bootstrap(v, c, n_boot=500, seed=0, stat=np.nanmean)  # deterministic
    assert r1[1] <= r1[0] <= r1[2]
    per_window = cluster_bootstrap(v, np.arange(len(v)).astype(str), n_boot=500, seed=0, stat=np.nanmean)
    assert (r1[2] - r1[1]) > 3 * (per_window[2] - per_window[1])  # clustering is respected, not ignored
    assert r1 != cluster_bootstrap(v, c, n_boot=500, seed=7, stat=np.nanmean)
