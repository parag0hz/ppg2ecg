"""X2 source-sensitivity diagnostics for one-step conditional generative maps.

Frozen definitions: docs/X2_ENDPOINT_IDENTITY_PREREGISTRATION.md sections 7, 8 and 12. All quantities are dimensionless
ratios that equal 0 for a map that is constant in the source latent and 1 for the identity map.

    R_source(i)  = mean_tau Var_k[F_k(i, tau)] / mean_tau Var_k[S_k(i, tau)]      (unbiased variance, ddof = 1)
    std_ret(i)   = sqrt(R_source(i))                                              (amplitude scale; NEVER conflated with R_source)
    D_pair(i)    = mean_{a<b} RMSE_tau(F_a, F_b) / mean_{a<b} RMSE_tau(S_a, S_b)
    beta(i)      = sum_k <F_k - Fbar, S_k - Sbar> / sum_k ||S_k - Sbar||^2
    rho_J        = ||J_x F . d|| / ||d||                                          (forward-mode JVP, unit-RMS directions)

`S` is the source perturbation that actually enters the map: S = x0 at t = 0, and S = (1 - t) x0 for the exploratory
oracle path-state profile. These are ORACLE diagnostics only when the state is built from the ground-truth target.
"""
from __future__ import annotations

import numpy as np
import torch


def source_stats(F: np.ndarray, S: np.ndarray, pairwise: bool = True) -> dict:
    """Per-window source-sensitivity statistics. F, S: [K, W, T] float arrays (endpoints and source perturbations)."""
    F = np.asarray(F, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if F.shape != S.shape or F.ndim != 3:
        raise ValueError(f"F {F.shape} and S {S.shape} must both be [K, W, T]")
    K = F.shape[0]
    if K < 2:
        raise ValueError("need at least 2 source samples")
    Fb, Sb = F.mean(0), S.mean(0)  # [W, T]
    dF, dS = F - Fb, S - Sb
    v_end = (dF**2).sum(0).mean(1) / (K - 1)  # [W]
    v_src = (dS**2).sum(0).mean(1) / (K - 1)
    r_source = v_end / np.maximum(v_src, 1e-30)
    num = (dF * dS).sum(0).sum(1)  # [W]
    den = (dS**2).sum(0).sum(1)
    beta = num / np.maximum(den, 1e-30)
    out = {"v_endpoint": v_end, "v_source": v_src, "r_source": r_source, "std_retention": np.sqrt(np.maximum(r_source, 0.0)), "beta": beta, "n_sources": K}
    if pairwise:
        W = F.shape[1]
        sf = np.zeros(W)
        ss = np.zeros(W)
        npairs = 0
        for a in range(K):
            for b in range(a + 1, K):
                sf += np.sqrt(((F[a] - F[b]) ** 2).mean(1))
                ss += np.sqrt(((S[a] - S[b]) ** 2).mean(1))
                npairs += 1
        out["pair_rmse_endpoint"] = sf / npairs
        out["pair_rmse_source"] = ss / npairs
        out["d_pair"] = (sf / npairs) / np.maximum(ss / npairs, 1e-30)
        out["n_pairs"] = npairs
    return out


@torch.no_grad()
def jvp_sensitivity(map_fn, x: torch.Tensor, d: torch.Tensor) -> dict:
    """rho_J = ||J.d|| / ||d|| for map_fn at x along d (forward-mode JVP; no autograd graph).

    map_fn must be the FULL endpoint map (e.g. x -> x + v_theta(x, c, 0)), so that J is J_x F directly.
    Returns per-sample rho_J and the secondary cosine between the velocity response (J.d - d) and -d.
    """
    y, Jd = torch.func.jvp(map_fn, (x,), (d,))
    nd = d.flatten(1).norm(dim=1)
    rho = Jd.flatten(1).norm(dim=1) / nd
    resp = (Jd - d).flatten(1)  # J_x v . d when map_fn = x + v(x)
    cos = (resp * (-d.flatten(1))).sum(1) / (resp.norm(dim=1).clamp_min(1e-30) * nd)
    return {"rho_J": rho.double().cpu().numpy(), "cos_resp_negd": cos.double().cpu().numpy(), "out_norm": y.flatten(1).norm(dim=1).double().cpu().numpy()}


def unit_rms_directions(n: int, T: int, seed: int, n_dir: int) -> torch.Tensor:
    """[n, n_dir, 1, T] Gaussian directions normalised to unit RMS, drawn window-major then direction-major."""
    g = torch.Generator().manual_seed(int(seed))
    d = torch.randn(n * n_dir, 1, T, generator=g)
    d = d / d.flatten(1).pow(2).mean(1).sqrt().reshape(-1, 1, 1)
    return d.reshape(n, n_dir, 1, T)


def source_bank(seed: int, n: int, T: int) -> torch.Tensor:
    """The frozen X2 source bank for one seed: identical construction to the historical evaluation draw
    (`scripts/eval_a0_nfe_curve.py:108-109`), so seed 0 reproduces the frozen paired noise exactly."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(int(n), 1, int(T), generator=g)


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0, stat=np.nanmedian):
    """Bootstrap `stat` by resampling CLUSTERS with replacement (windows within a cluster move together)."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    labs = np.unique(clusters)
    groups = [values[clusters == c] for c in labs]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(labs), len(labs))
        draws[b] = stat(np.concatenate([groups[i] for i in pick]))
    return float(stat(values)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-window RMSE over the time axis. a, b: [W, T]."""
    return np.sqrt(((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2).mean(-1))


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-window Pearson correlation over the time axis; NaN when either window is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    za, zb = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    den = np.sqrt((za**2).sum(-1) * (zb**2).sum(-1))
    out = np.full(len(a), np.nan)
    m = den > 1e-20
    out[m] = (za * zb).sum(-1)[m] / den[m]
    return out
