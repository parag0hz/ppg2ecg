"""Exact patient-cluster bootstrap of the KANFlow FD (>= 3000 windows: Gaussian Frechet distance on the raw flattened
waveforms, covariance with ddof = 1 plus 1e-4 I) from per-patient sufficient statistics.

A replicate draws patients with replacement; every window of a drawn patient enters as often as the patient is drawn,
exactly as `bf0_run.patient_bootstrap_indices`. Mean and covariance of the replicate are rebuilt from per-patient sums
(n_p, sum_p x, sum_p x x^T), and tr sqrtm(S_a S_r) is computed as sum sqrt(eig(S_r^1/2 S_a S_r^1/2)). The point estimate
always comes from `paper_metrics.kanflow_fd`; one replicate is re-checked against the official function at run time.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ppg2ecg.evaluation import paper_metrics as PMX

EPS = PMX.FD_EPS
_G: dict = {}


def patient_stats(X, pid, subs):
    X = np.asarray(X, np.float64)
    k = np.searchsorted(subs, np.asarray(pid))
    P, D = subs.size, X.shape[1]
    n = np.bincount(k, minlength=P).astype(np.float64)
    S = np.zeros((P, D))
    Q = np.zeros((P, D, D))
    order = np.argsort(k, kind="stable")
    bounds = np.searchsorted(k[order], np.arange(P + 1))
    for p in range(P):
        rows = X[order[bounds[p]:bounds[p + 1]]]
        S[p] = rows.sum(axis=0)
        Q[p] = rows.T @ rows
    return n, S, Q


def moments(w, st):
    n, S, Q = st
    N = float(w @ n)
    mu = (w @ S) / N
    cov = (np.tensordot(w, Q, axes=1) - N * np.outer(mu, mu)) / (N - 1.0)
    return mu, cov + EPS * np.eye(mu.size)


def sym_sqrt(S):
    lam, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(lam, 0.0, None))) @ V.T


def frechet_to_ref(mu_a, S_a, mu_r, S_r, R):
    """FD(a, r) with R = S_r^1/2 precomputed."""
    lam = np.linalg.eigvalsh(R @ S_a @ R)
    d = mu_a - mu_r
    return float(d @ d + np.trace(S_a) + np.trace(S_r) - 2.0 * np.sqrt(np.clip(lam, 0.0, None)).sum())


def _init(arm_stats, ref_stats):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    _G["arms"], _G["ref"] = arm_stats, ref_stats


def _draw(w):
    mu_r, S_r = moments(w, _G["ref"])
    R = sym_sqrt(S_r)
    return {k: frechet_to_ref(*moments(w, st), mu_r, S_r, R) for k, st in _G["arms"].items()}


def weights(resamples, P):
    return [np.bincount(r, minlength=P).astype(np.float64) for r in resamples]


def fd_bootstrap(arms: dict, ref, pid, pairs, resamples, workers: int = 16, check: bool = True) -> dict:
    """arms: name -> [N, D] generated sets (window-aligned with `ref`); pairs: [(a, b), ...] -> FD(a) - FD(b).
    Returns {"a-b": [official point difference, 2.5 %, 97.5 %]} plus per-arm official FD under "fd"."""
    pid = np.asarray(pid)
    subs = np.unique(pid)
    ref_st = patient_stats(ref, pid, subs)
    arm_st = {k: patient_stats(v, pid, subs) for k, v in arms.items()}
    W = weights(resamples, subs.size)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(arm_st, ref_st)) as pool:
        draws = list(pool.map(_draw, W, chunksize=8))
    fd = {k: float(PMX.kanflow_fd(v, ref)) for k, v in arms.items()}
    if check:
        rows = [np.flatnonzero(pid == s) for s in subs]
        idx = np.concatenate([rows[k] for k in resamples[0]])
        a = next(iter(arms))
        off = float(PMX.kanflow_fd(np.asarray(arms[a])[idx], np.asarray(ref)[idx]))
        if not abs(off - draws[0][a]) <= 1e-6 * max(1.0, abs(off)):
            raise RuntimeError(f"fast FD bootstrap disagrees with kanflow_fd: {draws[0][a]} vs {off}")
        full = _draw_full(arm_st, ref_st, subs.size)
        for k in arms:
            if not abs(full[k] - fd[k]) <= 1e-6 * max(1.0, abs(fd[k])):
                raise RuntimeError(f"fast FD point disagrees with kanflow_fd for {k}: {full[k]} vs {fd[k]}")
    out = {"fd": fd}
    for a, b in pairs:
        d = np.array([x[a] - x[b] for x in draws])
        out[f"{a}-{b}"] = [fd[a] - fd[b], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out


def _draw_full(arm_st, ref_st, P):
    _G["arms"], _G["ref"] = arm_st, ref_st
    return _draw(np.ones(P))
