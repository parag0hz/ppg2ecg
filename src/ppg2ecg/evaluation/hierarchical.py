"""Hierarchical bootstrap for C2 (preregistration section 10).

The training seed is the replication unit. Outer loop: resample the training seeds with replacement.
Inner loop: within each sampled seed, subject-stratified resample of development windows with equal
subject weight. The replicate statistic is the mean over the sampled seeds of the equal-subject-weight
mean effect.

This is a REPLICATION summary over a handful of training runs. It is not population-level inference.
"""
from __future__ import annotations

import numpy as np

N_REPLICATES, RNG_SEED = 5000, 20260902


def hierarchical_bootstrap(per_seed_effects: dict, subjects: np.ndarray,
                           n_rep: int = N_REPLICATES, seed: int = RNG_SEED) -> dict:
    """`per_seed_effects`: {training_seed -> per-window oriented difference array}, all on the same windows.

    Positive values must already mean "the intervention is better"; orientation is the caller's job.
    """
    keys = sorted(per_seed_effects)
    if not keys:
        raise ValueError("no seeds supplied")
    arrs = [np.asarray(per_seed_effects[k], dtype=np.float64) for k in keys]
    n = arrs[0].size
    if any(a.size != n for a in arrs):
        raise ValueError("every seed must supply the same number of per-window effects")
    s = np.asarray(subjects)
    if s.size != n:
        raise ValueError("subjects must align with the per-window effects")
    uniq = sorted(set(s.tolist()))
    idx = {u: np.flatnonzero(s == u) for u in uniq}

    def macro(a, picks):
        return float(np.mean([np.nanmean(a[picks[u]]) for u in uniq]))

    point_per_seed = {k: macro(a, idx) for k, a in zip(keys, arrs)}
    point = float(np.mean(list(point_per_seed.values())))

    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_rep))
    for b in range(int(n_rep)):
        chosen = rng.integers(0, len(keys), size=len(keys))          # outer: seeds with replacement
        vals = []
        for j in chosen:
            picks = {u: rng.choice(idx[u], idx[u].size, replace=True) for u in uniq}   # inner: windows
            vals.append(macro(arrs[j], picks))
        draws[b] = float(np.mean(vals))
    lo, hi = (float(v) for v in np.nanpercentile(draws, [2.5, 97.5]))
    e = np.array(list(point_per_seed.values()), dtype=np.float64)
    return {"point": point, "lo": lo, "hi": hi,
            "verdict": "improves" if lo > 0 else ("worsens" if hi < 0 else "unresolved"),
            "seed_effects": point_per_seed, "n_seeds": len(keys),
            "mean": float(e.mean()), "median": float(np.median(e)),
            "sd": float(e.std(ddof=1)) if e.size > 1 else float("nan"),
            "min": float(e.min()), "max": float(e.max()),
            "n_positive": int((e > 0).sum()), "n_rep": int(n_rep), "rng_seed": int(seed)}
