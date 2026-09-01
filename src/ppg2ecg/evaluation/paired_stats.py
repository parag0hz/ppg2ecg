"""Paired subject-stratified bootstrap for arm-vs-arm comparison on a shared window population.

Frozen by docs/C0_IMF_COMPRESSION_TARGET_PREREGISTRATION.md section 7. Every arm sees the same windows,
the same PPG, the same ground truth and the same Gaussian source, so comparisons are PAIRED: window
indices are resampled within subject and the SAME resampled index set is applied to both arms before the
difference is taken. Resampling the two arms independently would discard the pairing and inflate the
interval.
"""
from __future__ import annotations

import numpy as np

BOOT_N, BOOT_SEED = 2000, 20260901


def paired_subject_bootstrap(earlier, later, subjects, orient: str,
                             n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Oriented paired difference with an equal-subject-weight bootstrap CI.

    `orient` fixes the sign so that POSITIVE always means the later arm is better:
      "higher_better" -> later - earlier      (raw_corr, event F1 excess)
      "lower_better"  -> earlier - later      (deviations from 1, RMSEs)
    """
    if orient not in ("higher_better", "lower_better"):
        raise ValueError(f"unknown orientation {orient!r}")
    a = np.asarray(earlier, dtype=np.float64)
    b = np.asarray(later, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arms must have the same shape; got {a.shape} and {b.shape}")
    s = np.asarray(subjects)
    d = (b - a) if orient == "higher_better" else (a - b)

    uniq = sorted(set(s.tolist()))
    idx = {u: np.flatnonzero(s == u) for u in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot))
    for k in range(int(n_boot)):
        # one resampled index set per subject, applied to the SAME per-window differences
        draws[k] = float(np.mean([np.nanmean(d[rng.choice(idx[u], idx[u].size, replace=True)]) for u in uniq]))
    point = float(np.mean([np.nanmean(d[idx[u]]) for u in uniq]))
    lo, hi = (float(v) for v in np.nanpercentile(draws, [2.5, 97.5]))
    return {"point": point, "lo": lo, "hi": hi, "orient": orient,
            "verdict": "improves" if lo > 0 else ("worsens" if hi < 0 else "unresolved"),
            "n_pairs": int(d.size), "n_boot": int(n_boot), "seed": int(seed)}
