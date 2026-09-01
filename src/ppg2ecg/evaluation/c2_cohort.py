"""C2 frozen visual-atlas cohort (preregistration section 14). METADATA ONLY.

Selection uses the subject, the PPG site and the window index. It never touches model output, error,
F1, morphology or any measure of visual quality, and it is defined before any C2 prediction is loaded.
"""
from __future__ import annotations

import hashlib

import numpy as np

ATLAS_SALT = "c2-visual-atlas-v1"
SITES = ("sternum", "head", "wrist", "ankle")
PER_STRATUM = 8


def _rank_key(subject: str, site: str, window_index: int) -> str:
    return hashlib.sha256(f"{ATLAS_SALT}|{subject}|{site}|{int(window_index)}".encode()).hexdigest()


def atlas_cohort(subject: str, sites: np.ndarray, window_index: np.ndarray,
                 per_stratum: int = PER_STRATUM) -> dict[str, np.ndarray]:
    """Smallest-`per_stratum` SHA256 ranks within each (subject, site) stratum.

    `sites` and `window_index` are the metadata arrays of the ALREADY-FROZEN evaluation subset, so the
    cohort is a subset of the C0/C1 evaluation population and introduces no new window selection.
    """
    out = {}
    for site in SITES:
        m = np.flatnonzero(np.asarray(sites) == site)
        if m.size == 0:
            out[site] = np.zeros(0, dtype=np.int64)
            continue
        keys = [_rank_key(subject, site, int(window_index[i])) for i in m]
        out[site] = np.sort(m[np.argsort(keys, kind="stable")[:per_stratum]])
    return out
