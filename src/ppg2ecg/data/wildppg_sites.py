"""Deterministic reconstruction of the WildPPG test-window (subject, site) cluster labels.

The frozen evaluation `test_inputs.npz` files store only `sid` (subject), because `scripts/eval_a0_nfe_curve.py:39-48`
reads just `x`, `y`, `window_start_s` from the processed per-subject archives. The measurement *site*
(sternum / head / wrist / ankle) is present in `data/processed/wildppg_8s/<subject>.npz` under key `site`
(written by `scripts/build_processed_wildppg.py`). The evaluation concatenated subjects in manifest test order and
then applied the deterministic stride subsample `stride = ceil(len / subsample)` (`eval_a0_nfe_curve.py:103-105`,
`--subsample 4096` in `scripts/run_a4_pipeline.sh`), i.e. 24,094 + 22,790 = 46,884 -> stride 12 -> 3,907 windows.

This helper reproduces that mapping and FAILS LOUDLY (raises) if anything does not match: no silent fallback to
subject-only clustering is possible. Used by the X2 analysis (docs/X2_ENDPOINT_IDENTITY_PREREGISTRATION.md sec. 4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EXPECTED_SITES = ("ankle", "head", "sternum", "wrist")


class ClusterLabelError(RuntimeError):
    """Raised when the (subject, site) labels cannot be reconstructed exactly."""


def wildppg_test_site_labels(processed: Path, subjects, n_expected: int, starts_expected=None, subsample: int = 4096) -> np.ndarray:
    """Return the per-test-window measurement site as a [n_expected] array of str.

    processed        directory holding <subject>.npz (data/processed/wildppg_8s)
    subjects         test subjects IN MANIFEST ORDER (e.g. ["kjd", "ssx"])
    n_expected       number of frozen test windows (3907)
    starts_expected  optional `test_inputs['starts']`; when given it must match element-wise
    subsample        the evaluation's --subsample value (4096)
    """
    processed = Path(processed)
    sites, starts = [], []
    for s in subjects:
        f = processed / f"{s}.npz"
        if not f.exists():
            raise ClusterLabelError(f"missing processed archive {f}; cannot reconstruct WildPPG site labels")
        d = np.load(f)
        for key in ("site", "window_start_s", "x"):
            if key not in d:
                raise ClusterLabelError(f"{f} has no '{key}' key (keys: {list(d.keys())})")
        if len(d["site"]) != len(d["x"]) or len(d["window_start_s"]) != len(d["x"]):
            raise ClusterLabelError(f"{f}: site/window_start_s length does not match x")
        sites.append(np.asarray(d["site"]).astype(str))
        starts.append(np.asarray(d["window_start_s"]))
    site_all, start_all = np.concatenate(sites), np.concatenate(starts)
    total = len(site_all)
    stride = -(-total // subsample) if total > subsample else 1
    site_sub, start_sub = site_all[::stride], start_all[::stride]
    if len(site_sub) != n_expected:
        raise ClusterLabelError(f"reconstructed {len(site_sub)} windows (total {total}, stride {stride}) but {n_expected} were expected")
    if starts_expected is not None:
        exp = np.asarray(starts_expected)
        if len(exp) != n_expected or not np.array_equal(start_sub.astype(exp.dtype), exp):
            raise ClusterLabelError("reconstructed window_start_s does not match the frozen test_inputs['starts']")
    found = tuple(sorted(set(site_sub.tolist())))
    if found != EXPECTED_SITES:
        raise ClusterLabelError(f"unexpected site set {found}; expected {EXPECTED_SITES}")
    return site_sub


def wildppg_clusters(sid, site) -> np.ndarray:
    """Combine subject and site into "<subject>/<site>" cluster labels, asserting every cluster is non-empty."""
    sid, site = np.asarray(sid).astype(str), np.asarray(site).astype(str)
    if len(sid) != len(site):
        raise ClusterLabelError(f"sid ({len(sid)}) and site ({len(site)}) lengths differ")
    clusters = np.array([f"{a}/{b}" for a, b in zip(sid, site)])
    labels, counts = np.unique(clusters, return_counts=True)
    n_expected = len(set(sid.tolist())) * len(EXPECTED_SITES)
    if len(labels) != n_expected or counts.min() == 0:
        raise ClusterLabelError(f"expected {n_expected} non-empty (subject, site) clusters, got {dict(zip(labels.tolist(), counts.tolist()))}")
    return clusters
