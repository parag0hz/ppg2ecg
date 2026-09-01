"""C1 target-interval exposure control: the ONLY thing that may differ between C1 arms.

Frozen by docs/C1_INTERVAL_EXPOSURE_CONTROL_PREREGISTRATION.md (commit b32c952).

The intervention post-processes `sample_tr` at TRAINING TIME ONLY. It deliberately does not touch
`tr_kw = dict(p_mean, p_std, data_proportion)`, because `train_a2.py:122` builds the deterministic
checkpoint-selection banks with `make_imf_banks(..., **tr_kw)` -- changing tr_kw would change the
selection criterion itself between arms and confound the experiment.

`sample_tr`'s exact-h=0 branch is deterministic by ROW POSITION (the first int(B*data_proportion) rows),
not a Bernoulli draw. The forced-interval branch mirrors that idiom: of the remaining positive-h rows, the
SECOND HALF BY POSITION is forced. That realises the preregistered 25%/25% split exactly rather than in
expectation, and preserves the exact-h=0 probability untouched.
"""
from __future__ import annotations

import torch

from .imeanflow import sample_tr

#: frozen arm names; "B" is a no-op replay of the historical sampler
ARMS = ("B", "H25", "H50")
FORCED_H = {"B": None, "H25": 0.25, "H50": 0.50}


def sample_tr_c1(batch: int, generator: torch.Generator | None = None, arm: str = "B", **tr_kwargs):
    """`sample_tr` for arm B; for H25/H50 the second positional half of the positive-h rows is forced.

    For a forced row: t ~ Uniform[h, 1] and r = t - h, drawn from the same dedicated (t, r) generator.
    Returns (t [B,1], r [B,1], fm_mask [B,1] bool) exactly like `sample_tr`.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown C1 arm {arm!r}; expected one of {ARMS}")
    t, r, fm = sample_tr(batch, generator, **tr_kwargs)
    h = FORCED_H[arm]
    if h is None:                                   # ARM B: byte-identical to the historical sampler
        return t, r, fm
    pos = (~fm).reshape(-1)                          # positive-h rows, by construction a positional suffix
    idx = torch.nonzero(pos, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return t, r, fm
    forced = idx[idx.numel() // 2:]                  # second half of the positive rows, by position
    if forced.numel() == 0:
        return t, r, fm
    u = torch.rand(forced.numel(), 1, generator=generator)
    t_f = h + (1.0 - h) * u                          # t ~ Uniform[h, 1]
    t = t.clone(); r = r.clone()
    t[forced] = t_f
    r[forced] = t_f - h
    return t, r, fm


def exposure_stats(h: torch.Tensor) -> dict:
    """Monte-Carlo exposure summary for one sampler (preregistration section 14)."""
    x = h.reshape(-1).double()
    pos = x[x > 0]
    return {
        "p_h_eq_0": float((x == 0).double().mean()),
        "p_h_eq_0.25": float((x == 0.25).double().mean()),
        "p_h_eq_0.50": float((x == 0.50).double().mean()),
        "positive_median": float(pos.median()) if pos.numel() else float("nan"),
        "p_h_ge_0.125": float((x >= 0.125).double().mean()),
        "p_h_ge_0.25": float((x >= 0.25).double().mean()),
        "p_h_ge_0.5": float((x >= 0.5).double().mean()),
        "p_h_ge_0.7": float((x >= 0.7).double().mean()),
        "max_h": float(x.max()),
    }
