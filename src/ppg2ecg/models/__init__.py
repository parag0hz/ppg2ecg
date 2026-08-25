"""Backbones. v0 policy: the ONLY backbone is upstream PENGUIN's Flow-SSM (S5), imported unmodified.

No new architecture (KAN / Mamba / attention / new losses) is added until docs/PREREGISTRATION_V0.md H1 is confirmed.
"""
from __future__ import annotations

from ppg2ecg.utils.upstream import import_upstream_penguin

# Shipped PPG-DaLiA config (external/PENGUIN/config/model.yaml + preprocess.yaml), see docs/PENGUIN_AUDIT.md
PENGUIN_DALIA_CFG = dict(n_step=25, sample_rate=128, h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0)


def build_penguin_backbone(**overrides):
    """Instantiate upstream `PENGUIN(...)` with the shipped PPG-DaLiA hyper-parameters (+overrides)."""
    PENGUIN = import_upstream_penguin()
    cfg = {**PENGUIN_DALIA_CFG, **overrides}
    return PENGUIN(**cfg)


def count_params(model, exclude_prefixes: tuple[str, ...] = ()) -> dict:
    total = sum(p.numel() for p in model.parameters())
    dead = sum(p.numel() for n, p in model.named_parameters() if any(k in n for k in exclude_prefixes))
    return {"total": total, "excluded": dead, "effective": total - dead}
