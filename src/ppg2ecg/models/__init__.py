"""Backbones. v0 policy: the ONLY backbone is upstream PENGUIN's Flow-SSM (S5), imported unmodified.

No new architecture (KAN / Mamba / attention / new losses) is added until docs/PREREGISTRATION_V0.md H1 is confirmed.
"""
from __future__ import annotations

from ppg2ecg.utils.upstream import import_upstream_penguin

# Shipped PPG-DaLiA config (external/PENGUIN/config/model.yaml + preprocess.yaml), see docs/PENGUIN_AUDIT.md
PENGUIN_DALIA_CFG = dict(n_step=25, sample_rate=128, h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0)


def build_penguin_backbone(**overrides):
    """Instantiate upstream `PENGUIN(...)` with the shipped PPG-DaLiA hyper-parameters (+overrides).

    BB1: `arch="attn"` (with optional `attn_heads`) swaps every S5 mixer for self-attention, see attn_backbone.py.
    KN1: `arch="kan"` (with `kan_blocks` = "outer" | "all") swaps the FFN MLPs for RBF-KAN layers, see kan_backbone.py.
    Without `arch` (every historical checkpoint) the call is unchanged."""
    PENGUIN = import_upstream_penguin()
    arch = overrides.pop("arch", "s5")
    heads = overrides.pop("attn_heads", 4)
    kan_blocks = overrides.pop("kan_blocks", "outer")
    cfg = {**PENGUIN_DALIA_CFG, **overrides}
    if arch == "attn":
        from ppg2ecg.models.attn_backbone import build_attn_backbone
        return build_attn_backbone(PENGUIN, cfg, heads)
    if arch == "kan":
        from ppg2ecg.models.kan_backbone import build_kan_backbone
        return build_kan_backbone(PENGUIN, cfg, kan_blocks)
    assert arch == "s5", arch
    return PENGUIN(**cfg)


def count_params(model, exclude_prefixes: tuple[str, ...] = ()) -> dict:
    total = sum(p.numel() for p in model.parameters())
    dead = sum(p.numel() for n, p in model.named_parameters() if any(k in n for k in exclude_prefixes))
    return {"total": total, "excluded": dead, "effective": total - dead}
