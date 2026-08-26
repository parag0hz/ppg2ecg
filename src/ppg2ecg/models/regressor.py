"""S5ConditionalMeanRegressor — deterministic MSE regression control on the SAME PENGUIN Flow-SSM/S5 backbone (A5).

Construction (docs/A5_CONDITIONAL_MEAN_CONTROL_PREREGISTRATION.md §3): instantiate the unmodified upstream PENGUIN model,
then DELETE the two generative-input modules — `pre_conv_target` (stem of the noisy state x_t) and `timestep_embedder`
(time conditioning) — and run the remaining modules with x_t-embedding = 0 and conditioning vector = 0:
    PPG --pre_conv_ppg--> Flow-SSM blocks (PPG stream + target stream with zero input, adaLN driven by its biases only)
        --sum of block outputs--> final_layer(cond = 0) --> ECG prediction
No module is added or widened; parameter count = PENGUIN − 578,048 (= 4,568,707 − 528,640 − 49,408 = 3,990,659, of which
264,192 are the never-called `cross_attn` weights inherited from upstream). Trained with plain MSE, it is an
**MSE conditional-mean proxy** (the squared-error-optimal deterministic predictor is E[ECG | PPG]); we never claim the network
attains the exact conditional expectation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ppg2ecg.models import build_penguin_backbone

REMOVED_MODULES = ("pre_conv_target", "timestep_embedder")


class S5ConditionalMeanRegressor(nn.Module):
    def __init__(self, sample_rate: int = 128, h_dim: int = 128, ssm_block_num: int = 4, ssm_ratio: float = 2.0, mlp_ratio: float = 2.0):
        super().__init__()
        bb = build_penguin_backbone(n_step=1, sample_rate=sample_rate, h_dim=h_dim, ssm_block_num=ssm_block_num, ssm_ratio=ssm_ratio, mlp_ratio=mlp_ratio)
        for name in REMOVED_MODULES:  # generative inputs removed: no x_t stem, no time embedding
            delattr(bb, name)
        self.backbone = bb
        self.h_dim = h_dim

    def forward(self, ppg: torch.Tensor) -> torch.Tensor:
        """ppg [B, 1, T] -> ECG prediction [B, 1, T]. Mirrors upstream forward_step with x_t_emb = 0 and cond = 0."""
        bb = self.backbone
        ppg_e = bb.pre_conv_ppg(ppg)
        z_e = torch.zeros_like(ppg_e)
        cond = ppg_e.new_zeros(ppg_e.shape[0], self.h_dim)
        all_dx = torch.zeros_like(z_e)
        for blk in bb.flow_ssm_list:
            ppg_e, dx = blk(ppg_e, z_e, cond)
            all_dx = all_dx + dx
        return bb.final_layer(all_dx, cond)


def count_regressor_params(model: S5ConditionalMeanRegressor) -> dict:
    total = sum(p.numel() for p in model.parameters())
    dead = sum(p.numel() for n, p in model.named_parameters() if "cross_attn" in n or n.endswith("revin"))
    return {"total": total, "dead_cross_attn_revin": dead, "effective": total - dead}
