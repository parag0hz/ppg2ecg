"""S5ConditionalMeanRegressor — deterministic MSE regression control on the SAME PENGUIN Flow-SSM/S5 backbone (A5).

Construction (docs/A5_CONDITIONAL_MEAN_CONTROL_PREREGISTRATION.md §3): instantiate the unmodified upstream PENGUIN model,
then DELETE the two generative-input modules — `pre_conv_target` (stem of the noisy state x_t) and `timestep_embedder`
(time conditioning) — and run the remaining modules with the target-stream input replaced by a LEARNED CONSTANT state token
(information-free, the deterministic analogue of the 1-NFE noise input) and conditioning vector = 0:
    PPG --pre_conv_ppg--> Flow-SSM blocks (PPG stream + target stream fed with the constant token, adaLN driven by its biases)
        --sum of block outputs--> final_layer(cond = 0) --> ECG prediction
Amendment 1 (prereg §16): the originally pre-registered zero target-stream input cannot train at all — with x_t-embedding = 0
the block outputs are identically 0, `final_layer.linear.weight` is zero-initialised upstream, and only `final_layer.linear.bias`
ever receives gradient (permanent dead-start; A5a converged to a constant). The token (h_dim = 128 parameters, N(0, 0.02²)
init under the run seed) is the minimal change that restores gradient flow; nothing else is added or widened.
Parameter count = 4,568,707 − 528,640 − 49,408 + 128 = 3,990,787 (264,192 of which are upstream's never-called `cross_attn`).
Trained with plain MSE, it is an **MSE conditional-mean proxy** (the squared-error-optimal deterministic predictor is
E[ECG | PPG]); we never claim the network attains the exact conditional expectation.
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
        self.state_token = nn.Parameter(torch.randn(1, h_dim, 1) * 0.02)  # constant target-stream input (Amendment 1)

    def forward(self, ppg: torch.Tensor) -> torch.Tensor:
        """ppg [B, 1, T] -> ECG prediction [B, 1, T]. Mirrors upstream forward_step with x_t_emb = constant token and cond = 0."""
        bb = self.backbone
        ppg_e = bb.pre_conv_ppg(ppg)
        z_e = self.state_token.expand(ppg_e.shape[0], -1, ppg_e.shape[-1])
        cond = ppg_e.new_zeros(ppg_e.shape[0], self.h_dim)
        all_dx = torch.zeros_like(z_e)
        for blk in bb.flow_ssm_list:
            ppg_e, dx = blk(ppg_e, z_e, cond)
            all_dx = all_dx + dx
        return bb.final_layer(all_dx, cond)


def count_regressor_params(model: S5ConditionalMeanRegressor) -> dict:
    """total = all parameters; dead_cross_attn_revin = upstream's never-called modules; inactive_adaln_weights = the adaLN
    `Linear.weight`s, which receive zero gradient because cond = 0 (SiLU(0) = 0; functionally redundant with the adaLN biases);
    effective = total − dead − inactive."""
    total = sum(p.numel() for p in model.parameters())
    dead = sum(p.numel() for n, p in model.named_parameters() if "cross_attn" in n or n.endswith("revin"))
    inactive = sum(p.numel() for n, p in model.named_parameters() if "adaLN_modulation" in n and n.endswith("weight"))
    return {"total": total, "dead_cross_attn_revin": dead, "inactive_adaln_weights_cond0": inactive, "effective": total - dead - inactive, "state_token": model.state_token.numel()}
