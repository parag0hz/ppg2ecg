import torch

from ppg2ecg.models import build_penguin_backbone, count_params
from ppg2ecg.models.regressor import REMOVED_MODULES, S5ConditionalMeanRegressor, count_regressor_params


def test_regressor_is_backbone_minus_generative_inputs():
    torch.manual_seed(0)
    reg = S5ConditionalMeanRegressor(h_dim=16, ssm_block_num=2)
    bb = build_penguin_backbone(n_step=1, sample_rate=128, h_dim=16, ssm_block_num=2, ssm_ratio=2.0, mlp_ratio=2.0)
    removed = sum(p.numel() for n, p in bb.named_parameters() if n.split(".")[0] in REMOVED_MODULES)
    assert count_regressor_params(reg)["total"] == count_params(bb)["total"] - removed
    assert not any(hasattr(reg.backbone, m) for m in REMOVED_MODULES)
    kept = {n for n, _ in reg.backbone.named_parameters()}
    assert all(n in {k for k, _ in bb.named_parameters()} for n in kept)  # no new parameters


def test_regressor_full_config_param_count():
    reg = S5ConditionalMeanRegressor()
    assert count_regressor_params(reg)["total"] == 4568707 - 578048 == 3990659


def test_regressor_deterministic_and_conditioned():
    torch.manual_seed(1)
    reg = S5ConditionalMeanRegressor(h_dim=16, ssm_block_num=2).eval()
    with torch.no_grad():
        for n, p in reg.named_parameters():
            if "final_layer.linear" in n or "adaLN_modulation" in n:
                p.add_(0.05 * torch.randn_like(p))
    ppg = torch.randn(3, 1, 256)
    with torch.no_grad():
        a, b = reg(ppg), reg(ppg)
        c = reg(ppg + 0.5)
    assert a.shape == (3, 1, 256) and torch.equal(a, b) and not torch.allclose(a, c)
    loss = ((reg(ppg) - torch.randn(3, 1, 256)) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in reg.parameters() if p.grad is not None)
