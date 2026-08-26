import torch

from ppg2ecg.models import build_penguin_backbone, count_params
from ppg2ecg.models.regressor import REMOVED_MODULES, S5ConditionalMeanRegressor, count_regressor_params


def test_regressor_is_backbone_minus_generative_inputs_plus_state_token():
    torch.manual_seed(0)
    reg = S5ConditionalMeanRegressor(h_dim=16, ssm_block_num=2)
    bb = build_penguin_backbone(n_step=1, sample_rate=128, h_dim=16, ssm_block_num=2, ssm_ratio=2.0, mlp_ratio=2.0)
    removed = sum(p.numel() for n, p in bb.named_parameters() if n.split(".")[0] in REMOVED_MODULES)
    c = count_regressor_params(reg)
    assert c["total"] == count_params(bb)["total"] - removed + 16 and c["state_token"] == 16
    assert not any(hasattr(reg.backbone, m) for m in REMOVED_MODULES)
    bb_names = {k for k, _ in bb.named_parameters()}
    assert all(n in bb_names for n, _ in reg.backbone.named_parameters())  # the only new parameter is the state token
    assert [n for n, _ in reg.named_parameters() if not n.startswith("backbone.")] == ["state_token"]


def test_regressor_full_config_param_count():
    reg = S5ConditionalMeanRegressor()
    assert count_regressor_params(reg)["total"] == 4568707 - 578048 + 128 == 3990787


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


def _grad_nonzero_names(reg, steps):
    opt = torch.optim.AdamW(reg.parameters(), lr=1e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        x, y = torch.randn(4, 1, 256, generator=g), torch.randn(4, 1, 256, generator=g)
        opt.zero_grad()
        ((reg(x) - y) ** 2).mean().backward()
        opt.step()
    x, y = torch.randn(4, 1, 256, generator=g), torch.randn(4, 1, 256, generator=g)
    opt.zero_grad()
    ((reg(x) - y) ** 2).mean().backward()
    return {n for n, p in reg.named_parameters() if p.grad is not None and float(p.grad.abs().max()) > 0}


def test_zero_state_input_is_a_permanent_dead_start_and_state_token_fixes_it():
    """Amendment 1 (prereg §16): with a zero target-stream input only final_layer.linear.bias ever trains."""
    torch.manual_seed(2)
    dead = S5ConditionalMeanRegressor(h_dim=16, ssm_block_num=2)
    with torch.no_grad():
        dead.state_token.zero_()
    live = {n for n in _grad_nonzero_names(dead, steps=3) if n != "state_token"}
    assert live == {"backbone.final_layer.linear.bias"}, live
    torch.manual_seed(2)
    reg = S5ConditionalMeanRegressor(h_dim=16, ssm_block_num=2)
    # structurally inactive: upstream's never-called cross_attn/revin, adaLN Linear.weights (cond = 0 -> SiLU(0) = 0),
    # and the last block's PPG-stream MLP (its output is not consumed; identical in the generative backbone)
    last = f"backbone.flow_ssm_list.{len(reg.backbone.flow_ssm_list) - 1}.mlp_ppg"
    trainable = {n for n, p in reg.named_parameters() if "cross_attn" not in n and not n.endswith("revin") and not ("adaLN_modulation" in n and n.endswith("weight")) and not n.startswith(last)}
    live = _grad_nonzero_names(reg, steps=8)
    missing = trainable - live
    assert not missing, sorted(missing)
    with torch.no_grad():
        out = reg(torch.randn(4, 1, 256))
    assert float(out.std(dim=(1, 2)).min()) > 0 and not torch.equal(out[0], out[1])  # input-dependent (still small after 8 steps)
