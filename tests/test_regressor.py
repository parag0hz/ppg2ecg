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


# ---------------------------------------------------------------- A6: capacity-matched full-backbone deterministic control
def test_full_backbone_regressor_param_parity_and_determinism():
    from ppg2ecg.models.regressor import S5FullBackboneRegressor, count_full_backbone_params

    torch.manual_seed(3)
    reg = S5FullBackboneRegressor(h_dim=16, ssm_block_num=2)
    bb = build_penguin_backbone(n_step=1, sample_rate=128, h_dim=16, ssm_block_num=2, ssm_ratio=2.0, mlp_ratio=2.0)
    assert {n for n, _ in reg.backbone.named_parameters()} == {n for n, _ in bb.named_parameters()}  # identical module set
    assert count_full_backbone_params(reg)["total"] == count_params(bb)["total"]
    full = S5FullBackboneRegressor()
    assert count_full_backbone_params(full)["total"] == 4568707 and count_full_backbone_params(full)["effective"] == 4304513
    ppg = torch.randn(3, 1, 256)
    with torch.no_grad():  # at init the upstream zero-initialised final linear makes the output identically 0 -> perturb as in the A5 test
        for n, p in reg.named_parameters():
            if "final_layer.linear" in n or "adaLN_modulation" in n:
                p.add_(0.05 * torch.randn_like(p))
        a, b, c = reg(ppg), reg(ppg), reg(ppg + 0.5)
    assert torch.equal(a, b) and a.shape == (3, 1, 256) and not torch.allclose(a, c)
    assert "target" not in S5FullBackboneRegressor.forward.__code__.co_varnames  # no target information in the forward pass


def test_full_backbone_regressor_no_dead_start():
    """Prereg A6 §6: at step 0 only the final layer has gradient (upstream zero-initialises final_layer.linear.weight — identical in the
    generative model); by step 5 the state stem / timestep embedder / adaLN / final layer and by step 20 every pathway incl. the S5
    blocks and the PPG stem receive non-zero gradient, unlike the A5 zero-state dead start where nothing ever does."""
    from ppg2ecg.models.regressor import S5FullBackboneRegressor

    torch.manual_seed(4)
    reg = S5FullBackboneRegressor(h_dim=16, ssm_block_num=2)
    opt = torch.optim.AdamW(reg.parameters(), lr=1e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    groups = {"pre_conv_target": "backbone.pre_conv_target", "timestep_embedder": "backbone.timestep_embedder", "pre_conv_ppg": "backbone.pre_conv_ppg", "ssm_ppg": ".ssm_ppg", "ssm_target": ".ssm_target", "adaLN": "adaLN_modulation", "final_layer": "backbone.final_layer"}
    feats = {}
    reg.backbone.final_layer.register_forward_hook(lambda m, i, o: feats.__setitem__("in", i[0].detach()))
    early = {"pre_conv_target", "timestep_embedder", "adaLN", "final_layer"}
    for step in range(21):
        x, y = torch.randn(4, 1, 256, generator=g), torch.randn(4, 1, 256, generator=g)
        opt.zero_grad()
        ((reg(x) - y) ** 2).mean().backward()
        gn = {name: sum(float(p.grad.norm()) for n, p in reg.named_parameters() if pat in n and p.grad is not None) for name, pat in groups.items()}
        if step == 0:
            assert float(feats["in"].abs().max()) > 0  # final-layer input non-zero (x_const stem output)
            assert gn["final_layer"] > 0 and all(gn[k] == 0 for k in groups if k != "final_layer"), gn
        if step == 5:
            assert all(gn[k] > 0 for k in early), gn
        if step == 20:
            assert all(v > 0 for v in gn.values()), gn
        opt.step()


def test_full_backbone_cond_scale_keeps_every_pathway_trainable():
    """A6 Amendment (prereg §2c): cond = cond_scale * E(t_const) with cond_scale = 0.05 keeps the adaLN weights active (cond != 0),
    matches upstream forward_step when cond_scale = 1, and all pathways receive gradient by step 20."""
    from ppg2ecg.models.regressor import S5FullBackboneRegressor

    torch.manual_seed(5)
    reg = S5FullBackboneRegressor(h_dim=16, ssm_block_num=2, x_const=1.0, t_const=0.5, cond_scale=0.05)
    ref = S5FullBackboneRegressor(h_dim=16, ssm_block_num=2, x_const=1.0, t_const=0.5, cond_scale=1.0)
    ref.load_state_dict(reg.state_dict())
    with torch.no_grad():
        for m in (reg, ref):
            for n, p in m.named_parameters():
                if "final_layer.linear" in n or "adaLN_modulation" in n:
                    torch.manual_seed(6)
                    p.add_(0.05 * torch.randn_like(p))
        x = torch.randn(2, 1, 256)
        ref.cond_scale = 0.05
        assert torch.allclose(reg(x), ref(x), atol=1e-5)  # explicit path == upstream path with a scaled cond
        ref.cond_scale = 1.0
        assert not torch.allclose(reg(x), ref(x))  # the scale matters (adaLN inputs differ)
    opt = torch.optim.AdamW(reg.parameters(), lr=1e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    for step in range(21):
        xb, yb = torch.randn(4, 1, 256, generator=g), torch.randn(4, 1, 256, generator=g)
        opt.zero_grad()
        ((reg(xb) - yb) ** 2).mean().backward()
        if step == 20:
            for pat in ("pre_conv_target", "timestep_embedder", "pre_conv_ppg", ".ssm_ppg", ".ssm_target", "adaLN_modulation.1.weight", "final_layer"):
                assert sum(float(p.grad.norm()) for n, p in reg.named_parameters() if pat in n and p.grad is not None) > 0, pat
        opt.step()
