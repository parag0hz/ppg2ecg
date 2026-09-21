import torch

from ppg2ecg.flow.imeanflow import MeanFlowS5, imeanflow_loss
from ppg2ecg.models import build_penguin_backbone, count_params
from ppg2ecg.models.kan_backbone import KANFFN, KANLinear
from ppg2ecg.utils.upstream import import_upstream_penguin

CFG = dict(n_step=1, sample_rate=128, h_dim=16, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0)


def test_default_build_still_upstream():
    torch.manual_seed(0); a = build_penguin_backbone(**CFG)
    torch.manual_seed(0); b = import_upstream_penguin()(**CFG)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_placement_outer_vs_all():
    outer = build_penguin_backbone(**CFG, arch="kan", kan_blocks="outer")
    every = build_penguin_backbone(**CFG, arch="kan", kan_blocks="all")
    is_kan = lambda m: [isinstance(b.mlp_ppg, KANFFN) and isinstance(b.mlp_target, KANFFN) for b in m.flow_ssm_list]  # noqa: E731
    assert is_kan(outer) == [True, False, False, True] and is_kan(every) == [True] * 4
    assert count_params(every)["total"] > count_params(outer)["total"] > count_params(build_penguin_backbone(**CFG))["total"]
    build_penguin_backbone(**dict(CFG, arch="kan", kan_blocks="all")).load_state_dict(every.state_dict())


def test_kanlinear_shape_and_bounded_basis():
    lin = KANLinear(8, 5)
    y = lin(torch.randn(2, 7, 8) * 50)                       # large inputs stay finite: the basis sees tanh(x / tau)
    assert y.shape == (2, 7, 5) and torch.isfinite(y).all()


def test_imf_loss_with_jvp_backprops_through_kan():
    torch.manual_seed(0)
    net = MeanFlowS5(build_penguin_backbone(**CFG, arch="kan", kan_blocks="all"))
    x, e, p = torch.randn(3, 1, 64), torch.randn(3, 1, 64), torch.randn(3, 1, 64)
    t, r = torch.tensor([[0.9], [0.5], [0.3]]), torch.tensor([[0.1], [0.5], [0.0]])
    loss, _ = imeanflow_loss(net, x, p, e, t, r)
    loss.backward()
    g = net.backbone.flow_ssm_list[0].mlp_target.kan.spline.weight.grad
    assert torch.isfinite(loss) and g is not None and torch.isfinite(g).all()
