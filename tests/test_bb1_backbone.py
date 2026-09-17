import torch

from ppg2ecg.flow.imeanflow import MeanFlowS5, imeanflow_loss
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.models.attn_backbone import SelfAttnMixer
from ppg2ecg.utils.upstream import import_upstream_penguin

CFG = dict(n_step=1, sample_rate=128, h_dim=16, ssm_block_num=2, ssm_ratio=2.0, mlp_ratio=2.0)


def test_default_build_is_upstream_unchanged():
    torch.manual_seed(0)
    a = build_penguin_backbone(**CFG)
    torch.manual_seed(0)
    b = import_upstream_penguin()(**CFG)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_attn_build_has_no_s5_and_reloads_from_cfg():
    m = build_penguin_backbone(**CFG, arch="attn", attn_heads=4)
    assert all(isinstance(b.ssm_ppg, SelfAttnMixer) and isinstance(b.ssm_target, SelfAttnMixer) for b in m.flow_ssm_list)
    assert not any("ssm_ppg.A" in k or ".B" in k.split("ssm_")[-1][:2] for k in m.state_dict())
    m2 = build_penguin_backbone(**dict(CFG, arch="attn", attn_heads=4))
    m2.load_state_dict(m.state_dict())


def test_attn_imf_loss_backprops():
    torch.manual_seed(0)
    net = MeanFlowS5(build_penguin_backbone(**CFG, arch="attn"))
    x, e, p = torch.randn(3, 1, 64), torch.randn(3, 1, 64), torch.randn(3, 1, 64)
    t, r = torch.tensor([[0.9], [0.5], [0.3]]), torch.tensor([[0.1], [0.5], [0.0]])
    loss, _ = imeanflow_loss(net, x, p, e, t, r)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
