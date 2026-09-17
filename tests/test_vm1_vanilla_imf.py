import math
import sys
from pathlib import Path

import pytest
import torch

from ppg2ecg.flow.imf_vanilla import sample_vimf, vimf_loss
from ppg2ecg.models.imf_dit import ImfDiT1d, apply_rope, count_params, load_official_weights, rope_tables

SMALL = dict(seq_len=64, patch=8, hidden=48, depth=3, n_heads=4, aux_head_depth=1)


def test_rope_matches_official_complex_form():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 3, 8)
    cos, sin = rope_tables(8, 5)
    freqs = torch.complex(cos, sin)
    xc = x.to(torch.float32).view(torch.complex64).reshape(x.shape[:-1] + (-1,))       # official torch branch
    ref = (xc * freqs[None, :, None, :]).view(torch.float32).reshape(x.shape)
    assert torch.allclose(apply_rope(x, cos, sin), ref, atol=1e-6)


def test_zero_init_outputs_and_shapes():
    net = ImfDiT1d(**SMALL)
    x, p = torch.randn(3, 1, 64), torch.randn(3, 1, 64)
    b = torch.full((3, 1, 1), 0.5)
    u, v = net(x, b, b * 4, b * 0, b * 2, p, torch.tensor([True, False, False]))
    assert u.shape == v.shape == (3, 1, 64)
    assert torch.count_nonzero(u) == 0 and torch.count_nonzero(v) == 0            # zero final layers, as official
    assert count_params(net, inference=True) < count_params(net)


def test_loss_backprops_into_u_and_v_heads():
    torch.manual_seed(0)
    net = ImfDiT1d(**SMALL)
    with torch.no_grad():                               # leave the zero-init point so every path carries signal
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x, p = torch.randn(8, 1, 64), torch.randn(8, 1, 64)
    loss, info = vimf_loss(net, x, p, torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss)
    g = {n: q.grad for n, q in net.named_parameters()}
    assert g["u_final_layer.linear.weight"].abs().sum() > 0
    assert g["v_final_layer.linear.weight"].abs().sum() > 0
    assert g["shared_blocks.0.attn.q_proj.weight"].abs().sum() > 0


def test_sampler_step_count():
    net = ImfDiT1d(**SMALL)
    z, k = sample_vimf(net, torch.randn(2, 1, 64), torch.randn(2, 1, 64), 3, 2.0, 0.4, 0.65)
    assert k == 3 and z.shape == (2, 1, 64)


CKPT = Path(__file__).resolve().parents[1] / "data/pretrained/imf/iMF-B-2.pth"


@pytest.mark.skipif(not CKPT.exists(), reason="official checkpoint not downloaded")
def test_official_b2_loads_trunk():
    net = ImfDiT1d(seq_len=512, patch=8, hidden=768, depth=12, n_heads=12, aux_head_depth=8)
    rep = load_official_weights(net, str(CKPT))
    assert rep["n_v_copied_from_u"] == 8 * 13                      # 8 v-head blocks x 13 tensors each
    assert not any(k.startswith(("shared_blocks", "u_heads")) for k in rep["fresh_init"])
    assert set(rep["skipped_from_ckpt"]) >= {"x_embedder.proj.weight", "u_final_layer.linear.weight"}
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert torch.equal(net.state_dict()["shared_blocks.0.attn.q_proj.weight"],
                       sd["net.shared_blocks.0.attn.q_proj._flax_linear.weight"])
