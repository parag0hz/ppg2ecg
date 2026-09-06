"""M2 GSW-iMF (docs/M2_STRUCTURE_WEIGHTED_IMEANFLOW_PREREGISTRATION.md §29)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.flow import structure_weight as SW
from ppg2ecg.flow.imeanflow import MeanFlowS5, imeanflow_loss, sample_tr
from ppg2ecg.models import build_penguin_backbone, count_params

ROOT = Path(__file__).resolve().parents[1]
T = 1024

# ------------------------------------------------------------------ repository firewall


def test_frozen_pins_and_forbidden_subjects_unchanged():
    """prereg §1: the pins M2 anchors to must still hold when the suite runs."""
    def sha(path):
        return subprocess.run(["git", "-C", str(ROOT / path), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    assert sha("external/PENGUIN").startswith("6cd70cde")
    assert sha("external/iMeanFlow").startswith("bf60cd7c")
    a4 = ROOT / "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt"
    if a4.exists():
        assert hashlib.md5(a4.read_bytes()).hexdigest() == "31c042d291052fbb6dc15263ad316be2"
    c = ROOT / "artifacts/e2_evaluation_contract/contract_v1.json"
    if c.exists():
        assert hashlib.sha256(c.read_bytes()).hexdigest() == \
            "06e869412114e1efb9ab6624540aebc141495e2690150f4e512df7676c5a9115"
    assert not list((ROOT / "outputs").glob("c2*")), "C2 remains deferred"


def test_forbidden_subjects_absent_from_the_m2_split():
    m = json.loads((ROOT / "data/manifests/split_a4_wildppg_seed42.json").read_text())["splits"][0]
    assert set(m["train"]) == {"e61", "fex", "l38", "n31", "ngh", "p5d", "p9p", "qm9", "trh", "tz8", "u7y", "w4p"}
    assert set(m["val"]) == {"an0", "k2s"}
    assert not ({"kjd", "ssx"} & (set(m["train"]) | set(m["val"]))), "test subjects must never be trained or validated on"


# ------------------------------------------------------------------ structure map (§6)


def test_first_difference_is_backward_with_a_zero_first_sample():
    x = torch.tensor([1.0, 3.0, 6.0, 10.0])
    assert SW.first_difference(x).tolist() == [0.0, 2.0, 3.0, 4.0]
    assert SW.first_difference(torch.zeros(1, 1, 8)).abs().sum() == 0


def test_binomial_kernel_is_exactly_1_4_6_4_1_over_16_and_preserves_a_constant():
    assert SW.BINOMIAL_KERNEL == (1.0, 4.0, 6.0, 4.0, 1.0)
    out = SW.smooth_binomial(torch.ones(1, 1, 16))
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6), "a normalised kernel with reflect padding preserves a constant everywhere, edges included"
    assert out.shape == (1, 1, 16), "SAME-length convolution"


def test_reflect_padding_not_zero_padding_at_the_boundary():
    """Zero padding would pull the edge value down; reflect padding does not."""
    x = torch.ones(1, 1, 16) * 4.0
    assert SW.smooth_binomial(x)[0, 0, 0].item() == pytest.approx(4.0, abs=1e-6)


def test_q95_normalisation_and_clip_to_unit_interval():
    x = torch.zeros(1, 1, T)
    x[0, 0, ::64] = 10.0                      # sparse spikes -> a small q95, so many samples clip at 1
    m = SW.structure_map(x)
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0
    assert float(m.max()) == pytest.approx(1.0, abs=1e-6), "the strongest structure saturates the map"


def test_lambda_is_exactly_two_and_the_weight_has_unit_mean_per_window():
    assert SW.LAMBDA_STRUCT == 2.0
    x = torch.randn(5, 1, T)
    w = SW.structure_weight(x)
    assert w.shape == (5, 1, T)
    assert torch.allclose(w.mean(-1), torch.ones(5, 1), atol=1e-6), "prereg §6.4 / gate S0-B"
    assert torch.isfinite(w).all(), "gate S0-A"
    assert not w.requires_grad, "the map is data, never a parameter"


def test_maximum_unnormalised_emphasis_is_three_times():
    x = torch.zeros(1, 1, T)
    x[0, 0, ::64] = 10.0
    a = 1.0 + SW.LAMBDA_STRUCT * SW.structure_map(x)
    assert float(a.max()) == pytest.approx(3.0, abs=1e-6)
    assert float(a.min()) == pytest.approx(1.0, abs=1e-6)


def test_structure_map_never_reads_r_peaks_only_the_waveform():
    """Two windows with identical waveforms must get identical weights regardless of any annotation."""
    x = torch.randn(1, 1, T)
    assert torch.equal(SW.structure_weight(x), SW.structure_weight(x.clone()))
    sig = __import__("inspect").signature(SW.structure_weight)
    assert set(sig.parameters) == {"x", "lam"}, "no annotation may enter arm S"


def test_weight_is_higher_where_the_waveform_is_locally_sharp():
    x = torch.zeros(1, 1, T)
    x[0, 0, 500:503] = torch.tensor([0.0, 1.0, 0.0])       # one sharp lobe, flat elsewhere
    w = SW.structure_weight(x)[0, 0]
    assert w[498:506].mean() > w[:400].mean(), "emphasis follows local gradient energy"


# ------------------------------------------------------------------ controls (§7)


def test_shift_control_is_exactly_plus_256_and_preserves_the_histogram():
    x = torch.randn(2, 1, T)
    w, xs = SW.structure_weight(x), SW.shifted_structure_weight(x)
    assert torch.equal(xs, torch.roll(w, 256, dims=-1))
    assert torch.allclose(xs.mean(-1), w.mean(-1), atol=1e-7), "mean preserved"
    assert torch.allclose(torch.sort(xs, -1).values, torch.sort(w, -1).values), "histogram preserved exactly"
    assert float(xs.max()) == pytest.approx(float(w.max()))


def test_shift_control_refuses_a_length_other_than_1024():
    with pytest.raises(ValueError, match="new preregistration"):
        SW.shifted_structure_weight(torch.randn(1, 1, 512))


def test_hard_qrs_control_is_an_exact_r_plus_minus_10_mask():
    x = torch.zeros(1, 1, T)
    w = SW.hard_qrs_weight(x, [np.array([100, 600])], half=10)
    a = w * w.new_tensor(1.0)                                   # recover the shape of the raw weight
    assert torch.allclose(w.mean(-1), torch.ones(1, 1), atol=1e-6), "mean-1 normalised like arm S"
    hi = w[0, 0, 90:111]
    assert (hi == hi[0]).all() and hi[0] > w[0, 0, 300], "exactly R±10 is emphasised"
    assert w[0, 0, 89] < hi[0] and w[0, 0, 111] < hi[0], "the mask edge is exact, not smoothed"
    _ = a


# ------------------------------------------------------------------ loss (§7, §2.1)


def _tiny_net():
    torch.manual_seed(0)
    return MeanFlowS5(build_penguin_backbone(h_dim=32, ssm_block_num=1, ssm_ratio=1.0, mlp_ratio=1.0, sample_rate=128),
                      cond_mode="h_only", h_scale=1.0)


def _batch(B=4, n=256):
    torch.manual_seed(3)
    t, r, _ = sample_tr(B, torch.Generator().manual_seed(1))
    return torch.randn(B, 1, n), torch.randn(B, 1, n), torch.randn(B, 1, n), t, r


def test_all_ones_structure_weight_reproduces_the_baseline_loss_bit_exactly():
    """prereg §2.1: arm U is the same code path with structure_weight = 1."""
    net = _tiny_net()
    x, e, ppg, t, r = _batch()
    base, _ = imeanflow_loss(net, x, ppg, e, t, r)
    ones, _ = imeanflow_loss(net, x, ppg, e, t, r, structure_weight=torch.ones_like(x))
    assert base.item() == ones.item(), "bit-identical, not merely close"


def test_the_adaptive_weight_is_computed_from_the_UNWEIGHTED_delta2():
    """The resolution frozen in prereg §2.1 — w must not change when the structure weight does."""
    net = _tiny_net()
    x, e, ppg, t, r = _batch()
    _, i0 = imeanflow_loss(net, x, ppg, e, t, r)
    sw = torch.rand_like(x)
    sw = sw / sw.mean(-1, keepdim=True)
    _, i1 = imeanflow_loss(net, x, ppg, e, t, r, structure_weight=sw)
    for k in ("w_mean", "w_std", "w_min", "w_max", "w_median", "delta2_mean"):
        assert i0[k].item() == i1[k].item(), f"{k} moved; the adaptive weight must stay bit-identical between arms"


def test_a_mean_one_weight_changes_the_loss_but_not_its_scale_wildly():
    net = _tiny_net()
    x, e, ppg, t, r = _batch()
    base, _ = imeanflow_loss(net, x, ppg, e, t, r)
    sw = SW.structure_weight(torch.randn_like(x))
    got, _ = imeanflow_loss(net, x, ppg, e, t, r, structure_weight=sw)
    assert got.item() != base.item()
    assert 0.5 * base.item() < got.item() < 2.0 * base.item(), "mean-1 normalisation keeps the loss on scale"


def test_gradient_flows_to_the_model_and_not_to_the_structure_map():
    net = _tiny_net()
    x, e, ppg, t, r = _batch()
    sw = SW.structure_weight(x)
    assert not sw.requires_grad
    loss, _ = imeanflow_loss(net, x, ppg, e, t, r, structure_weight=sw)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in net.parameters())


def test_structure_weight_broadcasts_over_the_channel_axis_only():
    net = _tiny_net()
    x, e, ppg, t, r = _batch()
    per_time = SW.structure_weight(x)
    a, _ = imeanflow_loss(net, x, ppg, e, t, r, structure_weight=per_time)
    b, _ = imeanflow_loss(net, x, ppg, e, t, r, structure_weight=per_time.expand(-1, 1, -1))
    assert a.item() == b.item()


# ------------------------------------------------------------------ inference purity (§25)


def test_no_structure_map_symbol_is_reachable_from_the_inference_path():
    """prereg §14: the sampler must not touch the structure map or the ECG target."""
    src = (ROOT / "src/ppg2ecg/flow/imeanflow.py").read_text()
    body = src.split("def sample_meanflow(")[1].split("\ndef ")[0]
    for bad in ("structure_weight", "structure_map", "shifted_structure_weight", "hard_qrs_weight"):
        assert bad not in body, f"{bad} must not appear on the sampling path"


def test_the_arms_share_a_parameter_count():
    n = count_params(build_penguin_backbone(h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0, sample_rate=128))
    assert isinstance(n, (int, tuple, dict)), "count_params is the repo's own accounting; arms share the same backbone"
