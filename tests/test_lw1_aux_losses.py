import torch

from ppg2ecg.flow.aux_losses import mae_pcc_loss, stft_loss


def _sig(n=4, T=512):
    torch.manual_seed(0)
    t = torch.linspace(0, 4, T)
    return torch.stack([torch.sin(2 * torch.pi * (1.1 + 0.1 * i) * t) for i in range(n)]).unsqueeze(1)


def test_perfect_prediction_is_zero_and_constant_is_about_one():
    x = _sig()
    for alpha in (0.0, 0.5, 1.0):
        assert mae_pcc_loss(x, x, alpha)[0] < 1e-4
        const = x.mean(-1, keepdim=True).expand_as(x) + 1e-3 * torch.randn_like(x)
        assert 0.8 < float(mae_pcc_loss(const, x, alpha)[0]) < 1.3
    assert stft_loss(x, x)[0] < 1e-5 and 0.9 < float(stft_loss(torch.zeros_like(x), x)[0]) <= 1.0001


def test_stft_loss_tolerates_a_small_shift_but_mae_does_not():
    x = torch.zeros(1, 1, 512); x[..., 100] = 1.0; x[..., 300] = 1.0          # two spikes
    shifted = torch.roll(x, 6, dims=-1)                                       # ~47 ms late
    flat = torch.zeros_like(x)                                               # the "blurred to nothing" prediction
    assert stft_loss(shifted, x)[0] < stft_loss(flat, x)[0]                  # spectral: shifted spikes beat no spikes
    assert mae_pcc_loss(shifted, x, 1.0)[0] > mae_pcc_loss(flat, x, 1.0)[0]  # MAE: no spikes beat shifted spikes


def test_gradients_flow():
    x = _sig(); p = (x + 0.3 * torch.randn_like(x)).requires_grad_(True)
    (mae_pcc_loss(p, x, 0.5)[0] + stft_loss(p, x)[0]).backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
