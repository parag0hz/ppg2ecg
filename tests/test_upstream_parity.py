"""Parity of our re-statements with the UNMODIFIED upstream code (external/PENGUIN)."""
import numpy as np
import pytest
import torch

from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows
from ppg2ecg.flow.cfm import cfm_targets, euler_x1_estimate
from ppg2ecg.flow.samplers import heun_sample
from ppg2ecg.utils.upstream import UPSTREAM_SRC, import_upstream_penguin, import_upstream_preprocess

pytestmark = pytest.mark.skipif(not UPSTREAM_SRC.exists(), reason="upstream checkout missing")


def test_preprocess_matches_upstream():
    from types import SimpleNamespace

    up = import_upstream_preprocess()
    cfg = SimpleNamespace(preprocess=SimpleNamespace(resample_rate=128, segment_len=4))
    rng = np.random.default_rng(0)
    for native, kw in ((256, PPG_KW), (2800, ECG_KW)):
        x = rng.standard_normal((6, native)).cumsum(axis=1)
        a = up(x, cfg, kw["bandpass"], kw["freq_range"], kw["zscore"], kw["normalize"])
        b = preprocess_windows(x, 128, 4, **kw)
        assert a.shape == b.shape == (6, 512)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_heun_sampler_matches_upstream_bit_exact(device):
    PENGUIN = import_upstream_penguin()
    torch.manual_seed(0)
    model = PENGUIN(n_step=3, sample_rate=128, h_dim=16, ssm_block_num=2, ssm_ratio=2.0, mlp_ratio=2.0).to(device).eval()
    ppg = torch.randn(4, 512, device=device)
    with torch.no_grad():
        torch.manual_seed(7)
        up = model(ppg)
        torch.manual_seed(7)
        x0 = torch.randn(4, 1, 512).to(device)  # upstream draws on CPU then .to(device)
        ours, nfe = heun_sample(lambda x, t: model.forward_step(x, ppg.unsqueeze(1), t), x0, 3)
    assert nfe == 6
    assert torch.equal(up, ours.squeeze(1)), float((up - ours.squeeze(1)).abs().max())


def test_cfm_targets_match_upstream_train_flow():
    PENGUIN = import_upstream_penguin()
    torch.manual_seed(0)
    model = PENGUIN(n_step=1, sample_rate=128, h_dim=16, ssm_block_num=1, ssm_ratio=2.0, mlp_ratio=2.0).train()
    ppg, ecg = torch.randn(3, 512), torch.randn(3, 512)
    torch.manual_seed(11)
    pred_x1_up = model(ppg, target_signal=ecg)  # upstream train_flow: rand(t) then randn(x0)
    torch.manual_seed(11)
    t = torch.rand(3, 1)
    x0 = torch.randn(3, 1, 512)
    x_t, v_star, t, x0 = cfm_targets(ecg.unsqueeze(1), t, x0)
    assert torch.equal(v_star, model.dx_t)
    v_pred = model.forward_step(x_t, ppg.unsqueeze(1), t)
    assert torch.equal(v_pred, model.pred_dx_t)
    assert torch.equal(euler_x1_estimate(x_t, v_pred, t).squeeze(1), pred_x1_up)


def test_upstream_hr_metric_time_compression_with_4s_segments():
    """Upstream HeartRateError with segment_len=4 resamples the 8 s window to 512 samples => HR estimates double."""
    import neurokit2 as nk
    from scipy import signal
    from types import SimpleNamespace

    from ppg2ecg.utils.upstream import add_upstream_to_path

    add_upstream_to_path()
    from utils.help_func import calc_ecg_hr, compute_metrics  # noqa: PLC0415  (upstream)

    fs = 128
    x = nk.ecg_simulate(duration=8, sampling_rate=fs, heart_rate=60, random_state=1, method="ecgsyn")
    x = ((x - x.min()) / (x.max() - x.min()) * 2 - 1)[None]
    hr_direct = calc_ecg_hr(x, fs, filter=False)[0]
    hr_compressed = calc_ecg_hr(signal.resample(x, 128 * 4, axis=1), fs, filter=False)[0]  # upstream RR_seqlen with segment_len=4
    assert abs(hr_direct - 60) < 3
    assert abs(hr_compressed - 120) < 6  # doubled
    # a 120 bpm window becomes 240 bpm after compression -> no beats -> -1 -> masked -> error reported as 0.0
    y = nk.ecg_simulate(duration=8, sampling_rate=fs, heart_rate=120, random_state=1, method="ecgsyn")
    y = ((y - y.min()) / (y.max() - y.min()) * 2 - 1)[None]
    assert calc_ecg_hr(signal.resample(y, 512, axis=1), fs, filter=False)[0] == -1
    cfg4 = SimpleNamespace(preprocess=SimpleNamespace(segment_len=4))
    assert compute_metrics(torch.tensor(y, dtype=torch.float32).reshape(-1), torch.tensor(y, dtype=torch.float32).reshape(-1), "HeartRateError", cfg4) == 0.0
