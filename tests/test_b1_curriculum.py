import numpy as np
import torch

from ppg2ecg.flow.imeanflow import MeanFlowS5, imeanflow_loss, sample_tr
from ppg2ecg.flow.imeanflow_curriculum import LAMBDA_DEFAULT, curriculum_beta, imeanflow_loss_b1, progress_s
from ppg2ecg.models import build_penguin_backbone


def _net(seed=0):
    torch.manual_seed(seed)
    return MeanFlowS5(build_penguin_backbone(n_step=1, sample_rate=128, h_dim=16, ssm_block_num=2))


def _batch(seed=1, B=8, T=256):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 1, T, generator=g)
    ppg = torch.randn(B, 1, T, generator=g)
    e = torch.randn(B, 1, T, generator=g)
    t, r, fm = sample_tr(B, g)
    return x, ppg, e, t, r, fm


def test_A_vanilla_loss_parity_and_B_gradient_parity():
    net = _net()
    x, ppg, e, t, r, fm = _batch()
    l0, i0 = imeanflow_loss(net, x, ppg, e, t, r)
    net.zero_grad()
    l0.backward()
    g0 = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    l1, i1 = imeanflow_loss_b1(net, x, ppg, e, t, r, fm, beta=None)
    assert abs(float(l0) - float(l1)) <= 1e-6 * max(1.0, abs(float(l0)))
    assert abs(float(i0["delta2_mean"]) - float(i1["delta2_mean"])) < 1e-6
    net.zero_grad()
    l1.backward()
    for n, p in net.named_parameters():
        if n in g0:
            assert torch.allclose(g0[n], p.grad, rtol=1e-5, atol=1e-7), n


def test_C_end_of_schedule_parity_and_schedule_values():
    assert progress_s(0, 1000) == 1.0 and progress_s(500, 1000) == 0.5 and progress_s(1000, 1000) == 0.0 and progress_s(2000, 1000) == 0.0
    _, _, _, t, r, fm = _batch(2, B=64)
    b_end = curriculum_beta(t, r, fm, s=0.0)
    assert torch.allclose(b_end, torch.ones_like(b_end))
    net = _net()
    x, ppg, e, t, r, fm = _batch(3)
    le, _ = imeanflow_loss_b1(net, x, ppg, e, t, r, fm, beta=curriculum_beta(t, r, fm, 0.0))
    l0, _ = imeanflow_loss(net, x, ppg, e, t, r)
    assert abs(float(le) - float(l0)) <= 1e-6 * max(1.0, abs(float(l0)))


def test_D_small_gap_ordering_and_E_mean_normalization():
    g = torch.Generator().manual_seed(12345)
    t, r, fm = sample_tr(200_000, g)
    b1 = curriculum_beta(t, r, fm, s=1.0)
    nb = ~fm.reshape(-1)
    h = (t - r).reshape(-1)[nb]
    b = b1.reshape(-1)[nb]
    order = torch.argsort(h)
    assert torch.all(b[order][:-1] >= b[order][1:] - 1e-6)  # beta non-increasing in h at s=1
    assert b[order][0] > b[order][-1]  # strictly greater across the range
    assert abs(float(b.mean()) - 1.0) < 0.01  # E-mean normalization (prereg tolerance 0.01)


def test_E2_boundary_invariance_across_schedule():
    _, _, _, t, r, fm = _batch(4, B=64)
    for s in (1.0, 0.6, 0.3, 0.0):
        b = curriculum_beta(t, r, fm, s)
        assert torch.allclose(b[fm], torch.ones_like(b[fm]))


def test_F_beta_has_no_grad_and_J_consumes_no_rng():
    x, ppg, e, t, r, fm = _batch(5)
    state = torch.get_rng_state()
    b = curriculum_beta(t, r, fm, 0.7)
    assert torch.equal(state, torch.get_rng_state())  # no RNG consumed
    assert not b.requires_grad


def test_G_adaptive_weight_isolation():
    net = _net()
    x, ppg, e, t, r, fm = _batch(6)
    torch.manual_seed(0)
    _, i_a = imeanflow_loss_b1(net, x, ppg, e, t, r, fm, beta=None)
    torch.manual_seed(0)
    _, i_b = imeanflow_loss_b1(net, x, ppg, e, t, r, fm, beta=curriculum_beta(t, r, fm, 1.0))
    assert torch.allclose(i_a["per_sample"]["w"], i_b["per_sample"]["w"])  # w identical regardless of beta
    assert torch.allclose(i_a["per_sample"]["delta2"], i_b["per_sample"]["delta2"])


def test_H_jvp_and_V_parity():
    net = _net()
    x, ppg, e, t, r, fm = _batch(7)
    _, i0 = imeanflow_loss(net, x, ppg, e, t, r)
    _, i1 = imeanflow_loss_b1(net, x, ppg, e, t, r, fm, beta=curriculum_beta(t, r, fm, 0.9))
    for k in ("mse", "delta2_mean", "u_abs_mean", "dudt_abs_mean", "v_tangent_abs_mean"):
        assert abs(float(i0[k]) - float(i1[k])) < 1e-6, k  # u, du/dt, V untouched by beta


def test_K_paired_streams_identical_between_arms():
    """The two arms consume identical RNG: same seeds -> identical (t, r, e, fm) sequences whether or not beta is computed."""
    def stream(with_beta):
        g = torch.Generator().manual_seed(42)
        out = []
        for step in range(5):
            t, r, fm = sample_tr(16, g)
            e = torch.randn(16, 1, 32, generator=g)
            if with_beta:
                curriculum_beta(t, r, fm, progress_s(step, 100))
            out.append((t, r, fm, e))
        return out
    a, b = stream(False), stream(True)
    for (ta, ra, fa, ea), (tb, rb, fb, eb) in zip(a, b):
        assert torch.equal(ta, tb) and torch.equal(ra, rb) and torch.equal(fa, fb) and torch.equal(ea, eb)


def test_L_lambda_matches_frozen_calibration():
    g = torch.Generator().manual_seed(12345)
    t, r, fm = sample_tr(2_000_000, g)
    h = (t - r)[~fm]
    lam = 1.0 / float((1 - h).double().mean())
    assert abs(lam - LAMBDA_DEFAULT) < 2e-3


def test_KL_fixed_budget_driver_integration(tmp_path):
    """K/L (integration): early stopping never terminates; final + fraction checkpoints saved; paired probe identical across arms."""
    import json as _json
    import sys

    from ppg2ecg.training import train_b1_fixed_compute as TB

    outs = {}
    for arm in ("vanilla", "curriculum"):
        out = tmp_path / arm
        argv = ["--exp-name", f"smoke_{arm}", "--out-dir", str(out), "--arm", arm, "--t-schedule", "6",
                "--epochs", "3", "--patience", "1", "--min-delta", "1e9",  # forces the historical trigger immediately
                "--h-dim", "16", "--blocks", "2", "--limit-windows", "2", "--batch-size", "16", "--micro-batch", "8",
                "--val-batch", "8", "--n-val-banks", "1", "--gen-diag-every", "0", "--probe-batches", "4"]
        TB.main(argv)
        outs[arm] = out
        log = (out / "training_log.csv").read_text().strip().splitlines()
        assert len(log) - 1 == 3, log  # all 3 rounds ran despite patience=1 (no early termination)
        summ = _json.loads((out / "training_summary.json").read_text())
        assert summ["historical_early_stop_round"] is not None and summ["epochs_run"] == 3
        assert (out / "checkpoint_final.pt").exists() and (out / "checkpoint_frac000.pt").exists()
        assert (out / "schedule_state.csv").exists() and (out / "gap_bins_train.csv").exists()
        assert summ["schedule_s_final"] == 0.0  # schedule completed at the fixed budget
    pa = _json.loads((outs["vanilla"] / "paired_randomness_probe.json").read_text())
    pb = _json.loads((outs["curriculum"] / "paired_randomness_probe.json").read_text())
    assert pa["sha256"] == pb["sha256"]  # identical data/noise/(t,r)/mask streams in both arms
    import csv as _csv
    rows = list(_csv.DictReader(open(outs["curriculum"] / "gap_bins_train.csv")))
    b_small = [float(r["beta_mean"]) for r in rows if r["bin"].startswith("[0.0") and float(r["s"]) > 0.4]
    b_large = [float(r["beta_mean"]) for r in rows if r["bin"].startswith("[0.7") and float(r["s"]) > 0.4]
    if b_small and b_large:
        assert b_small[0] > b_large[0]  # early emphasis on small gaps
    bd = [float(r["beta_mean"]) for r in rows if r["bin"] == "boundary(r=t)"]
    assert all(abs(b - 1.0) < 1e-9 for b in bd)
