"""Static / synthetic validation for C1, run BEFORE any training.

Protocol: docs/C1_INTERVAL_EXPOSURE_CONTROL_PREREGISTRATION.md (b32c952).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.evaluation.event_reliability import TEST_SUBJECTS, WildPPGTestFirewallError, assert_no_test_subjects
from ppg2ecg.flow.imeanflow import imf_bank_hash, make_imf_banks, sample_tr
from ppg2ecg.flow.interval_exposure import ARMS, FORCED_H, exposure_stats, sample_tr_c1

ROOT = Path(__file__).resolve().parents[1]
TR_KW = dict(p_mean=-0.4, p_std=1.0, data_proportion=0.5)
TRAINER = (ROOT / "src/ppg2ecg/training/train_a2.py").read_text()
PREFLIGHT = (ROOT / "scripts/preflight_c1_rng_control.py").read_text()


# ---------------------------------------------------------------- 1 baseline parity
def test_arm_b_is_bit_identical_to_the_historical_sampler():
    for n in (32, 64, 1000):
        a = sample_tr(n, torch.Generator().manual_seed(7), **TR_KW)
        b = sample_tr_c1(n, torch.Generator().manual_seed(7), arm="B", **TR_KW)
        for x, y in zip(a, b):
            assert torch.equal(x, y)


def test_arm_b_consumes_the_tr_stream_identically():
    """Arm B must leave the (t, r) generator in the same state as the historical sampler."""
    g1, g2 = torch.Generator().manual_seed(43), torch.Generator().manual_seed(43)
    for _ in range(5):
        sample_tr(32, g1, **TR_KW)
        sample_tr_c1(32, g2, arm="B", **TR_KW)
    assert torch.equal(g1.get_state(), g2.get_state())


def test_interventions_do_consume_extra_tr_randomness():
    g1, g2 = torch.Generator().manual_seed(43), torch.Generator().manual_seed(43)
    sample_tr(32, g1, **TR_KW)
    sample_tr_c1(32, g2, arm="H50", **TR_KW)
    assert not torch.equal(g1.get_state(), g2.get_state())


# ---------------------------------------------------------------- 2 exact-h=0 branch preserved
def test_exact_h0_probability_preserved_in_every_arm():
    for arm in ARMS:
        t, r, fm = sample_tr_c1(200_000, torch.Generator().manual_seed(1), arm=arm, **TR_KW)
        h = (t - r).reshape(-1)
        assert float((h == 0).double().mean()) == pytest.approx(0.5, abs=1e-9)
        assert torch.equal((h == 0).reshape(-1, 1), fm)      # the h=0 rows are exactly the fm rows


def test_forced_rows_never_touch_the_h0_branch():
    for arm in ("H25", "H50"):
        t, r, fm = sample_tr_c1(64, torch.Generator().manual_seed(2), arm=arm, **TR_KW)
        h = (t - r).reshape(-1)
        assert torch.all(h[fm.reshape(-1)] == 0)


# ---------------------------------------------------------------- 3/4 forced interval is exact
@pytest.mark.parametrize("arm,target", [("H25", 0.25), ("H50", 0.50)])
def test_forced_interval_is_exactly_the_target(arm, target):
    assert FORCED_H[arm] == target
    t, r, fm = sample_tr_c1(20_000, torch.Generator().manual_seed(3), arm=arm, **TR_KW)
    h = (t - r).reshape(-1)
    assert float((h == target).double().mean()) == pytest.approx(0.25, abs=1e-9)
    forced = h == target
    np.testing.assert_allclose((t.reshape(-1)[forced] - r.reshape(-1)[forced]).numpy(), target, atol=1e-7)


def test_forced_mass_is_exactly_a_quarter_and_positional():
    t, r, fm = sample_tr_c1(64, torch.Generator().manual_seed(4), arm="H50", **TR_KW)
    h = (t - r).reshape(-1)
    assert int((h == 0).sum()) == 32 and int((h == 0.5).sum()) == 16
    assert torch.all(h[48:] == 0.5)                          # the forced block is the positional tail


# ---------------------------------------------------------------- 5/6 domain
def test_forced_samples_stay_in_domain_and_t_is_uniform_on_the_right_range():
    for arm, tgt in (("H25", 0.25), ("H50", 0.50)):
        t, r, _ = sample_tr_c1(50_000, torch.Generator().manual_seed(5), arm=arm, **TR_KW)
        assert torch.all(r >= 0) and torch.all(t <= 1) and torch.all(r <= t)
        forced = (t - r).reshape(-1) == tgt
        tf = t.reshape(-1)[forced]
        assert float(tf.min()) >= tgt - 1e-9 and float(tf.max()) <= 1.0
        assert float(tf.mean()) == pytest.approx((tgt + 1.0) / 2, abs=0.01)   # Uniform[tgt, 1]


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError):
        sample_tr_c1(8, torch.Generator().manual_seed(0), arm="H75", **TR_KW)


# ---------------------------------------------------------------- 7 no loss / protocol change
def test_only_the_sampler_call_site_changed_in_the_trainer():
    assert "sample_tr_c1(Bc, tr_gen, arm=args.c1_arm, **tr_kw)" in TRAINER
    assert "sample_tr(Bc, tr_gen" not in TRAINER
    # tr_kw must NOT carry the arm, or the selection banks would differ between arms
    assert 'tr_kw = dict(p_mean=args.p_mean, p_std=args.p_std, data_proportion=args.data_proportion)' in TRAINER
    assert "make_imf_banks(len(x_va), T, args.n_val_banks, args.bank_seed, **tr_kw)" in TRAINER
    for frozen in ("norm_p=args.norm_p, norm_eps=args.norm_eps", "AdamW", 'default="B"'):
        assert frozen in TRAINER


def test_selection_banks_are_arm_independent():
    """make_imf_banks depends only on tr_kw, which C1 never changes."""
    a = imf_bank_hash(make_imf_banks(256, 1024, 4, 1000, **TR_KW))
    b = imf_bank_hash(make_imf_banks(256, 1024, 4, 1000, **TR_KW))
    assert a == b


# ---------------------------------------------------------------- 8-11 RNG control artefact
def test_rng_control_preflight_asserts_the_four_shared_streams():
    for k in ("init_hash", "order_hash", "noise_hash", "banks_hash"):
        assert f'"{k}"' in PREFLIGHT or k in PREFLIGHT
    assert "tr_hash_B_differs_from_H25" in PREFLIGHT and "tr_hash_B_differs_from_H50" in PREFLIGHT


def test_rng_control_result_if_present_passed():
    p = ROOT / "artifacts/c1_interval_exposure/rng_control.json"
    if not p.exists():
        pytest.skip("preflight not run yet")
    d = json.loads(p.read_text())
    assert d["pass"] is True
    for k in ("identical_init_hash", "identical_order_hash", "identical_noise_hash", "identical_banks_hash"):
        assert d["checks"][k] is True
    assert d["checks"]["p_h0_preserved"] and d["checks"]["baseline_matches_record"]


# ---------------------------------------------------------------- 12 firewall
def test_test_subject_firewall():
    for bad in TEST_SUBJECTS:
        with pytest.raises(WildPPGTestFirewallError):
            assert_no_test_subjects(["an0", "k2s", bad])
    for src in (PREFLIGHT,):
        body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        assert "kjd" not in body and "ssx" not in body


# ---------------------------------------------------------------- 13 checkpoint protection
def test_c1_never_targets_the_frozen_a4_output_directory():
    for src in (PREFLIGHT,):
        assert "a4_imeanflow_wildppg_seed42" not in src or "checkpoint" not in src
    assert (ROOT / "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt").exists()


# ---------------------------------------------------------------- exposure summary
def test_exposure_stats_shape_and_baseline_record():
    g = torch.Generator().manual_seed(20260901)
    t, r, _ = sample_tr_c1(200_000, g, arm="B", **TR_KW)
    s = exposure_stats(t - r)
    assert s["p_h_eq_0"] == pytest.approx(0.50, abs=0.01)
    assert s["positive_median"] == pytest.approx(0.201, abs=0.01)
    assert s["p_h_ge_0.5"] == pytest.approx(0.042, abs=0.005)
    assert s["p_h_ge_0.7"] == pytest.approx(0.004, abs=0.003)


def test_h50_raises_target_interval_exposure_and_h25_does_not():
    out = {}
    for arm in ARMS:
        t, r, _ = sample_tr_c1(200_000, torch.Generator().manual_seed(6), arm=arm, **TR_KW)
        out[arm] = exposure_stats(t - r)
    assert out["H50"]["p_h_ge_0.5"] > out["B"]["p_h_ge_0.5"] + 0.15
    assert out["H25"]["p_h_ge_0.5"] < out["B"]["p_h_ge_0.5"] + 0.01
    # the two interventions replace the SAME half of positive rows, so their sub-0.25 exposure matches
    assert out["H25"]["p_h_ge_0.25"] == pytest.approx(out["H50"]["p_h_ge_0.25"], abs=1e-9)
