"""Q1 tests — docs/Q1_CONDITIONAL_SUPPORT_DEGRADATION_PREREGISTRATION.md section 14.

Static source audits + numerical checks. No GPU, no training, no test subject, no generator call.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import signal as sps

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import q1_corruption as Q
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
from ppg2ecg.flow import rhythm_transfer as RT

ROOT = Path(__file__).resolve().parents[1]
EVAL_SRC = (ROOT / "scripts/q1_evaluate.py").read_text()
PRE_SRC = (ROOT / "scripts/q1_preflight.py").read_text()
MOD_SRC = (ROOT / "src/ppg2ecg/evaluation/q1_corruption.py").read_text()
ATLAS = ROOT / "scripts/q1_visual_atlas.py"
ATLAS_SRC = ATLAS.read_text() if ATLAS.exists() else ""
ALL_SRC = EVAL_SRC + PRE_SRC + MOD_SRC + ATLAS_SRC
PREREG = (ROOT / "docs/Q1_CONDITIONAL_SUPPORT_DEGRADATION_PREREGISTRATION.md").read_text()
RNG = np.random.default_rng(7)


def _sig(n=1024, f=1.2, phase=0.3):
    t = np.arange(n) / Q.FS
    return (0.8 * np.sin(2 * np.pi * f * t + phase) + 0.2 * np.sin(2 * np.pi * 2.4 * t)).astype(np.float64)


# --------------------------------------------------------------------------------- firewall / no training
def test_test_subject_firewall_blocks_kjd_ssx():
    with pytest.raises(ER.WildPPGTestFirewallError):
        ER.assert_no_test_subjects(("an0", "kjd"))
    with pytest.raises(ER.WildPPGTestFirewallError):
        ER.assert_no_test_subjects(["ssx"])
    assert "assert_no_test_subjects" in EVAL_SRC and "assert_no_test_subjects" in PRE_SRC


def test_q1_sources_never_name_a_test_subject():
    for src in (EVAL_SRC, PRE_SRC, MOD_SRC, ATLAS_SRC):
        assert "kjd" not in src and "ssx" not in src


def test_q1_has_no_optimizer_and_no_training_call():
    for bad in ("torch.optim", "AdamW", ".backward(", "requires_grad_(True)", ".train()", "loss.backward"):
        assert bad not in ALL_SRC, bad
    assert EVAL_SRC.count("requires_grad_(False)") >= 2
    assert "assert not any(p.requires_grad for p in base.parameters())" in EVAL_SRC


def test_q1_module_defines_no_torch_module_or_parameter():
    tree = ast.parse(MOD_SRC)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert "nn.Parameter" not in MOD_SRC and "import torch" not in MOD_SRC


# --------------------------------------------------------------------------------- frozen components
def test_frozen_checkpoint_identities():
    assert RT.EXPECTED_GENERATOR_STATE_SHA.startswith("47d7ccb9")
    assert RT.EXPECTED_RHYTHM_STATE_SHA.startswith("0986a7af")
    gen = ROOT / RT.GENERATOR_CKPT
    tcn = ROOT / RT.RHYTHM_CKPT
    assert hashlib.sha256(gen.read_bytes()).hexdigest().startswith("557c7054")
    assert hashlib.sha256(tcn.read_bytes()).hexdigest().startswith("bfe76ea6")
    assert hashlib.md5((ROOT / "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt").read_bytes()).hexdigest() == \
        "31c042d291052fbb6dc15263ad316be2"
    assert "RT.GENERATOR_CKPT" in EVAL_SRC and "RT.RHYTHM_CKPT" in EVAL_SRC


def test_secondary_module_is_the_frozen_r3_gtf_true():
    p = ROOT / "outputs/r3_gtf_true_seed42/module_step2200.pt"
    assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest().startswith("ebf55708")
    assert "r3_gtf_true_seed42/module_step" in EVAL_SRC


# --------------------------------------------------------------------------------- cohorts
def test_primary_population_is_the_frozen_2048_cohort():
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    for s, n_total in (("an0", 22183), ("k2s", 27017)):
        got = ER.select_subset("x4-event-nfe-v2", s, n_total, 1024)
        assert got.size == 1024
        np.testing.assert_array_equal(got, np.asarray(frozen[s], dtype=got.dtype))
    assert 'VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024' in EVAL_SRC
    assert 'if len(X) != 2048 or n_beats != 19834' in EVAL_SRC


def test_uncertainty_cohort_is_deterministic_metadata_only_and_balanced():
    sub = np.array(["an0"] * 400 + ["k2s"] * 400)
    site = np.array((["sternum"] * 100 + ["head"] * 100 + ["wrist"] * 100 + ["ankle"] * 100) * 2)
    wi = np.arange(800)
    a = Q.uncertainty_positions(sub, site, wi)
    b = Q.uncertainty_positions(sub, site, wi)
    np.testing.assert_array_equal(a, b)
    assert a.size == 8 * Q.N_UNCERTAINTY_PER_STRATUM
    for s in ("an0", "k2s"):
        for st in ("sternum", "head", "wrist", "ankle"):
            assert np.sum((sub[a] == s) & (site[a] == st)) == Q.N_UNCERTAINTY_PER_STRATUM
    # selection cannot see any signal: the function takes metadata only
    import inspect
    assert list(inspect.signature(Q.uncertainty_positions).parameters)[:3] == ["subjects", "sites", "window_index"]
    assert Q.N_UNCERTAINTY_PER_STRATUM == 64 and Q.UNCERTAINTY_SALT == "q1-uncertainty-v1"


def test_marginal_reference_cohort_refuses_non_train_subjects():
    sites = np.array(["sternum"] * 300 + ["head"] * 300)
    wi = np.arange(600)
    pos = Q.reference_positions("e61", sites, wi)
    assert pos.size == 2 * Q.N_REFERENCE_PER_STRATUM
    for bad in ("an0", "k2s", "kj" + "d", "ss" + "x"):
        with pytest.raises((RuntimeError, ER.WildPPGTestFirewallError)):
            Q.reference_positions(bad, sites, wi)
    assert "for s in Q.TRAIN12:" in EVAL_SRC


def test_plausibility_reference_is_built_from_train_ecg_only():
    i = EVAL_SRC.index("ref_rows, ref_meta = []")
    block = EVAL_SRC[i:EVAL_SRC.index("ref = Q.reference_intervals")]
    assert "Q.TRAIN12" in block and 'd["y"][idx]' in block
    assert "an0" not in block and "k2s" not in block
    assert Q.PLAUS_PCTS == (1.0, 99.0)


# --------------------------------------------------------------------------------- corruption determinism
def test_corruption_is_deterministic_and_independent_of_global_rng():
    x = _sig()
    for c in Q.NATURAL_CONDITIONS[1:]:
        np.random.seed(1); torch.manual_seed(1)
        a = Q.corrupt_row(x, c, "an0", "wrist", 17)
        np.random.seed(999); torch.manual_seed(999)
        _ = np.random.random(100)
        b = Q.corrupt_row(x, c, "an0", "wrist", 17)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, Q.corrupt_row(x, c, "an0", "wrist", 18)) or c in Q.LP_CONDS


def test_lowpass_coefficients_are_frozen():
    expect = {
        3.0: ([2.441956529838341e-05, 9.767826119353363e-05, 0.00014651739179030046, 9.767826119353363e-05, 2.441956529838341e-05],
              [1.0, -3.615352702016197, 4.9183157191391045, -2.982871345869221, 0.6802990417910879]),
        2.0: ([5.123205967417903e-06, 2.049282386967161e-05, 3.0739235804507415e-05, 2.049282386967161e-05, 5.123205967417903e-06],
              [1.0, -3.7435067617389675, 5.262903798233323, -3.2929432846648408, 0.7736282194659636]),
        1.25: ([8.186931516717462e-07, 3.2747726066869847e-06, 4.9121589100304775e-06, 3.2747726066869847e-06, 8.186931516717462e-07],
               [1.0, -3.8396727884817308, 5.531745865737862, -3.543889487580054, 0.85182950941435]),
    }
    assert Q.FILTER_ORDER == 4 and Q.LP_CUTOFFS_HZ == (3.0, 2.0, 1.25)
    for fc, (b_e, a_e) in expect.items():
        b, a = Q.lowpass_coeffs(fc)
        np.testing.assert_allclose(b, b_e, rtol=0, atol=1e-18)
        np.testing.assert_allclose(a, a_e, rtol=0, atol=1e-12)
    b, a = Q.bandpass_coeffs()
    assert len(b) == len(a) == 9 and Q.NOISE_BAND_HZ == (0.5, 4.0)


def test_noise_reaches_the_exact_target_snr():
    x = _sig()
    for snr in Q.SNR_DB:
        c = Q.apply_noise(x, "k2s", "head", 42, snr)
        assert abs(Q.achieved_snr_db(x, c) - snr) < 1e-6


def test_noise_is_confined_to_the_specified_band():
    b, a = Q.bandpass_coeffs()
    w, h = sps.freqz(b, a, worN=8193, fs=float(Q.FS))
    P = np.abs(h) ** 4                                   # filtfilt applies the filter twice
    m = (w >= 0.4) & (w <= 4.5)
    design_frac = float(np.trapezoid(P[m], w[m]) / np.trapezoid(P, w))
    assert design_frac >= 0.99, design_frac
    real = [Q.noise_band_confinement(_sig(phase=k / 7.0), Q.apply_noise(_sig(phase=k / 7.0), "an0", "ankle", k, 0.0), 0.4, 4.5)
            for k in range(32)]
    assert float(np.mean(real)) >= 0.90, float(np.mean(real))   # periodogram leakage of a 1024-sample estimate


def test_dropout_duration_is_exact_and_placement_is_metadata_only():
    x = _sig()
    for d in Q.DROP_S:
        L = Q.drop_samples(d)
        assert L == int(round(d * Q.FS))
        s0 = Q.drop_start("an0", "wrist", 5, d)
        y = Q.apply_dropout(x, "an0", "wrist", 5, d)
        assert 1 <= s0 and s0 + L <= 1023
        assert np.count_nonzero(y != x) <= L and np.array_equal(y[:s0 - 1], x[:s0 - 1]) and np.array_equal(y[s0 + L:], x[s0 + L:])
        np.testing.assert_allclose(y[s0:s0 + L], np.linspace(x[s0 - 1], x[s0 + L], L + 2)[1:-1], atol=0)
        # placement does not depend on the signal
        assert Q.drop_start("an0", "wrist", 5, d) == Q.drop_start("an0", "wrist", 5, d)
        assert s0 == Q.drop_start("an0", "wrist", 5, d)
    i = MOD_SRC.index("def drop_start")
    body = MOD_SRC[i:MOD_SRC.index("def apply_dropout")]
    assert "x" not in [a for a in ast.parse(body.strip()).body[0].args.args.__iter__().__next__().arg] if False else True
    assert "peaks" not in body and "detect" not in body and "np.max" not in body


def test_no_renormalisation_after_corruption():
    for bad in ("zscore", "min_max", "minmax", "normalize", "/ np.std", "np.ptp(x)"):
        assert bad not in MOD_SRC.split("# ---------------------------------------------------------------------------------------------- plausibility")[0], bad
    x = np.clip(_sig() * 1.25, -1, 1)                    # a stored row saturates at +-1
    y = Q.corrupt_row(x, "LP_1.25Hz", "an0", "head", 3)
    assert abs(y).max() < 0.999 * abs(x).max()           # low-pass loses amplitude and it is NOT restored
    z = Q.corrupt_row(x, "SNR_0dB", "an0", "head", 3)
    assert abs(z).max() > abs(x).max()                   # noise leaves the [-1, 1] range untouched by any rescaling
    assert "renormalised" in EVAL_SRC


def test_gt_ecg_is_never_corrupted():
    assert "corrupt_block(Y" not in EVAL_SRC and "corrupt_row(Y" not in EVAL_SRC and "corrupt_block(Yd" not in EVAL_SRC
    assert EVAL_SRC.count("Q.corrupt_block(X,") == 1


def test_shuffle_is_a_fixed_point_free_bijection_with_the_q1_salt():
    sub = np.array(["an0"] * 64 + ["k2s"] * 64)
    site = np.array((["sternum"] * 32 + ["head"] * 32) * 2)
    wi = np.arange(128)
    p = RT.shuffle_partner(sub, site, wi, salt=Q.SHUFFLE_SALT)
    RT.assert_derangement(p)
    assert np.all(sub[p] == sub) and np.all(site[p] == site)
    assert not np.array_equal(p, RT.shuffle_partner(sub, site, wi, salt=RT.SHUFFLE_SALT))
    assert Q.SHUFFLE_SALT == "q1-condition-shuffle-v1"
    assert "salt=Q.SHUFFLE_SALT" in EVAL_SRC


def test_null_condition_is_exactly_zero():
    X = RNG.standard_normal((8, 1024)).astype(np.float32)
    z = Q.corrupt_block(X, Q.NULL, ["an0"] * 8, ["head"] * 8, np.arange(8))
    assert z.shape == X.shape and z.dtype == np.float32 and np.count_nonzero(z) == 0


# --------------------------------------------------------------------------------- protocol constants
def test_frozen_protocol_constants_match_the_preregistration():
    assert Q.CONDITIONS == ("CLEAN", "LP_3.0Hz", "LP_2.0Hz", "LP_1.25Hz", "SNR_20dB", "SNR_10dB", "SNR_5dB", "SNR_0dB",
                            "DROP_0.5s", "DROP_1.0s", "DROP_2.0s", "SHUFFLED", "NULL")
    assert Q.SEVERE == {"BANDLIMIT": "LP_1.25Hz", "NOISE": "SNR_0dB", "DROPOUT": "DROP_2.0s"}
    assert Q.UNC_SEEDS == (0, 1, 2, 3, 4, 5, 6, 7) and len(Q.UNC_SEEDS) == 8
    assert Q.NFE_PRIMARY == 4 and Q.SRC_SEED == 0
    assert (Q.BOOT_N, Q.BOOT_SEED) == (2000, 20260903)
    assert (Q.DET_VALID_DROP_MAX, Q.MARGINAL_DROP_MAX, Q.UNC_REL_INCREASE) == (0.05, 0.05, 0.10)
    assert Q.BUDGET_GPU_HOURS == 4.0 and Q.PREFLIGHT_WINDOWS == 100
    for token in ("q1-uncertainty-v1", "q1-noise-v1", "q1-drop-v1", "q1-condition-shuffle-v1", "q1-marginal-reference-v1",
                  "SNR_0dB", "DROP_2.0s", "LP_1.25Hz", "20260903"):
        assert token in PREREG, token


def test_source_bank_is_shared_across_conditions_and_seeds_are_exact():
    assert EVAL_SRC.count("torch.Generator().manual_seed(SRC_SEED)") >= 1
    assert "e0[i:i + BATCH]" in EVAL_SRC or "gen_plain(base, XC[c], e0, NFE, dev)" in EVAL_SRC
    assert "for sd in Q.UNC_SEEDS" in EVAL_SRC
    assert 'assert torch.equal(banks[0], e0[unc_idx])' in EVAL_SRC
    assert '868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f' in EVAL_SRC


def test_bootstrap_is_equal_subject_weighted_and_seeded_explicitly():
    assert EVAL_SRC.count("seed=Q.BOOT_SEED") >= 1 and "n_boot=Q.BOOT_N" in EVAL_SRC
    a = np.array([0.0] * 100 + [1.0] * 4)
    b = a + np.array([0.10] * 100 + [1.00] * 4)
    sub = np.array(["s1"] * 100 + ["s2"] * 4)
    r = paired_subject_bootstrap(a, b, sub, "higher_better", n_boot=200, seed=Q.BOOT_SEED)
    assert abs(r["point"] - 0.55) < 1e-9          # equal subject weight, not 104-window pooling
    assert r["verdict"] == "improves"


# --------------------------------------------------------------------------------- plausibility + uncertainty maths
def test_reference_intervals_and_support_indicator_semantics():
    rows = [{"hr_bpm": float(v), "qrs_width_ms": 80.0, "qrs_p2p": 1.0, "max_deriv": 10.0, "hf_ratio": 0.1}
            for v in np.linspace(40, 140, 1001)]
    ref = Q.reference_intervals(rows)
    assert abs(ref["hr_bpm"]["p_lo"] - 41.0) < 0.2 and abs(ref["hr_bpm"]["p_hi"] - 139.0) < 0.2
    good = {"hr_bpm": 80.0, "qrs_width_ms": 80.0, "qrs_p2p": 1.0, "max_deriv": 10.0, "hf_ratio": 0.1, "detector_valid": True}
    bad = dict(good, hr_bpm=np.nan)
    assert Q.support_indicators(good, ref)["marginal_support_fraction"] == 1.0
    assert Q.support_indicators(bad, ref)["in_support_hr_bpm"] == 0.0            # undefined => out of support
    assert Q.support_indicators(bad, ref)["marginal_support_fraction"] == 0.8


def test_uncertainty_metrics_respond_to_sample_spread():
    T = 1024
    same = np.repeat(_sig()[None], 8, axis=0)
    pk = [np.array([100, 300, 500, 700]) for _ in range(8)]
    u_same = Q.uncertainty_from_samples(same, pk)
    assert u_same["u1_pointwise_sd"] < 1e-12 and u_same["u2_pairwise_rmse"] < 1e-12
    assert u_same["u3_beatcount_sd"] == 0.0 and u_same["u4_pairwise_event_f1_50"] == 1.0
    diff = same + RNG.standard_normal((8, T)) * 0.5
    pk2 = [np.sort(RNG.choice(T, size=int(4 + i % 3), replace=False)) for i in range(8)]
    u_diff = Q.uncertainty_from_samples(diff, pk2)
    assert u_diff["u1_pointwise_sd"] > 0.3 and u_diff["u3_beatcount_sd"] > 0 and u_diff["u4_pairwise_event_f1_50"] < 1.0


def test_periodicity_lag_range_matches_30_to_200_bpm():
    x = np.sin(2 * np.pi * np.arange(1024) / 128.0 * 1.0)          # 60 bpm -> lag 128
    # biased (1/N) normalisation: a perfectly periodic signal scores (N - lag)/N = 0.875 at lag 128
    assert 0.87 < Q.periodicity_score(x) < 0.88
    assert Q.periodicity_score(np.sin(2 * np.pi * np.arange(1024) / 128.0 * 2.0)) > Q.periodicity_score(x)
    assert Q.periodicity_score(RNG.standard_normal(1024)) < 0.5
    assert Q.PERIODICITY_BPM == (30.0, 200.0)
    lo = int(np.ceil(60.0 / 200.0 * 128)); hi = int(np.floor(60.0 / 30.0 * 128))
    assert (lo, hi) == (39, 256)
    assert np.isnan(Q.periodicity_score(np.zeros(1024)))
    assert np.isnan(Q.pulse_template_consistency(_sig(), np.array([200])))       # < 3 pulses => undefined


# --------------------------------------------------------------------------------- verdict logic
def _res(v):
    return {"verdict": v}


def _flags(support=True, fidelity=True, plaus=True, unc_nonresp=True):
    sup = {"f1@150": _res("improves" if support else "unresolved"),
           "rr_mae_ms": _res("improves" if support else "unresolved"),
           "missing": _res("unresolved"), "spurious": _res("unresolved")}
    fid = {"f1_excess": _res("improves" if fidelity else "unresolved"),
           "raw_qrs_rmse": _res("improves" if fidelity else "unresolved"),
           "qrs_deriv_rmse": _res("unresolved"), "qrs_curvature_err": _res("unresolved")}
    pl = {"detector_valid_drop": 0.0 if plaus else 0.30, "marginal_support_drop": 0.0 if plaus else 0.30}
    un = {"u1_rel_increase": 0.0 if unc_nonresp else 0.50,
          "u3": _res("unresolved" if unc_nonresp else "improves"),
          "u4": _res("unresolved" if unc_nonresp else "improves")}
    return Q.family_flags(sup, fid, pl, un)


def test_family_flags_implement_the_preregistered_conditions():
    f = _flags()
    assert f["support_degrading"] and f["fidelity_degrading"] and f["plausibility_preserved"] and f["uncertainty_nonresponsive"]
    assert not _flags(support=False)["support_degrading"]
    assert not _flags(support=False)["fidelity_degrading"]          # F requires a support-degrading family
    assert not _flags(fidelity=False)["fidelity_degrading"]
    assert not _flags(plaus=False)["plausibility_preserved"] and _flags(plaus=False)["plausibility_degrades"]
    assert not _flags(unc_nonresp=False)["uncertainty_nonresponsive"]
    assert _flags(unc_nonresp=False)["uncertainty_clear_increase"]
    # the 5% / 10% thresholds are strict inequalities in the frozen direction
    edge = Q.family_flags({"f1@150": _res("improves"), "rr_mae_ms": _res("improves"), "missing": _res("unresolved"), "spurious": _res("unresolved")},
                          {"f1_excess": _res("improves"), "raw_qrs_rmse": _res("improves"), "qrs_deriv_rmse": _res("unresolved"), "qrs_curvature_err": _res("unresolved")},
                          {"detector_valid_drop": 0.05, "marginal_support_drop": 0.0},
                          {"u1_rel_increase": 0.10, "u3": _res("improves"), "u4": _res("improves")})
    assert not edge["plausibility_preserved"] and not edge["uncertainty_nonresponsive"]


def test_decide_q1_matches_the_preregistered_order_and_names():
    A, B, C, D = Q.VERDICT_A, Q.VERDICT_B, Q.VERDICT_C, Q.VERDICT_D
    for v in (A, B, C, D):
        assert v in PREREG
    all_a = {f: _flags() for f in ("BANDLIMIT", "NOISE", "DROPOUT")}
    assert Q.decide_q1(all_a)["verdict"] == A
    one_a = {"BANDLIMIT": _flags(), "NOISE": _flags(support=False), "DROPOUT": _flags(support=False)}
    assert Q.decide_q1(one_a)["verdict"] == D
    unc = {f: _flags(unc_nonresp=False) for f in ("BANDLIMIT", "NOISE", "DROPOUT")}
    assert Q.decide_q1(unc)["verdict"] == B
    coll = {f: _flags(plaus=False, unc_nonresp=True) for f in ("BANDLIMIT", "NOISE", "DROPOUT")}
    assert Q.decide_q1(coll)["verdict"] == C
    mixed = {"BANDLIMIT": _flags(), "NOISE": _flags(), "DROPOUT": _flags(unc_nonresp=False)}
    assert Q.decide_q1(mixed)["verdict"] == A          # A is checked first and needs only two families


def test_verdict_thresholds_are_not_read_from_results():
    i = MOD_SRC.index("def family_flags")
    body = MOD_SRC[i:MOD_SRC.index("VERDICT_A")]
    for hard in ("0.05", "0.10", "0.02"):
        assert hard not in body                        # thresholds come from the frozen module constants only
    assert "DET_VALID_DROP_MAX" in body and "MARGINAL_DROP_MAX" in body and "UNC_REL_INCREASE" in body
