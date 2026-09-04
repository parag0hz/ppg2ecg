import subprocess
import sys

import numpy as np
import pytest

from ppg2ecg.evaluation import metrics as M
from ppg2ecg.evaluation import paper_metrics as PM

FS = 128


# --------------------------------------------------------------------------- batch contract
@pytest.mark.parametrize("fn", [PM.mse, PM.prd, PM.cosine_similarity, PM.snr_db, PM.discrete_frechet, PM.dtw_distance,
                                PM.region_rmse, PM.rpeak_prf_at, PM.sbp_abs_err_mmhg, PM.dbp_abs_err_mmhg,
                                PM.respiratory_rate_abs_err_bpm, PM.pooled_mae, PM.pooled_rmse, PM.paper_metric_table])
def test_1d_input_raises_instead_of_broadcasting(fn):
    x = np.linspace(0, 1, 64)
    with pytest.raises(ValueError, match="2-D"):
        fn(x, x)


@pytest.mark.parametrize("fn", [PM.respiratory_rate_bpm, PM.default_feature_map])
def test_1d_single_argument_raises(fn):
    with pytest.raises(ValueError, match="2-D"):
        fn(np.linspace(0, 1, 64))


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="same shape"):
        PM.mse(np.zeros((2, 8)), np.zeros((2, 9)))


def test_every_function_returns_one_value_per_window():
    rng = np.random.default_rng(0)
    t = rng.standard_normal((5, 256))
    p = t + 0.1 * rng.standard_normal((5, 256))
    for v in (PM.mse(p, t), PM.cosine_similarity(p, t), PM.snr_db(p, t), PM.discrete_frechet(p, t), PM.dtw_distance(p, t),
              PM.prd(p, t)["prd_raw"], PM.sbp_abs_err_mmhg(p, t), PM.respiratory_rate_abs_err_bpm(p, t, FS)):
        assert v.shape == (5,) and v.dtype == np.float64


# --------------------------------------------------------------------------- pointwise closed forms
def test_pointwise_closed_forms():
    t = np.array([[1.0, -1.0, 1.0, -1.0]])
    assert PM.mse(t + 0.5, t) == pytest.approx(0.25)
    assert PM.snr_db(t + 0.5, t) == pytest.approx(10 * np.log10(4.0))          # sum t^2 = 4, sum err^2 = 1
    assert PM.cosine_similarity(t, t) == pytest.approx(1.0)
    assert PM.cosine_similarity(np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]])) == pytest.approx(0.0)
    assert np.isposinf(PM.snr_db(t, t)[0])                                     # exact reconstruction, not a clipped max
    assert np.isnan(PM.snr_db(np.zeros((1, 4)), np.zeros((1, 4)))[0])          # 0/0 is undefined, not 0 dB


def test_prd_variants_differ_on_offset_target_and_coincide_when_zero_mean():
    t0 = np.array([[1.0, -1.0, 1.0, -1.0]])                                    # zero mean
    z = PM.prd(t0 + 0.1, t0)
    assert z["prd_raw"] == pytest.approx(10.0) and z["prd_meansub"] == pytest.approx(10.0)
    t3 = t0 + 3.0                                                              # sum t^2 = 40, sum (t-mean)^2 = 4
    o = PM.prd(t3 + 0.1, t3)
    assert o["prd_raw"] == pytest.approx(100 * np.sqrt(0.04 / 40.0))
    assert o["prd_meansub"] == pytest.approx(10.0)
    assert o["prd_raw"][0] < o["prd_meansub"][0]                               # never collapse the two columns


def test_pooled_rmse_is_not_the_mean_of_window_rmses():
    t = np.zeros((2, 4))
    p = np.array([[1.0, 1, 1, 1], [3.0, 3, 3, 3]])                             # per-window RMSE 1 and 3
    assert PM.mean_window_rmse(p, t) == pytest.approx(2.0)
    assert PM.pooled_rmse(p, t) == pytest.approx(np.sqrt(5.0))                 # sqrt taken once, after pooling
    assert PM.pooled_rmse(p, t) > PM.mean_window_rmse(p, t)                    # Jensen, always
    assert PM.pooled_mae(p, t) == pytest.approx(2.0)


# --------------------------------------------------------------------------- discrete Frechet / DTW
def test_discrete_frechet_identity_offset_and_symmetry():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 97))
    assert np.allclose(PM.discrete_frechet(x, x), 0.0)
    for c in (0.0, 0.7, -2.5):
        assert PM.discrete_frechet(x + c, x) == pytest.approx(abs(c))          # the (0,0) pair alone forces |c|
    y = rng.standard_normal((3, 97))
    assert np.allclose(PM.discrete_frechet(x, y), PM.discrete_frechet(y, x))


def test_discrete_frechet_matches_the_textbook_recurrence():
    def brute(p, t):
        ca = np.full((len(p), len(t)), np.inf)
        for i in range(len(p)):
            for j in range(len(t)):
                d = abs(p[i] - t[j])
                prev = [ca[a, b] for a, b in ((i - 1, j), (i - 1, j - 1), (i, j - 1)) if a >= 0 and b >= 0]
                ca[i, j] = d if not prev else max(d, min(prev))
        return ca[-1, -1]

    rng = np.random.default_rng(2)
    for _ in range(20):
        p, t = rng.standard_normal(11), rng.standard_normal(11)
        assert PM.discrete_frechet(p[None], t[None])[0] == pytest.approx(brute(p, t))


def test_discrete_frechet_is_not_dtw_and_is_bounded_by_it():
    p = np.array([[0.0, 0.0, 0.0, 0.0]])
    t = np.array([[0.0, 1.0, 1.0, 0.0]])                                       # two unit-cost columns to cross
    assert PM.discrete_frechet(p, t) == pytest.approx(1.0)                     # max over the pairing
    assert PM.dtw_distance(p, t, band=None) == pytest.approx(2.0)              # sum over the warping
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((6, 128)), rng.standard_normal((6, 128))
    assert (PM.dtw_distance(a, b, band=None) >= PM.discrete_frechet(a, b) - 1e-9).all()


def test_dtw_identity_band_and_monotonicity():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 64))
    assert np.allclose(PM.dtw_distance(x, x, band=8), 0.0)
    y = rng.standard_normal((2, 64))
    wide, narrow = PM.dtw_distance(x, y, band=None), PM.dtw_distance(x, y, band=2)
    assert (narrow >= wide - 1e-9).all()                                       # a band can only remove paths


# --------------------------------------------------------------------------- Gaussian / FID-style Frechet
def test_gaussian_frechet_closed_forms():
    d = 5
    eye = np.eye(d)
    mu = np.zeros(d)
    assert PM.gaussian_frechet(mu, eye, mu, eye) == pytest.approx(0.0, abs=1e-9)
    shift = np.full(d, 0.5)
    assert PM.gaussian_frechet(shift, eye, mu, eye) == pytest.approx(d * 0.25)
    assert PM.gaussian_frechet(mu, 4 * eye, mu, 9 * eye) == pytest.approx(d * (2 - 3) ** 2)


def test_fid_frechet_self_is_zero_and_symmetric():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((256, 8))
    b = rng.standard_normal((256, 8)) + 1.0
    assert PM.fid_frechet(a, a) < 1e-6
    assert PM.fid_frechet(a, b) == pytest.approx(PM.fid_frechet(b, a), rel=1e-9)
    assert PM.fid_frechet(a, b) > PM.fid_frechet(a, a)
    with pytest.raises(ValueError, match="dimensionality"):
        PM.fid_frechet(a, rng.standard_normal((256, 7)))


def test_kanflow_fd_regimes():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((40, 64))
    assert PM.kanflow_fd(x, x, small_set=0) < 1e-6                             # raw flattened-waveform regime, deterministic
    same = PM.kanflow_fd(x, x)                                                 # PCA<=32 + 5 bootstrap trials
    shifted = PM.kanflow_fd(x + 3.0, x)
    assert same < shifted and shifted > 1.0
    assert PM.kanflow_fd(x, x, seed=0) == PM.kanflow_fd(x, x, seed=0)          # seeded => reproducible


def test_default_feature_map_shape_and_determinism():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((6, 256))
    f = PM.default_feature_map(x, FS)
    assert f.shape == (6, 36) and np.isfinite(f).all()
    assert np.array_equal(f, PM.default_feature_map(x, FS))


# --------------------------------------------------------------------------- R-peak driven metrics
def _peak_pair(ref, hyp, n_samples=512):
    dummy = np.zeros((1, n_samples))
    return dummy, dummy, ([np.asarray(ref)], [np.asarray(hyp)])


def test_rpeak_prf_at_is_tolerance_sensitive():
    p, t, peaks = _peak_pair([100, 200, 300, 400], [102, 205, 350, 398, 450])
    at50 = PM.rpeak_prf_at(p, t, FS, 50.0, peaks=peaks)                        # 50 ms = 6.4 samples
    assert (at50["n_tp"][0], at50["n_fp"][0], at50["n_fn"][0]) == (3, 2, 1)
    assert at50["rpeak_precision"][0] == pytest.approx(0.6) and at50["rpeak_recall"][0] == pytest.approx(0.75)
    at25 = PM.rpeak_prf_at(p, t, FS, 25.0, peaks=peaks)                        # 25 ms = 3.2 samples, drops the 5-sample hit
    assert (at25["n_tp"][0], at25["n_fp"][0], at25["n_fn"][0]) == (2, 3, 2)
    at100 = PM.rpeak_prf_at(p, t, FS, 100.0, peaks=peaks)                      # 100 ms = 12.8 samples, still 3 (350/450 are 50 off)
    assert at100["n_tp"][0] == 3
    assert at25["rpeak_f1"][0] < at50["rpeak_f1"][0] <= at100["rpeak_f1"][0]


def test_micro_and_macro_f1_aggregate_differently():
    tp, fp, fn = np.array([3.0, 1.0]), np.array([2.0, 0.0]), np.array([1.0, 0.0])
    f1 = np.array([2 * 0.6 * 0.75 / (0.6 + 0.75), 1.0])
    assert PM.micro_f1(tp, fp, fn) == pytest.approx(2 * (4 / 6) * (4 / 5) / ((4 / 6) + (4 / 5)))
    n_ref = np.array([4.0, 1.0])                                               # both segments are evaluable
    assert PM.macro_f1(f1, n_ref) == (pytest.approx(float(np.mean(f1))), 0)
    assert PM.micro_f1(tp, fp, fn) < PM.macro_f1(f1, n_ref)[0]                  # the sign KANFlow reports for short windows
    assert np.isnan(PM.macro_f1(np.array([np.nan, np.nan]), n_ref)[0])
    tp_nan = np.array([3.0, np.nan])
    assert PM.micro_f1(tp_nan, fp, fn) == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))   # nan windows drop out of the pool


def test_region_rmse_splits_the_error_by_region():
    n = 512
    t = np.zeros((1, n))
    p = np.zeros((1, n))
    peaks = [np.array([100, 300])]
    half = int(round(PM.QRS_HALF_MS / 1000 * FS))                              # 6 samples at 128 Hz
    p[0, 100 - half:100 + half + 1] = 2.0                                      # error lives only inside the QRS regions
    out = PM.region_rmse(p, t, FS, peaks=peaks)
    assert out["non_qrs_rmse"][0] == pytest.approx(0.0)
    assert out["qrs_region_rmse"][0] == pytest.approx(np.sqrt((2 * half + 1) * 4.0 / (2 * (2 * half + 1))))
    assert out["qrs_region_frac"][0] == pytest.approx(2 * (2 * half + 1) / n)
    assert PM.qrs_region_rmse(p, t, FS, peaks=peaks) == out["qrs_region_rmse"]
    assert PM.non_qrs_rmse(p, t, FS, peaks=peaks) == out["non_qrs_rmse"]


def test_beat_level_metrics_reproduce_rhythm_morphology_metrics(synth_ecg):
    sig, fs = synth_ecg
    t = sig[None]
    p = np.convolve(sig, np.ones(3) / 3, mode="same")[None]
    ref = M.rhythm_morphology_metrics(p, t, fs)
    got = PM.beat_level_metrics(p, t, fs)
    for k in ("hr_ref", "hr_pred", "hr_abs_err", "rr_mae_ms", "qrs_width_err_ms", "morph_corr", "n_ref_beats", "n_pred_beats"):
        np.testing.assert_allclose(got[k], ref[k], equal_nan=True, err_msg=k)


def test_hr_pooled_errors():
    hr_ref = np.array([60.0, 80.0, np.nan])
    hr_pred = np.array([63.0, 76.0, 100.0])
    assert PM.hr_mae_bpm(hr_pred, hr_ref) == pytest.approx(3.5)
    assert PM.hr_rmse_bpm(hr_pred, hr_ref) == pytest.approx(np.sqrt((9 + 16) / 2))
    assert np.isnan(PM.hr_mae_bpm(np.array([np.nan]), np.array([np.nan])))


# --------------------------------------------------------------------------- PENGUIN's non-ECG columns
def test_respiratory_rate_dominant_frequency():
    fs, dur = 4, 60                                                            # 60 s window => 1/60 Hz = 1 bpm resolution
    n = fs * dur
    tt = np.arange(n) / fs
    ref = np.sin(2 * np.pi * 0.25 * tt)[None]                                  # 15 breaths/min, exactly on a bin
    pred = np.sin(2 * np.pi * (1 / 3) * tt)[None]                              # 20 breaths/min
    assert PM.respiratory_rate_bpm(ref, fs) == pytest.approx(15.0)
    assert PM.respiratory_rate_bpm(pred, fs) == pytest.approx(20.0)
    assert PM.respiratory_rate_abs_err_bpm(pred, ref, fs) == pytest.approx(5.0)


def test_sbp_dbp_errors_are_window_extrema():
    t = np.array([[80.0, 120.0, 95.0]])
    p = np.array([[85.0, 130.0, 90.0]])
    assert PM.sbp_abs_err_mmhg(p, t) == pytest.approx(10.0)
    assert PM.dbp_abs_err_mmhg(p, t) == pytest.approx(5.0)


# --------------------------------------------------------------------------- nan behaviour and the table
def test_nan_window_yields_nan_everywhere_and_never_crashes(synth_ecg):
    sig, fs = synth_ecg
    t = np.stack([sig, sig])
    p = np.stack([sig + 0.02, sig.copy()])
    p[1, 500] = np.nan
    tab = PM.paper_metric_table(p, t, fs)
    for k, v in tab.items():
        assert np.isnan(v[1]), k                                               # the nan window is nan in every column
    for k in ("mae", "rmse", "mse", "prd_raw", "cosine_similarity", "snr_db", "discrete_frechet", "dtw",
              "rpeak_f1_50ms", "hr_abs_err", "qrs_region_rmse", "non_qrs_rmse"):
        assert np.isfinite(tab[k][0]), k                                       # ... and the clean window is unaffected
    t_inf = t.copy()
    t_inf[1, 3] = np.inf
    assert np.isnan(PM.discrete_frechet(p[:1].repeat(2, 0), t_inf)[1])


def test_paper_metric_table_columns_are_all_declared_in_the_spec(synth_ecg):
    sig, fs = synth_ecg
    x = np.stack([sig, sig + 0.05])
    tab = PM.paper_metric_table(x, np.stack([sig, sig]), fs)
    missing = sorted(set(tab) - set(PM.SPEC_BY_KEY))
    assert not missing, missing
    assert all(v.shape == (2,) for v in tab.values())


def test_paper_metric_spec_is_well_formed():
    keys = [s.key for s in PM.PAPER_METRIC_SPEC]
    assert len(keys) == len(set(keys))
    for s in PM.PAPER_METRIC_SPEC:
        assert s.orientation in PM.ORIENTATIONS, s.key
        assert s.level in ("window", "dataset") and s.modality in ("ecg", "resp", "abp"), s.key
        assert s.reported_by and s.definition.strip(), s.key
    assert PM.SPEC_BY_KEY["fid_ecgfounder"].impl is None                       # the one metric we cannot compute
    assert {s.key for s in PM.PAPER_METRIC_SPEC if "PENGUIN" in s.reported_by} >= {
        "hr_mae_bpm", "resp_rate_abs_err_bpm", "sbp_abs_err_mmhg", "dbp_abs_err_mmhg"}
    assert {s.key for s in PM.PAPER_METRIC_SPEC if "KANFlow" in s.reported_by} >= {
        "pooled_mae", "pooled_rmse", "kanflow_fd", "micro_f1", "macro_f1", "rr_mae_ms_macro", "hr_mae_bpm"}


def test_quadratic_and_rpeak_flags_drop_their_columns(synth_ecg):
    sig, fs = synth_ecg
    x, y = sig[None], (sig + 0.05)[None]
    fast = PM.paper_metric_table(y, x, fs, with_rpeaks=False, with_quadratic=False)
    assert "discrete_frechet" not in fast and "rpeak_f1_50ms" not in fast and "mae" in fast
    quad = PM.paper_metric_table(y, x, fs, with_rpeaks=False, with_quadratic=True)
    assert "dtw" in quad and "hr_ref" not in quad


def test_paper_metric_pooled_reuses_the_table(synth_ecg):
    sig, fs = synth_ecg
    rng = np.random.default_rng(8)
    t = np.stack([sig] * 6)
    p = t + 0.05 * rng.standard_normal(t.shape)
    tab = PM.paper_metric_table(p, t, fs)
    pooled = PM.paper_metric_pooled(p, t, fs, table=tab)
    assert set(pooled) == {"pooled_mae", "pooled_rmse", "mean_window_rmse", "kanflow_fd", "fid_default_features",
                           "micro_f1", "macro_f1", "rr_mae_ms_macro", "hr_mae_bpm", "hr_rmse_bpm",
                           "n_excluded_windows", "n_macro_f1_excluded_segments"}
    assert all(np.isfinite(v) for v in pooled.values())
    assert pooled["n_excluded_windows"] == 0 and pooled["n_macro_f1_excluded_segments"] == 0   # nothing to drop here
    assert pooled["pooled_mae"] == pytest.approx(float(np.mean(tab["mae"])))
    assert pooled["macro_f1"] == pytest.approx(float(np.nanmean(tab["rpeak_f1_50ms"])))
    assert set(pooled) <= set(PM.SPEC_BY_KEY)


# --------------------------------------------------------------------------- nan-mask leak (MAJOR 1 / MAJOR 2)
def test_paper_metric_pooled_excludes_nan_windows_from_every_pooled_number():
    """A window with a nan must be DROPPED from the pooled numbers, never scored as an all-zero waveform."""
    rng = np.random.default_rng(0)
    t = rng.standard_normal((4, 128))
    p = t + 1.0                                                                # |p - t| = 1 on every finite sample
    p[3, 7] = np.nan                                                           # window 3 is unusable
    out = PM.paper_metric_pooled(p, t, FS, with_rpeaks=False, with_quadratic=False)
    assert out["n_excluded_windows"] == 1                                      # the exclusion is never invisible
    assert out["pooled_mae"] == pytest.approx(1.0)                             # 0.75 if the zeroed window is scored
    assert out["pooled_rmse"] == pytest.approx(1.0)                            # sqrt(0.75) if it is scored
    assert out["mean_window_rmse"] == pytest.approx(1.0)                       # 0.75 if it is scored
    assert out["pooled_mae"] == pytest.approx(PM.pooled_mae(p, t))             # agrees with the standalone functions
    assert out["pooled_rmse"] == pytest.approx(PM.pooled_rmse(p, t))
    assert out["mean_window_rmse"] == pytest.approx(PM.mean_window_rmse(p, t))
    assert out["kanflow_fd"] == pytest.approx(PM.kanflow_fd(p, t))


def test_paper_metric_pooled_agrees_whether_or_not_the_table_is_supplied(synth_ecg):
    """The two call paths (table supplied vs. computed internally) must agree on EVERY key, nan windows included."""
    sig, fs = synth_ecg
    t = np.stack([sig, sig, sig])
    p = t + 0.05 * np.random.default_rng(1).standard_normal(t.shape)
    p[2, 500] = np.nan
    internal = PM.paper_metric_pooled(p, t, fs)
    supplied = PM.paper_metric_pooled(p, t, fs, table=PM.paper_metric_table(p, t, fs))
    assert set(internal) == set(supplied)
    for k in internal:
        assert internal[k] == supplied[k] or (np.isnan(internal[k]) and np.isnan(supplied[k])), k
    assert internal["n_excluded_windows"] == 1
    assert internal["n_macro_f1_excluded_segments"] == 1                       # the nan window is not evaluable
    assert internal["macro_f1"] == pytest.approx(1.0)                          # 2/3 if the nan window is scored F1 = 0


# --------------------------------------------------------------------------- macro-F1 evaluability (MINOR 3)
def test_macro_f1_excludes_segments_with_no_reference_beats():
    f1 = np.array([1.0, 0.0, 0.0])
    value, n_excluded = PM.macro_f1(f1, np.array([5.0, 4.0, 0.0]))             # window 2 has no reference beat
    assert value == pytest.approx(0.5) and n_excluded == 1                     # mean over windows 0 and 1 only
    kept, none_excluded = PM.macro_f1(f1, np.array([5.0, 4.0, 3.0]))           # window 2 IS evaluable here ...
    assert kept == pytest.approx(1.0 / 3.0) and none_excluded == 0             # ... a genuine recall failure scores 0
    v, n = PM.macro_f1(np.array([np.nan, 0.0]), np.array([np.nan, 0.0]))       # nothing evaluable at all
    assert np.isnan(v) and n == 2
    assert "reference beats" in PM.SPEC_BY_KEY["macro_f1"].definition          # the rule is in the report spec too


# --------------------------------------------------------------------------- region width docstring (MINOR 4)
def test_region_rmse_docstring_states_the_implemented_region_width():
    half = int(round(PM.QRS_HALF_MS / 1000.0 * FS))
    assert (half, 2 * half + 1) == (6, 13)                                     # what the code actually covers
    frac = PM.region_rmse(np.zeros((1, 512)), np.zeros((1, 512)), FS, peaks=[np.array([100])])["qrs_region_frac"][0]
    assert frac == pytest.approx(13 / 512)
    assert "13 samples" in PM.region_rmse.__doc__ and "101 samples" not in PM.region_rmse.__doc__


# --------------------------------------------------------------------------- complex sqrtm guard (MINOR 5)
def test_gaussian_frechet_raises_on_a_materially_complex_sqrtm():
    with pytest.raises(ValueError, match="imaginary"):
        PM.gaussian_frechet(np.zeros(2), np.eye(2), np.zeros(2), -np.eye(2))   # sqrtm(-I) = i I


def test_complex_sqrtm_guard_survives_python_O():
    """The guard must not be an `assert`: python -O strips those and silently .real()s a complex result."""
    code = ("import numpy as np\nfrom ppg2ecg.evaluation import paper_metrics as PM\n"
            "print(PM.gaussian_frechet(np.zeros(2), np.eye(2), np.zeros(2), -np.eye(2)))\n")
    r = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert r.returncode != 0 and "ValueError" in r.stderr, r.stdout + r.stderr


# --------------------------------------------------------------------------- Sakoe-Chiba radius (MINOR 6)
def test_dtw_band_radius_matches_a_brute_force_banded_dp():
    """Pin the band EXACTLY: |i - j| <= band, no wider. Compared against an independent O(T^2) banded DP."""
    def brute(p, t, band):
        ca = np.full((len(p), len(t)), np.inf)
        for i in range(len(p)):
            for j in range(len(t)):
                if abs(i - j) > band:
                    continue                                                   # outside the Sakoe-Chiba band
                d = abs(p[i] - t[j])
                prev = [ca[a, b] for a, b in ((i - 1, j), (i - 1, j - 1), (i, j - 1)) if a >= 0 and b >= 0]
                ca[i, j] = d if not prev else d + min(prev)
        return ca[-1, -1]

    rng = np.random.default_rng(11)
    for band in (1, 2, 5):
        for _ in range(8):
            p, t = rng.standard_normal(13), rng.standard_normal(13)
            assert PM.dtw_distance(p[None], t[None], band=band)[0] == pytest.approx(brute(p, t, band)), band


# --------------------------------------------------------------------------- modality contract (MINOR 7)
def test_paper_metric_table_emits_only_ecg_modality_columns(synth_ecg):
    sig, fs = synth_ecg
    tab = PM.paper_metric_table(np.stack([sig, sig + 0.05]), np.stack([sig, sig]), fs)
    assert {PM.SPEC_BY_KEY[k].modality for k in tab} == {"ecg"}
    non_ecg = {s.key for s in PM.PAPER_METRIC_SPEC if s.modality != "ecg"}
    assert non_ecg == {"resp_rate_abs_err_bpm", "sbp_abs_err_mmhg", "dbp_abs_err_mmhg"}
    assert not non_ecg & set(tab)                                              # never "missing", just another modality
    assert all(callable(f) for f in (PM.respiratory_rate_abs_err_bpm, PM.sbp_abs_err_mmhg, PM.dbp_abs_err_mmhg))
    assert "ecg-modality columns only" in PM.__doc__.lower()
