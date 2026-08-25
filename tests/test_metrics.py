import numpy as np
import pytest

from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.metrics import concat_consecutive, evaluate_windows, rhythm_morphology_metrics, signal_metrics, summarize


def test_signal_metrics_identity_and_scale():
    x = np.random.default_rng(0).standard_normal((4, 512))
    m = signal_metrics(x, x)
    assert np.allclose(m["mae"], 0) and np.allclose(m["rmse"], 0) and np.allclose(m["pcc"], 1)
    m2 = signal_metrics(2 * x, x)
    assert np.allclose(m2["pcc"], 1) and (m2["mae"] > 0).all()


def test_rpeak_matching_and_prf():
    ref = np.array([100, 200, 300, 400])
    pred = np.array([102, 205, 350, 398, 450])  # 3 hits (tol 50 ms @128 Hz = 6.4 samples), 2 FP, 1 FN
    m, fp, fn = R.match_rpeaks(ref, pred, fs=128, tol_ms=50)
    assert [i for i, _ in m] == [0, 1, 3] and fp == 2 and fn == 1
    p, r, f = R.prf(len(m), fp, fn)
    assert p == pytest.approx(3 / 5) and r == pytest.approx(3 / 4)


def test_synthetic_ecg_self_consistency(synth_ecg):
    sig, fs = synth_ecg
    rp = R.detect_rpeaks(sig, fs)
    assert 7 <= len(rp) <= 11, rp  # ~70 bpm over 8 s => 9 beats
    assert abs(R.hr_bpm(rp, fs) - 70) < 5
    out = rhythm_morphology_metrics(sig[None], sig[None], fs)
    assert out["rpeak_f1"][0] == 1.0 and out["hr_abs_err"][0] == 0.0 and out["rr_mae_ms"][0] == 0.0
    assert out["morph_corr"][0] == pytest.approx(1.0) and out["qrs_width_err_ms"][0] == 0.0


def test_degradation_is_detected(synth_ecg):
    sig, fs = synth_ecg
    rng = np.random.default_rng(0)
    blurred = np.convolve(sig, np.ones(5) / 5, mode="same") + 0.05 * rng.standard_normal(sig.size)  # mild blur + noise
    out = rhythm_morphology_metrics(blurred[None], sig[None], fs)
    assert out["rpeak_f1"][0] > 0.5, out  # beats still found ...
    assert out["morph_corr"][0] < 0.999 and out["qrs_width_err_ms"][0] >= 0.0  # ... but shape is measurably degraded
    shifted = np.roll(sig, int(0.2 * fs))  # 200 ms shift => no peak matches within 50 ms
    out2 = rhythm_morphology_metrics(shifted[None], sig[None], fs)
    assert out2["rpeak_f1"][0] < 0.5


def test_no_beats_gives_nan_not_zero():
    flat = np.zeros((1, 1024))
    out = rhythm_morphology_metrics(flat, flat, 128)
    assert np.isnan(out["hr_abs_err"][0]) and out["rpeak_f1"][0] == 0.0


def test_concat_and_summarize(synth_ecg):
    sig, fs = synth_ecg
    w4 = sig.reshape(2, -1)  # two 4 s windows
    assert concat_consecutive(w4, 2).shape == (1, 1024)
    res = evaluate_windows(w4, w4, fs=fs, hr_window_segments=2)
    s = summarize(res, n_boot=50)
    assert s["mae"]["mean"] == 0.0 and s["rpeak_f1"]["mean"] == 1.0 and s["rpeak_f1"]["n"] == 1
