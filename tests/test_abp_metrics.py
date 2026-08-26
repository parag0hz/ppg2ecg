import numpy as np

from ppg2ecg.evaluation.abp_metrics import FS, evaluate_abp, systolic_peaks, gt_prominence, window_metrics


def synth_abp(n_s=8, hr=72, sbp=120, dbp=70, fs=FS, phase=0.0):
    t = np.arange(n_s * fs) / fs
    f = hr / 60
    ph = (t * f + phase) % 1.0
    pulse = np.exp(-((ph - 0.15) ** 2) / (2 * 0.05**2)) + 0.35 * np.exp(-((ph - 0.45) ** 2) / (2 * 0.08**2))  # systolic + dicrotic bump
    pulse = (pulse - pulse.min()) / (pulse.max() - pulse.min())
    return dbp + (sbp - dbp) * pulse


def test_perfect_prediction():
    y = synth_abp()
    m = window_metrics(y.copy(), y)
    assert m["sbp_win_ae"] == 0 and m["dbp_win_ae"] == 0 and m["sbp_beat_ae"] == 0 and m["peak_f1"] == 1.0 and abs(m["morph_corr"] - 1) < 1e-9
    assert m["rmse"] == 0 and m["slope_ratio"] == 1.0 and m["pulse_count_ratio"] == 1.0 and 9 <= m["n_peaks_gt"] <= 10 and m["peak_region_frac"] > 0.3


def test_attenuated_and_flat_predictions():
    y = synth_abp()
    att = y.mean() + 0.3 * (y - y.mean())  # amplitude-attenuated, aligned
    m = window_metrics(att, y)
    assert m["peak_f1"] == 1.0 and abs(m["amp_ratio"] - 0.3) < 0.02 and abs(m["pp_ratio"] - 0.3) < 0.05 and m["slope_ratio"] < 0.35 and m["morph_corr"] > 0.99
    assert m["sbp_beat_ae"] > 20 and m["dbp_beat_ae"] > 5  # attenuation shows up as SBP/DBP error even with perfect timing
    flat = np.full_like(y, y.mean())
    mf = window_metrics(flat, y)
    assert mf["n_peaks_pr"] == 0 and mf["peak_f1"] == 0.0 and np.isnan(mf["morph_corr"]) and mf["rmse"] < m["rmse"] + 20  # flat has no peaks, but a deceptively modest RMSE
    assert mf["rmse_peak"] > mf["rmse_nonpeak"]


def test_phase_shift_hurts_timing_not_pressure():
    y = synth_abp()
    sh = synth_abp(phase=0.15)  # 125 ms shift at 72 bpm -> beyond the 100 ms tolerance
    m = window_metrics(sh, y)
    assert m["sbp_beat_ae"] < 1 and m["dbp_beat_ae"] < 1 and m["peak_f1"] < 0.5 and m["rmse"] > 15


def test_batch_and_prominence():
    y = np.stack([synth_abp(), synth_abp(hr=100, sbp=150, dbp=80)])
    pw = evaluate_abp(y * 0.5 + y.mean(axis=1, keepdims=True) * 0.5, y)
    assert pw["sbp_win_ae"].shape == (2,) and np.all(pw["peak_f1"] == 1.0)
    assert systolic_peaks(np.zeros(1024), gt_prominence(y[0])).size == 0
