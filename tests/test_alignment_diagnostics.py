import numpy as np

from ppg2ecg.evaluation.alignment_diagnostics import FS, beat_level_analysis, beat_segments_gt, event_timing, global_lag, oracle_local_shift, segment_stats, shift_crop


def synth_ecg(n_s=8, hr=72, fs=FS, phase=0.0, amp=1.0, seed=0):
    t = np.arange(n_s * fs) / fs
    f = hr / 60
    ph = (t * f + phase) % 1.0
    qrs = np.exp(-((ph - 0.5) ** 2) / (2 * 0.012**2)) - 0.25 * np.exp(-((ph - 0.47) ** 2) / (2 * 0.008**2)) - 0.3 * np.exp(-((ph - 0.53) ** 2) / (2 * 0.008**2))
    tw = 0.25 * np.exp(-((ph - 0.72) ** 2) / (2 * 0.05**2))
    rng = np.random.default_rng(seed)
    return amp * (qrs + tw) + rng.normal(0, 0.005, len(t))


def gt_rpeaks(sig, fs=FS):
    from scipy.signal import find_peaks

    pk, _ = find_peaks(sig, distance=int(0.4 * fs), height=0.5 * sig.max())
    return pk


def test_A_zero_shift_identity_and_J_deterministic():
    y = synth_ecg()
    lag, c = global_lag(y, y, 32)
    assert lag == 0 and c > 0.999
    r = gt_rpeaks(y)
    a1 = beat_level_analysis(y, y, r)
    a2 = beat_level_analysis(y, y, r)
    assert np.all(a1["shift_samples"] == 0) and np.allclose(a1["oracle_corr"], 1.0, atol=1e-6)
    assert np.array_equal(a1["oracle_corr"], a2["oracle_corr"])  # deterministic


def test_B_known_shift_recovery_and_I_no_wraparound():
    y = synth_ecg()
    for k in (-20, -7, 5, 25):
        pred = np.zeros_like(y)
        if k >= 0:
            pred[k:] = y[: len(y) - k]  # prediction late by k (no wrap: the vacated edge is zero, not y's tail)
        else:
            pred[: len(y) + k] = y[-k:]
        lag, _ = global_lag(pred, y, 32)
        assert lag == k, (k, lag)
        pa, ga, off = shift_crop(pred, y, lag)
        assert len(pa) == len(ga) == len(y) - abs(k) and np.allclose(pa, ga, atol=1e-9)
    # a lag beyond the search range must not be "found" by wrapping
    pred = np.roll(y, 300)
    lag, _ = global_lag(pred, y, 32)
    assert abs(lag) <= 32


def test_C_no_amplitude_cheating_and_D_translation_only():
    y = synth_ecg()
    r = gt_rpeaks(y)
    att = 0.3 * y + 0.1  # attenuated + offset, perfectly aligned
    res = beat_level_analysis(att, y, r)
    assert np.all(res["shift_samples"] == 0)  # alignment does not move an already-aligned signal
    assert np.allclose(res["oracle_p2p_ratio"], 0.3, atol=0.02) and np.allclose(res["oracle_slope_ratio"], 0.3, atol=0.02)
    assert np.allclose(res["oracle_corr"], 1.0, atol=1e-3)  # correlation is scale-free; amplitude ratios expose the attenuation
    # translation only: a locally shifted beat is recovered with the same amplitude statistics as the unshifted one
    sh = np.zeros_like(y)
    sh[10:] = att[:-10]
    res2 = beat_level_analysis(sh, y, r)
    assert np.all(res2["shift_samples"] == 10) and np.allclose(res2["oracle_p2p_ratio"], res["oracle_p2p_ratio"], atol=1e-6)


def test_E_flattened_prediction_no_fake_morphology():
    y = synth_ecg()
    r = gt_rpeaks(y)
    flat = np.full_like(y, y.mean())
    res = beat_level_analysis(flat, y, r)
    assert res["n_beats"] > 5
    assert np.all(np.isnan(res["oracle_corr"]) | (res["oracle_corr"] <= 0.0)) and np.all(res["oracle_p2p_ratio"] < 1e-6)
    smooth = np.convolve(y, np.ones(9) / 9, mode="same")  # smoothed (~70 ms box): shape correlation survives better than sharpness
    res2 = beat_level_analysis(smooth, y, r)
    assert res2["oracle_corr"].mean() > res2["oracle_slope_ratio"].mean() and res2["oracle_slope_ratio"].mean() < 0.7 and res2["oracle_qrs_energy_ratio"].mean() < 0.9
    assert np.all(np.abs(res2["shift_samples"]) <= 2)  # smoothing an asymmetric QRS moves the correlation peak by at most ~16 ms


def test_F_G_missing_and_spurious_events_counted():
    y = synth_ecg(hr=60)
    r_gt = gt_rpeaks(y)
    pred = y.copy()
    # remove one beat (flatten its window) and add a spurious beat between two others
    k = r_gt[3]
    pred[k - 30 : k + 30] = pred[k - 31]
    mid = (r_gt[5] + r_gt[6]) // 2
    seg = y[r_gt[5] - 30 : r_gt[5] + 30]
    pred[mid - 30 : mid + 30] = seg
    ev = event_timing(y, pred)
    assert ev["n_missing"] >= 1 and ev["n_spurious"] >= 1
    assert ev["n_matched"] + ev["n_missing"] == ev["n_ref"] and ev["n_matched"] + ev["n_spurious"] == ev["n_pred"]


def test_H_detector_independent_runs_without_predicted_peaks():
    y = synth_ecg()
    r = gt_rpeaks(y)
    flat = np.zeros_like(y)
    ev = event_timing(y, flat)
    assert ev["n_pred"] == 0 and ev["n_matched"] == 0
    res = beat_level_analysis(flat, y, r)  # still analyses every GT beat
    assert res["n_beats"] == len([1 for a, b, _ in beat_segments_gt(y, r, margin=19)[0]])


def test_local_shift_bounds_and_segment_stats():
    y = synth_ecg()
    r = gt_rpeaks(y)
    segs, _ = beat_segments_gt(y, r, margin=19)
    a, b, rl = segs[2]
    d, c = oracle_local_shift(y, y, a, b, 19)
    assert d == 0 and c > 0.999
    st = segment_stats(y[a:b], y[a:b], r_index=rl)
    assert abs(st["qrs_energy_ratio"] - 1) < 1e-9 and abs(st["r_amp_ratio"] - 1) < 1e-9 and st["hf_ratio_pred"] == st["hf_ratio_gt"]
