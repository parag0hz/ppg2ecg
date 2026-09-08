"""D3 metric ports (docs/D3_PENGUIN_SIX_DATASET_PREREGISTRATION.md §5)."""
from __future__ import annotations

import numpy as np
import pytest

from ppg2ecg.evaluation import penguin_metrics as PM


def test_sbp_dbp_are_per_window_max_and_min_differences_in_label_units():
    t = np.array([[80.0, 120.0, 90.0], [70.0, 140.0, 100.0]])
    p = np.array([[85.0, 115.0, 95.0], [60.0, 150.0, 90.0]])
    assert PM.sbp_error(p, t).tolist() == [5.0, 10.0]     # |115-120|, |150-140|
    assert PM.dbp_error(p, t).tolist() == [5.0, 10.0]     # |85-80|,  |60-70|
    assert PM.sbp_error(t, t).sum() == 0.0


def test_dominant_bpm_recovers_a_known_sinusoid():
    fs, T = 128, 128 * 60                                  # 60 s so the FFT grid contains 1/60 Hz multiples
    n = np.arange(T)
    for bpm in (12.0, 20.0, 30.0):                         # 0.2, 1/3, 0.5 Hz — all on the grid
        x = np.sin(2 * np.pi * (bpm / 60.0) * n / fs)[None, :]
        assert PM.dominant_bpm(x, fs)[0] == pytest.approx(bpm, abs=1e-9)


def test_resp_rate_error_is_zero_for_an_identical_low_frequency_signal():
    fs, T = 128, 128 * 60
    x = np.sin(2 * np.pi * (15 / 60.0) * np.arange(T) / fs)[None, :]
    assert PM.resp_rate_error(x, x, fs)[0] == pytest.approx(0.0, abs=1e-9)


def test_resp_rate_error_recovers_a_known_offset():
    fs, T = 128, 128 * 60
    n = np.arange(T)
    t = np.sin(2 * np.pi * (12 / 60.0) * n / fs)[None, :]
    p = np.sin(2 * np.pi * (18 / 60.0) * n / fs)[None, :]
    assert PM.resp_rate_error(p, t, fs)[0] == pytest.approx(6.0, abs=1e-6)


def test_prediction_only_low_pass_is_preserved_as_upstream_asymmetry():
    """Upstream filters the PREDICTION at 1 Hz and leaves the TARGET untouched (help_func.py:190-193).

    The asymmetry is demonstrated on a signal whose high-frequency component DOMINATES: the target then
    follows its 5 Hz peak (300 bpm) because nothing removes it, while the same waveform fed as the
    prediction is low-passed first and falls back to its 15 bpm component. An amplitude below 1.0 would
    not show this -- the 0.25 Hz peak would still win the argmax -- so the component is scaled to 2.0.
    """
    fs, T = 128, 128 * 60
    n = np.arange(T)
    low = np.sin(2 * np.pi * (15 / 60.0) * n / fs)
    mixed = (low + 2.0 * np.sin(2 * np.pi * 5.0 * n / fs))[None, :]
    assert PM.dominant_bpm(mixed, fs)[0] == pytest.approx(300.0, abs=1e-6), "unfiltered target follows the 5 Hz peak"
    # same waveform as the PREDICTION: the 1 Hz low-pass removes the 5 Hz component, so it reports 15 bpm
    assert PM.resp_rate_error(mixed, mixed, fs)[0] == pytest.approx(285.0, abs=1e-6)
    # and against a clean 15 bpm target the filtered prediction agrees exactly
    assert PM.resp_rate_error(mixed, low[None, :], fs)[0] == pytest.approx(0.0, abs=1e-9)


def test_segments_per_metric_window_matches_the_upstream_assert():
    assert PM.segments_per_metric_window(60, 4) == 15
    assert PM.segments_per_metric_window(8, 8) == 1
    with pytest.raises(ValueError, match="window_size % segment_len"):
        PM.segments_per_metric_window(60, 8)               # exactly why the resp task uses 4 s


def test_concat_windows_drops_the_remainder_and_preserves_order():
    x = np.arange(30, dtype=np.float64).reshape(6, 5)
    out = PM.concat_windows(x, 4)
    assert out.shape == (1, 20)
    assert out[0].tolist() == list(range(20))
