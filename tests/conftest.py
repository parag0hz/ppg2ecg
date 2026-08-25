import ppg2ecg.utils.mkl_warmup  # noqa: F401  (MKL warm-up before any test module imports torch, docs/ENVIRONMENT.md)
import numpy as np
import pytest


@pytest.fixture
def synth_ecg():
    """8 s synthetic ECG at 128 Hz, 70 bpm, deterministic (neurokit2)."""
    nk = pytest.importorskip("neurokit2")
    fs = 128
    sig = nk.ecg_simulate(duration=8, sampling_rate=fs, heart_rate=70, random_state=0, method="ecgsyn")
    return np.asarray(sig, dtype=np.float64), fs
