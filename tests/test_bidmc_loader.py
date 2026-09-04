"""BIDMC loader + processed-corpus builder (src/ppg2ecg/data/bidmc.py, scripts/build_processed_bidmc.py).

Everything structural runs on synthetic WFDB format-16 records written into tmp_path, so the suite collects and
passes without data/raw/BIDMC; the two checks that read the real corpus skip when it is absent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.data.bidmc import ECG_SIGNAL, FS, PPG_SIGNAL, participant_files, read_header, read_signals, windows_for_participant

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "BIDMC"
PROCESSED = ROOT / "data" / "processed" / "bidmc_8s"
needs_raw = pytest.mark.skipif(not (RAW / "bidmc01.dat").exists(), reason="data/raw/BIDMC not present")


def wfdb_checksum(channel: np.ndarray) -> int:
    """16-bit signed sum of the channel's samples, as stored in the header's checksum field."""
    s = int(channel.astype(np.int64).sum()) % 65536
    return s - 65536 if s > 32767 else s


def write_record(d: Path, name: str, names: list[str], data: np.ndarray, gains: list[float], baselines: list[int | None],
                 fs: int = FS, age: str = "64", sex: str = "F", location: str = "micu", adczeros: list[int] | None = None) -> Path:
    """Write name.hea/name.dat as WFDB format 16; data is [n_sig, n_samples] int16 in header order.

    adczeros writes the header's ADC-zero field per channel (default 0, as the released BIDMC headers have it); it is the
    documented fallback for an omitted (baseline) field, so a non-zero value there is what distinguishes the fallback."""
    n_sig, n_samples = data.shape
    adcz = [0] * n_sig if adczeros is None else adczeros
    lines = [f"{name} {n_sig} {fs} {n_samples}"]
    for i, nm in enumerate(names):
        base = "" if baselines[i] is None else f"({baselines[i]})"
        lines.append(f"{name}.dat 16 {gains[i]}{base}/mV 0 {adcz[i]} {int(data[i, 0])} {wfdb_checksum(data[i])} 0 {nm}, ")
    lines.append(f"#<age>: {age} <sex>: {sex} <location>: {location} <source>: synthetic")
    (d / f"{name}.hea").write_text("\n".join(lines) + "\n")
    data.T.astype("<i2").tofile(d / f"{name}.dat")
    return d / name


def synth_channels(n_sig: int, n_samples: int, seed: int = 0) -> np.ndarray:
    """Pulsatile int16 channels, each at a different rate/phase so no two channels are equal."""
    t = np.arange(n_samples) / FS
    rng = np.random.default_rng(seed)
    return np.stack([(8000 * np.sin(2 * np.pi * (1.0 + 0.3 * i) * t + i) + rng.integers(-50, 50, n_samples)).astype(np.int16) for i in range(n_sig)])


@pytest.fixture
def record(tmp_path):
    """A 6-signal record with PLETH and lead II buried among other leads, in a deliberately non-canonical order."""
    names = ["RESP", "I", PPG_SIGNAL, "V", ECG_SIGNAL, "AVR"]
    data = synth_channels(len(names), FS * 24)
    gains = [65534.0, 32670.9091, 138229.4494, 42414.137, 116796.6584, 3247.07]
    baselines = [-32767, -16367, -63710, -11643, -60167, -54739]
    path = write_record(tmp_path, "bidmc01", names, data, gains, baselines, age="90+", sex="?")
    return path, names, data, np.array(gains), np.array(baselines)


def test_header_parses_names_gains_baselines_and_demographics(record):
    path, names, data, gains, baselines = record
    h = read_header(path)
    assert h.record == "bidmc01" and h.n_sig == 6 and h.fs == FS and h.n_samples == FS * 24
    assert h.names == names  # channel order preserved, trailing ", " of the released descriptions stripped
    assert np.array_equal(h.gains, gains) and np.array_equal(h.baselines, baselines)
    assert h.units == ["mV"] * 6 and h.checksums == [wfdb_checksum(c) for c in data]
    assert (h.age, h.sex, h.location) == ("90+", "?", "micu")  # age stays text: the corpus ships "90+" and "NaN"


def test_absent_baseline_field_falls_back_to_adc_zero(tmp_path):
    """Channel 0 omits (baseline) and carries a NON-ZERO ADC-zero (-13), so the fallback cannot be confused with 0."""
    data = synth_channels(2, FS * 8)
    path = write_record(tmp_path, "bidmc02", [PPG_SIGNAL, ECG_SIGNAL], data, [100.0, 200.0], [None, -7], adczeros=[-13, 5])
    assert "16 100.0/mV 0 -13 " in path.with_suffix(".hea").read_text()  # the field under test is really written non-zero
    h = read_header(path)
    assert np.array_equal(h.baselines, np.array([-13.0, -7.0]))  # ch0 -> its ADC-zero; ch1 -> its explicit (baseline), NOT its ADC-zero 5
    _, sig = read_signals(path)
    assert np.allclose(sig[0], (data[0].astype(np.float64) + 13) / 100.0)  # the fallback is what the physical scaling uses


def test_signals_are_interleaved_int16_scaled_by_gain_and_baseline(record):
    path, names, data, gains, baselines = record
    h, sig = read_signals(path)
    assert sig.shape == (6, FS * 24)
    assert np.allclose(sig, (data.astype(np.float64) - baselines[:, None]) / gains[:, None])


def test_corrupt_dat_is_caught_by_the_header_checksum(record):
    path, _, data, _, _ = record
    raw = np.fromfile(path.with_suffix(".dat"), dtype="<i2")
    raw[17] += 3
    raw.tofile(path.with_suffix(".dat"))
    with pytest.raises(AssertionError):
        read_signals(path)


def test_truncated_dat_is_caught(record):
    path, _, _, _, _ = record
    np.fromfile(path.with_suffix(".dat"), dtype="<i2")[:-6].tofile(path.with_suffix(".dat"))
    with pytest.raises(AssertionError):
        read_signals(path)


def test_lead_ii_is_selected_and_no_other_lead_is_substituted(record):
    path, names, data, gains, baselines = record
    w = windows_for_participant(path, segment_len=8)
    for lead, i in [(ECG_SIGNAL, names.index(ECG_SIGNAL)), (PPG_SIGNAL, names.index(PPG_SIGNAL))]:
        phys = (data[i].astype(np.float64) - baselines[i]) / gains[i]
        got = w.ecg if lead == ECG_SIGNAL else w.ppg
        assert np.allclose(got[0], phys[: FS * 8]) and np.allclose(got[-1], phys[-FS * 8 :])
    assert w.notes["ecg_channel_used"] == ECG_SIGNAL
    for other in ("I", "V", "AVR"):  # the decoys must not be what ended up in .ecg
        i = names.index(other)
        assert not np.allclose(w.ecg[0], ((data[i].astype(np.float64) - baselines[i]) / gains[i])[: FS * 8])


def test_record_without_lead_ii_yields_zero_windows_and_says_so(tmp_path):
    names = ["RESP", PPG_SIGNAL, "I", "MCL", "AVR"]
    path = write_record(tmp_path, "bidmc03", names, synth_channels(len(names), FS * 24), [100.0] * 5, [0] * 5)
    w = windows_for_participant(path, segment_len=8)
    assert w.notes["ecg_channel_used"] is None
    assert w.ppg.shape == (0, FS * 8) and w.ecg.shape == (0, FS * 8) and len(w.window_index) == 0


@pytest.mark.parametrize("n_samples,segment_len", [(60001, 8), (60000, 8), (FS * 8, 8), (FS * 8 + 1, 8), (60001, 4), (60001, 10)])
def test_window_count_arithmetic_and_non_overlap(tmp_path, n_samples, segment_len):
    names = [PPG_SIGNAL, ECG_SIGNAL]
    data = synth_channels(2, n_samples)
    path = write_record(tmp_path, "bidmc04", names, data, [100.0, 200.0], [-5, 7])
    w = windows_for_participant(path, segment_len)
    win = FS * segment_len
    assert len(w.ppg) == max(0, (n_samples - win) // win + 1) == n_samples // win
    assert np.array_equal(w.window_index, np.arange(len(w.ppg)))
    phys = (data[1].astype(np.float64) - 7) / 200.0
    for k in range(len(w.ecg)):  # consecutive, non-overlapping, in temporal order
        assert np.array_equal(w.ecg[k], phys[k * win : (k + 1) * win])
    assert (w.fs_ppg, w.fs_ecg) == (FS, FS)


def test_record_shorter_than_one_window_raises(tmp_path):
    """PENGUIN's sliding_window_view idiom has no zero-window path; every BIDMC record is 480 s, so this only pins behaviour."""
    path = write_record(tmp_path, "bidmc05", [PPG_SIGNAL, ECG_SIGNAL], synth_channels(2, FS * 8 - 1), [100.0, 200.0], [0, 0])
    with pytest.raises(ValueError):
        windows_for_participant(path, segment_len=8)


def test_participant_files_excludes_numerics_and_records_missing_their_dat(tmp_path):
    data = synth_channels(2, FS * 8)
    for name in ("bidmc01", "bidmc07", "bidmc53"):
        write_record(tmp_path, name, [PPG_SIGNAL, ECG_SIGNAL], data, [100.0, 200.0], [0, 0])
    write_record(tmp_path, "bidmc01n", ["HR", "PULSE"], synth_channels(2, 481), [9362.0, 13106.8], [-856623, -1199272], fs=1)
    write_record(tmp_path, "bidmc32", [PPG_SIGNAL, ECG_SIGNAL], data, [100.0, 200.0], [0, 0])
    (tmp_path / "bidmc32.dat").unlink()  # header shipped, signal file absent (as for bidmc32/bidmc49 on disk)
    assert [p.name for p in participant_files(tmp_path)] == ["bidmc01", "bidmc07", "bidmc53"]
    assert all(p.suffix == "" for p in participant_files(tmp_path))


def load_builder(tmp_path, monkeypatch):
    """Import scripts/build_processed_bidmc.py as a module with its ROOT rebased onto tmp_path."""
    spec = importlib.util.spec_from_file_location("build_processed_bidmc", ROOT / "scripts" / "build_processed_bidmc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    return mod


def run_builder(mod, monkeypatch, raw: Path, out: Path, segment_len: int = 8):
    monkeypatch.setattr(sys, "argv", ["build_processed_bidmc.py", "--raw", str(raw), "--segment-len", str(segment_len), "--out", str(out)])
    mod.main()
    return json.loads((out / "MANIFEST.json").read_text())


def test_builder_emits_exactly_the_contract_keys_dtypes_and_shapes(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    for name in ("bidmc01", "bidmc02"):
        write_record(raw, name, ["RESP", PPG_SIGNAL, ECG_SIGNAL, "V"], synth_channels(4, FS * 40, seed=int(name[-1])), [65534.0, 138229.4494, 32670.9091, 42414.137], [-32767, -63710, -16367, -11643])
    man = run_builder(load_builder(tmp_path, monkeypatch), monkeypatch, raw, out)
    assert man["total_windows"] == 10 and man["total_dropped"] == 0 and man["n_subjects"] == 2  # 40 s / 8 s = 5 windows each
    assert set(man) >= {"built", "dataset", "segment_len_s", "resample_rate", "samples_per_window", "ppg_preprocess", "ecg_preprocess", "dtype", "total_windows", "total_dropped", "files"}
    assert (man["segment_len_s"], man["resample_rate"], man["samples_per_window"], man["dtype"]) == (8, 128, 1024, "float32")
    assert set(man["files"]["bidmc01"]) >= {"path", "raw_file", "raw_sha256", "n_windows", "n_dropped_nonfinite_or_constant", "fs_ppg", "fs_ecg", "sha256", "notes"}
    assert man["files"]["bidmc01"]["notes"]["ecg_channel_used"] == ECG_SIGNAL
    d = np.load(out / "bidmc01.npz")
    assert set(d.files) == {"x", "y", "window_index", "window_start_s", "subject"}
    assert d["x"].dtype == d["y"].dtype == np.float32 and d["x"].shape == d["y"].shape == (5, 1024)
    assert d["window_index"].dtype == d["window_start_s"].dtype == np.int32
    assert np.array_equal(d["window_index"], np.arange(5)) and np.array_equal(d["window_start_s"], np.arange(5) * 8)
    assert d["subject"].shape == () and d["subject"].dtype.kind == "U" and str(d["subject"]) == "bidmc01"
    assert np.isfinite(d["x"]).all() and np.abs(d["x"]).max() == pytest.approx(1.0)  # per-window min-max to [-1, 1]


def test_builder_accepts_an_out_dir_outside_the_repo_root(tmp_path, monkeypatch):
    """--out may point anywhere: with the real ROOT, an outside path is recorded absolute instead of raising in relative_to."""
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    write_record(raw, "bidmc01", [PPG_SIGNAL, ECG_SIGNAL], synth_channels(2, FS * 16), [100.0, 200.0], [-5, 7])
    spec = importlib.util.spec_from_file_location("build_processed_bidmc_real_root", ROOT / "scripts" / "build_processed_bidmc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # ROOT is NOT rebased here, so out really is outside it
    assert mod.ROOT == ROOT and not out.is_relative_to(ROOT)
    man = run_builder(mod, monkeypatch, raw, out)
    assert man["files"]["bidmc01"]["path"] == str(out / "bidmc01.npz") and man["total_windows"] == 2


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the 1e308 window deliberately overflows inside the filter
def test_builder_drop_rule_is_nonfinite_or_zero_std_before_and_nonfinite_after(tmp_path, monkeypatch):
    """One dropped window per failure mode: NaN in either signal, zero std in either signal, non-finite post-filter."""
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    rec = write_record(raw, "bidmc01", [PPG_SIGNAL, ECG_SIGNAL], synth_channels(2, FS * 8), [100.0, 200.0], [0, 0])
    mod = load_builder(tmp_path, monkeypatch)
    win, t = FS * 8, np.arange(FS * 8) / FS
    ppg = np.stack([np.sin(2 * np.pi * 1.2 * t + k) for k in range(8)])
    ecg = np.stack([np.sin(2 * np.pi * 1.1 * t + k) for k in range(8)])
    ppg[1, 3] = np.nan          # non-finite PPG
    ecg[2, 9] = np.inf          # non-finite ECG
    ppg[3] = 0.5                # constant PPG (zero std)
    ecg[4] = -2.0               # constant ECG (zero std)
    ppg[5] *= 1e308             # finite in, overflows to non-finite through resample+filtfilt
    from ppg2ecg.data.bidmc import BidmcWindows
    monkeypatch.setattr(mod, "windows_for_participant", lambda f, L: BidmcWindows("bidmc01", ppg, ecg, np.arange(8), FS, FS, {"ecg_channel_used": ECG_SIGNAL}))
    man = run_builder(mod, monkeypatch, raw, out)
    assert man["total_dropped"] == 5 and man["total_windows"] == 3
    assert man["files"]["bidmc01"]["n_dropped_nonfinite_or_constant"] == 5
    d = np.load(out / "bidmc01.npz")
    assert np.array_equal(d["window_index"], np.array([0, 6, 7]))  # survivors keep their temporal index
    assert np.array_equal(d["window_start_s"], np.array([0, 48, 56])) and d["x"].shape == (3, 1024)
    assert rec.with_suffix(".dat").exists() and win == 1000


@needs_raw
def test_real_record_parses_and_the_waveforms_are_the_expected_ones():
    h, sig = read_signals(RAW / "bidmc01")
    assert (h.n_sig, h.fs, h.n_samples) == (5, 125, 60001) and h.names == ["RESP", "PLETH", "V", "AVR", "II"]
    ppg, ecg = sig[h.names.index(PPG_SIGNAL)], sig[h.names.index(ECG_SIGNAL)]
    f = np.fft.rfftfreq(ppg.size, 1 / FS)
    band = (f > 0.5) & (f < 3.0)
    dom = f[band][np.abs(np.fft.rfft(ppg - ppg.mean()))[band].argmax()]
    assert ppg.min() > 0 and 0.7 < dom < 2.5  # PLETH is a positive pulsatile waveform at a plausible heart rate
    dev = ecg - np.median(ecg)
    assert dev[np.abs(dev).argmax()] > 0 and float(((dev / ecg.std()) ** 3).mean()) > 1.0  # lead II: sharp positive R peaks
    assert len(participant_files(RAW)) == len(list(RAW.glob("bidmc[0-9][0-9].dat")))


@pytest.mark.skipif(not (PROCESSED / "MANIFEST.json").exists(), reason="data/processed/bidmc_8s not built")
def test_built_corpus_matches_the_processed_schema():
    man = json.loads((PROCESSED / "MANIFEST.json").read_text())
    assert man["samples_per_window"] == 1024 and man["total_windows"] == sum(v["n_windows"] for v in man["files"].values())
    for subject, entry in man["files"].items():
        d = np.load(PROCESSED / f"{subject}.npz")
        assert set(d.files) == {"x", "y", "window_index", "window_start_s", "subject"}
        assert d["x"].shape == d["y"].shape == (entry["n_windows"], 1024) and d["x"].dtype == np.float32
        assert str(d["subject"]) == subject and np.isfinite(d["x"]).all() and np.isfinite(d["y"]).all()
