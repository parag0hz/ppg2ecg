"""CapnoBase loader + corpus-builder tests (src/ppg2ecg/data/capnobase.py, scripts/build_processed_capnobase.py).

Everything that needs the 42 raw .mat files is skipped when data/raw/CapnoBase/files is absent; the encoding, overlap and
drop-rule tests run against HDF5 cases synthesised here, including the object-reference encoding of the artifact field.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from ppg2ecg.data import capnobase
from ppg2ecg.data.capnobase import FS, artifact_intervals, participant_files, peaks_inside_artifacts, window_artifact_mask, windows_for_participant

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/CapnoBase/files"
BUILDER = ROOT / "scripts/build_processed_capnobase.py"
N_RAW_SAMPLES = 144001  # 480 s at 300 Hz, endpoint included
needs_raw = pytest.mark.skipif(not RAW.exists(), reason="raw CapnoBase not downloaded")


def _chars(f: h5py.File, name: str, text: str) -> None:
    f.create_dataset(name, data=np.array([ord(c) for c in text], dtype=np.uint16).reshape(-1, 1))


def _empty(f: h5py.File, name: str) -> None:
    """MATLAB v7.3 writes [] as a (2,) uint64 dims vector tagged MATLAB_empty."""
    d = f.create_dataset(name, data=np.zeros(2, dtype=np.uint64))
    d.attrs["MATLAB_empty"] = np.uint8(1)


def _write_case(path: Path, ecg: np.ndarray, ppg: np.ndarray, ecg_peaks=(), ppg_peaks=(), ecg_artif=None, ppg_artif=None, as_refs: bool = False) -> Path:
    """Synthesise one CapnoBase-shaped v7.3 file; artifact intervals are 1-based [start, end] pairs like the shipped data."""
    with h5py.File(path, "w") as f:
        f.create_dataset("signal/ecg/y", data=ecg.reshape(1, -1))
        f.create_dataset("signal/pleth/y", data=ppg.reshape(1, -1))
        for ch in ("ecg", "pleth"):
            f.create_dataset(f"param/samplingrate/{ch}", data=np.full((1, 1), float(FS)))
        f.create_dataset("labels/ecg/peak/x", data=np.asarray(ecg_peaks, dtype=np.float64).reshape(-1, 1))
        f.create_dataset("labels/pleth/peak/x", data=np.asarray(ppg_peaks, dtype=np.float64).reshape(-1, 1))
        for ch, iv in (("ecg", ecg_artif), ("pleth", ppg_artif)):
            if iv is None:
                _empty(f, f"labels/{ch}/artif/x")
            elif as_refs:  # cell-array encoding: the field holds object references into #refs#
                tgt = f.create_dataset(f"#refs#/{ch}", data=np.asarray(iv, dtype=np.float64).reshape(-1, 1))
                ref = f.create_dataset(f"labels/{ch}/artif/x", shape=(1,), dtype=h5py.ref_dtype)
                ref[0] = tgt.ref
            else:
                f.create_dataset(f"labels/{ch}/artif/x", data=np.asarray(iv, dtype=np.float64).reshape(-1, 1))
        _chars(f, "param/case/id", path.stem)
        _chars(f, "meta/treatment/ventilation", "spontaneous")
        f.create_dataset("meta/subject/age", data=np.full((1, 1), 5.0))
    return path


def _synthetic_signal(rng, n=N_RAW_SAMPLES):
    return np.sin(2 * np.pi * 1.2 * np.arange(n) / FS) + 0.01 * rng.standard_normal(n)


# ---------------------------------------------------------------- encoding / index conventions


def test_artifact_intervals_three_encodings_agree(tmp_path):
    rng = np.random.default_rng(0)
    sig = _synthetic_signal(rng)
    iv = [901, 1200, 5000, 5400]  # 1-based interleaved [start, end] pairs, exactly as shipped
    plain = _write_case(tmp_path / "0001_8min.mat", sig, sig, ecg_artif=iv, ppg_artif=None)
    refs = _write_case(tmp_path / "0002_8min.mat", sig, sig, ecg_artif=iv, ppg_artif=None, as_refs=True)
    with h5py.File(plain, "r") as f:
        a, empty = artifact_intervals(f, "ecg"), artifact_intervals(f, "pleth")
    with h5py.File(refs, "r") as f:
        assert h5py.check_dtype(ref=f["labels/ecg/artif/x"].dtype) is not None  # the branch under test is really exercised
        b = artifact_intervals(f, "ecg")
    np.testing.assert_array_equal(a, np.array([[900, 1199], [4999, 5399]]))  # 1-based -> 0-based inclusive
    np.testing.assert_array_equal(a, b)
    assert empty.shape == (0, 2) and a.dtype == np.int64


def test_zero_start_interval_is_clipped_not_negative(tmp_path):
    sig = _synthetic_signal(np.random.default_rng(1))
    p = _write_case(tmp_path / "0003_8min.mat", sig, sig, ecg_artif=[0, 656])  # shipped case 0115 really contains a 0 start
    with h5py.File(p, "r") as f:
        np.testing.assert_array_equal(artifact_intervals(f, "ecg"), np.array([[0, 655]]))


@needs_raw
def test_peak_labels_are_one_based():
    """The loader returns label-1; that index, not the raw label, is the local extremum of the raw signal."""
    path = participant_files(RAW)[0]
    with h5py.File(path, "r") as f:
        ecg = np.asarray(f["signal/ecg/y"]).ravel()
        labels = np.asarray(f["labels/ecg/peak/x"]).ravel().astype(int)
    at_zero_based = sum(int(np.argmax(ecg[p - 7 : p + 6])) == 6 for p in labels[3:-3])  # p-1 centred
    at_raw = sum(int(np.argmax(ecg[p - 6 : p + 7])) == 6 for p in labels[3:-3])
    assert at_zero_based > 0.95 * len(labels[3:-3]) > at_raw * 20
    w = windows_for_participant(path)
    first = w.ecg_peaks_in_window[0]
    assert first[0] == labels[0] - 1 and (first < FS * 8).all() and (first >= 0).all()


@needs_raw
def test_one_based_evidence_in_the_module_docstring_reproduces_over_all_42_files():
    """Re-derive the docstring's 1-based-index evidence: argmax over signal[p-6:p+7] for every label whose neighbourhood is
    in range (p-6 >= 0 and p+7 <= n), counted at label-1 (index 5) and at the label itself (index 6)."""
    considered, at_label_minus_1, at_label, shipped = ({"ecg": 0, "pleth": 0} for _ in range(4))
    for path in participant_files(RAW):
        with h5py.File(path, "r") as f:
            for ch in ("ecg", "pleth"):
                sig = np.asarray(f[f"signal/{ch}/y"]).ravel()
                lab = np.asarray(f[f"labels/{ch}/peak/x"]).ravel().astype(np.int64)
                shipped[ch] += lab.size
                inrange = lab[(lab - 6 >= 0) & (lab + 7 <= sig.size)]
                arg = np.array([np.argmax(sig[p - 6 : p + 7]) for p in inrange])
                considered[ch] += inrange.size
                at_label_minus_1[ch] += int((arg == 5).sum())
                at_label[ch] += int((arg == 6).sum())
    assert shipped == {"ecg": 27633, "pleth": 27992} and considered == {"ecg": 27631, "pleth": 27991}
    assert at_label_minus_1 == {"ecg": 26936, "pleth": 24745} and at_label == {"ecg": 2, "pleth": 421}
    doc = capnobase.__doc__
    assert f"label-1 for {at_label_minus_1['ecg']}/{considered['ecg']} ECG and {at_label_minus_1['pleth']}/{considered['pleth']} PPG labels (offset 0 for {at_label['ecg']} and {at_label['pleth']})" in doc
    assert f"{shipped['ecg']} ECG and all but 1 of the {shipped['pleth']} pleth labels" in doc


@needs_raw
def test_interval_pairs_are_row_major():
    """vals.reshape(-1, 2) yields sorted disjoint start<end intervals for every non-empty channel-case."""
    n_nonempty = 0
    for path in participant_files(RAW):
        with h5py.File(path, "r") as f:
            for ch in ("ecg", "pleth"):
                iv = artifact_intervals(f, ch)
                n_nonempty += len(iv) > 0
                assert (iv[:, 0] < iv[:, 1]).all() and (iv[1:, 0] >= iv[:-1, 1]).all(), (path.name, ch, iv)
                assert iv.max(initial=0) < N_RAW_SAMPLES
    assert n_nonempty == 28


# ---------------------------------------------------------------- overlap / audit logic


def test_peaks_inside_artifacts_strict_interior():
    iv = np.array([[100, 200], [300, 400]])
    peaks = np.array([50, 100, 101, 199, 200, 250, 350, 400, 500])  # endpoints 100/200/400 are NOT interior
    assert peaks_inside_artifacts(peaks, iv) == 3  # 101, 199, 350
    inclusive = int(((peaks[:, None] >= iv[:, 0]) & (peaks[:, None] <= iv[:, 1])).any(axis=1).sum())
    assert inclusive == 6  # strict and inclusive genuinely differ when a peak sits on an endpoint
    assert peaks_inside_artifacts(peaks, np.zeros((0, 2))) == 0
    assert peaks_inside_artifacts(np.array([]), iv) == 0
    assert peaks_inside_artifacts(peaks + 1, iv + 1) == 3  # invariant to a joint 1-based -> 0-based shift


def test_window_artifact_mask_is_inclusive_overlap():
    L = 10
    assert window_artifact_mask(np.array([[9, 9]]), 3, L).tolist() == [True, False, False]  # last sample of window 0
    assert window_artifact_mask(np.array([[10, 10]]), 3, L).tolist() == [False, True, False]  # first sample of window 1
    assert window_artifact_mask(np.array([[9, 10]]), 3, L).tolist() == [True, True, False]  # straddles the boundary
    assert window_artifact_mask(np.zeros((0, 2)), 3, L).tolist() == [False, False, False]
    assert window_artifact_mask(np.array([[0, 29]]), 3, L).all()


@needs_raw
def test_shipped_peaks_are_not_screened_against_artifacts():
    """Audit reproduced from the shipped labels: 177 pleth and 7 ECG peaks lie strictly inside an artifact interval."""
    tot = {"ecg": 0, "pleth": 0}
    cases = {"ecg": 0, "pleth": 0}
    with_artifacts = {"ecg": 0, "pleth": 0}
    for path in participant_files(RAW):
        with h5py.File(path, "r") as f:
            for ch in ("ecg", "pleth"):
                iv = artifact_intervals(f, ch)
                peaks = np.asarray(f[f"labels/{ch}/peak/x"]).ravel().astype(np.int64) - 1
                n = peaks_inside_artifacts(peaks, iv)
                tot[ch] += n
                cases[ch] += n > 0
                with_artifacts[ch] += len(iv) > 0
    assert tot == {"ecg": 7, "pleth": 177}
    assert cases == {"ecg": 4, "pleth": 19} and with_artifacts == {"ecg": 9, "pleth": 19}


@needs_raw
def test_loader_shapes_and_masks():
    w = windows_for_participant(participant_files(RAW)[0])
    assert w.subject == "0009" and w.fs_ppg == w.fs_ecg == FS
    assert w.ppg.shape == w.ecg.shape == (60, FS * 8) and w.window_index.tolist() == list(range(60))
    assert w.ecg_artifact_mask.shape == w.ppg_artifact_mask.shape == (60,) and w.ecg_artifact_mask.dtype == bool
    assert len(w.ecg_peaks_in_window) == 60 and sum(len(p) for p in w.ecg_peaks_in_window) == 815
    assert not w.ecg_artifact_mask.any() and not w.ppg_artifact_mask.any()  # case 0009 ships no artifact intervals


@needs_raw
def test_loader_labels_but_never_drops_artifact_windows():
    w = windows_for_participant(RAW / "0370_8min.mat")  # 6 pleth artifact intervals -> 9 flagged windows
    assert len(w.ppg) == 60 and int(w.ppg_artifact_mask.sum()) == 9 and w.n_ppg_peaks_in_artifacts == 23


# ---------------------------------------------------------------- builder


def _build(raw_dir: Path, out: Path, screen: str) -> dict:
    subprocess.run([sys.executable, str(BUILDER), "--raw", str(raw_dir), "--out", str(out), "--artifact-screen", screen], check=True, cwd=ROOT, capture_output=True)
    return json.loads((out / "MANIFEST.json").read_text())


def test_drop_rule_nonfinite_and_constant_windows(tmp_path):
    """A window is dropped if either signal has a non-finite sample OR zero std, counted before preprocessing."""
    rng = np.random.default_rng(2)
    ecg, ppg = _synthetic_signal(rng), _synthetic_signal(rng)
    L = FS * 8
    ecg[3 * L + 5] = np.nan  # window 3: non-finite in the ECG
    ppg[7 * L : 8 * L] = 1.0  # window 7: constant PPG
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_case(raw / "0004_8min.mat", ecg, ppg, ecg_peaks=[100.0, 200.0])
    man = _build(raw, tmp_path / "out", "none")
    assert man["total_windows"] == 58 and man["total_dropped"] == 2
    d = np.load(tmp_path / "out" / "0004.npz")
    assert d["window_index"].tolist() == [i for i in range(60) if i not in (3, 7)]


def test_synthetic_screening_modes(tmp_path):
    rng = np.random.default_rng(3)
    sig = _synthetic_signal(rng)
    L = FS * 8
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_case(raw / "0005_8min.mat", sig, sig, ecg_artif=[2 * L + 1, 2 * L + 50], ppg_artif=[5 * L + 1, 5 * L + 50])  # 1-based
    assert _build(raw, tmp_path / "none", "none")["total_windows"] == 60
    ecg_only = _build(raw, tmp_path / "ecg", "drop_ecg")
    assert ecg_only["total_windows"] == 59 and ecg_only["total_screened_artifact"] == 1
    both = _build(raw, tmp_path / "any", "drop_any")
    assert both["total_windows"] == 58 and both["total_screened_artifact"] == 2
    assert np.load(tmp_path / "any" / "0005.npz")["window_index"].tolist() == [i for i in range(60) if i not in (2, 5)]


def test_manifest_records_per_file_artifact_label_coverage(tmp_path):
    """MANIFEST['artifact_label_coverage'] counts FILES (not windows) carrying >= 1 shipped artifact interval per channel."""
    rng = np.random.default_rng(4)
    sig = _synthetic_signal(rng)
    L = FS * 8
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_case(raw / "0006_8min.mat", sig, sig, ecg_artif=[L + 1, L + 50], ppg_artif=None)  # ECG only
    _write_case(raw / "0007_8min.mat", sig, sig, ecg_artif=None, ppg_artif=[L + 1, L + 50])  # pleth only
    _write_case(raw / "0008_8min.mat", sig, sig, ecg_artif=[L + 1, L + 50], ppg_artif=[3 * L + 1, 3 * L + 50])  # both
    _write_case(raw / "0010_8min.mat", sig, sig)  # neither
    cov = _build(raw, tmp_path / "out", "none")["artifact_label_coverage"]
    assert {k: cov[k] for k in ("ecg", "pleth", "both", "any", "neither")} == {"ecg": 2, "pleth": 2, "both": 1, "any": 3, "neither": 1}


@needs_raw
def test_artifact_label_coverage_of_the_42_shipped_files_matches_the_builder_docstring():
    """9 files carry ECG intervals, 19 pleth, 7 both -> 21 carry at least one and 21 carry neither (re-measured here)."""
    e, g = [], []
    for path in participant_files(RAW):
        with h5py.File(path, "r") as f:
            e.append(len(artifact_intervals(f, "ecg")) > 0)
            g.append(len(artifact_intervals(f, "pleth")) > 0)
    cov = {"ecg": sum(e), "pleth": sum(g), "both": sum(a and b for a, b in zip(e, g)), "any": sum(a or b for a, b in zip(e, g)), "neither": sum(not (a or b) for a, b in zip(e, g))}
    assert len(e) == 42 and cov == {"ecg": 9, "pleth": 19, "both": 7, "any": 21, "neither": 21}
    doc = " ".join(BUILDER.read_text().split('"""')[1].split())
    assert f"{cov['ecg']} carry ECG intervals, {cov['pleth']} carry pleth intervals, {cov['both']} carry both, hence {cov['any']} carry at least one and {cov['neither']} carry NEITHER" in doc


@needs_raw
def test_npz_contract_and_manifest(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "0370_8min.mat").symlink_to(RAW / "0370_8min.mat")
    man = _build(raw, tmp_path / "out", "none")
    man_clean = _build(raw, tmp_path / "clean", "drop_any")
    d = np.load(tmp_path / "out" / "0370.npz")
    assert set(d.files) == {"x", "y", "window_index", "window_start_s", "subject", "n_expert_rpeaks"}
    assert d["x"].shape == d["y"].shape == (60, 1024) and d["x"].dtype == d["y"].dtype == np.float32
    assert d["window_index"].dtype == d["window_start_s"].dtype == d["n_expert_rpeaks"].dtype == np.int32
    assert d["subject"].shape == () and str(d["subject"]) == "0370" and d["subject"].dtype.kind == "U"
    assert d["window_start_s"].tolist() == (d["window_index"] * 8).tolist()
    assert np.isfinite(d["x"]).all() and np.isfinite(d["y"]).all() and abs(d["x"]).max() == pytest.approx(1.0)
    with h5py.File(RAW / "0370_8min.mat", "r") as f:
        n_labels = len(np.asarray(f["labels/ecg/peak/x"]).ravel())
    assert int(d["n_expert_rpeaks"].sum()) == n_labels  # every shipped R-peak lands in exactly one window
    for m, screen, n in ((man, "none", 60), (man_clean, "drop_any", 51)):
        assert m["artifact_screen"] == screen and m["total_windows"] == n
        assert m["peaks_inside_artifacts_audit"]["ecg"] == 0 and m["peaks_inside_artifacts_audit"]["pleth"] == 23
        assert set(m) >= {"built", "dataset", "segment_len_s", "resample_rate", "samples_per_window", "ppg_preprocess", "ecg_preprocess", "dtype", "total_windows", "total_dropped", "files"}
        assert set(m["files"]["0370"]) >= {"path", "raw_file", "raw_sha256", "n_windows", "n_dropped_nonfinite_or_constant", "fs_ppg", "fs_ecg", "sha256", "notes"}
    assert man_clean["total_screened_artifact"] == 9 and man["total_screened_artifact"] == 0
