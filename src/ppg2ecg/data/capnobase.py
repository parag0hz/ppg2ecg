"""CapnoBase TBME-RR raw access (Borealis doi:10.5683/SP2/NLB8IT, the 42 x 8-min `NNNN_8min.mat` v7.3 files) with the same
non-overlapping windowing as ppg2ecg.data.wildppg (PENGUIN load_data.py L27-77: sliding_window_view(x, fs*L)[::fs*L]).

Layout (verified on all 42 files): signal/{ecg,pleth}/y [1, 144001] float64 @ 300 Hz (param/samplingrate/{ecg,pleth}),
labels/{ecg,pleth}/peak/x [n, 1] float64 expert peak indices, labels/{ecg,pleth}/artif/x the expert artifact intervals.

Index conventions, measured not assumed (see tests/test_capnobase_loader.py):
  * peak labels are 1-BASED MATLAB indices -- the argmax of the signal over the +-6 sample neighbourhood signal[p-6:p+7] of
    a label p sits at label-1 for 26936/27631 ECG and 24745/27991 PPG labels (offset 0 for 2 and 421); this loader returns
    label-1. The denominators are the labels whose neighbourhood is fully in range (p-6 >= 0 and p+7 <= n): all but 2 of the
    27633 ECG and all but 1 of the 27992 pleth labels shipped in the 42 files (re-measured 2026-09-04 over every file).
  * artifact intervals are flat interleaved [start, end] pairs, i.e. row-major [k, 2]: reading vals.reshape(-1, 2) yields
    strictly increasing, sorted, disjoint intervals for all 28 non-empty channel-cases, vals.reshape(2, -1).T for only the
    7 single-interval ones (where both readings coincide). Values live in the same 1-based domain as the peaks but are
    clamped to [0, 144001], so they are converted with the same -1 and clipped at 0.

CAVEAT the corpus builder exists to record: the shipped peak labels are NOT screened against the shipped artifact
intervals -- 177 pleth peaks (19/19 cases that have pleth artifacts) and 7 ECG peaks (4/9 cases) fall strictly inside an
artifact interval. This loader only LABELS artifact windows; dropping them is scripts/build_processed_capnobase.py's job.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

FS = 300  # param/samplingrate/{ecg,pleth}, identical for both channels in every file
CHANNELS = ("ecg", "pleth")


def participant_files(raw_root: str | Path) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(Path(raw_root) / "*_8min.mat")))


def _matlab_char(d: h5py.Dataset) -> str:
    """MATLAB char arrays are stored as uint16 code points."""
    return "" if d.attrs.get("MATLAB_empty") else "".join(chr(c) for c in np.asarray(d).ravel())


def artifact_intervals(h5: h5py.File, channel: str) -> np.ndarray:
    """Expert artifact intervals of `channel` ('ecg' or 'pleth') as [k, 2] int64, 0-based INCLUSIVE raw-domain bounds.

    Three on-disk encodings occur for labels/<channel>/artif/x: MATLAB `[]` (a (2,) uint64 dims vector carrying
    MATLAB_empty=1), a plain float64 (2k, 1) vector of interleaved start/end pairs, or -- when the field was saved as a
    cell array -- a (k,) object-reference array whose elements must be dereferenced through the file to reach the values.
    """
    d = h5[f"labels/{channel}/artif/x"]
    if d.attrs.get("MATLAB_empty"):
        return np.zeros((0, 2), dtype=np.int64)
    if h5py.check_dtype(ref=d.dtype) is not None:
        vals = np.concatenate([np.asarray(h5[r]).ravel() for r in np.asarray(d).ravel()])  # object-reference dereference
    else:
        vals = np.asarray(d).ravel()
    iv = vals.reshape(-1, 2).astype(np.int64) - 1  # 1-based -> 0-based, same convention as the peak labels
    return np.clip(iv, 0, None)  # a start of 0 in the shipped file means "from the first sample"


def peaks_inside_artifacts(peaks: np.ndarray, intervals: np.ndarray) -> int:
    """Number of peaks lying STRICTLY inside (start < p < end) any interval; both arguments must share an index domain.

    Strict vs inclusive is immaterial on the shipped labels (no peak sits exactly on an interval endpoint) but is the
    conservative reading of "inside", and the counts are invariant to the joint 1-based -> 0-based shift.
    """
    p = np.asarray(peaks).reshape(-1, 1)
    iv = np.asarray(intervals).reshape(-1, 2)
    return int(((p > iv[:, 0]) & (p < iv[:, 1])).any(axis=1).sum())


def window_artifact_mask(intervals: np.ndarray, n_windows: int, window_len: int) -> np.ndarray:
    """[n_windows] bool, True where the window's closed sample range [w*L, (w+1)*L - 1] intersects ANY interval."""
    starts = np.arange(n_windows) * window_len
    ends = starts + window_len - 1
    iv = np.asarray(intervals).reshape(-1, 2)
    return ((iv[None, :, 0] <= ends[:, None]) & (iv[None, :, 1] >= starts[:, None])).any(axis=1)


@dataclass
class CapnoWindows:
    subject: str  # 4-digit case id, e.g. '0009'
    ppg: np.ndarray  # [n, fs*seg] raw
    ecg: np.ndarray  # [n, fs*seg] raw
    window_index: np.ndarray  # [n] int (temporal index within the recording)
    fs_ppg: int
    fs_ecg: int
    notes: str
    ecg_artifact_mask: np.ndarray  # [n] bool, window overlaps any ECG artifact interval
    ppg_artifact_mask: np.ndarray  # [n] bool, window overlaps any pleth artifact interval
    ecg_peaks_in_window: list[np.ndarray]  # per window, expert R-peak indices 0-based and relative to the window start
    ecg_artifact_intervals: np.ndarray  # [k, 2] 0-based inclusive
    ppg_artifact_intervals: np.ndarray
    n_ecg_peaks_in_artifacts: int  # audit: shipped ECG peaks strictly inside a shipped ECG artifact interval
    n_ppg_peaks_in_artifacts: int


def windows_for_participant(path: str | Path, segment_len: int = 8) -> CapnoWindows:
    """Non-overlapping segment_len-second windows of the raw 300 Hz signals, with expert artifact / peak labels attached."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        ecg, ppg = np.asarray(h5["signal/ecg/y"]).ravel(), np.asarray(h5["signal/pleth/y"]).ravel()
        fs_ecg, fs_ppg = int(np.asarray(h5["param/samplingrate/ecg"]).ravel()[0]), int(np.asarray(h5["param/samplingrate/pleth"]).ravel()[0])
        ecg_peaks = np.asarray(h5["labels/ecg/peak/x"]).ravel().astype(np.int64) - 1  # 1-based MATLAB -> 0-based
        ppg_peaks = np.asarray(h5["labels/pleth/peak/x"]).ravel().astype(np.int64) - 1
        ecg_iv, ppg_iv = artifact_intervals(h5, "ecg"), artifact_intervals(h5, "pleth")
        case_id = _matlab_char(h5["param/case/id"])
        notes = f"ventilation={_matlab_char(h5['meta/treatment/ventilation'])} age={np.asarray(h5['meta/subject/age']).ravel()[0]:g}"
    assert fs_ecg == fs_ppg == FS and case_id == path.stem, (path.name, fs_ecg, fs_ppg, case_id)
    subject = case_id.split("_")[0]
    L = FS * segment_len
    ecg_w = np.lib.stride_tricks.sliding_window_view(ecg, L)[::L]
    ppg_w = np.lib.stride_tricks.sliding_window_view(ppg, L)[::L]
    n = min(len(ecg_w), len(ppg_w))  # equal by construction (one shared clock), aligned explicitly like the WildPPG loader
    ecg_w, ppg_w = ecg_w[:n], ppg_w[:n]
    starts = np.arange(n) * L
    per_window = [ecg_peaks[(ecg_peaks >= s) & (ecg_peaks < s + L)] - s for s in starts]
    return CapnoWindows(
        subject=subject,
        ppg=ppg_w,
        ecg=ecg_w,
        window_index=np.arange(n),
        fs_ppg=fs_ppg,
        fs_ecg=fs_ecg,
        notes=notes,
        ecg_artifact_mask=window_artifact_mask(ecg_iv, n, L),
        ppg_artifact_mask=window_artifact_mask(ppg_iv, n, L),
        ecg_peaks_in_window=per_window,
        ecg_artifact_intervals=ecg_iv,
        ppg_artifact_intervals=ppg_iv,
        n_ecg_peaks_in_artifacts=peaks_inside_artifacts(ecg_peaks, ecg_iv),
        n_ppg_peaks_in_artifacts=peaks_inside_artifacts(ppg_peaks, ppg_iv),
    )
