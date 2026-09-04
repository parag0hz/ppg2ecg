"""BIDMC PPG and Respiration Dataset (PhysioNet, doi:10.13026/C2208R) raw access + PENGUIN-faithful windowing
(windowing idiom mirrors src/ppg2ecg/data/wildppg.py / external/PENGUIN/src/utils/load_data.py L27-77).

Records bidmc01..bidmc53: WFDB format-16, 5-7 signals at 125 Hz, 60001 samples (480 s), derived from MIMIC-II.
PPG = 'PLETH' (fingertip, unitless), ECG = lead 'II'; the companion bidmcNNn.* numerics records carry no waveform
and are excluded. wfdb is not installed in this venv, so header and signal file are parsed directly per the WFDB
header spec (physionet.org/physiotools/wag/header-5.htm): each signal line is
"<file> <fmt> <gain>(<baseline>)/<units> <adcres> <adczero> <initval> <checksum> <blocksize> <description>", the
.dat is little-endian int16 interleaved across channels in header order, and the physical value is
(raw - baseline) / gain. The parse is checked against the per-channel 16-bit checksum stored in the header, which
pins endianness, channel count and interleaving; no shipped record contains the format-16 invalid-sample marker
(-32768), so every raw sample maps to a finite physical value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = 125
PPG_SIGNAL = "PLETH"
ECG_SIGNAL = "II"  # lead II; never substituted by another lead (see windows_for_participant)

_SIG_RE = re.compile(
    r"^(?P<file>\S+)\s+(?P<fmt>\d+)\s+(?P<gain>[0-9.eE+-]+)(?:\((?P<baseline>-?\d+)\))?(?:/(?P<units>\S+))?"
    r"\s+(?P<adcres>\d+)\s+(?P<adczero>-?\d+)\s+(?P<init>-?\d+)\s+(?P<cksum>-?\d+)\s+(?P<blocksize>\d+)\s*(?P<desc>.*)$"
)
_INFO_RE = re.compile(r"<age>:\s*(?P<age>\S+)\s+<sex>:\s*(?P<sex>\S+)\s+<location>:\s*(?P<location>\S+)")


@dataclass
class BidmcHeader:
    record: str
    n_sig: int
    fs: int
    n_samples: int
    names: list[str]  # signal descriptions in .dat channel order
    gains: np.ndarray  # [n_sig] adu per physical unit
    baselines: np.ndarray  # [n_sig] adu value of physical zero
    units: list[str]
    checksums: list[int]
    age: str  # kept as text: the released headers use "90+" / "NaN"
    sex: str
    location: str


def read_header(record_path: str | Path) -> BidmcHeader:
    """Parse bidmcNN.hea; record_path may carry any suffix (or none) and is resolved to the .hea sibling."""
    lines = [ln for ln in Path(record_path).with_suffix(".hea").read_text().splitlines() if ln.strip()]
    record, n_sig, fs, n_samples = lines[0].split()[:4]
    sigs = [_SIG_RE.match(ln).groupdict() for ln in lines[1 : 1 + int(n_sig)]]
    assert all(s["fmt"] == "16" for s in sigs), (record, [s["fmt"] for s in sigs])
    info = _INFO_RE.search("\n".join(lines[1 + int(n_sig) :]))
    return BidmcHeader(
        record=record,
        n_sig=int(n_sig),
        fs=int(fs),
        n_samples=int(n_samples),
        names=[s["desc"].strip().rstrip(",") for s in sigs],  # trailing ", " is part of the released descriptions
        gains=np.array([float(s["gain"]) for s in sigs]),
        baselines=np.array([float(s["baseline"] if s["baseline"] is not None else s["adczero"]) for s in sigs]),
        units=[s["units"] for s in sigs],
        checksums=[int(s["cksum"]) for s in sigs],
        age=info["age"],
        sex=info["sex"],
        location=info["location"],
    )


def read_signals(record_path: str | Path) -> tuple[BidmcHeader, np.ndarray]:
    """-> (header, [n_sig, n_samples] float64 in physical units), channels in header order."""
    hdr = read_header(record_path)
    raw = np.fromfile(Path(record_path).with_suffix(".dat"), dtype="<i2")
    assert raw.size == hdr.n_sig * hdr.n_samples, (hdr.record, raw.size, hdr.n_sig * hdr.n_samples)
    raw = raw.reshape(hdr.n_samples, hdr.n_sig).T
    sums = [int(c.astype(np.int64).sum()) % 65536 for c in raw]  # WFDB checksum: 16-bit signed sum of the channel
    assert [s - 65536 if s > 32767 else s for s in sums] == hdr.checksums, (hdr.record, sums, hdr.checksums)
    return hdr, (raw.astype(np.float64) - hdr.baselines[:, None]) / hdr.gains[:, None]


def participant_files(raw_root: str | Path) -> list[Path]:
    """Suffix-less record paths of the bidmcNN waveform records (bidmcNNn numerics are not matched by the glob);
    a record is listed only once both its .hea and its .dat are on disk."""
    return sorted(p.with_suffix("") for p in Path(raw_root).glob("bidmc[0-9][0-9].hea") if p.with_suffix(".dat").exists())


@dataclass
class BidmcWindows:
    subject: str
    ppg: np.ndarray  # [n, fs*seg] raw PLETH (NU)
    ecg: np.ndarray  # [n, fs*seg] raw lead II (mV)
    window_index: np.ndarray  # [n] int, temporal index within the recording
    fs_ppg: int
    fs_ecg: int
    notes: dict


def windows_for_participant(record_path: str | Path, segment_len: int = 8) -> BidmcWindows:
    """Non-overlapping windows exactly as PENGUIN (sliding_window_view[::win]); PPG = PLETH, ECG = lead II.
    A record without lead II yields zero windows, flagged by notes['ecg_channel_used'] = None, rather than a
    silently substituted lead (I / V / AVR / MCL also appear in this corpus)."""
    hdr, sig = read_signals(record_path)
    assert PPG_SIGNAL in hdr.names, (hdr.record, hdr.names)
    win = hdr.fs * segment_len
    notes = {"age": hdr.age, "sex": hdr.sex, "location": hdr.location, "n_samples": hdr.n_samples,
             "ecg_channel_used": ECG_SIGNAL if ECG_SIGNAL in hdr.names else None}
    if notes["ecg_channel_used"] is None:
        empty = np.zeros((0, win))
        return BidmcWindows(hdr.record, empty, empty.copy(), np.zeros(0, np.int64), hdr.fs, hdr.fs, notes)
    ppg_w = np.lib.stride_tricks.sliding_window_view(sig[hdr.names.index(PPG_SIGNAL)], win)[::win]
    ecg_w = np.lib.stride_tricks.sliding_window_view(sig[hdr.names.index(ECG_SIGNAL)], win)[::win]
    return BidmcWindows(hdr.record, ppg_w, ecg_w, np.arange(len(ppg_w)), hdr.fs, hdr.fs, notes)
