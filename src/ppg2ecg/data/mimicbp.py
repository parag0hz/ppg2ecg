"""MIMIC-BP (Harvard Dataverse doi:10.7910/DVN/DBM1NF, v2.2) loader — PENGUIN-faithful windowing (upstream load_data.py L130-146).

Raw layout: data/raw/MIMIC-BP/{ppg,abp,ecg,resp,labels}/p<ID>_<kind>.npy ; waveforms [30 segments, 3750 samples] at 125 Hz (raw, ABP in
mmHg), labels [30, 2] = (SBP, DBP) median per segment. Upstream windows each 30 s segment into non-overlapping segment_len windows
(`sliding_window_view(x, (1, fs*L))[:, ::fs*L]`) -> floor(30/L) windows per segment (L = 8 -> 3, the last 6 s dropped) and pairs PPG/ABP
window-by-window (same indices, so alignment is preserved). Subject identity = file prefix p<ID>; official split lists shipped with the data.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS_RAW = 125
SEG_S = 30
N_SEG = 30


@dataclass
class SubjectWindows:
    pid: str
    ppg: np.ndarray  # [n_windows, fs*L] raw
    abp: np.ndarray  # [n_windows, fs*L] raw mmHg
    segment_idx: np.ndarray  # [n_windows]
    window_start_s: np.ndarray  # [n_windows] start within the 30 s segment
    label_sbp: np.ndarray  # [n_windows] segment-level median SBP label (mmHg)
    label_dbp: np.ndarray


def subject_ids(raw: Path) -> list[str]:
    return sorted(p.name.split("_")[0] for p in (raw / "ppg").glob("p*_ppg.npy"))


def official_split(raw: Path) -> dict[str, list[str]]:
    return {k: sorted(ast.literal_eval((raw / f"{k}_subjects.txt").read_text().strip())) for k in ("train", "val", "test")}


def windows_for_subject(raw: Path, pid: str, segment_len: int = 8) -> SubjectWindows:
    ppg = np.load(raw / "ppg" / f"{pid}_ppg.npy")
    abp = np.load(raw / "abp" / f"{pid}_abp.npy")
    lab = np.load(raw / "labels" / f"{pid}_labels.npy")
    assert ppg.shape == abp.shape == (N_SEG, FS_RAW * SEG_S) and lab.shape == (N_SEG, 2), (pid, ppg.shape, abp.shape, lab.shape)
    L = FS_RAW * segment_len
    # upstream: sliding_window_view(x, (1, L))[:, ::L].reshape(-1, L)  == non-overlapping windows per segment, segment-major order
    xw = np.lib.stride_tricks.sliding_window_view(ppg, (1, L))[:, ::L].reshape(-1, L)
    yw = np.lib.stride_tricks.sliding_window_view(abp, (1, L))[:, ::L].reshape(-1, L)
    n_per = xw.shape[0] // N_SEG
    seg = np.repeat(np.arange(N_SEG), n_per)
    start = np.tile(np.arange(n_per) * segment_len, N_SEG).astype(np.float32)
    return SubjectWindows(pid, xw.astype(np.float64), yw.astype(np.float64), seg, start, lab[seg, 0], lab[seg, 1])
