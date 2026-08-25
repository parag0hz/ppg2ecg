"""PPG-DaLiA raw access + PENGUIN-faithful windowing.

Dataset: Reiss et al. 2019, UCI #495, CC BY 4.0. 15 subjects S1..S15, ~2 h each.
  wrist  (Empatica E4)  : BVP 64 Hz  (also ACC 32, EDA 4, TEMP 4)
  chest  (RespiBAN)     : ECG 700 Hz (also ACC 700, Resp 700)
  extras: 'activity' (700 Hz labels), 'label' (reference HR per 8 s window, 2 s shift), 'rpeaks' (700 Hz sample indices)
Upstream loader: external/PENGUIN/src/utils/load_data.py L80-95.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUBJECTS: tuple[str, ...] = tuple(f"S{i}" for i in range(1, 16))
ECG_FS = 700
BVP_FS = 64
SEGMENT_LEN_S = 4


def find_subject_pkl(raw_root: Path, subject: str) -> Path:
    raw_root = Path(raw_root)
    candidates = [
        raw_root / "PPG-DaLiA" / "PPG_FieldStudy" / subject / f"{subject}.pkl",  # upstream layout
        raw_root / "PPG_FieldStudy" / subject / f"{subject}.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(raw_root.rglob(f"{subject}.pkl"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{subject}.pkl not found under {raw_root}")


@dataclass
class SubjectRaw:
    subject: str
    ecg: np.ndarray  # [N_ecg] @ 700 Hz
    bvp: np.ndarray  # [N_bvp] @ 64 Hz
    rpeaks: np.ndarray | None  # 700 Hz sample indices (dataset-provided reference)
    hr_label: np.ndarray | None  # reference HR per 8 s window (2 s shift)
    activity: np.ndarray | None
    path: str

    @property
    def ecg_seconds(self) -> float:
        return len(self.ecg) / ECG_FS

    @property
    def bvp_seconds(self) -> float:
        return len(self.bvp) / BVP_FS


def load_subject_raw(raw_root: Path, subject: str) -> SubjectRaw:
    p = find_subject_pkl(raw_root, subject)
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="latin1")  # same as upstream
    ecg = np.asarray(d["signal"]["chest"]["ECG"]).squeeze()
    bvp = np.asarray(d["signal"]["wrist"]["BVP"]).squeeze()
    return SubjectRaw(
        subject=subject,
        ecg=ecg,
        bvp=bvp,
        rpeaks=np.asarray(d["rpeaks"]).squeeze() if "rpeaks" in d else None,
        hr_label=np.asarray(d["label"]).squeeze() if "label" in d else None,
        activity=np.asarray(d["activity"]).squeeze() if "activity" in d else None,
        path=str(p),
    )


def window_nonoverlap(x: np.ndarray, win: int) -> np.ndarray:
    """Exactly upstream: sliding_window_view(x, win)[::win]  -> [n, win]; trailing < win samples dropped."""
    return np.lib.stride_tricks.sliding_window_view(x, win)[::win]


@dataclass
class SubjectWindows:
    subject: str
    ppg: np.ndarray  # [n, 64*seg]  raw (pre-preprocess)
    ecg: np.ndarray  # [n, 700*seg]
    n_ppg_windows: int
    n_ecg_windows: int
    truncated_to: int
    window_start_s: np.ndarray  # [n] start time of each window in seconds


def windows_for_subject(raw: SubjectRaw, segment_len: int = SEGMENT_LEN_S, align: str = "truncate") -> SubjectWindows:
    """Non-overlapping segment_len-second windows from both streams, assuming a common t=0 (dataset is synchronised).

    Upstream does NOT check that the PPG and ECG window counts agree; we record both counts and
    (align='truncate') keep the first min(n_ppg, n_ecg) windows so x[i] and y[i] always cover the same seconds.
    """
    ppg = window_nonoverlap(raw.bvp, BVP_FS * segment_len)
    ecg = window_nonoverlap(raw.ecg, ECG_FS * segment_len)
    n_p, n_e = len(ppg), len(ecg)
    n = min(n_p, n_e)
    if align == "strict" and n_p != n_e:
        raise ValueError(f"{raw.subject}: PPG windows {n_p} != ECG windows {n_e}")
    return SubjectWindows(
        subject=raw.subject,
        ppg=ppg[:n],
        ecg=ecg[:n],
        n_ppg_windows=n_p,
        n_ecg_windows=n_e,
        truncated_to=n,
        window_start_s=np.arange(n) * segment_len,
    )
