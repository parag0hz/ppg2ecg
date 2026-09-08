"""D3 loaders for PENGUIN's respiratory and ABP tasks — mirrors external/PENGUIN/src/utils/load_data.py exactly.

Frozen by docs/D3_PENGUIN_SIX_DATASET_PREREGISTRATION.md §4. Windowing is upstream's idiom,
`sliding_window_view(x, fs*L)[::fs*L]`, so the trailing partial window is dropped rather than padded.

Per-task segment lengths are forced by PENGUIN's own contradiction (prereg §3): their `sample_num`
bookkeeping is 8 s for every dataset, but `train.py:43` asserts `window_size % segment_len == 0` and
`RespRateError` has `window_size: 60`, so the respiratory metric cannot run at 8 s.
"""
from __future__ import annotations

import glob
import pickle
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# (ppg_fs, label_fs) exactly as external/PENGUIN/config/preprocess.yaml
BIDMC_FS = (125, 125)
WESAD_FS = (64, 700)
UCI_FS = (125, 125)


@dataclass(frozen=True)
class TaskWindows:
    subject: str
    ppg: np.ndarray      # [n, ppg_fs * L] raw
    label: np.ndarray    # [n, label_fs * L] raw
    window_index: np.ndarray
    fs_ppg: int
    fs_label: int
    notes: dict


def _win(x: np.ndarray, w: int) -> np.ndarray:
    """Upstream windowing: non-overlapping, trailing remainder dropped."""
    if len(x) < w:
        return np.zeros((0, w), dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=np.float64), w)[::w]


def bidmc_files(raw: str | Path) -> list[Path]:
    """PENGUIN indexes bidmc_{sub_idx+1:02d}_Signals.csv for sub_idx 0..52."""
    return [Path(raw) / f"bidmc_{i + 1:02d}_Signals.csv" for i in range(53)]


def load_bidmc_resp(path: Path, segment_len: int = 4) -> TaskWindows:
    """PPG->respiration. Columns " PLETH" and " RESP", both 125 Hz (load_data.py:169-183)."""
    d = pd.read_csv(path)
    ppg, resp = d[" PLETH"].values, d[" RESP"].values
    fp, fl = BIDMC_FS
    w_ppg, w_lab = _win(ppg, fp * segment_len), _win(resp, fl * segment_len)
    n = min(len(w_ppg), len(w_lab))
    return TaskWindows(path.stem.replace("_Signals", ""), w_ppg[:n], w_lab[:n], np.arange(n, dtype=np.int32),
                       fp, fl, {"n_samples": int(len(ppg)), "source": path.name})


def wesad_files(raw: str | Path) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(f"{raw}/S*/S*.pkl"))]


def load_wesad_resp(path: Path, segment_len: int = 4) -> TaskWindows:
    """PPG->respiration. wrist BVP 64 Hz, chest Resp 700 Hz (load_data.py:150-167)."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    ppg = np.asarray(d["signal"]["wrist"]["BVP"]).squeeze()
    resp = np.asarray(d["signal"]["chest"]["Resp"]).squeeze()
    fp, fl = WESAD_FS
    w_ppg, w_lab = _win(ppg, fp * segment_len), _win(resp, fl * segment_len)
    n = min(len(w_ppg), len(w_lab))
    return TaskWindows(path.parent.name, w_ppg[:n], w_lab[:n], np.arange(n, dtype=np.int32),
                       fp, fl, {"n_ppg": int(len(ppg)), "n_resp": int(len(resp)), "source": path.name})


def uci_subject_ids(n_subjects: int = 8) -> list[int]:
    return list(range(n_subjects))


def load_uci_abp(raw: str | Path, sub_idx: int, segment_len: int = 8) -> TaskWindows:
    """PPG->ABP. PENGUIN's indexing verbatim (load_data.py:99-129): Part_{sub_idx//4+1}.mat, records
    [sub_idx%2 * 1500 : (sub_idx%2+1) * 1500], column 0 = PPG, column 1 = ABP, both 125 Hz.
    ABP is left in raw mmHg (label_bandpass/zscore/normalize all False), so SBP/DBP are in mmHg.
    """
    part = sub_idx // 4 + 1
    lo, hi = sub_idx % 2 * 1500, (sub_idx % 2 + 1) * 1500
    fp, fl = UCI_FS
    ppg_w, abp_w = [], []
    with h5py.File(Path(raw) / f"Part_{part}.mat", "r") as f:
        refs = f[f"Part_{part}"][lo:hi, 0]
        for ref in refs:
            s = f[ref][:]
            ppg_w.append(_win(s[:, 0], fp * segment_len))
            abp_w.append(_win(s[:, 1], fl * segment_len))
    ppg = np.concatenate([a for a in ppg_w if len(a)]) if any(len(a) for a in ppg_w) else np.zeros((0, fp * segment_len))
    abp = np.concatenate([a for a in abp_w if len(a)]) if any(len(a) for a in abp_w) else np.zeros((0, fl * segment_len))
    n = min(len(ppg), len(abp))
    return TaskWindows(f"uci{sub_idx:02d}", ppg[:n], abp[:n], np.arange(n, dtype=np.int32), fp, fl,
                       {"part": part, "record_range": [lo, hi], "n_records": len(refs)})
