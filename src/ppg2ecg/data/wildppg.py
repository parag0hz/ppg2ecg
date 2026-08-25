"""WildPPG raw access + PENGUIN-faithful windowing (mirror of external/PENGUIN/src/utils/load_data.py L27-77).

PENGUIN (config/preprocess.yaml WildPPG block): ppg_fs 128, label ECG at sternum 128 Hz, colors [g], locations
[sternum, head, wrist, ankle]; each location's green-PPG windows are treated as separate samples paired with the SAME ECG
window (ECG tiled 4x). Filters: PPG band-pass 0.5-4 Hz, ECG high-pass 0.5 Hz, per-window z-score + min-max (same as DaLiA).
Site/colour selection for A4 is a frozen pre-registration decision (docs/A3_A4_REPLICATION_PREREGISTRATION.md Part II).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

LOCATIONS = ("sternum", "head", "wrist", "ankle")


def load_wildppg_participant(path: str | Path) -> dict:
    """Verbatim logic of PENGUIN load_wildppg_participant (load_data.py L29-51): nested dicts per body location / sensor."""
    loaded = scipy.io.loadmat(str(path))
    loaded["id"] = loaded["id"][0]
    loaded["notes"] = "" if len(loaded["notes"]) == 0 else loaded["notes"][0]
    for bodyloc in LOCATIONS:
        bodyloc_data = {}
        sensors = loaded[bodyloc][0].dtype.names
        for sensor_name, sensor_data in zip(sensors, loaded[bodyloc][0][0]):
            bodyloc_data[sensor_name] = {}
            field_names = sensor_data[0][0].dtype.names
            for sensor_field, field_data in zip(field_names, sensor_data[0][0]):
                bodyloc_data[sensor_name][sensor_field] = field_data[0]
                if sensor_field == "fs":
                    bodyloc_data[sensor_name][sensor_field] = bodyloc_data[sensor_name][sensor_field][0]
        loaded[bodyloc] = bodyloc_data
    return loaded


def participant_files(raw_root: Path) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(Path(raw_root) / "WildPPG_Part_*.mat")))


@dataclass
class WildWindows:
    subject: str
    ppg: np.ndarray  # [n, fs*seg] raw
    ecg: np.ndarray  # [n, fs*seg] raw (tiled to match ppg rows)
    site: np.ndarray  # [n] str
    window_index: np.ndarray  # [n] int (temporal index within the recording)
    fs_ppg: int
    fs_ecg: int
    notes: str


def windows_for_participant(path: Path, segment_len: int = 8, locations=LOCATIONS, color: str = "g") -> WildWindows:
    """Non-overlapping windows exactly as PENGUIN (sliding_window_view[::win]); ECG from sternum, PPG per location."""
    d = load_wildppg_participant(path)
    ecg, ecg_fs = np.asarray(d["sternum"]["ecg"]["v"]).squeeze(), int(d["sternum"]["ecg"]["fs"])
    ecg_w = np.lib.stride_tricks.sliding_window_view(ecg, ecg_fs * segment_len)[:: ecg_fs * segment_len]
    ppgs, sites, idxs, fs_ppg = [], [], [], None
    for loc in locations:
        ppg, ppg_fs = np.asarray(d[loc][f"ppg_{color}"]["v"]).squeeze(), int(d[loc][f"ppg_{color}"]["fs"])
        fs_ppg = ppg_fs if fs_ppg is None else fs_ppg
        assert ppg_fs == fs_ppg, (loc, ppg_fs, fs_ppg)
        w = np.lib.stride_tricks.sliding_window_view(ppg, ppg_fs * segment_len)[:: ppg_fs * segment_len]
        n = min(len(w), len(ecg_w))  # PENGUIN tiles ECG assuming equal counts; we align explicitly and record the counts
        ppgs.append(w[:n])
        sites += [loc] * n
        idxs.append(np.arange(n))
    ppg_all = np.concatenate(ppgs)
    idx_all = np.concatenate(idxs)
    ecg_all = np.concatenate([ecg_w[: len(p)] for p in ppgs])
    return WildWindows(subject=str(d["id"]), ppg=ppg_all, ecg=ecg_all, site=np.array(sites), window_index=idx_all, fs_ppg=fs_ppg, fs_ecg=ecg_fs, notes=str(d["notes"]))
