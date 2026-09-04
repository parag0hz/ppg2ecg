"""VitalDB raw access + PENGUIN-faithful windowing — same interface shape as src/ppg2ecg/data/wildppg.py.

Raw layout is what scripts/dl_vitaldb.py wrote: data/raw/VitalDB/cases/case_%05d.npz with keys srate () float32 (500.0),
caseid () int32, ECG_II (N,) float32, PLETH (N,) float32 — SNUADC/ECG_II and SNUADC/PLETH exactly as delivered by
api.vitaldb.net, no preprocessing. The waveforms contain NaN dropouts; they are left in here and the frozen drop rule in
scripts/build_processed_vitaldb.py removes the affected windows.

SCALE POLICY (deliberate design decision, recorded verbatim in the corpus MANIFEST as SELECTION_RULE): the full download is
6156 cases / ~9M 8 s windows, which is neither comparable to the other corpora in this benchmark (largest natural corpus:
WildPPG 8 s = 389,355 windows) nor feasible to train on. select_cases() takes an order-deterministic caseid-ascending prefix
of the eligible cases until the cumulative *estimated* window count first reaches --target-windows. No randomness.

SUBJECT IDENTITY = caseid, formatted 'case_%05d'. VitalDB's 6388 cases map to 6090 subjectids (data/raw/VitalDB/cases.csv),
so caseid-level splitting is very slightly weaker than subject-level splitting; we accept this and state it in the manifest.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS_RAW = 500  # SNUADC waveform rate for both tracks
MIN_DURATION_S = 600.0
MIN_FINITE_FRAC = 0.95
TARGET_WINDOWS = 390_000  # ~= WildPPG 8 s corpus (389,355 windows)

SELECTION_RULE = (
    "1. list all case files, sort by caseid ascending; "
    "2. keep only cases where BOTH ECG_II and PLETH exist, are finite-fraction >= min_finite_frac, and duration >= min_duration_s; "
    "3. from that eligible list, take cases in ascending caseid order until the cumulative estimated window count "
    "(floor(N/(srate*segment_len)) per case) first reaches or exceeds target_windows (the crossing case is included); "
    "4. record the exact selected caseid list, the cumulative count, and this rule text in the MANIFEST."
)


@dataclass(frozen=True)
class CaseSummary:
    """Eligibility statistics for one case; cheap enough to recompute, expensive enough to keep out of the build loop."""

    caseid: int
    path: Path
    n_samples: int  # min(len(PLETH), len(ECG_II))
    fs: int
    duration_s: float
    finite_frac_ppg: float
    finite_frac_ecg: float
    has_both: bool
    est_windows: int  # floor(n_samples / (fs*segment_len)) == the exact window count of windows_for_participant

    @property
    def subject(self) -> str:
        return f"case_{self.caseid:05d}"


@dataclass
class Selection:
    cases: list[CaseSummary]  # the selected caseid-ascending prefix
    ineligible: list[CaseSummary]  # cases skipped by step 2 inside the scanned range (kept so the manifest can justify each)
    n_scanned: int
    cum_est_windows: int


@dataclass
class VitalWindows:
    subject: str
    ppg: np.ndarray  # [n, fs*seg] raw (NaN dropouts kept)
    ecg: np.ndarray  # [n, fs*seg] raw (same wall-clock windows as ppg: one 500 Hz clock for both tracks)
    window_index: np.ndarray  # [n] int (temporal index within the case)
    fs_ppg: int
    fs_ecg: int
    notes: dict


def case_id(path: str | Path) -> int:
    """'.../case_00042.npz' -> 42 (file-name order == caseid order, which is what the selection rule is defined on)."""
    return int(Path(path).stem.split("_")[1])


def cases_dir(raw_root: str | Path) -> Path:
    """Accept either data/raw/VitalDB or data/raw/VitalDB/cases (tests write fixtures straight into tmp_path)."""
    root = Path(raw_root)
    return root / "cases" if (root / "cases").is_dir() else root


def participant_files(raw_root: str | Path) -> list[Path]:
    return sorted(cases_dir(raw_root).glob("case_*.npz"), key=case_id)


def summarize_case(path: str | Path, segment_len: int = 8) -> CaseSummary:
    """One pass per waveform, loaded and freed one array at a time (npz members are read lazily on __getitem__)."""
    path = Path(path)
    with np.load(path) as z:
        names = set(z.files)
        fs, caseid = int(z["srate"]), int(z["caseid"])
        assert caseid == case_id(path), (path, caseid)
        if not ("ECG_II" in names and "PLETH" in names):
            return CaseSummary(caseid, path, 0, fs, 0.0, 0.0, 0.0, False, 0)
        ppg = z["PLETH"]
        n_ppg, finite_ppg = ppg.size, float(np.isfinite(ppg).mean())
        del ppg
        ecg = z["ECG_II"]
        n_ecg, finite_ecg = ecg.size, float(np.isfinite(ecg).mean())
        del ecg
    n = min(n_ppg, n_ecg)
    return CaseSummary(caseid, path, n, fs, n / fs, finite_ppg, finite_ecg, True, n // (fs * segment_len))


def is_eligible(s: CaseSummary, min_duration_s: float = MIN_DURATION_S, min_finite_frac: float = MIN_FINITE_FRAC) -> bool:
    """Selection rule step 2."""
    return s.has_both and min(s.finite_frac_ppg, s.finite_frac_ecg) >= min_finite_frac and s.duration_s >= min_duration_s


def iter_case_summaries(raw_root: str | Path, segment_len: int = 8) -> Iterator[CaseSummary]:
    for p in participant_files(raw_root):
        yield summarize_case(p, segment_len)


def eligible_cases(raw_root: str | Path, min_duration_s: float = MIN_DURATION_S, min_finite_frac: float = MIN_FINITE_FRAC, segment_len: int = 8) -> list[CaseSummary]:
    """Selection rule steps 1-2 over the whole corpus (full 27 GB scan; select_cases() is the early-stopping version)."""
    return [s for s in iter_case_summaries(raw_root, segment_len) if is_eligible(s, min_duration_s, min_finite_frac)]


def select_from_eligible(eligible: Iterable[CaseSummary], target_windows: int = TARGET_WINDOWS, max_cases: int | None = None) -> list[CaseSummary]:
    """Selection rule step 3, pure: the caseid-ascending prefix whose cumulative est_windows first reaches target_windows."""
    picked: list[CaseSummary] = []
    cum = 0
    for s in sorted(eligible, key=lambda c: c.caseid):
        if cum >= target_windows or (max_cases is not None and len(picked) >= max_cases):
            break
        picked.append(s)
        cum += s.est_windows
    return picked


def select_cases(raw_root: str | Path, target_windows: int = TARGET_WINDOWS, max_cases: int | None = None, min_duration_s: float = MIN_DURATION_S, min_finite_frac: float = MIN_FINITE_FRAC, segment_len: int = 8) -> Selection:
    """The whole rule with early stopping: eligibility is per-case and the order is fixed, so scanning only up to the
    crossing case yields exactly select_from_eligible(eligible_cases(...)) without touching the rest of the 27 GB."""
    picked: list[CaseSummary] = []
    skipped: list[CaseSummary] = []
    cum = n_scanned = 0
    for p in participant_files(raw_root):
        if cum >= target_windows or (max_cases is not None and len(picked) >= max_cases):
            break
        s = summarize_case(p, segment_len)
        n_scanned += 1
        if not is_eligible(s, min_duration_s, min_finite_frac):
            skipped.append(s)
            continue
        picked.append(s)
        cum += s.est_windows
    return Selection(picked, skipped, n_scanned, cum)


def windows_for_participant(path: str | Path, segment_len: int = 8) -> VitalWindows:
    """Non-overlapping windows exactly as PENGUIN (sliding_window_view[::win]); both tracks share one 500 Hz clock, so
    window k of the PPG and of the ECG are the same wall-clock interval [seg*k, seg*(k+1)) s of the case."""
    path = Path(path)
    with np.load(path) as z:
        fs, caseid = int(z["srate"]), int(z["caseid"])
        assert caseid == case_id(path), (path, caseid)
        ppg_raw, ecg_raw = z["PLETH"], z["ECG_II"]
        n = min(ppg_raw.size, ecg_raw.size)
        win = fs * segment_len
        n_win = n // win
        finite_ppg, finite_ecg = float(np.isfinite(ppg_raw[:n]).mean()), float(np.isfinite(ecg_raw[:n]).mean())
        # identical to sliding_window_view(x[: n_win*win], win)[::win] for n_win > 0, and correct (0, win) for n_win == 0
        ppg = ppg_raw[: n_win * win].reshape(n_win, win).astype(np.float64)  # materialise as float64, then free the case
        ecg = ecg_raw[: n_win * win].reshape(n_win, win).astype(np.float64)
        del ppg_raw, ecg_raw
    notes = {"caseid": caseid, "duration_s": n / fs, "finite_frac_ppg": finite_ppg, "finite_frac_ecg": finite_ecg}
    return VitalWindows(subject=f"case_{caseid:05d}", ppg=ppg, ecg=ecg, window_index=np.arange(n_win), fs_ppg=fs, fs_ecg=fs, notes=notes)
