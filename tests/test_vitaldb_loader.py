"""VitalDB loader + scale-policy selection rule (src/ppg2ecg/data/vitaldb.py) on synthetic npz fixtures.

Nothing here touches the 27 GB raw corpus: every case is written into tmp_path with the documented raw contract
(srate (), caseid (), ECG_II (N,), PLETH (N,) float32). The fixtures use a low srate so the arrays stay small; the loader
is rate-generic and reads srate from the file, so the arithmetic under test is the same one that runs at 500 Hz.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows
from ppg2ecg.data.vitaldb import (
    CaseSummary,
    eligible_cases,
    is_eligible,
    participant_files,
    select_cases,
    select_from_eligible,
    summarize_case,
    windows_for_participant,
)

FS = 50  # fixture sampling rate (real corpus: 500); windows are fs*8 samples either way
ROOT = Path(__file__).resolve().parents[1]


def write_case(root, caseid: int, n_samples: int, fs: int = FS, n_nan: int = 0, keys=("PLETH", "ECG_II"), seed: int | None = None) -> None:
    """One synthetic case file; the first n_nan samples of both waveforms are NaN (VitalDB's leading dropouts)."""
    rng = np.random.default_rng(caseid if seed is None else seed)
    arrs = {k: rng.normal(size=n_samples).astype(np.float32) for k in keys}
    for a in arrs.values():
        a[:n_nan] = np.nan
    np.savez(root / f"case_{caseid:05d}.npz", srate=np.float32(fs), caseid=np.int32(caseid), **arrs)


def test_participant_files_are_caseid_ascending(tmp_path):
    for cid in (13, 2, 100, 7):
        write_case(tmp_path, cid, 800)
    assert [p.name for p in participant_files(tmp_path)] == ["case_00002.npz", "case_00007.npz", "case_00013.npz", "case_00100.npz"]


def test_summary_matches_the_raw_npz_contract(tmp_path):
    write_case(tmp_path, 7, 4321, n_nan=100)
    s = summarize_case(tmp_path / "case_00007.npz", segment_len=8)
    assert (s.caseid, s.subject, s.fs, s.n_samples, s.has_both) == (7, "case_00007", FS, 4321, True)
    assert s.duration_s == 4321 / FS and s.est_windows == 4321 // (FS * 8) == 10
    assert s.finite_frac_ppg == s.finite_frac_ecg == pytest.approx(1 - 100 / 4321)


def test_missing_track_is_not_eligible(tmp_path):
    write_case(tmp_path, 1, 4000, keys=("PLETH",))
    s = summarize_case(tmp_path / "case_00001.npz")
    assert not s.has_both and s.est_windows == 0 and not is_eligible(s, min_duration_s=0.0, min_finite_frac=0.0)


def test_eligibility_thresholds_are_inclusive(tmp_path):
    """>= on both criteria: duration exactly 600 s and finite fraction exactly 0.95 are KEPT; one sample less is not."""
    write_case(tmp_path, 1, 600 * FS, n_nan=int(0.05 * 600 * FS))  # exactly 600.0 s, exactly 0.95 finite
    write_case(tmp_path, 2, 600 * FS - 1)  # too short by one sample
    write_case(tmp_path, 3, 600 * FS, n_nan=int(0.05 * 600 * FS) + 1)  # one NaN too many
    write_case(tmp_path, 4, 900 * FS)
    got = eligible_cases(tmp_path, min_duration_s=600.0, min_finite_frac=0.95)
    assert [c.caseid for c in got] == [1, 4]


def test_selection_stops_at_the_first_case_that_reaches_the_target(tmp_path):
    for cid in range(1, 8):
        write_case(tmp_path, cid, 10 * FS * 8)  # 10 estimated windows each
    sel = select_cases(tmp_path, target_windows=35, min_duration_s=0.0, min_finite_frac=0.0)
    assert [c.caseid for c in sel.cases] == [1, 2, 3, 4] and sel.cum_est_windows == 40  # 30 < 35 <= 40, crossing case included
    assert sel.n_scanned == 4 and sel.ineligible == []  # cases 5-7 are never opened
    exact = select_cases(tmp_path, target_windows=30, min_duration_s=0.0, min_finite_frac=0.0)
    assert [c.caseid for c in exact.cases] == [1, 2, 3] and exact.cum_est_windows == 30  # exact hit stops immediately
    assert select_cases(tmp_path, target_windows=0, min_duration_s=0.0, min_finite_frac=0.0).cases == []


def test_selection_skips_ineligible_cases_without_stopping_and_is_deterministic(tmp_path):
    write_case(tmp_path, 1, 700 * FS)  # 87 windows
    write_case(tmp_path, 2, 100 * FS)  # too short -> skipped, does not end the scan
    write_case(tmp_path, 3, 700 * FS, n_nan=100 * FS)  # 0.857 finite -> skipped
    write_case(tmp_path, 4, 700 * FS)
    write_case(tmp_path, 5, 700 * FS)
    sel = select_cases(tmp_path, target_windows=150)
    assert [c.caseid for c in sel.cases] == [1, 4] and sel.n_scanned == 4 and [c.caseid for c in sel.ineligible] == [2, 3]
    assert [c.caseid for c in select_cases(tmp_path, target_windows=150).cases] == [1, 4]  # rerun: identical
    pure = select_from_eligible(eligible_cases(tmp_path), target_windows=150)  # early-stopping == full-scan + pure rule
    assert [c.caseid for c in pure] == [c.caseid for c in sel.cases]


def test_max_cases_caps_the_selection(tmp_path):
    for cid in range(1, 6):
        write_case(tmp_path, cid, 700 * FS)
    sel = select_cases(tmp_path, target_windows=10_000, max_cases=2)
    assert [c.caseid for c in sel.cases] == [1, 2] and sel.n_scanned == 2


def test_select_from_eligible_is_pure_and_order_independent():
    """Step 3 sorts by caseid itself, so the rule does not depend on the order the summaries arrive in."""
    cs = [CaseSummary(cid, None, 0, FS, 1e4, 1.0, 1.0, True, 100) for cid in (5, 1, 3, 2, 4)]
    assert [c.caseid for c in select_from_eligible(cs, target_windows=250)] == [1, 2, 3]


def test_windows_are_non_overlapping_and_aligned(tmp_path):
    write_case(tmp_path, 9, 400 * 5 + 137)  # 5 full windows of fs*8 = 400 samples + a remainder that must be dropped
    w = windows_for_participant(tmp_path / "case_00009.npz", segment_len=8)
    raw = np.load(tmp_path / "case_00009.npz")
    assert w.subject == "case_00009" and (w.fs_ppg, w.fs_ecg) == (FS, FS)
    assert w.ppg.shape == w.ecg.shape == (5, 400) and w.ppg.dtype == w.ecg.dtype == np.float64
    assert np.array_equal(w.window_index, np.arange(5))
    for k in range(5):
        assert np.array_equal(w.ppg[k], raw["PLETH"][k * 400 : (k + 1) * 400])  # window k == the k-th contiguous block
        assert np.array_equal(w.ecg[k], raw["ECG_II"][k * 400 : (k + 1) * 400])  # both tracks share one clock
    assert w.notes == {"caseid": 9, "duration_s": (400 * 5 + 137) / FS, "finite_frac_ppg": 1.0, "finite_frac_ecg": 1.0}


def test_unequal_track_lengths_use_the_shorter_track(tmp_path):
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "case_00011.npz", srate=np.float32(FS), caseid=np.int32(11), PLETH=rng.normal(size=1700).astype(np.float32), ECG_II=rng.normal(size=1200).astype(np.float32))
    assert summarize_case(tmp_path / "case_00011.npz").n_samples == 1200
    assert windows_for_participant(tmp_path / "case_00011.npz").ppg.shape == (3, 400)


def test_case_shorter_than_one_window_yields_an_empty_windows_object(tmp_path):
    """A 4 s case at segment_len=8 has n_win == 0: the loader must return empty (0, fs*8) arrays, not raise."""
    write_case(tmp_path, 42, FS * 4)  # half a window
    w = windows_for_participant(tmp_path / "case_00042.npz", segment_len=8)
    assert w.ppg.shape == w.ecg.shape == (0, FS * 8) and w.ppg.dtype == w.ecg.dtype == np.float64
    assert w.window_index.shape == (0,) and w.subject == "case_00042"
    assert w.notes["duration_s"] == 4.0 and w.notes["finite_frac_ppg"] == 1.0


def test_nans_survive_the_loader_and_the_frozen_drop_rule_removes_them(tmp_path):
    """The loader must NOT impute or clip: windows 0 (NaN) and 2 (constant) are dropped only by the builder's rule."""
    rng = np.random.default_rng(3)
    ppg = rng.normal(size=400 * 4).astype(np.float32)
    ecg = rng.normal(size=400 * 4).astype(np.float32)
    ppg[:400] = np.nan  # window 0: dropout
    ecg[800:1200] = 0.0  # window 2: flatline (zero std)
    np.savez(tmp_path / "case_00021.npz", srate=np.float32(FS), caseid=np.int32(21), PLETH=ppg, ECG_II=ecg)
    w = windows_for_participant(tmp_path / "case_00021.npz")
    assert np.isnan(w.ppg[0]).all() and np.isfinite(w.ppg[1:]).all()
    keep = (np.isfinite(w.ppg).all(axis=1) & np.isfinite(w.ecg).all(axis=1)) & ((w.ppg.std(axis=1) > 0) & (w.ecg.std(axis=1) > 0))
    assert list(np.flatnonzero(keep)) == [1, 3]
    x = preprocess_windows(w.ppg[keep], 128, 8, **PPG_KW)
    y = preprocess_windows(w.ecg[keep], 128, 8, **ECG_KW)
    assert x.shape == y.shape == (2, 1024) and np.isfinite(x).all() and np.isfinite(y).all()


def test_builder_accepts_an_out_dir_outside_the_repo_root(tmp_path, monkeypatch):
    """--out may point anywhere: with the real ROOT, an outside path is recorded absolute instead of raising in relative_to."""
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    write_case(raw, 9, FS * 8 * 3)
    spec = importlib.util.spec_from_file_location("build_processed_vitaldb_real_root", ROOT / "scripts" / "build_processed_vitaldb.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # ROOT is NOT rebased here, so out really is outside it
    assert mod.ROOT == ROOT and not out.is_relative_to(ROOT)
    monkeypatch.setattr(sys, "argv", ["build_processed_vitaldb.py", "--raw", str(raw), "--out", str(out), "--min-duration-s", "0", "--target-windows", "1"])
    mod.main()
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["total_windows"] == 3 and man["files"]["case_00009"]["path"] == str(out / "case_00009.npz")


def test_caseid_array_must_agree_with_the_file_name(tmp_path):
    np.savez(tmp_path / "case_00005.npz", srate=np.float32(FS), caseid=np.int32(6), PLETH=np.zeros(400, np.float32), ECG_II=np.zeros(400, np.float32))
    with pytest.raises(AssertionError):
        summarize_case(tmp_path / "case_00005.npz")
