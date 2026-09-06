"""D2 baseline predictors (docs/D2_BASELINE_FLOOR_PREREGISTRATION.md §4)."""
from __future__ import annotations

import numpy as np
import pytest

from ppg2ecg.evaluation import baselines as B

FS = B.FS


def spike_train(T: int, positions, amp: float = 1.0) -> np.ndarray:
    x = np.zeros(T)
    x[np.asarray(positions, dtype=int)] = amp
    return x


def test_cyclic_partner_stays_within_subject_and_is_a_derangement_when_possible():
    subj = np.array(["a", "a", "a", "b", "b", "c"])
    p = B.cyclic_partner(subj)
    assert (subj[p] == subj).all(), "a partner must belong to the same subject"
    assert (p[:5] != np.arange(5)).all(), "subjects with >= 2 rows must map to a different row"
    assert p[5] == 5, "a single-row subject maps to itself and is counted by the caller, not silently scored"
    assert sorted(p[:3]) == [0, 1, 2] and sorted(p[3:5]) == [3, 4], "partners permute within a subject"


def test_place_is_additive_and_clips_at_the_edges_without_wrapping():
    t = np.array([1.0, 2.0, 1.0])
    out = B._place(np.array([1, 5]), t, 8)
    assert out.tolist() == [1, 2, 1, 0, 1, 2, 1, 0]
    edge = B._place(np.array([0, 7]), t, 8)                       # half of each template falls outside
    assert edge.tolist() == [2, 1, 0, 0, 0, 0, 1, 2], "edges clip; nothing wraps around"
    overlap = B._place(np.array([2, 3]), t, 6)
    assert overlap.tolist() == [0, 1, 3, 3, 1, 0], "overlapping templates SUM, as a real beat train does"


def test_fit_pat_identifies_the_offset_only_up_to_the_matcher_tolerance_and_breaks_ties_to_the_smallest():
    """The +-50 ms matcher is 6.4 samples wide, so a synthetic exact-offset signal has a PLATEAU of maximisers.

    The contract is therefore not "recovers the true offset" but "returns a maximiser, and the smallest one when
    several tie". On real corpora the maximum is a single sharp peak (measured on CapnoBase train: width 1 at
    offset 31), so the tie-break does not bind there — it exists only to make the synthetic case deterministic.
    """
    n = 8
    pk = np.arange(100, 100 + n * 100, 100)
    true_off = 20
    ppg = [pk.copy() for _ in range(4)]
    gt = [pk + true_off for _ in range(4)]
    off, rate = B.fit_pat(ppg, gt)
    tol_samples = int(round(B.MATCH_TOL_MS / 1000 * FS))
    assert rate == pytest.approx(1.0)
    assert abs(off - true_off) <= tol_samples, "the returned offset must be inside the tolerance plateau"
    plateau = [o for o in range(129) if B.match_rate([p + o for p in ppg], gt) == pytest.approx(1.0)]
    assert off == min(plateau), "ties break to the smallest maximiser, deterministically"
    # a signal that matches at NO offset must not silently return a large arbitrary one
    off0, _ = B.fit_pat([np.zeros(0, dtype=np.int64)] * 4, gt)
    assert off0 == 0, "no evidence -> the smallest offset, deterministically"


def test_match_rate_is_recall_against_the_reference_and_is_nan_without_reference_beats():
    ref = [np.array([100, 300, 500])]
    assert B.match_rate([np.array([100, 300, 500])], ref) == pytest.approx(1.0)
    assert B.match_rate([np.array([100])], ref) == pytest.approx(1 / 3)
    assert B.match_rate([np.array([100 + int(0.2 * FS)])], ref) == 0.0, "200 ms is outside the 50 ms tolerance"
    assert np.isnan(B.match_rate([np.array([100])], [np.zeros(0, dtype=np.int64)]))


def test_beat_stack_drops_edge_beats_rather_than_zero_padding_them():
    T, half = 100, 10
    y = np.arange(T, dtype=np.float64)[None, :]
    st = B._beat_stack(y, [np.array([5, 50, 95])], half)
    assert st.shape == (1, 2 * half + 1), "only the fully-interior beat survives"
    assert st[0, half] == 50.0, "the surviving beat is centred on its own position"


def test_qrs_template_is_zero_at_its_edges_so_placement_injects_no_step():
    """prereg §4 calls B1 'zero outside' its support; a template sitting on the ECG baseline would step at each edge."""
    T, n = 1024, 10
    pk = np.arange(80, 80 + n * 90, 90)
    base = -0.4
    y = np.stack([spike_train(T, pk, 1.0) + base for _ in range(6)])
    x = np.stack([spike_train(T, pk - 20, 1.0) for _ in range(6)])
    fit = B.fit_on_train(y, x)
    assert abs(fit.qrs_template[0] + fit.qrs_template[-1]) < 1e-9, "template ends are centred on zero"
    assert fit.mean_waveform.shape == (T,)


def test_predictors_return_the_right_shape_and_b3_is_window_independent():
    T = 256
    fit = B.TrainFit(pat_offset=3, pat_train_match_rate=0.5, qrs_template=np.array([0.0, 1.0, 0.0]),
                     mean_beat=np.array([0.0, 0.5, 0.0]), mean_waveform=np.arange(T, dtype=np.float64),
                     n_train_beats=7)
    out = B.b3(5, fit)
    assert out.shape == (5, T) and (out == fit.mean_waveform).all(), "B3 uses nothing from the window"
    p = B.predict_template([np.array([10]), np.array([20, 30])], fit.qrs_template, T)
    assert p.shape == (2, T) and p[0, 10] == 1.0 and p[1, 20] == 1.0 and p[1, 30] == 1.0
