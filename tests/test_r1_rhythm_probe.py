"""R1 static/synthetic tests (docs/R1_PPG_GLOBAL_RHYTHM_OBSERVABILITY_PREREGISTRATION.md, c7481f9).

No real data, no checkpoint, no GPU required.
"""
import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.event_reliability import WildPPGTestFirewallError, assert_no_test_subjects
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.probes import rhythm_tcn as M

ROOT = Path(__file__).resolve().parents[1]
FS = 128


def _code_without_docstrings(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


# ---------------------------------------------------------------- firewall / split
def test_firewall_rejects_test_subjects():
    with pytest.raises(WildPPGTestFirewallError):
        assert_no_test_subjects(("an0", "kjd"))
    with pytest.raises(WildPPGTestFirewallError):
        assert_no_test_subjects(["ssx"])
    assert_no_test_subjects(C.TRAIN12 + C.VAL)


def test_r1_subject_sets_exclude_test_and_are_disjoint():
    for s in ("kjd", "ssx"):
        assert s not in C.TRAIN12 and s not in C.VAL
    assert len(C.TRAIN12) == 12 and set(C.TRAIN12).isdisjoint(C.VAL)
    for path in ("scripts/r1_train_probe.py", "scripts/r1_evaluate.py", "scripts/r1_ppi_rr_audit.py"):
        src = _code_without_docstrings(ROOT / path)
        assert "kjd" not in src and "ssx" not in src, path
        assert "assert_no_test_subjects" in src, path


def test_internal_dev_split_is_deterministic_and_inside_train12():
    a, b = C.internal_dev_split(), C.internal_dev_split()
    assert a == b
    assert len(a["internal_dev"]) == 2 and len(a["probe_train"]) == 10
    assert set(a["internal_dev"]).isdisjoint(a["probe_train"])
    assert set(a["internal_dev"]) | set(a["probe_train"]) == set(C.TRAIN12)
    assert set(a["internal_dev"]).isdisjoint(C.VAL)


def test_training_script_never_names_validation_subjects():
    src = _code_without_docstrings(ROOT / "scripts/r1_train_probe.py")
    assert "C.VAL" not in src and "an0" not in src and "k2s" not in src
    assert "split['internal_dev']" in src and "split['probe_train']" in src   # ast.unparse -> single quotes


# ---------------------------------------------------------------- PPG-only input, no ECG in the model
def test_model_forward_takes_ppg_and_site_only():
    params = list(inspect.signature(M.RhythmTCN.forward).parameters)
    assert params == ["self", "ppg", "site"]
    src = _code_without_docstrings(ROOT / "src/ppg2ecg/probes/rhythm_tcn.py")
    assert re.search(r"ecg", src, re.IGNORECASE) is None
    assert "attention" not in src.lower() and "Mamba" not in src and "KAN" not in src


def test_model_shape_and_module_inventory():
    net = M.RhythmTCN(M.GLOBAL_DILATIONS)
    x = torch.randn(3, 1, 1024)
    assert net(x).shape == (3, 1, 1024)
    kinds = {type(m).__name__ for m in net.modules()}
    assert kinds <= {"RhythmTCN", "_Block", "ModuleList", "Conv1d", "GELU"}


def test_evaluation_feeds_ppg_only_and_uses_ecg_for_labels_only():
    src = _code_without_docstrings(ROOT / "scripts/r1_evaluate.py")
    # every model call goes through probs(net, X, S, dev) whose X is built from d["x"] (PPG)
    assert "d['x'][idx]" in src and "d['y'][int(i)]" in src              # ast.unparse -> single quotes
    assert re.search(r"net\(\s*[^)]*d\['y'\]", src) is None
    assert "detect_rpeaks(d['y']" in src


# ---------------------------------------------------------------- cohorts
def _fake_meta(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sites = rng.choice(C.SITES, size=n)
    wi = rng.permutation(10_000)[:n]
    return sites, wi


def test_cohort_positions_deterministic_sorted_capped_and_site_pure():
    sites, wi = _fake_meta()
    a = C.cohort_positions("an0", sites, wi, 20)
    b = C.cohort_positions("an0", sites, wi, 20)
    for s in C.SITES:
        assert np.array_equal(a[s], b[s])
        assert len(a[s]) == min(20, int((sites == s).sum()))
        assert np.all(np.diff(a[s]) > 0)
        assert np.all(sites[a[s]] == s)
    big = C.cohort_positions("an0", sites, wi, 10_000)
    for s in C.SITES:
        assert len(big[s]) == int((sites == s).sum())


def test_cohort_depends_on_salt_and_subject_not_on_array_order():
    sites, wi = _fake_meta()
    a = C.cohort_positions("an0", sites, wi, 15)
    b = C.cohort_positions("an0", sites, wi, 15, salt=C.VISUAL_SALT)
    c = C.cohort_positions("k2s", sites, wi, 15)
    assert any(not np.array_equal(a[s], b[s]) for s in C.SITES)
    assert any(not np.array_equal(a[s], c[s]) for s in C.SITES)
    perm = np.random.default_rng(1).permutation(len(wi))
    d = C.cohort_positions("an0", sites[perm], wi[perm], 15)
    for s in C.SITES:
        assert set(wi[a[s]].tolist()) == set(wi[perm][d[s]].tolist())


def test_same_validation_windows_for_every_model_and_control():
    assert C.n_per_for("an0") == C.N_VAL_PER == 1024 and C.n_per_for("k2s") == 1024
    assert C.n_per_for("e61") == C.N_TRAIN_PER == 2048
    src = _code_without_docstrings(ROOT / "scripts/r1_evaluate.py")
    # validation arrays are built exactly once and every arm/control indexes the same Xv/gt_v
    assert src.count("for sub in C.VAL:") >= 1
    assert src.count("C.cohort_positions(sub, d['site'], d['window_index'], C.n_per_for(sub))") == 2  # dev + val


def test_visual_atlas_cohort_is_metadata_only_eight_per_stratum():
    sites, wi = _fake_meta()
    v = C.cohort_positions("an0", sites, wi, C.N_VISUAL_PER, salt=C.VISUAL_SALT)
    assert all(len(v[s]) == 8 for s in C.SITES)
    assert C.VISUAL_SALT == "r1-visual-v1" and C.COHORT_SALT == "r1-global-rhythm-observability-v1"


# ---------------------------------------------------------------- receptive field / capacity
def test_receptive_fields_meet_frozen_bounds():
    g = M.RhythmTCN(M.GLOBAL_DILATIONS)
    l = M.RhythmTCN(M.LOCAL_DILATIONS)
    assert g.rf == 2041 and g.rf >= 1024
    assert l.rf == 65 and l.rf / FS * 1000.0 <= 512.0
    assert M.receptive_field((1,), k=5, convs_per_block=2) == 9


def test_global_and_local_have_identical_parameter_count_and_shapes():
    g = M.RhythmTCN(M.GLOBAL_DILATIONS)
    l = M.RhythmTCN(M.LOCAL_DILATIONS)
    assert M.n_trainable(g) == M.n_trainable(l) == 328_897
    assert [p.shape for p in g.parameters()] == [p.shape for p in l.parameters()]
    torch.manual_seed(42); g2 = M.RhythmTCN(M.GLOBAL_DILATIONS)
    torch.manual_seed(42); l2 = M.RhythmTCN(M.LOCAL_DILATIONS)
    for a, b in zip(g2.parameters(), l2.parameters()):
        assert torch.equal(a, b)                              # identical init under the shared seed


def test_site_variant_adds_only_film_embedding_and_starts_as_identity():
    s = M.RhythmTCN(M.GLOBAL_DILATIONS, n_sites=4)
    assert M.n_trainable(s) - 328_897 == 4 * 2 * 64
    g = M.RhythmTCN(M.GLOBAL_DILATIONS)
    g.load_state_dict({k: v for k, v in s.state_dict().items() if not k.startswith("film.")})
    x = torch.randn(2, 1, 1024)
    assert torch.allclose(s(x, torch.tensor([0, 3])), g(x))


# ---------------------------------------------------------------- soft target / events
def test_sigma_and_refractory_conversions():
    assert M.SIGMA_SAMPLES == pytest.approx(12.8)
    assert M.REFRACTORY_SAMPLES == 32
    assert M.THRESH_GRID[0] == 0.05 and M.THRESH_GRID[-1] == 0.95 and len(M.THRESH_GRID) == 19
    assert M.TOL_MS == (50.0, 100.0, 150.0, 200.0, 250.0)
    assert [t / 1000.0 * FS for t in M.TOL_MS] == pytest.approx([6.4, 12.8, 19.2, 25.6, 32.0])


def test_soft_event_field_is_max_combined_gaussian():
    y = M.soft_event_field([100, 120], 1024)
    assert y.dtype == np.float32 and y.shape == (1024,)
    assert y[100] == pytest.approx(1.0) and y[120] == pytest.approx(1.0)
    assert y.max() <= 1.0 + 1e-6                               # max, never sum
    y1 = M.soft_event_field([500], 1024)
    assert y1[500 + 13] == pytest.approx(np.exp(-(13 ** 2) / (2 * 12.8 ** 2)), rel=1e-5)
    assert np.all(M.soft_event_field([], 1024) == 0)


def test_extract_events_threshold_nms_and_edges():
    p = np.zeros(1024); p[100] = 0.9; p[120] = 0.6; p[200] = 0.7; p[300] = 0.04
    ev = M.extract_events(p, 0.05)
    assert ev.tolist() == [100, 200]                           # 120 suppressed (20 < 32), 300 below threshold
    p2 = np.zeros(1024); p2[100] = 0.5; p2[133] = 0.5
    assert M.extract_events(p2, 0.5).tolist() == [100, 133]    # 33 > 32 survives
    p3 = np.linspace(1, 0, 1024)                               # edge maximum at 0
    assert M.extract_events(p3, 0.5).tolist() == [0]
    assert M.extract_events(np.zeros(1024), 0.05).size == 0


# ---------------------------------------------------------------- matching / RR
def test_matcher_is_one_to_one_and_symmetric_in_denominators():
    gt = np.array([100, 300, 500]); pred = np.array([102, 104, 300, 700])
    m, fp, fn = R.match_rpeaks(gt, pred, FS, 50.0)
    assert len(m) == 2 and fp == 2 and fn == 1
    assert len({i for i, _ in m}) == len(m) and len({j for _, j in m}) == len(m)
    p, r, f = R.prf(len(m), fp, fn)
    assert p == pytest.approx(0.5) and r == pytest.approx(2 / 3)


def test_rr_pairs_require_consecutive_matched_beats():
    import importlib.util
    spec = importlib.util.spec_from_file_location("r1_evaluate", ROOT / "scripts/r1_evaluate.py")
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    gt = np.array([100, 200, 300, 400]); pred = np.array([101, 199, 405])
    s = ev.score_window(gt, pred, 150.0)
    rr = ev.rr_window(gt, pred, s["matches"])
    assert len(rr) == 1                                         # only (100,200) both matched
    assert rr[0][0] == pytest.approx(100 / FS * 1000) and rr[0][1] == pytest.approx(98 / FS * 1000)


# ---------------------------------------------------------------- threshold freeze ordering
def test_threshold_selected_on_internal_dev_before_validation_is_loaded():
    src = (ROOT / "scripts/r1_evaluate.py").read_text()
    i_thr = src.index('threshold_selection.json')
    i_val = src.index("for sub in C.VAL:")
    assert i_thr < i_val
    assert "F1@150ms on INTERNAL_DEV" in src
    sel = src[src.index("_cache_internal_dev.npz"):i_thr]
    assert "C.VAL" not in sel and "Xv" not in sel


# ---------------------------------------------------------------- controls
def test_derangement_has_no_fixed_points_and_is_a_permutation():
    rng = np.random.default_rng(0)
    for n in (2, 3, 7, 64, 1024):
        p = C.derangement(n, rng)
        assert sorted(p.tolist()) == list(range(n))
        assert not np.any(p == np.arange(n))
    with pytest.raises(ValueError):
        C.derangement(1, rng)


def test_circular_offsets_are_one_to_four_seconds():
    off = C.circular_offsets(20_000, np.random.default_rng(20260902))
    assert off.min() >= 128 and off.max() <= 512
    assert off.min() == 128 and off.max() == 512                 # inclusive bounds reached
    assert np.array_equal(off, C.circular_offsets(20_000, np.random.default_rng(20260902)))


def test_circular_shift_preserves_window_content():
    x = np.random.default_rng(0).standard_normal(1024)
    y = np.roll(x, 300)
    assert np.allclose(np.sort(x), np.sort(y)) and not np.allclose(x, y)


# ---------------------------------------------------------------- bootstrap
def test_paired_bootstrap_weights_subjects_equally():
    subj = np.array(["A"] * 900 + ["B"] * 100)
    earlier = np.zeros(1000)
    later = np.concatenate([np.zeros(900), np.ones(100)])
    r = paired_subject_bootstrap(earlier, later, subj, "higher_better", n_boot=200, seed=1)
    assert r["point"] == pytest.approx(0.5)                    # not the window-pooled 0.1
    assert r["verdict"] == "improves"
    r2 = paired_subject_bootstrap(later, earlier, subj, "lower_better", n_boot=200, seed=1)
    assert r2["point"] == pytest.approx(0.5)
