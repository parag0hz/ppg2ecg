"""E0 unit tests (docs/E0_WWDET_SPONTANEOUS_EVENT_AUDIT_PREREGISTRATION.md §10). Synthetic data, the committed
reproduction record and static checks only; no event-level outcome is computed here."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.eventaudit import features as FT
from ppg2ecg.eventaudit import probe as PR
from ppg2ecg.eventaudit import taxonomy as TX

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def e0():
    sys.path.insert(0, str(ROOT / "scripts"))
    import e0_event_audit
    return e0_event_audit


# ------------------------------------------------------------------------------------------ 1 frozen reproduction
def test_frozen_c0a_outputs_reproduce(e0):
    rep = json.loads((ROOT / "artifacts/e0_wwdet_event_audit/reproduction.json").read_text())
    for role in ("val", "holdout"):
        assert rep[role]["raw_equal_c0a"] is True
        assert all(v == 0.0 for v in rep[role]["max_abs_dev_vs_c0a"].values())
    src = inspect.getsource(e0.stage_reproduce)
    assert "STOP" in src and "1e-9" in src


# ------------------------------------------------------------------------------------------ 2-7 taxonomy
def _cls():
    ref = np.array([50, 150, 250, 350])
    placed = np.array([52, 148, 300, 450])         # 300 and 450 are false placed events; 250 and 350 are missed
    rendered = np.array([51, 149, 152, 251, 301, 400, 449])
    return ref, placed, rendered, TX.classify(ref, placed, rendered)


def test_matching_is_one_to_one():
    _, _, _, c = _cls()
    r = c["ref_idx"][c["ref_idx"] >= 0]
    p = c["placed_idx"][c["placed_idx"] >= 0]
    assert len(set(r)) == len(r) and len(set(p)) == len(p)


def test_types_mutually_exclusive_and_exhaustive():
    _, _, rendered, c = _cls()
    assert len(c["type"]) == rendered.size
    assert set(c["type"]) <= set(TX.TYPES)
    n = TX.counts(c)
    assert sum(n.values()) == rendered.size


def test_type_definitions():
    _, _, _, c = _cls()
    for ty, r, p in zip(c["type"], c["ref_idx"], c["placed_idx"]):
        assert (ty == "A") == (r >= 0 and p >= 0)
        assert (ty == "B") == (r < 0 and p >= 0)
        assert (ty == "C") == (r >= 0 and p < 0)
        assert (ty == "D") == (r < 0 and p < 0)


def test_ambiguous_nearby_events():
    ref, placed, rendered, c = _cls()
    ty = dict(zip(rendered.tolist(), c["type"].tolist()))
    # two rendered detections near one placed / reference beat: only the closer one is assigned
    assert ty[149] == "A" and ty[152] == "D"
    assert ty[51] == "A"
    assert ty[251] == "C"          # missed reference beat, no placed event nearby
    assert ty[301] == "B"          # false placed event preserved
    assert ty[400] == "D"
    assert ty[449] == "B"
    # a rendered detection between a reference beat and a placed event 60 ms apart is assigned to both axes on its own
    c2 = TX.classify(np.array([100]), np.array([108]), np.array([104]))
    assert c2["type"].tolist() == ["A"]
    acc = TX.accounting(c, ref.size, placed.size)
    assert acc["tp_rendered"] - acc["tp_placed"] == acc["C"] - acc["L"]
    assert acc["fp_rendered"] - acc["fp_placed"] == acc["D"] - acc["M"]


# ------------------------------------------------------------------------------------------ 8-10 counterfactuals
def test_hard_guard_removes_only_off_placed_detections():
    _, placed, rendered, _ = _cls()
    kept = TX.hard_guard(rendered, placed)
    tol = 50.0 / 1000 * 128
    for t in rendered:
        assert (t in kept) == bool(np.min(np.abs(placed - t)) <= tol)
    assert 152 in kept               # near a placed event although it lost the one-to-one assignment
    assert TX.hard_guard(rendered, np.array([], int)).size == 0


def test_oracles():
    _, _, rendered, c = _cls()
    o1 = TX.remove_types(rendered, c["type"], ("D",))
    o2 = TX.remove_types(rendered, c["type"], ("B", "D"))
    keep1 = [t for t, ty in zip(rendered, c["type"]) if ty != "D"]
    keep2 = [t for t, ty in zip(rendered, c["type"]) if ty in ("A", "C")]
    assert o1.tolist() == keep1 and o2.tolist() == keep2


# ------------------------------------------------------------------------------------------ 11-12 localization
def test_phase_in_unit_interval_and_undefined_at_edges():
    placed = np.array([100, 200, 300])
    for t in range(100, 300):          # t on the last placed event has no next event
        loc = TX.localize(t, placed)
        assert 0.0 <= loc["phase"] <= 1.0 and loc["region"] == "between"
    for t, reg in ((20, "before_first"), (400, "after_last")):
        loc = TX.localize(t, placed)
        assert loc["region"] == reg and np.isnan(loc["phase"])
    loc = TX.localize(50, np.array([], int))
    assert loc["region"] == "no_placed" and np.isnan(loc["phase"]) and np.isnan(loc["dist_nearest_ms"])
    assert TX.bin_index(1.0, TX.PHASE_EDGES) == len(TX.PHASE_EDGES) - 2
    assert TX.bin_index(0.5, TX.PHASE_EDGES) == 3
    assert TX.bin_index(600.0, TX.ABS_EDGES_MS) == 4


# ------------------------------------------------------------------------------------------ 13-16 template / probe
def test_qrs_template_from_train_only(e0):
    src = inspect.getsource(e0.stage_train_infer)
    assert 'render_role("train"' in src and "FT.qrs_template(Y, ref)" in src
    for f in (e0.stage_evaluate, e0.evaluate_role, e0.stage_reproduce):
        assert "qrs_template(" not in inspect.getsource(f)
    y = np.sin(np.linspace(0, 6, 512))
    tpl = FT.qrs_template([y], [np.array([5, 100, 508])])     # 5 and 508 have incomplete supports
    assert np.allclose(tpl, y[90:111])
    c, l2 = FT.template_similarity(y, 100, tpl)
    assert c == pytest.approx(1.0) and l2 == pytest.approx(0.0, abs=1e-9)


def test_standardization_uses_train_only():
    rng = np.random.default_rng(0)
    A = rng.normal(5, 2, (200, 3))
    A[::7, 1] = np.nan
    Bm = rng.normal(-3, 1, (50, 3))
    p = PR.fit_prep(A, ["a", "b", "c"])
    Z = PR.transform(p, Bm)
    assert np.allclose(Z[:, 0], (Bm[:, 0] - p.mean[0]) / p.std[0])
    assert p.indicator_cols.tolist() == [1] and p.names[-1] == "b_missing"
    assert abs(p.mean[0] - np.mean(A[:, 0])) < 1e-12


def test_probe_cv_is_patient_grouped_and_grid_fixed():
    rng = np.random.default_rng(1)
    n = 400
    groups = np.repeat(np.arange(40), 10)
    y = rng.integers(0, 2, n)
    F = np.c_[y + rng.normal(0, 1, n), rng.normal(0, 1, n)]
    pb = PR.fit_probe(F, y, groups)
    assert set(pb.cv) == {str(c) for c in PR.C_GRID} and pb.c in PR.C_GRID
    assert all(len(v["fold_auroc"]) == PR.N_FOLDS for v in pb.cv.values())
    from sklearn.model_selection import GroupKFold
    for tr, te in GroupKFold(n_splits=PR.N_FOLDS).split(F, y, groups):
        assert not (set(groups[tr]) & set(groups[te]))
    assert "GroupKFold" in inspect.getsource(PR.fit_probe) and "class_weight=\"balanced\"" in inspect.getsource(PR._model)


def test_val_holdout_never_enter_probe_fitting(e0):
    src = inspect.getsource(e0.stage_evaluate)
    assert src.count("PR.fit_probe(") == 1 and 'event_level_train.parquet' in src
    assert "fit_probe" not in inspect.getsource(e0.evaluate_role) and "fit_prep" not in inspect.getsource(e0.evaluate_role)
    assert "PR.predict(" in inspect.getsource(e0.evaluate_role)


# ------------------------------------------------------------------------------------------ 17-18 bootstrap / atlas
def test_patient_bootstrap_is_clustered():
    rng = np.random.default_rng(2)
    pid_c = np.repeat([1, 2, 3], 5)
    pid_d = np.repeat([1, 2, 3, 4], 4)
    vc, vd = rng.normal(1, 1, pid_c.size), rng.normal(0, 1, pid_d.size)
    subs = np.array([1, 2, 3, 4])
    r = np.array([0, 0, 2, 3])                                # patient 1 twice, 3, 4; patient 2 absent
    one = PR.cohens_d_cluster(vc, pid_c, vd, pid_d, [r] * 3, subs)
    take = lambda v, p: np.concatenate([v[p == subs[k]] for k in r])  # noqa: E731
    c, d = take(vc, pid_c), take(vd, pid_d)
    pooled = np.sqrt(((c.size - 1) * c.var(ddof=1) + (d.size - 1) * d.var(ddof=1)) / (c.size + d.size - 2))
    assert one[1] == pytest.approx((c.mean() - d.mean()) / pooled)
    full = (vc.mean() - vd.mean()) / np.sqrt(((vc.size - 1) * vc.var(ddof=1) + (vd.size - 1) * vd.var(ddof=1)) / (vc.size + vd.size - 2))
    assert one[0] == pytest.approx(full)


def test_atlas_sampling_is_fixed(e0):
    import bf0_run as B
    keys = [f"{w}:{t}" for w in range(50) for t in (10, 200)]
    assert np.array_equal(B.salted_rank("e0-atlas-v1-C", keys), B.salted_rank("e0-atlas-v1-C", keys))
    src = inspect.getsource(e0.stage_atlas)
    assert 'salted_rank(f"e0-atlas-v1-{ty}"' in src and 'salted_rank("e0-atlas-v1-diff"' in src and "[:12]" in src


# ------------------------------------------------------------------------------------------ frozen case rule
def _q(**kw):
    q = {"dD": [0.10, 0.09, 0.11], "dC": [0.03, 0.02, 0.04], "dFP": [0.11, 0.10, 0.12], "share_C": 0.25,
         "hg_dF1": [-0.004, -0.006, -0.002], "hg_dFP": [-0.12, -0.13, -0.11], "hg_dRecall": [-0.01, -0.012, -0.008],
         "o1_dF1": [0.01, 0.008, 0.012], "o1_dFP": [-0.10, -0.11, -0.09], "auroc_p3": 0.85}
    q.update(kw)
    return q


def test_case_rules():
    assert PR.case_of(PR.criteria(_q(), "val")) == "E0-A"
    assert PR.case_of(PR.criteria(_q(auroc_p3=0.70), "val")) == "E0-C"
    assert PR.criteria(_q(auroc_p3=0.77), "holdout")["K3_separable"] is True
    b = _q(share_C=0.05, dC=[0.005, 0.001, 0.009], hg_dF1=[0.006, 0.004, 0.008], hg_dRecall=[-0.004, -0.006, -0.002], auroc_p3=0.6)
    assert PR.case_of(PR.criteria(b, "val")) == "E0-B"
    assert PR.case_of(PR.criteria(_q(dD=[0.03, 0.02, 0.04]), "val")) == "E0-C"          # D explains < half of the FP gap
    sel = PR.criteria(_q(hg_dRecall=[-0.03, -0.035, -0.025]), "val")
    assert sel["K2c_hardguard_recall_material_loss"] and sel["guard_must_be_selective"]
    assert PR.final_case("E0-A", "E0-A") == "E0-A" and PR.final_case("E0-A", "E0-C") == "E0-C"


# ------------------------------------------------------------------------------------------ features
def test_features_are_finite_on_a_synthetic_qrs():
    t = np.arange(512)
    x = 0.05 * np.sin(2 * np.pi * t / 128) + np.exp(-0.5 * ((t - 200) / 2.5) ** 2)
    tpl = x[190:211].copy()
    f = FT.waveform_features(x, 201, tpl)
    assert FT.refine(x, 201) == 200
    assert f["prominence"] > 0.9 and f["width_ms"] > 0 and f["max_pos_slope"] > 0 > f["max_neg_slope"]
    assert f["qrs_corr"] == pytest.approx(1.0) and 0 < f["hf_frac"] < 1
    seg = FT.segment(x, 5)
    assert seg.size == 2 * FT.H_LOCAL + 1 and np.isnan(seg[: FT.H_LOCAL - 5]).all() and seg[FT.H_LOCAL] == x[5]
    pf = FT.ppg_features(np.linspace(0, 1, 512), 100, np.array([90, 140]))
    assert pf["ppg_peak_lag_ms"] == pytest.approx(-10 * 1000 / 128)
    assert set(FT.GROUPS["P3"]) >= set(FT.GROUPS["P1"]) | set(FT.GROUPS["P2"])
    assert not set(FT.GROUPS["P1"]) & set(FT.GROUPS["P2"])


def test_no_model_training_in_e0(e0):
    src = inspect.getsource(e0)
    for bad in ("optimizer", ".backward(", "AdamW", "train()"):
        assert bad not in src
    assert "render_ww" in src and "load_net" not in src.replace("C0.load_net", "")


def test_event_table_keeps_localization_and_features(e0):
    """Regression: features 11-14 equal the localization of every row; no merge duplicates."""
    class Ex:
        map = staticmethod(lambda f, it, chunksize=1: map(f, it))
    t = np.arange(512)
    q = lambda pos, a=1.0: a * np.exp(-0.5 * ((t[:, None] - np.asarray(pos)[None, :]) / 2.5) ** 2).sum(axis=1)  # noqa: E731
    ref, events = [np.array([60, 160, 260, 360])], [np.array([60, 160, 360])]
    ww = q([60, 160, 260, 360, 430]) - 0.5
    c0 = q([60, 160, 360]) - 0.5
    det = {"WW": [np.array([60, 160, 260, 360, 430])], "C0": [np.array([60, 160, 360])]}
    cls = {k: {0: TX.classify(ref[0], events[0], det[k][0])} for k in ("WW", "C0")}
    df = e0.event_table("val", np.array([7]), ref, events, det, cls, {"WW": ww[None], "C0": c0[None]},
                        np.sin(t / 20.0)[None].astype(np.float32), q([100])[90:111], Ex())
    w = df[df["arm"] == "WW"].set_index("sample")
    assert w.loc[260, "type"] == "C" and w.loc[430, "type"] == "D"
    assert w.loc[260, "phase"] == pytest.approx(TX.localize(260, events[0])["phase"])
    assert w.loc[260, "prev_rr_ms"] == pytest.approx(100 * 1000 / 128) and np.isfinite(w.loc[430, "prominence"])
    assert np.isnan(w.loc[60, "prominence"]) and not [c for c in df.columns if c.endswith(("_x", "_y"))]
