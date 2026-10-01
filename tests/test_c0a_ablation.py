"""C0-A unit tests (docs/C0A_COHERENTBEAT_ABLATION_PREREGISTRATION.md §9). Synthetic data and the V1 manifest only."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.beatfirst import model as BM
from ppg2ecg.coherentbeat import ablation as AB
from ppg2ecg.coherentbeat import model as CM
from ppg2ecg.coherentbeat import split as SP

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ca():
    sys.path.insert(0, str(ROOT / "scripts"))
    import c0a_ablation
    return c0a_ablation


# ------------------------------------------------------------------------------------------ frozen C0 reuse (1-4, 17)
def test_c0_split_reused_and_free_of_old_heldout_patients():
    m = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())
    sp = SP.arch_split(m)
    assert {r: len(sp[r]["patients"]) for r in sp} == {"train": 3470, "val": 433, "holdout": 434}
    old = SP.old_heldout_patients(m)
    allp = set().union(*(set(sp[r]["patients"]) for r in sp))
    assert not (allp & set(old["val"])) and not (allp & set(old["test"]))


def test_c0a_reuses_c0_data_detector_events_and_preprocessing(ca):
    ev = inspect.getsource(ca.evaluate)
    assert "load_role(role)" in ev and "C0.reference_peaks(role, Y, ex)" in ev
    assert 'np.load(C0.OUT / f"{role}_renders.npz")' in ev and "C0.detect_events(X, dev, ex)" in ev   # frozen events, re-verified
    assert "STOP: frozen C0 events not reproduced" in ev
    assert "return C0.load_arch(role)" in inspect.getsource(ca.load_role)
    for fn in (ca.stage_train_pm, ca._train_window_model):
        assert 'C0.load_arch("train")' in inspect.getsource(fn)
    assert not any(k.startswith("train_detector") for k in ca.STAGES)                # the detector is never retrained


def test_every_waveform_arm_uses_the_same_event_sequence(ca):
    ev = inspect.getsource(ca.evaluate)
    for call in ("C0.render_bf0det(X, events, dev)", 'C0.render_c0("c0_localonly", X, events, dev, rr_edge)',
                 'C0.render_c0("c0", X, events, dev, rr_edge)', "render_pm(X, events, dev)",
                 "render_const(X, events, dev, rr_edge)", "render_ww(X, events, dev)"):
        assert call in ev


def test_holdout_sealed_without_c0a_freeze(ca, tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "ART", tmp_path)
    monkeypatch.setattr(ca, "FREEZE", tmp_path / "holdout_ablation_freeze.json")
    with pytest.raises(PermissionError):
        ca.load_role("holdout")
    with pytest.raises(PermissionError):
        ca.stage_evaluate_holdout(None, None)


# ------------------------------------------------------------------------------------------ PM-BF0 (5-7)
def test_pm_bf0_keeps_the_bf0_family_and_is_chosen_by_parameter_count_only():
    cands = AB.pm_bf0_candidates()
    key, n, mis = AB.select_closest(cands)
    assert key == min(cands, key=lambda c: abs(cands[c] - AB.C0_PARAMS))
    assert abs(mis) <= 0.02 and n == BM.n_params(BM.BeatFlowNet(ch=key))
    pm, bf = BM.BeatFlowNet(ch=key), BM.BeatFlowNet()
    assert type(pm) is type(bf)
    assert [b.c1.dilation for b in pm.blocks] == [b.c1.dilation for b in bf.blocks]
    assert pm.t_dim == bf.t_dim and pm.cond[0].out_features == bf.cond[0].out_features == BM.COND_HIDDEN
    assert pm.blocks[0].c1.kernel_size == bf.blocks[0].c1.kernel_size


def test_select_closest_ties_and_mismatch():
    assert AB.select_closest({"a": 90, "b": 110, "c": 120}, target=100) == ("a", 90, -0.1)
    assert AB.select_closest({"x": 101}, target=100)[2] == pytest.approx(0.01)


# ------------------------------------------------------------------------------------------ CONST (8-10)
def test_const_global_is_constant_has_no_time_varying_path_and_keeps_c0_local_branch():
    torch.manual_seed(0)
    c0 = CM.CoherentBeat(use_global=True).eval()
    const = AB.ConstGlobalLocal().eval()
    assert not hasattr(const, "global_head") and not hasattr(const, "basis")
    assert BM.n_params(const) == BM.n_params(c0)
    shared = {k: v for k, v in c0.state_dict().items() if not k.startswith("global_head")}
    const.load_state_dict(shared | {k: v for k, v in const.state_dict().items() if k.startswith("const_head")})
    x = torch.randn(3, 512)
    ev = [np.array([90, 200, 330, 450]), np.array([256]), np.array([], int)]
    with torch.no_grad():
        _, g0, l0 = c0(x, ev, 100.0, return_parts=True)
        _, gc, lc = const(x, ev, 100.0, return_parts=True)
    torch.testing.assert_close(lc, l0)                                              # identical local design and weights
    assert torch.all(gc == gc[:, :1])                                               # exactly constant over time
    assert torch.any(g0 != g0[:, :1])                                               # (C0's field does vary)
    assert inspect.getsource(AB.ConstGlobalLocal.global_field).count("expand") == 1


# ------------------------------------------------------------------------------------------ WW-DET (11-15)
def test_ww_det_uses_the_raster_outputs_512_directly_and_has_no_local_renderer():
    cands = AB.ww_candidates()
    key, n, mis = AB.select_closest(cands)
    assert key == min(cands, key=lambda c: (abs(cands[c] - AB.C0_PARAMS), list(cands).index(c)))
    assert abs(mis) <= 0.05
    torch.manual_seed(1)
    net = AB.WWDet(*key).eval()
    x = torch.randn(2, 512)
    r1 = torch.from_numpy(AB.event_raster([np.array([100, 300]), np.array([200])]))
    r2 = torch.from_numpy(AB.event_raster([np.array([150, 350]), np.array([250])]))
    with torch.no_grad():
        y1, y2 = net(x, r1), net(x, r2)
    assert y1.shape == (2, 512) and not torch.allclose(y1, y2)
    src = inspect.getsource(AB.WWDet.__init__) + inspect.getsource(AB.WWDet.forward)   # code, not the class docstring
    for banned in ("envelope", "supports", "bump", "tau", "index_add", "global_field", "assemble"):
        assert banned not in src
    r = AB.event_raster([np.array([100])])[0]
    assert r[100] == pytest.approx(1.0) and r[100 + 3] == pytest.approx(np.exp(-9 / (2 * AB.RASTER_SIGMA ** 2)), rel=1e-5)


# ------------------------------------------------------------------------------------------ determinism (16)
def test_no_stochastic_path_in_new_models():
    torch.manual_seed(2)
    x = torch.randn(2, 512)
    ev = [np.array([100, 260, 410]), np.array([128, 384])]
    for net, args in ((AB.ConstGlobalLocal().eval(), (ev, 100.0)), (AB.WWDet(72, 5).eval(), (torch.from_numpy(AB.event_raster(ev)),))):
        with torch.no_grad():
            torch.testing.assert_close(net(x, *args), net(x, *args))
    assert "randn" not in inspect.getsource(AB) and "rand(" not in inspect.getsource(AB)


# ------------------------------------------------------------------------------------------ statistics (18-20)
def test_paired_patient_bootstrap_and_matched_fd(ca):
    ev = inspect.getsource(ca.evaluate)
    assert "C0.cluster_ci(per[a][m] - per[b][m], Pid)" in ev
    assert 'C0.fd_diff_ci(waves["C0"], waves[other], Y, Pid)' in ev
    import c0_coherentbeat as C0
    assert C0.BOOT_N == 2000 and C0.BOOT_SEED == 20261001
    rng = np.random.default_rng(0)
    a, b, Y = (rng.normal(size=(40, 8)) for _ in range(3))
    C0._FD.update(a=a, b=b, Y=Y)
    idx = np.array([0, 0, 5, 7, 7, 9] * 6)
    from ppg2ecg.evaluation import paper_metrics as PMX
    assert C0._fd_draw(idx) == pytest.approx(PMX.kanflow_fd(a[idx], Y[idx]) - PMX.kanflow_fd(b[idx], Y[idx]))


def test_c0_reproduction_deviation_is_strict(ca):
    st = {"f1": [0.79, 0.77, 0.81], "fd": 23.7, "mae": [0.29, 0.28, 0.30]}
    assert ca.c0_reproduction_dev({"f1": [0.79, 0.77, 0.81], "fd": 23.7, "mae": [0.29, 0.28, 0.30]}, st) == 0.0
    assert ca.c0_reproduction_dev({"f1": [0.79, 0.77, 0.81], "fd": 23.7001, "mae": [0.29, 0.28, 0.30]}, st) > 1e-9
    assert ca.c0_reproduction_dev({"f1": [float("nan"), 0.77, 0.81], "fd": 23.7, "mae": [0.29, 0.28, 0.30]}, st) == float("inf")
    assert "STOP: frozen C0 metrics not reproduced" in inspect.getsource(ca.evaluate)


# ------------------------------------------------------------------------------------------ claim rules
def ok_m1():
    return {"fp": [-0.1, -0.2, -0.01], "fd": [-2.0, -3.0, -0.5], "corr": [0.0, -0.01, 0.01],
            "rhythm_f1": [-0.001, -0.002, 0.0], "rhythm_rr": [0.0, -0.1, 0.1]}


def test_claim_m1(ca):
    assert ca.claim_m1(ok_m1()) == "SUPPORTED"
    for k, v in (("fp", [-0.1, -0.2, 0.0]), ("fd", [0.5, -0.5, 1.0]), ("corr", [-0.01, -0.021, 0.0]),
                 ("rhythm_f1", [-0.01, -0.03, 0.0]), ("rhythm_rr", [1.0, 0.0, 2.5])):
        assert ca.claim_m1(ok_m1() | {k: v}) == "NOT SUPPORTED"


def test_claim_m2(ca):
    base = {"fd": [-2.0, -3.0, -0.5], "f1": [0.0, -0.005, 0.005], "corr": [0.0, -0.005, 0.005]}
    assert ca.claim_m2(base) == "SUPPORTED"
    assert ca.claim_m2(base | {"fd": [-0.2, -1.0, 0.3]}) == "NOT SUPPORTED"       # indistinguishable on FD
    assert ca.claim_m2(base | {"f1": [-0.01, -0.025, 0.0]}) == "NOT SUPPORTED"     # meaningful event disadvantage
    assert ca.claim_m2(base | {"corr": [-0.01, -0.03, 0.0]}) == "NOT SUPPORTED"


def test_claim_m3(ca):
    base = {"fp": [-0.1, -0.2, -0.01], "fd": [-2.0, -3.0, -0.5], "corr": [0.0, -0.01, 0.01]}
    assert ca.claim_m3(base) == "STRONG"
    assert ca.claim_m3(base | {"fd": [0.2, -0.5, 0.9]}) == "PARTIAL"               # FP better, FD not significantly worse
    assert ca.claim_m3(base | {"fp": [0.0, -0.05, 0.05]}) == "PARTIAL"             # FD better, FP not significantly worse
    assert ca.claim_m3(base | {"fd": [1.0, 0.5, 1.5]}) == "NOT SUPPORTED"          # FP better but FD significantly worse
    assert ca.claim_m3(base | {"fp": [0.05, 0.01, 0.1], "fd": [0.5, 0.1, 1.0]}) == "NOT SUPPORTED"
    assert ca.claim_m3(base | {"corr": [-0.02, -0.03, -0.01]}) == "NOT SUPPORTED"


def test_carrier_replication_matrix_and_c1(ca):
    assert ca.claim_carrier({"fd": [-5, -6, -4], "fp": [-0.1, -0.2, -0.05]}) == "SUPPORTED"
    assert ca.claim_carrier({"fd": [-5, -6, -4], "fp": [0.1, 0.05, 0.2]}) == "NOT SUPPORTED"
    v, h = {"fp": [-0.2, -0.3, -0.1], "fd": [-5, -6, -4]}, {"fp": [-0.1, -0.2, 0.01], "fd": [-4, -5, -3]}
    assert ca.replication("SUPPORTED", "SUPPORTED", v, h, ("fp", "fd")) == "REPLICATED"
    assert ca.replication("SUPPORTED", "NOT SUPPORTED", v, h, ("fp", "fd")) == "DIRECTIONALLY CONSISTENT"
    assert ca.replication("SUPPORTED", "NOT SUPPORTED", v, h | {"fd": [1, 0.5, 2]}, ("fp", "fd")) == "NOT REPLICATED"
    val = {"M1": "SUPPORTED", "M2": "NOT SUPPORTED", "M3": "PARTIAL", "CARRIER": "SUPPORTED"}
    rep = {"M1": "REPLICATED", "M2": "REPLICATED", "M3": "DIRECTIONALLY CONSISTENT", "CARRIER": "REPLICATED"}
    fm = ca.final_matrix(val, rep)
    assert fm == {"capacity_only_explanation": "DISFAVORED", "shared_absolute_context_carrier": "SUPPORTED",
                  "time_varying_global_field": "NOT SUPPORTED", "explicit_global_local_decomposition": "PARTIAL"}
    assert ca.c1_decision(val, rep).startswith("GO")
    assert ca.c1_decision(val | {"M1": "NOT SUPPORTED"}, rep) == "NO-GO"
    assert ca.c1_decision(val | {"M3": "NOT SUPPORTED"}, rep) == "NO-GO"
    assert ca.c1_decision(val, rep | {"M3": "NOT REPLICATED"}) == "NO-GO"
