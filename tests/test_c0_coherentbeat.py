"""C0 unit tests (docs/C0_COHERENTBEAT_PREREGISTRATION.md §9). Synthetic data, plus the V1 split manifest (metadata only)."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.coherentbeat import geometry as G
from ppg2ecg.coherentbeat import model as CM
from ppg2ecg.coherentbeat import split as SP

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/split_v1_vitaldb_seed42.json"


@pytest.fixture(scope="module")
def c0():
    sys.path.insert(0, str(ROOT / "scripts"))
    import c0_coherentbeat
    return c0_coherentbeat


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


# ------------------------------------------------------------------------------------------ split (tests 1-3)
def test_arch_split_counts_disjoint_deterministic_and_excludes_old_heldout(manifest):
    a, b = SP.arch_split(manifest), SP.arch_split(manifest)
    assert a == b                                                                   # never reshuffles
    n = {r: len(a[r]["patients"]) for r in a}
    assert n == {"train": 3470, "val": 433, "holdout": 434}
    sets = {r: set(a[r]["patients"]) for r in a}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["holdout"]) and not (sets["val"] & sets["holdout"])
    old = SP.old_heldout_patients(manifest)
    allp = set().union(*sets.values())
    assert not (allp & set(old["val"])) and not (allp & set(old["test"]))
    poc = manifest["extra"]["patient_of_case"]
    for r in a:                                                                     # whole patients, V1 train cases only
        assert all(int(poc[c]) in sets[r] for c in a[r]["cases"])
        assert set(a[r]["cases"]) <= set(manifest["splits"][0]["train"])
    assert sum(len(a[r]["cases"]) for r in a) == len(manifest["splits"][0]["train"])


def test_window_offsets_cover_the_v1_train_order(manifest):
    offs = SP.window_offsets(manifest)
    cases = manifest["splits"][0]["train"]
    assert offs[cases[0]][0] == 0
    assert offs[cases[-1]][1] == manifest["extra"]["n_windows"]["train"]


# ------------------------------------------------------------------------------------------ holdout guard (4, 22)
def test_holdout_is_sealed_without_a_freeze_manifest(c0, tmp_path, monkeypatch):
    monkeypatch.setattr(c0, "ART", tmp_path)
    monkeypatch.setattr(c0, "FREEZE", tmp_path / "holdout_freeze_manifest.json")
    with pytest.raises(PermissionError):
        c0.load_arch("holdout")
    with pytest.raises(PermissionError):
        c0.reference_peaks("holdout", np.zeros((1, 512)), None)
    with pytest.raises(PermissionError):
        c0.stage_evaluate_holdout(None, None)
    (tmp_path / "holdout_freeze_manifest.json").write_text(json.dumps({"sha256": {}}))
    (tmp_path / "qualification.json").write_text(json.dumps({"verdict": "FAILED"}))
    with pytest.raises(PermissionError):                                            # a failed ARCH-VAL never unseals
        c0.check_freeze()
    with pytest.raises(ValueError):
        c0.load_arch("test")


def test_stage_a_never_touches_holdout(c0):
    for fn in (c0.stage_audit, c0.stage_train_detector, c0.stage_train_bf0det, c0._train_c0, c0.stage_evaluate_val):
        src = inspect.getsource(fn)
        assert 'load_arch("holdout")' not in src and 'reference_peaks("holdout"' not in src     # metadata counts only
    assert 'load_arch("train")' in inspect.getsource(c0._train_c0)


# ------------------------------------------------------------------------------------------ geometry (5-13)
def test_spline_basis_shape_and_determinism():
    Bm = G.spline_basis()
    assert Bm.shape == (512, 17)
    np.testing.assert_array_equal(Bm, G.spline_basis())
    assert G.cubic_bspline(0.0) == pytest.approx(2 / 3) and G.cubic_bspline(2.0) == 0.0
    np.testing.assert_allclose(Bm[32:480].sum(axis=1), 1.0, atol=1e-12)          # partition of unity away from the ends


def test_bump_centre_boundary_and_flat_derivatives():
    assert G.bump(0.0) == 1.0 and G.bump(1.0) == 0.0 and G.bump(1.5) == 0.0
    u = np.array([0.9, 0.98, 0.99, 0.999])
    vals = G.bump(u)
    assert np.all(np.diff(vals) < 0) and vals[1] < 1e-10 and vals[-1] < 1e-200                  # decays to 0
    h = 1e-4
    for uu in (1.0 - 1e-3,):                                                        # slope -> 0 at the boundary
        assert abs((G.bump(uu + h) - G.bump(uu - h)) / (2 * h)) < 1e-12
    assert abs((G.bump(h) - G.bump(0.0)) / h) < 1e-3                                # zero slope at the centre (b ~ 1 - u^2)
    t = torch.tensor([0.0, 0.5, 0.999, 1.0, 2.0])
    np.testing.assert_allclose(G.bump(t).numpy(), G.bump(t.numpy()), rtol=1e-5, atol=1e-30)


def test_supports_use_physical_time_and_045_rr_clipping():
    L, R = G.supports([100, 160, 400], rr_edge=100.0)
    np.testing.assert_allclose(L, [min(38.4, 45.0), min(38.4, 0.45 * 60), min(38.4, 0.45 * 240)])
    np.testing.assert_allclose(R, [min(57.6, 0.45 * 60), min(57.6, 0.45 * 240), min(57.6, 45.0)])
    Ls, Rs = G.supports([250], rr_edge=90.0)                                         # edges: TRAIN median RR only
    assert (Ls[0], Rs[0]) == (min(38.4, 40.5), min(57.6, 40.5))
    assert G.L_MAX == pytest.approx(38.4) and G.R_MAX == pytest.approx(57.6)


def test_edge_rr_is_the_arch_train_median(c0):
    assert '["arch_train"]["median_reference_rr_samples"]' in inspect.getsource(c0.train_rr_median)
    for fn in (c0.evaluate, c0._train_c0):
        assert "train_rr_median()" in inspect.getsource(fn)


# ------------------------------------------------------------------------------------------ model (14-18)
def unit_q(net):
    """Make q == 1 everywhere so the local part equals the envelope (exposes any renormalization or warping)."""
    with torch.no_grad():
        net.local_head.weight.zero_()
        net.local_head.bias.fill_(1.0)
    return net


def test_local_output_is_the_envelope_in_absolute_time_and_zero_outside_support():
    torch.manual_seed(0)
    net = unit_q(CM.CoherentBeat(use_global=True).eval())
    ev = [np.array([100, 200, 400]), np.array([256])]
    with torch.no_grad():
        y, g, loc = net(torch.randn(2, 512), ev, rr_edge=100.0, return_parts=True)
    for b, e in enumerate(ev):
        L, R = G.supports(e, 100.0)
        expect = np.zeros(512)
        for r, l_, r_ in zip(e, L, R):
            tau = np.arange(G.TAU_LO, G.TAU_HI + 1)
            t = r + tau
            ok = (t >= 0) & (t < 512)
            expect[t[ok]] += G.envelope(tau, l_, r_)[ok]
        np.testing.assert_allclose(loc[b].numpy(), expect, atol=1e-6)               # no renormalization, no warping
        outside = expect == 0
        assert np.all(loc[b].numpy()[outside] == 0.0)                               # exactly zero outside support
        np.testing.assert_array_equal(y[b].numpy()[outside], g[b].numpy()[outside])  # the waveform is g there
    assert y.shape == (2, 512) and np.all(np.isfinite(y.numpy()))


def test_waveform_exists_everywhere_and_global_field_is_deterministic():
    torch.manual_seed(1)
    net = CM.CoherentBeat(use_global=True).eval()
    x = torch.randn(1, 512)
    with torch.no_grad():
        y1, g1, _ = net(x, [np.zeros(0, int)], 100.0, return_parts=True)
        y2, g2, _ = net(x, [np.zeros(0, int)], 100.0, return_parts=True)
    torch.testing.assert_close(y1, g1)                                              # no events: x_hat = g
    torch.testing.assert_close(g1, g2)
    assert torch.any(g1 != 0)
    lo = CM.CoherentBeat(use_global=False).eval()
    with torch.no_grad():
        assert torch.all(lo(x, [np.zeros(0, int)], 100.0) == 0)                     # LOCAL-ONLY: g = 0


def test_batch_size_invariance_and_finite_gradients():
    torch.manual_seed(2)
    net = CM.CoherentBeat(use_global=True)
    x = torch.randn(3, 512)
    ev = [np.array([60, 170, 290, 410]), np.array([128, 384]), np.array([], int)]
    net.eval()
    with torch.no_grad():
        full = net(x, ev, 100.0)
        single = torch.cat([net(x[i:i + 1], [ev[i]], 100.0) for i in range(3)])
    torch.testing.assert_close(full, single, atol=1e-5, rtol=1e-5)
    net.train()
    loss = (net(x, ev, 100.0) - torch.randn(3, 512)).abs().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in net.parameters() if p.requires_grad)


def test_parameter_budget():
    from ppg2ecg.beatfirst.model import BeatFlowNet
    c, b = CM.n_params(CM.CoherentBeat(use_global=True)), CM.n_params(BeatFlowNet())
    assert c <= 1.5 * b


# ------------------------------------------------------------------------------------------ evaluation wiring (19-21)
def test_all_waveform_arms_share_one_event_sequence_from_the_retrained_detector(c0):
    src = inspect.getsource(c0.evaluate)
    assert "events = detect_events(X, dev, ex)" in src
    for call in ('render_bf0det(X, events, dev)', 'render_c0("c0_localonly", X, events, dev, rr_edge)',
                 'render_c0("c0", X, events, dev, rr_edge)'):
        assert call in src
    assert 'load_net("detector", dev)' in inspect.getsource(c0.detect_events)
    assert "OUT / f\"{name}.pt\"" in inspect.getsource(c0.load_net)                 # C0's own checkpoints only


def test_patient_clustered_bootstrap_equal_weight(c0):
    pid = np.array([0] * 9 + [1])
    d = np.array([1.0] * 9 + [0.0])
    pt, lo, hi = c0.cluster_ci(d, pid, n_rep=200)
    assert pt == pytest.approx(0.5) and 0.0 <= lo <= hi <= 1.0
    assert c0.BOOT_N == 2000 and c0.BOOT_SEED == 20261001


def test_gates_and_verdicts(c0):
    ok = {"dF1": [0.0, -0.019, 0.01], "dRR": [0.1, -0.2, 1.9], "dFP": [-0.1, -0.2, -0.01], "dFD": [-1.0, -2.0, -0.1],
          "dCorr": [0.0, -0.019, 0.01]}
    g = c0.gates(ok)
    assert all(g.values()) and c0.verdict(g, "val") == "QUALIFIED" and c0.verdict(g, "holdout") == "STRONG"
    bad4 = c0.gates(ok | {"dCorr": [-0.03, -0.04, -0.02]})
    assert c0.verdict(bad4, "val") == "PARTIAL" and c0.verdict(bad4, "holdout") == "PARTIAL"
    for k, v in (("dF1", [-0.01, -0.021, 0.0]), ("dRR", [1.0, 0.0, 2.0]), ("dFP", [-0.1, -0.2, 0.0]), ("dFD", [0.1, -0.1, 0.2])):
        assert c0.verdict(c0.gates(ok | {k: v}), "val") == "FAILED"


def test_boundary_points_and_fp_matching(c0):
    b = c0.boundary_points(np.array([100, 200]), 100.0)
    np.testing.assert_allclose(sorted(b), [100 - 38.4, 145.0, 200 - 38.4, 245.0])
    assert c0.fp_detections([100, 300], [106, 300, 450]) == [450]
    assert c0.BOUNDARY_NEAR == 2
