"""SF0 unit tests (docs/SF0_SCALEFLOW_PREREGISTRATION.md §12): synthetic tensors, the V1 / C0 manifests, the committed
split artifacts and static checks only; no SF-VAL outcome is computed here."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.coherentbeat import ablation as AB
from ppg2ecg.coherentbeat import split as SP
from ppg2ecg.scaleflow import model as M

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/sf0_scaleflow"


@pytest.fixture(scope="module")
def sf():
    sys.path.insert(0, str(ROOT / "scripts"))
    import sf0_scaleflow
    return sf0_scaleflow


@pytest.fixture(scope="module")
def split():
    return json.loads((ART / "split_manifest.json").read_text())


# ------------------------------------------------------------------------------------------ 1-4 split
def test_split_counts_disjoint_and_clean(sf, split):
    tr, va = split["roles"]["sf_train"]["patients"], split["roles"]["sf_val"]["patients"]
    assert (len(tr), len(va)) == (3037, 433) and not set(tr) & set(va)
    c0 = json.loads((ROOT / "artifacts/c0_coherentbeat/split_manifest.json").read_text())["roles"]
    assert set(tr) | set(va) == set(int(p) for p in c0["train"]["patients"])
    assert not (set(tr) | set(va)) & (set(c0["val"]["patients"]) | set(c0["holdout"]["patients"]))
    m = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())
    old = SP.old_heldout_patients(m)
    assert not (set(tr) | set(va)) & (set(old["test"]) | set(old["val"]))
    assert sf.sf_split(c0["train"]["patients"]) == {"sf_train": tr, "sf_val": va}          # deterministic, never reshuffled


# ------------------------------------------------------------------------------------------ 5-6 detector / raster
def test_detector_trained_on_sf_train_and_one_raster_for_all(sf):
    src = inspect.getsource(sf.stage_train_detector)
    assert 'load_role("sf_train"' in src and "sf_val" not in src
    ev = inspect.getsource(sf.evaluate)
    assert ev.count("detector_events(X, dev, ex)") == 1 and "R = AB.event_raster(events)" in ev
    assert 'generate("wwl1", X, R, noise, dev)' in ev and 'for k, n in (("WWFM", "wwfm"), ("IND", "ind"), ("SF", "sf")):' in ev
    assert "waves[k] = generate(n, X, R, noise, dev)" in ev


# ------------------------------------------------------------------------------------------ 7-9 Haar
def test_haar_fixed_orthonormal_and_invertible():
    x = torch.randn(4, 512, dtype=torch.float64)
    c, m, f = M.haar(x)
    assert (c.shape[-1], m.shape[-1], f.shape[-1]) == (128, 128, 256)
    assert (M.ihaar(c, m, f) - x).abs().max() < 1e-6
    assert torch.allclose((c ** 2).sum() + (m ** 2).sum() + (f ** 2).sum(), (x ** 2).sum())   # orthonormal (Parseval)
    e = torch.zeros(512, dtype=torch.float64); e[0] = 1.0
    c, m, f = M.haar(e)
    assert c[0] == pytest.approx(0.5) and m[0] == pytest.approx(0.5) and f[0] == pytest.approx(1 / np.sqrt(2))
    assert not any(isinstance(v, torch.nn.Parameter) for v in vars(M).values())


# ------------------------------------------------------------------------------------------ 10-12 flow matching
def test_fm_path_target_and_time_range():
    x1, x0 = torch.randn(5, 512), torch.randn(5, 512)
    t = torch.rand(5)
    xt, u = M.fm_path(x1, x0, t)
    assert torch.allclose(xt, (1 - t[:, None]) * x0 + t[:, None] * x1) and torch.allclose(u, x1 - x0)
    g = torch.Generator().manual_seed(0)
    seen = []

    class Probe(torch.nn.Module):
        def forward(self, xt, t, ppg, raster):
            seen.append(t.clone())
            return torch.zeros_like(xt)
    M.fm_loss(Probe(), torch.randn(64, 512), torch.randn(64, 512), torch.zeros(64, 512), g)
    assert ((seen[0] >= 0) & (seen[0] <= 1)).all()


# ------------------------------------------------------------------------------------------ 13-17 architectures
def test_wwfm_has_no_multiresolution_path():
    src = inspect.getsource(M.WWFM)
    assert "haar" not in src.lower() and "Branch" not in src
    assert M.WWFM(63).dec_in.in_channels == 64 + 2


def test_independent_has_no_cross_scale_communication():
    torch.manual_seed(0)
    ind = M.ScaleFM(16, coupled=False).eval()
    assert not any(n.startswith("proj") for n, _ in ind.named_modules())
    assert ind.mid.stem.in_channels == 3 and ind.fine.stem.in_channels == 3
    x, ppg, r, t = torch.randn(2, 512), torch.randn(2, 512), torch.zeros(2, 512), torch.rand(2)
    _, (vc, vm, vf) = ind(x, t, ppg, r, return_parts=True)
    xc, xm, xf = M.haar(x)
    x2 = M.ihaar(xc + 1.0, xm, xf)                       # perturb only the coarse coefficients
    _, (vc2, vm2, vf2) = ind(x2, t, ppg, r, return_parts=True)
    close = lambda a, b: torch.allclose(a, b, atol=1e-5)  # noqa: E731  (float32 Haar round trip ~1e-7)
    assert not close(vc, vc2) and close(vm, vm2) and close(vf, vf2)


def test_coupled_topology_coarse_to_fine_only():
    torch.manual_seed(0)
    sfm = M.ScaleFM(16, coupled=True).eval()
    assert sfm.mid.stem.in_channels == 3 + M.PROJ and sfm.fine.stem.in_channels == 3 + 2 * M.PROJ
    x, ppg, r, t = torch.randn(2, 512), torch.randn(2, 512), torch.zeros(2, 512), torch.rand(2)
    _, (vc, vm, vf) = sfm(x, t, ppg, r, return_parts=True)
    xc, xm, xf = M.haar(x)
    _, (vc2, vm2, vf2) = sfm(M.ihaar(xc + 1.0, xm, xf), t, ppg, r, return_parts=True)
    close = lambda a, b: torch.allclose(a, b, atol=1e-5)  # noqa: E731
    assert not close(vm, vm2) and not close(vf, vf2)        # coarse -> mid, coarse -> fine
    _, (vc3, vm3, vf3) = sfm(M.ihaar(xc, xm + 1.0, xf), t, ppg, r, return_parts=True)
    assert close(vc, vc3) and not close(vf, vf3)             # mid -> fine, no mid -> coarse
    _, (vc4, vm4, vf4) = sfm(M.ihaar(xc, xm, xf + 1.0), t, ppg, r, return_parts=True)
    assert close(vc, vc4) and close(vm, vm4)                 # no fine -> coarse / mid


def test_inverse_transform_before_waveform_loss_and_same_objective(sf):
    src = inspect.getsource(M.ScaleFM.forward)
    assert "v = ihaar(vc, vm, vf)" in src
    tr = inspect.getsource(sf._train)
    assert tr.count("M.fm_loss(net, Yt[bd], Xt[bd], Rt[bd], gd)") == 1 and "abs().mean()" in tr
    fl = inspect.getsource(M.fm_loss)
    assert "((model(xt, t, ppg, raster) - u) ** 2).mean()" in fl


# ------------------------------------------------------------------------------------------ 19-21 exclusions
def test_no_auxiliary_loss_beat_renderer_or_guard(sf):
    msrc = inspect.getsource(M)
    for bad in ("guard", "assemble", "supports", "envelope", "spectral", "pcc", "peak_loss", "Attention", "LSTM", "GRU"):
        assert bad not in msrc
    tr = inspect.getsource(sf._train)
    assert [ln.strip() for ln in tr.splitlines() if ln.strip().startswith("loss =")] == [
        "loss = (net(Xt[bd], Rt[bd]) - Yt[bd]).abs().mean()", "loss = M.fm_loss(net, Yt[bd], Xt[bd], Rt[bd], gd)"]


# ------------------------------------------------------------------------------------------ 22-24 noise / sampling
def test_noise_map_deterministic_shared_and_euler8():
    a = M.window_noise([5, 5, 7], [0, 1, 0])
    b = M.window_noise([5, 5, 7], [0, 1, 0])
    assert np.array_equal(a, b) and not np.allclose(a[0], a[1]) and a.shape == (3, 512)
    assert not np.allclose(M.window_noise([5], [0], k=1), a[:1])
    calls = []

    class Count(torch.nn.Module):
        def forward(self, x, t, ppg, raster):
            calls.append(float(t[0]))
            return torch.ones_like(x)
    out = M.euler(Count(), torch.zeros(2, 512), torch.zeros(2, 512), torch.zeros(2, 512), 8)
    assert len(calls) == 8 and calls == [i / 8 for i in range(8)] and torch.allclose(out, torch.ones(2, 512))


def test_same_initial_noise_across_fm_arms(sf):
    ev = inspect.getsource(sf.evaluate)
    assert ev.count("noise = M.window_noise(Pid, wid)") == 1
    gen = inspect.getsource(sf.generate)
    assert "M.euler(net, torch.from_numpy(noise[i:i + bs])" in gen


# ------------------------------------------------------------------------------------------ 25-28 rules / seal / bootstrap / shuffle
def test_parameter_selection_by_count_only():
    src = inspect.getsource(M.param_match)
    assert "n_params" in src and "fd" not in src.lower() and "val" not in src.lower()
    pm = json.loads((ART / "parameter_match.json").read_text())
    assert abs(pm["SF"]["vs_600k"]) < 0.05 and abs(pm["IND"]["vs_SF"]) < 0.05 and abs(pm["WWFM"]["vs_SF"]) < 0.05
    assert pm == {**pm, **{k: v for k, v in M.param_match().items()}}


def test_test_loader_sealed(sf, tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "FREEZE", tmp_path / "final_test_freeze_manifest.json")
    with pytest.raises(PermissionError):
        sf.load_test()
    (tmp_path / "final_test_freeze_manifest.json").write_text(json.dumps({"qualified": False, "freshness_audit": "CLEAN", "sha256": {}}))
    with pytest.raises(PermissionError):
        sf.load_test()
    assert "load_test()" in inspect.getsource(sf.evaluate)


def test_bootstrap_clustered_with_sf0_seed(sf):
    assert sf.BOOT_SEED == 20261002 and sf.BOOT_N == 2000
    assert "B.patient_bootstrap_indices(pid, BOOT_N, BOOT_SEED)" in inspect.getsource(sf.fd_diff_ci)
    assert "C0.cluster_ci(v, pid, BOOT_N, BOOT_SEED)" in inspect.getsource(sf.ci)


def test_ppg_shuffle_preserves_other_inputs(sf):
    ev = inspect.getsource(sf.evaluate)
    assert 'generate("sf", X[perm], R, noise, dev)' in ev and 'generate("sf", X, R[perm], noise, dev)' in ev
    p = sf.shuffle_perm(100)
    assert np.array_equal(p, sf.shuffle_perm(100)) and sorted(p.tolist()) == list(range(100))


def test_gates_and_g1():
    good = {"fd_sf_wwfm": [-2, -3, -1], "fd_sf_best": [-2, -3, -1], "corr_sf_bestcorr": [0, -0.01, 0.01], "fp_sf_wwl1": [0.01, 0, 0.04],
            "recall_sf_wwl1": [0, -0.005, 0.005], "fd_sf_ind": [-1, -2, -0.5], "corr_sf_ind": [0, -0.01, 0.01], "fd_shuf_cond": [5, 4, 6],
            "corr_shuf_cond": [-0.1, -0.12, -0.08]}
    assert M.gates(good)["QUALIFIED"]
    assert not M.gates({**good, "fd_shuf_cond": [1, -0.5, 2]})["G7"]
    assert not M.gates({**good, "fp_sf_wwl1": [0.04, 0.03, 0.06]})["G5"]
    assert M.g1_label({"fd": [-2, -3, -1], "corr": [0, -0.01, 0.01], "fp": [0, -0.01, 0.01]}) == "FM BETTER"
    assert M.g1_label({"fd": [2, 1, 3], "corr": [-0.01, -0.02, 0.0], "fp": [0, -0.01, 0.01]}) == "NO BENEFIT"
    assert "coupling unsupported" in M.failure_categories({**M.gates(good), "G6": False}, "FM BETTER")
