"""AF0 unit tests (docs/AF0_ANCHORFLOW_DEVELOPMENT_PROTOCOL.md §11): synthetic tensors, the committed split artifacts,
the V1 / C0 / SF0 manifests and static checks only; no AF-DEV or AF-LOCK outcome is computed here."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.anchorflow import fastfd as FF
from ppg2ecg.anchorflow import model as A
from ppg2ecg.coherentbeat import split as SP
from ppg2ecg.evaluation import paper_metrics as PMX
from ppg2ecg.scaleflow import model as SM

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/af0_anchorflow"


@pytest.fixture(scope="module")
def af():
    sys.path.insert(0, str(ROOT / "scripts"))
    import af0_anchorflow
    return af0_anchorflow


@pytest.fixture(scope="module")
def split():
    return json.loads((ART / "split_manifest.json").read_text())["roles"]


# ------------------------------------------------------------------------------------------ 1-3 split
def test_split_counts_disjoint_and_exclusions(af, split):
    sets = {r: set(split[r]["patients"]) for r in ("af_train", "af_dev", "af_lock")}
    assert {r: len(s) for r, s in sets.items()} == {"af_train": 2400, "af_dev": 300, "af_lock": 337}
    assert not (sets["af_train"] & sets["af_dev"] or sets["af_train"] & sets["af_lock"] or sets["af_dev"] & sets["af_lock"])
    sf = json.loads((ROOT / "artifacts/sf0_scaleflow/split_manifest.json").read_text())["roles"]
    allp = set().union(*sets.values())
    assert allp == set(sf["sf_train"]["patients"]) and not allp & set(sf["sf_val"]["patients"])
    c0 = json.loads((ROOT / "artifacts/c0_coherentbeat/split_manifest.json").read_text())["roles"]
    assert not allp & (set(c0["val"]["patients"]) | set(c0["holdout"]["patients"]))
    old = SP.old_heldout_patients(json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text()))
    assert not allp & (set(old["val"]) | set(old["test"]))
    assert af.af_split(sf["sf_train"]["patients"]) == {r: sorted(s) for r, s in sets.items()}


# ------------------------------------------------------------------------------------------ 4, 30, 32 seals
def test_lock_sealed_and_no_old_test(af, tmp_path, monkeypatch):
    monkeypatch.setattr(af, "LOCK_FREEZE", tmp_path / "lock_freeze_manifest.json")
    with pytest.raises(PermissionError):
        af.load_role("af_lock", None)
    (tmp_path / "lock_freeze_manifest.json").write_text(json.dumps({"winner": None, "sha256": {}}))
    with pytest.raises(PermissionError):
        af.check_lock_freeze()
    src = inspect.getsource(af)
    assert 'splits"][0]["test"]' not in src and "load_test" not in src and 'load_arch("val")' not in src and 'load_arch("holdout")' not in src
    assert src.count('old["test"]') == 1                                             # metadata-only exclusion check in the split
    assert "check_lock_freeze()" in inspect.getsource(af.stage_eval_lock) and "lock_context" in inspect.getsource(af.stage_eval_cand)


# ------------------------------------------------------------------------------------------ 5-9 frozen detector / anchor / residual
def test_detector_and_anchor_trained_on_af_train_only(af):
    for f in (af.stage_train_detector, af.stage_train_anchor, af.stage_train):
        src = inspect.getsource(f)
        assert 'load_role("af_train"' in src and "af_dev" not in src and "af_lock" not in src


def test_anchor_frozen_and_residual_definition(af):
    tr = inspect.getsource(af.stage_train)
    assert "y1 = norm.encode(Yt[bd] - Mt[bd])" in tr and "anchor" not in tr.split("def stage_train")[1].split("opt = ")[0].replace("Mt", "")
    pr = inspect.getsource(af.stage_prep)
    assert "B.sha256_file(OUT / \"anchor.pt\") == h" in pr and 'if role == "af_train":' in pr and "r = (Y - mu" in pr
    assert "load_ckpt(\"anchor\", dev)" in inspect.getsource(af.anchor_out) and ".eval()" in inspect.getsource(af.load_ckpt)


def test_residual_stats_train_only_and_normalizer_roundtrip():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(300, 512)) * np.linspace(0.1, 1.0, 512)
    for kind in ("A0", "A1", "A2", "waveform"):
        n = A.Normalizer.fit(kind, R)
        r = torch.from_numpy(R[:7])
        assert torch.allclose(n.decode(n.encode(r)), r, atol=1e-10)
        assert A.Normalizer.from_json(n.to_json()).to_json() == n.to_json()
    n0 = A.Normalizer.fit("A0", R)
    Z = A.to_bands(torch.from_numpy(R)).numpy()
    assert n0.loc[0] == pytest.approx(Z[:, :128].mean()) and n0.scale[2] == pytest.approx(Z[:, 256:].std())


# ------------------------------------------------------------------------------------------ 10-13 transform / flow / output
def test_haar_exact_flow_target_and_output_composition(af):
    x = torch.randn(3, 512, dtype=torch.float64)
    assert (A.from_bands(A.to_bands(x)) - x).abs().max() < 1e-6
    y1, y0, t = torch.randn(4, 512), torch.randn(4, 512), torch.rand(4)
    yt, u = SM.fm_path(y1, y0, t)
    assert torch.allclose(u, y1 - y0) and torch.allclose(yt, (1 - t[:, None]) * y0 + t[:, None] * y1)
    ev = inspect.getsource(af.stage_eval_cand)
    assert "xs = mu + gen_residual(" in ev and "Ks = np.stack([mu[S] + gen_residual(" in ev
    assert "norm.decode(y.double())" in inspect.getsource(af.gen_residual)


# ------------------------------------------------------------------------------------------ 14-17 architectures
def test_vanilla_has_no_haar_and_independent_has_no_coupling():
    assert "haar" not in inspect.getsource(A.VanillaResidualFM).lower()
    ind = A.ResidualScaleFM(16, None, "B0").eval()
    assert not any(n.startswith("proj") for n, _ in ind.named_modules())
    y, x, r, mu, t = torch.randn(2, 512), torch.randn(2, 512), torch.zeros(2, 512), torch.randn(2, 512), torch.rand(2)
    v = ind(y, t, x, r, mu)
    y2 = y.clone(); y2[:, :128] += 1.0                                              # perturb the coarse band only
    v2 = ind(y2, t, x, r, mu)
    assert not torch.allclose(v[:, :128], v2[:, :128]) and torch.allclose(v[:, 128:], v2[:, 128:], atol=1e-6)


@pytest.mark.parametrize("coupling", ["C0", "C1", "C2"])
def test_coupled_topology(coupling):
    torch.manual_seed(0)
    m = A.ResidualScaleFM(16, coupling, "B0").eval()
    y, x, r, mu, t = torch.randn(2, 512), torch.randn(2, 512), torch.zeros(2, 512), torch.randn(2, 512), torch.rand(2)
    v = m(y, t, x, r, mu)
    yc = y.clone(); yc[:, :128] += 1.0
    vc = m(yc, t, x, r, mu)
    assert not torch.allclose(v[:, 128:256], vc[:, 128:256], atol=1e-6) and not torch.allclose(v[:, 256:], vc[:, 256:], atol=1e-6)
    yf = y.clone(); yf[:, 256:] += 1.0
    vf = m(yf, t, x, r, mu)
    assert torch.allclose(v[:, :256], vf[:, :256], atol=1e-6)                      # no fine -> coarse / mid path


@pytest.mark.parametrize("anchor", ["B0", "B1", "B2"])
def test_condition_includes_anchor(anchor):
    torch.manual_seed(0)
    m = A.ResidualScaleFM(16, "C1", anchor).eval()
    y, x, r, t = torch.randn(2, 512), torch.randn(2, 512), torch.zeros(2, 512), torch.rand(2)
    assert not torch.allclose(m(y, t, x, r, torch.zeros(2, 512)), m(y, t, x, r, torch.ones(2, 512)), atol=1e-6)
    v = A.VanillaResidualFM(16).eval()
    assert not torch.allclose(v(y, t, x, r, torch.zeros(2, 512)), v(y, t, x, r, torch.ones(2, 512)), atol=1e-6)


# ------------------------------------------------------------------------------------------ 18-19 exclusions
def test_no_prohibited_loss_or_block(af):
    msrc = inspect.getsource(A) + inspect.getsource(af.stage_train)
    for bad in ("Attention", "Transformer", "LSTM", "GRU", "Mamba", "discriminator", "spectral_loss", "peak_loss", "kanflow_fd(", "pcc"):
        assert bad not in msrc
    assert [ln.strip() for ln in inspect.getsource(af.stage_train).splitlines() if ln.strip().startswith("loss =")] == [
        "loss = SM.fm_loss(net, Yt[bd], Xt[bd], Rt[bd], gd)", "loss = A.fm_loss(net, y1, Xt[bd], Rt[bd], Mt[bd], gd, cfg.get(\"ppg_dropout\", 0.0))"]
    assert "((model(yt, t, ppg, raster, mu) - u) ** 2).mean()" in inspect.getsource(A.fm_loss)


# ------------------------------------------------------------------------------------------ 20-23 seeds / K16 / residual FD
def test_seed_mapping_k16_and_mean(af):
    a = SM.window_noise([3, 3], [10, 11]); b = SM.window_noise([3, 3], [10, 11])
    assert np.array_equal(a, b) and not np.allclose(SM.window_noise([3], [10], k=5), a[:1])
    assert np.array_equal(af.k16_subset(5000), af.k16_subset(5000)) and af.k16_subset(5000).size == 2000
    gb = inspect.getsource(af.generative_block)
    assert "xbar, xmed = Ks.mean(axis=0), np.median(Ks, axis=0)" in gb
    assert '"residual_fd": float(PMX.kanflow_fd(r_gen, r_real))' in gb and "r_gen, r_real = x_single - mu, Y - mu" in gb


# ------------------------------------------------------------------------------------------ 24-26 controls
def test_condition_controls(af):
    ev = inspect.getsource(af.stage_eval_cand)
    assert "x_s1 = mu + gen_residual(cid, X[perm], R, mu32, ctx[\"noise\"], dev, nfe)" in ev
    assert "mu_s2 = anchor_out(X[perm], R, dev)" in ev and "x_s2 = mu_s2 + gen_residual(cid, X[perm], R, mu_s2" in ev
    assert "x_as = mu + gen_residual(cid, X, R, mu32[perm], ctx[\"noise\"], dev, nfe)" in ev


# ------------------------------------------------------------------------------------------ 27-29 budget / AF-DEV only
def test_budget_and_dev_only(af, tmp_path, monkeypatch):
    assert af.MAX_CANDIDATES == 10 and af.MAX_JOBS == 30
    monkeypatch.setattr(af, "ART", tmp_path)
    for i in range(30):
        af.register_job(f"j{i}")
    with pytest.raises(SystemExit):
        af.register_job("j30")
    for i in range(1, 11):
        af.new_candidate(f"c{i:02d}", "", "x", "y")
    with pytest.raises(SystemExit):
        af.new_candidate("c11", "", "x", "y")
    assert 'load_role("af_dev"' in inspect.getsource(af.dev_context) and "af_lock" not in inspect.getsource(af.dev_context)


# ------------------------------------------------------------------------------------------ 31 clustered bootstrap / fast FD exactness
def test_fast_fd_bootstrap_matches_official():
    sys.path.insert(0, str(ROOT / "scripts"))
    import bf0_run as B
    rng = np.random.default_rng(1)
    pid = np.repeat(np.arange(60), 55)
    ref = rng.normal(size=(pid.size, 64)).cumsum(axis=1) * 0.1
    a = ref + rng.normal(scale=0.3, size=ref.shape)
    res = B.patient_resamples(60, 3, 20261002)
    subs = np.unique(pid)
    FF._G["arms"], FF._G["ref"] = {"a": FF.patient_stats(a, pid, subs)}, FF.patient_stats(ref, pid, subs)
    rows = [np.flatnonzero(pid == s) for s in subs]
    for r in res:
        idx = np.concatenate([rows[k] for k in r])
        fast = FF._draw(np.bincount(r, minlength=60).astype(float))["a"]
        assert fast == pytest.approx(PMX.kanflow_fd(a[idx], ref[idx]), rel=1e-7)


def test_gates_rule(af):
    good = {"fd_vs_anchor": [-5, -6, -4], "fd_vs_fullsf": [0.2, -0.3, 0.8], "corr16_vs_anchor": [0, -0.01, 0.01], "fp16_vs_anchor": [0.01, 0, 0.03],
            "recall16_vs_anchor": [0, -0.005, 0.005], "rfd_vs_b2": [-1, -2, -0.5], "rfd_vs_b3": [-0.5, -0.8, -0.1], "rfd_s1_minus_cond": [1, 0.5, 2],
            "rfd_anchorshuf_minus_cond": [5, 4, 6], "diversity_ratio": 0.9}
    assert af.gates(good)["overall"]
    assert not af.gates({**good, "fd_vs_fullsf": [0.5, 0.0, 1.2]})["D1"]
    assert not af.gates({**good, "diversity_ratio": 1.6})["D7"]
    assert not af.gates({**good, "rfd_s1_minus_cond": [0.2, -0.1, 0.5]})["D6"]
