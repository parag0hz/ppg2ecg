"""DP0 unit tests (docs/DP0_DUALREADOUT_PREREGISTRATION.md §14): synthetic tensors, the committed split / design artifacts,
the AF0 / SF0 / C0 / V1 manifests and static checks only; no DP-DEV or AF-LOCK outcome is computed here."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.coherentbeat import split as SP
from ppg2ecg.dualreadout import model as D
from ppg2ecg.scaleflow import model as SM

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/dp0_dualreadout"


@pytest.fixture(scope="module")
def dp():
    sys.path.insert(0, str(ROOT / "scripts"))
    import dp0_dualreadout
    return dp0_dualreadout


@pytest.fixture(scope="module")
def split():
    return json.loads((ART / "split_manifest.json").read_text())["roles"]


@pytest.fixture(scope="module")
def af0():
    return json.loads((ROOT / "artifacts/af0_anchorflow/split_manifest.json").read_text())["roles"]


def tiny(share="S0"):
    torch.manual_seed(0)
    return D.DualReadout(share, 16, 16)


# ------------------------------------------------------------------------------------------ 1-6 split
def test_split_counts_and_disjoint(dp, split, af0):
    tr, dv = set(split["dp_train"]["patients"]), set(split["dp_dev"]["patients"])
    assert len(tr) == 2100 and len(dv) == 300 and not tr & dv
    assert tr | dv == set(af0["af_train"]["patients"])
    assert dp.dp_split(af0["af_train"]["patients"]) == {"dp_train": sorted(tr), "dp_dev": sorted(dv)}


def test_split_exclusions(split, af0):
    allp = set(split["dp_train"]["patients"]) | set(split["dp_dev"]["patients"])
    assert not allp & set(af0["af_dev"]["patients"])
    assert not allp & set(af0["af_lock"]["patients"]) and len(af0["af_lock"]["patients"]) == 337
    c0 = json.loads((ROOT / "artifacts/c0_coherentbeat/split_manifest.json").read_text())["roles"]
    assert not allp & (set(c0["val"]["patients"]) | set(c0["holdout"]["patients"]))
    sf = json.loads((ROOT / "artifacts/sf0_scaleflow/split_manifest.json").read_text())["roles"]
    assert not allp & set(sf["sf_val"]["patients"])
    old = SP.old_heldout_patients(json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text()))
    assert not allp & (set(old["val"]) | set(old["test"]))


# ------------------------------------------------------------------------------------------ 7, 8, 34, 35 seals
def test_af_lock_sealed_before_freeze(dp, tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "LOCK_FREEZE", tmp_path / "lock_freeze_manifest.json")
    with pytest.raises(PermissionError):
        dp.load_role("af_lock", None)
    (tmp_path / "lock_freeze_manifest.json").write_text(json.dumps({"winner": None, "sha256": {}}))
    with pytest.raises(PermissionError):
        dp.check_lock_freeze()
    f = ART / "_pytest_uncommitted_lock_freeze_manifest.json"
    monkeypatch.setattr(dp, "LOCK_FREEZE", f)
    try:
        f.write_text(json.dumps({"winner": "S0", "sha256": {}}))
        with pytest.raises(PermissionError):                                       # a winner but an uncommitted manifest
            dp.check_lock_freeze()
    finally:
        f.unlink(missing_ok=True)


def test_lock_evaluation_requires_freeze(dp):
    assert inspect.getsource(dp.stage_eval_lock).splitlines()[1].strip() == "check_lock_freeze()"
    assert 'context("af_lock"' in inspect.getsource(dp.stage_eval_lock)
    assert "check_lock_freeze()" in inspect.getsource(dp.load_role)
    fr = inspect.getsource(dp.stage_freeze)
    assert 'sel["winner"] != share' in fr and "outputs/dp0_dualreadout/detector.pt" in fr


def test_old_test_closed(dp):
    with pytest.raises(PermissionError):
        dp.load_test()
    src = inspect.getsource(dp)
    for bad in ('splits"][0]["test"]', 'load_arch("val")', 'load_arch("holdout")', 'load_role("af_dev"'):
        assert bad not in src
    assert src.count('old["test"]') == 1 and src.count('af["af_dev"]') == 1          # metadata-only exclusion checks in the split


def test_dp_dev_outcomes_need_committed_prereg(dp):
    assert "check_prereg_committed()" in inspect.getsource(dp.stage_eval_dev)
    src = inspect.getsource(dp.check_prereg_committed)
    assert "ls-files" in src and "diff" in src and "PREREG" in src


# ------------------------------------------------------------------------------------------ 9-11 training sources
def test_training_uses_dp_train_only(dp):
    for f in (dp.stage_train_detector, dp._train_data):
        src = inspect.getsource(f)
        assert 'load_role("dp_train"' in src and "dp_dev" not in src and "af_lock" not in src
    for f in (dp.stage_train_point, dp.stage_train_gen, dp.stage_train_dual):
        src = inspect.getsource(f)
        assert "_train_data(ex, dev)" in src and "dp_dev" not in src and "af_lock" not in src
    assert 'load_role("dp_train"' not in inspect.getsource(dp.stage_eval_dev)


def test_specialist_point_is_point_only_l1(dp):
    net = dp.build("point")
    assert type(net).__name__ == "WWDet" and D.n_params(net) == 593_577
    src = inspect.getsource(dp.stage_train_point)
    assert "(net(Xt[b], Rt[b]) - Yt[b]).abs().mean()" in src and "fm_loss" not in src


def test_specialist_gen_is_exact_scaleflow(dp):
    net = dp.build("gen")
    assert isinstance(net, SM.ScaleFM) and net.coupled and D.n_params(net) == 598_333
    src = inspect.getsource(dp.stage_train_gen)
    assert "SM.fm_loss(net, Yt[b], Xt[b], Rt[b], gd)" in src
    x1 = torch.randn(4, 512)
    xt, u = SM.fm_path(x1, torch.zeros(4, 512), torch.full((4,), 0.25))
    assert torch.allclose(xt, 0.25 * x1) and torch.allclose(u, x1)
    assert "SM.euler(net, x0, x, r, nfe) if name == \"gen\"" in inspect.getsource(dp.gen_out) and dp.NFE == 8


def test_identical_event_raster_and_shuffle(dp):
    ev = inspect.getsource(dp.evaluate)
    assert "point_out(a, X, R, dev)" in ev and 'gen_out(a, X, R, ctx["noise"], dev)' in ev
    assert 'gen_out(a, X[perm], R, ctx["noise"], dev)' in ev                       # PPG shuffled; raster, noise, model kept
    cx = inspect.getsource(dp.context)
    assert cx.count("detector_events(") == 1 and '"R": AB.event_raster(ev)' in cx
    assert np.array_equal(dp.shuffle_perm(50), np.random.default_rng(20261002).permutation(50))


# ------------------------------------------------------------------------------------------ 13-18 dual structure
def test_shared_encoder_gets_condition_only_and_xt_private():
    net = tiny()
    assert net.stem.in_channels == 2 and list(inspect.signature(net.trunk).parameters) == ["ppg", "raster"]
    seen = []
    net.stem.register_forward_hook(lambda m, i, o: seen.append(i[0].detach().clone()))
    ppg, ras, xt = torch.randn(2, 512), torch.rand(2, 512), torch.randn(2, 512)
    net.flow(xt, torch.rand(2), ppg, ras)
    assert len(seen) == 1 and torch.equal(seen[0], torch.stack([ppg, ras], dim=1))
    c1 = net.encode_flow(ppg, ras)
    assert all(torch.equal(a, b) for a, b in zip(c1, net.encode_flow(ppg, ras)))


def test_point_head_has_no_noise_or_time():
    net = tiny()
    assert list(inspect.signature(net.point).parameters) == ["ppg", "raster"]
    assert not any("temb" in n or ".tp." in n for n, _ in net.point_dec.named_parameters())
    ppg, ras = torch.randn(2, 512), torch.rand(2, 512)
    assert torch.equal(net.point(ppg, ras), net.point(ppg, ras))
    mu, cond = net.both(ppg, ras)
    assert torch.allclose(mu, net.point(ppg, ras)) and cond[0].shape == (2, 64, 256) and cond[1].shape == (2, 64, 128)


def test_flow_head_uses_xt_and_time():
    net = tiny()
    for p in net.flow_dec.parameters():
        torch.nn.init.normal_(p, std=0.2)
    ppg, ras = torch.randn(2, 512), torch.rand(2, 512)
    cond = net.encode_flow(ppg, ras)
    xt, t = torch.randn(2, 512), torch.full((2,), 0.3)
    v = net.velocity(xt, t, cond)
    assert not torch.allclose(v, net.velocity(xt + 1.0, t, cond)) and not torch.allclose(v, net.velocity(xt, t + 0.4, cond))


def test_outputs_never_added_and_no_anchor_path(dp):
    msrc = inspect.getsource(D)
    assert "anchor" not in msrc.lower().replace("no_anchor_residual_path", "")
    assert "point" not in inspect.getsource(D.euler).replace("Point", "") and "point" not in inspect.getsource(D.FlowDecoder)
    gsrc = inspect.getsource(dp.gen_out)
    assert "point" not in gsrc and "D.euler(net, x0, x, r, nfe)" in gsrc
    ssrc = inspect.getsource(dp)
    assert "anchorflow import model" not in ssrc and "Normalizer" not in ssrc and "anchor_out" not in ssrc
    assert json.loads((ART / "sharing_graph.json").read_text())["readouts"]["never_added"] is True


# ------------------------------------------------------------------------------------------ 19-22 ownership / adapters
@pytest.mark.parametrize("share,n_shared,shared_params", [("S0", 8, 328_896), ("S1", 6, 246_720), ("S2", 0, 192)])
def test_ownership_graph(share, n_shared, shared_params):
    w = D.choose_widths(593_577, 598_333)
    net = D.DualReadout(share, w["w_point"], w["w_flow"])
    assert len(net.shared_blocks) == n_shared and len(net.point_tower) == len(net.flow_tower) == 8 - n_shared
    assert [b.c1.dilation[0] for b in net.shared_blocks] == list(D.ENC_DIL[:n_shared])
    assert [b.c1.dilation[0] for b in net.point_tower] == [b.c1.dilation[0] for b in net.flow_tower] == list(D.ENC_DIL[n_shared:])
    own = net.ownership()
    for n, o in own.items():
        top = n.split(".")[0]
        assert o == {"stem": "shared", "shared_blocks": "shared"}.get(top, "point" if top.startswith("point_") else "flow")
    a = net.accounting()
    assert a["shared"] == shared_params and a["point_path"] == w["point_path"] and a["flow_path"] == w["flow_path"]
    assert a["total"] == D.n_params(net) == a["shared"] + a["point_private"] + a["flow_private"]
    assert set(D.SHARED_STAGES[share]) == {"S0": {"E0", "E1", "E2"}, "S1": {"E0", "E1"}, "S2": {"E0"}}[share]


def test_widths_frozen_by_parameter_count():
    w = D.choose_widths(593_577, 598_333)
    assert (w["w_point"], w["w_flow"]) == (71, 30) and (w["point_path"], w["flow_path"]) == (590_503, 599_589)
    acc = json.loads((ART / "parameter_accounting.json").read_text())
    assert {s: acc[s]["total"] for s in ("S0", "S1", "S2")} == {"S0": 861_196, "S1": 943_372, "S2": 1_189_900}


def test_adapters_small_and_identical():
    for s in ("S0", "S1", "S2"):
        net = tiny(s)
        assert isinstance(net.point_adapter.conv, torch.nn.Conv1d) and net.point_adapter.conv.kernel_size == (1,)
        assert net.point_adapter.conv.in_channels == net.point_adapter.conv.out_channels == 64
        assert isinstance(net.point_adapter.act, torch.nn.GELU) and isinstance(net.flow_adapter.act, torch.nn.GELU)
        w = D.choose_widths(593_577, 598_333)
        a = D.DualReadout(s, w["w_point"], w["w_flow"]).accounting()
        assert a["adapters"] == 8_320 and a["adapter_fraction"] < 0.05
    assert not any("attn" in n.lower() or "attention" in n.lower() for n, _ in tiny().named_modules())


# ------------------------------------------------------------------------------------------ 23-28 round-robin training
def _losses(net):
    g = torch.Generator().manual_seed(1)
    ppg, ras, y = torch.randn(4, 512, generator=g), torch.rand(4, 512, generator=g), torch.randn(4, 512, generator=g)
    gd = torch.Generator().manual_seed(2)
    return (lambda: (net.point(ppg, ras) - y).abs().mean()), (lambda: D.fm_loss(net, y, ppg, ras, gd))


def _snap(params):
    return [p.detach().clone() for p in params]


@pytest.mark.parametrize("task", ["point", "flow"])
def test_substep_masks(task):
    net = tiny("S1")
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    groups = net.groups()
    pl, fl = _losses(net)
    before = {k: _snap(v) for k, v in groups.items()}
    D.substep(net, opt, groups, task, pl if task == "point" else fl, 1.0)
    other = "flow" if task == "point" else "point"
    assert all(torch.equal(a, b) for a, b in zip(before[other], groups[other]))     # other task's private params untouched (incl. decay)
    assert any(not torch.equal(a, b) for a, b in zip(before["shared"], groups["shared"]))
    assert any(not torch.equal(a, b) for a, b in zip(before[task], groups[task]))
    assert all(p not in opt.state or opt.state[p] == {} for p in groups[other])


def test_round_robin_counts_order_and_shared_updates():
    net = tiny("S0")
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    pl, fl = _losses(net)
    seq = []
    res = D.train_round_robin(net, opt, 6, pl, fl, 1.0, on_cycle=lambda c, t, l: seq.append((c, t)))
    assert res["counts"] == {"point": 6, "flow": 6}
    assert res["first_task_by_cycle"] == ["point", "flow"] * 3
    assert seq[:4] == [(1, "point"), (1, "flow"), (2, "flow"), (2, "point")]
    st = opt.state
    g = net.groups()
    assert all(int(st[p]["step"]) == 12 for p in g["shared"]) and all(int(st[p]["step"]) == 6 for p in g["point"] + g["flow"])
    assert D.cycle_order(1) == ("point", "flow") and D.cycle_order(20000) == ("flow", "point")


def test_dual_training_budget(dp):
    src = inspect.getsource(dp.stage_train_dual)
    assert 'D.train_round_robin(net, opt, PROTO["steps"], point_loss, flow_loss, PROTO["clip"], on_cycle)' in src
    assert 'res["counts"] != {"point": PROTO["steps"], "flow": PROTO["steps"]}' in src
    assert dp.PROTO == {"steps": 20000, "batch": 64, "lr": 1e-3, "wd": 0.01, "clip": 1.0} and dp.SEED == 42
    assert src.count("torch.optim.AdamW(") == 1 and "torch.Generator().manual_seed(SEED)" in src
    for bad in ("GradNorm", "pcgrad", "PCGrad", "lambda_", "weight * "):
        assert bad not in src


# ------------------------------------------------------------------------------------------ 29-30 sampling
def test_deterministic_noise_mapping(dp):
    a = SM.window_noise([7, 7, 9], [1, 2, 3])
    assert np.array_equal(a, SM.window_noise([7, 7, 9], [1, 2, 3])) and not np.array_equal(a[0], a[1])
    assert SM.noise_seed(7, 1) == SM.noise_seed(7, 1) and SM.NOISE_SALT == 20261002
    assert '"noise": SM.window_noise(Pid, wid)' in inspect.getsource(dp.context)


def test_euler8_exactly_eight_nfe():
    net = tiny()
    calls = {"v": 0, "e": 0}
    v0, e0 = net.velocity, net.encode_flow
    net.velocity = lambda *a, **k: (calls.__setitem__("v", calls["v"] + 1), v0(*a, **k))[1]
    net.encode_flow = lambda *a, **k: (calls.__setitem__("e", calls["e"] + 1), e0(*a, **k))[1]
    D.euler(net, torch.randn(2, 512), torch.randn(2, 512), torch.rand(2, 512), 8)
    assert calls == {"v": 8, "e": 1}


# ------------------------------------------------------------------------------------------ 32-33 accounting / bootstrap / gates
def test_parameter_saving_and_e1():
    sep = 593_577 + 598_333
    assert abs(D.saving(861_196, sep) - (1 - 861_196 / 1_191_910)) < 1e-12
    assert D.e1_pass(861_196, sep) and D.e1_pass(943_372, sep) and not D.e1_pass(1_189_900, sep)
    assert D.e1_pass(int(0.85 * sep), sep) and not D.e1_pass(int(0.85 * sep) + 1, sep)


def test_patient_clustered_bootstrap(dp):
    pid_a = np.repeat(np.arange(30), 2)
    pid_b = np.repeat(np.arange(30), np.arange(30) % 5 + 1)
    vals = np.random.default_rng(0).normal(size=30)
    ca, cb = dp.ci(vals[pid_a], pid_a), dp.ci(vals[pid_b], pid_b)
    assert np.allclose(ca, cb)                                                         # patients, not windows, are resampled
    assert "B.patient_resamples(np.unique(Pid).size, BOOT_N, BOOT_SEED)" in inspect.getsource(dp.evaluate)
    assert dp.BOOT_N == 2000 and dp.BOOT_SEED == 20261002


def test_gates_and_selection_rules():
    sep = 1_191_910
    ok = {"corr": [0.0, -0.019, 0.01], "fp": [0.0, -0.01, 0.049], "recall": [0.0, -0.009, 0.01], "fd": [0.0, -1.0, 0.99],
          "fd_shuf": [5.0, 0.1, 9.0], "corr_shuf": [-0.2, -0.3, -0.01]}
    assert D.gates(ok, 861_196, sep)["QUALIFIED"]
    for k, bad in (("corr", [0.0, -0.021, 0.0]), ("fp", [0.0, 0.0, 0.051]), ("recall", [0.0, -0.011, 0.0]), ("fd", [0.0, 0.0, 1.01]),
                   ("fd_shuf", [1.0, -0.1, 2.0]), ("corr_shuf", [-0.1, -0.2, 0.01])):
        assert not D.gates(ok | {k: bad}, 861_196, sep)["QUALIFIED"]
    assert not D.gates(ok, 1_189_900, sep)["QUALIFIED"]
    c = {"S0": {"qualified": True, "saving": 0.27, "fd": 9.0, "corr": 0.8, "latency_both_ms": 9},
         "S1": {"qualified": True, "saving": 0.20, "fd": 5.0, "corr": 0.9, "latency_both_ms": 5},
         "S2": {"qualified": False, "saving": 0.0, "fd": 1.0, "corr": 0.99, "latency_both_ms": 1}}
    assert D.select_winner(c) == "S0"
    assert D.select_winner({k: v | {"qualified": False} for k, v in c.items()}) is None
