"""GX1 tests (docs/GX1_ABP_GENERALIZATION_PREREGISTRATION.md): transferred topology, split, seals, target interface,
frozen evaluator / margins, gate rules and the adapter on synthetic subjects. No GX-LOCK value is read here."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.dualreadout import model as D

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/gx1_abp"


@pytest.fixture(scope="module")
def gx():
    sys.path.insert(0, str(ROOT / "scripts"))
    import gx1_abp
    return gx1_abp


def test_transferred_topology_exact(gx):
    net = gx.F.build("S1")
    assert [b.c1.dilation[0] for b in net.shared_blocks] == [1, 2, 4, 8, 16, 32]
    assert len(net.point_tower) == len(net.flow_tower) == 2
    acc = gx.F.accounting()
    assert (acc["S1"]["total"], acc["S1"]["shared"], acc["separate_waveform_params"]) == (943_372, 246_720, 1_191_910)
    src = inspect.getsource(gx.stage_train)
    assert 'F.stage_train_dual(e, d, "S1")' in src and "S0" not in src and "S2" not in src
    assert gx.SEEDS == (42, 43, 44)


def test_only_target_interface_changes(gx):
    src = inspect.getsource(gx.stage_train)
    assert "F.SEED, F.OUT, F._save, F._train_data = seed, OUT / f\"seed{seed}\", _saver(f\"seed{seed}\"), _gx_train_data" in src
    tr = inspect.getsource(gx._gx_train_data)
    assert 'load_role("train")' in tr and "(A - tn.mu) / tn.sigma" in tr and "AB.event_raster(ref)" in tr
    n = json.loads((ROOT / "artifacts/a8_abp_scale_control/normalization.json").read_text())
    t = gx.tnorm()
    assert (t.mu, t.sigma) == (n["mu_train"], n["sigma_train"]) and n["n_train_subjects"] == 1100
    assert gx.F.PROTO == {"steps": 20000, "batch": 64, "lr": 1e-3, "wd": 0.01, "clip": 1.0} and gx.F.NFE == 8


def test_split_official_and_disjoint(gx):
    sm = json.loads((ART / "split_manifest.json").read_text())
    r = {k: set(v["subjects"]) for k, v in sm["roles"].items()}
    assert {k: len(v) for k, v in r.items()} == {"gx_train": 1100, "gx_dev": 195, "gx_lock": 229}
    assert not (r["gx_train"] & r["gx_dev"] or r["gx_train"] & r["gx_lock"] or r["gx_dev"] & r["gx_lock"])
    assert set(gx.role_subjects("lock")) == r["gx_lock"] and set(gx.role_subjects("train")) == r["gx_train"]


def test_lock_sealed_before_freeze(gx, tmp_path, monkeypatch):
    monkeypatch.setattr(gx, "FREEZE", tmp_path / "final_freeze_manifest.json")
    for f in (lambda: gx.load_role("lock"), lambda: gx.build_role("lock", None), gx.check_final_freeze):
        with pytest.raises(PermissionError):
            f()
    assert inspect.getsource(gx.stage_eval_lock).split('"""')[0].count("check_final_freeze()") == 1
    with pytest.raises(SystemExit):
        gx.stage_build(None, None, "lock")


def test_evaluator_and_margins_frozen_from_train(gx):
    ev = json.loads((ART / "evaluator_manifest.json").read_text())
    assert "kanflow_fd" in ev["D_ABP"] and ev["pre_gx1_use_on_abp"][0]["commit"] == "c753d13"
    m = json.loads((ART / "margins.json").read_text())
    assert abs(m["G3_margin_mmhg2"] - m["G3_trace_abp_mmhg2"] / m["G3_trace_ecg_dp_train"]) < 1e-6
    assert abs(m["G2_mae_margin_z"] - 0.02 * m["train_target_sd_z"]) < 1e-12 and m["G1_corr_margin"] == -0.02
    src = inspect.getsource(gx.stage_margins)
    assert 'load_role("train")' in src and 'load_role("lock")' not in src and 'load_role("dev")' not in src


def test_gate_rules(gx):
    m = {"G2_mae_margin_z": 0.02, "G3_margin_mmhg2": 2785.0}
    ok = {"corr": [0, -0.019, 0], "mae_z": [0, 0, 0.0199], "D": [0, 0, 2784.0], "D_shuf": [5, 1, 9], "corr_shuf": [-0.2, -0.3, -0.1]}
    assert gx.gates(ok, m)["PASS"]
    for k, bad in (("corr", [0, -0.021, 0]), ("mae_z", [0, 0, 0.0201]), ("D", [0, 0, 2786.0]), ("D_shuf", [1, -1, 3]), ("corr_shuf", [-0.1, -0.2, 0.01])):
        assert not gx.gates(ok | {k: bad}, m)["PASS"], k


def test_wave_corr_and_bp(gx):
    t = np.linspace(0, 4, 512)
    y = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    assert np.allclose(gx.wave_corr(y, y), 1) and np.allclose(gx.wave_corr(-y, y), -1)
    e = gx.bp_errors(y + 2.0, y)
    assert np.allclose(e["sbp"], 2) and np.allclose(e["dbp"], 2) and np.allclose(e["map"], 2)


def test_shuffle_keeps_raster_and_noise_and_no_search(gx):
    s = inspect.getsource(gx.evaluate_pair)
    assert 'DP3.gen_infer(n, X[perm], R, ctx["noise"], dev)' in s and 'DP3.gen_infer(n, X, R, ctx["noise"], dev)' in s
    assert "np.random.default_rng(SHUF_SEED).permutation" in inspect.getsource(gx.context)
    d = inspect.getsource(gx.stage_train_diag)
    assert "not FREEZE.exists()" in d and 'share not in ("S0", "S2")' in d
    assert "multiseed_summary.json" in inspect.getsource(gx.stage_eval_diag)
    assert "argmax" not in inspect.getsource(gx) and "best_seed" not in inspect.getsource(gx)


def _fake(root, pid, rng):
    t = np.arange(30 * 3750) / 125.0
    ecg = sum(np.exp(-0.5 * ((t - r) / 0.012) ** 2) for r in np.arange(0.3, t[-1], 0.8))
    ppg = np.sin(2 * np.pi * (t - 0.2) / 0.8) + 0.05 * rng.standard_normal(t.size)
    abp = 80 + 20 * np.sin(2 * np.pi * (t - 0.15) / 0.8)
    for kind, a in (("ppg", ppg), ("abp", abp), ("ecg", ecg + 0.01 * rng.standard_normal(t.size))):
        (root / kind).mkdir(parents=True, exist_ok=True)
        a = a.reshape(30, 3750).copy()
        if kind == "abp":
            a[3, :500] = 80.0                                            # constant ABP window -> R2
        np.save(root / kind / f"{pid}_{kind}.npy", a)


def test_adapter_deterministic(gx, tmp_path, monkeypatch):
    _fake(tmp_path, "p000009", np.random.default_rng(0))
    monkeypatch.setattr(gx, "MIMIC", tmp_path)
    _, d1, l1 = gx._subject((2, "p000009"))
    _, d2, l2 = gx._subject((2, "p000009"))
    assert np.array_equal(d1["x"], d2["x"]) and np.array_equal(d1["abp_mmhg"], d2["abp_mmhg"]) and l1 == l2
    assert d1["x"].shape[1] == 512 and d1["abp_mmhg"].shape[1] == 512 and abs(float(d1["abp_mmhg"].mean()) - 80) < 1
    assert any(r[3].startswith("R2") and r[1] == 3 and r[2] == 0 for r in l1)
    assert int(d1["wid"].min()) >= 2 * 210 and int(d1["wid"].max()) < 3 * 210
