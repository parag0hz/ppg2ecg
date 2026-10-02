"""DP3 tests (docs/DP3_MIMICBP_TARGET_BLIND_PREREGISTRATION.md §tests): frozen S1, seeds, training protocol, the MIMIC-BP
adapter on synthetic subjects, the seals and the gate rules. No MIMIC-BP waveform value is read here."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.dualreadout import model as D
from ppg2ecg.scaleflow import model as SM

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/dp3_mimicbp"


@pytest.fixture(scope="module")
def dp():
    sys.path.insert(0, str(ROOT / "scripts"))
    import dp3_mimicbp
    return dp3_mimicbp


# ------------------------------------------------------------------------------------------ 1-5 frozen S1
def test_frozen_s1_accounting_and_graph(dp):
    acc = dp.F.accounting()
    s1 = acc["S1"]
    assert (s1["total"], s1["shared"], s1["adapters"], acc["separate_waveform_params"]) == (943_372, 246_720, 8_320, 1_191_910)
    assert round(100 * D.saving(s1["total"], acc["separate_waveform_params"]), 2) == 20.85
    net = dp.build("S1")
    assert [b.c1.dilation[0] for b in net.shared_blocks] == [1, 2, 4, 8, 16, 32]
    assert [b.c1.dilation[0] for b in net.point_tower] == [b.c1.dilation[0] for b in net.flow_tower] == [1, 2]
    own = net.ownership()
    assert {k: sum(v == k for v in own.values()) for k in ("shared", "point", "flow")} == {"shared": 26, "point": 34, "flow": 140}


# ------------------------------------------------------------------------------------------ 6-11 checkpoints / seeds / detector / data
def test_seed42_hashes_match_dp0(dp):
    h = json.loads((ROOT / "artifacts/dp0_dualreadout/checkpoint_hashes.json").read_text())
    for role, k in (("point", "point"), ("gen", "gen"), ("S1", "S1")):
        assert dp.ckpt_file(role, 42) == ROOT / f"outputs/dp0_dualreadout/{k}.pt"
        assert dp.B.sha256_file(dp.ckpt_file(role, 42)) == h[k]
    assert dp.B.sha256_file(ROOT / "outputs/dp0_dualreadout/detector.pt") == h["detector"]


def test_new_seed_configs_differ_only_by_seed(dp):
    assert dp.NEW_SEEDS == (43, 44) and dp.SEEDS == (42, 43, 44)
    src = inspect.getsource(dp.stage_train)
    assert "F.SEED = seed" in src and "F.OUT = OUT / f\"seed{seed}\"" in src and "F._save = _saver(seed)" in src
    assert "seed" not in json.dumps(dp.config_spec("S1"), default=str).lower().replace("dp_train", "")
    p = ART / "training_seed_manifest.json"
    if p.exists():
        jobs = json.loads(p.read_text())["jobs"]
        for role in ("point", "gen", "S1"):
            hs = {j["config_sha256"] for j in jobs if j["role"] == role}
            assert hs == {dp.config_hash(role)} or not hs
        assert all(j["nan_steps"] == 0 for j in jobs)


def test_detector_frozen_and_rule(dp):
    assert (dp.F.DET["threshold"], dp.F.DET["refractory"], dp.F.DET["sigma_ms"]) == (0.35, 32, 20.0)
    src = inspect.getsource(dp)
    assert "stage_train_detector" not in src and 'DP0_OUT / "detector.pt"' in inspect.getsource(dp.load_detector)


def test_training_population_is_dp_train_only(dp):
    sp = json.loads((ROOT / "artifacts/dp0_dualreadout/split_hashes.json").read_text())["dp_train"]
    assert dp.config_spec("point")["split_sha256"] == sp
    assert 'load_role("dp_train"' in inspect.getsource(dp.F._train_data)
    for f in (dp.stage_train, dp._saver, dp.config_spec):
        s = inspect.getsource(f)
        assert "MIMIC" not in s and "_subject_windows" not in s and "external" not in s.lower()


# ------------------------------------------------------------------------------------------ 12-20 training / inference protocol
def test_no_external_training_or_finetuning(dp):
    for f in (dp.evaluate_seed, dp.stage_eval_external, dp.external_context, dp._subject_windows, dp.point_infer, dp.gen_infer):
        s = inspect.getsource(f)
        assert "backward(" not in s and "optim" not in s and "train()" not in s


def test_losses_exact(dp):
    assert "(net(Xt[b], Rt[b]) - Yt[b]).abs().mean()" in inspect.getsource(dp.F.stage_train_point)
    assert "SM.fm_loss(net, Yt[b], Xt[b], Rt[b], gd)" in inspect.getsource(dp.F.stage_train_gen)
    x1 = torch.randn(3, 512)
    xt, u = SM.fm_path(x1, torch.zeros(3, 512), torch.full((3,), 0.5))
    assert torch.allclose(xt, 0.5 * x1) and torch.allclose(u, x1)


def test_round_robin_masks_and_update_counts(dp):
    torch.manual_seed(0)
    net = D.DualReadout("S1", 16, 16)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    g = net.groups()
    ppg, ras, y = torch.randn(4, 512), torch.rand(4, 512), torch.randn(4, 512)
    gd = torch.Generator().manual_seed(1)
    before = [p.detach().clone() for p in g["flow"]]
    D.substep(net, opt, g, "point", lambda: (net.point(ppg, ras) - y).abs().mean(), 1.0)
    assert all(torch.equal(a, b) for a, b in zip(before, g["flow"]))
    before = [p.detach().clone() for p in g["point"]]
    D.substep(net, opt, g, "flow", lambda: D.fm_loss(net, y, ppg, ras, gd), 1.0)
    assert all(torch.equal(a, b) for a, b in zip(before, g["point"]))
    res = D.train_round_robin(net, opt, 4, lambda: (net.point(ppg, ras) - y).abs().mean(), lambda: D.fm_loss(net, y, ppg, ras, gd), 1.0)
    assert res["counts"] == {"point": 4, "flow": 4} and res["first_task_by_cycle"] == ["point", "flow", "point", "flow"]
    assert dp.F.PROTO["steps"] == 20000
    assert 'res["counts"] != {"point": PROTO["steps"], "flow": PROTO["steps"]}' in inspect.getsource(dp.F.stage_train_dual)


def test_euler8_and_haar(dp):
    assert dp.F.NFE == 8
    torch.manual_seed(0)
    net = D.DualReadout("S1", 16, 16)
    n = {"v": 0}
    v0 = net.velocity
    net.velocity = lambda *a, **k: (n.__setitem__("v", n["v"] + 1), v0(*a, **k))[1]
    D.euler(net, torch.randn(2, 512), torch.randn(2, 512), torch.rand(2, 512), dp.F.NFE)
    assert n["v"] == 8
    x = torch.randn(3, 512, dtype=torch.float64)
    assert torch.allclose(SM.ihaar(*SM.haar(x)), x) and torch.allclose(sum((c ** 2).sum() for c in SM.haar(x)), (x ** 2).sum())
    assert "F.NFE" in inspect.getsource(dp.gen_infer)


# ------------------------------------------------------------------------------------------ 21-26 MIMIC-BP adapter (synthetic subjects)
def _fake_subject(root, pid, rng, nan_window=False, flat_window=False):
    t = np.arange(30 * 3750) / 125.0
    ecg = np.zeros(30 * 3750)
    for r in np.arange(0.3, t[-1], 0.8):
        ecg += np.exp(-0.5 * ((t - r) / 0.012) ** 2)
    ppg = np.sin(2 * np.pi * (t - 0.2) / 0.8) + 0.05 * rng.standard_normal(t.size)
    ppg, ecg = ppg.reshape(30, 3750), ecg.reshape(30, 3750) + 0.01 * rng.standard_normal((30, 3750))
    if nan_window:
        ppg[0, 10] = np.nan
    if flat_window:
        ecg[1, 500:1000] = 0.0
    for kind, a in (("ppg", ppg), ("ecg", ecg)):
        (root / kind).mkdir(parents=True, exist_ok=True)
        np.save(root / kind / f"{pid}_{kind}.npy", a)


def test_adapter_deterministic_windows_and_exclusions(dp, tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    _fake_subject(tmp_path, "p000007", rng, nan_window=True, flat_window=True)
    monkeypatch.setattr(dp, "MIMIC", tmp_path)
    pid, data, log = dp._subject_windows((3, "p000007"))
    pid2, data2, log2 = dp._subject_windows((3, "p000007"))
    assert np.array_equal(data["x"], data2["x"]) and np.array_equal(data["wid"], data2["wid"]) and log == log2
    assert data["x"].shape[1] == 512 and data["y"].shape[1] == 512 and dp.WIN_PER_SEG == 7 and dp.WIN_RAW == 500
    rules = {r[3].split(" ")[0] for r in log}
    assert "R2" in rules                                                   # NaN PPG window and the flat ECG window
    assert dp.window_ids(3, 0, 0) == 3 * 210 and dp.window_ids(3, 29, 6) == 4 * 210 - 1
    assert set(data["wid"].tolist()) <= set(range(3 * 210, 4 * 210)) and 3 * 210 + 0 not in set(data["wid"].tolist())
    assert len(data["wid"]) + sum(r[3].startswith(("R2", "R3", "R4")) for r in log) == 210


def test_resampler_and_normalization_frozen(dp):
    src = inspect.getsource(dp.preprocess_windows)
    assert "signal.resample(x, resample_rate * segment_len, axis=1)" in src
    a = np.random.default_rng(1).standard_normal((3, 500))
    y1, y2 = dp.preprocess_windows(a, 128, 4, **dp.PPG_KW), dp.preprocess_windows(a, 128, 4, **dp.PPG_KW)
    assert y1.shape == (3, 512) and np.array_equal(y1, y2) and np.allclose(y1.min(1), -1) and np.allclose(y1.max(1), 1)
    assert dp.PPG_KW == {"bandpass": True, "freq_range": (0.5, 4), "zscore": True, "normalize": True}
    assert dp.ECG_KW == {"bandpass": True, "freq_range": (0.5, -1), "zscore": True, "normalize": True}


def test_channel_rules_and_no_alignment_search(dp):
    s = inspect.getsource(dp._subject_windows)
    assert s.count('MIMIC / "ppg" / f"{pid}_ppg.npy"') == 1 and s.count('MIMIC / "ecg" / f"{pid}_ecg.npy"') == 1
    assert "ppg[:, :WIN_PER_SEG * WIN_RAW]" in s and "ecg[:, :WIN_PER_SEG * WIN_RAW]" in s
    src = inspect.getsource(dp)
    for bad in ("np.roll", "lag", "correlate(", "shift(", "argmax(corr", "best_lead", "abp\" /"):
        assert bad not in src, bad
    rules = json.loads((ART / "external_exclusion_rules.json").read_text())
    assert [r[:2] for r in rules["order"]] == ["R1", "R2", "R3", "R4", "R5"] and dp.HR_RANGE == (30.0, 200.0)


# ------------------------------------------------------------------------------------------ 27-28 seals
def test_external_blocked_before_prereg_and_freeze(dp, tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "FREEZE", tmp_path / "final_freeze_manifest.json")
    with pytest.raises(PermissionError):
        dp.check_final_freeze()
    (tmp_path / "final_freeze_manifest.json").write_text(json.dumps({"sha256": {}}))
    monkeypatch.setattr(dp, "PREREG", "docs/__missing_prereg__.md")
    with pytest.raises(PermissionError):
        dp.check_final_freeze()
    for f in (dp.stage_eval_external, dp.external_context):
        assert inspect.getsource(f).split(":", 1)[1].strip().startswith(("check_final_freeze()", '"""'))
        assert "check_final_freeze()" in inspect.getsource(f)
    src = inspect.getsource(dp)
    assert src.count("_subject_windows, list(enumerate(ids))") == 1                # only inside external_context


# ------------------------------------------------------------------------------------------ 29-33 evaluation rules
def test_noise_pairing_shuffle_and_bootstrap(dp):
    s = inspect.getsource(dp.evaluate_seed)
    assert 'gen_infer(nets["gen"], X, R, ctx["noise"], dev)' in s and 'gen_infer(nets["S1"], X, R, ctx["noise"], dev)' in s
    assert 'X[perm], R, ctx["noise"]' in s                                         # PPG shuffled; raster and noise kept
    assert '"noise": SM.window_noise(Pid, W)' in inspect.getsource(dp.external_context)
    pid_a, pid_b = np.repeat(np.arange(20), 2), np.repeat(np.arange(20), np.arange(20) % 4 + 1)
    v = np.random.default_rng(0).normal(size=20)
    assert np.allclose(dp.ci(v[pid_a], pid_a), dp.ci(v[pid_b], pid_b))
    assert dp.BOOT_N == 2000 and dp.BOOT_SEED == 20261002 and dp.SHUF_SEED == 20261002


def test_cached_separate_baseline_is_numerically_identical(dp):
    torch.manual_seed(0)
    g = SM.ScaleFM(16, True).eval()
    x0, ppg, ras = torch.randn(2, 512), torch.randn(2, 512), torch.rand(2, 512)
    assert torch.equal(SM.euler(g, x0, ppg, ras, 8), dp.euler_cached(g, x0, ppg, ras, 8))


def test_gates_and_no_best_seed(dp, tmp_path, monkeypatch):
    sep = 1_191_910
    ok = {"corr": [0, -0.019, 0], "fp": [0, 0, 0.049], "recall": [0, -0.009, 0], "fd": [0, 0, 0.99], "fd_shuf": [1, 0.1, 2], "corr_shuf": [-1, -1, -0.01]}
    assert D.gates(ok, 943_372, sep)["QUALIFIED"]
    for k, bad in (("corr", [0, -0.021, 0]), ("fp", [0, 0, 0.051]), ("recall", [0, -0.011, 0]), ("fd", [0, 0, 1.01]),
                   ("fd_shuf", [1, -0.1, 2]), ("corr_shuf", [-1, -1, 0.01])):
        assert not D.gates(ok | {k: bad}, 943_372, sep)["QUALIFIED"]
    assert not D.gates(ok, int(0.85 * sep) + 1, sep)["E1"]
    monkeypatch.setattr(dp, "ART", tmp_path)

    def fake(passes):
        g = {"P1": passes, "P2": True, "G1": True, "CONDITION": True, "E1": True, "QUALIFIED": passes}
        o = {"gates": g, "point": {k: {"corr": [0.8], "pm_fp_rate": [0.5], "pm_recall": [0.7]} for k in ("P", "S1")}, "gen": {k: {"fd": 10.0} for k in ("G", "S1")},
             "comparisons": {k: [0.0, 0, 0] for k in ("corr", "fp", "recall", "fd", "fd_shuf", "corr_shuf")}}
        return o
    dp._multiseed_summary({42: fake(False), 43: fake(True), 44: fake(True)})
    s = json.loads((tmp_path / "external_multiseed_summary.json").read_text())
    assert s["primary_seed42"] == "FAILED" and s["robustness"] == "ROBUST-2/3" and s["paper_readiness"].startswith("INTERNAL + CROSS-TASK")
    dp._multiseed_summary({42: fake(True), 43: fake(True), 44: fake(False)})
    s = json.loads((tmp_path / "external_multiseed_summary.json").read_text())
    assert s["primary_seed42"] == "CONFIRMED" and s["paper_readiness"] == "GO WITH SEED LIMITATION"
    src = inspect.getsource(dp)
    assert "argmax" not in src.replace("np.argsort", "") and "best_seed" not in src and "allres[42][\"gates\"][\"QUALIFIED\"]" in src


def test_no_architecture_search_and_target_blind_pass(dp):
    src = inspect.getsource(dp)
    assert "DualReadout(" not in src and "choose_widths(" not in src and "width" not in inspect.getsource(dp.build)
    a = json.loads((ART / "target_blind_audit.json").read_text())
    assert a["verdict"] == "PASS" and a["prior_mimicbp_ecg_target_use"] == "NONE" and a["evidence_label"] == dp.EVIDENCE_LABEL
    assert not a["code_paths"]["python_lines_loading_mimicbp_ecg_arrays"] and a["scan"]["unadjudicated_files"] == []
