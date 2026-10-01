"""R1 unit tests (docs/R1_RHYTHMFIELD_WW_PREREGISTRATION.md §11). Synthetic tensors, the V1 manifest, frozen checkpoint
hashes and static checks only; no R1 metric is computed here."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.coherentbeat import ablation as AB
from ppg2ecg.coherentbeat import model as CM
from ppg2ecg.coherentbeat import split as SP
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, soft_event_field
from ppg2ecg.rhythmfield import model as RM

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def r1():
    sys.path.insert(0, str(ROOT / "scripts"))
    import r1_rhythmfield
    return r1_rhythmfield


def _sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ 1-5 frozen data / models
def test_arch_split_reused_without_holdout(r1):
    m = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())
    sp = SP.arch_split(m)
    assert {r: len(sp[r]["patients"]) for r in sp} == {"train": 3470, "val": 433, "holdout": 434}
    src = inspect.getsource(r1)
    assert 'load_arch("train")' in src and 'load_arch("val")' in src and 'load_arch("holdout")' not in src


def test_old_test_sealed_before_final_freeze(r1, tmp_path, monkeypatch):
    monkeypatch.setattr(r1, "FREEZE", tmp_path / "final_test_freeze_manifest.json")
    with pytest.raises(PermissionError):
        r1.load_test()
    (tmp_path / "final_test_freeze_manifest.json").write_text(json.dumps({"selected": "NONE", "freshness_audit": "CLEAN", "sha256": {}}))
    with pytest.raises(PermissionError):
        r1.load_test()
    (tmp_path / "final_test_freeze_manifest.json").write_text(json.dumps({"selected": "SOFT", "freshness_audit": "NOT CLEAN", "sha256": {}}))
    with pytest.raises(PermissionError):
        r1.load_test()


@pytest.mark.parametrize("path,record,key", [("outputs/c0a_coherentbeat_ablation/ww.pt", "artifacts/c0a_coherentbeat_ablation/checkpoint_hashes.json", "ww"),
                                             ("outputs/c0_coherentbeat/c0.pt", "artifacts/c0_coherentbeat/checkpoint_hashes.json", "c0"),
                                             ("outputs/c0_coherentbeat/detector.pt", "artifacts/c0_coherentbeat/checkpoint_hashes.json", "detector")])
def test_frozen_checkpoints_reproduce(path, record, key):
    if not (ROOT / path).exists():
        pytest.skip("frozen checkpoint not on this machine")
    assert _sha(ROOT / path) == json.loads((ROOT / record).read_text())[key]


# ------------------------------------------------------------------------------------------ 6-9 SOFT
def test_soft_field_is_pre_threshold_detector_output(r1):
    src = inspect.getsource(r1.detector_field)
    assert "torch.sigmoid(det(" in src and "extract" not in src and "threshold" not in src.split('"""')[2]
    ev = inspect.getsource(r1.evaluate)
    assert "fdet = detector_field(X, dev)" in ev and "render_soft(X, fdet, dev)" in ev
    assert "_extract, list(fdet)" in ev                    # placed events are extracted from the same field, for the other arms


def test_soft_decoder_never_sees_thresholded_events(r1):
    tr = inspect.getsource(r1._train)
    assert "detector_field(X, dev)" in tr and "event_raster" not in tr and "extract" not in tr
    assert "event_raster" not in inspect.getsource(r1.render_soft)


def test_soft_uses_dense_field_channel_and_ww_family():
    soft = RM.soft_rhythm_ww()
    assert isinstance(soft, AB.WWDet) and soft.dec_in.in_channels == CM.CH + 1      # one conditioning channel, no second raster
    assert RM.n_params(soft) == 593577                                              # WW-DET exactly
    x, f = torch.randn(2, 512), torch.rand(2, 512)
    assert soft(x, f).shape == (2, 512)


# ------------------------------------------------------------------------------------------ 10-14 JOINT
def test_joint_structure():
    j = RM.JointRhythmFieldWW()
    assert sum(isinstance(m, CM.Encoder) for m in j.modules()) == 1
    assert not any(isinstance(m, RhythmTCN) for m in j.modules())
    pb = RM.param_breakdown(j)
    assert pb["waveform"] == 593577 and pb["rhythm_head"] == 4225 and pb["total"] <= 1.15 * 593577
    y, z = j(torch.randn(3, 512))
    assert y.shape == (3, 512) and z.shape == (3, 512)
    assert j.dec_in.in_channels == CM.CH + 1


def test_joint_field_feeds_decoder_and_gradients_flow():
    torch.manual_seed(0)
    j = RM.JointRhythmFieldWW()
    x = torch.randn(4, 512)
    y, z = j(x)
    y.abs().mean().backward()                               # waveform loss only
    assert j.rhythm_head.net[2].weight.grad.abs().sum() > 0  # reaches the rhythm head through p_R (not detached)
    j.zero_grad()
    y, z = j(x)
    tgt = torch.from_numpy(np.stack([soft_event_field(np.array([100, 300]), 512, 2.56)] * 4))
    torch.nn.functional.binary_cross_entropy_with_logits(z, tgt).backward()   # rhythm loss only
    assert j.rhythm_head.net[2].weight.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in j.head.parameters())
    with torch.no_grad():                                   # changing the field changes the waveform
        j.rhythm_head.net[2].bias += 5.0
        y2, _ = j(x)
    assert not torch.allclose(y.detach(), y2)


# ------------------------------------------------------------------------------------------ 15-18 exclusions
def test_no_guard_renderer_stochastic_path_or_extra_losses(r1):
    msrc = inspect.getsource(RM)
    for bad in ("guard", "supports", "envelope", "assemble", "randn", "torch.rand", "dropout", "Dropout"):
        assert bad not in msrc
    j = RM.JointRhythmFieldWW().eval()
    x = torch.randn(2, 512)
    assert torch.equal(j(x)[0], j(x)[0])
    tr = inspect.getsource(r1._train)
    for bad in ("spectral", "pcc", "kanflow_fd", "hr_abs", "peak_loss", "adversarial", "morph", "corr"):
        assert bad not in tr.lower()
    assert [ln.strip() for ln in tr.splitlines() if ln.strip().startswith("loss =")] == [
        "loss = (net(Xt[bd], Ft[bd].float()) - Yt[bd]).abs().mean()", "loss = (y - Yt[bd]).abs().mean() + RM.LAMBDA_RHYTHM * lr_"]
    assert tr.count("abs().mean()") == 2 and "LAMBDA_RHYTHM" in tr and RM.LAMBDA_RHYTHM == 1.0


# ------------------------------------------------------------------------------------------ 19 rhythm target
def test_rhythm_target_convention_and_train_only(r1):
    t = r1.rhythm_target([np.array([100, 228])])
    assert np.allclose(t[0], soft_event_field(np.array([100.0, 228.0]), 512, 20.0 / 1000 * 128))
    tr = inspect.getsource(r1._train)
    assert 'C0.load_arch("train")' in tr and 'reference_peaks("train"' in tr and "rhythm_target(ref)" in tr


# ------------------------------------------------------------------------------------------ 20-23 statistics / rules
def test_patient_macro_pools_within_patient():
    rows = RM.patient_macro_rows([3, 1, 0, 2], [1, 0, 0, 2], [0, 1, 4, 0], [7, 7, 9, 9])
    assert rows["patients"].tolist() == [7, 9]
    assert rows["precision"][0] == pytest.approx(4 / 5) and rows["recall"][0] == pytest.approx(4 / 5)
    assert rows["recall"][1] == pytest.approx(2 / 6) and rows["fp_rate"][1] == pytest.approx(1.0)
    assert rows["f1"][1] == pytest.approx(4 / (4 + 2 + 4))
    assert np.isnan(RM.patient_macro_rows([0], [0], [3], [1])["precision"][0])


def test_paired_comparisons_and_test_seal(r1):
    ev = inspect.getsource(r1.evaluate)
    assert "pm[cand][m] - pm[o][m]" in ev and 'subs = pm["WW"]["patients"]' in ev and "C0.fd_diff_ci(waves[cand]" in ev
    assert "load_test()" in ev and "check_final_freeze()" in inspect.getsource(r1.load_test)
    assert "VERDICT: CLEAN" in inspect.getsource(r1.stage_freeze)


def _d(**kw):
    d = {"fp_vs_ww": [-0.1, -0.12, -0.08], "fp_vs_c0": [0.01, 0.0, 0.02], "recall_vs_ww": [0.0, -0.002, 0.002],
         "fd_vs_ww": [0.5, 0.2, 0.8], "fd_vs_c0": [-3.0, -3.5, -2.5], "corr_vs_ww": [0.0, -0.005, 0.005]}
    d.update(kw)
    return d


def test_gates_selection_and_verdict():
    assert RM.val_gates(_d())["QUALIFIED"]
    assert not RM.val_gates(_d(fp_vs_c0=[0.02, 0.01, 0.031]))["V2"]
    assert not RM.val_gates(_d(fd_vs_c0=[-1.0, -2.0, 0.1]))["V4"]
    assert not RM.val_gates(_d(recall_vs_ww=[-0.003, -0.006, 0.0]))["V3"]
    assert RM.select_candidate({"SOFT": False, "JOINT": False}, {}, {}) == "NONE"
    assert RM.select_candidate({"SOFT": False, "JOINT": True}, {"SOFT": 1, "JOINT": 2}, {"SOFT": 0, "JOINT": 0}) == "JOINT"
    assert RM.select_candidate({"SOFT": True, "JOINT": True}, {"SOFT": 20.0, "JOINT": 19.5}, {"SOFT": 0.4, "JOINT": 0.5}) == "JOINT"
    assert RM.select_candidate({"SOFT": True, "JOINT": True}, {"SOFT": 20.0, "JOINT": 19.9}, {"SOFT": 0.40, "JOINT": 0.45}) == "SOFT"
    assert RM.select_candidate({"SOFT": True, "JOINT": True}, {"SOFT": 20.0, "JOINT": 19.9}, {"SOFT": 0.405, "JOINT": 0.40}) == "SOFT"
    assert RM.test_verdict(_d())["verdict"] == "STRONG"
    assert RM.test_verdict(_d(fp_vs_c0=[0.03, 0.02, 0.05]))["verdict"] == "PARTIAL"
    assert RM.test_verdict(_d(fp_vs_c0=[0.05, 0.04, 0.07]))["verdict"] == "FAILED"
    assert RM.test_verdict(_d(fp_vs_ww=[0.0, -0.01, 0.01]))["verdict"] == "FAILED"
    assert RM.test_verdict(_d(recall_vs_ww=[-0.008, -0.011, -0.005]))["verdict"] == "FAILED"
