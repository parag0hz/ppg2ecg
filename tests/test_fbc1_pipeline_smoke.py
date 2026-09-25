"""FBC1 end-to-end smoke test on SYNTHETIC HR banks (no real data is read): grid-independent stages fullval -> nested ->
test -> figure run in a temporary artifact directory with tiny bootstrap counts."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fbc1_run as F  # noqa: E402


def _fake_split(split):
    P = F.N_VAL_PATIENTS if split == "val" else F.N_TEST_PATIENTS
    rng = np.random.default_rng(0 if split == "val" else 1)
    pid = np.repeat(np.arange(1000, 1000 + P), 3)
    ref = rng.normal(75, 12, len(pid))
    banks = {}
    for mi, m in enumerate(F.MODELS):
        for S in (1, 2, 4, 8, 16, 32):
            sd = 4.0 + 0.3 * np.log2(S) * (mi + 1)
            Z = ref[None] + rng.normal(0.5 * np.log2(S), sd, (32 // S, len(pid)))
            Z[rng.random(Z.shape) < 0.02] = np.nan
            banks[(m, S)] = F.snap(Z)
    return banks, F.snap(ref), pid


def test_pipeline_smoke(tmp_path, monkeypatch):
    art, out = tmp_path / "art", tmp_path / "out"
    art.mkdir(); out.mkdir()
    monkeypatch.setattr(F, "ART", art)
    monkeypatch.setattr(F, "OUT", out)
    monkeypatch.setattr(F, "load_split", _fake_split)
    monkeypatch.setattr(F, "require_committed", lambda p: None)
    monkeypatch.setattr(F, "sha", lambda p: "0")
    monkeypatch.setattr(F, "git", lambda *a: SimpleNamespace(stdout="", returncode=0))
    monkeypatch.setattr(F, "NBOOT_VAL", 5)
    monkeypatch.setattr(F, "NBOOT_GATE", 5)
    monkeypatch.setattr(F, "NBOOT_TEST", 5)
    monkeypatch.setattr(F, "NSUB", 20)
    (art / "candidate_grid.json").write_text(json.dumps({"fallback_constant_bpm": 73.143}))
    rng = np.random.default_rng(3)
    ups = np.arange(1000, 1000 + F.N_VAL_PATIENTS)
    subs = {str(n): [sorted(int(x) for x in rng.choice(ups, n, replace=False)) for _ in range(20)] for n in F.NCAL}
    (art / "calibration_subsets.json").write_text(json.dumps({"subsets": subs}))
    F.fullval_stage()
    fz = json.loads((art / "frozen_method.json").read_text())
    assert all(0.05 <= d <= 0.25 for d in fz["delta_near"].values())
    F.nested_stage()
    near = json.loads((art / "near_optimality.json").read_text())
    assert near["verdict_B32"]["verdict"] in ("STRONG", "PARTIAL", "FAILED")
    F.test_stage()
    lt = json.loads((art / "legacy_test_confirmation.json").read_text())
    assert lt["label"].startswith("Legacy-test confirmation")
    F.figure_stage()
    for f in ("validation_full_grid.csv", "fullval_reference.json", "nested_calibration_results.csv", "selection_frequency.csv",
              "regret_curves.csv", "paired_risk.csv", "bootstrap.json", "figure.png"):
        assert (art / f).exists(), f
