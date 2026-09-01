"""Static / synthetic validation for C2, run BEFORE any C2 training.

Protocol: docs/C2_COMPUTE_MATCHED_MULTISEED_INTERVAL_PREREGISTRATION.md (f5120f9).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation.c2_cohort import ATLAS_SALT, PER_STRATUM, SITES, atlas_cohort
from ppg2ecg.evaluation.hierarchical import N_REPLICATES, RNG_SEED, hierarchical_bootstrap

ROOT = Path(__file__).resolve().parents[1]
TRAINER = (ROOT / "src/ppg2ecg/training/train_a2.py").read_text()
SEEDS = (40, 41, 42, 43, 44)


# ---------------------------------------------------------------- compute matching
def test_round_structure_gives_14409_steps_not_14520():
    """batch_rounds truncates a round at the epoch boundary, so 66 rounds != 66*220."""
    n_train, batch, spr, rounds = 293271, 64, 220, 66
    per_epoch = -(-n_train // batch)
    assert per_epoch == 4583
    it, total, short = 0, 0, []
    for _ in range(rounds):
        take = min(spr, per_epoch - it)
        total += take
        if take != spr:
            short.append(take)
        it = 0 if it + take >= per_epoch else it + take
    assert total == 14409 and total != 66 * 220
    assert short == [183, 183, 183]


def test_step_structure_is_independent_of_arm_and_seed():
    """The truncation pattern depends only on window count, batch size and steps_per_round."""
    def steps(n_train, batch=64, spr=220, rounds=66):
        per_epoch = -(-n_train // batch); it = 0; tot = 0
        for _ in range(rounds):
            take = min(spr, per_epoch - it); tot += take
            it = 0 if it + take >= per_epoch else it + take
        return tot
    assert steps(293271) == steps(293271) == 14409          # deterministic, no seed/arm term


def test_trainer_counts_optimiser_steps():
    assert '"opt_steps": 0}' in TRAINER
    assert 'state["opt_steps"] += 1' in TRAINER
    assert '"opt_steps": state["opt_steps"]' in TRAINER
    assert '"opt_steps", "event"]' in TRAINER
    # the counter must increment exactly where opt.step() is called, once
    assert TRAINER.count('state["opt_steps"] += 1') == 1
    assert TRAINER.count("                opt.step()\n") == 1


# ---------------------------------------------------------------- visual atlas cohort
def test_atlas_cohort_is_metadata_only_and_deterministic():
    # the EXECUTABLE code (docstrings stripped) must not reference any model-derived quantity,
    # and the module must not import anything that could supply one
    import ast as _ast
    mod = ROOT / "src/ppg2ecg/evaluation/c2_cohort.py"
    tree = _ast.parse(mod.read_text())
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.Module, _ast.ClassDef)):
            node.body = [n for n in node.body
                         if not (isinstance(n, _ast.Expr) and isinstance(n.value, _ast.Constant)
                                 and isinstance(n.value.value, str))]
    code = _ast.unparse(tree).lower()
    for forbidden in ("f1", "morph", "corr", "rmse", "error", "predict", "model", "beat_level", "rpeak"):
        assert forbidden not in code, f"model-derived term {forbidden!r} in executable cohort code"
    imports = {n.module for n in _ast.walk(tree) if isinstance(n, _ast.ImportFrom) and n.module}
    imports |= {a.name for n in _ast.walk(tree) if isinstance(n, _ast.Import) for a in n.names}
    assert imports <= {"hashlib", "numpy", "__future__"}, imports
    assert ATLAS_SALT == "c2-visual-atlas-v1" and PER_STRATUM == 8
    rng = np.random.default_rng(0)
    sites = np.array(rng.choice(SITES, 400))
    wi = np.arange(400)
    a = atlas_cohort("an0", sites, wi)
    b = atlas_cohort("an0", sites, wi)
    for s in SITES:
        np.testing.assert_array_equal(a[s], b[s])
    assert atlas_cohort("k2s", sites, wi)["head"].tolist() != a["head"].tolist()   # subject-specific


def test_atlas_cohort_covers_eight_strata_of_eight_on_the_frozen_subset():
    total = 0
    for s in ("an0", "k2s"):
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        idx = ER.select_subset("x4-event-nfe-v2", s, len(d["x"]), 1024)
        c = atlas_cohort(s, d["site"][idx], d["window_index"][idx])
        assert set(c) == set(SITES)
        for site in SITES:
            assert c[site].size == PER_STRATUM
            assert np.all(np.asarray(d["site"][idx])[c[site]] == site)   # stratum purity
        total += sum(v.size for v in c.values())
    assert total == 64


# ---------------------------------------------------------------- hierarchical bootstrap
def test_hierarchical_bootstrap_reports_seed_level_summary():
    s = np.array(["an0"] * 60 + ["k2s"] * 60)
    eff = {k: np.full(120, 0.01 * (i + 1)) for i, k in enumerate(SEEDS)}
    r = hierarchical_bootstrap(eff, s, n_rep=200)
    assert r["n_seeds"] == 5 and r["n_positive"] == 5
    assert r["point"] == pytest.approx(0.03)                 # mean of 0.01..0.05
    assert r["median"] == pytest.approx(0.03)
    assert r["min"] == pytest.approx(0.01) and r["max"] == pytest.approx(0.05)
    assert set(r["seed_effects"]) == set(SEEDS)


def test_hierarchical_bootstrap_resamples_seeds_not_just_windows():
    """With per-seed constants, all interval width must come from the OUTER seed resample."""
    s = np.array(["an0"] * 50 + ["k2s"] * 50)
    eff = {k: np.full(100, v) for k, v in zip(SEEDS, (0.0, 0.0, 0.0, 0.0, 1.0))}
    r = hierarchical_bootstrap(eff, s, n_rep=800)
    assert r["hi"] - r["lo"] > 0.1, "a window-only bootstrap would give a zero-width interval here"
    assert r["verdict"] == "unresolved"


def test_hierarchical_bootstrap_is_deterministic_and_uses_frozen_settings():
    assert N_REPLICATES == 5000 and RNG_SEED == 20260902
    s = np.array(["an0"] * 20 + ["k2s"] * 20)
    g = np.random.default_rng(1)
    eff = {k: g.normal(size=40) for k in SEEDS}
    assert hierarchical_bootstrap(eff, s, n_rep=100) == hierarchical_bootstrap(eff, s, n_rep=100)


def test_hierarchical_bootstrap_rejects_misaligned_input():
    s = np.array(["an0"] * 10)
    with pytest.raises(ValueError):
        hierarchical_bootstrap({40: np.zeros(10), 41: np.zeros(11)}, s, n_rep=10)
    with pytest.raises(ValueError):
        hierarchical_bootstrap({40: np.zeros(11)}, s, n_rep=10)
    with pytest.raises(ValueError):
        hierarchical_bootstrap({}, s, n_rep=10)


# ---------------------------------------------------------------- preflight artefact
def test_c2_rng_control_if_present_passed_for_every_seed():
    p = ROOT / "artifacts/c2_compute_matched_multiseed/rng_control.json"
    if not p.exists():
        pytest.skip("C2 preflight not run yet")
    d = json.loads(p.read_text())
    assert d["pass"] is True
    for sd in SEEDS:
        for k in ("init_hash", "order_hash", "noise_hash", "banks_hash"):
            assert d["checks"][f"seed{sd}_identical_{k}"] is True
        assert d["checks"][f"seed{sd}_tr_all_differ"] is True
    for k in ("init_hash", "order_hash", "noise_hash"):
        assert d["checks"][f"seeds_differ_{k}"] is True


def test_c2_preflight_never_names_a_test_subject():
    src = (ROOT / "scripts/preflight_c2_rng_control.py").read_text()
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "kjd" not in body and "ssx" not in body
