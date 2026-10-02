"""DP2 Stage A tests (docs/DP2_LOCAL_DATASET_AUDIT.md): the local dataset audit is metadata-only, every dataset gets exactly
one eligibility label, and the frozen primary-selection rule. No waveform is loaded here."""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/dp2_external"


@pytest.fixture(scope="module")
def au():
    sys.path.insert(0, str(ROOT / "scripts"))
    import dp2_dataset_audit
    return dp2_dataset_audit


def test_audit_never_loads_full_waveforms(au):
    src = inspect.getsource(au)
    for bad in ("loadmat(", "pickle", "read_pickle", "rdrecord", "rdsamp", "read_csv(", ".dat\"", "torch.load"):
        assert bad not in src, bad
    assert all("mmap_mode=\"r\"" in ln for ln in src.splitlines() if "np.load(" in ln)
    assert "h5py.File(p, \"r\")" in src and "f[p.stem].shape" in src                          # HDF5: shapes only
    assert re.search(r"VitalDB/cases\"\)\.glob", src) and "np.load(RAW / \"VitalDB" not in src  # VitalDB arrays never opened


def test_npy_probe_is_memory_mapped(au, tmp_path, monkeypatch):
    p = tmp_path / "x.npy"
    np.save(p, np.zeros((30, 3750)))
    seen = {}
    real = np.load
    monkeypatch.setattr(au.np, "load", lambda f, **k: (seen.update(k), real(f, **k))[1])
    assert au.npy_shape(p) == ([30, 3750], "float64") and seen == {"mmap_mode": "r"}


def test_excluded_roots_and_no_old_test(au):
    assert au.EXCLUDED_ROOTS == ("/home/kwy00/sci", "/home/kwy00/taeho")
    src = inspect.getsource(au)
    for ln in src.splitlines():                                                            # named only as exclusions
        if "/home/kwy00/sci" in ln or "/home/kwy00/taeho" in ln:
            assert not any(k in ln for k in ("walk", "glob", "open(", "Path(", "iterdir"))
    assert "load_test" not in src and "load_arch" not in src


def test_exactly_one_label_per_dataset():
    inv = json.loads((ART / "dataset_inventory.json").read_text())["datasets"]
    el = json.loads((ART / "dataset_eligibility.json").read_text())["datasets"]
    names = [d["dataset_name"] for d in inv]
    assert len(names) == len(set(names)) and set(names) == set(el)
    labels = {"FRESH-PRIMARY-CANDIDATE", "FRESH-SECONDARY-CANDIDATE", "CONTAMINATED", "INELIGIBLE", "PROVENANCE-UNCERTAIN"}
    assert all(e["label"] in labels for e in el.values())
    for e in el.values():
        if e["label"] == "CONTAMINATED":
            assert e["evidence"] and any(u.startswith("USED FOR") for u in e["prior_use"])     # contamination only with evidence


def test_structural_ineligibility_consistent():
    inv = {d["dataset_name"]: d for d in json.loads((ART / "dataset_inventory.json").read_text())["datasets"]}
    el = json.loads((ART / "dataset_eligibility.json").read_text())["datasets"]
    for n, d in inv.items():
        if d["PPG_present"] == "NO" or d["ECG_present"] == "NO" or d["subject_identifier_available"] == "NO":
            assert el[n]["label"] == "INELIGIBLE", n


def test_primary_selection_rule(au):
    sel = json.loads((ART / "primary_dataset_selection.json").read_text())
    assert sel["selected_dataset"] is None and sel["verdict"] == "DP2 NO ELIGIBLE PRIMARY DATASET"
    assert au.select_primary(au.ELIGIBILITY, {k: {"subjects": 0} for k in au.ELIGIBILITY}) is None
    fake = {"A": ("FRESH-PRIMARY-CANDIDATE", ""), "B": ("FRESH-PRIMARY-CANDIDATE", ""), "C": ("CONTAMINATED", "")}
    assert au.select_primary(fake, {"A": {"subjects": 250}, "B": {"subjects": 900}, "C": {"subjects": 5000}}) == "B"
