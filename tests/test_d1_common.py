"""D1 multi-dataset benchmark: the split rule, the corpus-identity pin, the test-window cap and the DaLiA builder.

Every test here is a regression test for a defect that was found in the D1 drivers, so each one names the thing it
would have caught. Nothing in this file trains, loads a checkpoint or produces a real-data metric: the corpora are
read only for their MANIFEST bookkeeping (subject ids, window counts, per-file hashes).
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import d1_common as C  # noqa: E402

D1_KEYS = ("dalia", "bidmc", "capnobase", "vitaldb")


def load_script(name: str, stem: str | None = None):
    spec = importlib.util.spec_from_file_location(stem or name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def built(key: str) -> bool:
    return (C.corpus(key).processed_dir / "MANIFEST.json").exists()


# ----------------------------------------------------------------------------------------------------------------------
# §5 split rule
# ----------------------------------------------------------------------------------------------------------------------
def test_make_d1_split_is_deterministic_and_independent_of_input_order():
    subs = [f"case_{i:05d}" for i in range(1, 38)]
    ref = C.make_d1_split(subs, "t")
    assert C.make_d1_split(subs, "t") == ref                       # determinism
    for shuffled in (list(reversed(subs)), sorted(subs), list(np.random.default_rng(7).permutation(subs))):
        got = C.make_d1_split([str(s) for s in shuffled], "t")
        assert {k: got[k] for k in ("train", "val", "test")} == {k: ref[k] for k in ("train", "val", "test")}


def test_make_d1_split_partitions_exactly_and_takes_the_train_ceiling():
    for n in range(3, 60):
        subs = [f"s{i}" for i in range(n)]
        sp = C.make_d1_split(subs, "t")
        tr, va, te = (set(sp[k]) for k in ("train", "val", "test"))
        assert tr | va | te == set(subs)
        assert len(tr) + len(va) + len(te) == n and not (tr & va) and not (tr & te) and not (va & te)
        assert len(tr) == min(-(-70 * n // 100), n - 2)
        assert sp["n_subjects"] == n and sp["seed"] == C.SEED


def test_make_d1_split_raises_below_three_and_keeps_val_and_test_non_empty_from_three_to_ten():
    for n in (0, 1, 2):
        with pytest.raises(ValueError, match="cannot give a non-empty"):
            C.make_d1_split([f"s{i}" for i in range(n)], "t")
    for n in range(3, 11):
        sp = C.make_d1_split([f"s{i}" for i in range(n)], "t")
        assert len(sp["val"]) >= 1 and len(sp["test"]) >= 1 and len(sp["train"]) >= 1


def test_assert_no_forbidden_subjects_fires_for_kjd_or_ssx_in_any_of_the_three_lists():
    """prereg D1 §4: kjd/ssx are never LOADED, so TEST is forbidden too — not just train and val."""
    for field in ("train", "val", "test"):
        for bad in ("kjd", "ssx"):
            split = {"train": ["a"], "val": ["b"], "test": ["c"]}
            split[field] = [*split[field], bad]
            with pytest.raises(C.ForbiddenSubjectError, match=bad):
                C.assert_no_forbidden_subjects(split, "unit")
    C.assert_no_forbidden_subjects({"train": ["a"], "val": ["b"], "test": ["c"]}, "unit")


def test_corpus_subjects_excludes_the_never_loaded_wildppg_holdouts():
    """The exclusion is at the POOL, so no D1 split can place kjd/ssx anywhere, including test."""
    if not built("wildppg"):
        pytest.skip("wildppg corpus not built")
    c = C.corpus("wildppg")
    subs = C.corpus_subjects(c)
    stored = set(C.corpus_manifest(c)["files"])
    assert {"kjd", "ssx"} <= stored, "fixture assumption: both hold-outs are stored in the corpus"
    assert not ({"kjd", "ssx"} & set(subs))
    assert len(subs) == len(stored) - 2


@pytest.mark.parametrize("key", D1_KEYS)
def test_committed_split_manifests_regenerate_from_the_current_corpora(key):
    """The committed partition must be exactly what the frozen §5 rule produces from the corpus that is on disk."""
    if not built(key):
        pytest.skip(f"{key} corpus not built")
    c = C.corpus(key)
    committed, _ = C.read_split_manifest(c.manifest_path)
    regenerated = C.make_d1_split(C.corpus_subjects(c), c.key)
    for k in ("train", "val", "test"):
        assert list(committed[k]) == list(regenerated[k]), f"{key}: {k} partition moved"
    assert committed["n_subjects"] == len(C.corpus_subjects(c)) == regenerated["n_subjects"]


# ----------------------------------------------------------------------------------------------------------------------
# Corpus-identity pin  (the manifests carried STALE MANIFEST.json hashes and nothing checked them)
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", D1_KEYS)
def test_committed_pin_matches_the_corpus_on_disk(key):
    if not built(key):
        pytest.skip(f"{key} corpus not built")
    c = C.corpus(key)
    split, extra = C.read_split_manifest(c.manifest_path)
    assert extra["corpus_identity_sha256"] == C.corpus_identity_sha256(c)
    C.assert_corpus_identity(c, split, extra)


def test_corpus_identity_hash_ignores_the_built_timestamp_but_not_the_windows(tmp_path):
    files = {"S1": {"n_windows": 3, "sha256": "aa"}, "S2": {"n_windows": 5, "sha256": "bb"}}
    stub = SimpleNamespace(processed_dir=tmp_path)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"built": "2026-01-01T00:00:00", "files": files}))
    h0 = C.corpus_identity_sha256(stub)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"built": "2026-09-04T12:00:00", "files": files}))
    assert C.corpus_identity_sha256(stub) == h0                       # a rebuild alone must not move the pin
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"built": "x", "files": files | {"S2": {"n_windows": 6, "sha256": "bb"}}}))
    assert C.corpus_identity_sha256(stub) != h0                       # a changed window count must
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"built": "x", "files": files | {"S2": {"n_windows": 5, "sha256": "cc"}}}))
    assert C.corpus_identity_sha256(stub) != h0                       # a changed array must


@pytest.mark.parametrize("key", D1_KEYS)
def test_tampering_with_a_manifest_makes_ensure_split_manifest_fail_loudly(key, tmp_path, monkeypatch):
    """A stale pin, a missing pin and an incomplete subject list are all HARD failures, never warnings."""
    if not built(key):
        pytest.skip(f"{key} corpus not built")
    c = C.corpus(key)
    payload = json.loads(c.manifest_path.read_text())
    tampered = tmp_path / c.manifest_path.name
    monkeypatch.setattr(type(c), "manifest_path", property(lambda self: tampered))

    payload["extra"]["corpus_identity_sha256"] = "0" * 64
    tampered.write_text(json.dumps(payload))
    with pytest.raises(C.CorpusIdentityError, match="corpus identity changed"):
        C.ensure_split_manifest(c, write=False)

    payload["extra"].pop("corpus_identity_sha256")
    tampered.write_text(json.dumps(payload))
    with pytest.raises(C.CorpusIdentityError, match="no corpus_identity_sha256 pin"):
        C.ensure_split_manifest(c, write=False)

    payload["extra"]["corpus_identity_sha256"] = C.corpus_identity_sha256(c)
    payload["splits"][0]["test"] = payload["splits"][0]["test"][1:]
    tampered.write_text(json.dumps(payload))
    with pytest.raises(C.CorpusIdentityError, match="split covers"):
        C.ensure_split_manifest(c, write=False)


def test_refresh_refuses_to_move_a_subject_between_partitions(tmp_path, monkeypatch):
    if not built("dalia"):
        pytest.skip("dalia corpus not built")
    c = C.corpus("dalia")
    payload = json.loads(c.manifest_path.read_text())
    payload["splits"][0]["train"], payload["splits"][0]["val"] = payload["splits"][0]["val"], payload["splits"][0]["train"]
    moved = tmp_path / c.manifest_path.name
    moved.write_text(json.dumps(payload))
    monkeypatch.setattr(type(c), "manifest_path", property(lambda self: moved))
    with pytest.raises(C.SplitPartitionChanged, match="would MOVE subjects"):
        C.ensure_split_manifest(c, write=True, refresh=True)
    assert json.loads(moved.read_text()) == payload  # nothing was written


# ----------------------------------------------------------------------------------------------------------------------
# Test-window cap  (the old rule `x[::-(-n//cap)]` HALVED a subject instead of trimming it)
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, 10, 1023, 1024, 1025, 2048, 2049])
@pytest.mark.parametrize("cap", [0, 1024])
def test_capped_indices_returns_exactly_min_n_cap_unique_ascending_indices(n, cap):
    idx = C.capped_indices(n, cap)
    assert idx.ndim == 1 and idx.dtype.kind == "i"
    assert len(idx) == (n if cap <= 0 else min(n, cap))
    assert len(set(idx.tolist())) == len(idx)
    assert np.all(np.diff(idx) > 0) if len(idx) > 1 else True
    assert (idx >= 0).all() and (idx < max(n, 1)).all() if n else len(idx) == 0
    assert C.capped_indices(n, cap).tolist() == idx.tolist()  # deterministic


def test_capped_indices_does_not_halve_a_subject_just_above_the_cap():
    """PPG-DaLiA S9 = 1070 and S14 = 1119 windows: the integer-stride rule gave stride 2 and dropped ~50%."""
    for n in (1070, 1119):
        assert len(C.capped_indices(n, 1024)) == 1024
        assert len(np.arange(n)[:: -(-n // 1024)]) < 600  # what the old rule did


def test_load_test_keeps_exactly_min_n_cap_windows_per_subject(tmp_path):
    ev = load_script("d1_evaluate")
    for s, n in (("A", 1070), ("B", 30)):
        np.savez(tmp_path / f"{s}.npz", x=np.zeros((n, 8), np.float32), y=np.zeros((n, 8), np.float32),
                 window_index=np.arange(n, dtype=np.int32))
    stub = SimpleNamespace(processed_dir=tmp_path)
    x, _y, sid, widx = ev.load_test(stub, ["A", "B"], 1024)
    assert len(x) == 1024 + 30 and int((sid == "A").sum()) == 1024 and int((sid == "B").sum()) == 30
    assert np.all(np.diff(widx[sid == "A"]) > 0)
    x0, _, sid0, _ = ev.load_test(stub, ["A", "B"], 0)
    assert len(x0) == 1100 and int((sid0 == "A").sum()) == 1070


def test_evaluator_default_is_no_cap():
    """Default 1024 silently subsampled every corpus with >1024 windows in a test subject, including PPG-DaLiA."""
    args = load_script("d1_evaluate").build_parser().parse_args(["--corpus", "bidmc"])
    assert args.max_test_windows_per_subject == 0


# ----------------------------------------------------------------------------------------------------------------------
# --metric-chunk must not silently keep the LAST chunk's set-level scalars
# ----------------------------------------------------------------------------------------------------------------------
def test_metric_chunk_refuses_to_emit_set_level_columns():
    ev = load_script("d1_evaluate")
    ev.assert_chunking_has_no_set_level(0, {"fid_default_features", "prd"})       # no chunking: allowed
    ev.assert_chunking_has_no_set_level(256, set())                               # chunking, nothing set-level: allowed
    with pytest.raises(ValueError, match="fid_default_features") as e:
        ev.assert_chunking_has_no_set_level(256, {"fid_default_features", "kid_default"})
    assert "kid_default" in str(e.value) and "without --metric-chunk" in str(e.value)


# ----------------------------------------------------------------------------------------------------------------------
# The waveform npz must be a SUBSET of the scored population (FIG 1 must describe rows that exist in the table)
# ----------------------------------------------------------------------------------------------------------------------
def test_waveform_rows_index_into_the_scored_population():
    ev = load_script("d1_evaluate")
    sid = np.asarray(["A"] * 5 + ["B"] * 3 + ["C"] * 20)
    rows = ev.waveform_rows(sid, ["A", "B", "C"], 4)
    assert rows.tolist() == [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
    assert sid[rows].tolist() == ["A"] * 4 + ["B"] * 3 + ["C"] * 4
    assert (rows < len(sid)).all() and len(set(rows.tolist())) == len(rows)
    assert ev.waveform_rows(np.asarray([], dtype="<U1"), [], 4).tolist() == []


# ----------------------------------------------------------------------------------------------------------------------
# scripts/build_processed_dalia.py
# ----------------------------------------------------------------------------------------------------------------------
def synth_window(rng, n=1024):
    w = rng.standard_normal(n)
    w = 2 * (w - w.min()) / (w.max() - w.min()) - 1.0          # exactly [-1, 1], as the upstream min-max leaves it
    return w


def write_dalia_pickles(src: Path, n_windows: int, drop_at=()):
    """15 upstream-shaped pickles; every index in `drop_at` is made constant so the frozen drop rule removes it."""
    from ppg2ecg.data.dalia import SUBJECTS
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, _s in enumerate(SUBJECTS):
        x = np.stack([synth_window(rng) for _ in range(n_windows)])
        y = np.stack([synth_window(rng) for _ in range(n_windows)])
        for j in drop_at:
            x[j] = 0.0                                          # std == 0 -> dropped
        with open(src / f"subject{i}.pkl", "wb") as fh:
            pickle.dump({"x_data": x, "y_data": y}, fh)


def test_builder_preserves_the_recording_window_index_when_a_window_is_dropped(tmp_path, monkeypatch):
    """window_index must be the position in the RECORDING, not a renumbering over the survivors."""
    src, out = tmp_path / "src", tmp_path / "out"
    write_dalia_pickles(src, n_windows=6, drop_at=(2, 3))
    mod = load_script("build_processed_dalia")
    monkeypatch.setattr(sys, "argv", ["build_processed_dalia.py", "--src", str(src), "--out", str(out)])
    mod.main()
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["total_dropped"] == 2 * 15 and man["total_windows"] == 4 * 15
    for s in ("S1", "S7", "S15"):
        z = np.load(out / f"{s}.npz")
        assert z["window_index"].tolist() == [0, 1, 4, 5]          # the gap at 2/3 is preserved
        assert z["window_start_s"].tolist() == [0, 8, 32, 40]      # and so is the true recording offset
        assert z["window_index"].tolist() != list(range(4))


def test_builder_accepts_an_out_directory_outside_the_repo(tmp_path, monkeypatch):
    """`str(p.relative_to(ROOT))` raised ValueError for any --out that is not under ROOT."""
    src, out = tmp_path / "src", tmp_path / "out"
    write_dalia_pickles(src, n_windows=2)
    mod = load_script("build_processed_dalia")
    assert not out.is_relative_to(mod.ROOT)
    monkeypatch.setattr(sys, "argv", ["build_processed_dalia.py", "--src", str(src), "--out", str(out)])
    mod.main()
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["files"]["S1"]["path"] == str(out / "S1.npz") and man["total_windows"] == 2 * 15
