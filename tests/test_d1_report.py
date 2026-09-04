"""D1 read-out tests — scripts/d1_report.py and the FIG 3 geometry of scripts/d1_figures.py.

Everything here runs against the synthetic evaluator tree of `tests/fixtures/make_fake_d1.py`: NO checkpoint,
NO GPU, NO real corpus and nothing under the repository's `outputs/`. The tree is built in `tmp_path` and
`d1_common.ROOT` is pointed at it, so `Corpus.processed_dir` / `manifest_path` / `out_dir` resolve inside the
fixture and the real data directories are never touched.

What is pinned here:
  * the report's row count on a known tree, and the `n_windows` / cap columns that make the evaluation
    population visible (the per-subject cap must never be invisible in a human-facing artefact);
  * the four skip paths — missing corpus directory, missing CSV, missing waveform npz, empty CSV — each of
    which must WARN and continue, never crash and never fabricate a row;
  * summary_by_nfe.csv agreeing with a recomputation from per_window_metrics.csv;
  * the loud failure on a k-fold manifest, the mandatory --corpus/--all choice, the documented INTERFACES
    note, and the FIG 3 annotation lane.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import d1_common as C  # noqa: E402
import d1_figures as F  # noqa: E402
import d1_report as R  # noqa: E402

from tests.fixtures.make_fake_d1 import make_fake_d1, patch_root  # noqa: E402


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _clean_warnings():
    R.WARNINGS.clear()
    yield
    R.WARNINGS.clear()


def build(tmp_path, monkeypatch, **kw) -> dict:
    """A fake evaluator tree with `d1_common.ROOT` pointed at it."""
    info = make_fake_d1(tmp_path / "tree", **kw)
    patch_root(monkeypatch, tmp_path / "tree")
    return info


def run_report(info, monkeypatch, tmp_path, *extra) -> tuple[int, str, list[dict]]:
    out = tmp_path / "bench"
    monkeypatch.setattr(sys, "argv", ["d1_report.py", "--eval-root", str(info["eval_root"]),
                                      "--out-root", str(out), *extra])
    rc = R.main()
    rows = list(csv.DictReader(open(out / "results_table.csv", newline="")))
    return rc, (out / "RESULTS.md").read_text(), rows


def expected_rows(info) -> int:
    """One TABLE 1 row per corpus per spec column at the headline NFE + one TABLE 2 row per headline metric
    at every other NFE. Columns outside PAPER_METRIC_SPEC (`mae_ew`) never become rows."""
    spec_cols = [c for c in info["per_window_columns"] + info["subject_level_columns"] if c in R.SPEC]
    heads = [m for m in R.HEADLINE if m in spec_cols]
    return len(info["corpora"]) * (len(spec_cols) + len(heads) * (len(info["nfes"]) - 1))


# ---------------------------------------------------------------------------------------------- row count
def test_report_row_count_from_a_known_fake_tree(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch, stored_windows=24)
    rc, md, rows = run_report(info, monkeypatch, tmp_path, "--all")
    assert rc == 0
    assert len(rows) == expected_rows(info) == 32
    assert {r["corpus"] for r in rows} == set(info["corpora"])
    assert all(r["metric_key"] in R.SPEC for r in rows)
    assert "mae_ew" not in {r["metric_key"] for r in rows}
    # the three corpora with no evaluator output are named as skipped, never silently dropped
    assert "**skipped (no evaluator output): wildppg, capnobase, vitaldb**" in md


# ---------------------------------------------------------------------------------------------- skip paths
def test_skip_missing_corpus_directory_warns_and_continues(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch)
    rc, md, rows = run_report(info, monkeypatch, tmp_path, "--corpus", "dalia", "--corpus", "wildppg")
    assert rc == 0
    assert {r["corpus"] for r in rows} == {"dalia"}
    assert any("wildppg: no evaluator output" in w for w in R.WARNINGS)
    assert "### Warnings raised while building this report" in md


def test_skip_missing_summary_csv_warns_and_continues(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch)
    (info["eval_dirs"]["bidmc"] / "summary_by_nfe.csv").unlink()
    rc, _md, rows = run_report(info, monkeypatch, tmp_path, "--all")
    assert rc == 0
    assert any("summary_by_nfe.csv missing" in w for w in R.WARNINGS)
    bidmc = [r for r in rows if r["corpus"] == "bidmc"]
    assert bidmc and {r["ci_source"] for r in bidmc} == {"recomputed here"}
    assert len(rows) == expected_rows(info)          # a missing CSV costs no row, it only changes the source


def test_skip_missing_waveform_npz_warns_and_continues(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch)
    (info["eval_dirs"]["bidmc"] / "waveforms_nfe1.npz").unlink()
    evs = {k: R.load_corpus(info["eval_root"], C.corpus(k)) for k in info["corpora"]}
    assert F.waveforms(evs["bidmc"], 1) is None
    assert any("waveforms_nfe1.npz missing" in w for w in R.WARNINGS)
    written = F.fig1_qualitative(evs, list(info["corpora"]), tmp_path, {"dalia": 1, "bidmc": 1})
    assert written, "FIG 1 must still be drawn from the corpora that do have waveforms"


def test_skip_empty_csv_warns_and_continues(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch)
    p = info["eval_dirs"]["bidmc"] / "per_subject_metrics.csv"
    p.write_text(p.read_text().splitlines()[0] + "\n")          # header only: no data rows
    (info["eval_dirs"]["bidmc"] / "eval_meta.json").unlink()
    rc, _md, rows = run_report(info, monkeypatch, tmp_path, "--all")
    assert rc == 0
    assert any("per_subject_metrics.csv holds no data rows" in w for w in R.WARNINGS)
    assert {r["corpus"] for r in rows} == {"dalia"}, "an empty table must not become rows"


# ---------------------------------------------------------------------------------- summary vs recomputation
@pytest.mark.parametrize("metric", ["mae", "rmse", "hr_abs_err"])
def test_summary_row_equals_recomputation_from_per_window(tmp_path, monkeypatch, metric):
    info = build(tmp_path, monkeypatch)
    ev = R.load_corpus(info["eval_root"], C.corpus("dalia"))
    from_file = R.stat(ev, metric, info["nfes"][0])
    from_windows = R.stat(dict(ev, summary={}), metric, info["nfes"][0])   # forces the recompute path
    assert from_file["source"] == "summary_by_nfe.csv"
    assert from_windows["source"] == "recomputed here"
    for k in ("macro", "lo", "hi", "pooled"):
        assert from_file[k] == pytest.approx(from_windows[k], rel=1e-9, abs=1e-12), k
    assert from_file["n_windows"] == from_windows["n_windows"] == info["n_test_windows"]


# ---------------------------------------------------------------------------------- the per-subject cap [MAJOR]
def test_cap_is_disclosed_under_table1_and_in_table3(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch, stored_windows=24, cap=1024)
    assert info["capped"]
    _rc, md, rows = run_report(info, monkeypatch, tmp_path, "--all")
    table1, table3 = md.split("### TABLE 2")[0], md.split("### TABLE 3")[1]
    assert "**Evaluation population — CAPPED.**" in table1, "the cap must be disclosed directly under TABLE 1"
    assert f"{info['n_test_windows']} of {info['split_test_windows']} test windows" in table1
    assert f"cap {info['cap']}/subject" in table1
    assert "`population_rule`" in table1 and "no outcome-dependent selection" in table1
    assert "Test windows evaluated" in table3 and "Cap (windows/subject)" in table3
    assert f"| {info['n_test_windows']} | {info['cap']} |" in table3
    assert all(int(r["n_windows"]) > 0 for r in rows)
    assert {int(r["n_windows"]) for r in rows if r["level"] == "window"} == {info["n_test_windows"]}


def test_cap_counts_fall_back_to_the_evaluator_csv_columns(tmp_path, monkeypatch):
    """With eval_meta.json gone, the realised count still comes from the evaluator's own CSV column — and the
    cap, which no CSV carries, stays None instead of being guessed."""
    info = build(tmp_path, monkeypatch, stored_windows=24, write_meta=False)
    ci = R.cap_info(R.load_corpus(info["eval_root"], C.corpus("dalia")))
    assert ci["n_eval"] == info["n_test_windows"]
    assert "n_test_windows_total" in ci["n_eval_source"]
    assert ci["n_available"] == info["split_test_windows"]
    assert ci["capped"] is True
    assert ci["cap"] is None and ci["cap_source"] is None


def test_no_cap_disclosure_when_the_population_is_complete(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch, stored_windows=None)      # every stored window was evaluated
    _rc, md, _rows = run_report(info, monkeypatch, tmp_path, "--all")
    assert "Evaluation population — CAPPED" not in md
    assert f"| {info['n_test_windows']} | {info['cap']} |" in md.split("### TABLE 3")[1]


def test_missing_cap_fields_warn_and_are_never_invented(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch, stored_windows=24, write_meta=False)
    _rc, md, _rows = run_report(info, monkeypatch, tmp_path, "--all")
    assert any("max_test_windows_per_subject" in w and "NOT inferred" in w for w in R.WARNINGS)
    table3 = md.split("### TABLE 3")[1]
    assert f"| {info['n_test_windows']} | ? |" in table3, "the realised count is known, the cap is not"
    assert "cap 1024" not in md


# ---------------------------------------------------------------------------------- k-fold manifest [MINOR 3]
def test_kfold_manifest_fails_loudly(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch)
    c = C.corpus("dalia")
    p = info["root"] / c.manifest
    one = json.loads(p.read_text())["splits"][0]
    p.write_text(json.dumps({"splits": [dict(one, fold=0), dict(one, fold=1)], "extra": {}}))
    with pytest.raises(SystemExit) as e:
        R.split_sizes(c, {})
    assert "2 splits" in str(e.value) and "fold" in str(e.value)


# ---------------------------------------------------------------------------------- --corpus / --all [MINOR 4]
@pytest.mark.parametrize("mod", [R, F], ids=["report", "figures"])
@pytest.mark.parametrize("flags", [[], ["--corpus", "dalia", "--all"]], ids=["neither", "both"])
def test_exactly_one_of_corpus_or_all_is_required(tmp_path, monkeypatch, mod, flags):
    info = build(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [f"{mod.__name__}.py", "--eval-root", str(info["eval_root"]),
                                      "--out-root", str(tmp_path / "bench"), *flags])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code != 0


def test_all_selects_every_benchmark_corpus(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch)
    _rc, md, _rows = run_report(info, monkeypatch, tmp_path, "--all")
    for key in C.BENCH_KEYS:
        assert key in md


# ---------------------------------------------------------------------------------- documented note [MINOR 5]
def test_interfaces_note_matches_the_real_corpus_dataclass():
    block = R.__doc__.split("INTERFACES")[1]
    fields = block.split("d1_common.Corpus fields")[1].split("d1_common.Corpus properties")[0]
    props = block.split("d1_common.Corpus properties")[1].split("splits.read_manifest")[0]
    def names(seg):
        return set(re.findall(r"[a-z][a-z0-9_]*", seg))

    assert names(fields) == {f.name for f in dataclasses.fields(C.Corpus)}
    assert names(props) == {n for n, v in vars(C.Corpus).items() if isinstance(v, property)}


# ---------------------------------------------------------------------------------- TABLE 1 grouping [MINOR 2]
def test_restatement_columns_are_grouped_and_labelled(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch)
    _rc, md, _rows = run_report(info, monkeypatch, tmp_path, "--all")
    table1 = md.split("### TABLE 2")[0]
    blocks = [b for b in table1.split("**TABLE 1 — ") if b.startswith("block") or b.startswith("restatement")]
    restatement = [b for b in blocks if b.startswith("restatement")]
    assert len(restatement) == 1, "the restatements must live in one block of their own"
    for key, of in R.RESTATEMENTS.items():
        header_of = [b for b in blocks if f"| {R.head_label(of)} |" in b]
        assert header_of and not any(f"| {R.head_label(key)} |" in b for b in header_of), \
            f"{key} must not sit next to the column it restates"
        assert f"| {R.head_label(key)} |" in restatement[0]
    assert "ALGEBRAIC RESTATEMENTS" in table1 and "not independent confirmations" in table1
    # the preamble must no longer claim the coincidence away
    assert "The pooled window mean is a separate column of `results_table.csv` and is never merged" not in table1


# ---------------------------------------------------------------------------------- FIG 3 geometry [MINOR 6]
def test_fig3_nfe50_annotation_is_clear_of_the_rule_and_the_spine(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch)
    evs = {k: R.load_corpus(info["eval_root"], C.corpus(k)) for k in info["corpora"]}
    grabbed: dict = {}
    monkeypatch.setattr(F, "save", lambda fig, out, stem, caption: grabbed.setdefault("fig", fig) and [])
    F.fig3_nfe_tradeoff(evs, list(info["corpora"]), tmp_path, [m for m in R.HEADLINE if m in R.SPEC])
    fig = grabbed["fig"]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    annotated = 0
    for ax in fig.axes:
        assert ax.get_xlim()[1] > R.PENGUIN_NFE, "the NFE-50 rule must not sit on the right spine"
        for t in ax.texts:
            if "NFE 50" not in t.get_text():
                continue
            annotated += 1
            bb, rule = t.get_window_extent(rend), ax.transData.transform((R.PENGUIN_NFE, 0))[0]
            assert bb.x0 > rule, "the label is struck through by the NFE-50 rule"
            assert bb.x1 < ax.get_window_extent().x1 - 1.0, "the label is clipped against the right spine"
    assert annotated == len([m for m in R.HEADLINE if m in R.SPEC])


# ---------------------------------------------------------------------------------- the fixture itself
def test_fixture_reproduces_the_restatement_identity(tmp_path, monkeypatch):
    """The fake tree is only useful if it has the property the report has to survive."""
    info = build(tmp_path, monkeypatch)
    ev = R.load_corpus(info["eval_root"], C.corpus("dalia"))
    for key, of in R.RESTATEMENTS.items():
        a, b = R.stat(ev, key, info["nfes"][0]), R.stat(ev, of, info["nfes"][0])
        assert a["macro"] == pytest.approx(b["macro"], rel=1e-12)
    assert R.stat(ev, "pooled_rmse", info["nfes"][0])["macro"] > R.stat(ev, "rmse", info["nfes"][0])["macro"]


def test_fixture_writes_nothing_outside_its_root(tmp_path, monkeypatch):
    info = build(tmp_path, monkeypatch)
    run_report(info, monkeypatch, tmp_path, "--all")
    assert not (ROOT / "outputs" / "d1_dalia_seed42").exists()
    assert set(p.name for p in (tmp_path / "tree").iterdir()) == {"data", "outputs"}
