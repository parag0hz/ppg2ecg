"""D1 report — docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md sections 7.1-7.4 and 11.

Every D1 number is read from the frozen evaluator output written by `scripts/d1_evaluate.py`
(`<eval-root>/d1_<corpus>_seed42/eval/{per_window_metrics,per_subject_metrics,summary_by_nfe}.csv` and
`eval_meta.json`). Nothing is recomputed from waveforms, no generator is run and no result is hardcoded:
the metric identity, orientation and attribution come from `PAPER_METRIC_SPEC`, the corpus identity from
`scripts/d1_common.py`, and the `mean [95 % CI]` cells from the evaluator's own subject-clustered
bootstrap rows (recomputed with `d1_common.subject_cluster_bootstrap` only when a summary row is absent,
so a figure and a table can never disagree about a number).

The only literals are the PUBLISHED literature table of TABLE 4 — each row carries its citation and a
mandatory non-comparability note, and no published number ever enters `results_table.csv`.

This module also owns the D1 read-out layer; `scripts/d1_figures.py` imports it rather than re-reading
the evaluator tree.

INTERFACES this module depends on (checked against the real definitions by tests/test_d1_report.py, so an
upstream rename fails a test instead of silently mis-documenting this file):

  d1_common.Corpus fields      key, name, processed, manifest, fs_ppg, fs_ecg, citation, target, trains,
                               checkpoint, notes
  d1_common.Corpus properties  included, out_dir, exp_name, processed_dir, manifest_path, checkpoint_path,
                               eval_dir
  splits.read_manifest(path)   -> a LIST of split dicts (one per fold); D1 manifests hold exactly one and
                               `one_split` refuses to guess when they do not
  evaluator artefacts read     per_window_metrics.csv, per_subject_metrics.csv, summary_by_nfe.csv,
                               eval_meta.json (keys used: nfe_grid, timings, split_sizes,
                               max_test_windows_per_subject, n_test_windows)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import d1_common as C  # noqa: E402

from ppg2ecg.data.splits import read_manifest  # noqa: E402
from ppg2ecg.evaluation.paper_metrics import PAPER_METRIC_SPEC  # noqa: E402

PREREG = ROOT / "docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md"
SPEC = {s.key: s for s in PAPER_METRIC_SPEC}
HEADLINE_NFE, PENGUIN_NFE = 1, 50                 # prereg section 6: one-step is the claim, NFE 50 is PENGUIN's point
HEADLINE = ("rmse", "pcc", "rpeak_f1_50ms", "hr_abs_err", "prd_meansub")
ARROW = {"lower_is_better": "↓", "higher_is_better": "↑", "neutral": "·"}
#: axis/table units. Amplitude metrics live on the frozen per-window z-score -> [-1,1] scale, so they are
#: dimensionless but NOT unitless in the physical sense: they are normalised-amplitude units, never mV.
UNITS = {"snr_db": "dB", "prd_raw": "%", "prd_meansub": "%",
         "mae": "norm. amplitude", "mse": "norm. amplitude^2", "rmse": "norm. amplitude",
         "pooled_mae": "norm. amplitude", "pooled_rmse": "norm. amplitude", "mean_window_rmse": "norm. amplitude",
         "qrs_region_rmse": "norm. amplitude", "non_qrs_rmse": "norm. amplitude",
         "discrete_frechet": "norm. amplitude", "dtw": "norm. amplitude, accumulated",
         "kanflow_fd": "squared feature distance", "fid_default_features": "squared feature distance",
         "fid_ecgfounder": "squared feature distance"}


def units(key: str) -> str:
    """Unit string for an axis label or a table header; `dimensionless` for ratios, F1 and correlations."""
    if key.endswith("_ms"):
        return "ms"
    if key.endswith("_bpm") or key in ("hr_ref", "hr_pred", "hr_abs_err"):
        return "bpm"
    if key.endswith("_mmhg"):
        return "mmHg"
    if key.startswith("n_"):
        return "count"
    return UNITS.get(key, "dimensionless")
BLOCK = 6                                         # TABLE 1 metric columns per legibility block
#: dataset-level keys whose per-subject value is BY CONSTRUCTION the mean of the named per-window column over the
#: same subject: with equal-length windows `pooled_mae` IS that subject's mean per-window MAE and
#: `mean_window_rmse` IS its mean per-window RMSE, so the subject-macro means printed in TABLE 1 are algebraically
#: identical — a restatement, never an independent confirmation. `pooled_rmse` is deliberately NOT here: its
#: square root is taken after pooling, so it differs from `rmse` by the Jensen gap.
RESTATEMENTS: dict[str, str] = {"pooled_mae": "mae", "mean_window_rmse": "rmse"}
EFF_KEYS = (("gen_samples_per_s", "samples/s ↑"), ("gen_wall_clock_s", "wall-clock (s) ↓"),
            ("bench_latency_ms_median", "latency (ms, batch) ↓"), ("peak_mem_MiB", "peak mem (MiB) ↓"))

# ---------------------------------------------------------------- literature constants (TABLE 4 only)
PAPER_PROTOCOL = {
    "PENGUIN":    {"window": "8 s (paper) / 4 s as shipped", "band": "PPG 0.5-4 Hz", "nfe": "50", "params": "62.53 M",
                   "split": "13/1/1 single hold-out, filesystem-dependent test subject (docs/PENGUIN_AUDIT.md)"},
    "KANFlow":    {"window": "4 s (512 samples @128 Hz)", "band": "PPG band not stated in the frozen spec", "nfe": "not stated",
                   "params": "not stated", "split": "not stated; NeuroKit2 SQI screening applied before metrics"},
    "PPGFlowECG": {"window": "10 s (1280 samples @128 Hz)", "band": "PPG 0.5-8 Hz", "nfe": "not stated", "params": "not stated",
                   "split": "not stated; PPG template SQI intersected with nk.ecg_quality(zhao2018)"},
}
OURS_PROTOCOL = {"window": "8 s (1024 samples @128 Hz)", "band": "PPG 0.5-4 Hz (PENGUIN-inherited; the narrowest published band)",
                 "params": "4,568,707", "split": "subject-level 70/15/15, seed 42, per corpus"}
PUBLISHED = (
    # (paper, their dataset, our corpus key or None, metric key, value or None, citation, row-specific note)
    ("PENGUIN", "PPG-DaLiA", "dalia", "hr_abs_err", 15.64, "arXiv:2602.03858 Table 1, via docs/PENGUIN_AUDIT.md row 38",
     "their 15.64 rests on ONE test subject and the shipped HR metric is 2x time-compressed at "
     "segment_len=4 (PENGUIN_AUDIT rows 34-35)"),
    ("PENGUIN", "WildPPG", "wildppg", "hr_abs_err", 12.97, "arXiv:2602.03858 Table 1, via docs/PENGUIN_AUDIT.md row 38",
     "same shipped-metric caveats as the DaLiA row"),
    ("PPGFlowECG", "BIDMC", "bidmc", "fid_ecgfounder", 54.22, "PPGFlowECG (2025), per-dataset FID",
     "we do not compute FID on ECGFounder at all (no checkpoint); our fid_default_features is a SURROGATE "
     "feature space and is not comparable to any published FID"),
    ("PPGFlowECG", "VitalDB", "vitaldb", "fid_ecgfounder", 27.09, "PPGFlowECG (2025), per-dataset FID",
     "we do not compute FID on ECGFounder at all (no checkpoint); our fid_default_features is a SURROGATE "
     "feature space and is not comparable to any published FID"),
    ("PPGFlowECG", "MIMIC-AFib", None, "fid_ecgfounder", 37.69, "PPGFlowECG (2025), per-dataset FID",
     "MIMIC-AFib is not a D1 corpus, and its cohort has five irreconcilable definitions (PREPROCESSING_CONVENTIONS_SURVEY 3.5)"),
    ("PPGFlowECG", "MCMED", None, "fid_ecgfounder", 12.84, "PPGFlowECG (2025), per-dataset FID", "MCMED is not a D1 corpus"),
    ("PPGFlowECG", "not stated in the frozen spec", None, "hr_abs_err", 1.80, "PPGFlowECG (2025) Fig. 3a",
     "reported against 2.16 bpm read directly from PPG; the dataset behind Fig. 3a is not recorded in the "
     "frozen metric spec, so no corpus may be attached"),
    ("KANFlow", "BIDMC", "bidmc", "hr_abs_err", None, "KANFlow Table III",
     "per-dataset value absent from the frozen metric spec (only the 0.65-12.18 bpm range is recorded); "
     "left empty rather than interpolated"),
    ("KANFlow", "CapnoBase", "capnobase", "hr_abs_err", None, "KANFlow Table III",
     "per-dataset value absent from the frozen metric spec; left empty rather than interpolated"),
    ("KANFlow", "VitalDB", "vitaldb", "hr_abs_err", None, "KANFlow Table III",
     "per-dataset value absent from the frozen metric spec; left empty rather than interpolated"),
    ("KANFlow", "BIDMC", "bidmc", "rr_mae_ms", None, "KANFlow Table III",
     "per-dataset value absent from the frozen metric spec (only the 13.4-46.9 ms range is recorded); "
     "left empty rather than interpolated"),
    ("KANFlow", "BIDMC", "bidmc", "kanflow_fd", None, "KANFlow Table II",
     "KANFlow's FD uses PCA<=32 with 5-trial averaging below 3000 test segments and the raw 512-dim space "
     "above it; KANFlow itself states FD must be read WITHIN one dataset only"),
)
WARNINGS: list[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"[warn] {msg}", flush=True)


# ---------------------------------------------------------------- io primitives
def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, newline=""))) if path.exists() else []


def fnum(row: dict, key: str) -> float:
    v = row.get(key, "")
    return np.nan if v in ("", None, "nan", "NaN", "None") else float(v)


def inum(row: dict, key: str) -> int | None:
    v = fnum(row, key)
    return None if not np.isfinite(v) else int(v)


# ---------------------------------------------------------------- evaluator tree
def eval_dir(eval_root: Path, c: C.Corpus) -> Path:
    return eval_root / c.exp_name / "eval"


def load_corpus(eval_root: Path, c: C.Corpus) -> dict | None:
    """Everything the tables and the figures need for one corpus, or None when the evaluator has not run."""
    d = eval_dir(eval_root, c)
    if not (d / "per_subject_metrics.csv").exists():
        return None
    sub = read_csv(d / "per_subject_metrics.csv")
    if not sub:
        warn(f"{c.key}: per_subject_metrics.csv holds no data rows (header only) — corpus skipped; "
             "an empty evaluator table is never turned into a row")
        return None
    summ = read_csv(d / "summary_by_nfe.csv")
    meta_p = d / "eval_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if not meta:
        warn(f"{c.key}: eval_meta.json missing — corpus descriptors and timings fall back to the CSVs")
    if not summ:
        warn(f"{c.key}: summary_by_nfe.csv missing — CIs are recomputed here with d1_common.subject_cluster_bootstrap")
    nfes = sorted({int(float(r["nfe"])) for r in sub})
    cols = [k for k in (sub[0] if sub else {}) if k not in ("corpus", "subject", "nfe", "n_windows")]
    # per_window_metrics.csv is NOT read here: at VitalDB scale it is hundreds of thousands of rows and the
    # report never needs it while the evaluator's summary rows exist. `per_window` / `window_values` read it lazily.
    return {"key": c.key, "corpus": c, "dir": d, "per_subject": sub, "per_window_path": d / "per_window_metrics.csv",
            "per_window_cache": None, "meta": meta, "nfes": nfes, "columns": cols,
            "summary": {(int(float(r["nfe"])), r["metric"]): r for r in summ},
            "kind": {r["metric"]: r.get("kind", "") for r in summ}}


def per_window(ev: dict) -> list[dict]:
    """The per-window rows, read and cached on first use only."""
    if ev["per_window_cache"] is None:
        ev["per_window_cache"] = read_csv(ev["per_window_path"])
    return ev["per_window_cache"]


def per_window_header(ev: dict) -> list[str]:
    """Column names of per_window_metrics.csv without reading the body."""
    if ev["per_window_cache"] is not None:
        return list(ev["per_window_cache"][0]) if ev["per_window_cache"] else []
    if not ev["per_window_path"].exists():
        return []
    with open(ev["per_window_path"], newline="") as fh:
        return next(csv.reader(fh), [])


def has(ev: dict, metric: str) -> bool:
    return metric in ev["columns"]


def stat(ev: dict, metric: str, nfe: int) -> dict:
    """`mean [95 % CI]` for one metric at one NFE — the evaluator's own bootstrap row whenever it exists."""
    row = ev["summary"].get((nfe, metric))
    if row is not None:
        return {"macro": fnum(row, "subject_macro_mean"), "lo": fnum(row, "macro_ci_lo"), "hi": fnum(row, "macro_ci_hi"),
                "pooled": fnum(row, "pooled_window_mean"), "n_subjects": inum(row, "n_subjects"),
                "n_windows": inum(row, "n_windows"), "kind": row.get("kind", ""), "source": "summary_by_nfe.csv"}
    if not has(ev, metric):
        return {"macro": np.nan, "lo": np.nan, "hi": np.nan, "pooled": np.nan, "n_subjects": None,
                "n_windows": None, "kind": "", "source": "absent"}
    # a per-window column is bootstrapped over windows; a subject-level column exists only in the subject file
    src = per_window(ev) if metric in per_window_header(ev) else ev["per_subject"]
    rows = [r for r in src if int(float(r["nfe"])) == nfe]
    b = C.subject_cluster_bootstrap(np.asarray([fnum(r, metric) for r in rows], dtype=np.float64),
                                    np.asarray([r["subject"] for r in rows]))
    return {"macro": b["subject_macro_mean"], "lo": b["macro_ci_lo"], "hi": b["macro_ci_hi"],
            "pooled": b["pooled_window_mean"], "n_subjects": b["n_subjects"], "n_windows": b["n_windows"],
            "kind": "recomputed", "source": "recomputed here"}


def subject_values(ev: dict, metric: str, nfe: int) -> tuple[np.ndarray, list[str]]:
    """Per-subject values in natural subject order — the population behind the box/strip plot of FIG 5."""
    rows = [r for r in ev["per_subject"] if int(float(r["nfe"])) == nfe]
    rows.sort(key=lambda r: C.natural_key(r["subject"]))
    return np.asarray([fnum(r, metric) for r in rows], dtype=np.float64), [r["subject"] for r in rows]


def window_values(ev: dict, *metrics: str, nfe: int) -> list[np.ndarray]:
    """Stream the named per-window columns at one NFE without materialising the whole file (VitalDB scale)."""
    head = per_window_header(ev)
    if not head or any(m not in head for m in metrics) or "nfe" not in head:
        return [np.zeros(0) for _ in metrics]
    infe, idx = head.index("nfe"), [head.index(m) for m in metrics]
    out = [[] for _ in metrics]
    with open(ev["per_window_path"], newline="") as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for row in rd:
            if int(float(row[infe])) != nfe:
                continue
            for o, i in zip(out, idx):
                v = row[i]
                o.append(np.nan if v in ("", "nan", "NaN", "None") else float(v))
    return [np.asarray(o, dtype=np.float64) for o in out]


def efficiency(ev: dict, nfe: int) -> dict:
    """Generation timing as the evaluator recorded it (eval_meta timings, else any summary row at that NFE)."""
    t = (ev["meta"].get("timings") or {}).get(str(nfe))
    if isinstance(t, dict):
        return {k: float(t[k]) if t.get(k) is not None else np.nan for k, _ in EFF_KEYS}
    row = next((r for (n, _m), r in ev["summary"].items() if n == nfe), None)
    return {k: (fnum(row, k) if row else np.nan) for k, _ in EFF_KEYS}


def one_split(path: Path) -> dict:
    """The single split of a D1 manifest. `splits.read_manifest` returns a LIST (one entry per fold): a k-fold
    manifest is refused loudly here rather than silently reported as its first fold."""
    splits = read_manifest(path)
    if len(splits) != 1:
        folds = [s.get("fold", i) for i, s in enumerate(splits)]
        raise SystemExit(f"{path}: manifest holds {len(splits)} splits (folds {folds}); the D1 report describes ONE "
                         "split per corpus and must not silently take the first — say which fold is being reported "
                         "and write a single-split manifest for it before re-running")
    return splits[0]


def split_sizes(c: C.Corpus, meta: dict) -> dict | None:
    """Subjects/windows per split from the committed manifest + corpus MANIFEST; never regenerates either."""
    if c.manifest_path.exists() and (c.processed_dir / "MANIFEST.json").exists():
        return C.split_sizes(c, one_split(c.manifest_path))
    s = meta.get("split_sizes")
    return s if isinstance(s, dict) else None


#: where the evaluation-population fields live. `scripts/d1_evaluate.py` writes them into eval_meta.json and
#: carries the counts as CSV columns; both spellings of the realised total are accepted so a rename in the
#: evaluator degrades to a warning instead of a silent blank. Nothing outside this map is a source for them.
CAP_FIELDS: dict[str, tuple[str, ...]] = {
    "cap": ("max_test_windows_per_subject",),                       # the per-subject cap that was applied
    "n_eval": ("n_test_windows", "n_test_windows_total"),           # windows actually scored
    "n_available": ("n_test_windows_available",),                   # windows the test split holds
}


def _first_value(rows: list[dict], keys: tuple[str, ...]):
    for k in keys:
        for r in rows:
            v = r.get(k)
            if v not in ("", None):
                return v, k
    return None, None


def cap_info(ev: dict) -> dict:
    """The per-subject evaluation cap, the number of test windows actually scored, and how many the split holds.

    Read from `eval_meta.json` first, then from the matching column of the evaluator CSVs, and only then
    derived: the realised total by summing the per-subject `n_windows` column, the available total from the
    committed split manifest + corpus MANIFEST. Every value carries the artefact it came from. Nothing is
    invented: a field no artefact carries comes back None, is printed as `?`, and is named in a warning.
    """
    if ev.get("cap_cache") is not None:
        return ev["cap_cache"]                       # one read, so a missing field warns once, not once per table
    meta, rows = ev["meta"], ev["per_subject"] + list(ev["summary"].values())
    out = {k: None for k in CAP_FIELDS} | {f"{k}_source": None for k in CAP_FIELDS} | {"capped": None}
    for key, names in CAP_FIELDS.items():
        v, src = next(((meta[n], "eval_meta.json") for n in names if meta.get(n) not in ("", None)), (None, None))
        if v is None:
            v, col = _first_value(rows, names)
            src = f"the `{col}` column of the evaluator CSVs" if v is not None else None
        if v is not None:
            out[key], out[f"{key}_source"] = int(float(v)), src
    if out["n_eval"] is None:
        hn = headline_nfe_of(ev)
        n = [inum(r, "n_windows") for r in ev["per_subject"] if int(float(r["nfe"])) == hn]
        if n and all(v is not None for v in n):
            out["n_eval"], out["n_eval_source"] = int(sum(n)), f"sum of the per-subject `n_windows` column at NFE {hn}"
    if out["n_available"] is None:
        full = (split_sizes(ev["corpus"], meta) or {}).get("test", {}).get("n_windows")
        if full is not None:
            out["n_available"], out["n_available_source"] = int(full), "the split manifest + corpus MANIFEST.json"
    rule = meta.get("population_rule")
    out["rule"] = str(rule) if isinstance(rule, str) and rule.strip() else None
    if isinstance(meta.get("capped"), bool):
        out["capped"] = meta["capped"]
    elif out["n_eval"] is not None and out["n_available"] is not None:
        out["capped"] = out["n_eval"] < out["n_available"]
    for key, names in CAP_FIELDS.items():
        if out[key] is None:
            warn(f"{ev['key']}: the evaluator recorded no {' / '.join(names)} in eval_meta.json or in its CSV "
                 "columns — that part of the evaluation-population disclosure is printed as `?` and NOT inferred")
    ev["cap_cache"] = out
    return out


def cap_disclosure(evs: dict, order: list[str]) -> list[str]:
    """The one-line disclosure that goes under TABLE 1 whenever any corpus was evaluated on a capped subsample."""
    capped, unknown, rules = [], [], []
    for key in order:
        ci = cap_info(evs[key])
        if not ci["cap"]:
            continue
        if ci["capped"] is False:
            continue
        if ci["capped"] and ci["n_eval"] is not None and ci["n_available"] is not None:
            capped.append(f"{evs[key]['corpus'].name} {ci['n_eval']} of {ci['n_available']} test windows "
                          f"(cap {ci['cap']}/subject)")
            if ci["rule"] and ci["rule"] not in rules:
                rules.append(ci["rule"])
        else:
            unknown.append(f"{evs[key]['corpus'].name} (cap {ci['cap']}/subject)")
    out = []
    if capped:
        out.append("**Evaluation population — CAPPED.** Every number above was computed on a SUBSAMPLE of the test "
                   "split, not on all of it: " + "; ".join(capped) + " — the per-subject cap of "
                   "`--max-test-windows-per-subject`. Selection rule, quoted from the evaluator's own "
                   "`population_rule`: " + ("; ".join(f'"{r}"' for r in rules) if rules else
                                            "NOT RECORDED in eval_meta.json") +
                   ". Per-corpus counts and caps are in TABLE 3; the count of values behind each cell is the "
                   "`n_windows` column of `results_table.csv`.")
    if unknown:
        out.append("**Evaluation population — cap declared, extent unknown.** A per-subject cap is recorded for "
                   + "; ".join(unknown) + ", but the realised window count or the size of the test split is "
                   "missing from the evaluator artefacts, so the fraction evaluated cannot be stated here "
                   "(TABLE 3, and the warnings at the end of this report).")
    return out


def headline_nfe_of(ev: dict, default: int = HEADLINE_NFE) -> int:
    grid = ev["meta"].get("nfe_grid") or ev["nfes"]
    return default if default in grid else min(grid)


# ---------------------------------------------------------------- formatting
def fmt(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    a = abs(v)
    if a and (a < 1e-3 or a >= 1e5):
        return f"{v:.3e}"
    return f"{v:.3f}" if a < 10 else (f"{v:.2f}" if a < 1000 else f"{v:.1f}")


def cell(s: dict) -> str:
    return "—" if not np.isfinite(s["macro"]) else f"{fmt(s['macro'])} [{fmt(s['lo'])}, {fmt(s['hi'])}]"


def esc(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def md_table(header: list[str], rows: list[list[str]], numeric_from: int = 1) -> list[str]:
    align = ["---" if i < numeric_from else "---:" for i in range(len(header))]
    return ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "|".join(align) + "|"] \
        + ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows]


def prereg_section(title: str) -> list[str]:
    """Quote a section of the frozen preregistration verbatim rather than paraphrasing it."""
    if not PREREG.exists():
        return []
    out, on = [], False
    for ln in PREREG.read_text().splitlines():
        if ln.startswith("## "):
            if on:
                break
            on = ln.strip().endswith(title)
            continue
        if on:
            out.append(ln.rstrip())
    return out


def head_label(key: str) -> str:
    u = units(key)
    return f"{SPEC[key].name} {ARROW[SPEC[key].orientation]}" + ("" if u == "dimensionless" else f" ({u})")


# ---------------------------------------------------------------- tables
def table1(evs, order, long_rows) -> list[str]:
    keys = [k for k in SPEC if k not in RESTATEMENTS]
    restated = [k for k in SPEC if k in RESTATEMENTS]
    ident = "; ".join(f"`{SPEC[k].name}` = `{SPEC[RESTATEMENTS[k]].name}`" for k in restated)
    nb = (len(keys) + BLOCK - 1) // BLOCK
    out = ["### TABLE 1 — headline table, every metric in `PAPER_METRIC_SPEC` at the headline NFE", "",
           f"Every cell is `mean [95 % CI]` of the **subject-macro mean**: each test subject contributes exactly one "
           f"value, the mean is taken over subjects with equal subject weight, and the interval is the evaluator's "
           f"subject-clustered bootstrap ({C.BOOTSTRAP_N} replicates, seed {C.BOOTSTRAP_SEED}; prereg section 6). "
           "What *one value per subject* means is set by the metric's `level`: a `window`-level key is that "
           "subject's mean over its own windows, a `dataset`-level key is that metric computed once over that "
           "subject's windows. The window-WEIGHTED aggregate of the same values, `pooled_window_mean`, is a "
           "separate column of `results_table.csv` and never enters a cell here — it is an aggregation weighting, "
           "not the same thing as the metric keys spelled `pooled_*`.", "",
           f"**Two of these columns are ALGEBRAIC RESTATEMENTS of another column, not independent confirmations:** "
           f"{ident}. With equal-length windows a subject's pooled MAE *is* its mean per-window MAE, and the "
           "mean-of-windows RMSE *is* its mean per-window RMSE, so the subject-macro means coincide exactly — "
           "identical values are arithmetic, not agreement between two measurements. They are kept (KANFlow's "
           "Eq. 21 and PPGFlowECG's ambiguous reading both have to be reproducible) but are grouped into their own "
           "block below, away from the columns they restate. `RMSE (pooled)` is NOT one of them: its square root is "
           "taken after pooling, so it exceeds `RMSE` by the Jensen gap.", "",
           f"Columns are otherwise split into blocks of {BLOCK} purely for legibility; the single machine-readable "
           "form is `results_table.csv`. `—` means the evaluator produced no such column for that corpus (see the "
           "not-produced table below).", ""]
    groups = [(f"**TABLE 1 — block {i + 1} of {nb}**", keys[i * BLOCK:(i + 1) * BLOCK], "") for i in range(nb)]
    if restated:
        groups.append(("**TABLE 1 — restatement block: dataset-level keys that RESTATE a column above**", restated,
                       f"Read as a definition check, never as confirmation: {ident} by construction."))
    missing: dict[str, list[str]] = {}
    for title, blk, note in groups:
        rows = []
        for key in order:
            ev = evs[key]
            hn = headline_nfe_of(ev)
            row = [f"{ev['corpus'].name} (NFE {hn})"]
            for m in blk:
                s = stat(ev, m, hn)
                row.append(cell(s))
                if not has(ev, m):
                    missing.setdefault(m, []).append(key)
                    continue
                long_rows.append({"corpus": key, "corpus_name": ev["corpus"].name, "nfe": hn, "is_headline_nfe": "true",
                                  "metric_key": m, "metric_name": SPEC[m].name, "orientation": SPEC[m].orientation,
                                  "level": SPEC[m].level, "modality": SPEC[m].modality,
                                  "reported_by": ";".join(SPEC[m].reported_by), "subject_macro_mean": repr(s["macro"]),
                                  "macro_ci_lo": repr(s["lo"]), "macro_ci_hi": repr(s["hi"]),
                                  "pooled_window_mean": repr(s["pooled"]), "n_subjects": s["n_subjects"],
                                  "n_windows": s["n_windows"],
                                  "ci_source": s["source"], "boot_n": C.BOOTSTRAP_N, "boot_seed": C.BOOTSTRAP_SEED})
            rows.append(row)
        out += [title, ""] + ([note, ""] if note else [])
        out += md_table(["Dataset (NFE)"] + [head_label(m) for m in blk], rows) + [""]
    disclosure = cap_disclosure(evs, order)
    if disclosure:
        out += disclosure + [""]
    out += ["**Metric attribution and definition (footnote to TABLE 1).**", ""]
    out += md_table(["Metric", "Key", "Units", "Orientation", "Level", "Modality", "Reported by", "Definition as implemented"],
                    [[SPEC[m].name, f"`{m}`", units(m), f"{ARROW[SPEC[m].orientation]} {SPEC[m].orientation.replace('_', ' ')}",
                      SPEC[m].level, SPEC[m].modality, ", ".join(SPEC[m].reported_by), SPEC[m].definition]
                     for m in list(SPEC)], numeric_from=99) + [""]
    if missing:
        out += ["**Metrics named in `PAPER_METRIC_SPEC` that the evaluator did not produce** "
                "(listed, never silently dropped — prereg section 6):", ""]
        out += md_table(["Metric", "Key", "Missing for", "Reason"],
                        [[SPEC[m].name, f"`{m}`", ", ".join(v),
                          ("no implementing callable in paper_metrics (`impl` is None): "
                           "requires an external artefact this repo does not ship" if SPEC[m].impl is None
                           else f"modality `{SPEC[m].modality}` is not part of the D1 ECG benchmark"
                           if SPEC[m].modality != "ecg" else "no such column in per_subject_metrics.csv")]
                         for m, v in missing.items()], numeric_from=99) + [""]
    return out


def table2(evs, order, long_rows) -> list[str]:
    heads = [m for m in HEADLINE if m in SPEC]
    header = ["Dataset", "NFE"] + [head_label(m) for m in heads] + [lab for _, lab in EFF_KEYS]
    rows = []
    for key in order:
        ev = evs[key]
        hn = headline_nfe_of(ev)
        for nfe in ev["nfes"]:
            row = [ev["corpus"].name, str(nfe)]
            for m in heads:
                s = stat(ev, m, nfe)
                row.append(cell(s))
                if nfe != hn and has(ev, m):
                    long_rows.append({"corpus": key, "corpus_name": ev["corpus"].name, "nfe": nfe, "is_headline_nfe": "false",
                                      "metric_key": m, "metric_name": SPEC[m].name, "orientation": SPEC[m].orientation,
                                      "level": SPEC[m].level, "modality": SPEC[m].modality,
                                      "reported_by": ";".join(SPEC[m].reported_by), "subject_macro_mean": repr(s["macro"]),
                                      "macro_ci_lo": repr(s["lo"]), "macro_ci_hi": repr(s["hi"]),
                                      "pooled_window_mean": repr(s["pooled"]), "n_subjects": s["n_subjects"],
                                      "n_windows": s["n_windows"],
                                      "ci_source": s["source"], "boot_n": C.BOOTSTRAP_N, "boot_seed": C.BOOTSTRAP_SEED})
            e = efficiency(ev, nfe)
            rows.append(row + [fmt(e[k]) for k, _ in EFF_KEYS])
    return ["### TABLE 2 — NFE sweep", "",
            f"NFE {HEADLINE_NFE} is the headline (one-step is the method's claim); NFE {PENGUIN_NFE} is PENGUIN's "
            "published operating point and exists so that a budget-matched reading is possible (prereg section 6). "
            "Timing columns are whatever the evaluator measured on its own hardware and are not a controlled "
            "comparison against any published throughput.", ""] + md_table(header, rows, numeric_from=2)


def table3(evs, order) -> list[str]:
    header = ["Dataset", "Native fs PPG / ECG (Hz)", "Train subj / windows", "Val subj / windows",
              "Test subj / windows", "Test windows evaluated", "Cap (windows/subject)", "Trained by D1", "Source",
              "Disclosure flags"]
    rows, deviations = [], []
    for key in order:
        ev, c, meta = evs[key], evs[key]["corpus"], evs[key]["meta"]
        ss = split_sizes(c, meta)
        ci = cap_info(ev)
        if ss is None:
            warn(f"{key}: split sizes unavailable (needs {c.manifest} and {c.processed}/MANIFEST.json) — reported as ?")

        def part(name, ss=ss):
            if not ss or name not in ss:
                return "? / ?"
            return f"{ss[name].get('n_subjects', '?')} / {ss[name].get('n_windows', '?')}"

        flags = []
        n_train = (ss or {}).get("train", {}).get("n_windows")
        if n_train is not None:
            flags.append(f"small_corpus={'true' if n_train < C.MIN_TRAIN_WINDOWS else 'false'} "
                         f"(threshold {C.MIN_TRAIN_WINDOWS} train windows; hyperparameters NOT adjusted in response)")
        if key.startswith("capnobase"):
            flags.append("artifact screening: " + ("shipped expert artifact intervals dropped (SECONDARY disclosure row)"
                                                   if key.endswith("clean") else "none — this is the PRIMARY unscreened row"))
        if key == "vitaldb":
            flags.append("VitalDB caseid-subset rule frozen in the corpus MANIFEST; caseid is a marginally weaker "
                         "cluster than subjectid (6,388 cases map to 6,090 subject ids)")
        if key == "wildppg":
            flags.append("standing rule: subjects kjd and ssx never enter a train or val list in any D1 stage")
        if not c.trains:
            flags.append(f"NOT trained by D1 — reuses the frozen checkpoint {c.checkpoint}")
            deviations.append(f"`{key}`: prereg section 4 describes the D1 WildPPG run as a NEW training run on the "
                              f"14 eligible subjects, while `scripts/d1_common.py` marks this corpus `trains=False` and "
                              f"reuses `{c.checkpoint}` with the manifest `{c.manifest}`. Recorded here per prereg "
                              "section 12; the report author must resolve it in a dated `## Deviations` section.")
        if c.notes:
            flags.append(c.notes)
        if ci["cap"] and ci["capped"] and ci["n_eval"] is not None and ci["n_available"] is not None:
            flags.append(f"evaluation population CAPPED: {ci['n_eval']} of {ci['n_available']} test windows scored "
                         f"({ci['cap']}/subject; selection rule quoted under TABLE 1 from the evaluator's "
                         "`population_rule`)")
        rows.append([c.name, f"{c.fs_ppg} / {c.fs_ecg}", part("train"), part("val"), part("test"),
                     "?" if ci["n_eval"] is None else str(ci["n_eval"]),
                     "?" if ci["cap"] is None else ("none (every test window)" if ci["cap"] == 0 else str(ci["cap"])),
                     "yes" if c.trains else "no", c.citation, "; ".join(flags)])
    out = ["### TABLE 3 — corpus descriptors and disclosure flags", "",
           "Subjects and windows come from the committed split manifest and the corpus `MANIFEST.json` (never "
           "regenerated here); `?` means those files are not present next to this report. Native sampling rates and "
           "citations come from `scripts/d1_common.py`.", "",
           "**Test subj / windows** is the FULL test split. **Test windows evaluated** is how many of those windows "
           "the evaluator actually scored — `n_test_windows` from `eval_meta.json`, or the same count from its CSV "
           "columns — and **Cap** is its `max_test_windows_per_subject` (`--max-test-windows-per-subject`); the "
           "evaluator's own `population_rule` says how the subsample was taken and is quoted in the disclosure "
           "under TABLE 1. When the two window counts differ, every metric in this report for that corpus was "
           "computed on the subsample, not on the split; `?` means the evaluator recorded no such field and the "
           "value is NOT inferred here.",
           ""] + md_table(header, rows, numeric_from=99)
    if deviations:
        out += ["", "**Configuration facts that differ from the preregistration** (prereg section 12 requires these in "
                "the report, never by editing the preregistration):", ""] + [f"- {d}" for d in deviations]
    return out


def table4(evs, order, published) -> list[str]:
    header = ["Paper", "Their dataset", "Metric", "Their value", "Ours (same corpus, headline NFE)",
              "Mandatory non-comparability note"]
    rows = []
    for paper, ds, key, m, value, cite, extra in published:
        prot, ours, hn = PAPER_PROTOCOL.get(paper, {}), "—", HEADLINE_NFE
        if key in evs:
            ev = evs[key]
            hn = headline_nfe_of(ev)
            ours = cell(stat(ev, m, hn)) if has(ev, m) else "— (metric not produced)"
        elif key is not None:
            ours = "— (corpus not run)"
        note = (f"window {OURS_PROTOCOL['window']} vs {prot.get('window', 'not stated')}; "
                f"{OURS_PROTOCOL['band']} vs {prot.get('band', 'not stated')}; NFE {hn} vs {prot.get('nfe', 'not stated')}; "
                f"params {OURS_PROTOCOL['params']} vs {prot.get('params', 'not stated')}; "
                f"split {OURS_PROTOCOL['split']} vs {prot.get('split', 'not stated')}; ours is the subject-macro mean "
                f"with equal subject weight, theirs is an unspecified aggregation. {extra}.")
        rows.append([paper, ds, SPEC[m].name if m in SPEC else m, "—" if value is None else fmt(float(value)), ours, note])
    return ["### TABLE 4 — published-number context (NOT a head-to-head)", "",
            "**No row here is a like-for-like comparison and none may be quoted as one.** "
            "`docs/PREPROCESSING_CONVENTIONS_SURVEY.md` section 6 establishes that no cross-paper number in this field "
            "is comparable without restating the pipeline, so every row carries the pipeline differences explicitly. "
            "A blank *their value* is a number not present in the frozen metric spec, left empty rather than "
            "interpolated (prereg section 7.4); a blank *ours* is a corpus or a metric we did not run.", ""] \
        + md_table(header, rows, numeric_from=99) \
        + ["", "Citations: " + "; ".join(sorted({f"{p} — {c}" for p, _, _, _, _, c, _ in published})), ""]


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="D1 multi-dataset benchmark report (prereg section 7)")
    ap.add_argument("--eval-root", default=str(ROOT / "outputs"), help="directory holding d1_<corpus>_seed42/eval/")
    ap.add_argument("--out-root", default=str(ROOT / "outputs/d1_bench"), help="where RESULTS.md and results_table.csv go")
    ap.add_argument("--corpus", action="append", default=[], choices=list(C.BENCH_KEYS),
                    help="corpus key; repeatable. Exactly one of --corpus / --all is required")
    ap.add_argument("--all", action="store_true",
                    help=f"report every benchmark corpus ({' '.join(C.BENCH_KEYS)}); exactly one of --corpus / --all "
                         "is required, so the corpus set is always stated explicitly")
    ap.add_argument("--published", default=None,
                    help="optional JSON list of published-number rows replacing the built-in TABLE 4 set")
    args = ap.parse_args()
    if bool(args.corpus) == bool(args.all):
        ap.error("give exactly one of --corpus (repeatable) or --all")

    eval_root, out_root = Path(args.eval_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    evs, order, skipped = {}, [], []
    for key in (list(C.BENCH_KEYS) if args.all else args.corpus):
        c = C.corpus(key)
        ev = load_corpus(eval_root, c)
        if ev is None:
            skipped.append(key)
            warn(f"{key}: no evaluator output under {eval_dir(eval_root, c)} — corpus skipped")
            continue
        evs[key] = ev
        order.append(key)
    if not evs:
        raise SystemExit(f"no D1 evaluator output found under {eval_root}")

    published = [tuple(r) for r in json.loads(Path(args.published).read_text())] if args.published else PUBLISHED
    long_rows: list[dict] = []
    L = ["# D1 — Multi-Dataset Benchmark of the Frozen Methodology: RESULTS", "",
         f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `scripts/d1_report.py`. "
         "Preregistration: `docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md`. This is a **descriptive benchmark**: "
         "one fixed methodology, one fixed recipe, trained and evaluated separately inside each corpus.", "",
         f"- corpora reported: {', '.join(order)}"
         + (f"; **skipped (no evaluator output): {', '.join(skipped)}**" if skipped else ""),
         f"- eval root: `{eval_root}`",
         f"- metric spec: `ppg2ecg.evaluation.paper_metrics.PAPER_METRIC_SPEC` ({len(SPEC)} metrics)",
         f"- corpus config: `scripts/d1_common.py` ({len(C.BENCH_KEYS)} benchmark corpora; excluded: "
         + ", ".join(c.key for c in C.CORPORA if not c.included) + ")",
         f"- uncertainty: subject-clustered bootstrap, {C.BOOTSTRAP_N} replicates, seed {C.BOOTSTRAP_SEED} (prereg section 6)",
         "- aggregation: subject-macro mean primary, pooled window mean secondary; the two are never merged into one cell", ""]
    L += table1(evs, order, long_rows) + [""]
    L += table2(evs, order, long_rows) + [""]
    L += table3(evs, order) + [""]
    L += table4(evs, order, published) + [""]

    extra = {k: [c for c in evs[k]["columns"] if c not in SPEC] for k in order}
    extra = {k: v for k, v in extra.items() if v}
    if extra:
        L += ["### Appendix — evaluator columns outside `PAPER_METRIC_SPEC`", "",
              "Reported so that nothing the evaluator wrote is invisible. Columns suffixed `_ew` are the "
              "`evaluation.metrics.evaluate_windows` value of a name the paper table also produces; the two "
              "definitions differ and are never averaged together.", ""]
        L += md_table(["Dataset", "Unmapped columns"],
                      [[evs[k]["corpus"].name, ", ".join(v)] for k, v in extra.items()], numeric_from=99) + [""]
    if WARNINGS:
        L += ["### Warnings raised while building this report", ""] + [f"- {w}" for w in WARNINGS] + [""]

    L += ["## Claim boundary", "",
          "Quoted verbatim from the frozen preregistration section 11 — no D1 output may be worded to exceed it."]
    cb = prereg_section("Claim boundary")
    L += cb if cb else ["", "(preregistration not found; the claim boundary MUST be read from "
                        "`docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md` section 11 before any D1 number is quoted)"]
    L += ["", "In plain terms — what this benchmark **does** establish: the performance of one fixed methodology, under "
          "one fixed recipe, trained and tested separately inside each corpus, over a stated metric set at stated "
          "inference budgets, with subject-clustered uncertainty. What it **does not** establish: any ranking against a "
          "published method (pipeline, parameter count, split and inference budget all differ, and TABLE 4 exists to "
          "make that unmistakable); that one dataset is intrinsically harder than another (corpus size varies by two "
          "orders of magnitude and is a confound by construction — see FIG 6); and anything about generalisation "
          "across corpora, since no model is ever evaluated outside the corpus it was trained on.", ""]

    (out_root / "RESULTS.md").write_text("\n".join(L) + "\n")
    #: `n_windows` is the evaluator's own per-row count of the finite values behind the bootstrap — windows for a
    #: `per_window` metric, subject values for a `subject_level` one — i.e. how much data the cell rests on AFTER
    #: the per-subject evaluation cap. It comes from the bootstrap dict and was previously dropped here.
    fields = ["corpus", "corpus_name", "nfe", "is_headline_nfe", "metric_key", "metric_name", "orientation", "level",
              "modality", "reported_by", "subject_macro_mean", "macro_ci_lo", "macro_ci_hi", "pooled_window_mean",
              "n_subjects", "n_windows", "ci_source", "boot_n", "boot_seed"]
    with open(out_root / "results_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(long_rows)
    print(json.dumps({"results_md": str(out_root / "RESULTS.md"), "results_csv": str(out_root / "results_table.csv"),
                      "corpora": order, "skipped": skipped, "rows": len(long_rows), "warnings": len(WARNINGS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
