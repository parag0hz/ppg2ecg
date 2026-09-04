"""Synthetic D1 evaluator tree — the fixture the D1 read-out tests run against.

`make_fake_d1(root, ...)` writes a complete, self-consistent copy of the artefact layout that
`scripts/d1_evaluate.py` produces, under a caller-supplied directory (a pytest `tmp_path`):

    <root>/data/manifests/<corpus.manifest>                 split manifest, `splits.write_manifest` shape
    <root>/<corpus.processed>/MANIFEST.json                  {"files": {subject: {"n_windows": N}}}
    <root>/outputs/d1_<key>_seed42/eval/per_window_metrics.csv
                                        per_subject_metrics.csv
                                        summary_by_nfe.csv
                                        eval_meta.json
                                        waveforms_nfe<N>.npz

NO CHECKPOINT, NO GPU AND NO REAL DATA ARE INVOLVED. Every metric value is drawn from a seeded
`numpy.random.Generator` and is a *placeholder*, never a model result: these files exist so the report and
figure read-out layers can be exercised without a trained model. The tree is written where the caller asks
and nowhere else — in particular never under the repository's `outputs/`.

Two structural properties of the real evaluator are reproduced on purpose, because the report has to survive
them:

  * the subject-level restatement identity — a subject's `pooled_mae` is exactly the mean of that subject's
    per-window `mae`, and `mean_window_rmse` exactly the mean of its per-window `rmse`, so their subject-macro
    means coincide with the per-window columns while `pooled_rmse` (Jensen) does not;
  * the per-subject evaluation cap — `stored_windows` per subject in the corpus MANIFEST versus
    `n_eval_windows` actually scored, with `max_test_windows_per_subject` / `n_test_windows` in eval_meta.json.

Point `d1_common.ROOT` at the same directory (`patch_root`) so `Corpus.processed_dir`, `Corpus.manifest_path`
and `Corpus.out_dir` resolve inside the fake tree.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import d1_common as C  # noqa: E402

from ppg2ecg.data.splits import write_manifest  # noqa: E402

#: per-window columns written to per_window_metrics.csv. `mae_ew` is deliberately OUTSIDE PAPER_METRIC_SPEC so
#: the report's "columns outside the spec" appendix has something to report.
PER_WINDOW_COLUMNS: tuple[str, ...] = (
    "mae", "rmse", "pcc", "prd_meansub", "hr_ref", "hr_pred", "hr_abs_err", "rpeak_f1_50ms", "mae_ew")
#: subject-level scalars, written only to per_subject_metrics.csv (the evaluator's `sc` half)
SUBJECT_LEVEL_COLUMNS: tuple[str, ...] = ("pooled_mae", "pooled_rmse", "mean_window_rmse")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def _window_metrics(rng: np.random.Generator, n: int, nfe: int, bias: float) -> dict[str, np.ndarray]:
    """Placeholder per-window values: plausible ranges, deterministic, NOT a model result."""
    gain = 1.0 / (1.0 + 0.15 * np.log2(max(nfe, 1)))            # a mild, monotone "more NFE is better" shape
    mae = np.abs(0.30 * gain + bias + 0.02 * rng.standard_normal(n))
    rmse = mae * 1.4
    hr_ref = 60.0 + 12.0 * rng.random(n)
    return {"mae": mae, "rmse": rmse,
            "pcc": np.clip(0.80 - bias + 0.02 * rng.standard_normal(n), -1.0, 1.0),
            "prd_meansub": 45.0 * gain + 100.0 * bias + rng.standard_normal(n),
            "hr_ref": hr_ref, "hr_pred": hr_ref + 2.0 * rng.standard_normal(n),
            "hr_abs_err": np.abs(2.0 * rng.standard_normal(n)),
            "rpeak_f1_50ms": np.clip(0.75 - bias + 0.03 * rng.standard_normal(n), 0.0, 1.0),
            "mae_ew": mae * 1.01}


def _waveforms(rng: np.random.Generator, n: int, t: int, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic PPG / ECG / generated-ECG windows: Gaussian R peaks at ~1 Hz on a slow baseline."""
    tt = np.arange(t) / fs
    x, y, yh = [], [], []
    for _ in range(n):
        rr = 0.85 + 0.1 * rng.random()
        peaks = np.arange(0.3, tt[-1], rr)
        ecg = sum(np.exp(-0.5 * ((tt - p) / 0.012) ** 2) for p in peaks) - 0.15 * np.sin(2 * np.pi * tt / rr)
        y.append(ecg + 0.01 * rng.standard_normal(t))
        yh.append(0.95 * ecg + 0.03 * rng.standard_normal(t))
        x.append(np.sin(2 * np.pi * tt / rr - 0.6) + 0.05 * rng.standard_normal(t))
    return (np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(yh, np.float32))


def make_fake_d1(root: Path | str, corpora: tuple[str, ...] = ("dalia", "bidmc"), nfes: tuple[int, ...] = (1, 2),
                 n_subjects: int = 3, n_eval_windows: int = 6, stored_windows: int | None = None,
                 cap: int = 1024, waveform_windows: int = 2, seed: int = 20260904,
                 write_waveforms: bool = True, write_summary: bool = True, write_meta: bool = True) -> dict:
    """Build the fake tree under `root` and return the handles the tests need.

    Args:
      corpora: benchmark corpus keys (must exist in `d1_common.BY_KEY`).
      nfes: the NFE grid to fabricate; the first entry is the headline in the tests.
      n_subjects / n_eval_windows: test subjects and windows-per-subject actually "evaluated".
      stored_windows: windows per subject declared in the corpus MANIFEST. Larger than `n_eval_windows`
                      means the corpus was CAPPED (that is the state the report must disclose).
      cap: the value written as `max_test_windows_per_subject`; 0 means "no cap".
      write_summary / write_meta: drop summary_by_nfe.csv / eval_meta.json to exercise the skip paths.
    """
    root = Path(root)
    stored = int(stored_windows if stored_windows is not None else n_eval_windows)
    out: dict = {"root": root, "eval_root": root / "outputs", "corpora": list(corpora), "nfes": list(nfes),
                 "per_window_columns": list(PER_WINDOW_COLUMNS), "subject_level_columns": list(SUBJECT_LEVEL_COLUMNS),
                 "n_subjects": n_subjects, "n_eval_windows": n_eval_windows, "stored_windows": stored,
                 "cap": cap, "capped": stored > n_eval_windows, "subjects": {}, "eval_dirs": {},
                 "n_test_windows": n_subjects * n_eval_windows, "split_test_windows": n_subjects * stored}
    for ci, key in enumerate(corpora):
        c = C.corpus(key)
        rng = np.random.default_rng(seed + 1000 * ci)
        test = [f"F{i + 1}" for i in range(n_subjects)]
        train, val = [f"T{i + 1}" for i in range(4)], ["V1"]
        out["subjects"][key] = list(test)

        (root / c.processed).mkdir(parents=True, exist_ok=True)
        (root / c.processed / "MANIFEST.json").write_text(json.dumps(
            {"corpus": key, "fs": C.FS, "samples_per_window": C.SAMPLES_PER_WINDOW,
             "files": {s: {"n_windows": stored} for s in train + val + test}}, indent=1))
        write_manifest(root / c.manifest,
                       {"protocol": "D1-subject-holdout", "seed": C.SEED, "corpus": key, "rule": C.SPLIT_RULE,
                        "n_subjects": len(train + val + test), "train": train, "val": val, "test": test},
                       {"note": "SYNTHETIC fixture split — tests/fixtures/make_fake_d1.py, never a real corpus"})

        d = root / "outputs" / c.exp_name / "eval"
        d.mkdir(parents=True, exist_ok=True)
        out["eval_dirs"][key] = d
        pw_rows, sub_rows, summ_rows, timings = [], [], [], {}
        for nfe in nfes:
            per_subject_pw: dict[str, dict[str, np.ndarray]] = {}
            for si, s in enumerate(test):
                vals = _window_metrics(rng, n_eval_windows, nfe, bias=0.01 * si + 0.02 * ci)
                per_subject_pw[s] = vals
                for j in range(n_eval_windows):
                    pw_rows.append({"corpus": key, "subject": s, "window_index": int(j * max(stored // max(n_eval_windows, 1), 1)),
                                    "nfe": nfe, "n_test_windows_subject": n_eval_windows,
                                    "n_test_windows_total": n_subjects * n_eval_windows,
                                    **{k: float(v[j]) for k, v in vals.items()}})
                sub_rows.append({"corpus": key, "subject": s, "nfe": nfe, "n_windows": n_eval_windows,
                                 "n_test_windows_subject": n_eval_windows,
                                 "n_test_windows_total": n_subjects * n_eval_windows,
                                 **{k: float(np.mean(v)) for k, v in vals.items()},
                                 # the restatement identity the real evaluator also produces
                                 "pooled_mae": float(np.mean(vals["mae"])),
                                 "mean_window_rmse": float(np.mean(vals["rmse"])),
                                 "pooled_rmse": float(np.mean(vals["rmse"]) * 1.05)})
            timings[nfe] = {"gen_wall_clock_s": 1.0 * nfe, "gen_samples_per_s": (n_subjects * n_eval_windows) / (1.0 * nfe),
                            "gen_n_windows": n_subjects * n_eval_windows, "bench_latency_ms_median": 10.0 * nfe,
                            "bench_samples_per_s": 64.0 / nfe, "bench_batch": 64, "peak_mem_MiB": 512.0}
            subs = np.asarray([s for s in test for _ in range(n_eval_windows)])
            counts = {"n_test_windows_total": n_subjects * n_eval_windows, "n_test_subjects": n_subjects,
                      "test_windows_per_subject": ";".join(f"{s}={n_eval_windows}" for s in test)}
            for name in PER_WINDOW_COLUMNS:
                v = np.concatenate([per_subject_pw[s][name] for s in test])
                summ_rows.append({"corpus": key, "nfe": nfe, "metric": name, "kind": "per_window", **counts,
                                  **C.subject_cluster_bootstrap(v, subs), **timings[nfe]})
            for name in SUBJECT_LEVEL_COLUMNS:
                v = np.asarray([r[name] for r in sub_rows if r["nfe"] == nfe and r["corpus"] == key], dtype=np.float64)
                summ_rows.append({"corpus": key, "nfe": nfe, "metric": name, "kind": "subject_level", **counts,
                                  **C.subject_cluster_bootstrap(v, np.asarray(test)), **timings[nfe]})
            if write_waveforms:
                x, y, yh = _waveforms(rng, n_subjects * waveform_windows, C.SAMPLES_PER_WINDOW, C.FS)
                np.savez_compressed(d / f"waveforms_nfe{nfe}.npz", x=x, y=y, yhat=yh,
                                    subject=np.asarray([s for s in test for _ in range(waveform_windows)]),
                                    window_index=np.tile(np.arange(waveform_windows, dtype=np.int32), n_subjects))
        _write_csv(d / "per_window_metrics.csv", pw_rows)
        _write_csv(d / "per_subject_metrics.csv", sub_rows)
        if write_summary:
            _write_csv(d / "summary_by_nfe.csv", summ_rows)
        if write_meta:
            (d / "eval_meta.json").write_text(json.dumps(
                {"corpus": key, "dataset": c.name, "citation": c.citation, "processed": c.processed,
                 "manifest": c.manifest, "test_subjects": test,
                 "n_test_windows": n_subjects * n_eval_windows,
                 "n_test_windows_per_subject": {s: n_eval_windows for s in test},
                 "n_test_windows_available": n_subjects * stored,
                 "n_test_windows_available_per_subject": {s: stored for s in test},
                 "max_test_windows_per_subject": int(cap), "capped": stored > n_eval_windows,
                 "population_rule": "stored window order per subject; exactly min(n, cap) unique ascending indices "
                                    "when capped; no shuffling, no outcome-dependent selection",
                 "nfe_grid": list(nfes), "detector": "neurokit",
                 "bootstrap": {"n_boot": C.BOOTSTRAP_N, "seed": C.BOOTSTRAP_SEED, "unit": "subject cluster",
                               "primary": "subject_macro_mean", "secondary": "pooled_window_mean"},
                 "timings": {str(k): v for k, v in timings.items()},
                 "columns_per_window": sorted(PER_WINDOW_COLUMNS), "columns_subject_level": sorted(SUBJECT_LEVEL_COLUMNS),
                 "training": False, "synthetic_fixture": "tests/fixtures/make_fake_d1.py — placeholder values, not a model result"},
                indent=1))
    return out


def patch_root(monkeypatch, root: Path | str) -> None:
    """Resolve `Corpus.processed_dir` / `manifest_path` / `out_dir` inside the fake tree instead of the repo."""
    monkeypatch.setattr(C, "ROOT", Path(root))
