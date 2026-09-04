"""D1 per-corpus evaluation (docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md §6).

Frozen inference only — no weight is updated here. The checkpoint is rebuilt exactly as scripts/eval_c1_arms.py
does it (`model_cfg` -> build_penguin_backbone, `imf_cfg` -> cond_mode/h_scale, `target_norm` -> inverse before any
metric) and sampled with `event_reliability.sample_meanflow_schedule` on the uniform schedule, so NFE is asserted
rather than assumed. Generation runs on the corpus TEST subjects only.

Per NFE in {1, 2, 4, 10, 25, 50}: metrics (paper_metrics.paper_metric_table + metrics.evaluate_windows), generation
wall-clock and samples/s, a fixed-batch efficiency benchmark, and a subject-clustered bootstrap (2000 replicates,
seed 20260904) reporting the SUBJECT-MACRO mean as primary and the pooled window mean as a separate secondary
column. The two are never merged into one cell.

Evaluation population: EVERY test window of every test subject. --max-test-windows-per-subject defaults to 0 = no
cap, so no corpus is ever silently subsampled; a caller that needs the NFE-50 sweep to stay tractable on a large
corpus may pass e.g. --max-test-windows-per-subject 1024 for vitaldb (never for bidmc, capnobase or dalia, which
must be evaluated complete). When a cap is given it selects exactly min(n, cap) unique ascending indices per
subject (`d1_common.capped_indices`) — never a shuffle, never a selection on any outcome. The REALISED per-subject
and total window counts are written into eval_meta.json and into every output CSV as a column, so the size of the
evaluated population can never go missing downstream.

Run: .venv/bin/python scripts/d1_evaluate.py --corpus wildppg [--force] [--dry-run]
     .venv/bin/python scripts/d1_evaluate.py --all
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401  (MKL warm-up must precede `import torch`, docs/ENVIRONMENT.md)

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ppg2ecg.data.target_norm import TargetNorm  # noqa: E402
from ppg2ecg.evaluation import event_reliability as ER  # noqa: E402
from ppg2ecg.evaluation.efficiency import benchmark  # noqa: E402
from ppg2ecg.evaluation.metrics import evaluate_windows  # noqa: E402
from ppg2ecg.flow.imeanflow import MeanFlowS5  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402
from ppg2ecg.utils.upstream import assert_upstream_pinned  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_common import (  # noqa: E402
    BENCH_KEYS, BOOTSTRAP_N, BOOTSTRAP_SEED, FS, NFE_GRID, ROOT, SAMPLES_PER_WINDOW, WAVEFORM_WINDOWS_PER_SUBJECT,
    Corpus, assert_no_forbidden_subjects, capped_indices, corpus, ensure_split_manifest, git_sha, sha256_file,
    subject_cluster_bootstrap, window_counts,
)

#: same Gaussian source convention as scripts/eval_c1_arms.py (SRC_SEED = 0), redrawn per corpus population
SOURCE_SEED = 0
DETECTOR = "neurokit"
PAPER_METRICS_MODULE = "ppg2ecg.evaluation.paper_metrics"


# ----------------------------------------------------------------------------------------------------------------------
# Test population
# ----------------------------------------------------------------------------------------------------------------------
def load_test(c: Corpus, subjects: list[str], per_subject_cap: int):
    """Test windows in stored order. `per_subject_cap` > 0 keeps exactly min(n, cap) of them per subject.

    The selection is `d1_common.capped_indices`: unique, ascending, deterministic, outcome-blind. It replaces an
    integer-stride rule that was not a cap at all — for cap 1024 a subject with 1070 windows got stride 2 and was
    halved to 535 while being reported as complete.
    """
    xs, ys, sid, widx = [], [], [], []
    for s in subjects:
        d = np.load(c.processed_dir / f"{s}.npz")
        idx = capped_indices(len(d["x"]), per_subject_cap)
        xs.append(d["x"][idx].astype(np.float32))
        ys.append(d["y"][idx].astype(np.float32))
        sid += [s] * len(idx)
        widx.append(d["window_index"][idx].astype(np.int32))
    return np.concatenate(xs), np.concatenate(ys), np.asarray(sid), np.concatenate(widx)


def waveform_rows(sid: np.ndarray, subjects: list[str], k: int) -> np.ndarray:
    """Rows OF THE SCORED POPULATION used for the figure npz: the first k rows of each test subject, in order.

    The figure must show samples that a per_window_metrics.csv row describes, so the waveform set is a subset of the
    evaluated population and reuses that population's generated output — never a second load and a second sampling
    pass with different noise.
    """
    parts = [np.flatnonzero(sid == s)[:k] for s in subjects]
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)


def assert_chunking_has_no_set_level(metric_chunk: int, subj_cols) -> None:
    """A set-level statistic computed on a chunk is not the subject's statistic, and chunks cannot be averaged back.

    `acc[s]["sc"] |= sc` used to keep the LAST chunk's values silently. A Frechet distance (and every other
    distribution-level column) has no chunk mean, so chunking simply may not emit these columns.
    """
    cols = sorted(subj_cols)
    if metric_chunk and cols:
        raise ValueError(
            f"--metric-chunk {int(metric_chunk)} splits each test subject into blocks, so the set-level (subject-level) "
            f"columns {cols} would be computed on a block instead of on the subject. They are not chunk-averageable "
            f"(a Frechet distance has no mean over chunks). Re-run without --metric-chunk (0) to produce them.")


# ----------------------------------------------------------------------------------------------------------------------
# Model + sampling (rebuilt exactly as scripts/eval_c1_arms.py does it)
# ----------------------------------------------------------------------------------------------------------------------
def build_net(ckpt: Path, device: torch.device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("imf_cfg", {})
    net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                     h_scale=cfg.get("h_scale", 1.0)).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    net.requires_grad_(False)
    tn = ck.get("target_norm") or {"mu": 0.0, "sigma": 1.0, "source": "identity"}
    return net, ck, TargetNorm(float(tn["mu"]), float(tn["sigma"]), str(tn["source"]))


def uniform_schedule(nfe: int) -> list[float]:
    return ER.UNIFORM.get(nfe, [1.0 / nfe] * nfe)


@torch.no_grad()
def generate(net, x: np.ndarray, e_all: torch.Tensor, nfe: int, batch: int, device: torch.device):
    """Uniform-schedule MeanFlow sampling over the whole population; returns (pred, wall_clock_s). NFE is asserted."""
    sched, outs, got = uniform_schedule(nfe), [], set()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(x), batch):
        ppg = torch.from_numpy(x[i : i + batch]).to(device).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(net, ppg, e_all[i : i + batch].to(device), sched)
        got.add(int(k))
        outs.append(z.squeeze(1).float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    assert got == {nfe}, f"NFE parity violated at {nfe}: {got}"
    return np.concatenate(outs), dt


# ----------------------------------------------------------------------------------------------------------------------
# Metric columns: paper_metric_table (every PAPER_METRIC_SPEC column) + the repo's evaluate_windows
# ----------------------------------------------------------------------------------------------------------------------
def spec_names(spec) -> list[str]:
    """Column names of PAPER_METRIC_SPEC, whatever container the module uses (a mapping, or records carrying a name)."""
    if isinstance(spec, dict):
        return [str(k) for k in spec]
    out = []
    for s in spec:
        if isinstance(s, str):
            out.append(s)
        elif isinstance(s, dict):
            out.append(str(s["name"]))
        elif isinstance(s, (tuple, list)):
            out.append(str(s[0]))
        else:
            out.append(str(s.name))
    return out


def as_columns(obj, n: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Split a metric table into per-window columns (length n) and set-level scalars. Accepts a DataFrame, a flat
    dict of arrays, or one level of nesting (the shape `metrics.evaluate_windows` returns)."""
    flat: dict[str, np.ndarray] = {}
    if hasattr(obj, "columns") and hasattr(obj, "to_numpy"):
        flat = {str(c): np.asarray(obj[c], dtype=np.float64) for c in obj.columns}
    else:
        for k, v in dict(obj).items():
            if isinstance(v, dict):
                flat |= {str(k2): np.asarray(v2, dtype=np.float64) for k2, v2 in v.items()}
            else:
                flat[str(k)] = np.asarray(v, dtype=np.float64)
    per_window = {k: v for k, v in flat.items() if v.ndim == 1 and len(v) == n}
    scalars = {k: float(v.reshape(-1)[0]) for k, v in flat.items() if v.ndim == 0 or v.size == 1}
    return per_window, scalars


def score_task(task):
    """One (subject, block) of generated vs. reference windows. Runs in a worker process; the dataset-level half of
    PAPER_METRIC_SPEC (`paper_metric_pooled`) is computed at the block's scope, which is the whole subject unless
    --metric-chunk splits it."""
    subject, pred, gt = task
    pm = importlib.import_module(PAPER_METRICS_MODULE)
    table = pm.paper_metric_table(pred, gt, FS, detector=DETECTOR)
    pw, sc = as_columns(table, len(pred))
    sc |= as_columns(pm.paper_metric_pooled(pred, gt, FS, table=table), len(pred))[1]
    ew_pw, ew_sc = as_columns(evaluate_windows(pred, gt, fs=FS, hr_window_segments=1, detector=DETECTOR), len(pred))
    for src, dst in ((ew_pw, pw), (ew_sc, sc)):
        for k, v in src.items():
            dst[k if k not in dst else f"{k}_ew"] = v  # a name the paper table already produced keeps its own values
    return subject, pw, sc


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------------------------------------------------
# One corpus
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_corpus(c: Corpus, args) -> int:
    split = ensure_split_manifest(c)
    assert_no_forbidden_subjects(split, f"d1_evaluate[{c.key}]")
    test = list(split["test"])
    nfes = [int(v) for v in args.nfe.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out, done = c.eval_dir, c.out_dir / "EVAL_DONE"

    print(f"[d1-eval] corpus {c.key} ({c.name}) | checkpoint {c.checkpoint_path.relative_to(ROOT)} "
          f"({'present' if c.checkpoint_path.exists() else 'ABSENT'}) | test subjects {test} | NFE {nfes}")
    if args.dry_run:
        x, _y, sid, _w = load_test(c, test, args.max_test_windows_per_subject)
        realised = {s: int((sid == s).sum()) for s in test}
        print(f"[d1-eval] test population {len(x)} windows over {len(set(sid.tolist()))} subjects "
              f"(cap {args.max_test_windows_per_subject or 'none'}/subject) | device {device.type}")
        print(f"[d1-eval] realised windows/subject: {realised} | total {sum(realised.values())}")
        print(f"[d1-eval] outputs -> {out.relative_to(ROOT)}/{{per_window_metrics.csv,per_subject_metrics.csv,summary_by_nfe.csv,eval_meta.json,waveforms_nfe*.npz}}")
        print(f"[d1-eval] bootstrap {BOOTSTRAP_N}@{BOOTSTRAP_SEED} | source seed {SOURCE_SEED} | detector {DETECTOR} | "
              f"waveforms: first {WAVEFORM_WINDOWS_PER_SUBJECT} SCORED windows/subject "
              f"({len(waveform_rows(sid, test, WAVEFORM_WINDOWS_PER_SUBJECT))} rows of the scored population)")
        print(f"[d1-eval] {PAPER_METRICS_MODULE} importable: {importlib.util.find_spec(PAPER_METRICS_MODULE) is not None}")
        print("[d1-eval] --dry-run: nothing executed")
        return 0
    if done.exists() and not args.force:
        print(f"[d1-eval] {done.relative_to(ROOT)} exists -> skipping (use --force to re-evaluate)")
        return 0
    if not c.checkpoint_path.exists():
        print(f"[d1-eval] checkpoint missing at {c.checkpoint_path} -> nothing to evaluate")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    up = assert_upstream_pinned()
    net, ck, tnorm = build_net(c.checkpoint_path, device)
    x, y, sid, widx = load_test(c, test, args.max_test_windows_per_subject)
    assert x.shape[1] == SAMPLES_PER_WINDOW, x.shape
    gt = tnorm.inverse(y).astype(np.float64) if not tnorm.is_identity else y.astype(np.float64)
    e_all = torch.randn(len(x), 1, x.shape[1], generator=torch.Generator().manual_seed(SOURCE_SEED))
    src_hash = hashlib.sha256(e_all.numpy().tobytes()).hexdigest()
    n_windows_per_subject = {s: int((sid == s).sum()) for s in test}
    n_available_per_subject = {s: int(v) for s, v in window_counts(c).items() if s in set(test)}
    n_test_windows = int(len(x))
    wave_rows = waveform_rows(sid, test, WAVEFORM_WINDOWS_PER_SUBJECT)
    wave_pairs = [[str(sid[i]), int(widx[i])] for i in wave_rows]
    print(f"[d1-eval] {len(x)} test windows over {len(test)} subjects | checkpoint epoch {ck.get('epoch')} | "
          f"target_norm {tnorm.source} | device {device.type} | source sha256 {src_hash[:16]}", flush=True)
    print(f"[d1-eval] realised windows/subject (cap {args.max_test_windows_per_subject or 'none'}): "
          f"{n_windows_per_subject} | total {n_test_windows}", flush=True)

    order = [(s, np.flatnonzero(sid == s)) for s in test]
    per_window_rows, per_subject_rows, summary_rows, timing = [], [], [], {}
    cols_pw: dict[str, None] = {}
    subj_cols: list[str] = []
    xb, eb = torch.from_numpy(x[: args.batch_size]).to(device).unsqueeze(1), e_all[: args.batch_size].to(device)
    for nfe in nfes:
        pred_n, wall = generate(net, x, e_all, nfe, args.batch_size, device)
        pred = tnorm.inverse(pred_n).astype(np.float64) if not tnorm.is_identity else pred_n.astype(np.float64)
        sched = uniform_schedule(nfe)
        eff = benchmark(lambda: ER.sample_meanflow_schedule(net, xb, eb, sched), n_warmup=3,  # noqa: B023
                        n_repeats=args.bench_repeats, batch_size=len(xb), device=device.type)
        timing[nfe] = {"gen_wall_clock_s": wall, "gen_samples_per_s": len(x) / wall, "gen_n_windows": int(len(x)),
                       "bench_latency_ms_median": eff["latency_ms_median"], "bench_samples_per_s": eff.get("samples_per_s"),
                       "bench_batch": int(len(xb)), "peak_mem_MiB": eff.get("peak_mem_MiB")}

        blocks = [(s, i) for s, i in order] if not args.metric_chunk else [
            (s, i[j : j + args.metric_chunk]) for s, i in order for j in range(0, len(i), args.metric_chunk)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(score_task, [(s, pred[i], gt[i]) for s, i in blocks]))
        acc = {s: {"pw": {}, "sc": {}} for s in test}
        for s, pw, sc in results:
            for k, v in pw.items():
                acc[s]["pw"].setdefault(k, []).append(v)
            acc[s]["sc"] |= sc
        # a chunked block's set-level statistic is not the subject's; `|=` above would keep the LAST block's values
        assert_chunking_has_no_set_level(args.metric_chunk, {k for s in test for k in acc[s]["sc"]})
        nfe_pw: dict[str, list[np.ndarray]] = {}
        nfe_sub: list[str] = []
        for s, i in order:
            pw = {k: np.concatenate(v) for k, v in acc[s]["pw"].items()}
            sc = acc[s]["sc"]
            for j, w in enumerate(widx[i]):
                per_window_rows.append({"corpus": c.key, "subject": s, "window_index": int(w), "nfe": nfe,
                                        "n_test_windows_subject": n_windows_per_subject[s],
                                        "n_test_windows_total": n_test_windows,
                                        **{k: float(v[j]) for k, v in pw.items()}})
            per_subject_rows.append({"corpus": c.key, "subject": s, "nfe": nfe, "n_windows": int(len(i)),
                                     "n_test_windows_subject": n_windows_per_subject[s],
                                     "n_test_windows_total": n_test_windows,
                                     **{k: float(np.nanmean(v)) for k, v in pw.items()}, **sc})
            for k, v in pw.items():
                nfe_pw.setdefault(k, []).append(v)
            nfe_sub += [s] * len(i)
            cols_pw |= dict.fromkeys(pw)
        subj_cols = sorted({k for s in test for k in acc[s]["sc"]})
        subs_pw = np.asarray(nfe_sub)
        counts_col = {"n_test_windows_total": n_test_windows, "n_test_subjects": len(test),
                      "test_windows_per_subject": ";".join(f"{s}={n}" for s, n in n_windows_per_subject.items())}
        for name, chunks in nfe_pw.items():
            summary_rows.append({"corpus": c.key, "nfe": nfe, "metric": name, "kind": "per_window", **counts_col,
                                 **subject_cluster_bootstrap(np.concatenate(chunks), subs_pw), **timing[nfe]})
        for name in subj_cols:
            vals = np.asarray([acc[s]["sc"].get(name, np.nan) for s in test], dtype=np.float64)
            summary_rows.append({"corpus": c.key, "nfe": nfe, "metric": name, "kind": "subject_level", **counts_col,
                                 **subject_cluster_bootstrap(vals, np.asarray(test)), **timing[nfe]})
        print(f"[d1-eval] {c.key} NFE {nfe:2d}: {len(nfe_pw)} per-window + {len(subj_cols)} subject-level columns | "
              f"gen {wall:.1f}s ({len(x) / wall:.0f} win/s) | batch{len(xb)} {eff['latency_ms_median']:.0f} ms", flush=True)

        # the figure set is a SUBSET of the scored population and reuses its generated rows, so every panel of FIG 1
        # is described by a row of per_window_metrics.csv (same subject, same window_index, same generated sample)
        np.savez_compressed(out / f"waveforms_nfe{nfe}.npz", x=x[wave_rows], y=gt[wave_rows].astype(np.float32),
                            yhat=pred[wave_rows].astype(np.float32), subject=sid[wave_rows],
                            window_index=widx[wave_rows], population_row=wave_rows,
                            selection=np.asarray([f"{a}:{b}" for a, b in wave_pairs], dtype="U64"),
                            selection_rule=f"first {WAVEFORM_WINDOWS_PER_SUBJECT} SCORED windows of each test subject, "
                                           f"in stored order; identical rows and generated samples as per_window_metrics.csv")

    write_csv(out / "per_window_metrics.csv", per_window_rows)
    write_csv(out / "per_subject_metrics.csv", per_subject_rows)
    write_csv(out / "summary_by_nfe.csv", summary_rows)
    pm_spec = spec_names(importlib.import_module(PAPER_METRICS_MODULE).PAPER_METRIC_SPEC)
    meta = {"corpus": c.key, "dataset": c.name, "citation": c.citation,
            "checkpoint": str(c.checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": sha256_file(c.checkpoint_path),
            "checkpoint_epoch": ck.get("epoch"), "target_norm": {"mu": tnorm.mu, "sigma": tnorm.sigma, "source": tnorm.source},
            "processed": c.processed, "processed_manifest_sha256": sha256_file(c.processed_dir / "MANIFEST.json"),
            "manifest": c.manifest, "manifest_sha256": sha256_file(c.manifest_path), "test_subjects": test,
            "n_test_windows": n_test_windows, "n_test_windows_per_subject": n_windows_per_subject,
            "n_test_windows_available_per_subject": n_available_per_subject,
            "n_test_windows_available": int(sum(n_available_per_subject.values())),
            "max_test_windows_per_subject": int(args.max_test_windows_per_subject),
            "capped": any(n_windows_per_subject[s] < n_available_per_subject[s] for s in test),
            "population_rule": "stored window order per subject; exactly min(n, cap) unique ascending indices when "
                               "capped (d1_common.capped_indices); no shuffling, no outcome-dependent selection",
            "nfe_grid": nfes, "source_seed": SOURCE_SEED, "source_bank_sha256": src_hash,
            "bootstrap": {"n_boot": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "subject cluster",
                          "primary": "subject_macro_mean", "secondary": "pooled_window_mean"},
            "detector": DETECTOR, "waveform_windows_per_subject": WAVEFORM_WINDOWS_PER_SUBJECT,
            "waveform_selection": wave_pairs, "waveform_population_rows": [int(v) for v in wave_rows],
            "waveform_rule": "a SUBSET of the scored population: the first N windows of each test subject in stored "
                             "order, reusing that population's generated rows (same samples as per_window_metrics.csv)",
            "metric_chunk": int(args.metric_chunk), "timings": {str(k): v for k, v in timing.items()},
            "paper_metric_spec": pm_spec, "paper_metrics_not_produced": sorted(set(pm_spec) - set(cols_pw) - set(subj_cols)),
            "columns_per_window": sorted(cols_pw), "columns_subject_level": subj_cols,
            "upstream": up, "git": git_sha(), "device": str(device), "torch": torch.__version__,
            "training": False, "created": datetime.now().isoformat(timespec="seconds")}
    (out / "eval_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    done.write_text(json.dumps({"corpus": c.key, "nfe_grid": nfes, "n_test_windows": n_test_windows,
                                "n_test_windows_per_subject": n_windows_per_subject, "finished": meta["created"]}))
    if meta["paper_metrics_not_produced"]:
        print(f"[d1-eval] WARNING PAPER_METRIC_SPEC columns not produced: {meta['paper_metrics_not_produced']}")
    print(f"[d1-eval] wrote {out.relative_to(ROOT)}/ (per_window {len(per_window_rows)} rows, per_subject "
          f"{len(per_subject_rows)}, summary {len(summary_rows)})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(BENCH_KEYS))
    ap.add_argument("--all", action="store_true", help=f"evaluate every benchmark corpus in order: {' '.join(BENCH_KEYS)}")
    ap.add_argument("--nfe", default=",".join(str(n) for n in NFE_GRID))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bench-repeats", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--metric-chunk", type=int, default=0, help="0 = one metric task per test subject (set-level metrics stay subject-level); >0 splits a subject into blocks of N windows and REFUSES to emit set-level columns")
    ap.add_argument("--max-test-windows-per-subject", type=int, default=0,
                    help="0 (default) = evaluate EVERY test window; >0 keeps exactly min(n, N) deterministic unique "
                         "ascending indices per test subject. Only pass a cap for a large corpus (vitaldb) — bidmc, "
                         "capnobase and dalia must be evaluated complete. The realised counts are written to "
                         "eval_meta.json and to a column of every output CSV.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if bool(args.corpus) == bool(args.all):
        ap.error("give exactly one of --corpus or --all")
    rc = 0
    for k in (list(BENCH_KEYS) if args.all else [args.corpus]):
        rc |= evaluate_corpus(corpus(k), args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
