"""D2 driver — baseline floor and conditioning controls (docs/D2_BASELINE_FLOOR_PREREGISTRATION.md).

NO TRAINING. NO WEIGHT UPDATE. Inference and arithmetic on the IDENTICAL D1 test population, asserted per corpus
against D1's own per_window_metrics.csv row sequence (hard failure on mismatch), with the identical metric suite,
aggregation and bootstrap seed.

Arms: ours@NFE1 (re-read from D1's saved output, never re-sampled), B0..B5 (prereg §4).
Run: .venv/bin/python scripts/d2_baselines.py [--corpus KEY ...]
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import d1_common as C  # noqa: E402
import d1_evaluate as E  # noqa: E402

from ppg2ecg.evaluation import baselines as B  # noqa: E402
from ppg2ecg.evaluation import metrics as M  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as P  # noqa: E402

FS, NFE_HEADLINE, BOOT_N, BOOT_SEED = 128, 1, 2000, 20260904
OUT = ROOT / "outputs/d2_baselines"
ARMS = ("ours_nfe1", "B0_wrong_window", "B1_ppg_template", "B2_ppg_mean_beat",
        "B3_train_mean", "B4_mismatched_ppg", "B5_gt_timing")
LEAKY = {"B0_wrong_window": "uses test GT of the same subject; diagnostic only",
         "B5_gt_timing": "(GT-R leakage; diagnostic only)"}


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def d1_population(c: C.Corpus) -> list[tuple[str, int]]:
    """(subject, window_index) in D1's scored order, read from D1's own CSV — the thing D2 must reproduce."""
    p = c.out_dir / "eval/per_window_metrics.csv"
    seen, order = set(), []
    with open(p) as f:
        for r in csv.DictReader(f):
            if int(r["nfe"]) != NFE_HEADLINE:
                continue
            k = (r["subject"], int(r["window_index"]))
            if k not in seen:
                seen.add(k)
                order.append(k)
    return order


def load_split_arrays(c: C.Corpus, which: str) -> tuple[np.ndarray, np.ndarray]:
    split, _ = C.read_split_manifest(c.manifest_path)
    xs, ys = [], []
    for s in split[which]:
        d = np.load(c.processed_dir / f"{s}.npz")
        xs.append(d["x"].astype(np.float64))
        ys.append(d["y"].astype(np.float64))
    return np.concatenate(xs), np.concatenate(ys)


def score(pred: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    tab = dict(P.paper_metric_table(pred, y, FS))
    for group in M.evaluate_windows(pred, y, fs=FS).values():          # evaluate_windows nests by metric family
        for k, v in (group.items() if isinstance(group, dict) else []):
            a = np.asarray(v, dtype=np.float64)
            if a.ndim == 1 and len(a) == len(y):
                tab.setdefault(f"{k}_ew", a)
    out = {}
    for k, v in tab.items():
        a = np.asarray(v, dtype=np.float64) if np.ndim(v) else None
        if a is not None and a.ndim == 1 and len(a) == len(y):
            out[k] = a
    return out


def macro_ci(vals: np.ndarray, subj: np.ndarray) -> tuple[float, float, float]:
    """Subject-macro mean with a subject-clustered bootstrap CI — the same estimator D1 reports."""
    subs = np.unique(subj)
    per = np.array([np.nanmean(vals[subj == s]) for s in subs])
    if not np.isfinite(per).any():
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.array([np.nanmean(per[rng.integers(0, len(subs), len(subs))]) for _ in range(BOOT_N)])
    return float(np.nanmean(per)), float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def paired_ci(a: np.ndarray, b: np.ndarray, subj: np.ndarray) -> tuple[float, float, float]:
    """ours - baseline, resampling SUBJECTS (the cluster), on per-subject paired differences."""
    subs = np.unique(subj)
    per = np.array([np.nanmean(a[subj == s]) - np.nanmean(b[subj == s]) for s in subs])
    if not np.isfinite(per).any():
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.array([np.nanmean(per[rng.integers(0, len(subs), len(subs))]) for _ in range(BOOT_N)])
    return float(np.nanmean(per)), float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def run_corpus(key: str, args) -> dict:
    c = C.corpus(key)
    split, _ = C.read_split_manifest(c.manifest_path)
    C.assert_no_forbidden_subjects(split, f"d2({key})")
    cap = args.cap[key]
    x, y, sid, widx = E.load_test(c, split["test"], cap)
    x, y = x.astype(np.float64), y.astype(np.float64)

    want = d1_population(c)
    got = list(zip(sid.tolist(), widx.tolist()))
    assert got == want, f"{key}: D2 population differs from D1 ({len(got)} vs {len(want)} rows)"
    print(f"[d2] {key}: population matches D1 exactly — {len(got)} windows, {len(np.unique(sid))} subjects", flush=True)

    xt, yt = load_split_arrays(c, "train")
    if args.fit_windows and len(xt) > args.fit_windows:            # PAT/template fit budget, deterministic prefix
        xt, yt = xt[: args.fit_windows], yt[: args.fit_windows]
    t0 = time.perf_counter()
    fit = B.fit_on_train(yt, xt)
    print(f"[d2] {key}: PAT offset {fit.pat_offset} samples ({fit.pat_offset / FS * 1000:.1f} ms), "
          f"train match rate {fit.pat_train_match_rate:.3f}, {fit.n_train_beats} train beats, "
          f"fit {time.perf_counter() - t0:.0f}s", flush=True)

    partner = B.cyclic_partner(sid)
    singleton = int((partner == np.arange(len(sid))).sum())

    b1, b2, pos = B.b1_b2(x, fit)
    gt_pk = B.gt_rpeaks(y)
    b1_ceiling = B.match_rate(pos, gt_pk)     # the placement ceiling for B1/B2 on THIS test split, diagnostic
    preds = {
        "B0_wrong_window": y[partner],
        "B1_ppg_template": b1,
        "B2_ppg_mean_beat": b2,
        "B3_train_mean": B.b3(len(y), fit),
        "B5_gt_timing": B.b5_gt_timing(y, fit),
    }
    # ours@NFE1 and B4 are generated over the FULL population in one code path with the SAME noise tensor, so the
    # only difference between them is which window's PPG goes in. Reading `ours` from waveforms_nfe1.npz instead
    # would have scored it on that file's 16-rows-per-subject subset while every baseline saw all rows — the
    # paired comparison would not have been paired.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ck, _tn = E.build_net(c.checkpoint_path, dev)
    e_all = torch.randn(len(x), 1, x.shape[1], generator=torch.Generator().manual_seed(E.SOURCE_SEED))
    preds["ours_nfe1"], _dt = E.generate(net, x.astype(np.float32), e_all, NFE_HEADLINE, args.batch, dev)
    if not args.no_b4:
        preds["B4_mismatched_ppg"], _dt = E.generate(net, x[partner].astype(np.float32), e_all,
                                                     NFE_HEADLINE, args.batch, dev)
    del net
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    scored = {a: score(p, y) for a, p in preds.items()}
    keys = sorted(set().union(*[set(v) for v in scored.values()]))
    rows, prows = [], []
    for a in ARMS:
        if a not in scored:
            continue
        for k in keys:
            v = scored[a].get(k)
            if v is None:
                continue
            m, lo, hi = macro_ci(v, sid)
            rows.append({"corpus": key, "arm": a, "leakage": LEAKY.get(a, ""), "metric": k,
                         "subject_macro_mean": m, "ci_lo": lo, "ci_hi": hi,
                         "pooled_window_mean": float(np.nanmean(v)), "n_windows": int(np.isfinite(v).sum())})
            if a != "ours_nfe1" and "ours_nfe1" in scored and k in scored["ours_nfe1"]:
                d, dlo, dhi = paired_ci(scored["ours_nfe1"][k], v, sid)
                prows.append({"corpus": key, "baseline": a, "leakage": LEAKY.get(a, ""), "metric": k,
                              "ours_minus_baseline": d, "ci_lo": dlo, "ci_hi": dhi,
                              "excludes_zero": bool(np.isfinite(dlo) and (dlo > 0 or dhi < 0))})
    return {"corpus": key, "rows": rows, "paired": prows, "n_windows": len(y),
            "b1_placement_ceiling_test": float(b1_ceiling),
            "n_subjects": int(len(np.unique(sid))), "singleton_subject_rows": singleton,
            "pat_offset_samples": fit.pat_offset, "pat_train_match_rate": fit.pat_train_match_rate,
            "n_train_beats": fit.n_train_beats, "n_train_windows_used": int(len(xt)),
            "qrs_template_sha": sha(fit.qrs_template), "mean_beat_sha": sha(fit.mean_beat),
            "mean_waveform_sha": sha(fit.mean_waveform),
            "checkpoint": str(c.checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": C.sha256_file(c.checkpoint_path) if c.checkpoint_path.exists() else None,
            "arms": [a for a in ARMS if a in scored]}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", choices=list(C.BENCH_KEYS), default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--fit-windows", type=int, default=4000, help="deterministic prefix of train windows for the fit")
    ap.add_argument("--no-b4", action="store_true", help="skip the mismatched-PPG control (needs the checkpoint/GPU)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    args.cap = {"dalia": 0, "bidmc": 0, "capnobase": 0, "wildppg": 1024, "vitaldb": 1024}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    allrows, allpaired, meta = [], [], []
    for k in (args.corpus or list(C.BENCH_KEYS)):
        r = run_corpus(k, args)
        allrows += r.pop("rows")
        allpaired += r.pop("paired")
        meta.append(r)
        print(f"[d2] {k}: done ({r['n_subjects']} subjects, {r['n_windows']} windows)", flush=True)

    write_csv(out / "summary_by_arm.csv", allrows)
    write_csv(out / "paired_vs_ours.csv", allpaired)
    (out / "d2_meta.json").write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": C.git_sha(), "fs": FS, "nfe": NFE_HEADLINE,
         "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED}, "noise_seed": args.noise_seed, "caps": args.cap,
         "arms": list(ARMS), "leakage_labels": LEAKY, "corpora": meta,
         "preregistration": "docs/D2_BASELINE_FLOOR_PREREGISTRATION.md"}, indent=1))
    print(f"[d2] wrote {out}/summary_by_arm.csv ({len(allrows)} rows), paired_vs_ours.csv ({len(allpaired)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
