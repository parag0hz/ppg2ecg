"""V1 part 1 — cohorts + the ECG-R -> PPG-pulse delay audit and timing-prior feasibility. CPU only.

Frozen protocol: docs/V1_STEPWISE_VISUALIZATION_PREREGISTRATION.md (a73cafa).
NO MODEL IS LOADED. NO TRAINING. Test subjects kjd/ssx are never loaded.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv, json, subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.evaluation import v1_timing as V
from ppg2ecg.evaluation.event_reliability import assert_no_test_subjects

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v1_stepwise_visualization"
FS, T_LEN, WORKERS = 128, 1024, 12
SUBJECTS = V.TRAIN + V.VAL


def wcsv(p, rows):
    if rows:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _pair(args):
    """One window: GT R-peaks, PPG systolic peaks, matched delays, foot proxies."""
    ecg, ppg = args
    r = R.detect_rpeaks(np.asarray(ecg, dtype=np.float64), FS)
    p = S1.dsp_ppg_peaks(np.asarray(ppg, dtype=np.float64), FS)
    pairs, bnd, unm = V.match_r_to_ppg(r, p, T_LEN, FS)
    rows = []
    for i, j in pairs:
        prev = int(p[j - 1]) if j > 0 else None
        foot = V.ppg_foot(ppg, int(p[j]), prev, FS)
        rr = (int(r[i]) - int(r[i - 1])) / FS * 1000.0 if i > 0 else np.nan
        rows.append({"r_sample": int(r[i]), "ppg_peak_sample": int(p[j]),
                     "delay_samples": int(p[j] - r[i]), "delay_ms": float(p[j] - r[i]) / FS * 1000.0,
                     "preceding_RR_ms": rr, "estimated_HR": (60000.0 / rr) if np.isfinite(rr) and rr > 0 else np.nan,
                     "foot_sample": -1 if foot is None else int(foot),
                     "delay_foot_ms": np.nan if foot is None else float(foot - r[i]) / FS * 1000.0})
    return rows, len(r), len(p), bnd, unm


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_no_test_subjects(SUBJECTS)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print(f"[prov] HEAD {head} | {len(SUBJECTS)} subjects, {len(V.SITES)} sites", flush=True)

    coh_rows, delay_rows, skipped = [], [], []
    for sub in SUBJECTS:
        split = "train" if sub in V.TRAIN else "val"
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        c = V.cohorts(sub, d["site"], d["window_index"])
        for site in V.SITES:
            info = c[site]
            for name in ("viz", "metrics", "delay"):
                for pos in info[name]:
                    coh_rows.append({"subject": sub, "split": split, "site": site, "cohort": name,
                                     "array_pos": int(pos), "window_index": int(d["window_index"][pos])})
            idx = info["delay"]
            if idx.size == 0:
                skipped.append({"subject": sub, "site": site, "reason": "no windows for this site"})
                continue
            ecg = [d["y"][int(i)].astype(np.float64) for i in idx]
            ppg = [d["x"][int(i)].astype(np.float64) for i in idx]
            bad = [k for k, (a, b) in enumerate(zip(ecg, ppg))
                   if not (np.isfinite(a).all() and np.isfinite(b).all()) or a.std() < 1e-8 or b.std() < 1e-8]
            for k in bad:
                skipped.append({"subject": sub, "site": site, "reason": "non-finite or flat window",
                                "window_index": int(d["window_index"][int(idx[k])])})
            keep = [k for k in range(len(ecg)) if k not in set(bad)]
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                res = list(ex.map(_pair, [(ecg[k], ppg[k]) for k in keep], chunksize=8))
            nb = nm = nbnd = nunm = 0
            for k, (rows, n_r, n_p, b_, u_) in zip(keep, res):
                nb += n_r; nm += len(rows); nbnd += b_; nunm += u_
                for r_ in rows:
                    delay_rows.append({"subject": sub, "split": split, "site": site,
                                       "window_index": int(d["window_index"][int(idx[k])]), **r_})
            print(f"[D] {sub:4s} {site:8s} windows {len(keep):4d} GT beats {nb:6d} matched {nm:6d} "
                  f"({nm / max(nb, 1):.3f}) boundary {nbnd} unmatched {nunm}", flush=True)
    wcsv(OUT / "cohort_manifest.csv", coh_rows)
    wcsv(OUT / "r_to_ppg_peak_delays.csv", delay_rows)
    wcsv(OUT / "skipped_windows.csv", skipped)
    print(f"[D] total matched pairs {len(delay_rows)} | skipped {len(skipped)}", flush=True)

    D = {k: np.asarray([r[k] for r in delay_rows], dtype=object if k in ("subject", "split", "site") else float)
         for k in ("subject", "split", "site", "delay_ms", "delay_foot_ms", "preceding_RR_ms", "estimated_HR")}
    foot_fail = float(np.mean(~np.isfinite(D["delay_foot_ms"])))
    foot_ok = foot_fail <= V.FOOT_FAIL_ABORT
    print(f"[F] PPG-foot failure rate {foot_fail:.4f} (abort threshold {V.FOOT_FAIL_ABORT}) -> "
          f"{'RETAINED' if foot_ok else 'OMITTED'}", flush=True)

    summ = []
    for sub in SUBJECTS:
        for site in list(V.SITES) + ["ALL"]:
            m = (D["subject"] == sub) & ((D["site"] == site) if site != "ALL" else True)
            s = V.delay_summary(D["delay_ms"][m])
            row = {"subject": sub, "split": "train" if sub in V.TRAIN else "val", "site": site, **s}
            if foot_ok:
                row |= {f"foot_{k}": v for k, v in V.delay_summary(D["delay_foot_ms"][m]).items()}
            summ.append(row)
    for site in list(V.SITES) + ["ALL"]:
        for sp in ("train", "val", "all"):
            m = ((D["site"] == site) if site != "ALL" else np.ones(len(D["site"]), bool)) & \
                ((D["split"] == sp) if sp != "all" else np.ones(len(D["split"]), bool))
            summ.append({"subject": f"__{sp.upper()}__", "split": sp, "site": site, **V.delay_summary(D["delay_ms"][m])})
    wcsv(OUT / "delay_summary.csv", summ)
    for site in list(V.SITES) + ["ALL"]:
        for sp in ("train", "val"):
            r_ = next(x for x in summ if x["subject"] == f"__{sp.upper()}__" and x["site"] == site)
            print(f"[S] {sp:5s} {site:8s} n {r_['n']:7d} median {r_['median']:7.1f} ms  IQR {r_['iqr']:6.1f}  "
                  f"p5-p95 {r_['p5']:6.1f}-{r_['p95']:6.1f}  CV {r_['cv']:.3f}", flush=True)

    # ---------------- timing prior: TRAIN-ONLY statistics, validated on an0/k2s ----------------
    tr = D["split"] == "train"
    global_delay = float(np.median(D["delay_ms"][tr]))
    site_delay = {s: float(np.median(D["delay_ms"][tr & (D["site"] == s)])) for s in V.SITES}
    rr_tr = D["preceding_RR_ms"][tr]
    edges = np.percentile(rr_tr[np.isfinite(rr_tr)], [100 / 3, 200 / 3])
    hr_delay = {}
    for b in range(3):
        lo = -np.inf if b == 0 else edges[b - 1]
        hi = np.inf if b == 2 else edges[b]
        m = tr & np.isfinite(D["preceding_RR_ms"]) & (D["preceding_RR_ms"] >= lo) & (D["preceding_RR_ms"] < hi)
        hr_delay[b] = float(np.median(D["delay_ms"][m]))
    print(f"[T] train-only global delay {global_delay:.1f} ms | site {site_delay} | "
          f"RR tercile edges {edges.round(1).tolist()} -> {hr_delay}", flush=True)

    val_rows = []
    for sub in V.VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        c = V.cohorts(sub, d["site"], d["window_index"])
        for site in V.SITES:
            idx = c[site]["delay"]
            ecg = [d["y"][int(i)].astype(np.float64) for i in idx]
            ppg = [d["x"][int(i)].astype(np.float64) for i in idx]
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                rr_ = list(ex.map(_pair, list(zip(ecg, ppg)), chunksize=8))
            for k, (rows, n_r, n_p, _b, _u) in enumerate(rr_):
                r_true = R.detect_rpeaks(ecg[k], FS)
                p_ppg = S1.dsp_ppg_peaks(ppg[k], FS)
                if r_true.size == 0 or p_ppg.size == 0:
                    continue
                variants = {"A_global": np.full(p_ppg.size, global_delay),
                            "B_site": np.full(p_ppg.size, site_delay[site])}
                # C: HR-conditioned. Each PPG peak inherits the tercile of the RR preceding its matched R.
                cdel = np.full(p_ppg.size, hr_delay[1])
                for r_ in rows:
                    j = int(np.flatnonzero(p_ppg == r_["ppg_peak_sample"])[0])
                    rrv = r_["preceding_RR_ms"]
                    b = 0 if (np.isfinite(rrv) and rrv < edges[0]) else (2 if (np.isfinite(rrv) and rrv >= edges[1]) else 1)
                    cdel[j] = hr_delay[b]
                variants["C_hr"] = cdel
                for name, dl in variants.items():
                    pred = p_ppg - dl / 1000.0 * FS
                    err = V.nearest_abs_error_ms(r_true, pred, FS)
                    val_rows.append({"subject": sub, "site": site, "predictor": name,
                                     "n_gt_beats": int(r_true.size), "n_pred": int(pred.size),
                                     "mae_ms": float(np.mean(err)), "median_ae_ms": float(np.median(err)),
                                     **{f"cov_{t}ms": float(np.mean(err <= t)) for t in (25, 50, 100, 150)}})
    wcsv(OUT / "timing_prior_validation.csv", val_rows)
    agg = []
    for name in ("A_global", "B_site", "C_hr"):
        m = [r for r in val_rows if r["predictor"] == name]
        w = np.asarray([r["n_gt_beats"] for r in m], float)
        agg.append({"predictor": name, "n_gt_beats": int(w.sum()),
                    "n_pred": int(sum(r["n_pred"] for r in m)),
                    "mae_ms": float(np.average([r["mae_ms"] for r in m], weights=w)),
                    "median_ae_ms": float(np.average([r["median_ae_ms"] for r in m], weights=w)),
                    **{f"cov_{t}ms": float(np.average([r[f"cov_{t}ms"] for r in m], weights=w)) for t in (25, 50, 100, 150)}})
        print(f"[V] {name:9s} MAE {agg[-1]['mae_ms']:7.1f} ms  medAE {agg[-1]['median_ae_ms']:7.1f}  "
              f"cov25 {agg[-1]['cov_25ms']:.3f} cov50 {agg[-1]['cov_50ms']:.3f} "
              f"cov100 {agg[-1]['cov_100ms']:.3f} cov150 {agg[-1]['cov_150ms']:.3f}", flush=True)
    wcsv(OUT / "timing_prior_summary.csv", agg)

    (OUT / "provenance_delay.json").write_text(json.dumps({
        "head": head, "utc": datetime.now(timezone.utc).isoformat(), "protocol": "a73cafa",
        "subjects": list(SUBJECTS), "test_subjects_loaded": [], "model_loaded": False, "training": False,
        "delay_window_ms": [V.DELAY_LO_MS, V.DELAY_HI_MS], "salt": V.SALT,
        "cohort_sizes": {"viz": V.VIZ_N, "metrics": V.METRICS_N, "delay": V.DELAY_N},
        "train_only_global_delay_ms": global_delay, "train_only_site_delay_ms": site_delay,
        "train_rr_tercile_edges_ms": edges.tolist(), "train_only_hr_delay_ms": hr_delay,
        "ppg_foot_failure_rate": foot_fail, "ppg_foot_retained": bool(foot_ok),
        "n_matched_pairs": len(delay_rows), "n_skipped_windows": len(skipped)}, indent=2))
    print("\n[done] delay audit complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
