"""R1 step 7 — non-learned rhythm audit + cohort manifest. CPU only, no model.

PPI (PPG pulse intervals) vs RR (GT R-peak intervals), on one-to-one matched consecutive beats.
Measures rhythm-interval observability, NOT absolute R phase (that is V1's separate result).
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv, json, subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.evaluation import v1_timing as V
from ppg2ecg.evaluation.event_reliability import assert_no_test_subjects
from ppg2ecg.probes import r1_cohort as C

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/r1_global_rhythm"
FS, T_LEN = 128, 1024


def wcsv(p, rows):
    if rows:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _one(args):
    ecg, ppg = args
    r = R.detect_rpeaks(np.asarray(ecg, dtype=np.float64), FS)
    p = S1.dsp_ppg_peaks(np.asarray(ppg, dtype=np.float64), FS)
    pairs, _, _ = V.match_r_to_ppg(r, p, T_LEN, FS)
    out = []
    for (i, j), (i2, j2) in zip(pairs[:-1], pairs[1:]):
        if i2 != i + 1:                                     # only consecutive matched beats
            continue
        f1 = V.ppg_foot(ppg, int(p[j]), int(p[j - 1]) if j > 0 else None, FS)
        f2 = V.ppg_foot(ppg, int(p[j2]), int(p[j2 - 1]) if j2 > 0 else None, FS)
        out.append({"rr_ms": (r[i2] - r[i]) / FS * 1000.0,
                    "ppi_peak_ms": (p[j2] - p[j]) / FS * 1000.0,
                    "ppi_foot_ms": ((f2 - f1) / FS * 1000.0) if (f1 is not None and f2 is not None) else np.nan})
    return out, len(r)


def summarize(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return {"n": int(a.size)}
    e = np.abs(a - b); rel = e / np.maximum(b, 1e-9)
    return {"n": int(a.size), "mae_ms": float(e.mean()), "median_ae_ms": float(np.median(e)),
            "rmse_ms": float(np.sqrt(np.mean(e ** 2))),
            "pearson": float(pearsonr(a, b)[0]), "spearman": float(spearmanr(a, b)[0]),
            "rel_median": float(np.median(rel)),
            **{f"within_{t}ms": float(np.mean(e <= t)) for t in (25, 50, 100, 150)},
            "within_10pct": float(np.mean(rel <= 0.10)), "within_20pct": float(np.mean(rel <= 0.20))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    subs = C.TRAIN12 + C.VAL
    assert_no_test_subjects(subs)
    split = C.internal_dev_split()
    (OUT / "subject_split.json").write_text(json.dumps({**{k: list(v) for k, v in split.items()},
                                                        "validation": list(C.VAL), "dev_salt": C.DEV_SALT}, indent=2))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    coh, pairs_rows, summ = [], [], []
    for sub in subs:
        role = "validation" if sub in C.VAL else ("internal_dev" if sub in split["internal_dev"] else "probe_train")
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        Xs, Ys, WIs = d["x"], d["y"], d["window_index"]   # decompress ONCE: every d[key] access re-reads the npz
        pos = C.cohort_positions(sub, d["site"], d["window_index"], C.n_per_for(sub))
        for site in C.SITES:
            idx = pos[site]
            for k in idx:
                coh.append({"subject": sub, "role": role, "site": site, "array_pos": int(k),
                            "window_index": int(WIs[k])})
            ecg = [Ys[int(i)].astype(np.float64) for i in idx]
            ppg = [Xs[int(i)].astype(np.float64) for i in idx]
            with ProcessPoolExecutor(max_workers=12) as ex:
                res = list(ex.map(_one, list(zip(ecg, ppg)), chunksize=16))
            rr, pk, ft, nb = [], [], [], 0
            for (rows, n_r), k in zip(res, idx):
                nb += n_r
                for r_ in rows:
                    pairs_rows.append({"subject": sub, "role": role, "site": site,
                                       "window_index": int(WIs[int(k)]), **r_})
                    rr.append(r_["rr_ms"]); pk.append(r_["ppi_peak_ms"]); ft.append(r_["ppi_foot_ms"])
            sp = summarize(pk, rr); sf = summarize(ft, rr)
            summ.append({"subject": sub, "role": role, "site": site, "n_windows": int(idx.size), "n_gt_beats": nb,
                         **{f"peak_{k}": v for k, v in sp.items()}, **{f"foot_{k}": v for k, v in sf.items()}})
            print(f"[A] {sub:4s} {site:8s} n={idx.size:5d} pairs={sp.get('n',0):6d} | PEAK medAE {sp.get('median_ae_ms',float('nan')):6.1f} "
                  f"corr {sp.get('pearson',float('nan')):.3f} <=100ms {sp.get('within_100ms',float('nan')):.3f} | "
                  f"FOOT medAE {sf.get('median_ae_ms',float('nan')):6.1f} corr {sf.get('pearson',float('nan')):.3f}", flush=True)
    for role in ("probe_train", "internal_dev", "validation"):
        for site in list(C.SITES) + ["ALL"]:
            sel = [r for r in pairs_rows if r["role"] == role and (site == "ALL" or r["site"] == site)]
            sp = summarize([r["ppi_peak_ms"] for r in sel], [r["rr_ms"] for r in sel])
            sf = summarize([r["ppi_foot_ms"] for r in sel], [r["rr_ms"] for r in sel])
            summ.append({"subject": f"__{role}__", "role": role, "site": site, "n_windows": -1, "n_gt_beats": -1,
                         **{f"peak_{k}": v for k, v in sp.items()}, **{f"foot_{k}": v for k, v in sf.items()}})
            if site == "ALL":
                print(f"[S] {role:13s} PEAK medAE {sp['median_ae_ms']:6.1f} MAE {sp['mae_ms']:6.1f} corr {sp['pearson']:.3f} "
                      f"<=100 {sp['within_100ms']:.3f} | FOOT medAE {sf['median_ae_ms']:6.1f} MAE {sf['mae_ms']:6.1f} "
                      f"corr {sf['pearson']:.3f} <=100 {sf['within_100ms']:.3f}", flush=True)
    wcsv(OUT / "cohort_manifest.csv", coh)
    wcsv(OUT / "ppi_rr_pairs.csv", pairs_rows)
    wcsv(OUT / "ppi_rr_summary.csv", summ)
    (OUT / "provenance_audit.json").write_text(json.dumps({
        "head": head, "utc": datetime.now(timezone.utc).isoformat(), "prereg": "c7481f9",
        "cohort_windows": len(coh), "pairs": len(pairs_rows), "test_subjects_loaded": [], "model": None}, indent=2))
    print(f"\n[done] cohort {len(coh)} windows, {len(pairs_rows)} consecutive matched pairs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
