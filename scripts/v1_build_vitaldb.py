"""V1 corpus + patient-level split (docs/V1_VITALDB_PAIRED_PREREGISTRATION.md)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ppg2ecg.data.preprocess import ECG_KW, PPG_KW, preprocess_windows
from ppg2ecg.evaluation import rpeaks as R

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/VitalDB/cases"
OUT = ROOT / "data/processed/v1_vitaldb"
MAN = ROOT / "data/manifests/split_v1_vitaldb_seed42.json"
SEED, SEG, FS_RAW, FS = 20260917, 4, 500, 128
CAND = {"train": 128, "val": 32, "test": 32}
CAP = {"train": 64, "val": 16, "test": 16}


def plan():
    rows = [r for r in csv.DictReader(open(ROOT / "artifacts/v0_vitaldb_audit/cases_audit.csv")) if r["eligible"] == "True"]
    pats = sorted({int(r["subjectid"]) for r in rows})
    perm = np.random.default_rng(SEED).permutation(len(pats))
    n_te, n_va = round(0.20 * len(pats)), round(0.05 * len(pats))
    role_of = {}
    for j, i in enumerate(perm):
        role_of[pats[i]] = "test" if j < n_te else "val" if j < n_te + n_va else "train"
    return [(int(r["caseid"]), int(r["subjectid"]), role_of[int(r["subjectid"])]) for r in rows]


def one(job):
    caseid, pid, role = job
    with np.load(RAW / f"case_{caseid:05d}.npz") as z:
        ppg, ecg = z["PLETH"], z["ECG_II"]
        win = FS_RAW * SEG
        n_win = min(ppg.size, ecg.size) // win
        cand = np.unique(np.linspace(0, n_win - 1, CAND[role]).round().astype(int))
        P = np.stack([ppg[k * win:(k + 1) * win] for k in cand]).astype(np.float64)
        E = np.stack([ecg[k * win:(k + 1) * win] for k in cand]).astype(np.float64)
    ok = np.isfinite(P).all(1) & np.isfinite(E).all(1) & (P.std(1) > 0) & (E.std(1) > 0)
    cand, P, E = cand[ok], P[ok], E[ok]
    stats = {"cand": int(ok.size), "drop_raw": int((~ok).sum())}
    if len(cand) == 0:
        return caseid, pid, role, None, stats
    x = preprocess_windows(P, FS, SEG, **PPG_KW)
    y = preprocess_windows(E, FS, SEG, **ECG_KW)
    ok2 = np.isfinite(x).all(1) & np.isfinite(y).all(1)
    hr = np.array([R.hr_bpm(R.detect_rpeaks(row, FS, "neurokit"), FS) if good else np.nan for row, good in zip(y, ok2)])
    ok3 = ok2 & np.isfinite(hr) & (hr >= 30) & (hr <= 200)
    stats.update(drop_nonfinite_pre=int((~ok2).sum()), drop_ecg_quality=int((ok2 & ~ok3).sum()))
    cand, x, y = cand[ok3], x[ok3], y[ok3]
    if len(cand) > CAP[role]:
        keep = np.unique(np.linspace(0, len(cand) - 1, CAP[role]).round().astype(int))
        cand, x, y = cand[keep], x[keep], y[keep]
    stats["kept"] = int(len(cand))
    if len(cand) == 0:
        return caseid, pid, role, None, stats
    name = f"case_{caseid:05d}"
    np.savez(OUT / f"{name}.npz", x=x.astype(np.float32), y=y.astype(np.float32),
             window_index=cand.astype(np.int64), subjectid=np.int64(pid), subject=np.array(name))
    return caseid, pid, role, name, stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = plan()
    split, patient_of, counts, tot = {"train": [], "val": [], "test": []}, {}, {}, {}
    with ProcessPoolExecutor(18) as ex:
        for i, (caseid, pid, role, name, st) in enumerate(ex.map(one, jobs, chunksize=4), 1):
            for k, v in st.items():
                tot[f"{role}.{k}"] = tot.get(f"{role}.{k}", 0) + v
            if name:
                split[role].append(name); patient_of[name] = pid; counts[name] = st["kept"]
            if i % 500 == 0:
                print(f"[v1-build] {i}/{len(jobs)}", flush=True)
    pats = {r: {patient_of[c] for c in split[r]} for r in split}
    assert not (pats["train"] & pats["test"]) and not (pats["train"] & pats["val"]) and not (pats["val"] & pats["test"])
    man = {"splits": [{"protocol": "V1-patient-holdout", "seed": SEED, "corpus": "v1_vitaldb", "dataset": "VitalDB",
                       "task": "ECG", **{r: sorted(v) for r, v in split.items()}}],
           "extra": {"patient_of_case": patient_of, "windows_per_case": counts,
                     "n_cases": {r: len(v) for r, v in split.items()}, "n_patients": {r: len(v) for r, v in pats.items()},
                     "n_windows": {r: sum(counts[c] for c in split[r]) for r in split}, "drop_counts": tot,
                     "rule": "docs/V1_VITALDB_PAIRED_PREREGISTRATION.md"}}
    MAN.write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man["extra"][k] for k in ("n_cases", "n_patients", "n_windows", "drop_counts")}, indent=1))


if __name__ == "__main__":
    main()
