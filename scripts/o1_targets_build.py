"""O1 step 3 — build the frozen cohort, extract every ECG component target once per unique ECG window,
and write the validity / variability / train-only scaling artifacts (preregistration sections 4-8).

Read-only with respect to models: no probe is built or trained here. ECG is a TRAINING LABEL only.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o1_component_extractability"
CACHE = ART / "_cache_targets.npz"
SPLIT = C.internal_dev_split()
PROBE_TRAIN, INTERNAL_DEV, VALIDATION = SPLIT["probe_train"], SPLIT["internal_dev"], C.VAL
ROLE = {**{s: "probe_train" for s in PROBE_TRAIN}, **{s: "internal_dev" for s in INTERNAL_DEV},
        **{s: "validation" for s in VALIDATION}}


def _targets(y):
    t = OT.window_targets(np.asarray(y, dtype=np.float64))
    return [t[k] for k in OT.TARGETS] + [float(t["n_valid_qrs_beats"])]


def build_cohort():
    """R1's frozen balanced cohort, rebuilt exactly (2,048 per site for train/dev, 1,024 for validation)."""
    rows = []
    for s in list(PROBE_TRAIN) + list(INTERNAL_DEV) + list(VALIDATION):
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        SITE, WI = np.asarray(d["site"]).astype(str), d["window_index"].astype(np.int64)
        pos = C.cohort_positions(s, SITE, WI, C.n_per_for(s))
        for site in C.SITES:
            for p in pos[site]:
                rows.append({"subject": s, "role": ROLE[s], "site": site, "array_pos": int(p),
                             "window_index": int(WI[p])})
    return rows


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(list(PROBE_TRAIN) + list(INTERNAL_DEV) + list(VALIDATION))
    t0 = time.perf_counter()
    rows = build_cohort()
    with open(ART / "cohort_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    uniq = sorted({(r["subject"], r["window_index"]) for r in rows})
    print(f"[cohort] {len(rows)} rows | {len(uniq)} unique ECG windows | "
          f"train {sum(r['role']=='probe_train' for r in rows)} dev {sum(r['role']=='internal_dev' for r in rows)} "
          f"val {sum(r['role']=='validation' for r in rows)}", flush=True)

    ecg = {}
    for s in sorted({u[0] for u in uniq}):
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Y, WI = d["y"], d["window_index"].astype(np.int64)
        first = {}
        for i in range(len(WI)):
            first.setdefault(int(WI[i]), i)
        want = [w for (sub, w) in uniq if sub == s]
        ecg[s] = (np.stack([Y[first[w]] for w in want]).astype(np.float64), want)
    flat = [(s, w) for s in ecg for w in ecg[s][1]]
    Yall = np.concatenate([ecg[s][0] for s in ecg])
    with ProcessPoolExecutor(max_workers=12) as ex:
        T = np.asarray(list(ex.map(_targets, list(Yall), chunksize=32)), dtype=np.float64)
    key = {(s, w): i for i, (s, w) in enumerate(flat)}
    np.savez_compressed(CACHE, targets=T, subjects=np.array([s for s, _ in flat]),
                        window_index=np.array([w for _, w in flat], dtype=np.int64), names=np.array(OT.TARGETS))
    print(f"[targets] {T.shape[0]} unique ECG windows scored in {time.perf_counter()-t0:.0f} s", flush=True)

    role_of = {s: ROLE[s] for s in ROLE}
    idx_role = {r: np.array([key[(s, w)] for (s, w) in flat if role_of[s] == r]) for r in ("probe_train", "internal_dev", "validation")}
    val_rows = [r for r in rows if r["role"] == "validation"]
    n_val_unique = len({(r["subject"], r["window_index"]) for r in val_rows})

    validity, variability, scaling = [], [], {}
    for j, t in enumerate(OT.TARGETS):
        v_tr, v_dev, v_val = (T[idx_role[r], j] for r in ("probe_train", "internal_dev", "validation"))
        fin_val = np.isfinite(v_val)
        frac_val = float(fin_val.mean())
        med, iqr = float(np.nanmedian(v_tr)), float(np.nanpercentile(v_tr, 75) - np.nanpercentile(v_tr, 25))
        status = "PRIMARY" if frac_val >= OT.MIN_VALID_FRACTION else "SECONDARY / INSUFFICIENT COVERAGE"
        if not np.isfinite(iqr) or iqr <= 1e-12:
            status = "INSUFFICIENT TARGET VARIATION"
        validity.append({"target": t, "id": OT.TARGET_IDS[t], "unit": OT.UNITS[t],
                         "n_train": int(v_tr.size), "valid_train": int(np.isfinite(v_tr).sum()),
                         "n_dev": int(v_dev.size), "valid_dev": int(np.isfinite(v_dev).sum()),
                         "n_validation_unique_ecg": int(v_val.size), "valid_validation": int(fin_val.sum()),
                         "valid_fraction_validation": frac_val, "status": status,
                         "missing_reason": "fewer than 2 R peaks" if t in ("median_RR_ms",) else
                                           ("fewer than 3 R peaks" if t == "RR_IQR_ms" else
                                            ("no GT beat whose QRS core fits the window" if t.startswith("median_QRS") else "none"))})
        scaling[t] = {"center_train_median": med, "scale_train_IQR": iqr, "status": status}
        # variability once per UNIQUE ECG window
        parts, subs = [], []
        for r in ("probe_train", "internal_dev", "validation"):
            parts.append(T[idx_role[r], j]); subs.append(np.array([s for (s, _w) in flat if role_of[s] == r]))
        allv, alls = np.concatenate(parts), np.concatenate(subs)
        ok = np.isfinite(allv)
        allv, alls = allv[ok], alls[ok]
        smeans = np.array([allv[alls == s].mean() for s in sorted(set(alls.tolist()))])
        within = float(np.mean([allv[alls == s].var(ddof=0) for s in sorted(set(alls.tolist()))]))
        variability.append({"target": t, "id": OT.TARGET_IDS[t],
                            "train_median": med, "train_IQR": iqr,
                            "validation_median": float(np.nanmedian(v_val)),
                            "validation_IQR": float(np.nanpercentile(v_val, 75) - np.nanpercentile(v_val, 25)),
                            "total_variance": float(allv.var(ddof=0)),
                            "between_subject_variance": float(smeans.var(ddof=0)),
                            "within_subject_variance": within,
                            "within_over_total": float(within / max(allv.var(ddof=0), 1e-30))})
    for name, data in (("target_validity.csv", validity), ("target_variability.csv", variability)):
        with open(ART / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (ART / "target_scaling.json").write_text(json.dumps(
        {"rule": "y_z = (y - TRAIN median) / TRAIN IQR; validation statistics are never used",
         "computed_on": "unique ECG windows of the probe_train subjects", "targets": scaling}, indent=2))
    (ART / "target_definitions.json").write_text(json.dumps(
        {"targets": {t: {"id": OT.TARGET_IDS[t], "unit": OT.UNITS[t]} for t in OT.TARGETS},
         "detector": "ppg2ecg.evaluation.rpeaks.detect_rpeaks (neurokit, frozen R1/Q1/M1 configuration)",
         "qrs_core": {"CORE_samples": int(__import__("ppg2ecg.evaluation.m1_structural", fromlist=["CORE"]).CORE),
                      "beat_validity": "r-CORE-1 >= 0 and r+CORE+2 <= 1024 (M1 verbatim)",
                      "d1": "m1_structural.d1 (np.diff, no fs scaling)", "d2": "m1_structural.d2"},
         "width": "ppg2ecg.evaluation.rpeaks.qrs_width_ms (frozen; q_win 0.08 s, s_win 0.12 s)",
         "hf": "ppg2ecg.evaluation.metrics.hf_energy_ratio (>= 15 Hz, frozen)",
         "per_window_aggregation": "MEDIAN over valid GT beats"}, indent=2))
    prov = {"git": git_sha(ROOT), "utc": datetime.now(timezone.utc).isoformat(), "test_subjects_loaded": [],
            "cohort": {"probe_train": list(PROBE_TRAIN), "internal_dev": list(INTERNAL_DEV), "validation": list(VALIDATION),
                       "n_rows": len(rows), "n_unique_ecg_windows": len(uniq), "n_validation_unique_ecg": n_val_unique,
                       "per_site_cap_train": C.N_TRAIN_PER, "per_site_cap_validation": C.N_VAL_PER, "salt": C.COHORT_SALT},
            "libs": {"numpy": np.__version__, "python": platform.python_version()},
            "wall_s": time.perf_counter() - t0}
    (ART / "target_build_provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    for v in validity:
        print(f"  {v['id']} {v['target']:<32} valid_val {v['valid_fraction_validation']:.4f}  {v['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
