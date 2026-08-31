"""S1 G1 — HARD GATE ONLY: can the frozen event metric reward exact beat placement?

Protocol: docs/S1_METRIC_VALIDITY_PREREGISTRATION.md (b749339)
          + docs/S1_METRIC_VALIDITY_AMENDMENT_1.md (dc75079)

Runs G1 and nothing else. S1.2-S1.6 are NOT implemented here by design.
NO TRAINING. No model is loaded. No checkpoint is read or written. Test subjects are never touched.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401  (must precede torch/numpy-MKL work)

import json
import hashlib
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ppg2ecg.evaluation import stamping as ST
from ppg2ecg.evaluation.event_reliability import assert_no_test_subjects, select_subset
from ppg2ecg.evaluation.rpeaks import detect_rpeaks, match_rpeaks, prf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/wildppg_8s"
OUT = ROOT / "artifacts/s1_metric_validity"
FIG = OUT / "figures"

FS = 128
VAL_SUBJECTS = ("an0", "k2s")
NFE_SALT, NFE_TAKE = "x4-event-nfe-v2", 1024          # the frozen X4-0 stage-B population
TOLERANCES = (50.0, 75.0, 150.0)
GATE_ARM, GATE_TOL, GATE_THRESHOLD = "T-B", 50.0, 0.95


def _ecg(subject: str) -> np.ndarray:
    assert_no_test_subjects([subject])
    return np.load(DATA / f"{subject}.npz")["y"]


def _peaks_one(sig: np.ndarray) -> np.ndarray:
    return detect_rpeaks(np.asarray(sig, dtype=np.float64), FS)


def _peaks_many(sigs, workers: int = 12):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_peaks_one, sigs, chunksize=16))


def _score(ref, pred, tol_ms):
    m, fp, fn = match_rpeaks(ref, pred, FS, tol_ms=tol_ms)
    p, r, f = prf(len(m), fp, fn)
    return {"n_ref": int(len(ref)), "n_pred": int(len(pred)), "n_matched": int(len(m)),
            "missing": int(fn), "spurious": int(fp), "precision": p, "recall": r, "f1": f,
            "beats_ratio": (len(pred) / len(ref)) if len(ref) else np.nan}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    assert_no_test_subjects(VAL_SUBJECTS)
    geo = ST.template_geometry(FS)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print(f"[prov] HEAD {head}\n[prov] geometry {geo}", flush=True)

    # ---------------------------------------------------------------- Template A (train-only)
    print(f"[T] extracting train beats from {len(ST.TEMPLATE_SUBJECTS)} subjects "
          f"(excluding noisy-ECG {ST.TEMPLATE_EXCLUDED_NOISY})", flush=True)
    beats = ST.collect_train_beats(_ecg)
    t_a, meta = ST.build_template_a(beats)
    t_b = ST.crop_qrs(t_a, FS)
    t_c = ST.analytic_template_c(meta["final_ptp"], FS)
    np.save(OUT / "template_A.npy", t_a)
    meta |= {"sha256": ST.sha256_array(t_a), "sha256_file": hashlib.sha256((OUT / "template_A.npy").read_bytes()).hexdigest(),
             "subjects_included": list(ST.TEMPLATE_SUBJECTS), "subjects_excluded_noisy": list(ST.TEMPLATE_EXCLUDED_NOISY),
             "salt": ST.TEMPLATE_SALT, "n_take_per_subject": ST.TEMPLATE_N_TAKE,
             "qrs_crop_sha256": ST.sha256_array(t_b), "qrs_len": int(t_b.size), "geometry": geo}
    (OUT / "template_A_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[T] beats {meta['n_beats']} | len {meta['length']} | raw ptp {meta['raw_ptp']:.6f} "
          f"| target {meta['a_target_median_ptp']:.6f} | final {meta['final_ptp']:.6f}\n"
          f"[T] sha256 {meta['sha256']}", flush=True)

    # ---------------------------------------------------------------- frozen population + GT peaks
    pop, gt_sigs, gt_peaks = {}, [], []
    for s in VAL_SUBJECTS:
        y = _ecg(s)
        idx = select_subset(NFE_SALT, s, y.shape[0], NFE_TAKE)
        pop[s] = idx
        gt_sigs += [y[int(i)].astype(np.float64) for i in idx]
        print(f"[P] {s}: {idx.size} windows of {y.shape[0]}", flush=True)
    subj_of = np.array([s for s in VAL_SUBJECTS for _ in pop[s]])
    widx_of = np.concatenate([pop[s] for s in VAL_SUBJECTS])
    print(f"[P] detecting GT R-peaks on {len(gt_sigs)} windows", flush=True)
    gt_peaks = _peaks_many(gt_sigs)
    n_gt = int(sum(len(p) for p in gt_peaks))
    print(f"[P] GT beats {n_gt}", flush=True)

    # ---------------------------------------------------------------- Amendment 1 §1B overlap assertion
    need = geo["min_rr_samples_for_no_overlap"]
    rrs = np.concatenate([np.diff(p) for p in gt_peaks if len(p) >= 2])
    viol = [(subj_of[i], int(widx_of[i]), int(np.min(np.diff(p))))
            for i, p in enumerate(gt_peaks) if len(p) >= 2 and np.min(np.diff(p)) < need]
    overlap_check = {"required_min_rr_samples": int(need), "required_min_rr_ms": need / FS * 1000,
                     "observed_min_rr_samples": int(rrs.min()), "observed_min_rr_ms": float(rrs.min()) / FS * 1000,
                     "n_rr_intervals": int(rrs.size), "n_violating_windows": len(viol),
                     "violations": viol[:50], "passed": len(viol) == 0}
    (OUT / "overlap_check.json").write_text(json.dumps(overlap_check, indent=2))
    print(f"[O] min RR {overlap_check['observed_min_rr_samples']} samples "
          f"({overlap_check['observed_min_rr_ms']:.1f} ms); need >= {need} "
          f"({need / FS * 1000:.2f} ms) -> {'OK' if overlap_check['passed'] else 'VIOLATED'}", flush=True)
    if not overlap_check["passed"]:
        print(f"[O] STOP: {len(viol)} windows would overlap under T-B. Per Amendment 1 §1B the frozen "
              f"[-80,+120] ms interval is NOT adjusted. Reporting and stopping before G1.", flush=True)
        return 2

    # ---------------------------------------------------------------- build the four arms
    arms = {}
    r_full, r_qrs = geo["r_index_full"], geo["r_index_qrs"]
    arms["T-A"] = [ST.stamp(t_a, p, s.size, r_full) for p, s in zip(gt_peaks, gt_sigs)]
    arms["T-B"] = [ST.stamp(t_b, p, s.size, r_qrs) for p, s in zip(gt_peaks, gt_sigs)]
    arms["T-C"] = [ST.stamp(t_c, p, s.size, r_qrs) for p, s in zip(gt_peaks, gt_sigs)]
    arms["T-D"] = [ST.stamp(t_a, p, s.size, r_full, baseline=ST.lowfreq_baseline(s, FS, 1.0))
                   for p, s in zip(gt_peaks, gt_sigs)]
    tb_overlap = sum(ST.stamp_supports_overlap(p, r_qrs, t_b.size) for p in gt_peaks)
    ta_overlap = sum(ST.stamp_supports_overlap(p, r_full, t_a.size) for p in gt_peaks)
    print(f"[S] windows with overlapping supports: T-A {ta_overlap}/{len(gt_peaks)} (expected, declared) "
          f"| T-B {tb_overlap}/{len(gt_peaks)} (must be 0)", flush=True)
    assert tb_overlap == 0, "T-B overlap detected after the arithmetic check passed"

    # ---------------------------------------------------------------- score
    rows, per_window = [], []
    for arm, sigs in arms.items():
        print(f"[G] detecting on {arm}", flush=True)
        det = _peaks_many(sigs)
        for tol in TOLERANCES:
            recs = [_score(g, d, tol) for g, d in zip(gt_peaks, det)]
            for k, rec in enumerate(recs):
                per_window.append({"arm": arm, "tol_ms": tol, "subject": subj_of[k],
                                   "window_index": int(widx_of[k]), **rec})
            f1 = np.array([r["f1"] for r in recs])
            by_s = {s: f1[subj_of == s] for s in VAL_SUBJECTS}
            agg = {"arm": arm, "tol_ms": tol,
                   "f1_macro": float(np.mean([v.mean() for v in by_s.values()])),
                   "f1_pooled": float(f1.mean()),
                   "precision_macro": float(np.mean([np.mean([r["precision"] for r, s in zip(recs, subj_of) if s == k]) for k in VAL_SUBJECTS])),
                   "recall_macro": float(np.mean([np.mean([r["recall"] for r, s in zip(recs, subj_of) if s == k]) for k in VAL_SUBJECTS])),
                   "beats_ratio_macro": float(np.mean([np.nanmean([r["beats_ratio"] for r, s in zip(recs, subj_of) if s == k]) for k in VAL_SUBJECTS])),
                   "f1_sd": float(f1.std(ddof=1)), "f1_min": float(f1.min()), "f1_max": float(f1.max()),
                   "n_f1_eq_1": int((f1 >= 1.0 - 1e-12).sum()), "frac_f1_eq_1": float((f1 >= 1.0 - 1e-12).mean()),
                   "n_f1_lt_0.5": int((f1 < 0.5).sum()), "frac_f1_lt_0.5": float((f1 < 0.5).mean()),
                   "n_ref": int(sum(r["n_ref"] for r in recs)), "n_pred": int(sum(r["n_pred"] for r in recs)),
                   "n_matched": int(sum(r["n_matched"] for r in recs)),
                   "missing": int(sum(r["missing"] for r in recs)), "spurious": int(sum(r["spurious"] for r in recs)),
                   "n_windows": int(f1.size)}
            for s in VAL_SUBJECTS:
                agg[f"f1__{s}"] = float(by_s[s].mean())
            for q in range(1, 10):
                agg[f"f1_d{q}"] = float(np.percentile(f1, q * 10))
            rows.append(agg)
            print(f"[G] {arm} @{tol:>5.0f} ms: F1 macro {agg['f1_macro']:.4f} "
                  f"(an0 {agg['f1__an0']:.4f}, k2s {agg['f1__k2s']:.4f}) "
                  f"P {agg['precision_macro']:.4f} R {agg['recall_macro']:.4f} "
                  f"beats {agg['beats_ratio_macro']:.4f} | F1=1 {agg['frac_f1_eq_1']:.3f} "
                  f"F1<0.5 {agg['frac_f1_lt_0.5']:.3f}", flush=True)

    import csv
    with open(OUT / "g1_stamping_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT / "g1_per_window.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_window[0])); w.writeheader(); w.writerows(per_window)

    # ---------------------------------------------------------------- gate
    g = next(r for r in rows if r["arm"] == GATE_ARM and r["tol_ms"] == GATE_TOL)
    ta = next(r for r in rows if r["arm"] == "T-A" and r["tol_ms"] == GATE_TOL)
    verdict = "PASS" if g["f1_macro"] >= GATE_THRESHOLD else "FAIL"
    gate = {"gate_arm": GATE_ARM, "gate_tolerance_ms": GATE_TOL, "threshold": GATE_THRESHOLD,
            "observed_f1_macro": g["f1_macro"], "verdict": verdict,
            "t_a_f1_macro_at_gate_tol": ta["f1_macro"],
            "amendment": "docs/S1_METRIC_VALIDITY_AMENDMENT_1.md (T-A -> T-B); T-A is a control only",
            "population": {s: int(pop[s].size) for s in VAL_SUBJECTS}, "n_gt_beats": n_gt,
            "overlap_check": overlap_check}
    (OUT / "g1_gate.json").write_text(json.dumps(gate, indent=2))
    (OUT / "provenance.json").write_text(json.dumps({
        "head": head, "utc": datetime.now(timezone.utc).isoformat(),
        "prereg": "b749339", "amendment_1": "dc75079", "semantics_correction": "2e0c20b",
        "python": platform.python_version(), "numpy": np.__version__,
        "test_subjects_loaded": [], "models_loaded": [], "training": False,
        "population_salt": NFE_SALT, "n_take_per_subject": NFE_TAKE,
        "template_sha256": meta["sha256"], "items_run": ["G1"],
        "items_not_run": ["S1.2", "S1.3", "S1.4", "S1.5", "S1.6"]}, indent=2))

    # ---------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = int(np.argmax([len(p) for p in gt_peaks]))                    # the densest window: the overlap case
    t = np.arange(gt_sigs[k].size) / FS
    fig, ax = plt.subplots(5, 1, figsize=(13, 10), sharex=True)
    ax[0].plot(t, gt_sigs[k], "k", lw=0.8); ax[0].set_ylabel("GT ECG")
    ax[0].plot(np.asarray(gt_peaks[k]) / FS, gt_sigs[k][gt_peaks[k]], "r.", ms=6)
    for a, arm in zip(ax[1:], ["T-A", "T-B", "T-C", "T-D"]):
        a.plot(t, arms[arm][k], lw=0.8, color="tab:green" if arm == "T-B" else "tab:blue")
        for p in gt_peaks[k]:
            a.axvline(p / FS, color="tab:red", lw=0.7, alpha=0.35)
        a.set_ylabel(arm + (" (GATE)" if arm == "T-B" else ""))
    ax[-1].set_xlabel("time (s)")
    fig.suptitle(f"S1 G1 stamping arms — {subj_of[k]} window {widx_of[k]} "
                 f"({len(gt_peaks[k])} GT beats, min RR {int(np.min(np.diff(gt_peaks[k])))} samples)\n"
                 f"red = exact GT R-peak positions; T-B is the overlap-free hard gate")
    fig.tight_layout(); fig.savefig(FIG / "g1_stamping_examples.png", dpi=110); plt.close(fig)

    pw = np.array([[r["f1"] for r in per_window if r["arm"] == a and r["tol_ms"] == 50.0] for a in arms])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for i, a in enumerate(arms):
        ax[0].hist(pw[i], bins=40, histtype="step", lw=1.6, label=f"{a} (mean {pw[i].mean():.3f})")
    ax[0].axvline(GATE_THRESHOLD, color="k", ls="--", lw=1, label=f"gate {GATE_THRESHOLD}")
    ax[0].set_xlabel("per-window F1 @50 ms"); ax[0].set_ylabel("windows"); ax[0].legend(fontsize=8)
    for i, a in enumerate(arms):
        ax[1].plot(TOLERANCES, [r["f1_macro"] for r in rows if r["arm"] == a], "o-", label=a)
    ax[1].axhline(GATE_THRESHOLD, color="k", ls="--", lw=1)
    ax[1].set_xlabel("matching tolerance (ms)"); ax[1].set_ylabel("macro F1"); ax[1].legend(fontsize=8)
    fig.suptitle(f"S1 G1 — hard gate {GATE_ARM} @{GATE_TOL:.0f} ms = {g['f1_macro']:.4f} -> {verdict}")
    fig.tight_layout(); fig.savefig(FIG / "g1_f1_distributions.png", dpi=110); plt.close(fig)

    print(f"\n[GATE] {GATE_ARM} macro F1 @{GATE_TOL:.0f} ms = {g['f1_macro']:.4f} "
          f"vs threshold {GATE_THRESHOLD} -> {verdict}", flush=True)
    print(f"[GATE] T-A (control) @{GATE_TOL:.0f} ms = {ta['f1_macro']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
