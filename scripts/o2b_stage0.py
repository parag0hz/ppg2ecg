"""O2b Stage 0 — integer-grid canonicalization operator repair audit.

Prechecks (spacing, integer core), then the EXACT O2 round-trip metrics and the EXACT O2 Stage-0 gate.
No generator is built, loaded or trained anywhere in this script.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import importlib.util
import inspect
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import o2_warp as O2
from ppg2ecg.evaluation import o2b_warp as B
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2b_integer_grid"
O2ART = ROOT / "artifacts/o2_oracle_canonicalization"
O1ART = ROOT / "artifacts/o1_component_extractability"
_s = importlib.util.spec_from_file_location("o2_stage0_roundtrip", ROOT / "scripts/o2_stage0_roundtrip.py")
S0 = importlib.util.module_from_spec(_s); sys.modules[_s.name] = S0; _s.loader.exec_module(S0)   # EXACT O2 metric code
VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024
ALIGNED = S0.ALIGNED


def wcsv(p, rows):
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    t0 = time.perf_counter()
    X, Y, SUB, SITE, POS, WI = S0.load_cohort()                      # exact O2 cohort + frozen-subset assert
    Yd = Y.astype(np.float64)
    gt_pk = S0.pmap(S0._peaks, list(Yd))
    gt_tg = S0.pmap(S0._targets, list(Yd))
    n_beats = int(sum(len(p) for p in gt_pk))
    if len(X) != 2048 or n_beats != 19834:
        raise RuntimeError(f"frozen population facts differ: {len(X)} windows, {n_beats} beats (STOP)")
    iqr = {t: json.loads((O1ART / "target_scaling.json").read_text())["targets"][t]["scale_train_IQR"] for t in ALIGNED}
    wcsv(ART / "cohort_manifest.csv", [{"row": i, "subject": SUB[i], "site": SITE[i], "array_pos": int(POS[i]),
                                        "window_index": int(WI[i]), "K": int(len(gt_pk[i]))} for i in range(len(X))])
    print(f"[P] {len(X)} windows, {n_beats} GT beats", flush=True)

    # ---------------- operator + prechecks ----------------
    warps = [B.IntegerEventWarp(gt_pk[i]) for i in range(len(Y))]
    sched, spacing, core_rows, dist_rows, beat_rows = [], [], [], [], []
    n_k3 = n_bad = n_space = 0
    max_core = 0.0
    for i, w in enumerate(warps):
        r = np.asarray(gt_pk[i], dtype=np.float64)
        K = int(len(r))
        if w.identity:
            if w.status == "K<3":
                n_k3 += 1
            elif w.status == "integer spacing violated":
                n_space += 1
            else:
                n_bad += 1
            sched.append({"row": i, "K": K, "identity": 1, "status": w.status})
            continue
        qr, qi = np.asarray(w.q_real, float), np.asarray(w.q, np.int64)
        d_int = np.diff(qi); d_real = np.diff(qr)
        co = w.core_offsets(); max_core = max(max_core, float(co.max()))
        sched.append({"row": i, "K": K, "identity": 0, "status": w.status,
                      "q_int_first": int(qi[0]), "q_int_last": int(qi[-1]),
                      "all_integer_shift": int(np.all(np.mod(w.integer_shift(), 1.0) == 0.0)),
                      "core_offset_max": float(co.max())})
        spacing.append({"row": i, "K": K, "min_original_RR": float(np.min(np.diff(r))) if K > 1 else np.nan,
                        "min_real_canonical_spacing": float(d_real.min()), "min_int_canonical_spacing": int(d_int.min()),
                        "meets_min_21": int(d_int.min() >= B.MIN_INT_SPACING), "q_min": int(qi.min()), "q_max": int(qi.max())})
        core_rows.append({"row": i, "n_core_coords": int(co.size), "max_frac_offset": float(co.max()),
                          "mean_frac_offset": float(co.mean())})
        dist_rows.append({"row": i, "K": K, "max_abs_q_shift": float(np.max(np.abs(qi - qr))),
                          "median_abs_q_shift": float(np.median(np.abs(qi - qr))),
                          "canonical_RR_spread": int(d_int.max() - d_int.min()), "canonical_RR_std": float(d_int.std()),
                          "ideal_spacing": float(d_real[0]),
                          "relative_deviation": float(np.max(np.abs(d_int - d_real[0])) / max(d_real[0], 1e-9))})
        for k in range(K):
            beat_rows.append({"row": i, "beat": k, "r": float(r[k]), "q_real": float(qr[k]), "q_int": int(qi[k]),
                              "old_frac_offset": float(abs(qr[k] - r[k]) - round(abs(qr[k] - r[k]))) if False else
                              float(abs(abs(qr[k] - r[k]) - round(abs(qr[k] - r[k])))),
                              "new_frac_offset": 0.0, "old_shift": float(qr[k] - r[k]), "new_shift": float(qi[k] - r[k]),
                              "rounding_perturbation": float(abs(qi[k] - qr[k]))})
    wcsv(ART / "integer_schedule_manifest.csv", sched)
    wcsv(ART / "integer_spacing_audit.csv", spacing)
    wcsv(ART / "integer_grid_core_audit.csv", core_rows)
    wcsv(ART / "schedule_distortion.csv", dist_rows)
    wcsv(ART / "per_beat_grid_analysis.csv", beat_rows)
    precheck = {"n_windows": len(warps), "identity_K_lt_3": n_k3, "spacing_violations": n_space,
                "other_invalid": n_bad, "max_fractional_core_offset": max_core,
                "core_offset_tolerance": B.CORE_OFFSET_TOL, "min_int_spacing_required": B.MIN_INT_SPACING,
                "min_int_spacing_observed": int(min(s["min_int_canonical_spacing"] for s in spacing)) if spacing else None,
                "all_warps_valid": all(w.valid() for w in warps),
                "spacing_ok": n_space == 0 and n_bad == 0,
                "core_offset_ok": max_core <= B.CORE_OFFSET_TOL}
    precheck["passed"] = bool(precheck["spacing_ok"] and precheck["core_offset_ok"] and precheck["all_warps_valid"])
    print(f"[precheck] K<3 {n_k3} | spacing violations {n_space} | other invalid {n_bad} | "
          f"min int spacing {precheck['min_int_spacing_observed']} (>= {B.MIN_INT_SPACING}) | "
          f"max core frac offset {max_core:.3e} (<= {B.CORE_OFFSET_TOL}) -> {'PASS' if precheck['passed'] else 'FAIL'}", flush=True)

    prov = {"git": git_sha(ROOT), "prereg": "f2f3617", "utc": datetime.now(timezone.utc).isoformat(),
            "test_subjects_loaded": [], "n_windows": int(len(X)), "n_gt_beats": n_beats, "o1_train_iqr": iqr,
            "operator": "integer-grid oracle canonicalization operator", "generator_trained": False,
            "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()}}
    if not precheck["passed"]:
        dec = B.decide_o2b({}, precheck_ok=False)
        (ART / "decision.json").write_text(json.dumps({**dec, "precheck": precheck}, indent=2, default=float))
        (ART / "provenance.json").write_text(json.dumps({**prov, "precheck": precheck, "stopped": True}, indent=2, default=str))
        print(f"\n[O2b VERDICT] {dec['verdict']}", flush=True)
        return 0

    # ---------------- Stage-0 round trip (EXACT O2 metric code) ----------------
    rows, med = S0.roundtrip_metrics(Y, warps, gt_pk, gt_tg, iqr, "integer_grid")
    wcsv(ART / "warp_roundtrip_metrics.csv", rows)
    gate = O2.roundtrip_gate(med)
    checks = {"R0-1": med["raw_rmse"] <= 0.020, "R0-2": med["T6"] <= 0.020, "R0-3": med["T7"] <= 0.020,
              "R0-4a": med["T4"] <= 0.020, "R0-4b": med["T8"] <= 0.020,
              "R0-5": med["f1_at_50"] >= 0.98, "R0-6": med["beat_count_diff"] == 0.0}
    dec = B.decide_o2b(checks, precheck_ok=True)

    o2 = json.loads((O2ART / "stage0_result.json").read_text())["qrs_preserving_medians"]
    cmp_rows = []
    for k, thr in (("raw_rmse", 0.020), ("raw_corr", None), ("qrs_core_rmse", None), ("T4", 0.020), ("T6", 0.020),
                   ("T7", 0.020), ("T8", 0.020), ("f1_at_50", 0.98), ("beat_count_diff", 0.0)):
        cmp_rows.append({"metric": k, "o2_fractional": o2[k], "o2b_integer": med[k], "threshold": thr,
                         "repair_ratio": (med[k] / o2[k]) if (o2[k] not in (0, 0.0) and k in ("T4", "T6", "T7", "T8", "raw_rmse", "qrs_core_rmse")) else None})
    wcsv(ART / "o2_vs_o2b_comparison.csv", cmp_rows)

    # rounding perturbation vs O2b error (association only)
    pert = np.array([np.median([b["rounding_perturbation"] for b in beat_rows if b["row"] == i]) if any(b["row"] == i for b in beat_rows) else np.nan
                     for i in range(len(Y))])
    t6 = np.array([r["nAE_T6"] for r in rows], float); t7 = np.array([r["nAE_T7"] for r in rows], float)
    ok = np.isfinite(pert) & np.isfinite(t6) & np.isfinite(t7)
    rho = {"spearman_pert_T6": float(spearmanr(pert[ok], t6[ok]).statistic) if ok.sum() > 8 else None,
           "spearman_pert_T7": float(spearmanr(pert[ok], t7[ok]).statistic) if ok.sum() > 8 else None,
           "median_rounding_perturbation": float(np.nanmedian(pert)), "note": "association only; no causal language"}

    src = inspect.getsource(R.qrs_width_ms)
    t8 = {"function": "ppg2ecg.evaluation.rpeaks.qrs_width_ms", "q_win_s": 0.08, "s_win_s": 0.12,
          "q_search_samples": int(round(0.08 * OT.FS)), "s_search_samples": int(round(0.12 * OT.FS)),
          "inspected_span_relative_to_R": [-int(round(0.08 * OT.FS)), int(round(0.12 * OT.FS))],
          "protected_core": [-B.ANCHOR_W, B.ANCHOR_W],
          "can_inspect_outside_protected_core": bool(int(round(0.12 * OT.FS)) > B.ANCHOR_W),
          "samples_outside_core": int(round(0.12 * OT.FS)) - B.ANCHOR_W,
          "W_unchanged": True, "source": src}
    (ART / "t8_support_audit.json").write_text(json.dumps(t8, indent=2))

    stage0 = {**prov, "precheck": precheck, "medians": med, "gate": gate, "checks": checks,
              "o2_medians": o2, "rounding_perturbation_association": rho, "wall_s": time.perf_counter() - t0}
    (ART / "stage0_result.json").write_text(json.dumps(stage0, indent=2, default=str))
    (ART / "provenance.json").write_text(json.dumps({**prov, "precheck": precheck, "stopped": False,
                                                     "wall_s": time.perf_counter() - t0}, indent=2, default=str))
    (ART / "decision.json").write_text(json.dumps({**dec, "medians": med, "o2_medians": o2,
                                                   "precheck": precheck,
                                                   "terminology": "integer-grid oracle canonicalization operator",
                                                   "factorization_hypothesis": "NOT TESTED",
                                                   "generator_trained": False}, indent=2, default=float))
    print("\n[Stage 0] integer-grid medians:", json.dumps({k: round(v, 6) for k, v in med.items()}, indent=1), flush=True)
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}", flush=True)
    print(f"\n[O2b VERDICT] {dec['verdict']}", flush=True)
    print(f"[repair ratios] " + " ".join(f"{k} {med[k]/o2[k]:.4f}" for k in ("T4", "T6", "T7", "T8")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
