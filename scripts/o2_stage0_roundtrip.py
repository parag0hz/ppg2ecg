"""O2 Stage 0 — warp-only round-trip falsification (preregistration section 6, gate R0-1..R0-6).

No generator is built or trained here. If the gate fails, O2 stops with
CANONICALIZATION OPERATOR REJECTED.
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
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import m1_structural as M1
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import o2_warp as W
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2_oracle_canonicalization"
O1ART = ROOT / "artifacts/o1_component_extractability"
VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024
T4, T6, T7, T8 = "median_QRS_p2p", "median_QRS_max_abs_derivative", "median_QRS_curvature_energy", "median_QRS_width_ms"
ALIGNED = (T4, T6, T7, T8)


def _peaks(sig):
    return R.detect_rpeaks(np.asarray(sig, dtype=np.float64), OT.FS)


def _targets(sig):
    t = OT.window_targets(np.asarray(sig, dtype=np.float64))
    return {k: t[k] for k in OT.TARGETS} | {"n_valid_qrs_beats": t["n_valid_qrs_beats"]}


def pmap(fn, items, workers=12, chunk=16):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=chunk))


def load_cohort():
    X, Y, SUB, SITE, POS, WI = [], [], [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs, Ys, Ss, Ws = d["x"], d["y"], np.asarray(d["site"]).astype(str), d["window_index"]
        idx = ER.select_subset(SALT, s, len(Xs), TAKE)
        X.append(Xs[idx].astype(np.float32)); Y.append(Ys[idx].astype(np.float32))
        SUB.append(np.full(len(idx), s)); SITE.append(Ss[idx]); POS.append(idx); WI.append(Ws[idx].astype(np.int64))
    X, Y, SUB, SITE, POS, WI = (np.concatenate(v) for v in (X, Y, SUB, SITE, POS, WI))
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    for s in VAL:
        assert POS[SUB == s].tolist() == list(frozen[s]), f"frozen subset mismatch {s}"
    return X, Y, SUB, SITE, POS, WI


def roundtrip_metrics(Y, warps, gt_pk, gt_tg, iqr, tag):
    y = torch.from_numpy(Y).unsqueeze(1)
    rt = W.round_trip(y, warps).squeeze(1).numpy()
    rt_pk = pmap(_peaks, list(rt.astype(np.float64)))
    rt_tg = pmap(_targets, list(rt.astype(np.float64)))
    rows = []
    for i in range(len(Y)):
        a, b = Y[i].astype(np.float64), rt[i].astype(np.float64)
        m, fp, fn = R.match_rpeaks(gt_pk[i], rt_pk[i], OT.FS, tol_ms=50.0)
        _p, _r, f1 = R.prf(len(m), fp, fn)
        qm = M1.qrs_core_morphology(b, a, gt_pk[i])
        row = {"variant": tag, "row": i, "identity": int(warps[i].identity), "status": warps[i].status,
               "raw_rmse": float(np.sqrt(np.mean((b - a) ** 2))),
               "raw_corr": float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else np.nan,
               "qrs_core_rmse": float(qm["qrs_rmse_core"]), "f1_at_50": float(f1),
               "beat_count_diff": float(rt_tg[i]["beat_count"] - gt_tg[i]["beat_count"])}
        for t in ALIGNED:
            a_, b_ = gt_tg[i][t], rt_tg[i][t]
            row[f"nAE_{OT.TARGET_IDS[t]}"] = float(abs(b_ - a_) / iqr[t]) if np.isfinite(a_) and np.isfinite(b_) else np.nan
        rows.append(row)
    med = {"raw_rmse": float(np.nanmedian([r["raw_rmse"] for r in rows])),
           "raw_corr": float(np.nanmedian([r["raw_corr"] for r in rows])),
           "qrs_core_rmse": float(np.nanmedian([r["qrs_core_rmse"] for r in rows])),
           "f1_at_50": float(np.nanmedian([r["f1_at_50"] for r in rows])),
           "f1_at_50_mean": float(np.nanmean([r["f1_at_50"] for r in rows])),
           "beat_count_diff": float(np.nanmedian([r["beat_count_diff"] for r in rows])),
           "T4": float(np.nanmedian([r["nAE_T4"] for r in rows])), "T6": float(np.nanmedian([r["nAE_T6"] for r in rows])),
           "T7": float(np.nanmedian([r["nAE_T7"] for r in rows])), "T8": float(np.nanmedian([r["nAE_T8"] for r in rows]))}
    return rows, med


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    t0 = time.perf_counter()
    X, Y, SUB, SITE, POS, WI = load_cohort()
    Yd = Y.astype(np.float64)
    gt_pk = pmap(_peaks, list(Yd))
    gt_tg = pmap(_targets, list(Yd))
    n_beats = int(sum(len(p) for p in gt_pk))
    if len(X) != 2048 or n_beats != 19834:
        raise RuntimeError(f"frozen population facts differ: {len(X)} windows, {n_beats} beats (STOP)")
    iqr = {t: json.loads((O1ART / "target_scaling.json").read_text())["targets"][t]["scale_train_IQR"] for t in ALIGNED}
    print(f"[P] {len(X)} windows, {n_beats} GT beats | O1 IQR {({OT.TARGET_IDS[t]: round(v,4) for t, v in iqr.items()})}", flush=True)

    warps = [W.EventWarp(gt_pk[i]) for i in range(len(Y))]
    cwarps = [W.CenterOnlyWarp(gt_pk[i]) for i in range(len(Y))]
    man, slopes_all = [], []
    n_k3 = n_bad = 0
    for i, w in enumerate(warps):
        sl = w.slopes(); cs = w.core_slopes()
        if w.identity:
            if w.status == "K<3":
                n_k3 += 1
            else:
                n_bad += 1
        man.append({"row": i, "subject": SUB[i], "site": SITE[i], "array_pos": int(POS[i]), "window_index": int(WI[i]),
                    "K": int(len(gt_pk[i])), "identity": int(w.identity), "status": w.status, "valid": int(w.valid()),
                    "n_anchors": 0 if w.identity else int(len(w.src)),
                    "slope_min": float(sl.min()), "slope_max": float(sl.max()),
                    "core_slope_min": float(cs.min()), "core_slope_max": float(cs.max()),
                    "max_abs_shift": 0.0 if w.identity else float(np.max(np.abs(w.forward(np.arange(1024)) - np.arange(1024))))})
        slopes_all.append(sl)
    frac_bad = n_bad / len(warps)
    print(f"[warp] identity K<3 {n_k3} ({n_k3/len(warps):.4%}) | other fallback {n_bad} ({frac_bad:.4%}) | "
          f"all valid {all(w.valid() for w in warps)}", flush=True)
    allsl = np.concatenate(slopes_all)
    core = np.concatenate([w.core_slopes() for w in warps if not w.identity]) if any(not w.identity for w in warps) else np.ones(1)
    slope_rows = [{"quantity": "all_segments", **{f"p{q}": float(np.percentile(allsl, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}},
                  {"quantity": "qrs_core_segments", **{f"p{q}": float(np.percentile(core, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}}]
    with open(ART / "warp_manifest.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(man[0])); w_.writeheader(); w_.writerows(man)
    with open(ART / "warp_slope_distribution.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(slope_rows[0])); w_.writeheader(); w_.writerows(slope_rows)
    if not all(w.valid() for w in warps):
        raise RuntimeError("a warp is not monotone (STOP)")
    if frac_bad > W.FALLBACK_BUDGET:
        raise RuntimeError(f"non-(K<3) fallback fraction {frac_bad:.4%} exceeds {W.FALLBACK_BUDGET:.2%} (STOP)")

    rows, med = roundtrip_metrics(Y, warps, gt_pk, gt_tg, iqr, "qrs_preserving")
    crows, cmed = roundtrip_metrics(Y, cwarps, gt_pk, gt_tg, iqr, "center_only")
    with open(ART / "warp_roundtrip_metrics.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    with open(ART / "center_only_roundtrip_metrics.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(crows[0])); w_.writeheader(); w_.writerows(crows)
    gate = W.roundtrip_gate(med)
    out = {"git": git_sha(ROOT), "prereg": "1911fa6", "utc": datetime.now(timezone.utc).isoformat(),
           "test_subjects_loaded": [], "n_windows": int(len(X)), "n_gt_beats": n_beats,
           "identity_K_lt_3": n_k3, "identity_K_lt_3_fraction": n_k3 / len(warps),
           "other_fallback": n_bad, "other_fallback_fraction": frac_bad, "fallback_budget": W.FALLBACK_BUDGET,
           "all_warps_valid": True, "anchor_w": W.ANCHOR_W, "o1_train_iqr": iqr,
           "qrs_preserving_medians": med, "center_only_medians": cmed, "gate": gate,
           "slope_all_min": float(allsl.min()), "slope_all_max": float(allsl.max()),
           "core_slope_min": float(core.min()), "core_slope_max": float(core.max()),
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
           "wall_s": time.perf_counter() - t0}
    (ART / "stage0_result.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n[Stage 0] QRS-preserving medians:", json.dumps({k: round(v, 5) for k, v in med.items()}, indent=1), flush=True)
    print("[Stage 0] center-only medians   :", json.dumps({k: round(v, 5) for k, v in cmed.items()}, indent=1), flush=True)
    for k, v in gate["checks"].items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}", flush=True)
    print(f"\n[Stage 0] GATE {'PASSED' if gate['passed'] else 'FAILED -> CANONICALIZATION OPERATOR REJECTED'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
