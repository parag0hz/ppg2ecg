"""O3 stage B — the frozen PPG-only R1 schedule bridge (preregistration sections 21-23).

Commit-order steps 20-28. NO TRAINING: the R1 Global-TCN, B and O2c are all frozen and no optimizer exists.
This script refuses to run until stage A has written synthetic_curve_frozen.json, so R1 performance cannot
influence the synthetic tolerance definition.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.probes import r1_cohort as RC
from ppg2ecg.training.train_a0 import git_sha

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import o3_common as C  # noqa: E402

ROOT, ART = C.ROOT, C.ART
FROZEN = ART / "synthetic_curve_frozen.json"
ARM = "O2C-R1-SCHEDULE"
META: dict = {}


def main() -> int:
    if not FROZEN.exists():
        raise RuntimeError("synthetic_curve_frozen.json does not exist — stage A must complete first (STOP)")
    frozen = json.loads(FROZEN.read_text())
    bad = {f: [h, C.fsha(ART / f)] for f, h in frozen["artifact_sha256"].items() if C.fsha(ART / f) != h}
    if bad:
        raise RuntimeError(f"synthetic artifacts changed after the freeze (STOP): {list(bad)}")
    t_all = time.perf_counter()
    dev = torch.device("cuda")
    coh = C.load_cohort()
    X, Yd, SUB, SITE, WI = coh["X"], coh["Yd"], coh["SUB"], coh["SITE"], coh["WI"]
    gt_pk, gt_tg, iqr, CLUSTER = coh["gt_pk"], coh["gt_tg"], coh["iqr"], coh["cluster"]
    base, o2c, tcn, meta = C.load_models(dev, with_r1=True)
    META["frozen"] = meta
    e0 = C.source_bank(len(X))
    print(f"[frozen] synthetic curve verified, J_MAX={frozen['j_max_samples']} "
          f"({frozen['j_max_ms'] if frozen['j_max_ms'] is None else round(frozen['j_max_ms'], 3)} ms)", flush=True)

    # ---------------- 20-21. frozen R1 schedule extraction and precheck ----------------
    S = C.r1_schedules(tcn, X, dev)
    checks = [O3.precheck_schedule(s, require_non_identity=False) for s in S]
    short = [i for i, c in enumerate(checks) if c["M"] < 3]
    hard = []
    for i, c in enumerate(checks):
        if c["M"] < 3:
            continue
        ks = dict(c["checks"]); ks["non_identity"] = not c["identity"]
        if not all(ks.values()):
            hard.append({"row": i, "subject": SUB[i], "site": SITE[i], "M": c["M"],
                         **{k: bool(v) for k, v in ks.items()}})
    man = {"n_windows": len(S), "threshold": O3.R1_THRESHOLD, "refractory_samples": 32,
           "identity_M_lt_3": len(short), "identity_M_lt_3_fraction": len(short) / len(S),
           "hard_invalid": len(hard), "min_spacing": int(min(c["min_spacing"] for c in checks if c["M"] >= 2)),
           "min_M": int(min(c["M"] for c in checks)), "max_M": int(max(c["M"] for c in checks)),
           "mean_M": float(np.mean([len(s) for s in S])),
           "max_core_offset": float(max(c["max_core_offset"] for c in checks)),
           "gt_fallback_used": False, "correction_applied": "none"}
    C.wcsv(ART / "r1_schedule_manifest.csv", [man] + hard[:200])
    precheck_ok = not hard
    print(f"[R1] M {man['min_M']}..{man['max_M']} mean {man['mean_M']:.2f} | M<3 {len(short)} "
          f"({man['identity_M_lt_3_fraction']:.3%}) | hard invalid {len(hard)} | "
          f"{'PRECHECK OK' if precheck_ok else 'R1 BRIDGE FAILS PRECHECK'}", flush=True)

    # ---------------- 21. R1 schedule quality (always reported) ----------------
    q = [O3.schedule_quality(gt_pk[i], S[i]) for i in range(len(S))]
    QK = ("f1_at_50", "f1_at_100", "f1_at_150", "f1_at_200", "precision", "recall", "missing", "spurious",
          "beats_ratio_dev", "timing_median_ae_ms", "timing_mae_ms")
    qrow = {"arm": "R1-SCHEDULE", "n_windows": len(S), **{k: C.macro([r[k] for r in q], SUB) for k in QK}}
    site_q = [{"arm": "R1-SCHEDULE", "site": s, "n": int((SITE == s).sum()),
               **{k: C.macro([q[i][k] for i in np.flatnonzero(SITE == s)], SUB[SITE == s]) for k in QK}}
              for s in RC.SITES]
    C.wcsv(ART / "r1_schedule_quality.csv", [qrow] + site_q)
    print(f"[R1] schedule F1@50 {qrow['f1_at_50']:.4f} F1@150 {qrow['f1_at_150']:.4f} "
          f"MAE {qrow['timing_mae_ms']:.2f} ms beats dev {qrow['beats_ratio_dev']:.4f}", flush=True)

    if not precheck_ok:
        _finish(frozen, None, None, qrow, None, None, precheck_ok=False, t_all=t_all, dev=dev, coh=coh)
        return 0

    # ---------------- 22. R1 operator-floor diagnostic ----------------
    warps = C.build_warps(S)
    frows, fmed = C.S0.roundtrip_metrics(coh["Y"], warps, gt_pk, gt_tg, iqr, "R1")
    coupled = bool(fmed["T6"] > 0.020 or fmed["T7"] > 0.020)
    C.wcsv(ART / "r1_operator_floor.csv", [{"arm": "R1-SCHEDULE", "raw_rmse": fmed["raw_rmse"],
        "qrs_core_rmse": fmed["qrs_core_rmse"], "f1_at_50": fmed["f1_at_50"],
        "beat_count_diff": fmed["beat_count_diff"], "nAE_T4": fmed["T4"], "nAE_T6": fmed["T6"],
        "nAE_T7": fmed["T7"], "nAE_T8": fmed["T8"],
        "interpretation": "schedule/operator coupled" if coupled else "operator floor small",
        "note": "never subtracted, never used to correct a metric or a CI"}])
    print(f"[R1] operator floor raw RMSE {fmed['raw_rmse']:.5f} T6 {fmed['T6']:.5f} T7 {fmed['T7']:.5f} "
          f"-> {'schedule/operator coupled' if coupled else 'floor small'}", flush=True)

    # ---------------- 23. reference arms and the end-to-end R1 arm ----------------
    p_b = C.R2E.gen_plain(base, X, e0, C.NFE, dev)
    rows_b, _a, _b = C.R2E.score(p_b, Yd, gt_pk)
    mac_b, al_b = C.R2E.macro_rows(rows_b, SUB), C.aligned_rows(p_b, gt_tg, iqr)
    warps_o = C.build_warps([np.asarray(p, np.int64) for p in gt_pk])
    p_o, _c1 = C.o2c_predict(o2c, X, warps_o, e0, dev)
    rows_o, _c, _d = C.R2E.score(p_o, Yd, gt_pk)
    mac_o, al_o = C.R2E.macro_rows(rows_o, SUB), C.aligned_rows(p_o, gt_tg, iqr)
    p_r, _c2 = C.o2c_predict(o2c, X, warps, e0, dev)
    rows_r, _e, _f = C.R2E.score(p_r, Yd, gt_pk)
    mac_r, al_r = C.R2E.macro_rows(rows_r, SUB), C.aligned_rows(p_r, gt_tg, iqr)
    gen_pk = C.R2E.pmap(C.S0._peaks, list(p_r.astype(np.float64)))
    adh = [O3.adherence(S[i], gen_pk[i]) for i in range(len(S))]
    adh_row = {k: C.macro([r[k] for r in adh], SUB) for k in adh[0]}
    ALM = {a: {t: C.macro([r[C.NAE[t]] for r in al], SUB) for t in C.ALIGNED}
           for a, al in (("B", al_b), ("O2C-ORACLE", al_o), (ARM, al_r))}
    MAC = {"B": mac_b, "O2C-ORACLE": mac_o, ARM: mac_r}
    C.wcsv(ART / "r1_generator_metrics.csv",
           [{"arm": a, **{k: MAC[a][k] for k in C.EVENT_M}, **{C.NAE[t]: ALM[a][t] for t in C.ALIGNED},
             **{k: MAC[a][k] for k in C.STRUCT_M},
             **(adh_row if a == ARM else {k: "" for k in adh_row})} for a in ("B", "O2C-ORACLE", ARM)])
    print(f"[R1] {ARM} f1x {mac_r['f1_excess']:+.4f} | " +
          " ".join(f"{C.NAE[t]} {ALM[ARM][t]:.4f}" for t in C.ALIGNED) +
          f" | adherence F1@50 {adh_row['adherence_f1_at_50']:.4f}", flush=True)

    res = C.paired_boot(rows_b, al_b, rows_r, al_r, SUB, CLUSTER)
    C.wcsv(ART / "r1_paired_bootstrap.csv",
           [{"contrast": f"{ARM}_vs_B", "metric": m, "positive_means": "R1-schedule O2c better than B",
             **r} for m, r in res.items()])
    gates = O3.joint_gates(res)

    # ---------------- 24. R1 multi-source and G7 ----------------
    from ppg2ecg.evaluation import q1_corruption as Q
    unc = Q.uncertainty_positions(SUB, SITE, WI)
    assert len(unc) == 512
    SUBu, CLu = SUB[unc], CLUSTER[unc]
    wu = [warps[i] for i in unc]
    U = {}
    ms_extra = []
    for arm in ("B", ARM):
        Sm, adh_s = [], []
        for sd in Q.UNC_SEEDS:
            bank = C.source_bank(len(X), sd)[unc]
            if arm == "B":
                pr = C.R2E.gen_plain(base, X[unc], bank, C.NFE, dev)
            else:
                pr, _cn = C.o2c_predict(o2c, X[unc], wu, bank, dev)
            Sm.append(pr)
            if arm != "B":
                pk = C.R2E.pmap(C.S0._peaks, list(pr.astype(np.float64)))
                adh_s.append(C.macro([O3.adherence(S[unc[i]], pk[i])["adherence_f1_at_50"] for i in range(len(unc))], SUBu))
        Sm = np.stack(Sm)
        flat = C.R2E.pmap(C.S0._peaks, [Sm[s, i].astype(np.float64) for i in range(Sm.shape[1]) for s in range(Sm.shape[0])])
        rows = [Q.uncertainty_from_samples(Sm[:, i], flat[i * Sm.shape[0]:(i + 1) * Sm.shape[0]], gt_pk[unc[i]])
                for i in range(Sm.shape[1])]
        U[arm] = rows
        if arm == ARM:
            ms_extra.append({"arm": "R1-SCHEDULE", "condition": ARM, "rep": "-", "n_windows": len(rows),
                             "n_sources": len(Q.UNC_SEEDS),
                             "beat_count_SD": C.macro([r["u3_beatcount_sd"] for r in rows], SUBu),
                             "pairwise_event_F1_50": C.macro([r["u4_pairwise_event_f1_50"] for r in rows], SUBu),
                             "pairwise_event_F1_150": C.macro([r["u5_pairwise_event_f1_150"] for r in rows], SUBu),
                             "pointwise_waveform_SD": C.macro([r["u1_pointwise_sd"] for r in rows], SUBu),
                             "pairwise_waveform_RMSE": C.macro([r["u2_pairwise_rmse"] for r in rows], SUBu),
                             "adherence_f1_at_50_across_sources": float(np.mean(adh_s))})
    s1 = C.O1E.cluster_bootstrap(np.array([r["u3_beatcount_sd"] for r in U["B"]], float) -
                                 np.array([r["u3_beatcount_sd"] for r in U[ARM]], float), SUBu, CLu,
                                 n_boot=O3.BOOT_N, seed=O3.BOOT_SEED)
    s2 = C.O1E.cluster_bootstrap(np.array([r["u4_pairwise_event_f1_50"] for r in U[ARM]], float) -
                                 np.array([r["u4_pairwise_event_f1_50"] for r in U["B"]], float), SUBu, CLu,
                                 n_boot=O3.BOOT_N, seed=O3.BOOT_SEED)
    _append(ART / "multisource_metrics.csv", ms_extra)
    _append(ART / "multisource_bootstrap.csv",
            [{"arm": "R1-SCHEDULE", "gate": "G7a", "quantity": "beat-count SD (B - R1 arm)", **s1},
             {"arm": "R1-SCHEDULE", "gate": "G7b", "quantity": "pairwise event F1@50 (R1 arm - B)", **s2}])
    full = O3.bridge_gate(gates, s1, s2)
    _append(ART / "joint_benefit_gates.csv", [{"stage": "R1", "condition": ARM, "family": "R1", "level": "-",
                                               "rep": "-", **full}])
    print("[R1] gates " + " ".join(f"{k}:{'PASS' if full[k] else 'FAIL'}"
                                   for k in ("G1", "G2", "G3", "G4", "G5", "G6", "G7a", "G7b")) +
          f" -> {'SUPPORTED' if full['supported'] else 'NOT SUPPORTED'}", flush=True)

    # ---------------- 27. site-wise secondary ----------------
    srows = []
    for s in RC.SITES:
        k = np.flatnonzero(SITE == s)
        row = {"site": s, "n": int(k.size),
               "R1_schedule_f1_at_50": C.macro([q[i]["f1_at_50"] for i in k], SUB[k]),
               "R1_schedule_f1_at_150": C.macro([q[i]["f1_at_150"] for i in k], SUB[k]),
               "R1_schedule_beats_ratio_dev": C.macro([q[i]["beats_ratio_dev"] for i in k], SUB[k]),
               "R1_schedule_timing_mae_ms": C.macro([q[i]["timing_mae_ms"] for i in k], SUB[k])}
        for arm, rows_, al_ in (("B", rows_b, al_b), ("O2C-ORACLE", rows_o, al_o), (ARM, rows_r, al_r)):
            row[f"{arm}_f1_excess"] = C.macro([rows_[i]["f1_excess"] for i in k], SUB[k])
            row[f"{arm}_nAE_T6"] = C.macro([al_[i]["nAE_T6"] for i in k], SUB[k])
            row[f"{arm}_nAE_T7"] = C.macro([al_[i]["nAE_T7"] for i in k], SUB[k])
        srows.append(row)
    C.wcsv(ART / "r1_site_metrics.csv", srows)

    _finish(frozen, res, full, qrow, fmed, adh_row, precheck_ok=True, t_all=t_all, dev=dev, coh=coh,
            mac={"B": mac_b, "O2C-ORACLE": mac_o, ARM: mac_r}, alm=ALM, coupled=coupled, s1=s1, s2=s2)
    return 0


def _append(path, rows, key=("stage", "condition", "arm", "gate")):
    """Idempotent append: rows that collide with the new ones on `key` are replaced, so a re-run cannot duplicate."""
    import csv
    new = [{k: ("" if v is None else v) for k, v in r.items()} for r in rows]
    ids = {tuple(str(r.get(k, "")) for k in key) for r in new}
    old = [r for r in (list(csv.DictReader(open(path))) if path.exists() else [])
           if tuple(str(r.get(k, "")) for k in key) not in ids]
    C.wcsv(path, old + new)


def _finish(frozen, res, full, qrow, fmed, adh_row, precheck_ok, t_all, dev, coh, mac=None, alm=None,
            coupled=None, s1=None, s2=None):
    level_pass = {int(k): bool(v) for k, v in frozen["jitter_level_pass"].items()}
    r1_supported = bool(full["supported"]) if full else False
    dec = O3.decide_o3(level_pass, r1_supported, precheck_ok)
    tol = {"j_max_samples": frozen["j_max_samples"], "j_max_ms": frozen["j_max_ms"],
           "jitter_level_pass": frozen["jitter_level_pass"], "miss_pass": frozen["miss_pass"],
           "extra_pass": frozen["extra_pass"], "schedule_quality_at_j_max": frozen["schedule_quality_at_j_max"],
           "r1_schedule_quality": qrow, "r1_precheck_ok": precheck_ok,
           "r1_operator_floor": (None if fmed is None else {"raw_rmse": fmed["raw_rmse"], "T4": fmed["T4"],
                                                            "T6": fmed["T6"], "T7": fmed["T7"], "T8": fmed["T8"]}),
           "r1_operator_floor_coupled": coupled, "r1_adherence": adh_row,
           "r1_gates": full, "r1_effects": res, "synthetic_curve_frozen_sha256": C.fsha(FROZEN),
           "composite_sqi_used": False,
           "note": "the synthetic corruption model is not claimed to match the R1 error distribution"}
    (ART / "tolerance_summary.json").write_text(json.dumps(tol, indent=2, default=float))
    (ART / "decision.json").write_text(json.dumps(
        {**dec, "S1_form": s1, "S2_form": s2, "r1_arm": ARM,
         "oracle": "all synthetic arms perturb the GT R schedule and remain ORACLE DIAGNOSTICS; the R1 arm is "
                   "PPG-only at inference but the R1 probe was supervised with ECG R labels",
         "status": "exploratory mechanism bridge; development cohort only; two validation subjects; no test; "
                   "no training anywhere in O3",
         "training_performed": False,
         "miss_extra_characterise_failure_mode_only": True}, indent=2, default=float))
    (ART / "provenance_stage_b.json").write_text(json.dumps(
        {"stage": "B (frozen R1 bridge)", "git": git_sha(ROOT), "prereg": "60d1810",
         "utc": datetime.now(timezone.utc).isoformat(), "test_subjects_loaded": [], "training_performed": False,
         "r1_threshold": O3.R1_THRESHOLD, "r1_refractory_samples": 32, "correction_applied": "none",
         "synthetic_curve_frozen_sha256": C.fsha(FROZEN), "frozen_components": META.get("frozen"),
         "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
         "gpu": torch.cuda.get_device_name(0), "wall_s": time.perf_counter() - t_all,
         "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20)}, indent=2, default=str))
    print(f"\n[VERDICT] {dec['verdict']}", flush=True)
    print(f"[done] stage B in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
