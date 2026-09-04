"""E3 Stage 0 — candidate audit and the ORACLE COUNT ceiling (commit-order steps 9-12).

NO TRAINING. GT supplies the COUNT ONLY; every event location comes from the frozen R1 candidate scores.
This is an ORACLE COUNT DIAGNOSTIC and is not deployable.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import e3_beat_set as E3
from ppg2ecg.evaluation import event_geometry_contract as EG
from ppg2ecg.probes import rhythm_tcn as RTCN
from ppg2ecg.training.train_a0 import git_sha

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as H  # noqa: E402
import o3_common as C  # noqa: E402

ROOT, ART = H.ROOT, H.ART
SHORTAGE_STOP = 0.005


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "figures").mkdir(exist_ok=True)
    t0 = time.perf_counter()
    dev = torch.device("cuda")
    pt = subprocess.run([sys.executable, "-m", "pytest", "tests/test_e3_beat_set.py", "-o", "addopts=",
                         "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    if pt.returncode != 0:
        print(pt.stdout[-4000:]); raise RuntimeError("E3 tests fail; not evaluating")
    tests = next((ln for ln in reversed(pt.stdout.splitlines()) if "passed" in ln or "failed" in ln), "")
    print(f"[tests] {tests}", flush=True)

    (ART / "e2_contract_identity.json").write_text(json.dumps(H.assert_e2_contract(), indent=2))
    coh = C.load_cohort()
    X, Yd, SUB, SITE, gt_pk, CLUSTER = coh["X"], coh["Yd"], coh["SUB"], coh["SITE"], coh["gt_pk"], coh["cluster"]
    base, o2c, tcn, meta = C.load_models(dev, with_r1=True)
    (ART / "frozen_component_manifest.json").write_text(json.dumps(
        {**meta, "training_performed": False, "r1_threshold": E3.R1_THRESHOLD,
         "r1_refractory": E3.REFRACTORY}, indent=2, default=str))
    e0 = C.source_bank(len(X))
    print(f"[P] {len(X)} windows, {coh['n_beats']} GT beats, {len(set(CLUSTER.tolist()))} clusters", flush=True)

    # ---------------- 9. candidate extraction audit + capacity ----------------
    P = H.r1_scores(tcn, X, dev)
    cand_pos, cand_sc = H.candidates_for(P)
    frozen_r1 = C.r1_schedules(tcn, X, dev)
    exact = [np.array_equal(cand_pos[i][cand_sc[i] >= E3.R1_THRESHOLD], frozen_r1[i]) for i in range(len(X))]
    audit = {"score": "sigmoid(RhythmTCN(ppg)) -- the exact frozen R1 quantity",
             "threshold_free_rule": "rhythm_tcn.extract_events with ONLY the amplitude filter removed",
             "refractory": E3.REFRACTORY, "n_windows": len(X),
             "n_windows_reproducing_frozen_r1": int(sum(exact)),
             "bit_exact_reproduction": bool(all(exact)),
             "recalibration": "none", "smoothing": "none", "per_window_normalisation": "none"}
    (ART / "candidate_extraction_audit.json").write_text(json.dumps(audit, indent=2))
    if not all(exact):
        (ART / "decision.json").write_text(json.dumps(E3.decide_e3(precheck_ok=False), indent=2, default=float))
        raise RuntimeError(f"{E3.VERDICT_PRECHECK} (STOP): {len(X) - sum(exact)} windows differ")
    print(f"[cand] threshold-free candidates filtered at 0.35 reproduce frozen R1 on {sum(exact)}/{len(X)} "
          "windows (bit-exact)", flush=True)

    K = np.array([len(p) for p in gt_pk])
    M = np.array([len(p) for p in cand_pos])
    cov150 = []
    for i in range(len(X)):
        tol = 150.0 / 1000.0 * EG.FS
        g = np.asarray(gt_pk[i], np.int64)
        cov150.append(float(np.mean([np.any(np.abs(cand_pos[i] - int(x)) <= tol) for x in g])) if g.size else np.nan)
    short_frac = float(np.mean(M < K))
    C.wcsv(ART / "candidate_capacity.csv",
           [{"row": i, "subject": SUB[i], "site": SITE[i], "n_candidates": int(M[i]), "K_gt": int(K[i]),
             "candidates_minus_gt": int(M[i] - K[i]), "shortage": bool(M[i] < K[i]),
             "coverage_at_150ms": cov150[i]} for i in range(len(X))])
    cap = {"mean_candidates": float(M.mean()), "mean_gt": float(K.mean()),
           "median_candidates_minus_gt": float(np.median(M - K)), "min_candidates": int(M.min()),
           "shortage_fraction": short_frac, "shortage_stop_threshold": SHORTAGE_STOP,
           "candidate_coverage_at_150ms": float(np.nanmean(cov150)),
           "candidate_coverage_at_150ms_subject_macro": C.macro(cov150, SUB)}
    (ART / "candidate_capacity_summary.json").write_text(json.dumps(cap, indent=2))
    print(f"[cap ] candidates {M.mean():.2f} vs GT {K.mean():.2f} | shortage {short_frac:.4%} "
          f"| coverage@150ms {cap['candidate_coverage_at_150ms_subject_macro']:.4f}", flush=True)
    if short_frac > SHORTAGE_STOP:
        (ART / "decision.json").write_text(json.dumps(
            {"verdict": E3.VERDICT_CAPACITY, **cap}, indent=2, default=float))
        raise RuntimeError(f"{E3.VERDICT_CAPACITY} (STOP): shortage {short_frac:.4%} > {SHORTAGE_STOP:.2%}")

    # ---------------- 10. Stage-0 schedule evaluation ----------------
    S_R1 = frozen_r1
    S_OC, oc_short = H.arm_schedules("topk", cand_pos, cand_sc, counts=K)
    rows_r1 = H.schedule_block(gt_pk, S_R1)
    rows_oc = H.schedule_block(gt_pk, S_OC)
    mac = {"R1-0.35": H.macro_block(rows_r1, SUB), "ORACLE-COUNT-R1": H.macro_block(rows_oc, SUB)}
    C.wcsv(ART / "oracle_count_schedule_metrics.csv",
           [{"arm": a, "stage": "schedule", **mac[a]} for a in mac])
    eff = H.effects(rows_oc, rows_r1, H.SCHED_EFFECT_KEYS, SUB, CLUSTER, "ORACLE-COUNT_vs_R1")
    oc = E3.oracle_count_gates(eff)
    print(f"[S0  ] exact-set {mac['R1-0.35']['A5_exact_set_fraction']:.4f} -> "
          f"{mac['ORACLE-COUNT-R1']['A5_exact_set_fraction']:.4f} | T3 "
          f"{mac['R1-0.35']['T3_frac']:.4f} -> {mac['ORACLE-COUNT-R1']['T3_frac']:.4f} | "
          f"spurious {mac['R1-0.35']['A4_spurious_fraction']:.4f} -> "
          f"{mac['ORACLE-COUNT-R1']['A4_spurious_fraction']:.4f} | OC {oc}", flush=True)

    og, gen_mac, eff_g = None, {}, {}
    if oc["passed"]:
        # ---------------- 11. Stage-0 generator ceiling ----------------
        for arm, S in (("R1-0.35", S_R1), ("ORACLE-COUNT-R1", S_OC)):
            warps = C.build_warps(S)
            pred, _cn = C.o2c_predict(o2c, X, warps, e0, dev)
            Ppk = C.R2E.pmap(C.S0._peaks, list(pred.astype(np.float64)))
            r, _b = H.full_block(gt_pk, S, pred, Yd, Ppk)
            gen_mac[arm] = H.macro_block(r, SUB)
            gen_mac[arm]["_rows"] = r
        eff_g = H.effects(gen_mac["ORACLE-COUNT-R1"]["_rows"], gen_mac["R1-0.35"]["_rows"],
                          H.GEN_EFFECT_KEYS, SUB, CLUSTER, "ORACLE-COUNT_vs_R1_generator")
        og = E3.oracle_generator_gates(eff_g)
        C.wcsv(ART / "oracle_count_generator_metrics.csv",
               [{"arm": a, "stage": "generator", **{k: v for k, v in gen_mac[a].items() if k != "_rows"}}
                for a in gen_mac])
        print(f"[S0g ] PG F1@50 {gen_mac['R1-0.35']['PG_F1_50']:.4f} -> "
              f"{gen_mac['ORACLE-COUNT-R1']['PG_F1_50']:.4f} | T6 {gen_mac['R1-0.35']['C2_own_T6']:.4f} -> "
              f"{gen_mac['ORACLE-COUNT-R1']['C2_own_T6']:.4f} | OG {og}", flush=True)

    # site secondary (schedule side; no site causality claim)
    from ppg2ecg.probes import r1_cohort as RC
    srows = []
    for st in RC.SITES:
        idx = [i for i in range(len(X)) if SITE[i] == st]
        for arm, rws in (("R1-0.35", rows_r1), ("ORACLE-COUNT-R1", rows_oc)):
            srows.append({"arm": arm, "site": st, "n": len(idx),
                          **{k: C.macro([rws[i][k] for i in idx], SUB[idx]) for k in
                             ("A5_exact_set", "T3_frac", "T2_frac", "A3_missing_fraction",
                              "A4_spurious_fraction", "SG_F1_50")},
                          "B5_exact_set_mae_ms": float(np.nanmean([rws[i]["B5_exact_set_mae_ms"] for i in idx])),
                          "note": "secondary; no site causality claim"})
    C.wcsv(ART / "site_metrics.csv", srows)
    C.wcsv(ART / "oracle_count_bootstrap.csv", list(eff.values()) + list(eff_g.values()))
    C.wcsv(ART / "gates.csv", [{"stage": "stage0", "family": "OC", "gate": k, "result": bool(v)}
                               for k, v in oc.items()] +
           ([{"stage": "stage0", "family": "OG", "gate": k, "result": bool(v)} for k, v in og.items()]
            if og else []))
    (ART / "stage0_gates.json").write_text(json.dumps(
        {"OC": oc, "OG": og, "oracle_count_shortage_windows": int(sum(oc_short)),
         "effects": {k: {kk: v[kk] for kk in ("point", "lo", "hi", "NEW", "REF")} for k, v in eff.items()},
         "generator_effects": {k: {kk: v[kk] for kk in ("point", "lo", "hi", "NEW", "REF")}
                               for k, v in eff_g.items()}}, indent=2, default=float))
    (ART / "provenance_stage0.json").write_text(json.dumps(
        {"git": git_sha(ROOT), "prereg": "20f890b", "utc": datetime.now(timezone.utc).isoformat(),
         "test_subjects_loaded": [], "training_performed": False, "n_windows": int(len(X)),
         "nfe": C.NFE, "source_seed": C.SRC_SEED, "tests": tests,
         "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
         "gpu": torch.cuda.get_device_name(0), "wall_s": time.perf_counter() - t0}, indent=2, default=str))

    if not oc["passed"] or og is None or not og["passed"]:
        dec = E3.decide_e3(oc=oc, og=og)
        (ART / "decision.json").write_text(json.dumps(dec, indent=2, default=float))
        print(f"\n[VERDICT] {dec['verdict']} -- the ridge readout is NOT fitted", flush=True)
        return 0
    print("\n[stage0] ORACLE COUNT CEILING PASSES; proceed to the train-only threshold control and the "
          "count readout", flush=True)
    print(f"[done] stage 0 in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
