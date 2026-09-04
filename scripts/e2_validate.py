"""E2 — contract validation (commit-order steps 10-13).

MEASUREMENT-CONTRACT VALIDATION ONLY. NO TRAINING, NO WEIGHT UPDATE, NO NEW PREDICTOR, NO THRESHOLD TUNING:
B, O2c, the O2b operator and the R1 Global-TCN are loaded frozen and no optimizer is constructed. The E1
verdict is not revisited here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import event_geometry_contract as EG
from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.training.train_a0 import git_sha

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o3_common as C  # noqa: E402

ROOT = C.ROOT
ART = ROOT / "artifacts/e2_evaluation_contract"
E1ART = ROOT / "artifacts/e1_event_morphology_decomposition"
O3ART = ROOT / "artifacts/o3_schedule_tolerance"
REPRO_TOL = 1e-6
ARMS = [("ORACLE", "JITTER", 0), ("JITTER_8", "JITTER", 8), ("MISS1", "MISS", 1), ("EXTRA1", "EXTRA", 1)]
REP = 0
E1_SOURCES = ("beat_identity_manifest.csv", "chain_matching_metrics.csv", "coverage_metrics.csv",
              "topology_metrics.csv", "placement_metrics.csv", "own_center_beat_metrics.csv",
              "own_center_window_metrics.csv", "gt_anchored_local_metrics.csv", "alignment_sensitivity.csv",
              "synthetic_contrasts.csv", "topology_excess_damage.csv", "paired_bootstrap.csv",
              "r1_topology_strata.csv", "r1_timing_bins.csv", "r1_stratum_metrics.csv",
              "r1_site_topology.csv", "decision.json", "provenance.json")
# E1 column -> contract key (the _gt_T* entries exist only for the reproduction gate; they are NOT contract metrics)
E1MAP = {"own_nAE_T4": "C1_own_T4", "own_nAE_T6": "C2_own_T6", "own_nAE_T7": "C3_own_T7",
         "own_nAE_T8": "C4_own_T8", "own_local_raw_rmse": "C5_own_local_raw_rmse",
         "own_local_deriv_rmse": "C6_own_local_deriv_rmse", "own_local_curvature_err": "C7_own_local_curvature_err",
         "own_local_corr": "C8_own_local_corr", "gt_local_raw_rmse": "J1_gt_local_raw_rmse",
         "gt_local_deriv_rmse": "J2_gt_local_deriv_rmse", "gt_local_curvature_err": "J3_gt_local_curvature_err",
         "gt_local_corr": "J4_gt_local_corr", "gt_nAE_T4": "_gt_T4", "gt_nAE_T6": "_gt_T6",
         "gt_nAE_T7": "_gt_T7", "gt_nAE_T8": "_gt_T8"}
COVMAP = {"C1_schedule_to_gt_identity": "COV_C1_schedule_to_gt_identity",
          "C2_generated_to_supplied_adherence": "COV_C2_generated_to_supplied_adherence",
          "C3_full_chain_P_S_G": "COV_C3_full_chain"}


def rd(p):
    return list(csv.DictReader(open(p)))


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def boot(a_rows, b_rows, key, SUB, CLUSTER, label, note):
    """Damage(A,B) = Error_A - Error_B per window. POSITIVE = MORE DAMAGE (never mixed with other tables)."""
    a = np.array([r[key] for r in a_rows], float)
    b = np.array([r[key] for r in b_rows], float)
    d = a - b
    res = C.O1E.cluster_bootstrap(d, SUB, CLUSTER, n_boot=EG.BOOT_N, seed=EG.BOOT_SEED)
    return {"contrast": label, "metric": key, "orientation": "positive = more damage", "note": note,
            "A_macro": C.macro(a, SUB), "B_macro": C.macro(b, SUB),
            "point": res["point"], "lo": res["lo"], "hi": res["hi"],
            "damage_verdict": ("damage confirmed" if res["lo"] > 0 else
                               ("better than the reference" if res["hi"] < 0 else "unresolved")),
            "n_dropped_windows": int(np.sum(~np.isfinite(d)))}


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "figures").mkdir(exist_ok=True)
    t_all = time.perf_counter()
    dev = torch.device("cuda")
    pt = subprocess.run([sys.executable, "-m", "pytest", "tests/test_e2_contract.py", "-o", "addopts=",
                         "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    if pt.returncode != 0:
        print(pt.stdout[-4000:]); raise RuntimeError("E2 tests fail; not validating")
    tests_summary = next((ln for ln in reversed(pt.stdout.splitlines()) if "passed" in ln or "failed" in ln), "")
    print(f"[tests] {tests_summary}", flush=True)

    (ART / "source_artifact_manifest.json").write_text(json.dumps(
        {"e1_artifact_dir": str(E1ART.relative_to(ROOT)),
         "e1_artifact_sha256": {f: fsha(E1ART / f) for f in E1_SOURCES},
         "o3_artifact_sha256": {f: fsha(O3ART / f) for f in
                                ("r1_schedule_quality.csv", "r1_generator_metrics.csv", "multisource_metrics.csv")},
         "reused_without_rerun": ["own-centre and GT-anchored morphology reference rows", "chain coverage",
                                  "R1 reference-card inputs"],
         "rerun_required_for": ["per-window topology class and per-window schedule placement (no generator "
                                "inference needed, but E1 wrote only per-arm aggregates)",
                                "joint event fidelity P->G at 100/150/200 ms",
                                "the full detected event list P (E1 stored only chained beats)"],
         "prediction_source": "DETERMINISTIC FROZEN RERUN of exactly the five validation arms",
         "training_performed": False}, indent=2))
    (ART / "frozen_metric_manifest.json").write_text(json.dumps(
        {"contract_version": EG.VERSION,
         "contract_files_sha256": {f.name: fsha(f) for f in sorted(ART.glob("*_contract.json"))
                                   } | {"contract_v1.json": fsha(ART / "contract_v1.json"),
                                        "metric_taxonomy.json": fsha(ART / "metric_taxonomy.json")},
         "metric_ids": sorted(EG.CONTRACT["metrics"]), "n_metrics": len(EG.CONTRACT["metrics"]),
         "module": "src/ppg2ecg/evaluation/event_geometry_contract.py",
         "module_sha256": fsha(ROOT / "src/ppg2ecg/evaluation/event_geometry_contract.py")}, indent=2))

    coh = C.load_cohort()
    X, Yd, SUB, SITE, WI = coh["X"], coh["Yd"], coh["SUB"], coh["SITE"], coh["WI"]
    gt_pk, CLUSTER = coh["gt_pk"], coh["cluster"]
    base, o2c, tcn, meta = C.load_models(dev, with_r1=True)
    e0 = C.source_bank(len(X))
    print(f"[P] {len(X)} windows, {coh['n_beats']} GT beats, {len(set(CLUSTER.tolist()))} clusters", flush=True)

    SUPP, IDENT = {}, {}
    for name, fam, lv in ARMS:
        S, ID = [], []
        for i in range(len(X)):
            r = np.asarray(gt_pk[i], dtype=np.int64)
            S.append(O3.supplied_schedule(fam, lv, REP, r, SUB[i], SITE[i], int(WI[i])))
            ID.append(O3.retained_pairs(fam, lv, REP, r, S[-1], SUB[i], SITE[i], int(WI[i])))
        SUPP[name], IDENT[name] = S, ID
    SUPP["R1-SCHEDULE"], IDENT["R1-SCHEDULE"] = C.r1_schedules(tcn, X, dev), [None] * len(X)

    W, BEATS, MAC = {}, {}, {}
    for name in ("ORACLE", "JITTER_8", "MISS1", "EXTRA1", "R1-SCHEDULE"):
        warps = C.build_warps(SUPP[name])
        pred, _cn = C.o2c_predict(o2c, X, warps, e0, dev)
        P = C.R2E.pmap(C.S0._peaks, list(pred.astype(np.float64)))
        rows, brows = [], []
        for i in range(len(X)):
            row, beats = EG.apply_contract(gt_pk[i], SUPP[name][i], pred[i], Yd[i], P[i], IDENT[name][i])
            row["row"] = i
            rows.append(row)
            brows += [{"arm": name, "row": i, **b} for b in beats]
        W[name], BEATS[name] = rows, brows
        num = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        MAC[name] = {k: C.macro([r[k] for r in rows], SUB) for k in num}
        print(f"[arm ] {name:<12} T0 {sum(1 for r in rows if r['A6_topology_class'] == EG.T0):>4}/2048 | "
              f"covC3 {MAC[name]['COV_C3_full_chain']:.4f} | ownT6 {MAC[name]['C2_own_T6']:.4f} | "
              f"J2 {MAC[name]['J2_gt_local_deriv_rmse']:.4f} | adh {MAC[name]['AD_F1_50']:.4f}", flush=True)

    # ---------------- 10. reproduction gate against E1 ----------------
    e1m = {r["arm"]: r for r in rd(E1ART / "gt_anchored_local_metrics.csv")}
    e1c = {r["arm"]: r for r in rd(E1ART / "coverage_metrics.csv")}
    e1s = {r["stratum"]: r for r in rd(E1ART / "r1_topology_strata.csv")}
    repro, worst = [], 0.0
    for name in W:
        diffs = {}
        if name in e1m:                       # E1 wrote per-arm morphology rows for the synthetic arms only
            for col, key in E1MAP.items():
                diffs[key] = abs(MAC[name][key] - float(e1m[name][col]))
        pooled = EG.pooled_coverage(W[name])  # E1's coverage is a cohort count ratio, not a per-window macro
        for col, key in COVMAP.items():
            diffs[key] = abs(pooled[key] - float(e1c[name][col]))
        if name == "R1-SCHEDULE":             # reproduce R1 morphology against E1's per-stratum rows
            for cls_ in (EG.T0, EG.T1, EG.T2, EG.T3):
                idx = [r["row"] for r in W[name] if r["A6_topology_class"] == cls_]
                if not idx:
                    continue
                for col, key in (("own_T6", "C2_own_T6"), ("own_T7", "C3_own_T7"),
                                 ("gt_T6", "_gt_T6"), ("gt_T7", "_gt_T7")):
                    got = C.macro([W[name][i][key] for i in idx], SUB[idx])
                    diffs[f"{cls_}:{col}"] = abs(got - float(e1s[cls_][col]))
                got = float(np.nanmean([W[name][i]["B2_mae_ms"] for i in idx]))
                diffs[f"{cls_}:schedule_mae_ms"] = abs(got - float(e1s[cls_]["schedule_mae_ms"]))
        mx = max(diffs.values())
        worst = max(worst, mx)
        repro.append({"arm": name, "max_abs_diff": mx, "passed": bool(mx <= REPRO_TOL),
                      "n_checks": len(diffs), "morphology_row_in_e1": name in e1m,
                      "worst_metric": max(diffs, key=diffs.get)})
        if mx > REPRO_TOL:
            (ART / "decision.json").write_text(json.dumps(
                {"verdict": "E1 REPRODUCTION FAILED", "arm": name, "max_abs_diff": mx}, indent=2))
            raise RuntimeError(f"E1 REPRODUCTION FAILED (STOP): {name} max |Δ| {mx:.3e} on {repro[-1]['worst_metric']}")
    C.wcsv(ART / "e1_reproduction.csv", repro)
    print(f"[repro] all five arms reproduce E1 (max |Δ| {worst:.3e}; "
          f"{sum(r['n_checks'] for r in repro)} checks)", flush=True)

    # ---------------- 11-12. contract validation ----------------
    topo_rows = []
    for name in W:
        topo_rows.append({"arm": name, **EG.exact_set_summary(W[name], SUB),
                          "A1_abs_count_error": MAC[name]["A1_abs_count_error"],
                          "A2_beats_ratio_dev": MAC[name]["A2_beats_ratio_dev"],
                          "A3_missing_fraction": MAC[name]["A3_missing_fraction"],
                          "A4_spurious_fraction": MAC[name]["A4_spurious_fraction"]})
    C.wcsv(ART / "topology_validation.csv", topo_rows)
    TOPO = {r["arm"]: r for r in topo_rows}

    BLOCK = ["A1_abs_count_error", "A2_beats_ratio_dev", "A3_missing_fraction", "A4_spurious_fraction",
             "A5_exact_set_fraction", EG.T0, EG.T1, EG.T2, EG.T3,
             "B1_median_ae_ms", "B2_mae_ms", "B3_p90_ae_ms", "B4_p95_ae_ms",
             "B5_exact_set_mae_ms", "B6_exact_set_p90_ae_ms", "B5_n_T0_windows",
             "SG_F1_50", "SG_F1_100", "SG_F1_150", "SG_F1_200", "SG_PREC", "SG_REC",
             "PG_F1_50", "PG_F1_100", "PG_F1_150", "PG_F1_200", "PG_PREC", "PG_REC",
             "AD_F1_50", "AD_F1_100", "AD_MISS", "AD_SPUR", "AD_MAE",
             "C1_own_T4", "C2_own_T6", "C3_own_T7", "C4_own_T8", "C5_own_local_raw_rmse",
             "C6_own_local_deriv_rmse", "C7_own_local_curvature_err", "C8_own_local_corr",
             "J1_gt_local_raw_rmse", "J2_gt_local_deriv_rmse", "J3_gt_local_curvature_err", "J4_gt_local_corr",
             "D1_raw_rmse", "D2_deriv_rmse", "D3_curvature_err",
             "WD1_raw_rmse", "WD2_deriv_rmse", "WD3_curvature_err",
             "COV_C1_schedule_to_gt_identity", "COV_C2_generated_to_supplied_adherence", "COV_C3_full_chain",
             "COV_C4_gt_beats_excluded", "COV_C5_generated_beats_excluded"]
    # every frozen contract metric must reach an artifact; a silently dropped metric is a contract violation
    PUBLISHED = set(BLOCK) | {"A6_topology_class"}
    MISSING = [mid for mid in EG.CONTRACT["metrics"]
               if not any(k.startswith(mid.split("_")[0]) or mid in k for k in PUBLISHED)]
    if MISSING:
        raise RuntimeError(f"frozen contract metrics never published: {MISSING}")
    C.wcsv(ART / "contract_validation_metrics.csv",
           [{"arm": a, **{k: (TOPO[a][k] if k in TOPO[a] else MAC[a][k]) for k in BLOCK},
             **{f"pooled_{k}": v for k, v in EG.pooled_coverage(W[a]).items() if k.startswith("COV_")},
             "adherence_label": EG.adherence_label(MAC[a]["AD_F1_50"])} for a in
            ("ORACLE", "JITTER_8", "MISS1", "EXTRA1", "R1-SCHEDULE")])

    bt = []
    bt.append(boot(W["JITTER_8"], W["ORACLE"], "B2_mae_ms", SUB, CLUSTER, "JITTER_8_vs_ORACLE", "V2 placement axis responds"))
    bt.append(boot(W["JITTER_8"], W["ORACLE"], "J2_gt_local_deriv_rmse", SUB, CLUSTER, "JITTER_8_vs_ORACLE", "V3"))
    bt.append(boot(W["JITTER_8"], W["ORACLE"], "J3_gt_local_curvature_err", SUB, CLUSTER, "JITTER_8_vs_ORACLE", "V4"))
    bt.append(boot(W["JITTER_8"], W["ORACLE"], "D2_deriv_rmse", SUB, CLUSTER, "JITTER_8_vs_ORACLE",
                   "V6 DerivativePlacementExcess, same functional"))
    bt.append(boot(W["JITTER_8"], W["ORACLE"], "D3_curvature_err", SUB, CLUSTER, "JITTER_8_vs_ORACLE",
                   "V7 CurvaturePlacementExcess, same functional"))
    for k in ("WD2_deriv_rmse", "WD3_curvature_err"):
        bt.append(boot(W["JITTER_8"], W["ORACLE"], k, SUB, CLUSTER, "JITTER_8_vs_ORACLE",
                       "SECONDARY window-level variant of the placement excess; not a gate"))
    for arm in ("MISS1", "EXTRA1"):
        for k, lab in (("C2_own_T6", "T6"), ("C3_own_T7", "T7"), ("C1_own_T4", "T4"), ("C4_own_T8", "T8")):
            bt.append(boot(W[arm], W["ORACLE"], k, SUB, CLUSTER, f"{arm}_vs_ORACLE", f"V8-V10 topology axis {lab}"))
        for k, lab in (("C2_own_T6", "T6"), ("C3_own_T7", "T7")):
            bt.append(boot(W[arm], W["JITTER_8"], k, SUB, CLUSTER, f"{arm}_vs_JITTER_8",
                           f"V11-V12 topology excess over severe jitter, {lab}"))
    bt.append(boot(W["JITTER_8"], W["ORACLE"], "C2_own_T6", SUB, CLUSTER, "JITTER_8_vs_ORACLE", "reference: own-centre T6"))
    bt.append(boot(W["JITTER_8"], W["ORACLE"], "C3_own_T7", SUB, CLUSTER, "JITTER_8_vs_ORACLE", "reference: own-centre T7"))
    C.wcsv(ART / "contract_validation_bootstrap.csv", bt)
    B = {(r["contrast"], r["metric"]): r for r in bt}
    C.wcsv(ART / "placement_excess.csv",
           [{"quantity": "DerivativePlacementExcess", "definition":
             "[J2(J8) - C6(J8)] - [J2(ORACLE) - C6(ORACLE)], same functional",
             **{k: B[("JITTER_8_vs_ORACLE", "D2_deriv_rmse")][k] for k in ("point", "lo", "hi", "damage_verdict")}},
            {"quantity": "CurvaturePlacementExcess", "definition":
             "[J3(J8) - C7(J8)] - [J3(ORACLE) - C7(ORACLE)], same functional",
             **{k: B[("JITTER_8_vs_ORACLE", "D3_curvature_err")][k] for k in ("point", "lo", "hi", "damage_verdict")}},
            {"quantity": "DerivativePlacementExcess (window-level variant, SECONDARY)", "definition":
             "median(J2) - median(C6) per window, then the same contrast; reported because the contract "
             "defines D as a PER-BEAT difference and median(J-C) != median(J) - median(C)",
             **{k: B[("JITTER_8_vs_ORACLE", "WD2_deriv_rmse")][k] for k in ("point", "lo", "hi", "damage_verdict")}},
            {"quantity": "CurvaturePlacementExcess (window-level variant, SECONDARY)", "definition":
             "median(J3) - median(C7) per window, then the same contrast",
             **{k: B[("JITTER_8_vs_ORACLE", "WD3_curvature_err")][k] for k in ("point", "lo", "hi", "damage_verdict")}},
            {"quantity": "PROHIBITED cross-functional comparison", "definition":
             "own-centre T6 versus GT-anchored derivative RMSE", "point": "", "lo": "", "hi": "",
             "damage_verdict": "not computed: forbidden by contract_v1 prohibited_comparisons"}])

    # ---------------- V1-V13 ----------------
    r1 = MAC["R1-SCHEDULE"]
    blocks = {"A_event_set": ["A5_exact_set_fraction", "A1_abs_count_error", "A3_missing_fraction",
                              "A4_spurious_fraction"],
              "B_placement": ["B2_mae_ms", "B3_p90_ae_ms", "B5_exact_set_mae_ms", "B6_exact_set_p90_ae_ms"],
              "C_joint_event": ["PG_F1_50", "PG_F1_100", "PG_F1_150"],
              "D_adherence": ["AD_F1_50", "AD_MAE"],
              "E_own_centre": ["C1_own_T4", "C2_own_T6", "C3_own_T7", "C4_own_T8",
                               "C6_own_local_deriv_rmse", "C7_own_local_curvature_err", "COV_C3_full_chain"],
              "F_joint_structure": ["J1_gt_local_raw_rmse", "J2_gt_local_deriv_rmse",
                                    "J3_gt_local_curvature_err", "J4_gt_local_corr"]}
    r1_block_ok = {b: all(np.isfinite(TOPO["R1-SCHEDULE"][k] if k in TOPO["R1-SCHEDULE"] else r1[k]) for k in ks)
                   for b, ks in blocks.items()}
    V = {"V1": bool(TOPO["JITTER_8"][EG.T0] == len(W["JITTER_8"])),
         "V2": bool(B[("JITTER_8_vs_ORACLE", "B2_mae_ms")]["lo"] > 0),
         "V3": bool(B[("JITTER_8_vs_ORACLE", "J2_gt_local_deriv_rmse")]["lo"] > 0),
         "V4": bool(B[("JITTER_8_vs_ORACLE", "J3_gt_local_curvature_err")]["lo"] > 0),
         "V5": bool(MAC["JITTER_8"]["AD_F1_50"] >= EG.HIGH_ADHERENCE),
         "V6": bool(B[("JITTER_8_vs_ORACLE", "D2_deriv_rmse")]["lo"] > 0),
         "V7": bool(B[("JITTER_8_vs_ORACLE", "D3_curvature_err")]["lo"] > 0),
         "V8": bool(B[("MISS1_vs_ORACLE", "C2_own_T6")]["lo"] > 0),
         "V9": bool(B[("MISS1_vs_ORACLE", "C3_own_T7")]["lo"] > 0),
         "V10": bool(B[("EXTRA1_vs_ORACLE", "C2_own_T6")]["lo"] > 0
                     or B[("EXTRA1_vs_ORACLE", "C3_own_T7")]["lo"] > 0),
         "V11": bool(B[("MISS1_vs_JITTER_8", "C2_own_T6")]["lo"] > 0
                     and B[("MISS1_vs_JITTER_8", "C3_own_T7")]["lo"] > 0),
         "V12": bool(B[("EXTRA1_vs_JITTER_8", "C2_own_T6")]["lo"] > 0
                     or B[("EXTRA1_vs_JITTER_8", "C3_own_T7")]["lo"] > 0),
         "V13": bool(all(r1_block_ok.values()) and np.isfinite(r1["COV_C3_full_chain"]))}
    sanity = {"MISS1_all_T2": bool(TOPO["MISS1"][EG.T2] == len(W["MISS1"])),
              "EXTRA1_all_T3": bool(TOPO["EXTRA1"][EG.T3] == len(W["EXTRA1"]))}
    (ART / "validation_gates.json").write_text(json.dumps(
        {"gates": V, "sanity_checks": sanity, "r1_block_computable": r1_block_ok,
         "definitions": {"V1": "JITTER_8 topology 100% T0", "V2": "schedule timing MAE clearly increases",
                         "V3": "GT-anchored local derivative RMSE clearly worsens",
                         "V4": "GT-anchored local curvature clearly worsens",
                         "V5": f"P->S adherence >= {EG.HIGH_ADHERENCE}",
                         "V6": "DerivativePlacementExcess CI entirely > 0",
                         "V7": "CurvaturePlacementExcess CI entirely > 0",
                         "V8": "MISS1 own-centre T6 clearly worse than ORACLE",
                         "V9": "MISS1 own-centre T7 clearly worse than ORACLE",
                         "V10": "at least one EXTRA1 own-centre T6/T7 clearly worse than ORACLE",
                         "V11": "MISS1 excess over JITTER_8 > 0 with CI > 0 for both T6 and T7",
                         "V12": "EXTRA1 excess over JITTER_8 > 0 with CI > 0 for at least one of T6/T7",
                         "V13": "all six mandatory blocks computable for R1 with finite values and coverage"}},
        indent=2))

    dec = EG.decide_contract(V)
    verdict = dec["verdict"]
    (ART / "decision.json").write_text(json.dumps(
        {**dec, "sanity_checks": sanity,
         "contract_version": EG.VERSION, "e1_reproduction_max_abs_diff": worst, "e1_reproduction_tolerance": REPRO_TOL,
         "status": "measurement-contract construction after E1; not confirmation, not a model, not deployable",
         "e1_verdict_unchanged": "MIXED TOPOLOGY AND PLACEMENT LIMITATION",
         "training_performed": False,
         }, indent=2, default=float))

    # ---------------- R1 reference card, loaded from frozen artifacts ----------------
    strat = {r["stratum"]: r for r in rd(E1ART / "r1_topology_strata.csv")}
    o3q = next(r for r in rd(O3ART / "r1_schedule_quality.csv") if not r.get("site"))
    o3g = next(r for r in rd(O3ART / "r1_generator_metrics.csv") if r["arm"] == "O2C-R1-SCHEDULE")
    n_tot = sum(int(strat[s]["n_rows"]) for s in strat)
    card = {"contract_version": EG.VERSION, "role": "development reference card for the next schedule-predictor "
            "experiment; every value is LOADED from a frozen artifact, none is typed by hand",
            "cohort": "frozen O2/O3/E1 2,048-window development cohort (an0, k2s)",
            "exact_set_fraction": int(strat[EG.T0]["n_rows"]) / n_tot,
            "topology_distribution": {s: int(strat[s]["n_rows"]) / n_tot for s in strat},
            "dominant_error": max((s for s in strat if s != EG.T0), key=lambda s: int(strat[s]["n_rows"])),
            "schedule_f1_at_50": float(o3q["f1_at_50"]), "schedule_f1_at_150": float(o3q["f1_at_150"]),
            "schedule_timing_mae_ms": float(o3q["timing_mae_ms"]),
            "schedule_missing": float(o3q["missing"]), "schedule_spurious": float(o3q["spurious"]),
            "adherence_f1_at_50": float(o3g["adherence_f1_at_50"]),
            "exact_set_own_centre_T6": float(strat[EG.T0]["own_T6"]),
            "exact_set_own_centre_T7": float(strat[EG.T0]["own_T7"]),
            "sources": {"e1_r1_topology_strata.csv": fsha(E1ART / "r1_topology_strata.csv"),
                        "o3_r1_schedule_quality.csv": fsha(O3ART / "r1_schedule_quality.csv"),
                        "o3_r1_generator_metrics.csv": fsha(O3ART / "r1_generator_metrics.csv")}}
    (ART / "r1_reference_card.json").write_text(json.dumps(card, indent=2))

    C.wcsv(ART / "contract_validation_windows.csv", [{"arm": a, **r} for a in W for r in W[a]])
    C.wcsv(ART / "contract_validation_beats.csv", [b for a in BEATS for b in BEATS[a]])
    (ART / "provenance.json").write_text(json.dumps(
        {"git": git_sha(ROOT), "prereg": "14a26a7", "utc": datetime.now(timezone.utc).isoformat(),
         "test_subjects_loaded": [], "training_performed": False,
         "prediction_source": "DETERMINISTIC FROZEN RERUN (five validation arms)",
         "n_windows": int(len(X)), "n_gt_beats": coh["n_beats"], "n_clusters": int(len(set(CLUSTER.tolist()))),
         "nfe": C.NFE, "source_seed": C.SRC_SEED, "contract_version": EG.VERSION,
         "e1_reproduction_max_abs_diff": worst, "tests": tests_summary,
         "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
         "gpu": torch.cuda.get_device_name(0), "wall_s": time.perf_counter() - t_all,
         "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20)}, indent=2, default=str))
    print("\n[gates] " + " ".join(f"{k}:{'PASS' if v else 'FAIL'}" for k, v in V.items()), flush=True)
    print(f"[sanity] {sanity}", flush=True)
    print(f"[VERDICT] {verdict}", flush=True)
    print(f"[done] E2 validation in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
