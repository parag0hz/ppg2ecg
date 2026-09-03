"""O3 stage A — synthetic schedule-error tolerance sweep (preregistration sections 3, 5-19).

Commit-order steps 11-19. NO TRAINING: B, O2c and the O2b operator are loaded frozen and no optimizer exists.
Stage B (the frozen R1 bridge) may only run after this script has written synthetic_curve_frozen.json.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.training.train_a0 import git_sha

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import o3_common as C  # noqa: E402

ROOT, ART = C.ROOT, C.ART
FAMILIES = (("JITTER", (1, 2, 4, 6, 8)), ("MISS", (1, 2)), ("EXTRA", (1, 2)))
FROZEN_FILES = ("cohort_manifest.csv", "frozen_model_regression.json", "perturbation_manifest.csv",
                "schedule_precheck.csv", "schedule_quality_metrics.csv", "operator_floor_metrics.csv",
                "synthetic_generator_metrics.csv", "schedule_adherence.csv", "synthetic_paired_bootstrap.csv",
                "retained_gain.csv", "shape_only_diagnostic.csv")


def build_supplied(coh, family, level, rep):
    S, RP = [], []
    for i in range(len(coh["X"])):
        r = np.asarray(coh["gt_pk"][i], dtype=np.int64)
        s = O3.supplied_schedule(family, level, rep, r, coh["SUB"][i], coh["SITE"][i], int(coh["WI"][i]))
        S.append(s)
        RP.append(O3.retained_pairs(family, level, rep, r, s, coh["SUB"][i], coh["SITE"][i], int(coh["WI"][i])))
    return S, RP


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "figures").mkdir(exist_ok=True)
    t_all = time.perf_counter()
    dev = torch.device("cuda")
    pt = subprocess.run([sys.executable, "-m", "pytest", "tests/test_o3_schedule_tolerance.py", "-o", "addopts=",
                         "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    if pt.returncode != 0:
        print(pt.stdout[-4000:]); raise RuntimeError("O3 tests fail; not evaluating")
    tests_summary = next((ln for ln in reversed(pt.stdout.splitlines()) if "passed" in ln or "failed" in ln), "")
    print(f"[tests] {tests_summary}", flush=True)

    coh = C.load_cohort()
    X, Yd, SUB, SITE, WI = coh["X"], coh["Yd"], coh["SUB"], coh["SITE"], coh["WI"]
    gt_pk, gt_tg, iqr, CLUSTER = coh["gt_pk"], coh["gt_tg"], coh["iqr"], coh["cluster"]
    C.wcsv(ART / "cohort_manifest.csv", [{"row": i, "subject": SUB[i], "site": SITE[i],
                                          "array_pos": int(coh["POS"][i]), "window_index": int(WI[i]),
                                          "K": int(len(gt_pk[i])), "cluster": CLUSTER[i]} for i in range(len(X))])
    print(f"[P] {len(X)} windows, {coh['n_beats']} GT beats, {len(set(CLUSTER.tolist()))} clusters", flush=True)

    base, o2c, meta = C.load_models(dev)
    (ART / "frozen_component_manifest.json").write_text(json.dumps(
        {**meta, "training_performed": False, "optimizer_constructed": False}, indent=2, default=str))
    e0 = C.source_bank(len(X))

    # ---------------- 11. frozen model regression ----------------
    orc_S, orc_RP = build_supplied(coh, "JITTER", 0, 0)
    orc_pc = [O3.precheck_schedule(s) for s in orc_S]              # section 8: before ANY generator call
    C.wcsv(ART / "schedule_precheck.csv", [{"condition": "ORACLE", "family": "JITTER", "level": 0, "rep": 0,
        "n_windows": len(orc_S), "n_fail": int(sum(not c["passed"] for c in orc_pc)),
        "min_spacing": int(min(c["min_spacing"] for c in orc_pc)),
        "max_core_offset": float(max(c["max_core_offset"] for c in orc_pc)),
        "min_M": int(min(c["M"] for c in orc_pc)), "max_M": int(max(c["M"] for c in orc_pc))}])
    if any(not c["passed"] for c in orc_pc):
        raise RuntimeError("the GT (ORACLE) schedule fails the section 8 precheck (STOP)")
    p_b = C.R2E.gen_plain(base, X, e0, C.NFE, dev)
    rows_b, _a, _b = C.R2E.score(p_b, Yd, gt_pk)
    mac_b = C.R2E.macro_rows(rows_b, SUB)
    al_b = C.aligned_rows(p_b, gt_tg, iqr)
    warps_o = C.build_warps(orc_S)
    p_o, _can = C.o2c_predict(o2c, X, warps_o, e0, dev)
    rows_o, _c, _d = C.R2E.score(p_o, Yd, gt_pk)
    mac_o = C.R2E.macro_rows(rows_o, SUB)
    al_o = C.aligned_rows(p_o, gt_tg, iqr)
    alm_o = {t: C.macro([r[C.NAE[t]] for r in al_o], SUB) for t in C.ALIGNED}
    got = {**{k: mac_b[k] for k in C.FROZEN_B}, "O2C_f1_excess": mac_o["f1_excess"],
           **{f"O2C_{C.NAE[t]}": alm_o[t] for t in C.ALIGNED}}
    exp = {**C.FROZEN_B, "O2C_f1_excess": C.FROZEN_O2C["f1_excess"],
           **{f"O2C_{C.NAE[t]}": C.FROZEN_O2C[C.NAE[t]] for t in C.ALIGNED}}
    bad = {k: [got[k], v] for k, v in exp.items() if abs(got[k] - v) > C.REG_TOL}
    mx = max(abs(got[k] - v) for k, v in exp.items())
    (ART / "frozen_model_regression.json").write_text(json.dumps(
        {"tolerance": C.REG_TOL, "expected": exp, "reproduced": got, "max_abs_diff": mx,
         "passed": not bad, "mismatches": bad,
         "verdict": None if not bad else O3.VERDICT_REGRESSION}, indent=2))
    if bad:
        raise RuntimeError(f"{O3.VERDICT_REGRESSION} (STOP): {bad}")
    print(f"[reg] B and O2c ORACLE reproduced (max |Δ| {mx:.3e})", flush=True)

    # ---------------- 12-13. schedules, prechecks, quality, operator floor ----------------
    CONDS = [("ORACLE", "JITTER", 0, rep) for rep in O3.REPS] + \
            [(O3.condition_name(f, lv), f, lv, rep) for f, lvs in FAMILIES for lv in lvs for rep in O3.REPS]
    SUP, RET, pm, pc, sq, fl = {}, {}, [], [], [], []
    floor_med = {}
    for name, fam, lv, rep in CONDS:
        key = (name, rep)
        if name == "ORACLE" and rep > 0:
            SUP[key], RET[key] = SUP[("ORACLE", 0)], RET[("ORACLE", 0)]
        else:
            try:
                SUP[key], RET[key] = build_supplied(coh, fam, lv, rep)
            except ValueError as exc:                               # not enough interior beats / eligible intervals
                v = O3.VERDICT_EXTRA_INVALID if fam == "EXTRA" else "MISS PERTURBATION DESIGN INVALID"
                (ART / "decision.json").write_text(json.dumps(
                    {"verdict": v, "condition": name, "rep": rep, "reason": str(exc)}, indent=2))
                raise RuntimeError(f"{v} (STOP): {name} rep{rep}: {exc}") from None
        S = SUP[key]
        checks = [O3.precheck_schedule(s) for s in S]
        n_fail = int(sum(not c["passed"] for c in checks))
        agg = {}
        for k in checks[0]["checks"]:
            agg[f"fail_{k}"] = int(sum(not c["checks"][k] for c in checks))
        pc.append({"condition": name, "family": fam, "level": lv, "rep": rep, "n_windows": len(S),
                   "n_fail": n_fail, "min_spacing": int(min(c["min_spacing"] for c in checks)),
                   "max_core_offset": float(max(c["max_core_offset"] for c in checks)),
                   "min_M": int(min(c["M"] for c in checks)), "max_M": int(max(c["M"] for c in checks)), **agg})
        if n_fail:
            v = O3.VERDICT_JITTER_INVALID if fam == "JITTER" else (
                O3.VERDICT_EXTRA_INVALID if fam == "EXTRA" else "MISS PERTURBATION DESIGN INVALID")
            C.wcsv(ART / "schedule_precheck.csv", pc)
            (ART / "decision.json").write_text(json.dumps({"verdict": v, "condition": name, "rep": rep,
                                                           "n_fail": n_fail}, indent=2))
            raise RuntimeError(f"{v} (STOP): {name} rep{rep}, {n_fail} windows fail the precheck")
        shift = [np.abs(np.asarray(S[i], float) - np.asarray(gt_pk[i], float)).max()
                 if len(S[i]) == len(gt_pk[i]) else np.nan for i in range(len(S))]
        pm.append({"condition": name, "family": fam, "level": lv, "rep": rep, "salt": {"JITTER": O3.JITTER_SALT,
                   "MISS": O3.MISS_SALT, "EXTRA": O3.EXTRA_SALT}[fam], "n_windows": len(S),
                   "mean_M": float(np.mean([len(s) for s in S])), "mean_K": float(np.mean([len(p) for p in gt_pk])),
                   "max_abs_shift_samples": float(np.nanmax(shift)) if np.isfinite(shift).any() else np.nan,
                   "identity_rows": 0})
        q = [O3.schedule_quality(gt_pk[i], S[i]) for i in range(len(S))]
        sq.append({"condition": name, "family": fam, "level": lv, "rep": rep,
                   **{k: C.macro([r[k] for r in q], SUB) for k in
                      ("f1_at_50", "f1_at_100", "f1_at_150", "f1_at_200", "precision", "recall", "missing",
                       "spurious", "beats_ratio_dev", "timing_median_ae_ms", "timing_mae_ms")}})
        warps = C.build_warps(S)
        frows, fmed = C.S0.roundtrip_metrics(coh["Y"], warps, gt_pk, gt_tg, iqr, name)
        floor_med[key] = fmed
        fl.append({"condition": name, "family": fam, "level": lv, "rep": rep,
                   "raw_rmse": fmed["raw_rmse"], "qrs_core_rmse": fmed["qrs_core_rmse"],
                   "f1_at_50": fmed["f1_at_50"], "beat_count_diff": fmed["beat_count_diff"],
                   "nAE_T4": fmed["T4"], "nAE_T6": fmed["T6"], "nAE_T7": fmed["T7"], "nAE_T8": fmed["T8"],
                   "floor_exceeds_0.020": bool(fmed["T6"] > 0.020 or fmed["T7"] > 0.020),
                   "interpretation": ("OPERATOR-CONFOUNDED" if (lv > 0 and (fmed["T6"] > 0.020 or fmed["T7"] > 0.020))
                                      else "operator floor below the 0.020 margin"),
                   "stops_run": bool(fam == "JITTER" and lv == 1 and (fmed["T6"] > 0.020 or fmed["T7"] > 0.020))})
        print(f"[sched] {name:<10} rep{rep} | F1@50 {sq[-1]['f1_at_50']:.4f} MAE {sq[-1]['timing_mae_ms']:.2f} ms | "
              f"floor T6 {fmed['T6']:.5f} T7 {fmed['T7']:.5f}", flush=True)
    C.wcsv(ART / "perturbation_manifest.csv", pm)
    C.wcsv(ART / "schedule_precheck.csv", pc)
    C.wcsv(ART / "schedule_quality_metrics.csv", sq)
    C.wcsv(ART / "operator_floor_metrics.csv", fl)

    brittle = [(k, floor_med[k]["T6"], floor_med[k]["T7"]) for k in floor_med
               if k[0] == "JITTER_1" and (floor_med[k]["T6"] > 0.020 or floor_med[k]["T7"] > 0.020)]
    if brittle:
        (ART / "decision.json").write_text(json.dumps(
            {"verdict": O3.VERDICT_BRITTLE, "offending": [[k[0], k[1], t6, t7] for k, t6, t7 in brittle]}, indent=2))
        raise RuntimeError(f"{O3.VERDICT_BRITTLE} (STOP): {brittle}")
    print("[floor] J1 early-falsification gate PASS", flush=True)

    pf = ART / "runtime_preflight.json"
    if not pf.exists():
        print("\n[stop] steps 11-13 complete. Run scripts/o3_preflight.py (step 14), then re-run this script.",
              flush=True)
        return 0
    pfj = json.loads(pf.read_text())
    if pfj.get("stop"):
        raise RuntimeError(f"runtime preflight projected {pfj['projected_total_gpu_hours']:.2f} GPU-h "
                           f"> {pfj['budget_gpu_hours']} h budget (STOP)")
    print(f"[preflight] projected {pfj['projected_total_gpu_hours']:.3f} GPU-h "
          f"(budget {pfj['budget_gpu_hours']}) -> proceed", flush=True)

    # ---------------- 15-17. generator sweep, bootstrap, gates ----------------
    gm, ad, bt, so, gates = [], [], [], [], []
    ROWS = {("ORACLE", 0): (rows_o, al_o)}
    for name, fam, lv, rep in CONDS:
        key = (name, rep)
        S, RP = SUP[key], RET[key]
        if key in ROWS:
            rows_c, al_c = ROWS[key]; p_c = p_o
        elif name == "ORACLE":
            rows_c, al_c, p_c = rows_o, al_o, p_o
        else:
            warps = C.build_warps(S)
            p_c, _cn = C.o2c_predict(o2c, X, warps, e0, dev)
            rows_c, _e, _f = C.R2E.score(p_c, Yd, gt_pk)
            al_c = C.aligned_rows(p_c, gt_tg, iqr)
        mac_c = C.R2E.macro_rows(rows_c, SUB)
        alm_c = {t: C.macro([r[C.NAE[t]] for r in al_c], SUB) for t in C.ALIGNED}
        gen_pk = C.R2E.pmap(C.S0._peaks, list(p_c.astype(np.float64)))
        adh = [O3.adherence(S[i], gen_pk[i]) for i in range(len(S))]
        arow = {"condition": name, "family": fam, "level": lv, "rep": rep,
                **{k: C.macro([r[k] for r in adh], SUB) for k in adh[0]}}
        ad.append(arow)
        gm.append({"condition": name, "family": fam, "level": lv, "rep": rep,
                   **{k: mac_c[k] for k in C.EVENT_M}, **{C.NAE[t]: alm_c[t] for t in C.ALIGNED},
                   **{k: mac_c[k] for k in C.STRUCT_M}, "adherence_f1_at_50": arow["adherence_f1_at_50"]})
        res = C.paired_boot(rows_b, al_b, rows_c, al_c, SUB, CLUSTER)
        for m, r in res.items():
            bt.append({"condition": name, "family": fam, "level": lv, "rep": rep, "metric": m,
                       "orientation": "higher_better" if m == "f1_excess" else "lower_better",
                       "positive_means": "supplied-schedule O2c better than B", "B": r["B"], "arm": r["arm"],
                       "point": r["point"], "lo": r["lo"], "hi": r["hi"], "verdict": r["verdict"]})
        g = O3.joint_gates(res)
        gates.append({"stage": "synthetic", "condition": name, "family": fam, "level": lv, "rep": rep, **g})
        so.append({"condition": name, "family": fam, "level": lv, "rep": rep,
                   **C.shape_only(p_c, Yd, S, gt_pk, RP, iqr, SUB)})
        print(f"[gen ] {name:<10} rep{rep} | f1x {mac_c['f1_excess']:+.4f} T6 {alm_c[C.ALIGNED[1]]:.4f} "
              f"T7 {alm_c[C.ALIGNED[2]]:.4f} adh {arow['adherence_f1_at_50']:.4f} | "
              + " ".join(k for k in ("G1", "G2", "G3", "G4", "G5", "G6") if g[k]) +
              f" -> {'SURVIVES' if g['survives'] else 'FAILS'}", flush=True)
    C.wcsv(ART / "synthetic_generator_metrics.csv", gm)
    C.wcsv(ART / "schedule_adherence.csv", ad)
    C.wcsv(ART / "synthetic_paired_bootstrap.csv", bt)
    C.wcsv(ART / "shape_only_diagnostic.csv", so)
    C.wcsv(ART / "joint_benefit_gates.csv", gates)

    # ---------------- retained gain (descriptive, never clipped) ----------------
    GM = {(r["condition"], r["rep"]): r for r in gm}
    orc = GM[("ORACLE", 0)]
    rg = []
    for r in gm:
        row = {"condition": r["condition"], "family": r["family"], "level": r["level"], "rep": r["rep"],
               "event_gain_retention": O3.retention(r["f1_excess"], mac_b["f1_excess"], orc["f1_excess"], True)}
        for t in C.ALIGNED:
            k = C.NAE[t]
            row[f"morph_gain_retention_{k}"] = O3.retention(r[k], C.macro([x[k] for x in al_b], SUB), orc[k], False)
        rg.append(row)
    C.wcsv(ART / "retained_gain.csv", rg)

    # ---------------- 17-18. freeze the synthetic tolerance ----------------
    by_level = {}
    for g in gates:
        by_level.setdefault((g["family"], g["level"]), []).append(g)
    level_pass = {0: O3.level_survives(by_level[("JITTER", 0)])} if ("JITTER", 0) in by_level else {}
    orc_gates = [g for g in gates if g["condition"] == "ORACLE"]
    level_pass[0] = O3.level_survives(orc_gates)
    for j in (1, 2, 4, 6, 8):
        level_pass[j] = O3.level_survives(by_level[("JITTER", j)])
    miss_pass = {n: O3.level_survives(by_level[("MISS", n)]) for n in (1, 2)}
    extra_pass = {n: O3.level_survives(by_level[("EXTRA", n)]) for n in (1, 2)}
    jm = O3.j_max(level_pass)
    q_at = None if jm is None else [r for r in sq if r["condition"] == O3.condition_name("JITTER", jm)]
    frozen = {"utc": datetime.now(timezone.utc).isoformat(), "git": git_sha(ROOT), "prereg": "60d1810",
              "jitter_level_pass": {int(k): bool(v) for k, v in level_pass.items()},
              "j_max_samples": jm, "j_max_ms": (None if jm is None else jm / C.FS * 1000.0),
              "miss_pass": {int(k): bool(v) for k, v in miss_pass.items()},
              "extra_pass": {int(k): bool(v) for k, v in extra_pass.items()},
              "schedule_quality_at_j_max": q_at,
              "gates": [{k: g[k] for k in ("condition", "rep", "G1", "G2", "G3", "G4", "G5", "G6", "survives")}
                        for g in gates],
              "schedule_quality_table": sq,
              "level_summary": [{"condition": c, "family": f_, "level": l_,
                                 **{k: float(np.mean([r[k] for r in gm if r["condition"] == c]))
                                    for k in ("f1_excess", "nAE_T4", "nAE_T6", "nAE_T7", "nAE_T8",
                                              "adherence_f1_at_50")},
                                 "schedule_f1_at_50": float(np.mean([r["f1_at_50"] for r in sq if r["condition"] == c])),
                                 "schedule_timing_mae_ms": float(np.mean([r["timing_mae_ms"] for r in sq
                                                                          if r["condition"] == c])),
                                 "reps": sorted({r["rep"] for r in gm if r["condition"] == c}),
                                 "survives_all_reps": bool(O3.level_survives(
                                     [g for g in gates if g["condition"] == c]))}
                                for c, f_, l_ in sorted({(r["condition"], r["family"], r["level"]) for r in gm})],
              "artifact_sha256": {f: C.fsha(ART / f) for f in FROZEN_FILES},
              "note": "joint_benefit_gates.csv and the multisource files also receive stage-B rows later, so "
                      "they are not hashed here; the synthetic G1-G6 table and schedule-quality table are "
                      "embedded inline instead and are what stage B may not influence."}
    (ART / "synthetic_curve_frozen.json").write_text(json.dumps(frozen, indent=2, default=str))
    print(f"[freeze] J_MAX = {jm} samples | MISS {miss_pass} | EXTRA {extra_pass}", flush=True)

    # ---------------- 19. selected multi-source arms ----------------
    from ppg2ecg.evaluation import q1_corruption as Q
    unc = Q.uncertainty_positions(SUB, SITE, WI)
    assert len(unc) == 512
    SUBu, CLu = SUB[unc], CLUSTER[unc]
    banks = {sd: C.source_bank(len(X), sd)[unc] for sd in Q.UNC_SEEDS}
    ms_rows, U = [], {}
    plan = [("B", None, None), ("O2C-ORACLE", "ORACLE", 0), ("JITTER_4", "JITTER_4", 0),
            ("JITTER_8", "JITTER_8", 0), ("MISS1", "MISS_1", 0), ("EXTRA1", "EXTRA_1", 0)]
    for arm, cond, rep in plan:
        if arm == "B":
            Ss, wu = None, None
        else:
            Ss = [SUP[(cond, rep)][i] for i in unc]
            wu = C.build_warps(Ss)
        Sm, adh_s = [], []
        for sd in Q.UNC_SEEDS:
            if arm == "B":
                pr = C.R2E.gen_plain(base, X[unc], banks[sd], C.NFE, dev)
            else:
                pr, _cn = C.o2c_predict(o2c, X[unc], wu, banks[sd], dev)
            Sm.append(pr)
            pk = C.R2E.pmap(C.S0._peaks, list(pr.astype(np.float64)))
            if Ss is not None:
                adh_s.append(C.macro([O3.adherence(Ss[i], pk[i])["adherence_f1_at_50"] for i in range(len(unc))], SUBu))
        Sm = np.stack(Sm)
        flat = C.R2E.pmap(C.S0._peaks, [Sm[s, i].astype(np.float64) for i in range(Sm.shape[1]) for s in range(Sm.shape[0])])
        rows = [Q.uncertainty_from_samples(Sm[:, i], flat[i * Sm.shape[0]:(i + 1) * Sm.shape[0]], gt_pk[unc[i]])
                for i in range(Sm.shape[1])]
        U[arm] = rows
        ms_rows.append({"arm": arm, "condition": cond or "-", "rep": "-" if rep is None else rep,
                        "n_windows": len(rows), "n_sources": len(Q.UNC_SEEDS),
                        "beat_count_SD": C.macro([r["u3_beatcount_sd"] for r in rows], SUBu),
                        "pairwise_event_F1_50": C.macro([r["u4_pairwise_event_f1_50"] for r in rows], SUBu),
                        "pairwise_event_F1_150": C.macro([r["u5_pairwise_event_f1_150"] for r in rows], SUBu),
                        "pointwise_waveform_SD": C.macro([r["u1_pointwise_sd"] for r in rows], SUBu),
                        "pairwise_waveform_RMSE": C.macro([r["u2_pairwise_rmse"] for r in rows], SUBu),
                        "adherence_f1_at_50_across_sources": float(np.mean(adh_s)) if adh_s else np.nan})
        print(f"[ms  ] {arm:<12} bcSD {ms_rows[-1]['beat_count_SD']:.4f} pairF1 {ms_rows[-1]['pairwise_event_F1_50']:.4f}", flush=True)
    C.wcsv(ART / "multisource_metrics.csv", ms_rows)
    mb = []
    for arm in [p[0] for p in plan if p[0] != "B"]:
        s1 = C.O1E.cluster_bootstrap(np.array([r["u3_beatcount_sd"] for r in U["B"]], float) -
                                     np.array([r["u3_beatcount_sd"] for r in U[arm]], float), SUBu, CLu,
                                     n_boot=O3.BOOT_N, seed=O3.BOOT_SEED)
        s2 = C.O1E.cluster_bootstrap(np.array([r["u4_pairwise_event_f1_50"] for r in U[arm]], float) -
                                     np.array([r["u4_pairwise_event_f1_50"] for r in U["B"]], float), SUBu, CLu,
                                     n_boot=O3.BOOT_N, seed=O3.BOOT_SEED)
        mb.append({"arm": arm, "gate": "G7a-form", "quantity": "beat-count SD (B - arm)", **s1})
        mb.append({"arm": arm, "gate": "G7b-form", "quantity": "pairwise event F1@50 (arm - B)", **s2})
    C.wcsv(ART / "multisource_bootstrap.csv", mb)

    (ART / "provenance.json").write_text(json.dumps(
        {"stage": "A (synthetic)", "git": git_sha(ROOT), "prereg": "60d1810",
         "utc": datetime.now(timezone.utc).isoformat(), "test_subjects_loaded": [],
         "training_performed": False, "n_windows": int(len(X)), "n_gt_beats": coh["n_beats"],
         "n_clusters": int(len(set(CLUSTER.tolist()))), "nfe": C.NFE, "source_seed": C.SRC_SEED,
         "bootstrap": {"n": O3.BOOT_N, "seed": O3.BOOT_SEED}, "tests": tests_summary,
         "o1_train_iqr": iqr, "conditions": len(CONDS),
         "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
         "gpu": torch.cuda.get_device_name(0), "wall_s": time.perf_counter() - t_all,
         "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20)}, indent=2, default=str))
    print(f"[done] stage A in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
