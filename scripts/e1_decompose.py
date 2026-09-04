"""E1 — event topology / placement / own-centre morphology decomposition (commit-order steps 11-21).

POST-O3 DIAGNOSTIC. NO TRAINING: B, O2c, the O2b operator and the R1 Global-TCN are loaded frozen and no
optimizer is constructed. The R1 diagnostic assignment is computed only after every synthetic contrast has
been written to disk.
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from ppg2ecg.evaluation import e1_decompose as E1
from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.evaluation import rpeaks as RP
from ppg2ecg.probes import r1_cohort as RC
from ppg2ecg.training.train_a0 import git_sha

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import o3_common as C  # noqa: E402

ROOT = C.ROOT
ART = ROOT / "artifacts/e1_event_morphology_decomposition"
O3ART = ROOT / "artifacts/o3_schedule_tolerance"
REG_TOL_SCHED, REG_TOL_GEN = 1e-9, 1e-6
ARMS = [("ORACLE", "JITTER", 0), ("JITTER_2", "JITTER", 2), ("JITTER_4", "JITTER", 4),
        ("JITTER_8", "JITTER", 8), ("MISS1", "MISS", 1), ("EXTRA1", "EXTRA", 1)]
O3NAME = {"ORACLE": "ORACLE", "JITTER_2": "JITTER_2", "JITTER_4": "JITTER_4", "JITTER_8": "JITTER_8",
          "MISS1": "MISS_1", "EXTRA1": "EXTRA_1"}
REP = 0
T6K, T7K = E1.NAE[E1.ALIGNED[1]], E1.NAE[E1.ALIGNED[2]]


def rd(p):
    return list(csv.DictReader(open(p)))


def agree(a: float, b: float, tol: float) -> bool:
    """Both undefined counts as agreement: O3 records NaN for a quantity that a family leaves undefined."""
    if not np.isfinite(a) and not np.isfinite(b):
        return True
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)


def o3row(name, cond, rep=REP, arm=None):
    rows = rd(O3ART / name)
    for r in rows:
        if arm is not None and r.get("arm") == arm:
            return r
        if r.get("condition") == cond and str(r.get("rep")) == str(rep):
            return r
    raise KeyError(f"{name}: {cond} rep{rep} arm={arm}")


def analyse_arm(gen, Yd, S, P, pairs, gt_pk, iqr, n_time=E1.T_LEN):
    """Per-window own-centre + GT-anchored beat analysis over chained beats. Returns (window rows, beat rows)."""
    wrows, brows = [], []
    for i in range(len(gen)):
        s = np.asarray(S[i], dtype=np.int64)
        p = np.asarray(P[i], dtype=np.int64)
        gpos = np.asarray(gt_pk[i], dtype=np.int64)                   # GT sample positions
        ident = {int(a): int(b) for a, b in pairs[i]}                 # supplied index -> GT BEAT INDEX
        chain, _fp, _fn = RP.match_rpeaks(s, p, E1.FS, E1.TOL_CHAIN_MS)
        beats, n_chain = [], 0
        for si, pj in chain:
            if si not in ident:
                continue                                             # inserted beat: no GT identity
            n_chain += 1
            g = ident[si]
            gp = int(gpos[g])                                          # the GT beat's SAMPLE POSITION
            own = E1.beat_shape(gen[i], Yd[i], int(p[pj]), gp, iqr)
            gta = E1.beat_shape(gen[i], Yd[i], gp, gp, iqr)
            if own is None or gta is None:
                continue
            row = {"row": i, "supplied_index": si, "gen_index": int(pj), "gt_index": int(g),
                   "gen_pos": int(p[pj]), "sup_pos": int(s[si]), "gt_pos": gp,
                   "gen_to_gt_ae_ms": abs(int(p[pj]) - gp) / E1.FS * 1000.0,
                   "gen_to_sup_ae_ms": abs(int(p[pj]) - int(s[si])) / E1.FS * 1000.0,
                   **{f"own_{k}": v for k, v in own.items()},
                   **{f"gt_{k}": v for k, v in gta.items()}}
            beats.append(row); brows.append(row)
        keys = [f"own_{k}" for k in E1.SHAPE_KEYS] + [f"gt_{k}" for k in E1.SHAPE_KEYS] + \
               ["gen_to_gt_ae_ms", "gen_to_sup_ae_ms"]
        w = E1.window_median(beats, keys)
        w |= {"row": i, "n_supplied": int(s.size), "n_generated": int(p.size), "n_gt": len(set(ident.values())),
              "n_identity": len(ident), "n_chained": n_chain, "n_eligible_beats": len(beats)}
        wrows.append(w)
    return wrows, brows


def coverage(wrows, gt_pk, S):
    n_sup = sum(int(np.asarray(s).size) for s in S)
    n_gt = sum(len(g) for g in gt_pk)
    n_gen = sum(w["n_generated"] for w in wrows)
    n_id = sum(w["n_identity"] for w in wrows)
    n_ch = sum(w["n_chained"] for w in wrows)
    n_el = sum(w["n_eligible_beats"] for w in wrows)
    return {"C1_schedule_to_gt_identity": n_id / max(n_sup, 1),
            "C2_generated_to_supplied_adherence": n_ch / max(n_id, 1),
            "C3_full_chain_P_S_G": n_el / max(n_sup, 1),
            "C4_gt_beats_excluded": 1.0 - n_el / max(n_gt, 1),
            "C5_generated_beats_excluded": 1.0 - n_el / max(n_gen, 1),
            "n_supplied": n_sup, "n_gt_beats": n_gt, "n_generated": n_gen,
            "n_identity": n_id, "n_chained": n_ch, "n_eligible": n_el}


def boot(a_rows, b_rows, key, SUB, CLUSTER):
    """Damage(A,B) = Error_A - Error_B per window, then the frozen ECG-window clustered bootstrap."""
    a = np.array([r[key] for r in a_rows], float)
    b = np.array([r[key] for r in b_rows], float)
    d = a - b
    res = C.O1E.cluster_bootstrap(d, SUB, CLUSTER, n_boot=E1.BOOT_N, seed=E1.BOOT_SEED)
    res["n_dropped_windows"] = int(np.sum(~np.isfinite(d)))
    res["damage_verdict"] = ("damage confirmed" if res["lo"] > 0 else
                             ("better than the reference" if res["hi"] < 0 else "unresolved"))
    res["A_mean"] = C.macro(a, SUB)
    res["B_mean"] = C.macro(b, SUB)
    return res


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "figures").mkdir(exist_ok=True)
    t_all = time.perf_counter()
    dev = torch.device("cuda")
    pt = subprocess.run([sys.executable, "-m", "pytest", "tests/test_e1_decomposition.py", "-o", "addopts=",
                         "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    if pt.returncode != 0:
        print(pt.stdout[-4000:]); raise RuntimeError("E1 tests fail; not evaluating")
    tests_summary = next((ln for ln in reversed(pt.stdout.splitlines()) if "passed" in ln or "failed" in ln), "")
    print(f"[tests] {tests_summary}", flush=True)

    coh = C.load_cohort()
    X, Yd, SUB, SITE, WI = coh["X"], coh["Yd"], coh["SUB"], coh["SITE"], coh["WI"]
    gt_pk, iqr, CLUSTER = coh["gt_pk"], coh["iqr"], coh["cluster"]
    C.wcsv(ART / "cohort_manifest.csv", [{"row": i, "subject": SUB[i], "site": SITE[i],
                                          "window_index": int(WI[i]), "K": len(gt_pk[i]),
                                          "cluster": CLUSTER[i]} for i in range(len(X))])
    base, o2c, tcn, meta = C.load_models(dev, with_r1=True)
    (ART / "frozen_component_manifest.json").write_text(json.dumps(
        {**meta, "training_performed": False, "optimizer_constructed": False}, indent=2, default=str))
    e0 = C.source_bank(len(X))
    print(f"[P] {len(X)} windows, {coh['n_beats']} GT beats, {len(set(CLUSTER.tolist()))} clusters", flush=True)

    # ---------------- 11-12. schedules, reconstruction gate, generator reproduction gate ----------------
    SUPP, PAIRS = {}, {}
    sched_gate, gen_gate = [], []
    for name, fam, lv in ARMS:
        S, RPn = [], []
        for i in range(len(X)):
            r = np.asarray(gt_pk[i], dtype=np.int64)
            s = O3.supplied_schedule(fam, lv, REP, r, SUB[i], SITE[i], int(WI[i]))
            S.append(s)
            RPn.append(O3.retained_pairs(fam, lv, REP, r, s, SUB[i], SITE[i], int(WI[i])))
        SUPP[name] = S
        PAIRS[name] = [np.stack([pp[:, 0], pp[:, 1]], axis=1) if len(pp) else pp for pp in RPn]
        o3m = o3row("perturbation_manifest.csv", O3NAME[name])
        shift = [np.abs(np.asarray(S[i], float) - np.asarray(gt_pk[i], float)).max()
                 if len(S[i]) == len(gt_pk[i]) else np.nan for i in range(len(X))]   # the exact O3 expression
        got = {"mean_M": float(np.mean([len(s) for s in S])),
               "max_abs_shift_samples": float(np.nanmax(shift)) if np.isfinite(shift).any() else np.nan}
        q = [O3.schedule_quality(gt_pk[i], S[i]) for i in range(len(X))]
        o3q = o3row("schedule_quality_metrics.csv", O3NAME[name])
        for k in ("f1_at_50", "f1_at_150", "timing_mae_ms", "missing", "spurious", "beats_ratio_dev"):
            got[k] = C.macro([r[k] for r in q], SUB)
        exp = {"mean_M": float(o3m["mean_M"]), "max_abs_shift_samples": float(o3m["max_abs_shift_samples"]),
               **{k: float(o3q[k]) for k in ("f1_at_50", "f1_at_150", "timing_mae_ms", "missing",
                                             "spurious", "beats_ratio_dev")}}
        bad = {k: [got[k], v] for k, v in exp.items() if not agree(got[k], v, REG_TOL_SCHED)}
        fin = [abs(got[k] - v) for k, v in exp.items() if np.isfinite(got[k]) and np.isfinite(v)]
        sched_gate.append({"arm": name, "max_abs_diff": max(fin) if fin else 0.0,
                           "passed": not bad, "mismatches": json.dumps(bad), **{f"o3_{k}": v for k, v in exp.items()}})
        if bad:
            raise RuntimeError(f"SCHEDULE RECONSTRUCTION GATE FAILED (STOP): {name}: {bad}")
    print("[gate1] schedule reconstruction reproduces the frozen O3 manifests", flush=True)

    PRED, ROWS = {}, {}
    p_b = C.R2E.gen_plain(base, X, e0, C.NFE, dev)
    rows_b, _a, _b = C.R2E.score(p_b, Yd, gt_pk)
    mac_b = C.R2E.macro_rows(rows_b, SUB)
    al_b = C.aligned_rows(p_b, gt_tg := coh["gt_tg"], iqr)
    exp_b = {k: v for k, v in C.FROZEN_B.items()}
    bad = {k: [mac_b[k], v] for k, v in exp_b.items() if abs(mac_b[k] - v) > REG_TOL_GEN}
    gen_gate.append({"arm": "B", "max_abs_diff": max(abs(mac_b[k] - v) for k, v in exp_b.items()),
                     "passed": not bad, "mismatches": json.dumps(bad)})
    if bad:
        raise RuntimeError(f"GENERATOR REPRODUCTION GATE FAILED (STOP): B: {bad}")
    for name, _f, _l in ARMS:
        warps = C.build_warps(SUPP[name])
        pr, _cn = C.o2c_predict(o2c, X, warps, e0, dev)
        PRED[name] = pr
        rows, _c, _d = C.R2E.score(pr, Yd, gt_pk)
        mac = C.R2E.macro_rows(rows, SUB)
        al = C.aligned_rows(pr, gt_tg, iqr)
        alm = {t: C.macro([r[C.NAE[t]] for r in al], SUB) for t in C.ALIGNED}
        ROWS[name] = (rows, al, mac, alm)
        o3g = o3row("synthetic_generator_metrics.csv", O3NAME[name])
        exp = {"f1_excess": float(o3g["f1_excess"]), "qrs_deriv_rmse": float(o3g["qrs_deriv_rmse"]),
               "qrs_curvature_err": float(o3g["qrs_curvature_err"]),
               **{C.NAE[t]: float(o3g[C.NAE[t]]) for t in C.ALIGNED}}
        got = {"f1_excess": mac["f1_excess"], "qrs_deriv_rmse": mac["qrs_deriv_rmse"],
               "qrs_curvature_err": mac["qrs_curvature_err"], **{C.NAE[t]: alm[t] for t in C.ALIGNED}}
        bad = {k: [got[k], v] for k, v in exp.items() if abs(got[k] - v) > REG_TOL_GEN}
        gen_gate.append({"arm": name, "max_abs_diff": max(abs(got[k] - v) for k, v in exp.items()),
                         "passed": not bad, "mismatches": json.dumps(bad)})
        if bad:
            raise RuntimeError(f"GENERATOR REPRODUCTION GATE FAILED (STOP): {name}: {bad}")
        print(f"[gate2] {name:<10} reproduces the frozen O3 row (max |Δ| "
              f"{max(abs(got[k] - v) for k, v in exp.items()):.2e})", flush=True)
    (ART / "prediction_reuse_manifest.json").write_text(json.dumps(
        {"prediction_source": "DETERMINISTIC FROZEN RERUN",
         "reason": "O3 committed no per-window supplied schedule, generated waveform or detected event list; "
                   "predictions are never committed in this project, so beat-level decomposition requires a "
                   "deterministic rerun of the frozen models",
         "nfe": C.NFE, "source_seed": C.SRC_SEED, "cohort": "exact O3 primary cohort",
         "schedule_reconstruction_gate": sched_gate, "generator_reproduction_gate": gen_gate,
         "tolerances": {"schedule": REG_TOL_SCHED, "generator": REG_TOL_GEN},
         "training_performed": False}, indent=2, default=str))

    # ---------------- 13-16. synthetic identities, own-centre metrics, contrasts ----------------
    W, BEATS, COV, TOPO, PLACE = {}, {}, {}, {}, {}
    for name, _f, _l in ARMS:
        P = C.R2E.pmap(C.S0._peaks, list(PRED[name].astype(np.float64)))
        w, b = analyse_arm(PRED[name], Yd, SUPP[name], P, PAIRS[name], gt_pk, iqr)
        W[name], BEATS[name] = w, b
        COV[name] = coverage(w, gt_pk, SUPP[name])
        TOPO[name] = [E1.topology(gt_pk[i], SUPP[name][i]) for i in range(len(X))]
        PLACE[name] = [E1.placement(gt_pk[i], SUPP[name][i],
                                    [(int(bb), int(aa)) for aa, bb in PAIRS[name][i]]) for i in range(len(X))]
        adh = C.macro([O3.adherence(SUPP[name][i], P[i])["adherence_f1_at_50"] for i in range(len(X))], SUB)
        COV[name]["adherence_f1_at_50"] = adh
        print(f"[syn ] {name:<10} chain cov {COV[name]['C3_full_chain_P_S_G']:.4f} | own T6 "
              f"{C.macro([r[f'own_{T6K}'] for r in w], SUB):.4f} gt T6 {C.macro([r[f'gt_{T6K}'] for r in w], SUB):.4f} "
              f"| adh {adh:.4f}", flush=True)

    cov_rows = [{"arm": a, **{k: v for k, v in COV[a].items()}} for a in COV]
    C.wcsv(ART / "coverage_metrics.csv", cov_rows)
    C.wcsv(ART / "own_center_window_metrics.csv",
           [{"arm": a, **{k: v for k, v in r.items()}} for a in W for r in W[a]])
    C.wcsv(ART / "own_center_beat_metrics.csv",
           [{"arm": a, **r} for a in BEATS for r in BEATS[a]])          # every eligible beat, no cap
    C.wcsv(ART / "beat_identity_manifest.csv",
           [{"arm": a, "row": i, "n_supplied": int(len(SUPP[a][i])), "n_identity": int(len(PAIRS[a][i])),
             "n_no_gt_identity": int(len(SUPP[a][i]) - len(PAIRS[a][i]))}
            for a in SUPP for i in range(0, len(X), 8)])
    C.wcsv(ART / "chain_matching_metrics.csv",
           [{"arm": a, "n_supplied": COV[a]["n_supplied"], "n_identity": COV[a]["n_identity"],
             "n_chained": COV[a]["n_chained"], "n_eligible": COV[a]["n_eligible"],
             "adherence_f1_at_50": COV[a]["adherence_f1_at_50"],
             "adherence_label": "HIGH ADHERENCE" if COV[a]["adherence_f1_at_50"] >= E1.ADHERENCE_HIGH else "-"}
            for a in COV])
    C.wcsv(ART / "topology_metrics.csv",
           [{"arm": a, **{k: float(np.mean([t[k] for t in TOPO[a]])) for k in
                          ("K", "M", "count_error", "abs_count_error", "missing_events", "extra_events",
                           "missing_fraction", "spurious_fraction")},
             **{cls: int(sum(1 for t in TOPO[a] if t["topology_class"] == cls))
                for cls in (E1.T0, E1.T1, E1.T2, E1.T3)}} for a in TOPO])
    C.wcsv(ART / "placement_metrics.csv",
           [{"arm": a, **{k: float(np.nanmean([p[k] for p in PLACE[a]])) for k in
                          ("median_ae_ms", "mae_ms", "p90_ae_ms", "p95_ae_ms", "signed_mean_ms")},
             "gen_to_gt_ae_ms": C.macro([r["gen_to_gt_ae_ms"] for r in W[a]], SUB),
             "gen_to_sup_ae_ms": C.macro([r["gen_to_sup_ae_ms"] for r in W[a]], SUB)} for a in PLACE])
    C.wcsv(ART / "gt_anchored_local_metrics.csv",
           [{"arm": a, **{f"own_{k}": C.macro([r[f"own_{k}"] for r in W[a]], SUB) for k in E1.SHAPE_KEYS},
             **{f"gt_{k}": C.macro([r[f"gt_{k}"] for r in W[a]], SUB) for k in E1.SHAPE_KEYS}} for a in W])
    C.wcsv(ART / "alignment_sensitivity.csv",
           [{"arm": a, **{f"align_sens_{k}": C.macro([r[f"gt_{k}"] - r[f"own_{k}"] for r in W[a]], SUB)
                          for k in E1.SHAPE_KEYS}} for a in W])

    res, brows = {}, []
    def add(tag, a_rows, b_rows, key, note):
        r = boot(a_rows, b_rows, key, SUB, CLUSTER)
        res[tag] = r
        brows.append({"contrast": tag, "metric": key, "orientation": "positive = FIRST arm worse",
                      "note": note, **r})
    for arm, tag in (("JITTER_4", "JITTER4"), ("JITTER_8", "JITTER8"), ("MISS1", "MISS1"), ("EXTRA1", "EXTRA1")):
        for k, lab in ((f"own_{T6K}", "T6"), (f"own_{T7K}", "T7"), (f"own_{E1.NAE[E1.ALIGNED[0]]}", "T4"),
                       (f"own_{E1.NAE[E1.ALIGNED[3]]}", "T8")):
            add(f"{tag}_damage_{lab}", W[arm], W["ORACLE"], k, f"own-centre {lab} damage vs ORACLE")
    for k, lab in (("gt_local_deriv_rmse", "gt_anchored_local_deriv_rmse"),
                   ("gt_local_curvature_err", "gt_anchored_local_curvature_err")):
        add(f"{lab}_J4_vs_ORACLE", W["JITTER_4"], W["ORACLE"], k, "GT-anchored waveform-style damage vs ORACLE")
    for lab, k in (("T6", T6K), ("T7", T7K)):
        a = [{"d": r[f"gt_{k}"] - r[f"own_{k}"]} for r in W["JITTER_4"]]
        b = [{"d": r[f"gt_{k}"] - r[f"own_{k}"]} for r in W["ORACLE"]]
        add(f"{lab}_gt_minus_own_damage_J4", a, b, "d",
            "excess of GT-anchored over own-centre damage (same functional, same beats)")
    for lab, k in (("T6", T6K), ("T7", T7K)):
        add(f"excess_MISS_{lab}", W["MISS1"], W["JITTER_8"], f"own_{k}",
            "TopologyExcessDamage_MISS = Damage(MISS1,ORACLE) - Damage(JITTER8,ORACLE)")
        add(f"excess_EXTRA_{lab}", W["EXTRA1"], W["JITTER_8"], f"own_{k}",
            "TopologyExcessDamage_EXTRA = Damage(EXTRA1,ORACLE) - Damage(JITTER8,ORACLE)")
    for lab, ownk, gtk in (("T6", f"own_{T6K}", "gt_local_deriv_rmse"),
                           ("T7", f"own_{T7K}", "gt_local_curvature_err")):
        a = [{"d": r[gtk]} for r in W["JITTER_4"]]
        b = [{"d": r[gtk]} for r in W["ORACLE"]]
        gt_dmg = boot(a, b, "d", SUB, CLUSTER)
        own_dmg = res[f"JITTER4_damage_{lab}"]
        brows.append({"contrast": f"POSTHOC_own_{lab}_vs_gt_waveform_damage_J4", "metric": f"{ownk} | {gtk}",
                      "orientation": "positive = FIRST arm worse", "point": gt_dmg["point"] - own_dmg["point"],
                      "lo": "", "hi": "", "verdict": "", "damage_verdict": "",
                      "note": "SECONDARY, POST-HOC, NOT A GATE: the task wording pairs own-centre T6/T7 damage "
                              "against GT-anchored DERIVATIVE/CURVATURE damage, whereas the frozen "
                              "preregistration gate P3/P4 uses the same-functional GT-anchored T6/T7; both are "
                              "reported and only the frozen gate can move the verdict",
                      "own_damage_point": own_dmg["point"], "gt_waveform_damage_point": gt_dmg["point"],
                      "A_mean": gt_dmg["A_mean"], "B_mean": gt_dmg["B_mean"], "n_dropped_windows": 0})
    C.wcsv(ART / "paired_bootstrap.csv", brows)
    C.wcsv(ART / "synthetic_contrasts.csv",
           [{"arm": a, "schedule_mae_ms": float(np.nanmean([p["mae_ms"] for p in PLACE[a]])),
             "own_T6": C.macro([r[f"own_{T6K}"] for r in W[a]], SUB),
             "own_T7": C.macro([r[f"own_{T7K}"] for r in W[a]], SUB),
             "gt_T6": C.macro([r[f"gt_{T6K}"] for r in W[a]], SUB),
             "gt_T7": C.macro([r[f"gt_{T7K}"] for r in W[a]], SUB),
             "gt_local_deriv_rmse": C.macro([r["gt_local_deriv_rmse"] for r in W[a]], SUB),
             "own_local_deriv_rmse": C.macro([r["own_local_deriv_rmse"] for r in W[a]], SUB),
             "gt_local_curvature_err": C.macro([r["gt_local_curvature_err"] for r in W[a]], SUB),
             "own_local_curvature_err": C.macro([r["own_local_curvature_err"] for r in W[a]], SUB),
             "C3_full_chain": COV[a]["C3_full_chain_P_S_G"], "adherence_f1_at_50": COV[a]["adherence_f1_at_50"]}
            for a in ("ORACLE", "JITTER_2", "JITTER_4", "JITTER_8", "MISS1", "EXTRA1")])
    C.wcsv(ART / "topology_excess_damage.csv",
           [{"contrast": t, **{k: res[t][k] for k in ("point", "lo", "hi", "verdict", "A_mean", "B_mean")}}
            for t in ("excess_MISS_T6", "excess_MISS_T7", "excess_EXTRA_T6", "excess_EXTRA_T7")])
    pg = E1.placement_gates(res)
    tg = E1.topology_gates(res)
    cov_ok = all(COV[a]["C3_full_chain_P_S_G"] >= E1.COVERAGE_MIN for a in E1.SYNTHETIC_O2C_ARMS)
    print(f"[gates] P {pg} | C {tg} | coverage_ok {cov_ok}", flush=True)

    # ---------------- 17-18. ONLY NOW the R1 diagnostic assignment ----------------
    S_r1 = C.r1_schedules(tcn, X, dev)
    q = [O3.schedule_quality(gt_pk[i], S_r1[i]) for i in range(len(X))]
    o3q = o3row("r1_schedule_quality.csv", None, arm="R1-SCHEDULE")
    bad = {k: [C.macro([r[k] for r in q], SUB), float(o3q[k])] for k in
           ("f1_at_50", "f1_at_150", "timing_mae_ms", "missing", "spurious", "beats_ratio_dev")
           if abs(C.macro([r[k] for r in q], SUB) - float(o3q[k])) > REG_TOL_SCHED}
    if bad:
        raise RuntimeError(f"R1 SCHEDULE RECONSTRUCTION GATE FAILED (STOP): {bad}")
    warps = C.build_warps(S_r1)
    p_r1, _cn = C.o2c_predict(o2c, X, warps, e0, dev)
    rows_r1, _e, _f = C.R2E.score(p_r1, Yd, gt_pk)
    mac_r1 = C.R2E.macro_rows(rows_r1, SUB)
    al_r1 = C.aligned_rows(p_r1, gt_tg, iqr)
    o3r = o3row("r1_generator_metrics.csv", None, arm="O2C-R1-SCHEDULE")
    bad = {k: [mac_r1[k], float(o3r[k])] for k in ("f1_excess", "qrs_deriv_rmse", "qrs_curvature_err")
           if abs(mac_r1[k] - float(o3r[k])) > REG_TOL_GEN}
    if bad:
        raise RuntimeError(f"R1 GENERATOR REPRODUCTION GATE FAILED (STOP): {bad}")
    P_r1 = C.R2E.pmap(C.S0._peaks, list(p_r1.astype(np.float64)))
    pairs_r1 = []
    for i in range(len(X)):
        m, _ur, _up = E1.dp_match(gt_pk[i], S_r1[i], E1.FS, E1.TOL_IDENTITY_MS)
        pairs_r1.append(np.asarray([(j, g) for g, j in m], dtype=np.int64).reshape(-1, 2))
    W["R1-SCHEDULE"], BEATS["R1-SCHEDULE"] = analyse_arm(p_r1, Yd, S_r1, P_r1, pairs_r1, gt_pk, iqr)
    COV["R1-SCHEDULE"] = coverage(W["R1-SCHEDULE"], gt_pk, S_r1)
    COV["R1-SCHEDULE"]["adherence_f1_at_50"] = C.macro(
        [O3.adherence(S_r1[i], P_r1[i])["adherence_f1_at_50"] for i in range(len(X))], SUB)
    TOPO["R1-SCHEDULE"] = [E1.topology(gt_pk[i], S_r1[i]) for i in range(len(X))]
    PLACE["R1-SCHEDULE"] = [E1.placement(gt_pk[i], S_r1[i], [(int(g), int(j)) for j, g in pairs_r1[i]])
                            for i in range(len(X))]
    print(f"[R1  ] chain cov {COV['R1-SCHEDULE']['C3_full_chain_P_S_G']:.4f} | own T6 "
          f"{C.macro([r[f'own_{T6K}'] for r in W['R1-SCHEDULE']], SUB):.4f} | adh "
          f"{COV['R1-SCHEDULE']['adherence_f1_at_50']:.4f}", flush=True)

    for f, rows in (("coverage_metrics.csv", cov_rows), ):
        C.wcsv(ART / f, rows + [{"arm": "R1-SCHEDULE", **COV["R1-SCHEDULE"]}])
    strata = []
    cls = [t["topology_class"] for t in TOPO["R1-SCHEDULE"]]
    for c in (E1.T0, E1.T1, E1.T2, E1.T3):
        idx = [i for i in range(len(X)) if cls[i] == c]
        uw = {CLUSTER[i] for i in idx}
        per = {s: len({CLUSTER[i] for i in idx if SUB[i] == s}) for s in sorted(set(SUB.tolist()))}
        strata.append({"stratum": c, "n_rows": len(idx), "n_unique_ecg_windows": len(uw), **per,
                       "sufficient": E1.stratum_sufficient(len(uw), per),
                       "own_T6": C.macro([W["R1-SCHEDULE"][i][f"own_{T6K}"] for i in idx], SUB[idx]) if idx else np.nan,
                       "own_T7": C.macro([W["R1-SCHEDULE"][i][f"own_{T7K}"] for i in idx], SUB[idx]) if idx else np.nan,
                       "gt_T6": C.macro([W["R1-SCHEDULE"][i][f"gt_{T6K}"] for i in idx], SUB[idx]) if idx else np.nan,
                       "gt_T7": C.macro([W["R1-SCHEDULE"][i][f"gt_{T7K}"] for i in idx], SUB[idx]) if idx else np.nan,
                       "schedule_mae_ms": float(np.nanmean([PLACE["R1-SCHEDULE"][i]["mae_ms"] for i in idx])) if idx else np.nan})
    C.wcsv(ART / "r1_topology_strata.csv", strata)

    setc = [i for i in range(len(X)) if cls[i] == E1.T0]
    wrong = [i for i in range(len(X)) if cls[i] != E1.T0]
    bins = {}
    for i in range(len(X)):
        b = E1.timing_bin(PLACE["R1-SCHEDULE"][i]["mae_ms"])
        bins[i] = b
    tb_rows = []
    for b in ("A", "B", "C"):
        idx = [i for i in setc if bins[i] == b]
        uw = {CLUSTER[i] for i in idx}
        per = {s: len({CLUSTER[i] for i in idx if SUB[i] == s}) for s in sorted(set(SUB.tolist()))}
        tb_rows.append({"stratum": "R1-SET-CORRECT", "bin": b,
                        "range_ms": {"A": "<=16", "B": ">16 and <=32", "C": ">32"}[b],
                        "n_rows": len(idx), "n_unique_ecg_windows": len(uw), **per,
                        "sufficient": E1.stratum_sufficient(len(uw), per),
                        **{k: (C.macro([W["R1-SCHEDULE"][i][k] for i in idx], SUB[idx]) if idx else np.nan)
                           for k in (f"own_{T6K}", f"own_{T7K}", "gt_local_deriv_rmse", "gt_local_curvature_err",
                                     f"gt_{T6K}", f"gt_{T7K}")},
                        "f1_excess": C.macro([rows_r1[i]["f1_excess"] for i in idx], SUB[idx]) if idx else np.nan})
    C.wcsv(ART / "r1_timing_bins.csv", tb_rows)
    sm = []
    for lab, idx in (("R1-SET-CORRECT", setc), ("R1-TOPOLOGY-WRONG", wrong),
                     ("R1-UNDERCOUNT", [i for i in range(len(X)) if cls[i] == E1.T2]),
                     ("R1-OVERCOUNT", [i for i in range(len(X)) if cls[i] == E1.T3])):
        uw = {CLUSTER[i] for i in idx}
        per = {s: len({CLUSTER[i] for i in idx if SUB[i] == s}) for s in sorted(set(SUB.tolist()))}
        sm.append({"stratum": lab, "n_rows": len(idx), "n_unique_ecg_windows": len(uw), **per,
                   "sufficient": E1.stratum_sufficient(len(uw), per),
                   **{k: (C.macro([W["R1-SCHEDULE"][i][k] for i in idx], SUB[idx]) if idx else np.nan)
                      for k in (f"own_{T6K}", f"own_{T7K}", f"gt_{T6K}", f"gt_{T7K}",
                                "gt_local_deriv_rmse", "own_local_deriv_rmse")},
                   "schedule_mae_ms": float(np.nanmean([PLACE["R1-SCHEDULE"][i]["mae_ms"] for i in idx])) if idx else np.nan,
                   "f1_excess": C.macro([rows_r1[i]["f1_excess"] for i in idx], SUB[idx]) if idx else np.nan})
    C.wcsv(ART / "r1_stratum_metrics.csv", sm)

    tc = []
    for b in ("A", "B", "C"):
        a_idx = [i for i in wrong if bins[i] == b]
        b_idx = [i for i in setc if bins[i] == b]
        ua, ub = {CLUSTER[i] for i in a_idx}, {CLUSTER[i] for i in b_idx}
        pa = {s: len({CLUSTER[i] for i in a_idx if SUB[i] == s}) for s in sorted(set(SUB.tolist()))}
        pb = {s: len({CLUSTER[i] for i in b_idx if SUB[i] == s}) for s in sorted(set(SUB.tolist()))}
        ok = E1.stratum_sufficient(len(ua), pa) and E1.stratum_sufficient(len(ub), pb)
        row = {"bin": b, "n_topology_wrong": len(a_idx), "n_set_correct": len(b_idx),
               "unique_windows_wrong": len(ua), "unique_windows_correct": len(ub), "powered": ok,
               "note": "observational; within coarse matched-timing strata only; not causal"}
        for lab, k in (("T6", f"own_{T6K}"), ("T7", f"own_{T7K}")):
            row[f"wrong_{lab}"] = C.macro([W["R1-SCHEDULE"][i][k] for i in a_idx], SUB[a_idx]) if a_idx else np.nan
            row[f"correct_{lab}"] = C.macro([W["R1-SCHEDULE"][i][k] for i in b_idx], SUB[b_idx]) if b_idx else np.nan
        tc.append(row)
    C.wcsv(ART / "r1_timing_controlled_comparison.csv", tc)
    srows = []
    for st in RC.SITES:
        idx = [i for i in range(len(X)) if SITE[i] == st]
        wrong_i = [i for i in idx if cls[i] != E1.T0]
        srows.append({"site": st, "n": len(idx), "topology_error_fraction": len(wrong_i) / max(len(idx), 1),
                      "schedule_f1_at_50": C.macro([q[i]["f1_at_50"] for i in idx], SUB[idx]),
                      "schedule_mae_ms": float(np.nanmean([PLACE["R1-SCHEDULE"][i]["mae_ms"] for i in idx])),
                      **{k: C.macro([W["R1-SCHEDULE"][i][k] for i in idx], SUB[idx])
                         for k in (f"own_{T6K}", f"own_{T7K}", f"gt_{T6K}", f"gt_{T7K}")},
                      "note": "secondary; no site causality claim"})
    C.wcsv(ART / "r1_site_topology.csv", srows)

    # ---------------- 19-21. source-stability reuse, gates, verdict ----------------
    ms = {r["arm"]: r for r in rd(O3ART / "multisource_metrics.csv")}
    C.wcsv(ART / "source_stability_reuse.csv",
           [{"arm": a, "source": "frozen O3 multisource_metrics.csv (reused, not rerun)",
             "beat_count_SD": ms[k]["beat_count_SD"], "pairwise_event_F1_50": ms[k]["pairwise_event_F1_50"],
             "pointwise_waveform_SD": ms[k]["pointwise_waveform_SD"]}
            for a, k in (("ORACLE", "O2C-ORACLE"), ("JITTER_8", "JITTER_8"), ("MISS1", "MISS1"),
                         ("EXTRA1", "EXTRA1"), ("R1-SCHEDULE", "R1-SCHEDULE"))])
    C.wcsv(ART / "gates.csv",
           [{"family": "placement", "gate": k, "result": bool(v)} for k, v in pg.items()] +
           [{"family": "topology", "gate": k, "result": bool(v)} for k, v in tg.items()] +
           [{"family": "coverage", "gate": "C3>=0.80 on all synthetic arms", "result": bool(cov_ok)}])
    dec = E1.decide_e1(pg, tg, res, cov_ok, True)
    (ART / "decision.json").write_text(json.dumps(
        {**dec, "key_effects": {k: {kk: res[k][kk] for kk in ("point", "lo", "hi", "verdict")} for k in res},
         "status": "post-O3 problem-decomposition diagnostic; designed after the O3 results were known; "
                   "frozen post-hoc diagnostic criteria, not independent preregistered confirmation",
         "training_performed": False,
         "r1_identity": "target-derived diagnostic matching (monotonic one-to-one, +-150 ms); never used to "
                        "modify the supplied schedule"}, indent=2, default=float))
    (ART / "provenance.json").write_text(json.dumps(
        {"git": git_sha(ROOT), "prereg": "2d5bb54", "utc": datetime.now(timezone.utc).isoformat(),
         "test_subjects_loaded": [], "training_performed": False, "prediction_source": "DETERMINISTIC FROZEN RERUN",
         "n_windows": int(len(X)), "n_gt_beats": coh["n_beats"], "n_clusters": int(len(set(CLUSTER.tolist()))),
         "nfe": C.NFE, "source_seed": C.SRC_SEED, "bootstrap": {"n": E1.BOOT_N, "seed": E1.BOOT_SEED},
         "own_center_support": [E1.WIN_LO, E1.WIN_HI], "identity_tol_ms": E1.TOL_IDENTITY_MS,
         "chain_tol_ms": E1.TOL_CHAIN_MS, "tests": tests_summary, "o1_train_iqr": iqr,
         "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
         "gpu": torch.cuda.get_device_name(0), "wall_s": time.perf_counter() - t_all,
         "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20)}, indent=2, default=str))
    print(f"\n[VERDICT] {dec['verdict']}", flush=True)
    print(f"[done] E1 in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
