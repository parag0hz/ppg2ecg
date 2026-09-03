"""O1 evaluation — the ECG component-wise conditional extractability map.

Loads the frozen probes, scores validation (an0/k2s) TRUE / B0 / B1 / B2 / SS-SHUFFLE / XS-SHUFFLE, freezes the
clean component classification, then runs the secondary analyses (Q1 corruption transfer, site map, natural
quality) and the generator-utilization crosswalk built from the FROZEN Q1 artifacts. No probe is trained here and
no generator is run.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import q1_corruption as Q
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o1_component_extractability"
FIG = ART / "figures"
Q1ART = ROOT / "artifacts/q1_conditional_support"
VAL = C.VAL
WORKERS = 12
CORRUPTIONS = ("LP_1.25Hz", "SNR_0dB", "DROP_2.0s", "SHUFFLED", "NULL")


def wcsv(p, rows):
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)


def _ppg_rhythm(x):
    """B2 features: rhythm/site only. No amplitude, no morphology, no waveform sample."""
    pk = S1.dsp_ppg_peaks(np.asarray(x, dtype=np.float64), OT.FS)
    n = len(pk)
    if n >= 2:
        ppi = np.diff(np.asarray(pk, dtype=np.float64)) / OT.FS * 1000.0
        med = float(np.median(ppi))
        iqr = float(np.percentile(ppi, 75) - np.percentile(ppi, 25)) if len(ppi) >= 2 else 0.0
        return [float(n), med, iqr, 1.0]
    return [float(n), 0.0, 0.0, 0.0]


def rhythm_features(X, SITE, cache: Path | None = None):
    if cache is not None and cache.exists():
        F = np.load(cache)["F"]
    else:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            F = np.asarray(list(ex.map(_ppg_rhythm, list(X.astype(np.float64)), chunksize=64)), dtype=np.float64)
        if cache is not None:
            np.savez_compressed(cache, F=F)
    onehot = np.stack([(np.asarray(SITE) == s).astype(np.float64) for s in C.SITES], axis=1)
    return np.concatenate([F, onehot], axis=1)


def ridge_fit(Xf, y, alpha=OT.RIDGE_ALPHA):
    mu, sd = Xf.mean(0), Xf.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = np.concatenate([(Xf - mu) / sd, np.ones((len(Xf), 1))], axis=1)
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    A[-1, -1] -= alpha                                     # do not penalise the intercept
    w = np.linalg.solve(A, Z.T @ y)
    return {"mu": mu, "sd": sd, "w": w, "alpha": alpha}


def ridge_pred(m, Xf):
    Z = np.concatenate([(Xf - m["mu"]) / m["sd"], np.ones((len(Xf), 1))], axis=1)
    return Z @ m["w"]


def cluster_bootstrap(d, subjects, clusters, n_boot=OT.BOOT_N, seed=OT.BOOT_SEED):
    """Equal-subject-weight bootstrap over UNDERLYING ECG WINDOWS; all rows of a window move together."""
    d, subjects, clusters = np.asarray(d, float), np.asarray(subjects), np.asarray(clusters)
    ok = np.isfinite(d)
    d, subjects, clusters = d[ok], subjects[ok], clusters[ok]
    subs = sorted(set(subjects.tolist()))
    per = {}
    for s in subs:
        m = subjects == s
        cl = clusters[m]
        uniq, inv = np.unique(cl, return_inverse=True)
        groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        per[s] = (d[m], groups)
    point = float(np.mean([vals.mean() for vals, _ in per.values()]))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        acc = []
        for s in subs:
            vals, groups = per[s]
            pick = rng.integers(0, len(groups), len(groups))
            acc.append(float(np.mean(np.concatenate([vals[groups[i]] for i in pick]))))
        draws[b] = float(np.mean(acc))
    lo, hi = (float(v) for v in np.nanpercentile(draws, [2.5, 97.5]))
    return {"point": point, "lo": lo, "hi": hi, "n": int(d.size), "n_boot": int(n_boot), "seed": int(seed),
            "verdict": "improves" if lo > 0 else ("worsens" if hi < 0 else "unresolved")}


def metrics_row(pred_z, y_z, iqr):
    e = np.abs(pred_z - y_z)
    ok = np.isfinite(e)
    out = {"n": int(ok.sum()), "nMAE": float(np.mean(e[ok])), "MAE_units": float(np.mean(e[ok]) * iqr),
           "median_AE": float(np.median(e[ok])), "RMSE_units": float(np.sqrt(np.mean((pred_z[ok] - y_z[ok]) ** 2)) * iqr),
           "nRMSE": float(np.sqrt(np.mean((pred_z[ok] - y_z[ok]) ** 2)))}
    if ok.sum() > 2 and np.std(pred_z[ok]) > 1e-12:
        out["pearson"] = float(pearsonr(pred_z[ok], y_z[ok]).statistic)
        out["spearman"] = float(spearmanr(pred_z[ok], y_z[ok]).statistic)
        ss = float(np.sum((y_z[ok] - y_z[ok].mean()) ** 2))
        out["r2"] = float(1.0 - np.sum((pred_z[ok] - y_z[ok]) ** 2) / max(ss, 1e-30))
    else:
        out |= {"pearson": np.nan, "spearman": np.nan, "r2": np.nan}
    return out


@torch.no_grad()
def predict(net, X, dev, batch=512):
    out = []
    for i in range(0, len(X), batch):
        out.append(net(torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(dev).unsqueeze(1)).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    t_all = time.perf_counter()
    dev = torch.device("cuda")
    git = git_sha(ROOT)
    pt = subprocess.run([sys.executable, "-m", "pytest", "tests/test_o1_component_extractability.py", "-o", "addopts=",
                         "-q", "-rs", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    if pt.returncode != 0:
        print(pt.stdout[-4000:]); raise RuntimeError("O1 tests fail; not evaluating")
    tests_summary = next((ln for ln in reversed(pt.stdout.splitlines()) if "passed" in ln or "failed" in ln), "")
    print(f"[tests] {tests_summary}", flush=True)

    # ---------------- data ----------------
    z = np.load(ART / "_cache_targets.npz"); Tt = z["targets"]
    key = {(str(s), int(w)): i for i, (s, w) in enumerate(zip(z["subjects"], z["window_index"]))}
    scaling = json.loads((ART / "target_scaling.json").read_text())["targets"]
    rows = [r for r in csv.DictReader(open(ART / "cohort_manifest.csv"))]
    val_rows = [r for r in rows if r["role"] == "validation"]
    Xv, Yv, SUB, SITE, WI = [], [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs = d["x"]
        rr = [r for r in val_rows if r["subject"] == s]
        for site in C.SITES:
            sel = [r for r in rr if r["site"] == site]
            idx = np.array([int(r["array_pos"]) for r in sel])
            Xv.append(Xs[idx].astype(np.float32))
            Yv.append(np.stack([Tt[key[(s, int(r["window_index"]))]] for r in sel]))
            SUB.append(np.full(len(sel), s)); SITE.append(np.full(len(sel), site))
            WI.append(np.array([int(r["window_index"]) for r in sel], dtype=np.int64))
    Xv, Yv, SUB, SITE, WI = (np.concatenate(v) for v in (Xv, Yv, SUB, SITE, WI))
    CLUSTER = np.array([f"{a}|{b}" for a, b in zip(SUB, WI)])
    print(f"[val] {len(Xv)} rows | {len(set(CLUSTER))} unique ECG windows | subjects {sorted(set(SUB))}", flush=True)

    # training-side rows (for the baselines only; the probes never saw validation)
    Xtr, Ytr, SUBtr, SITEtr = [], [], [], []
    for s in C.internal_dev_split()["probe_train"]:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs, SITEs, WIs = d["x"], np.asarray(d["site"]).astype(str), d["window_index"].astype(np.int64)
        pos = C.cohort_positions(s, SITEs, WIs, C.n_per_for(s))
        for site in C.SITES:
            idx = pos[site]
            Xtr.append(Xs[idx].astype(np.float32)); Ytr.append(np.stack([Tt[key[(s, int(WIs[p]))]] for p in idx]))
            SUBtr.append(np.full(len(idx), s)); SITEtr.append(np.full(len(idx), site))
    Xtr, Ytr, SITEtr = np.concatenate(Xtr), np.concatenate(Ytr), np.concatenate(SITEtr)

    # ---------------- shuffles ----------------
    ss = OT.same_subject_shuffle(SUB, SITE, WI); OT.assert_derangement(ss)
    xs = OT.cross_subject_shuffle(SUB, SITE, WI); OT.assert_cross_subject(xs, SUB, SITE)
    wcsv(ART / "same_subject_shuffle_manifest.csv", [{"row": i, "subject": SUB[i], "site": SITE[i],
         "window_index": int(WI[i]), "partner_row": int(ss[i]), "partner_window_index": int(WI[ss[i]])} for i in range(len(Xv))])
    wcsv(ART / "cross_subject_shuffle_manifest.csv", [{"row": i, "subject": SUB[i], "site": SITE[i],
         "window_index": int(WI[i]), "partner_row": int(xs[i]), "partner_subject": SUB[xs[i]],
         "partner_site": SITE[xs[i]]} for i in range(len(Xv))])

    # ---------------- baselines ----------------
    Ftr = rhythm_features(Xtr, SITEtr, ART / "_cache_rhythm_train.npz")
    Fv = rhythm_features(Xv, SITE, ART / "_cache_rhythm_val.npz")
    base_rows, B = [], {}
    for j, t in enumerate(OT.TARGETS):
        c, iqr = scaling[t]["center_train_median"], scaling[t]["scale_train_IQR"]
        ytr_z, yv_z = (Ytr[:, j] - c) / iqr, (Yv[:, j] - c) / iqr
        ok_tr = np.isfinite(ytr_z)
        b0 = np.full(len(yv_z), float(np.median(ytr_z[ok_tr])))
        site_med = {s: float(np.median(ytr_z[ok_tr & (SITEtr == s)])) for s in C.SITES}
        b1 = np.array([site_med[s] for s in SITE])
        m = ridge_fit(Ftr[ok_tr], ytr_z[ok_tr])
        b2 = ridge_pred(m, Fv)
        B[t] = {"B0": b0, "B1": b1, "B2": b2, "y_z": yv_z, "iqr": iqr}
        for name, pred in (("B0", b0), ("B1", b1), ("B2", b2)):
            base_rows.append({"target": t, "id": OT.TARGET_IDS[t], "arm": name, **metrics_row(pred, yv_z, iqr)})
    wcsv(ART / "baseline_metrics.csv", base_rows)
    for t in OT.TARGETS:
        r = {x["arm"]: x for x in base_rows if x["target"] == t}
        print(f"[base] {OT.TARGET_IDS[t]} {t:<30} B0 {r['B0']['nMAE']:.4f} B1 {r['B1']['nMAE']:.4f} B2 {r['B2']['nMAE']:.4f}", flush=True)

    # ---------------- probes: TRUE / SS / XS ----------------
    man = {(r["target"], int(r["seed"])): r for r in csv.DictReader(open(ART / "probe_training_manifest.csv"))}
    probe_rows, AE = [], {}
    for t in OT.TARGETS:
        for seed in OT.SEEDS:
            ck = torch.load(ROOT / f"outputs/o1_{t}_seed{seed}/checkpoint_best.pt", map_location="cpu", weights_only=False)
            assert ck["target"] == t and ck["seed"] == seed
            net = OT.build_probe(seed).to(dev).eval(); net.load_state_dict(ck["state_dict"]); net.requires_grad_(False)
            y_z, iqr = B[t]["y_z"], B[t]["iqr"]
            p_true = predict(net, Xv, dev)
            p_ss = predict(net, Xv[ss], dev)
            p_xs = predict(net, Xv[xs], dev)
            for arm, p in (("TRUE", p_true), ("SS-SHUFFLE", p_ss), ("XS-SHUFFLE", p_xs)):
                probe_rows.append({"target": t, "id": OT.TARGET_IDS[t], "seed": seed, "arm": arm, **metrics_row(p, y_z, iqr)})
            AE[(t, seed)] = {"TRUE": np.abs(p_true - y_z), "SS": np.abs(p_ss - y_z), "XS": np.abs(p_xs - y_z),
                             "pred": p_true}
            del net
        r = [x for x in probe_rows if x["target"] == t and x["arm"] == "TRUE"]
        print(f"[probe] {OT.TARGET_IDS[t]} {t:<30} TRUE nMAE {np.mean([x['nMAE'] for x in r]):.4f} "
              f"rho {np.median([x['spearman'] for x in r]):+.3f}", flush=True)
    wcsv(ART / "probe_metrics.csv", probe_rows)
    wcsv(ART / "shuffle_metrics.csv", [r for r in probe_rows if r["arm"] != "TRUE"])

    # ---------------- skill, bootstrap, classification ----------------
    skill_rows, boot_rows, cls_rows, classes = [], [], [], {}
    for t in OT.TARGETS:
        iqr = B[t]["iqr"]
        mae_true = float(np.mean([x["nMAE"] for x in probe_rows if x["target"] == t and x["arm"] == "TRUE"]))
        mae_ss = float(np.mean([x["nMAE"] for x in probe_rows if x["target"] == t and x["arm"] == "SS-SHUFFLE"]))
        mae_xs = float(np.mean([x["nMAE"] for x in probe_rows if x["target"] == t and x["arm"] == "XS-SHUFFLE"]))
        mae_b0 = next(x["nMAE"] for x in base_rows if x["target"] == t and x["arm"] == "B0")
        mae_b1 = next(x["nMAE"] for x in base_rows if x["target"] == t and x["arm"] == "B1")
        mae_b2 = next(x["nMAE"] for x in base_rows if x["target"] == t and x["arm"] == "B2")
        d_ss = np.mean([AE[(t, s)]["SS"] - AE[(t, s)]["TRUE"] for s in OT.SEEDS], axis=0)
        d_xs = np.mean([AE[(t, s)]["XS"] - AE[(t, s)]["SS"] for s in OT.SEEDS], axis=0)
        d_b2 = np.mean([np.abs(B[t]["B2"] - B[t]["y_z"]) - AE[(t, s)]["TRUE"] for s in OT.SEEDS], axis=0)
        r_ss = cluster_bootstrap(d_ss, SUB, CLUSTER); r_xs = cluster_bootstrap(d_xs, SUB, CLUSTER)
        r_b2 = cluster_bootstrap(d_b2, SUB, CLUSTER)
        for nm, r in (("TRUE_vs_SS_SHUFFLE", r_ss), ("XS_minus_SS", r_xs), ("TRUE_vs_B2", r_b2)):
            boot_rows.append({"target": t, "id": OT.TARGET_IDS[t], "contrast": nm,
                              "positive_means": {"TRUE_vs_SS_SHUFFLE": "TRUE better than same-subject shuffle",
                                                 "XS_minus_SS": "cross-subject shuffle worse than same-subject shuffle",
                                                 "TRUE_vs_B2": "TRUE better than the rhythm/site baseline"}[nm],
                              "unit": "normalised AE (standardised target)", **r})
        rho_seeds = [x["spearman"] for x in probe_rows if x["target"] == t and x["arm"] == "TRUE"]
        beats_b0_all = all(x["nMAE"] < mae_b0 for x in probe_rows if x["target"] == t and x["arm"] == "TRUE")
        cls = OT.classify_component(OT.skill(mae_true, mae_b2), r_ss["lo"], float(np.median(rho_seeds)),
                                    beats_b0_all, mae_true < min(mae_b0, mae_b1), mae_true < mae_b2)
        classes[t] = cls["class"]
        skill_rows.append({"target": t, "id": OT.TARGET_IDS[t], "nMAE_TRUE": mae_true, "nMAE_SS": mae_ss,
                           "nMAE_XS": mae_xs, "nMAE_B0": mae_b0, "nMAE_B1": mae_b1, "nMAE_B2": mae_b2,
                           "Skill_R": OT.skill(mae_true, mae_b2), "Skill_W": OT.skill(mae_true, mae_ss),
                           "spearman_median": float(np.median(rho_seeds)), "spearman_seeds": ";".join(f"{v:+.4f}" for v in rho_seeds),
                           "MAE_units_TRUE": mae_true * iqr, "train_IQR": iqr, "unit": OT.UNITS[t]})
        cls_rows.append({"target": t, "id": OT.TARGET_IDS[t], **cls,
                         "ss_effect": r_ss["point"], "ss_lo": r_ss["lo"], "ss_hi": r_ss["hi"], "ss_verdict": r_ss["verdict"],
                         "xs_minus_ss": r_xs["point"], "xs_lo": r_xs["lo"], "xs_hi": r_xs["hi"], "xs_verdict": r_xs["verdict"],
                         "nMAE_TRUE": mae_true, "nMAE_B2": mae_b2, "nMAE_SS": mae_ss})
        print(f"[class] {OT.TARGET_IDS[t]} {t:<30} SkillR {cls['skill_r']:+.3f} SkillW {OT.skill(mae_true, mae_ss):+.3f} "
              f"rho {cls['rho_median']:+.3f} SS[{r_ss['lo']:+.4f},{r_ss['hi']:+.4f}] -> {cls['class']}", flush=True)
    wcsv(ART / "extractability_skill.csv", skill_rows)
    wcsv(ART / "component_classification.csv", cls_rows)
    wcsv(ART / "bootstrap_results.csv", boot_rows)

    # positive control + verdict (CLEAN map frozen here)
    pc = []
    for t in OT.POSITIVE_CONTROLS:
        r = [x for x in probe_rows if x["target"] == t and x["arm"] == "TRUE"]
        mae_b0 = next(x["nMAE"] for x in base_rows if x["target"] == t and x["arm"] == "B0")
        pc.append({"target": t, "rho_median": float(np.median([x["spearman"] for x in r])),
                   "nMAE_TRUE": float(np.mean([x["nMAE"] for x in r])), "nMAE_B0": mae_b0,
                   "passes": bool(float(np.median([x["spearman"] for x in r])) >= 0.70 and float(np.mean([x["nMAE"] for x in r])) < mae_b0)})
    pc_ok = any(p["passes"] for p in pc)
    var = {r["target"]: r for r in csv.DictReader(open(ART / "target_variability.csv"))}
    morph_ok = {t: float(var[t]["within_subject_variance"]) > 0 for t in OT.TARGETS}
    dec = OT.decide_o1(classes, pc_ok, tuple(OT.TARGETS), morph_ok)
    decision = {"verdict": dec["verdict"], "detail": dec, "classes": classes, "positive_control": pc,
                "positive_control_ok": pc_ok, "prereg": "b972dea", "terminology": "operational conditional extractability",
                "status": "problem-discovery / diagnostic; not independent confirmation",
                "frozen_before_secondary_analyses": True}
    (ART / "decision.json").write_text(json.dumps(decision, indent=2, default=float))
    print(f"\n[VERDICT] {dec['verdict']} | positive control {pc_ok} | high {dec['high']} | low {dec['low']}\n", flush=True)

    # ---------------- secondary: corruption transfer ----------------
    partner_q1 = None
    corr_rows = []
    for cond in CORRUPTIONS:
        if cond == Q.SHUFFLED:
            if partner_q1 is None:
                from ppg2ecg.flow import rhythm_transfer as RT
                partner_q1 = RT.shuffle_partner(SUB, SITE, WI, salt=Q.SHUFFLE_SALT)
            Xc = Q.corrupt_block(Xv, cond, SUB, SITE, WI, partner=partner_q1)
        else:
            Xc = Q.corrupt_block(Xv, cond, SUB, SITE, WI)
        for t in OT.TARGETS:
            y_z, iqr = B[t]["y_z"], B[t]["iqr"]
            ms = []
            for seed in OT.SEEDS:
                ck = torch.load(ROOT / f"outputs/o1_{t}_seed{seed}/checkpoint_best.pt", map_location="cpu", weights_only=False)
                net = OT.build_probe(seed).to(dev).eval(); net.load_state_dict(ck["state_dict"]); net.requires_grad_(False)
                ms.append(metrics_row(predict(net, Xc, dev), y_z, iqr))
                del net
            corr_rows.append({"target": t, "id": OT.TARGET_IDS[t], "condition": cond,
                              "nMAE": float(np.mean([m["nMAE"] for m in ms])),
                              "spearman": float(np.median([m["spearman"] for m in ms])),
                              "n_seeds": len(ms)})
        print(f"[corrupt] {cond:<10} done", flush=True)
    wcsv(ART / "corruption_transfer.csv", corr_rows)

    # ---------------- secondary: site map ----------------
    site_rows = []
    for t in OT.TARGETS:
        y_z, iqr = B[t]["y_z"], B[t]["iqr"]
        for site in C.SITES:
            m = SITE == site
            tr = float(np.mean([AE[(t, s)]["TRUE"][m].mean() for s in OT.SEEDS]))
            sh = float(np.mean([AE[(t, s)]["SS"][m].mean() for s in OT.SEEDS]))
            rho = float(np.median([spearmanr(AE[(t, s)]["pred"][m], y_z[m]).statistic for s in OT.SEEDS]))
            site_rows.append({"target": t, "id": OT.TARGET_IDS[t], "site": site, "n": int(m.sum()),
                              "nMAE_TRUE": tr, "nMAE_SS": sh, "true_minus_ss": sh - tr,
                              "Skill_W": OT.skill(tr, sh), "spearman": rho})
    wcsv(ART / "site_extractability.csv", site_rows)

    # ---------------- secondary: natural quality ----------------
    def _q(x):
        v = np.asarray(x, dtype=np.float64)
        pk = S1.dsp_ppg_peaks(v, OT.FS)
        return [Q.periodicity_score(v), Q.pulse_template_consistency(v, pk)]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        QS = np.asarray(list(ex.map(_q, list(Xv.astype(np.float64)), chunksize=64)), dtype=np.float64)
    nq_rows = []
    for qi, qname in enumerate(("periodicity_score", "pulse_template_consistency")):
        qof = np.full(len(Xv), -1, dtype=int)
        for sub in VAL:
            for site in C.SITES:
                m = np.flatnonzero((SUB == sub) & (SITE == site))
                v = QS[m, qi]
                fin = np.isfinite(v)
                if fin.sum() < 8:
                    continue
                edges = np.percentile(v[fin], [25, 50, 75])
                qof[m[fin]] = np.searchsorted(edges, v[fin], side="right")
        for t in OT.TARGETS:
            for q in range(4):
                sel = np.flatnonzero(qof == q)
                if sel.size == 0:
                    continue
                nq_rows.append({"score": qname, "quartile": q + 1, "n": int(sel.size), "target": t, "id": OT.TARGET_IDS[t],
                                "score_mean": float(np.nanmean(QS[sel, qi])),
                                "nMAE_TRUE": float(np.mean([AE[(t, s)]["TRUE"][sel].mean() for s in OT.SEEDS])),
                                "nMAE_SS": float(np.mean([AE[(t, s)]["SS"][sel].mean() for s in OT.SEEDS]))})
    wcsv(ART / "natural_quality_extractability.csv", nq_rows)

    # ---------------- generator-utilization crosswalk (frozen Q1 artifacts) ----------------
    from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
    q1 = defaultdict(list)
    for r in csv.DictReader(open(Q1ART / "generator_fidelity_metrics.csv")):
        q1[r["condition"]].append(r)
    q1cl, q1sh = q1["CLEAN"], q1["SHUFFLED"]
    q1sub = np.array([r["subject"] for r in q1cl])
    CROSSWALK = {"event_timing": ("f1_excess", "higher_better"), "beat_count": ("beats_ratio_dev", "lower_better"),
                 "median_RR_ms": ("beats_ratio_dev", "lower_better"), "RR_IQR_ms": (None, None),
                 "median_QRS_p2p": ("p2p_dev", "lower_better"), "median_QRS_energy": ("qrs_e_dev", "lower_better"),
                 "median_QRS_max_abs_derivative": ("qrs_deriv_rmse", "lower_better"),
                 "median_QRS_curvature_energy": ("qrs_curvature_err", "lower_better"),
                 "median_QRS_width_ms": (None, None), "ECG_HF_fraction": ("hf_err", "lower_better")}
    cw_rows, util = [], {}
    for comp, (metric, orient) in CROSSWALK.items():
        if metric is None:
            cw_rows.append({"component": comp, "generator_metric": "N/A", "note": "no exact frozen per-window metric exists"})
            util[comp] = None
            continue
        a = np.array([float(r[metric]) for r in q1sh], float)     # SHUFFLED
        b = np.array([float(r[metric]) for r in q1cl], float)     # CLEAN
        res = paired_subject_bootstrap(a, b, q1sub, orient, n_boot=OT.BOOT_N, seed=OT.BOOT_SEED)
        util[comp] = res
        cw_rows.append({"component": comp, "id": OT.TARGET_IDS.get(comp, "R1"), "generator_metric": metric,
                        "orientation": orient, "clean": float(np.nanmean(b)), "shuffled": float(np.nanmean(a)),
                        "utilization_effect": res["point"], "lo": res["lo"], "hi": res["hi"], "verdict": res["verdict"],
                        "positive_means": "correct PPG improves this generated component",
                        "population": "Q1 frozen 2,048-window development cohort (arm B, NFE 4, seed 0)"})
    wcsv(ART / "generator_utilization_crosswalk.csv", cw_rows)

    quad_rows = []
    for t in OT.TARGETS:
        u = util.get(t)
        extract = classes[t] in (OT.CLASS_A, OT.CLASS_B)
        if u is None:
            quad, note = "N/A", "no matched frozen generator metric"
        else:
            sens = u["verdict"] == "improves"
            quad = ("Q-A EXTRACTABLE + GENERATOR-SENSITIVE" if extract and sens else
                    "Q-B EXTRACTABLE + GENERATOR-INSENSITIVE (CANDIDATE UNDERUTILIZATION)" if extract and not sens else
                    "Q-C WEAKLY EXTRACTABLE + GENERATOR-SENSITIVE" if (not extract) and sens else
                    "Q-D WEAKLY EXTRACTABLE + GENERATOR-INSENSITIVE")
            note = ""
        quad_rows.append({"component": t, "id": OT.TARGET_IDS[t], "class": classes[t],
                          "extractable": bool(extract), "generator_metric": CROSSWALK[t][0] or "N/A",
                          "utilization_effect": (u["point"] if u else np.nan), "utilization_lo": (u["lo"] if u else np.nan),
                          "utilization_hi": (u["hi"] if u else np.nan), "utilization_verdict": (u["verdict"] if u else "n/a"),
                          "quadrant": quad, "note": note})
    wcsv(ART / "extractability_utilization_map.csv", quad_rows)

    prov = {"git": git, "prereg": "b972dea", "utc": datetime.now(timezone.utc).isoformat(),
            "test_subjects_loaded": [], "validation_subjects": list(VAL), "n_val_rows": int(len(Xv)),
            "n_val_unique_ecg_windows": int(len(set(CLUSTER))), "tests_ran": {"exit": pt.returncode, "summary": tests_summary},
            "bootstrap": {"unit": "underlying ECG window (subject|window_index)", "n_boot": OT.BOOT_N, "seed": OT.BOOT_SEED,
                          "note": "within-cohort uncertainty, NOT population generalisation"},
            "shuffle_salts": {"ss": OT.SS_SALT, "xs": OT.XS_SALT},
            "corruption_module": "ppg2ecg.evaluation.q1_corruption (frozen, reused byte-identically)",
            "crosswalk_population": "Q1 frozen 2,048-window cohort (different from the O1 8,192 validation cohort)",
            "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
            "gpu": torch.cuda.get_device_name(0), "wall_s": time.perf_counter() - t_all,
            "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20)}
    (ART / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"[done] O1 evaluation in {(time.perf_counter()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
