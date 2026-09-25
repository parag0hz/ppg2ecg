"""M2 — cost-matched width-vs-depth action prediction (docs/M2_ACTION_VALUE_PREREGISTRATION.md, frozen at 51e9343).

For every 4/8/4 partition p (A pilot at S, W width at S, D depth at 2S; disjoint seed IDs) and condition (model, S):
  L_W = |median(W_S) - y|, L_D = |median(D_2S) - y|, dQ = L_D - L_W (> 0: width better); pilot A only chooses the action.
Every metric is computed inside a partition and then averaged over partitions (never average first).

Stages:
  tables-val   validation rows (partition x condition x window), cached in outputs/m2_action_value/
  fit          validation-only fitting of B1, B2, P1, P2, P3, gate, flag rates -> artifacts/m2_action_value/frozen_policy.json
  tables-test  test rows (only after frozen_policy.json exists)
  test         single test evaluation + 5,000-replicate patient bootstrap -> metrics, verdict
  latency      measured latency of the width / depth actions (content-independent)
  figure       main figure
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/m2_run.py <stage>
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m1_run as M1  # noqa: E402  (frozen M1 helpers: bank loading + snapping, weighted ranks; read-only use)

OUT = ROOT / "outputs/m2_action_value"
ART = ROOT / "artifacts/m2_action_value"
PREREG = "docs/M2_ACTION_VALUE_PREREGISTRATION.md (51e9343)"
SEED, NPART, NBOOT, NBOOT_GATE, RBOOT, NWORK = 20260925, 32, 5000, 2000, 64, 8
ARMS = ("I", "D", "C")
NAME = {"I": "iMF", "D": "CD", "C": "PENGUIN"}
BASE_S = (1, 2, 4)
COND = [(a, S) for a in ARMS for S in BASE_S]          # frozen order iMF -> CD -> PENGUIN, S 1 -> 2 -> 4
ARM_OF_COND = np.array([ARMS.index(a) for a, _ in COND])
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
QGRID = tuple(np.round(np.arange(0.05, 0.951, 0.05), 2))
DELTA_MAE = 0.10
HR_RANGE = (30.0, 200.0)
OUTLIER_REL = 0.25
FULL = ["median", "MAD", "IQR", "SD", "range", "meanPair", "medPair", "bootMedInstab", "splitHalfInstab", "logS"]
LOG1P_FULL = list(range(1, 9))
P1_COLS = [3, 9]                                          # log1p SD, log S
PI4, PJ4 = np.triu_indices(4, 1)
N_TEST_PATIENTS = 1156


def sha(path):
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def partitions():
    m = json.loads((ART / "partition_manifest.json").read_text())
    P = [(np.array(p["A"]), np.array(p["W"]), np.array(p["D"])) for p in m["partitions"]]
    assert len(P) == NPART
    for A, W, D in P:
        assert len(A) == 4 and len(W) == 8 and len(D) == 4
        assert not (set(A) & set(W)) and not (set(A) & set(D)) and not (set(W) & set(D)) and set(A) | set(W) | set(D) == set(range(16))
    return P


# ============================================================================ rows
def _boot_instab(args):
    a, ci, pi, win = args
    out = np.empty(a.shape[1])
    for j in range(a.shape[1]):
        rng = np.random.default_rng([SEED, ci, int(win[j]), pi])
        idx = rng.integers(0, 4, size=(RBOOT, 4))
        out[j] = np.std(np.median(a[:, j][idx], axis=1), ddof=1)
    return out


def pilot_features(a, S):
    """a: (4, n) pilot HRs (stored order) -> (n, 10) raw features (bootMedInstab filled by caller)."""
    n = a.shape[1]
    med = np.median(a, 0)
    F = np.empty((n, 10))
    F[:, 0] = med
    F[:, 1] = np.median(np.abs(a - med), 0)
    F[:, 2] = np.percentile(a, 75, axis=0) - np.percentile(a, 25, axis=0)
    F[:, 3] = np.std(a, 0, ddof=1)
    F[:, 4] = a.max(0) - a.min(0)
    d = np.abs(a[PI4] - a[PJ4])
    F[:, 5] = d.mean(0)
    F[:, 6] = np.median(d, 0)
    F[:, 7] = np.nan
    F[:, 8] = np.abs(np.median(a[:2], 0) - np.median(a[2:], 0))
    F[:, 9] = np.log(S)
    return F


def action_estimate(Z, c_val, min_finite):
    """median of the finite draws (Z: (k, n)); c_val if fewer than min_finite finite draws."""
    nf = np.isfinite(Z).sum(0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(Z, 0)
    return np.where(nf >= min_finite, med, c_val), nf


def build_rows(banks, ref, pid, c_val, parts, n_workers=14):
    """rows for every (partition, condition, deployable window). banks: {(arm, S): (16, n)} incl. 2S."""
    rows = {k: [] for k in ("F", "eW", "eD", "LW", "LD", "LW50", "LD50", "nW", "nD", "flag_range", "flag_outlier", "cond", "win", "part", "pid", "y")}
    ex = ProcessPoolExecutor(n_workers) if n_workers > 1 else None
    for ci, (a, S) in enumerate(COND):
        YS, Y2S = banks[(a, S)], banks[(a, 2 * S)]
        jobs = []
        for p, (A, W, D) in enumerate(parts):
            pil = YS[A]
            inc = np.isfinite(ref) & np.isfinite(pil).all(0)
            idx = np.flatnonzero(inc)
            args = (pil[:, idx], ci, p, idx)
            jobs.append((p, idx, ex.submit(_boot_instab, args) if ex else _boot_instab(args)))
        for p, idx, fut in jobs:
            A, W, D = parts[p]
            pil, y = YS[A][:, idx], ref[idx]
            eW, nW = action_estimate(YS[W][:, idx], c_val, 1)
            eD, nD = action_estimate(Y2S[D][:, idx], c_val, 1)
            eW50, _ = action_estimate(YS[W][:, idx], c_val, 4)
            eD50, _ = action_estimate(Y2S[D][:, idx], c_val, 2)
            F = pilot_features(pil, S)
            F[:, 7] = fut.result() if ex else fut
            med = np.median(pil, 0)
            for k, v in (("F", F), ("eW", eW), ("eD", eD), ("LW", np.abs(eW - y)), ("LD", np.abs(eD - y)), ("LW50", np.abs(eW50 - y)), ("LD50", np.abs(eD50 - y)),
                         ("nW", nW), ("nD", nD), ("flag_range", ((pil < HR_RANGE[0]) | (pil > HR_RANGE[1])).any(0)),
                         ("flag_outlier", (np.abs(pil - med) > OUTLIER_REL * med).any(0)), ("cond", np.full(len(idx), ci)),
                         ("win", idx), ("part", np.full(len(idx), p)), ("pid", pid[idx]), ("y", y)):
                rows[k].append(v)
    if ex:
        ex.shutdown()
    d = {k: np.concatenate(v) for k, v in rows.items()}
    d["dQ"] = d["LD"] - d["LW"]
    d["dQ50"] = d["LD50"] - d["LW50"]
    return d


def transform_full(F):
    X = np.array(F, np.float64, copy=True)
    X[:, LOG1P_FULL] = np.log1p(np.clip(X[:, LOG1P_FULL], 0, None))
    return X


def load_banks(split):
    banks, ref, pid = M1.load_split(split)                 # (16, n) snapped to 1e-3, seeds 0-15
    return banks, ref, pid


def c_val_constant():
    return float(np.median(M1.snap(np.load(ROOT / "outputs/m1_pilot_gain_prediction/val_ref.npz")["ref_hr"])))


def table(split):
    cache = OUT / f"rows_{split}.npz"
    prov = {"partition_manifest_sha256": sha(ART / "partition_manifest.json"), "snap": M1.SNAP}
    if split == "test":
        assert (ART / "frozen_policy.json").exists(), "frozen policy must exist before any test row is built"
        prov["frozen_policy_sha256"] = sha(ART / "frozen_policy.json")
    if cache.exists():
        d = dict(np.load(cache))
        assert json.loads(str(d.pop("provenance"))) == prov, "stale cache"
        return d
    banks, ref, pid = load_banks(split)
    d = build_rows(banks, ref, pid, c_val_constant(), partitions())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(cache, provenance=json.dumps(prov), **d)
    return d


# ============================================================================ policy MAE (patient-macro, per partition)
def patient_means(loss, part, pid_idx, n_pat, mask=None):
    """(NPART, n_pat) per-patient mean loss per partition (NaN where the patient has no unit)."""
    out = np.full((NPART, n_pat), np.nan)
    for p in range(NPART):
        m = part == p if mask is None else (part == p) & mask
        s = np.bincount(pid_idx[m], loss[m], n_pat); c = np.bincount(pid_idx[m], None, n_pat)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[p] = s / c
    return out


def macro(pm, w=None):
    """mean over partitions of the (weighted) mean over patients with units."""
    if w is None:
        w = np.ones(pm.shape[1])
    ok = np.isfinite(pm)
    return float(np.mean((np.where(ok, pm, 0) @ w) / (ok @ w)))


def policy_mae(loss, part, pid_idx, n_pat, mask=None):
    return macro(patient_means(loss, part, pid_idx, n_pat, mask))


# ============================================================================ fitting (validation only)
def make_pipe(alpha):
    return Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=alpha))])


def fit_ridge_policy(X, d, pid_idx, n_pat, name, log):
    y, part = d["dQ"], d["part"]
    folds = list(GroupKFold(n_splits=5).split(X, y, d["pid"]))
    for tr, te in folds:
        assert not set(d["pid"][tr]) & set(d["pid"][te])
    best = None
    for al in ALPHAS:
        oof = np.empty(len(y))
        for tr, te in folds:
            oof[te] = make_pipe(al).fit(X[tr], y[tr]).predict(X[te])
        loss = np.where(oof > 0, d["LW"], d["LD"])
        mae = policy_mae(loss, part, pid_idx, n_pat)
        log.append({"policy": name, "param": f"alpha={al}", "oof_policy_mae": mae, "oof_width_rate": float(np.mean(oof > 0))})
        if best is None or mae < best[1] - 1e-12:
            best = (al, mae)
    pipe = make_pipe(best[0]).fit(X, y)
    sc, rg = pipe.named_steps["sc"], pipe.named_steps["rg"]
    return {"alpha": best[0], "oof_policy_mae": best[1], "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
            "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_)}


def ridge_predict(entry, X):
    return ((X - np.array(entry["scaler_mean"])) / np.array(entry["scaler_scale"])) @ np.array(entry["coef"]) + entry["intercept"]


def choose_threshold(z, LW, LD, part, pid_idx, n_pat, thresholds):
    best = None
    for t in thresholds:                                   # ascending; ties -> smallest t
        mae = policy_mae(np.where(z > t, LW, LD), part, pid_idx, n_pat)
        if best is None or mae < best[1] - 1e-12:
            best = (t, mae)
    return best


def fit_stage():
    assert not (OUT / "rows_test.npz").exists() and not (ART / "test_metrics.json").exists(), "no refit after test rows / metrics exist"
    d = table("val")
    up, pid_idx = np.unique(d["pid"], return_inverse=True); n_pat = len(up)
    X = transform_full(d["F"])
    log, fr = [], {}
    # static policies
    mae_W, mae_D = policy_mae(d["LW"], d["part"], pid_idx, n_pat), policy_mae(d["LD"], d["part"], pid_idx, n_pat)
    fr["B1_global_static"] = {"action": "WIDTH" if mae_W <= mae_D else "DEPTH", "val_mae_width": mae_W, "val_mae_depth": mae_D}
    b2 = {}
    for ci, (a, S) in enumerate(COND):
        m = d["cond"] == ci
        mw = policy_mae(d["LW"], d["part"], pid_idx, n_pat, m); md = policy_mae(d["LD"], d["part"], pid_idx, n_pat, m)
        b2[f"{NAME[a]}_S{S}"] = {"action": "WIDTH" if mw <= md else "DEPTH", "val_mae_width": mw, "val_mae_depth": md}
    fr["B2_condition_static"] = b2
    # P1 / P3 ridge
    fr["P1_SD_ridge"] = {"features": ["log1p SD(A)", "log S"], **fit_ridge_policy(X[:, P1_COLS], d, pid_idx, n_pat, "P1_SD_ridge", log)}
    fr["P3_full_ridge"] = {"features": [("log1p " if i in LOG1P_FULL else "") + f for i, f in enumerate(FULL)],
                           **fit_ridge_policy(X, d, pid_idx, n_pat, "P3_full_ridge", log)}
    # P2 threshold on z1 = log1p SD(A)
    z1 = X[:, 3]
    grid = [-np.inf] + [float(np.quantile(z1, q)) for q in QGRID] + [np.inf]
    t, mae = choose_threshold(z1, d["LW"], d["LD"], d["part"], pid_idx, n_pat, grid)
    for tt in grid:
        log.append({"policy": "P2_SD_threshold", "param": f"t={tt}", "oof_policy_mae": "", "in_sample_policy_mae":
                    policy_mae(np.where(z1 > tt, d["LW"], d["LD"]), d["part"], pid_idx, n_pat)})
    oof_loss = np.empty(len(z1))
    for tr, te in GroupKFold(n_splits=5).split(z1, z1, d["pid"]):
        u_tr, i_tr = np.unique(d["pid"][tr], return_inverse=True)
        g_tr = [-np.inf] + [float(np.quantile(z1[tr], q)) for q in QGRID] + [np.inf]
        tt, _ = choose_threshold(z1[tr], d["LW"][tr], d["LD"][tr], d["part"][tr], i_tr, len(u_tr), g_tr)
        oof_loss[te] = np.where(z1[te] > tt, d["LW"][te], d["LD"][te])
    fr["P2_SD_threshold"] = {"feature": "log1p SD(A)", "threshold": t, "threshold_quantile_grid": list(QGRID),
                             "in_sample_policy_mae": mae, "grouped_oof_policy_mae_of_selection": policy_mae(oof_loss, d["part"], pid_idx, n_pat)}
    fr["oracle_val_mae"] = policy_mae(np.minimum(d["LW"], d["LD"]), d["part"], pid_idx, n_pat)
    # informativeness gate (validation, per condition): action estimate vs c_val
    c_val = c_val_constant()
    gate = gate_validation(d, c_val, pid_idx, n_pat)
    fr["informativeness_gate"] = gate
    fr["flag_rates_validation"] = {"range": float(d["flag_range"].mean()), "outlier": float(d["flag_outlier"].mean()),
                                   "either": float((d["flag_range"] | d["flag_outlier"]).mean())}
    # validation action values (descriptive)
    with open(ART / "validation_action_values.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "n_rows", "mae_width", "mae_depth", "mean_dQ_patient_macro", "share_width_better", "share_tie",
                    "width_all_finite", "width_partial", "width_none", "depth_all_finite", "depth_partial", "depth_none"])
        for ci, (a, S) in enumerate(COND):
            m = d["cond"] == ci
            w.writerow([f"{NAME[a]}_S{S}", int(m.sum()), b2[f"{NAME[a]}_S{S}"]["val_mae_width"], b2[f"{NAME[a]}_S{S}"]["val_mae_depth"],
                        policy_mae(d["dQ"], d["part"], pid_idx, n_pat, m), float((d["dQ"][m] > 0).mean()), float((d["dQ"][m] == 0).mean()),
                        float((d["nW"][m] == 8).mean()), float(((d["nW"][m] > 0) & (d["nW"][m] < 8)).mean()), float((d["nW"][m] == 0).mean()),
                        float((d["nD"][m] == 4).mean()), float(((d["nD"][m] > 0) & (d["nD"][m] < 4)).mean()), float((d["nD"][m] == 0).mean())])
    with open(ART / "validation_cv.csv", "w", newline="") as fh:
        keys = ["policy", "param", "oof_policy_mae", "oof_width_rate", "in_sample_policy_mae"]
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in log:
            w.writerow({k: r.get(k, "") for k in keys})
    frozen = {"prereg": PREREG, "fallback_hr_c_val": c_val, "delta_mae_bpm": DELTA_MAE, "hr_range": HR_RANGE, "outlier_rel": OUTLIER_REL,
              "feature_transforms": {"P1": "[log1p SD(A) (ddof 1), log S] -> StandardScaler -> Ridge", "P2": "log1p SD(A) > t -> WIDTH",
                                     "P3": "10 features, log1p on spread features 1-8 -> StandardScaler -> Ridge"},
              "policies": fr, "n_val_rows": int(len(d["dQ"])), "n_val_patients": int(n_pat),
              "hashes": {"partition_manifest.json": sha(ART / "partition_manifest.json"), "scripts/m2_run.py": sha(__file__),
                         "scripts/m1_run.py": sha(ROOT / "scripts/m1_run.py"),
                         "val_bank_and_refs": "see artifacts/m1_pilot_gain_prediction/input_hashes.json"}}
    (ART / "frozen_policy.json").write_text(json.dumps(frozen, indent=1, default=float))
    print(json.dumps({k: (v if not isinstance(v, dict) or k in ("B1_global_static",) else {kk: vv for kk, vv in v.items() if kk in ("alpha", "oof_policy_mae", "threshold", "in_sample_policy_mae", "grouped_oof_policy_mae_of_selection", "coef")}) for k, v in fr.items() if k != "informativeness_gate"}, indent=1, default=float))
    print("B2:", {k: v["action"] for k, v in b2.items()})
    print("gate flags:", [k for k, v in gate.items() if v["flagged"]], "| flag rates", fr["flag_rates_validation"])


def gate_validation(d, c_val, pid_idx, n_pat):
    """per condition, width and depth estimators vs c_val on validation: MAE difference (patient-macro) and Spearman with
    the reference (window level, per condition); per partition, averaged over partitions; 2,000-replicate patient bootstrap."""
    rng = np.random.default_rng(SEED)
    Wb = [np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat).astype(float) for _ in range(NBOOT_GATE)]
    nc = len(COND)
    ec = np.abs(c_val - d["y"])
    diff_pm = {act: [patient_means(L - ec, d["part"], pid_idx, n_pat, d["cond"] == ci) for ci in range(nc)]
               for act, L in (("WIDTH", d["LW"]), ("DEPTH", d["LD"]))}
    prep = []
    for p in range(NPART):
        m = np.flatnonzero(d["part"] == p); c = d["cond"][m]
        prep.append((m, c, M1.RankPrep(d["eW"][m], c), M1.RankPrep(d["eD"][m], c), M1.RankPrep(d["y"][m], c)))

    def sp(w_pat):
        out = {"WIDTH": [], "DEPTH": []}
        for m, c, rw, rd, ry in prep:
            w = w_pat[pid_idx[m]]; r_y = ry.ranks(w)
            out["WIDTH"].append(M1.wcorr_seg(rw.ranks(w), r_y, w, c, nc)); out["DEPTH"].append(M1.wcorr_seg(rd.ranks(w), r_y, w, c, nc))
        return {k: np.nanmean(np.array(v), 0) for k, v in out.items()}
    sp0 = sp(np.ones(n_pat)); spb = [sp(w) for w in Wb]
    res = {}
    for ci, (a, S) in enumerate(COND):
        out = {}
        for act in ("WIDTH", "DEPTH"):
            dci = M1.ci([macro(diff_pm[act][ci], w) for w in Wb]); sci = M1.ci([b[act][ci] for b in spb])
            out[act] = {"mae_minus_const": macro(diff_pm[act][ci]), "mae_minus_const_ci": dci, "spearman_with_ref": float(sp0[act][ci]),
                        "spearman_ci": sci, "pass": bool(dci[1] < 0 and sci[0] > 0)}
        out["flagged"] = not (out["WIDTH"]["pass"] and out["DEPTH"]["pass"])
        res[f"{NAME[a]}_S{S}"] = out
    return res


# ============================================================================ test evaluation
_G = {}


def _prep_eval(d, frozen):
    up, pid_idx = np.unique(d["pid"], return_inverse=True); n_pat = len(up)
    X = transform_full(d["F"])
    pol = frozen["policies"]
    pred1 = ridge_predict(pol["P1_SD_ridge"], X[:, P1_COLS]); pred3 = ridge_predict(pol["P3_full_ridge"], X)
    act = {"B0_W": np.ones(len(X), bool), "B0_D": np.zeros(len(X), bool),
           "B1": np.full(len(X), pol["B1_global_static"]["action"] == "WIDTH"),
           "B2": np.array([pol["B2_condition_static"][f"{NAME[COND[c][0]]}_S{COND[c][1]}"]["action"] == "WIDTH" for c in range(len(COND))])[d["cond"]],
           "P1": pred1 > 0, "P2": X[:, 3] > pol["P2_SD_threshold"]["threshold"], "P3": pred3 > 0}
    loss = {k: np.where(v, d["LW"], d["LD"]) for k, v in act.items()}
    loss["oracle"] = np.minimum(d["LW"], d["LD"])
    loss50 = {k: np.where(v, d["LW50"], d["LD50"]) for k, v in act.items()}
    flag = d["flag_range"] | d["flag_outlier"]
    arm = ARM_OF_COND[d["cond"]]
    pm = {}
    for k, L in loss.items():
        pm[k] = patient_means(L, d["part"], pid_idx, n_pat)
    for k in ("P1", "B2"):
        pm[f"{k}_50"] = patient_means(loss50[k], d["part"], pid_idx, n_pat)
        pm[f"{k}_unflagged"] = patient_means(loss[k], d["part"], pid_idx, n_pat, ~flag)
        pm[f"{k}_flagged"] = patient_means(loss[k], d["part"], pid_idx, n_pat, flag)
        for ai, a in enumerate(ARMS):
            pm[f"{k}_model_{a}"] = patient_means(loss[k], d["part"], pid_idx, n_pat, arm == ai)
        for ci in range(len(COND)):
            pm[f"{k}_cond_{ci}"] = patient_means(loss[k], d["part"], pid_idx, n_pat, d["cond"] == ci)
    parts = []
    for p in range(NPART):
        m = np.flatnonzero(d["part"] == p)
        c = d["cond"][m]; dq = d["dQ"][m]; p1 = pred1[m]; p3 = pred3[m]
        parts.append({"idx": m, "cond": c, "dq": dq, "p1": p1, "p3": p3, "act1": act["P1"][m], "act2": act["P2"][m], "act3": act["P3"][m],
                      "actB2": act["B2"][m], "r_dq": M1.RankPrep(dq), "r_dq_c": M1.RankPrep(dq, c), "r_p1": M1.RankPrep(p1),
                      "r_p1_c": M1.RankPrep(p1, c), "r_p3": M1.RankPrep(p3), "r_p3_c": M1.RankPrep(p3, c)})
    return {"up": up, "pid_idx": pid_idx, "n_pat": n_pat, "pm": pm, "parts": parts, "pred1": pred1, "pred3": pred3, "act": act,
            "loss": loss, "flag": flag}


def _stats(ctx, wpat):
    pm = ctx["pm"]
    s = {f"mae_{k}": macro(v, wpat) for k, v in pm.items()}
    for k in ("B0_W", "B0_D", "B1", "B2"):
        s[f"delta_P1_minus_{k}"] = s["mae_P1"] - s[f"mae_{k}"]
    for k in ("P2", "P3"):
        s[f"delta_{k}_minus_B2"] = s[f"mae_{k}"] - s["mae_B2"]
    s["delta_P2_minus_P1"] = s["mae_P2"] - s["mae_P1"]
    s["delta_P1_minus_B2_50"] = s["mae_P1_50"] - s["mae_B2_50"]
    s["delta_P1_minus_B2_unflagged"] = s["mae_P1_unflagged"] - s["mae_B2_unflagged"]
    s["delta_P1_minus_B2_flagged"] = s["mae_P1_flagged"] - s["mae_B2_flagged"]
    for a in ARMS:
        s[f"delta_P1_minus_B2_model_{a}"] = s[f"mae_P1_model_{a}"] - s[f"mae_B2_model_{a}"]
    for ci in range(len(COND)):
        s[f"delta_P1_minus_B2_cond_{ci}"] = s[f"mae_P1_cond_{ci}"] - s[f"mae_B2_cond_{ci}"]
    s["regret_P1"] = s["mae_P1"] - s["mae_oracle"]; s["regret_B2"] = s["mae_B2"] - s["mae_oracle"]
    s["oracle_gap_recovery"] = (1 - s["regret_P1"] / s["regret_B2"]) if s["regret_B2"] > 0 else float("nan")
    acc = {k: [] for k in ("sp_p1", "sp_p1_within", "sp_p3", "sp_p3_within", "pearson_p1", "r2_p1", "acc_p1", "tie", "strata",
                           "wf_p1", "wf_p2", "wf_p3", "wf_b2")}
    z0 = None
    for pp in ctx["parts"]:
        w = wpat[ctx["pid_idx"][pp["idx"]]]
        z0 = np.zeros(len(w), np.int64)
        rdq, rdqc = pp["r_dq"].ranks(w), pp["r_dq_c"].ranks(w)
        acc["sp_p1"].append(M1.wcorr_seg(pp["r_p1"].ranks(w), rdq, w, z0, 1)[0])
        acc["sp_p1_within"].append(np.nanmean(M1.wcorr_seg(pp["r_p1_c"].ranks(w), rdqc, w, pp["cond"], len(COND))))
        acc["sp_p3"].append(M1.wcorr_seg(pp["r_p3"].ranks(w), rdq, w, z0, 1)[0])
        acc["sp_p3_within"].append(np.nanmean(M1.wcorr_seg(pp["r_p3_c"].ranks(w), rdqc, w, pp["cond"], len(COND))))
        o, p = pp["dq"], pp["p1"]; sw = w.sum(); mo, mp_ = (w * o).sum() / sw, (w * p).sum() / sw
        acc["pearson_p1"].append((w * (o - mo) * (p - mp_)).sum() / np.sqrt((w * (o - mo) ** 2).sum() * (w * (p - mp_) ** 2).sum()))
        acc["r2_p1"].append(1 - (w * (o - p) ** 2).sum() / (w * (o - mo) ** 2).sum())
        nz = o != 0
        acc["acc_p1"].append((w * nz * (pp["act1"] == (o > 0))).sum() / (w * nz).sum())
        acc["tie"].append((w * ~nz).sum() / sw)
        q25, q75 = M1.wquantile(pp["r_p1"], w, 0.25), M1.wquantile(pp["r_p1"], w, 0.75)
        top, bot = p >= q75, p <= q25
        acc["strata"].append((w * o * top).sum() / (w * top).sum() - (w * o * bot).sum() / (w * bot).sum())
        for k, a_ in (("wf_p1", pp["act1"]), ("wf_p2", pp["act2"]), ("wf_p3", pp["act3"]), ("wf_b2", pp["actB2"])):
            acc[k].append((w * a_).sum() / sw)
    s.update({{"sp_p1": "spearman_P1", "sp_p1_within": "spearman_P1_within_condition", "sp_p3": "spearman_P3",
               "sp_p3_within": "spearman_P3_within_condition", "pearson_p1": "pearson_P1", "r2_p1": "r2_P1",
               "acc_p1": "decision_accuracy_P1", "tie": "share_dQ_tied", "strata": "top_minus_bottom_quartile_dQ_P1",
               "wf_p1": "width_rate_P1", "wf_p2": "width_rate_P2", "wf_p3": "width_rate_P3", "wf_b2": "width_rate_B2"}[k]: float(np.mean(v))
              for k, v in acc.items()})
    return s


def _boot_chunk(bs):
    ctx = _G["ctx"]
    return [_stats(ctx, ctx["W"][b]) for b in bs]


def verdict_of(pt, bci):
    models_neg = sum(pt[f"delta_P1_minus_B2_model_{a}"] < 0 for a in ARMS)
    fail = {"a_P1_significantly_worse": bci["delta_P1_minus_B2"][0] > 0,
            "b_within_condition_association_nonpositive": pt["spearman_P1_within_condition"] <= 0,
            "c_no_pilot_rule_improves": all(pt[k] >= 0 for k in ("delta_P1_minus_B2", "delta_P2_minus_B2", "delta_P3_minus_B2"))}
    strong = {"1_ci_upper_below_0": bci["delta_P1_minus_B2"][1] < 0, "2_point_at_most_minus_delta": pt["delta_P1_minus_B2"] <= -DELTA_MAE,
              "3_ge2_models_negative": models_neg >= 2, "4_within_condition_ci_lower_gt_0": bci["spearman_P1_within_condition"][0] > 0}
    cat = "FAILED" if any(fail.values()) else ("STRONG" if all(strong.values()) else "PARTIAL")
    labels = []
    if cat == "PARTIAL":
        if not strong["1_ci_upper_below_0"] and pt["delta_P1_minus_B2"] < 0:
            labels.append("direction favourable, CI crosses 0")
        if pt["delta_P1_minus_B2"] >= 0:
            labels.append("P1 does not improve on condition-static" + (" (only P2/P3 improve)" if min(pt["delta_P2_minus_B2"], pt["delta_P3_minus_B2"]) < 0 else ""))
        if not strong["2_point_at_most_minus_delta"] and pt["delta_P1_minus_B2"] < 0:
            labels.append("below the practical margin")
        if not strong["3_ge2_models_negative"]:
            labels.append("one-model driven")
        if not strong["4_within_condition_ci_lower_gt_0"]:
            labels.append("within-condition association CI includes 0")
    go = cat == "STRONG" or (cat == "PARTIAL" and not strong["1_ci_upper_below_0"] and strong["2_point_at_most_minus_delta"]
                             and strong["3_ge2_models_negative"] and strong["4_within_condition_ci_lower_gt_0"])
    prefer_threshold = (bci["delta_P2_minus_P1"][0] <= 0 <= bci["delta_P2_minus_P1"][1]) or abs(pt["delta_P2_minus_P1"]) < DELTA_MAE / 2
    return {"category": cat, "labels": labels, "failed_routes": {k: bool(v) for k, v in fail.items()},
            "strong_items": {k: bool(v) for k, v in strong.items()}, "n_models_negative": int(models_neg),
            "go_M3": bool(go), "prefer_threshold_over_ridge": bool(prefer_threshold)}


def test_stage():
    frozen = json.loads((ART / "frozen_policy.json").read_text())
    d = table("test")
    ctx = _prep_eval(d, frozen)
    assert ctx["n_pat"] == N_TEST_PATIENTS
    rng = np.random.default_rng(SEED)
    ctx["W"] = [np.bincount(rng.integers(0, ctx["n_pat"], ctx["n_pat"]), minlength=ctx["n_pat"]).astype(float) for _ in range(NBOOT)]
    _G["ctx"] = ctx
    t0 = time.time()
    pt = _stats(ctx, np.ones(ctx["n_pat"]))
    print(f"  [test] point ({time.time() - t0:.0f}s)", flush=True)
    boots = []
    with mp.get_context("fork").Pool(NWORK) as pool:
        for i, r in enumerate(pool.imap(_boot_chunk, [list(range(j, min(j + 50, NBOOT))) for j in range(0, NBOOT, 50)])):
            boots += r
            if i % 10 == 0:
                print(f"  [test] bootstrap {len(boots)}/{NBOOT} ({time.time() - t0:.0f}s)", flush=True)
    bci = {k: M1.ci([b[k] for b in boots]) for k in pt}
    nonfin = {k: int(np.sum(~np.isfinite([b[k] for b in boots]))) for k in pt}
    ver = verdict_of(pt, bci)
    gate_flag = [k for k, v in frozen["policies"]["informativeness_gate"].items() if v.get("flagged")]
    metrics = {"prereg": PREREG, "frozen_policy_sha256": sha(ART / "frozen_policy.json"), "n_test_patients": ctx["n_pat"],
               "rows_per_partition_mean": float(len(d["dQ"]) / NPART), "point": pt, "ci95": bci, "verdict": ver,
               "gate_flagged_conditions": gate_flag}
    (ART / "test_metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (ART / "bootstrap.json").write_text(json.dumps({"replicates": NBOOT, "seed": SEED, "unit": "patient multiplicity over all units",
                                                    "statistic": "per partition, then mean over 32 partitions", "ci95": bci,
                                                    "n_nonfinite_replicates": nonfin}, indent=1, default=float))
    (ART / "model_specific.json").write_text(json.dumps({NAME[a]: {"mae_P1": pt[f"mae_P1_model_{a}"], "mae_B2": pt[f"mae_B2_model_{a}"],
                                                                    "delta": pt[f"delta_P1_minus_B2_model_{a}"], "ci": bci[f"delta_P1_minus_B2_model_{a}"]}
                                                         for a in ARMS}, indent=1, default=float))
    (ART / "condition_specific.json").write_text(json.dumps({f"{NAME[a]}_S{S}": {
        "B2_action": frozen["policies"]["B2_condition_static"][f"{NAME[a]}_S{S}"]["action"], "mae_P1": pt[f"mae_P1_cond_{ci}"],
        "mae_B2": pt[f"mae_B2_cond_{ci}"], "delta": pt[f"delta_P1_minus_B2_cond_{ci}"], "ci": bci[f"delta_P1_minus_B2_cond_{ci}"]}
        for ci, (a, S) in enumerate(COND)}, indent=1, default=float))
    miss = {}
    for ci, (a, S) in enumerate(COND):
        m = d["cond"] == ci
        miss[f"{NAME[a]}_S{S}"] = {"width_all_finite": float((d["nW"][m] == 8).mean()), "width_partial": float(((d["nW"][m] > 0) & (d["nW"][m] < 8)).mean()),
                                   "width_none_fallback": float((d["nW"][m] == 0).mean()), "depth_all_finite": float((d["nD"][m] == 4).mean()),
                                   "depth_partial": float(((d["nD"][m] > 0) & (d["nD"][m] < 4)).mean()), "depth_none_fallback": float((d["nD"][m] == 0).mean()),
                                   "width_below_50pct": float((d["nW"][m] < 4).mean()), "depth_below_50pct": float((d["nD"][m] < 2).mean())}
    (ART / "missingness.json").write_text(json.dumps({"test": miss, "sensitivity_50pct_rule": {"delta_P1_minus_B2": pt["delta_P1_minus_B2_50"],
                                                                                              "ci": bci["delta_P1_minus_B2_50"]}}, indent=1))
    (ART / "outlier_sensitivity.json").write_text(json.dumps({
        "rules": {"range": HR_RANGE, "pilot_relative_deviation": OUTLIER_REL}, "test_flag_rate": float(ctx["flag"].mean()),
        "validation_flag_rates": frozen["policies"]["flag_rates_validation"],
        **{k: {"point": pt[k], "ci": bci[k]} for k in ("delta_P1_minus_B2", "delta_P1_minus_B2_unflagged", "delta_P1_minus_B2_flagged",
                                                        "mae_P1_unflagged", "mae_B2_unflagged", "mae_P1_flagged", "mae_B2_flagged")}}, indent=1, default=float))
    np.savez(OUT / "test_policy_rows.npz", part=d["part"], cond=d["cond"], pid=d["pid"], dQ=d["dQ"], pred_P1=ctx["pred1"], pred_P3=ctx["pred3"],
             LW=d["LW"], LD=d["LD"], flag=ctx["flag"])
    print(json.dumps(ver, indent=1))
    for k in ("mae_B0_W", "mae_B0_D", "mae_B1", "mae_B2", "mae_P2", "mae_P1", "mae_P3", "mae_oracle", "delta_P1_minus_B2", "delta_P2_minus_B2",
              "delta_P3_minus_B2", "delta_P2_minus_P1", "spearman_P1", "spearman_P1_within_condition", "oracle_gap_recovery", "width_rate_P1"):
        print(f"  {k}: {pt[k]:+.4f} {bci[k]}")


# ============================================================================ latency
def latency_stage():
    import torch
    import dw1_depth_width as DW
    dev = torch.device("cuda")
    x1 = np.zeros((1, 512), np.float32) + np.sin(np.linspace(0, 20, 512, dtype=np.float32))
    res = {}
    for arm in ARMS:
        sampler = DW.make_sampler(arm, dev)

        def timed(X, S, calls, reps=7, _s=sampler):
            ts = []
            for r in range(reps + 1):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for c in range(calls):
                    _s(X, c, S)
                torch.cuda.synchronize()
                if r:
                    ts.append(1000 * (time.perf_counter() - t0))
            return float(np.median(ts))
        for S in BASE_S:
            res[f"{NAME[arm]}_S{S}"] = {
                "width_8xS_sequential_ms": timed(x1, S, 8), "depth_4x2S_sequential_ms": timed(x1, 2 * S, 4),
                "width_8xS_batched_ms": timed(np.repeat(x1, 8, 0), S, 1), "depth_4x2S_batched_ms": timed(np.repeat(x1, 4, 0), 2 * S, 1),
                "pilot_4xS_batched_ms": timed(np.repeat(x1, 4, 0), S, 1), "nfe_pilot": 4 * S, "nfe_width": 8 * S, "nfe_depth": 8 * S}
        del sampler; torch.cuda.empty_cache()
    (ART / "cost_accounting.json").write_text(json.dumps({
        "unit": "vector-field NFE per window; latency on RTX 5090 (median of 7 runs, one window, includes host-side sampler overhead and HR-free generation)",
        "design": {"pilot": "4*S", "width_action": "8*S (8 samples at S)", "depth_action": "8*S (4 samples at 2S)", "total": "12*S"},
        "term": "equal future vector-field NFE", "flops": "not measured (no FLOPs infrastructure in the repository)", "per_condition": res}, indent=1))
    print(json.dumps(res, indent=1))


# ============================================================================ figure
def figure_stage():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tm = json.loads((ART / "test_metrics.json").read_text()); pt, cb = tm["point"], tm["ci95"]
    z = np.load(OUT / "test_policy_rows.npz")
    fig = plt.figure(figsize=(21, 4.4))
    ax = [fig.add_subplot(1, 5, i + 1) for i in range(5)]
    a0 = ax[0]; a0.axis("off")
    a0.text(0.5, 0.92, "A_S: 4 pilot draws at depth S\n(4·S NFE, only decides)", ha="center", va="top", fontsize=9, bbox=dict(fc="#eef", ec="#557"))
    a0.annotate("", xy=(0.5, 0.58), xytext=(0.5, 0.72), arrowprops=dict(arrowstyle="->"))
    a0.text(0.5, 0.55, "policy (pilot statistics)", ha="center", va="center", fontsize=9, bbox=dict(fc="#ffe", ec="#997"))
    a0.annotate("", xy=(0.22, 0.30), xytext=(0.45, 0.48), arrowprops=dict(arrowstyle="->"))
    a0.annotate("", xy=(0.78, 0.30), xytext=(0.55, 0.48), arrowprops=dict(arrowstyle="->"))
    a0.text(0.22, 0.22, "WIDTH\n8 draws × S\nmedian(W_S)", ha="center", va="center", fontsize=9, bbox=dict(fc="#e8f4e8", ec="#585"))
    a0.text(0.78, 0.22, "DEPTH\n4 draws × 2S\nmedian(D_2S)", ha="center", va="center", fontsize=9, bbox=dict(fc="#f4e8e8", ec="#855"))
    a0.text(0.5, 0.04, "same future vector-field NFE (8·S); seed IDs A, W, D disjoint", ha="center", fontsize=8)
    a0.set_title("A. M2 design", fontsize=9)
    m0 = z["part"] == 0
    ax[1].hexbin(z["pred_P1"][m0], np.clip(z["dQ"][m0], -20, 20), gridsize=50, bins="log", cmap="Blues", mincnt=1)
    ax[1].axhline(0, color="#555", lw=0.6); ax[1].axvline(0, color="#555", lw=0.6)
    ax[1].set_xlabel("predicted ΔQ (P1, partition 0)"); ax[1].set_ylabel("observed ΔQ = L_D − L_W (bpm, clipped ±20)")
    ax[1].set_title(f"B. within-condition Spearman {pt['spearman_P1_within_condition']:+.3f} "
                    f"[{cb['spearman_P1_within_condition'][0]:+.3f}, {cb['spearman_P1_within_condition'][1]:+.3f}]", fontsize=8.5)
    names = [("B0_W", "always W"), ("B0_D", "always D"), ("B1", "global static"), ("B2", "condition static"), ("P2", "SD threshold"),
             ("P1", "SD ridge (P1)"), ("P3", "full ridge"), ("oracle", "oracle (not attainable)")]
    v = [pt[f"mae_{k}"] for k, _ in names]; lo = [pt[f"mae_{k}"] - cb[f"mae_{k}"][0] for k, _ in names]; hi = [cb[f"mae_{k}"][1] - pt[f"mae_{k}"] for k, _ in names]
    ax[2].barh(range(len(names)), v, xerr=[lo, hi], color=["#bbb"] * 3 + ["#555", "#8a5a00", "#1f4e79", "#6a3d9a", "#ddd"])
    ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels([n for _, n in names], fontsize=7.5); ax[2].invert_yaxis()
    ax[2].set_xlim(min(v) * 0.95, max(v) * 1.02); ax[2].set_xlabel("test policy MAE (bpm, patient-macro)"); ax[2].set_title("C. policies", fontsize=9)
    keys = [f"delta_P1_minus_B2_cond_{ci}" for ci in range(len(COND))]
    labs = [f"{NAME[a]} S{S}" for a, S in COND]
    ax[3].errorbar([pt[k] for k in keys], range(len(keys)), xerr=[[pt[k] - cb[k][0] for k in keys], [cb[k][1] - pt[k] for k in keys]],
                   fmt="o", ms=4, capsize=2, color="#1f4e79")
    ax[3].axvline(0, color="#555", lw=0.8); ax[3].set_yticks(range(len(keys))); ax[3].set_yticklabels(labs, fontsize=7.5); ax[3].invert_yaxis()
    ax[3].set_xlabel("MAE(P1) − MAE(condition static), bpm"); ax[3].set_title(f"D. per model / depth (overall {pt['delta_P1_minus_B2']:+.3f})", fontsize=9)
    ks = ["delta_P1_minus_B2", "delta_P1_minus_B2_unflagged", "delta_P1_minus_B2_flagged"]
    ax[4].errorbar([pt[k] for k in ks], range(3), xerr=[[pt[k] - cb[k][0] for k in ks], [cb[k][1] - pt[k] for k in ks]], fmt="o", ms=5, capsize=3, color="#a33")
    ax[4].axvline(0, color="#555", lw=0.8); ax[4].set_yticks(range(3)); ax[4].set_yticklabels(["all deployable", "pilot unflagged", "pilot flagged"])
    ax[4].invert_yaxis(); ax[4].set_xlabel("MAE(P1) − MAE(condition static), bpm"); ax[4].set_title("E. HR-extractor outlier sensitivity", fontsize=9)
    for a_ in ax[1:]:
        a_.grid(lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(ART / "figure.png", dpi=150)
    print("wrote", ART / "figure.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tables-val", "fit", "tables-test", "test", "latency", "figure"])
    st = ap.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    {"tables-val": lambda: table("val"), "fit": fit_stage, "tables-test": lambda: table("test"), "test": test_stage,
     "latency": latency_stage, "figure": figure_stage}[st]()
