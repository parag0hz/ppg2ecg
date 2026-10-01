"""Diagnostic separability probe, feature effect sizes and the frozen E0 case rule (E0 prereg §7-§9).

The probe is an L2-regularized logistic regression on frozen scalar features, label 1 = type C (spontaneous recovery),
0 = type D (spontaneous hallucination). It is fitted on ARCH-TRAIN only (patient-grouped CV for C, imputation and
standardization fitted on training data only) and applied unchanged to ARCH-VAL and ARCH-HOLDOUT. It is not an ECG model
and not a proposed guard.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

C_GRID = (0.01, 0.1, 1.0, 10.0)
N_FOLDS = 5
MARGIN = 0.02             # the program's F1 / recall materiality margin (C0 G1/G4, C0-A M1-M3)
SHARE_C_MIN = 0.20        # recovered share C / (C + D) above which type C counts as nontrivial
HALF = 0.5                # "most" / "at least half"
AUROC_BAR = {"val": 0.80, "holdout": 0.75}


# ----------------------------------------------------------------------------------------------- preprocessing
@dataclass
class Prep:
    median: np.ndarray
    indicator_cols: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    names: list = field(default_factory=list)


def fit_prep(F: np.ndarray, names=None) -> Prep:
    """Median imputation + one missingness indicator per feature with any nan, then z-scoring; all fitted on F only."""
    F = np.asarray(F, dtype=np.float64)
    med = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in F.T])
    ind = np.flatnonzero(~np.isfinite(F).all(axis=0))
    Z = _impute(F, med, ind)
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd[sd < 1e-12] = 1.0
    nm = list(names) if names is not None else [f"f{i}" for i in range(F.shape[1])]
    return Prep(med, ind, mu, sd, nm + [f"{nm[i]}_missing" for i in ind])


def _impute(F, med, ind):
    miss = ~np.isfinite(F)
    Z = np.where(miss, med[None, :], F)
    return np.concatenate([Z, miss[:, ind].astype(np.float64)], axis=1)


def transform(p: Prep, F: np.ndarray) -> np.ndarray:
    return (_impute(np.asarray(F, dtype=np.float64), p.median, p.indicator_cols) - p.mean) / p.std


def _model(c):
    return LogisticRegression(C=c, penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=5000)


def best_threshold(y, p) -> float:
    """Threshold maximizing balanced accuracy (predict C iff prob >= threshold); ties -> the lowest threshold."""
    best, thr = -1.0, 0.5
    for t in np.unique(p):
        ba = balanced_accuracy(y, p >= t)
        if ba > best + 1e-12:
            best, thr = ba, float(t)
    return thr


def balanced_accuracy(y, pred) -> float:
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    sens = np.mean(pred[y]) if y.any() else np.nan
    spec = np.mean(~pred[~y]) if (~y).any() else np.nan
    return float(np.nanmean([sens, spec]))


@dataclass
class Probe:
    prep: Prep
    model: LogisticRegression
    c: float
    threshold: float
    cv: dict


def fit_probe(F, y, groups, names=None) -> Probe:
    """Patient-grouped 5-fold CV over C_GRID (mean fold AUROC; ties -> smaller C), TRAIN-OOF threshold, refit on all."""
    F, y, groups = np.asarray(F, np.float64), np.asarray(y, int), np.asarray(groups)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(F, y, groups))
    cv = {}
    for c in C_GRID:
        oof, aucs = np.full(len(y), np.nan), []
        for tr, te in folds:
            p = fit_prep(F[tr], names)
            m = _model(c).fit(transform(p, F[tr]), y[tr])
            oof[te] = m.predict_proba(transform(p, F[te]))[:, 1]
            aucs.append(float(roc_auc_score(y[te], oof[te])) if len(np.unique(y[te])) == 2 else float("nan"))
        cv[c] = {"fold_auroc": aucs, "mean_auroc": float(np.nanmean(aucs)), "oof": oof}
    c_best = max(C_GRID, key=lambda c: (round(cv[c]["mean_auroc"], 12), -c))
    thr = best_threshold(y, cv[c_best]["oof"])
    p = fit_prep(F, names)
    m = _model(c_best).fit(transform(p, F), y)
    return Probe(p, m, float(c_best), thr, {str(c): {k: v for k, v in cv[c].items() if k != "oof"} for c in C_GRID})


def predict(probe: Probe, F) -> np.ndarray:
    return probe.model.predict_proba(transform(probe.prep, F))[:, 1]


def evaluate(probe: Probe, F, y, pid, resamples) -> dict:
    """AUROC / AUPRC (positive = type C) with patient-cluster bootstrap CIs, balanced accuracy at the TRAIN threshold,
    sensitivity for C, specificity against D, prevalence and a 10-bin quantile calibration table (descriptive)."""
    y = np.asarray(y, int)
    p = predict(probe, F)
    pid = np.asarray(pid)
    subs = np.unique(pid)
    rows = [np.flatnonzero(pid == s) for s in subs]
    au, ap = [], []
    for r in resamples:
        idx = np.concatenate([rows[k] for k in r])
        if len(np.unique(y[idx])) == 2:
            au.append(roc_auc_score(y[idx], p[idx]))
            ap.append(average_precision_score(y[idx], p[idx]))
    pred = p >= probe.threshold
    q = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    cal = []
    for a, b in zip(q[:-1], q[1:]):
        s = (p >= a) & ((p < b) if b < q[-1] else (p <= b))
        if s.any():
            cal.append({"p_lo": float(a), "p_hi": float(b), "n": int(s.sum()), "mean_pred": float(p[s].mean()), "frac_C": float(y[s].mean())})
    two = len(np.unique(y)) == 2
    return {"n": int(len(y)), "n_C": int(y.sum()), "n_D": int((1 - y).sum()), "prevalence_C": float(y.mean()) if len(y) else float("nan"),
            "auroc": [float(roc_auc_score(y, p)) if two else float("nan"), *_pct(au)],
            "auprc": [float(average_precision_score(y, p)) if two else float("nan"), *_pct(ap)],
            "balanced_accuracy": balanced_accuracy(y, pred), "sensitivity_C": float(np.mean(pred[y == 1])) if (y == 1).any() else float("nan"),
            "specificity_D": float(np.mean(~pred[y == 0])) if (y == 0).any() else float("nan"), "calibration": cal}


def _pct(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


# ----------------------------------------------------------------------------------------------- effect sizes
def cohens_d_cluster(vals_c, pid_c, vals_d, pid_d, resamples, subs) -> list[float]:
    """Cohen's d (C - D, pooled SD over events) and its patient-cluster bootstrap 95% CI via per-patient sufficient
    statistics; nan values are skipped. `subs` is the sorted patient list that `resamples` indexes."""
    def suff(v, p):
        v, p = np.asarray(v, np.float64), np.asarray(p)
        ok = np.isfinite(v)
        k = np.searchsorted(subs, p[ok])
        n = np.bincount(k, minlength=subs.size).astype(np.float64)
        s = np.bincount(k, weights=v[ok], minlength=subs.size)
        q = np.bincount(k, weights=v[ok] ** 2, minlength=subs.size)
        return n, s, q

    def d_of(w, sc, sd):
        out = []
        for (n, s, q) in (sc, sd):
            N, S, Q = w @ n, w @ s, w @ q
            m = S / np.maximum(N, 1)
            var = (Q - N * m ** 2) / np.maximum(N - 1, 1)
            out.append((N, m, var))
        (n1, m1, v1), (n2, m2, v2) = out
        pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / np.maximum(n1 + n2 - 2, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((n1 > 1) & (n2 > 1) & (pooled > 0), (m1 - m2) / pooled, np.nan)

    sc, sd = suff(vals_c, pid_c), suff(vals_d, pid_d)
    point = float(d_of(np.ones((1, subs.size)), sc, sd)[0])
    W = np.stack([np.bincount(r, minlength=subs.size) for r in resamples]).astype(np.float64)
    draws = d_of(W, sc, sd)
    draws = draws[np.isfinite(draws)]
    return [point, *_pct(draws)]


def patient_macro(vals, pid) -> dict:
    """Per-patient median, then the mean and median across patients (patients with >= 1 finite value)."""
    vals, pid = np.asarray(vals, np.float64), np.asarray(pid)
    per = [np.nanmedian(vals[pid == s]) for s in np.unique(pid) if np.isfinite(vals[pid == s]).any()]
    return {"patients": len(per), "mean_of_patient_medians": float(np.mean(per)) if per else float("nan"),
            "median_of_patient_medians": float(np.median(per)) if per else float("nan")}


# ----------------------------------------------------------------------------------------------- frozen case rule
def criteria(q: dict, role: str) -> dict:
    """q: per-population quantities ([point, lo, hi] triples unless noted), all WW-DET unless named:
    dD, dC, dFP (WW - C0 per window); share_C (pooled C / (C + D), point); hg_dF1, hg_dFP, hg_dRecall (hard guard - WW);
    o1_dF1, o1_dFP (oracle-1 - WW); auroc_p3 (point)."""
    k1 = q["dD"][1] > 0 and q["dD"][0] >= HALF * q["dFP"][0]
    k2a = q["dC"][1] > 0 and q["share_C"] >= SHARE_C_MIN
    k2b = q["hg_dF1"][2] < 0
    k2c = q["hg_dRecall"][2] < -MARGIN
    k3 = q["auroc_p3"] >= AUROC_BAR[role]
    k4 = q["o1_dFP"][2] < 0 and -q["o1_dFP"][0] >= HALF * q["dFP"][0] and q["o1_dF1"][1] > 0
    b = (not k2a) and (1.0 - q["share_C"]) >= HALF and q["hg_dF1"][1] > 0 and q["hg_dFP"][2] < 0 and q["hg_dRecall"][1] > -MARGIN
    return {"K1_excess_D": bool(k1), "K2a_C_nontrivial": bool(k2a), "K2b_hardguard_F1_worse": bool(k2b),
            "K2c_hardguard_recall_material_loss": bool(k2c), "K2": bool(k2a or k2b or k2c), "K3_separable": bool(k3),
            "K4_oracle_headroom": bool(k4), "B_conditions": bool(b),
            "guard_must_be_selective": bool(k2b or k2c)}


def case_of(c: dict) -> str:
    if c["K1_excess_D"] and c["K2"] and c["K3_separable"] and c["K4_oracle_headroom"]:
        return "E0-A"
    if c["B_conditions"]:
        return "E0-B"
    return "E0-C"


def final_case(case_val: str, case_holdout: str) -> str:
    """The ARCH-VAL case stands only if the previously opened ARCH-HOLDOUT gives the same case; otherwise E0-C."""
    return case_val if case_val == case_holdout else "E0-C"
