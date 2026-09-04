"""E3 — beat-set-first event geometry: threshold-free R1 candidates, top-K selection, count readout, gates.

Frozen by docs/E3_BEAT_SET_FIRST_EVENT_GEOMETRY_PREREGISTRATION.md.

Exactly ONE learned new object is permitted in E3: a StandardScaler + Ridge(alpha=1.0) count readout on
frozen R1 pre-logit features. No generator, backbone or operator weight is ever updated here, and every
evaluation metric is imported from the frozen E2 contract rather than restated.
"""
from __future__ import annotations

import numpy as np

from ppg2ecg.evaluation import event_geometry_contract as EG
from ppg2ecg.evaluation import o2b_warp as BW
from ppg2ecg.probes import rhythm_tcn as RT

R1_THRESHOLD = 0.35
REFRACTORY = RT.REFRACTORY_SAMPLES                       # 32, imported, never changed
THRESHOLD_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))   # 0.05 .. 0.95, exactly 19 values
K_STRUCTURAL_MAX = 32                                    # 1024 samples with a 32-sample refractory
RIDGE_ALPHA = 1.0
TRAIN12 = ("e61", "fex", "l38", "n31", "ngh", "p5d", "p9p", "qm9", "trh", "tz8", "u7y", "w4p")
ARMS = ("R1-0.35", "R1-TRAIN-THRESH", "ORACLE-COUNT-R1", "E3-RIDGE-COUNT")


# --------------------------------------------------------------------------- threshold-free candidates
def candidate_events(prob, refractory: int = REFRACTORY):
    """`rhythm_tcn.extract_events` with ONLY the amplitude-filter line removed.

    Peak definition, boundary behaviour, refractory logic and tie behaviour are byte-identical to the frozen
    detector. Filtering the result at `score >= 0.35` reproduces the frozen R1 event list exactly, because
    the NMS is greedy by DESCENDING score: a sub-threshold peak is processed after every supra-threshold one
    and can never suppress it, and the stable sort preserves their relative order.
    """
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    n = p.size
    lm = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    if n >= 2:
        if p[0] > p[1]:
            lm = np.concatenate([[0], lm])
        if p[-1] > p[-2]:
            lm = np.concatenate([lm, [n - 1]])
    if lm.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=np.float64)
    order = lm[np.argsort(-p[lm], kind="stable")]
    keep: list[int] = []
    for i in order:
        if all(abs(int(i) - int(j)) > refractory for j in keep):
            keep.append(int(i))
    pos = np.sort(np.asarray(keep, dtype=int))
    return pos, p[pos]


def reproduces_r1(prob, threshold: float = R1_THRESHOLD, refractory: int = REFRACTORY) -> bool:
    pos, sc = candidate_events(prob, refractory)
    return bool(np.array_equal(pos[sc >= threshold], RT.extract_events(prob, threshold, refractory)))


# --------------------------------------------------------------------------- top-K selector
def topk_select(positions, scores, k: int):
    """Score descending, ties by lower sample index first, first K, returned ascending in time.

    The refractory constraint belongs to candidate extraction and is never reapplied here: any subset of the
    candidates already satisfies it. No fallback, no GT correction, no timing movement.
    """
    pos = np.asarray(positions, dtype=np.int64)
    sc = np.asarray(scores, dtype=np.float64)
    k = int(k)
    shortage = bool(k > pos.size)
    if k <= 0:
        return np.zeros(0, dtype=np.int64), shortage
    order = np.lexsort((pos, -sc))                        # primary -score ascending, secondary index ascending
    return np.sort(pos[order[:min(k, pos.size)]]), shortage


# --------------------------------------------------------------------------- count readout
def count_features(h) -> np.ndarray:
    """z = concat(mean_t(H), max_t(H)) for the frozen pre-logit tensor H [C, T]. Nothing else is appended."""
    H = np.asarray(h, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"expected H as [C, T], got {H.shape}")
    return np.concatenate([H.mean(axis=1), H.max(axis=1)])


def to_int_count(y_hat) -> np.ndarray:
    """round_half_to_even, then the STRUCTURAL clip [0, 32] only. Never a data-dependent range."""
    k = BW.round_half_to_even(np.asarray(y_hat, dtype=np.float64))
    return np.clip(k, 0, K_STRUCTURAL_MAX).astype(np.int64)


def standardize_fit(X) -> tuple[np.ndarray, np.ndarray]:
    """Train-only mean/std; a zero-variance dimension gets std = 1."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    return mu, sd


# --------------------------------------------------------------------------- threshold control
def select_threshold(rows) -> dict:
    """Frozen lexicographic objective on TRAIN12 rows: max A5, then min A4, then min A3, then |t - 0.35|,
    then the lower numerical threshold. `rows` carry `threshold`, `A5`, `A4`, `A3`."""
    def key(r):
        return (-round(float(r["A5"]), 12), float(r["A4"]), float(r["A3"]),
                abs(float(r["threshold"]) - R1_THRESHOLD), float(r["threshold"]))
    best = sorted(rows, key=key)[0]
    return {"selected_threshold": float(best["threshold"]), "objective": "lexicographic: max A5 (ties 1e-12) "
            "-> min A4 -> min A3 -> min |t-0.35| -> lower t",
            "A5": float(best["A5"]), "A4": float(best["A4"]), "A3": float(best["A3"]),
            "grid": list(THRESHOLD_GRID), "population": "train12 only"}


# --------------------------------------------------------------------------- topology indicators
def topology_indicators(row: dict) -> dict:
    """Window-fraction metrics enter the bootstrap as per-window 0/1 indicators."""
    c = row["A6_topology_class"]
    return {"A5_exact_set": float(c == EG.T0), "T0_frac": float(c == EG.T0), "T1_frac": float(c == EG.T1),
            "T2_frac": float(c == EG.T2), "T3_frac": float(c == EG.T3)}


# --------------------------------------------------------------------------- gates
def _lo(e, m):
    return float(e[m]["lo"])


def _pt(e, m):
    return float(e[m]["point"])


def oracle_count_gates(e: dict) -> dict:
    """OC1-OC6. `e` maps a metric to a bootstrap dict oriented so that POSITIVE = the new arm is better."""
    g = {"OC1": bool(_lo(e, "A5_exact_set") > 0 and _pt(e, "A5_exact_set") >= 0.10),
         "OC2": bool(_lo(e, "T3_frac") > 0 and _pt(e, "T3_frac") >= 0.15),
         "OC3": bool(_lo(e, "A4_spurious_fraction") > 0 and _pt(e, "A4_spurious_fraction") >= 0.05),
         "OC4": bool(_lo(e, "A3_missing_fraction") > -0.020),
         "OC5": bool(_lo(e, "B5_exact_set_mae_ms") > -8.0),
         "OC6": bool(_lo(e, "SG_F1_50") > 0)}
    g["passed"] = bool(all(g[k] for k in ("OC1", "OC2", "OC3", "OC4", "OC5", "OC6")))
    return g


def oracle_generator_gates(e: dict) -> dict:
    """OG1-OG6."""
    g = {"OG1": bool(_lo(e, "PG_F1_50") > 0 and _pt(e, "PG_F1_50") >= 0.05),
         "OG2": bool(_lo(e, "C2_own_T6") > -0.020),
         "OG3": bool(_lo(e, "C3_own_T7") > -0.020),
         "OG4": bool(_lo(e, "C2_own_T6") > 0 or _lo(e, "C3_own_T7") > 0),
         "OG5": bool(_lo(e, "J2_gt_local_deriv_rmse") > -0.020),
         "OG6": bool(_lo(e, "J3_gt_local_curvature_err") > -0.020)}
    g["passed"] = bool(all(g[k] for k in ("OG1", "OG2", "OG3", "OG4", "OG5", "OG6")))
    return g


def predicted_count_gates(e: dict) -> dict:
    """PC1-PC7, E3-RIDGE-COUNT against R1-0.35."""
    g = {"PC1": bool(_lo(e, "A5_exact_set") > 0 and _pt(e, "A5_exact_set") >= 0.05),
         "PC2": bool(_lo(e, "T3_frac") > 0 and _pt(e, "T3_frac") >= 0.10),
         "PC3": bool(_lo(e, "A4_spurious_fraction") > 0 and _pt(e, "A4_spurious_fraction") >= 0.04),
         "PC4": bool(_lo(e, "A3_missing_fraction") > -0.020),
         "PC5": bool(_lo(e, "T2_frac") > -0.020),
         "PC6": bool(_lo(e, "B5_exact_set_mae_ms") > -8.0),
         "PC7": bool(_lo(e, "SG_F1_50") > 0)}
    g["passed"] = bool(all(g[f"PC{i}"] for i in range(1, 8)))
    return g


def threshold_control_gates(e: dict) -> dict:
    """TC1-TC3, E3-RIDGE-COUNT against R1-TRAIN-THRESH."""
    g = {"TC1": bool(_lo(e, "A5_exact_set") > 0 and _pt(e, "A5_exact_set") >= 0.03),
         "TC2": bool(_lo(e, "A3_missing_fraction") > -0.020),
         "TC3": bool(_lo(e, "A4_spurious_fraction") > -0.020)}
    g["passed"] = bool(all(g[k] for k in ("TC1", "TC2", "TC3")))
    return g


def downstream_gates(e: dict, adherence_f1_50: float) -> dict:
    """DG1-DG7, E3-RIDGE-COUNT against the R1-0.35 generator arm."""
    g = {"DG1": bool(adherence_f1_50 >= 0.90),
         "DG2": bool(_lo(e, "PG_F1_50") > 0 and _pt(e, "PG_F1_50") >= 0.05),
         "DG3": bool(_lo(e, "C2_own_T6") > -0.020),
         "DG4": bool(_lo(e, "C3_own_T7") > -0.020),
         "DG5": bool(_lo(e, "C2_own_T6") > 0 or _lo(e, "C3_own_T7") > 0),
         "DG6": bool(_lo(e, "J2_gt_local_deriv_rmse") > -0.020),
         "DG7": bool(_lo(e, "J3_gt_local_curvature_err") > -0.020)}
    g["passed"] = bool(all(g[f"DG{i}"] for i in range(1, 8)))
    return g


def downstream_control_gates(e: dict) -> dict:
    """DC1-DC3, E3-RIDGE-COUNT against the R1-TRAIN-THRESH generator arm."""
    g = {"DC1": bool(_lo(e, "PG_F1_50") > -0.02),
         "DC2": bool(_lo(e, "C2_own_T6") > -0.02 and _lo(e, "C3_own_T7") > -0.02),
         "DC3": bool(any(_lo(e, m) > 0 for m in ("PG_F1_50", "C2_own_T6", "C3_own_T7",
                                                 "J2_gt_local_deriv_rmse", "J3_gt_local_curvature_err")))}
    g["passed"] = bool(all(g[k] for k in ("DC1", "DC2", "DC3")))
    return g


VERDICT_A = "BEAT-SET-FIRST COUNT CONSTRAINT SUPPORTED"
VERDICT_B = "COUNT CORRECTION CEILING SUPPORTED, MINIMAL PPG COUNT READOUT INSUFFICIENT"
VERDICT_C = "TRAIN-ONLY THRESHOLD CONTROL SUFFICIENT"
VERDICT_D = "COUNT-CONSTRAINED SCHEDULE IMPROVES, DOWNSTREAM BENEFIT NOT SUPPORTED"
VERDICT_E = "COUNT-ONLY CEILING NOT SUPPORTED"
VERDICT_PRECHECK = "CANDIDATE EXTRACTION NOT IDENTICAL TO R1"
VERDICT_CAPACITY = "CANDIDATE CAPACITY INSUFFICIENT FOR A FAIR COUNT CEILING"


def decide_e3(oc=None, og=None, pc=None, tc=None, dg=None, dc=None, precheck_ok: bool = True) -> dict:
    """Preregistration section 18. Each stage may be None when an earlier stage stopped the experiment."""
    if not precheck_ok:
        v = VERDICT_PRECHECK
    elif oc is None or not oc["passed"] or og is None or not og["passed"]:
        v = VERDICT_E
    elif pc is None or not pc["passed"]:
        v = VERDICT_B
    elif tc is None or not tc["passed"]:
        v = VERDICT_C
    elif dg is None or not dg["passed"] or dc is None or not dc["passed"]:
        v = VERDICT_D
    else:
        v = VERDICT_A
    return {"verdict": v, "OC": oc, "OG": og, "PC": pc, "TC": tc, "DG": dg, "DC": dc,
            "precheck_ok": bool(precheck_ok),
            "ridge_fitted": bool(oc is not None and oc["passed"] and og is not None and og["passed"]),
            "generator_stage_reached": bool(dg is not None)}
