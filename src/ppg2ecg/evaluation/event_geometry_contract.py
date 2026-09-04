"""E2 — the frozen event-geometry evaluation contract.

Frozen by docs/E2_EVENT_SET_PLACEMENT_MORPHOLOGY_CONTRACT_PREREGISTRATION.md and by the machine-readable
`artifacts/e2_evaluation_contract/contract_v1.json`, which is the single source of truth for every constant.
Future PPG schedule experiments import THIS module; they must not restate metric definitions.

MEASUREMENT CONTRACT ONLY. Nothing here trains, updates or selects a model, and no arm name of any specific
experiment appears in a metric function.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ppg2ecg.evaluation import e1_decompose as E1
from ppg2ecg.evaluation import m1_structural as M1
from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.evaluation import rpeaks as R

CONTRACT_PATH = Path(__file__).resolve().parents[3] / "artifacts/e2_evaluation_contract/contract_v1.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text())
VERSION = CONTRACT["contract_version"]

TOL_SG_MS = float(CONTRACT["matching"]["S_to_G"]["tolerance_ms"])          # 150.0
TOL_PS_MS = float(CONTRACT["matching"]["P_to_S"]["tolerance_ms"])          # 50.0
TOLS_PG_MS = tuple(float(t) for t in CONTRACT["matching"]["P_to_G"]["tolerances_ms"])
SUPPORT = tuple(int(v) for v in CONTRACT["support"]["own_centre"])          # (-10, 15)
FS = int(CONTRACT["support"]["fs"])
T_LEN = int(CONTRACT["support"]["n_time"])
IQR = {"T4": CONTRACT["normalization"]["T4_train_IQR"], "T6": CONTRACT["normalization"]["T6_train_IQR"],
       "T7": CONTRACT["normalization"]["T7_train_IQR"], "T8": CONTRACT["normalization"]["T8_train_IQR"]}
BOOT_N = int(CONTRACT["bootstrap"]["n_replicates"])
BOOT_SEED = int(CONTRACT["bootstrap"]["rng"].split("(")[1].rstrip(")"))
HIGH_ADHERENCE = 0.90
assert f">= {HIGH_ADHERENCE:.2f}" in CONTRACT["terminology"]["descriptive_labels"]["HIGH ADHERENCE"]
T0, T1, T2, T3 = E1.T0, E1.T1, E1.T2, E1.T3

# the contract's constants and the frozen primitives must not drift apart
assert SUPPORT == (E1.WIN_LO, E1.WIN_HI) and FS == E1.FS and T_LEN == E1.T_LEN
assert TOL_SG_MS == E1.TOL_IDENTITY_MS and TOL_PS_MS == E1.TOL_CHAIN_MS
assert (BOOT_N, BOOT_SEED) == (E1.BOOT_N, E1.BOOT_SEED)
_IQRKEY = {"median_QRS_p2p": "T4", "median_QRS_max_abs_derivative": "T6",
           "median_QRS_curvature_energy": "T7", "median_QRS_width_ms": "T8"}
_TARGET_IQR = {t: IQR[_IQRKEY[t]] for t in E1.ALIGNED}


# --------------------------------------------------------------------------------- matching
def assign_schedule_to_gt(gt, sup, identity=None, tol_ms: float = TOL_SG_MS):
    """S -> G assignment. `identity` is the exact construction identity when one exists (synthetic
    perturbations of a known schedule); otherwise the frozen monotonic one-to-one DP assignment is used.

    Returns (pairs[(gt_index, sup_index)], n_unmatched_gt, n_unmatched_sup). Never modifies either schedule.
    """
    g = np.asarray(gt, dtype=np.int64)
    s = np.asarray(sup, dtype=np.int64)
    if identity is not None:
        pairs = [(int(b), int(a)) for a, b in np.asarray(identity, dtype=np.int64).reshape(-1, 2)]
        pairs.sort()
        return pairs, int(g.size - len(pairs)), int(s.size - len(pairs))
    m, ur, up = E1.dp_match(g, s, FS, tol_ms)
    return [(int(i), int(j)) for i, j in m], int(ur), int(up)


def chain_generated_to_supplied(sup, gen, tol_ms: float = TOL_PS_MS):
    """P -> S with the exact frozen greedy one-to-one matcher. Returns [(sup_index, gen_index)]."""
    m, _fp, _fn = R.match_rpeaks(np.asarray(sup, dtype=np.int64), np.asarray(gen, dtype=np.int64), FS, tol_ms)
    return [(int(i), int(j)) for i, j in m]


# --------------------------------------------------------------------------------- AXIS A
def classify_topology(gt, sup, pairs, n_unmatched_gt: int, n_unmatched_sup: int) -> dict:
    """A1-A6. T0 needs an exact count AND a complete one-to-one assignment within the S->G tolerance."""
    k, m = int(np.asarray(gt).size), int(np.asarray(sup).size)
    if m < k:
        cls = T2
    elif m > k:
        cls = T3
    elif n_unmatched_gt == 0 and n_unmatched_sup == 0:
        cls = T0
    else:
        cls = T1
    return {"K": k, "M": m, "A1_abs_count_error": abs(m - k), "A2_beats_ratio_dev": abs(m / max(k, 1) - 1.0),
            "A3_missing_fraction": n_unmatched_gt / max(k, 1), "A4_spurious_fraction": n_unmatched_sup / max(k, 1),
            "A6_topology_class": cls, "n_identity_matched": len(pairs)}


# --------------------------------------------------------------------------------- AXIS B
def placement_metrics(gt, sup, pairs) -> dict:
    """B1-B4 on identity-matched S->G pairs. Coverage is reported by the caller alongside these."""
    g = np.asarray(gt, dtype=np.int64)
    s = np.asarray(sup, dtype=np.int64)
    if not pairs:
        return {"B_n_pairs": 0, "B1_median_ae_ms": np.nan, "B2_mae_ms": np.nan,
                "B3_p90_ae_ms": np.nan, "B4_p95_ae_ms": np.nan, "B_signed_mean_ms": np.nan}
    e = np.asarray([int(s[j]) - int(g[i]) for i, j in pairs], dtype=np.float64) / FS * 1000.0
    a = np.abs(e)
    return {"B_n_pairs": int(a.size), "B1_median_ae_ms": float(np.median(a)), "B2_mae_ms": float(np.mean(a)),
            "B3_p90_ae_ms": float(np.percentile(a, 90)), "B4_p95_ae_ms": float(np.percentile(a, 95)),
            "B_signed_mean_ms": float(np.mean(e))}


# --------------------------------------------------------------------------------- joint event fidelity
def joint_event_fidelity(ref, pred, tols_ms=TOLS_PG_MS, prefix: str = "") -> dict:
    """F1 family. JOINT EVENT FIDELITY: existence, count and placement together. Never 'timing accuracy'."""
    a = np.asarray(ref, dtype=np.int64)
    b = np.asarray(pred, dtype=np.int64)
    out = {}
    for tol in tols_ms:
        m, fp, fn = R.match_rpeaks(a, b, FS, tol)
        p, r, f1 = R.prf(len(m), fp, fn)
        out[f"{prefix}F1_{int(tol)}"] = float(f1)
        if tol == 50.0:
            out[f"{prefix}PREC"] = float(p)
            out[f"{prefix}REC"] = float(r)
    return out


# --------------------------------------------------------------------------------- generator adherence
def adherence_metrics(sup, gen) -> dict:
    """AD_* on P -> S. HIGH ADHERENCE is descriptive only and never a success criterion."""
    s = np.asarray(sup, dtype=np.int64)
    p = np.asarray(gen, dtype=np.int64)
    n = max(int(s.size), 1)
    out = {}
    for tol in (50.0, 100.0):
        m, fp, fn = R.match_rpeaks(s, p, FS, tol)
        _pr, _rc, f1 = R.prf(len(m), fp, fn)
        out[f"AD_F1_{int(tol)}"] = float(f1)
        if tol == 50.0:
            out["AD_MISS"] = fn / n
            out["AD_SPUR"] = fp / n
            err = np.asarray([abs(int(p[j]) - int(s[i])) for i, j in m], dtype=np.float64) / FS * 1000.0
            out["AD_MAE"] = float(np.mean(err)) if err.size else np.nan
            out["AD_n_matched"] = int(err.size)
    return out


def adherence_label(f1_at_50: float) -> str:
    return "HIGH ADHERENCE" if np.isfinite(f1_at_50) and f1_at_50 >= HIGH_ADHERENCE else "-"


# --------------------------------------------------------------------------------- AXIS C and JOINT
def eligible(centre: int, n_time: int = T_LEN) -> bool:
    return E1.eligible(centre, n_time)


def _shape(gen_sig, gt_sig, c_gen: int, c_gt: int) -> dict | None:
    return E1.beat_shape(gen_sig, gt_sig, int(c_gen), int(c_gt), _TARGET_IQR)


def own_center_morphology(gen_sig, gt_sig, c_gen: int, c_gt: int) -> dict | None:
    """C1-C8: generated beat centred on its OWN detected event, GT beat on the GT event. No shift search."""
    s = _shape(gen_sig, gt_sig, c_gen, c_gt)
    if s is None:
        return None
    return {"C1_own_T4": s["nAE_T4"], "C2_own_T6": s["nAE_T6"], "C3_own_T7": s["nAE_T7"],
            "C4_own_T8": s["nAE_T8"], "C5_own_local_raw_rmse": s["local_raw_rmse"],
            "C6_own_local_deriv_rmse": s["local_deriv_rmse"],
            "C7_own_local_curvature_err": s["local_curvature_err"], "C8_own_local_corr": s["local_corr"]}


def gt_anchored_joint_structure(gen_sig, gt_sig, c_gt: int) -> dict | None:
    """J1-J4: BOTH windows centred on the GT event. GT-ANCHORED JOINT STRUCTURE -- shape AND placement."""
    s = _shape(gen_sig, gt_sig, c_gt, c_gt)
    if s is None:
        return None
    return {"J1_gt_local_raw_rmse": s["local_raw_rmse"], "J2_gt_local_deriv_rmse": s["local_deriv_rmse"],
            "J3_gt_local_curvature_err": s["local_curvature_err"], "J4_gt_local_corr": s["local_corr"],
            "_gt_T4": s["nAE_T4"], "_gt_T6": s["nAE_T6"], "_gt_T7": s["nAE_T7"], "_gt_T8": s["nAE_T8"]}


ALIGNMENT_PAIRS = (("D1_raw_rmse", "J1_gt_local_raw_rmse", "C5_own_local_raw_rmse"),
                   ("D2_deriv_rmse", "J2_gt_local_deriv_rmse", "C6_own_local_deriv_rmse"),
                   ("D3_curvature_err", "J3_gt_local_curvature_err", "C7_own_local_curvature_err"))


def alignment_sensitivity(own: dict, joint: dict) -> dict:
    """D1-D3 = GT-anchored minus own-centre, SAME FUNCTIONAL ONLY.

    Cross-functional differences (for example own-centre T6 against GT-anchored derivative RMSE) are
    prohibited by the contract and are not computable through this function.
    """
    return {d: float(joint[j] - own[o]) for d, j, o in ALIGNMENT_PAIRS}


# --------------------------------------------------------------------------------- aggregation
def aggregate_window_metrics(beat_rows: list, keys) -> dict:
    """Per-beat -> window MEDIAN, the first step of the frozen aggregation order."""
    return E1.window_median(beat_rows, list(keys))


BEAT_KEYS = ("C1_own_T4", "C2_own_T6", "C3_own_T7", "C4_own_T8", "C5_own_local_raw_rmse",
             "C6_own_local_deriv_rmse", "C7_own_local_curvature_err", "C8_own_local_corr",
             "J1_gt_local_raw_rmse", "J2_gt_local_deriv_rmse", "J3_gt_local_curvature_err",
             "J4_gt_local_corr", "_gt_T4", "_gt_T6", "_gt_T7", "_gt_T8",
             "D1_raw_rmse", "D2_deriv_rmse", "D3_curvature_err", "gen_to_gt_ae_ms", "gen_to_sup_ae_ms")


def coverage_block(n_sup: int, n_gt: int, n_gen: int, n_identity: int, n_chained: int, n_eligible: int) -> dict:
    """The five coverage quantities the contract requires beside every morphology number."""
    return {"COV_C1_schedule_to_gt_identity": n_identity / max(n_sup, 1),
            "COV_C2_generated_to_supplied_adherence": n_chained / max(n_identity, 1),
            "COV_C3_full_chain": n_eligible / max(n_sup, 1),
            "COV_C4_gt_beats_excluded": 1.0 - n_eligible / max(n_gt, 1),
            "COV_C5_generated_beats_excluded": 1.0 - n_eligible / max(n_gen, 1),
            "n_supplied": int(n_sup), "n_gt": int(n_gt), "n_generated": int(n_gen),
            "n_identity": int(n_identity), "n_chained": int(n_chained), "n_eligible": int(n_eligible)}


def pooled_coverage(window_rows) -> dict:
    """Cohort-level coverage: sum the counts over windows, then divide once.

    A coverage figure is a count ratio, not a per-beat value, so the frozen aggregation order
    ("unless a metric definition itself specifies otherwise") does not apply to it. The per-window
    COV_* fields remain available for window-level analysis.
    """
    n = {k: int(sum(int(r[k]) for r in window_rows)) for k in
         ("n_supplied", "n_gt", "n_generated", "n_identity", "n_chained", "n_eligible")}
    return {"COV_C1_schedule_to_gt_identity": n["n_identity"] / max(n["n_supplied"], 1),
            "COV_C2_generated_to_supplied_adherence": n["n_chained"] / max(n["n_identity"], 1),
            "COV_C3_full_chain": n["n_eligible"] / max(n["n_supplied"], 1),
            "COV_C4_gt_beats_excluded": 1.0 - n["n_eligible"] / max(n["n_gt"], 1),
            "COV_C5_generated_beats_excluded": 1.0 - n["n_eligible"] / max(n["n_generated"], 1), **n}


def exact_set_summary(window_rows, subjects=None) -> dict:
    """A5 and the T0-only placement metrics B5/B6, so no future experiment has to re-implement them."""
    rows = list(window_rows)
    t0 = [r for r in rows if r["A6_topology_class"] == T0]
    counts = {c: sum(1 for r in rows if r["A6_topology_class"] == c) for c in (T0, T1, T2, T3)}
    def _agg(key):
        if not t0:
            return np.nan
        v = np.asarray([r[key] for r in t0], dtype=np.float64)
        if subjects is None:
            return float(np.nanmean(v))
        sub = np.asarray(subjects)[[r["row"] for r in t0]]
        return float(np.mean([np.nanmean(v[sub == u]) for u in sorted(set(sub.tolist()))]))
    return {**counts, "A5_exact_set_fraction": counts[T0] / max(len(rows), 1),
            "B5_exact_set_mae_ms": _agg("B2_mae_ms"), "B6_exact_set_p90_ae_ms": _agg("B3_p90_ae_ms"),
            "B5_n_T0_windows": len(t0)}


def window_alignment_sensitivity(window_row: dict) -> dict:
    """Window-level same-functional sensitivity, median(J) - median(C).

    SECONDARY. The contract defines D1-D3 as PER-BEAT differences (contract_v1 metrics D1/D2/D3), which are
    then aggregated by the frozen order, so `median(J - C)` is the contracted quantity and this is reported
    beside it only because the two are not identical.
    """
    return {f"W{d}": float(window_row[j] - window_row[o]) for d, j, o in ALIGNMENT_PAIRS}


def apply_contract(gt, sup, gen_sig, gt_sig, gen_peaks, identity=None) -> tuple[dict, list]:
    """The complete frozen per-window block for ONE window. Returns (window row, per-beat rows).

    No arm name, experiment name or success threshold appears here: the contract measures, it does not judge.
    """
    g = np.asarray(gt, dtype=np.int64)
    s = np.asarray(sup, dtype=np.int64)
    p = np.asarray(gen_peaks, dtype=np.int64)
    pairs, n_ur, n_up = assign_schedule_to_gt(g, s, identity)
    topo = classify_topology(g, s, pairs, n_ur, n_up)
    place = placement_metrics(g, s, pairs)
    ident = {j: i for i, j in pairs}                       # supplied index -> GT index
    beats, n_chain = [], 0
    for si, pj in chain_generated_to_supplied(s, p):
        if si not in ident:
            continue                                        # a supplied event with no GT identity
        n_chain += 1
        gp = int(g[ident[si]])
        own = own_center_morphology(gen_sig, gt_sig, int(p[pj]), gp)
        joint = gt_anchored_joint_structure(gen_sig, gt_sig, gp)
        if own is None or joint is None:
            continue
        beats.append({"supplied_index": int(si), "gen_index": int(pj), "gt_index": int(ident[si]),
                      "gen_pos": int(p[pj]), "sup_pos": int(s[si]), "gt_pos": gp,
                      "gen_to_gt_ae_ms": abs(int(p[pj]) - gp) / FS * 1000.0,
                      "gen_to_sup_ae_ms": abs(int(p[pj]) - int(s[si])) / FS * 1000.0,
                      **own, **joint, **alignment_sensitivity(own, joint)})
    agg = aggregate_window_metrics(beats, BEAT_KEYS)
    row = {**topo, **place, **agg, **window_alignment_sensitivity(agg),
           **joint_event_fidelity(g, s, TOLS_PG_MS, prefix="SG_"),
           **joint_event_fidelity(g, p, TOLS_PG_MS, prefix="PG_"),
           **adherence_metrics(s, p),
           **coverage_block(int(s.size), int(g.size), int(p.size), len(ident), n_chain, len(beats))}
    return row, beats


# --------------------------------------------------------------------------------- acceptance tree
VERDICT_ACCEPTED = "EVENT-GEOMETRY EVALUATION CONTRACT ACCEPTED"
VERDICT_INVALID = "EVALUATION CONTRACT INVALID"
VERDICT_INCOMPLETE = "CONTRACT INCOMPLETE FOR NATURAL SCHEDULES"
GATE_IDS = tuple(f"V{i}" for i in range(1, 14))


def decide_contract(gates: dict) -> dict:
    """Preregistration section 20. V1-V12 are the controlled synthetic validation; V13 is natural-schedule
    completeness. A missing axis makes V13 false and the contract cannot be ACCEPTED."""
    missing = [g for g in GATE_IDS if g not in gates]
    if missing:
        raise KeyError(f"missing validation gates: {missing}")
    core = all(bool(gates[f"V{i}"]) for i in range(1, 13))
    if core and bool(gates["V13"]):
        v = VERDICT_ACCEPTED
    elif core:
        v = VERDICT_INCOMPLETE
    else:
        v = VERDICT_INVALID
    return {"verdict": v, "gates": {g: bool(gates[g]) for g in GATE_IDS},
            "core_synthetic_gates_pass": core,
            "licenses": ("a NEW preregistered predictor experiment" if v == VERDICT_ACCEPTED
                         else "nothing; do not train a predictor")}
