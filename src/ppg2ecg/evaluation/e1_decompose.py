"""E1 — event topology / placement / own-centre morphology decomposition.

Frozen by docs/E1_EVENT_PLACEMENT_MORPHOLOGY_DECOMPOSITION_PREREGISTRATION.md.

POST-O3 DIAGNOSTIC, designed after the O3 results were known: every threshold here is a frozen post-hoc
diagnostic criterion, not independent preregistered confirmation. NO TRAINING, NO NEW PREDICTOR, NO
THRESHOLD TUNING. All morphology primitives are the exact frozen O1/M1 ones, evaluated at a supplied centre.
"""
from __future__ import annotations

import numpy as np

from ppg2ecg.evaluation import m1_structural as M1
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.evaluation import rpeaks as R

FS, T_LEN = O3.FS, O3.T_LEN
TOL_IDENTITY_MS = 150.0                 # S -> G diagnostic assignment (R1 only)
TOL_CHAIN_MS = 50.0                     # P -> S, the exact frozen O3 adherence tolerance
WIN_LO, WIN_HI = -10, 15                # frozen own-centre support, spans the frozen QRS-width search
ELIG_LO, ELIG_HI = 11, 15               # a centre c is eligible iff c - 11 >= 0 and c + 15 <= T_LEN - 1
ALIGNED = O3.ALIGNED
NAE = {t: f"nAE_{OT.TARGET_IDS[t]}" for t in ALIGNED}
BOOT_N, BOOT_SEED = O3.BOOT_N, O3.BOOT_SEED          # 2000, 20260904
TIMING_BINS = ((0.0, 16.0, "A"), (16.0, 32.0, "B"), (32.0, float("inf"), "C"))
MIN_STRATUM_WINDOWS, MIN_STRATUM_PER_SUBJECT = 100, 30
ADHERENCE_HIGH = 0.90
COVERAGE_MIN = 0.80
PRIMARY_ARMS = ("B", "ORACLE", "JITTER_2", "JITTER_4", "JITTER_8", "MISS1", "EXTRA1", "R1-SCHEDULE")
SYNTHETIC_O2C_ARMS = ("ORACLE", "JITTER_2", "JITTER_4", "JITTER_8", "MISS1", "EXTRA1")


# ------------------------------------------------------------------------------- monotonic one-to-one matcher
def dp_match(ref, pred, fs: int = FS, tol_ms: float = TOL_IDENTITY_MS):
    """Monotonic one-to-one assignment. Frozen objective hierarchy (preregistration section 6):

    1. maximise the number of matches within `tol_ms`
    2. among equal-cardinality solutions, minimise the total absolute timing error
    3. deterministic tie break by transition preference match -> skip-ref -> skip-pred

    Returns (matches[(i_ref, j_pred)], n_unmatched_ref, n_unmatched_pred). Never modifies either schedule.
    """
    ref = np.asarray(ref, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    n, m = int(ref.size), int(pred.size)
    tol = tol_ms / 1000.0 * fs
    best = [[(0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            cands = []
            if i > 0 and j > 0:
                d = int(abs(int(ref[i - 1]) - int(pred[j - 1])))
                if d <= tol:
                    c, cost = best[i - 1][j - 1]
                    cands.append(((c + 1, -(cost + d)), "M", (c + 1, cost + d)))
            if i > 0:
                c, cost = best[i - 1][j]
                cands.append(((c, -cost), "R", (c, cost)))
            if j > 0:
                c, cost = best[i][j - 1]
                cands.append(((c, -cost), "P", (c, cost)))
            key = max(k for k, _o, _v in cands)
            for k, op, val in cands:                      # first in the frozen preference order wins a tie
                if k == key:
                    best[i][j], back[i][j] = val, op
                    break
    i, j, out = n, m, []
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif op == "R":
            i -= 1
        else:
            j -= 1
    out.reverse()
    return out, n - len(out), m - len(out)


# ------------------------------------------------------------------------------- axis A: topology
T0, T1, T2, T3 = "T0_EXACT_EVENT_SET", "T1_COUNT_CORRECT_SET_IMPERFECT", "T2_UNDERCOUNT", "T3_OVERCOUNT"


def topology(gt, sup, fs: int = FS) -> dict:
    """Preregistration section 9. Classification uses the monotonic +-150 ms assignment."""
    g = np.asarray(gt, dtype=np.int64)
    s = np.asarray(sup, dtype=np.int64)
    matches, n_ref_un, n_pred_un = dp_match(g, s, fs, TOL_IDENTITY_MS)
    k, m = int(g.size), int(s.size)
    if m < k:
        cls = T2
    elif m > k:
        cls = T3
    elif n_ref_un == 0 and n_pred_un == 0:
        cls = T0
    else:
        cls = T1
    return {"K": k, "M": m, "count_error": m - k, "abs_count_error": abs(m - k),
            "missing_events": n_ref_un, "extra_events": n_pred_un,
            "missing_fraction": n_ref_un / max(k, 1), "spurious_fraction": n_pred_un / max(k, 1),
            "n_identity_matched": len(matches), "topology_class": cls}


# ------------------------------------------------------------------------------- axis B: placement
def placement(gt, sup, pairs, fs: int = FS) -> dict:
    """Signed / absolute schedule timing error over identity-matched (gt_index, sup_index) pairs."""
    g = np.asarray(gt, dtype=np.int64)
    s = np.asarray(sup, dtype=np.int64)
    if len(pairs) == 0:
        return {"n_pairs": 0, "signed_mean_ms": np.nan, "median_ae_ms": np.nan, "mae_ms": np.nan,
                "p90_ae_ms": np.nan, "p95_ae_ms": np.nan}
    e = np.asarray([(int(s[j]) - int(g[i])) for i, j in pairs], dtype=np.float64) / fs * 1000.0
    a = np.abs(e)
    return {"n_pairs": int(a.size), "signed_mean_ms": float(np.mean(e)),
            "median_ae_ms": float(np.median(a)), "mae_ms": float(np.mean(a)),
            "p90_ae_ms": float(np.percentile(a, 90)), "p95_ae_ms": float(np.percentile(a, 95))}


# ------------------------------------------------------------------------------- axis C: own-centre morphology
def eligible(centre: int, n_time: int = T_LEN) -> bool:
    """Frozen eligibility: the [-11, +15] envelope of the own-centre support must fit inside the window."""
    c = int(centre)
    return bool(c - ELIG_LO >= 0 and c + ELIG_HI <= n_time - 1)


def _slices(sig, c: int):
    """(support [-10,+15], M1 extended slice [-11,+12)) around a centre; both inside the frozen envelope."""
    s = np.asarray(sig, dtype=np.float64)
    return s[c + WIN_LO:c + WIN_HI + 1], s[c - ELIG_LO:c + 12]


def beat_shape(gen, gt, c_gen: int, c_gt: int, iqr: dict) -> dict | None:
    """One paired beat compared in LOCAL coordinates: generated centred on c_gen, GT centred on c_gt.

    M1-M4 are the exact O1 per-beat primitives; W1-W4 are the frozen M1 local waveform functionals.
    No shift optimisation, no DTW, no amplitude scaling, no renormalisation.
    """
    if not (eligible(c_gen) and eligible(c_gt)):
        return None
    pg = O3.beat_primitives(gen, c_gen)
    pt = O3.beat_primitives(gt, c_gt)
    if pg is None or pt is None:
        return None
    out = {}
    for t in ALIGNED:
        a, b = pg[t], pt[t]
        out[NAE[t]] = float(abs(a - b) / iqr[t]) if np.isfinite(a) and np.isfinite(b) else np.nan
    wg, eg = _slices(gen, c_gen)
    wt, et = _slices(gt, c_gt)
    dg, dt = M1.d1(eg), M1.d1(et)
    cg, ct = M1.d2(eg), M1.d2(et)
    out["local_raw_rmse"] = float(np.sqrt(np.mean((wg - wt) ** 2)))
    out["local_deriv_rmse"] = float(np.sqrt(np.mean((dg - dt) ** 2)))
    out["local_curvature_err"] = float(np.mean(np.abs(cg - ct)))
    out["local_corr"] = (float(np.corrcoef(wg, wt)[0, 1])
                         if wg.std() > 1e-12 and wt.std() > 1e-12 else np.nan)
    return out


def window_median(beats: list, keys) -> dict:
    """Per-window MEDIAN over eligible beats (preregistration section 11)."""
    out = {}
    for k in keys:
        v = np.asarray([b[k] for b in beats if k in b], dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k] = float(np.median(v)) if v.size else np.nan
    return out


SHAPE_KEYS = tuple(NAE[t] for t in ALIGNED) + ("local_raw_rmse", "local_deriv_rmse", "local_curvature_err",
                                               "local_corr")


# ------------------------------------------------------------------------------- gates and verdict
def placement_gates(res: dict) -> dict:
    """P1-P4 (preregistration section 14). `res` maps a contrast name to a bootstrap dict."""
    g = {"P1": bool(res["gt_anchored_local_deriv_rmse_J4_vs_ORACLE"]["lo"] > 0),
         "P2": bool(res["gt_anchored_local_curvature_err_J4_vs_ORACLE"]["lo"] > 0),
         "P3": bool(res["T6_gt_minus_own_damage_J4"]["lo"] > 0),
         "P4": bool(res["T7_gt_minus_own_damage_J4"]["lo"] > 0)}
    g["supported"] = bool(all(g[k] for k in ("P1", "P2", "P3", "P4")))
    return g


def topology_gates(res: dict) -> dict:
    """C1-C5 (preregistration section 15)."""
    g = {"C1": bool(res["MISS1_damage_T6"]["lo"] > 0),
         "C2": bool(res["MISS1_damage_T7"]["lo"] > 0),
         "C3": bool(res["EXTRA1_damage_T6"]["lo"] > 0 or res["EXTRA1_damage_T7"]["lo"] > 0),
         "C4": bool(res["excess_MISS_T6"]["lo"] > 0 and res["excess_MISS_T7"]["lo"] > 0),
         "C5": bool(res["excess_EXTRA_T6"]["lo"] > 0 or res["excess_EXTRA_T7"]["lo"] > 0)}
    g["supported"] = bool(all(g[k] for k in ("C1", "C2", "C3", "C4", "C5")))
    return g


VERDICT_A = "EVENT-TOPOLOGY / COUNT PRIORITY SUPPORTED"
VERDICT_B = "FINE PLACEMENT PRIORITY SUPPORTED"
VERDICT_C = "MIXED TOPOLOGY AND PLACEMENT LIMITATION"
VERDICT_D = "DECOMPOSITION INCONCLUSIVE"


def decide_e1(pg: dict, tg: dict, res: dict, coverage_ok: bool, gates_ok: bool) -> dict:
    """Preregistration section 19. The synthetic contrasts alone select the verdict; R1 strata never do."""
    if not (coverage_ok and gates_ok):
        v = VERDICT_D
    elif pg["supported"] and tg["supported"]:
        v = VERDICT_A
    elif (not tg["C4"]
          and res["JITTER8_damage_T6"]["point"] >= res["MISS1_damage_T6"]["point"]
          and res["JITTER8_damage_T7"]["point"] >= res["MISS1_damage_T7"]["point"]
          and (res["JITTER8_damage_T6"]["lo"] > 0 or res["JITTER8_damage_T7"]["lo"] > 0)):
        v = VERDICT_B
    elif ((res["JITTER8_damage_T6"]["lo"] > 0 or res["JITTER8_damage_T7"]["lo"] > 0)
          and (res["MISS1_damage_T6"]["lo"] > 0 or res["MISS1_damage_T7"]["lo"] > 0)):
        v = VERDICT_C
    else:
        v = VERDICT_D
    return {"verdict": v, "placement_gates": pg, "topology_gates": tg,
            "coverage_precondition_ok": bool(coverage_ok), "integrity_gates_ok": bool(gates_ok),
            "selected_by": "synthetic intervention contrasts only",
            "r1_strata_can_override": False}


# ------------------------------------------------------------------------------- R1 strata helpers
def timing_bin(mae_ms: float) -> str | None:
    if not np.isfinite(mae_ms):
        return None
    for lo, hi, name in TIMING_BINS:
        if lo <= mae_ms <= hi if name == "A" else lo < mae_ms <= hi:
            return name
    return None


def stratum_sufficient(n_windows_total: int, per_subject: dict) -> bool:
    return bool(n_windows_total >= MIN_STRATUM_WINDOWS
                and per_subject and min(per_subject.values()) >= MIN_STRATUM_PER_SUBJECT)
