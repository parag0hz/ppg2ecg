"""O3 — supplied-schedule perturbations, prechecks, schedule-quality metrics and the frozen decision logic.

Frozen by docs/O3_SCHEDULE_ERROR_TOLERANCE_PREREGISTRATION.md.

NO TRAINING, NO WEIGHT UPDATE, NO NEW PREDICTOR anywhere in O3. Every synthetic schedule is derived from the
GT ECG R schedule, so all synthetic arms remain ORACLE DIAGNOSTICS. The O2b integer-grid operator is imported
unchanged; nothing here re-implements the schedule, the anchors, the rounding, the inverse or the resampler.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ppg2ecg.evaluation import m1_structural as M1
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import o2_warp as O2
from ppg2ecg.evaluation import o2b_warp as BW
from ppg2ecg.evaluation import rpeaks as R

FS, T_LEN = O2.FS, O2.T_LEN
MIN_INT_SPACING = BW.MIN_INT_SPACING                 # 21, imported
CORE_OFFSET_TOL = BW.CORE_OFFSET_TOL                 # 1e-6, imported
NONINF_MARGIN = O2.NONINF_MARGIN                     # 0.020, imported unchanged
F1_EXCESS_MIN = O2.F1_EXCESS_MIN                     # 0.10, imported unchanged

JITTER_LEVELS = (0, 1, 2, 4, 6, 8)
MISS_LEVELS = (0, 1, 2)
EXTRA_LEVELS = (0, 1, 2)
REPS = (0, 1, 2)
TOLS_MS = (50.0, 100.0, 150.0, 200.0)
NFE, SRC_SEED = 4, 0
BOOT_N, BOOT_SEED = 2000, 20260904
R1_THRESHOLD = 0.35
PREFLIGHT_WINDOWS, BUDGET_GPU_HOURS = 100, 2.0
JITTER_SALT, MISS_SALT, EXTRA_SALT = "o3-jitter-v1", "o3-miss-v1", "o3-extra-v1"
ALIGNED = ("median_QRS_p2p", "median_QRS_max_abs_derivative", "median_QRS_curvature_energy", "median_QRS_width_ms")
MS_ARMS = ("B", "O2C-ORACLE", "JITTER_4", "JITTER_8", "MISS1", "EXTRA1", "R1-SCHEDULE")


# ------------------------------------------------------------------------------------------ deterministic hash
def _u64(key: str) -> int:
    """First 8 bytes of SHA256 as a big-endian unsigned integer. No global RNG is ever consulted."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _row_key(salt: str, rep: int, subject: str, site: str, wi: int) -> str:
    return f"{salt}|{rep}|{subject}|{site}|{wi}"


# ------------------------------------------------------------------------------------------ family A: jitter
def jitter_schedule(r, rep: int, j: int, subject: str, site: str, wi: int, n_time: int = T_LEN) -> np.ndarray:
    """s_k = clip(r_k + delta_k, 0, n_time-1) with delta_k an integer uniform on {-J..+J} from a per-beat hash."""
    r = np.asarray(r, dtype=np.int64)
    base = _row_key(JITTER_SALT, rep, subject, site, wi)
    d = np.array([_u64(f"{base}|{k}|{j}") % (2 * int(j) + 1) - int(j) for k in range(r.size)], dtype=np.int64)
    return np.clip(r + d, 0, n_time - 1).astype(np.int64)


# ------------------------------------------------------------------------------------------ family B: miss
def miss_interior_order(r, rep: int, subject: str, site: str, wi: int) -> np.ndarray:
    """Interior beat indices 1..K-2 ordered by SHA256 rank ascending, ties broken by index ascending."""
    k = int(np.asarray(r).size)
    base = _row_key(MISS_SALT, rep, subject, site, wi)
    idx = list(range(1, k - 1))
    return np.asarray(sorted(idx, key=lambda i: (_u64(f"{base}|{i}"), i)), dtype=np.int64)


def miss_schedule(r, rep: int, n_del: int, subject: str, site: str, wi: int) -> np.ndarray:
    r = np.asarray(r, dtype=np.int64)
    if n_del == 0:
        return r.copy()
    order = miss_interior_order(r, rep, subject, site, wi)
    if order.size < n_del:
        raise ValueError(f"window has {order.size} interior beats, MISS{n_del} needs {n_del}")
    drop = set(order[:n_del].tolist())
    return np.asarray([v for i, v in enumerate(r) if i not in drop], dtype=np.int64)


# ------------------------------------------------------------------------------------------ family C: extra
def extra_midpoint(a: int, b: int) -> int:
    return int(BW.round_half_to_even([(float(a) + float(b)) / 2.0])[0])


def extra_eligible(r) -> np.ndarray:
    """Interval indices i whose integer midpoint keeps BOTH new spacings >= MIN_INT_SPACING."""
    r = np.asarray(r, dtype=np.int64)
    out = []
    for i in range(r.size - 1):
        m = extra_midpoint(int(r[i]), int(r[i + 1]))
        if m - int(r[i]) >= MIN_INT_SPACING and int(r[i + 1]) - m >= MIN_INT_SPACING:
            out.append(i)
    return np.asarray(out, dtype=np.int64)


def extra_interval_order(r, rep: int, subject: str, site: str, wi: int) -> np.ndarray:
    base = _row_key(EXTRA_SALT, rep, subject, site, wi)
    elig = extra_eligible(r).tolist()
    return np.asarray(sorted(elig, key=lambda i: (_u64(f"{base}|{i}"), i)), dtype=np.int64)


def extra_schedule(r, rep: int, n_ins: int, subject: str, site: str, wi: int) -> np.ndarray:
    r = np.asarray(r, dtype=np.int64)
    if n_ins == 0:
        return r.copy()
    order = extra_interval_order(r, rep, subject, site, wi)
    if order.size < n_ins:
        raise ValueError(f"window has {order.size} eligible intervals, EXTRA{n_ins} needs {n_ins}")
    add = [extra_midpoint(int(r[i]), int(r[i + 1])) for i in order[:n_ins]]
    return np.sort(np.concatenate([r, np.asarray(add, dtype=np.int64)]))


def retained_pairs(family: str, level: int, rep: int, r, s, subject: str, site: str, wi: int) -> np.ndarray:
    """(index into S, index into GT R) for supplied events that keep their originating GT beat identity.

    JITTER keeps every event's identity, MISS keeps the retained events', EXTRA excludes inserted events.
    """
    r = np.asarray(r, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    if family == "JITTER":
        if s.size != r.size:
            return np.zeros((0, 2), dtype=np.int64)
        return np.stack([np.arange(s.size), np.arange(r.size)], axis=1).astype(np.int64)
    if family == "MISS":
        if level == 0:
            kept = np.arange(r.size, dtype=np.int64)
        else:
            drop = set(miss_interior_order(r, rep, subject, site, wi)[:level].tolist())
            kept = np.asarray([i for i in range(r.size) if i not in drop], dtype=np.int64)
        return np.stack([np.arange(kept.size, dtype=np.int64), kept], axis=1)
    if family == "EXTRA":
        pos = np.searchsorted(s, r)
        ok = (pos < s.size) & (s[np.clip(pos, 0, max(s.size - 1, 0))] == r)
        return np.stack([pos[ok], np.arange(r.size, dtype=np.int64)[ok]], axis=1).astype(np.int64)
    raise ValueError(family)


def supplied_schedule(family: str, level: int, rep: int, r, subject: str, site: str, wi: int) -> np.ndarray:
    if family == "JITTER":
        return jitter_schedule(r, rep, level, subject, site, wi)
    if family == "MISS":
        return miss_schedule(r, rep, level, subject, site, wi)
    if family == "EXTRA":
        return extra_schedule(r, rep, level, subject, site, wi)
    raise ValueError(family)


def condition_name(family: str, level: int) -> str:
    return "ORACLE" if level == 0 else f"{family}_{level}"


# ------------------------------------------------------------------------------------------ precheck
def precheck_schedule(s, n_time: int = T_LEN, require_non_identity: bool = True) -> dict:
    """Preregistration section 8. Returns per-check booleans plus the constructed warp's status."""
    s = np.asarray(s)
    integer = bool(np.issubdtype(s.dtype, np.integer) or np.all(np.equal(np.mod(s, 1), 0)))
    si = np.asarray(s, dtype=np.int64)
    finite = bool(np.all(np.isfinite(np.asarray(s, dtype=np.float64))))
    increasing = bool(si.size >= 2 and np.all(np.diff(si) > 0))
    in_bounds = bool(si.size > 0 and si.min() >= 0 and si.max() <= n_time - 1)
    spacing = int(np.diff(si).min()) if si.size >= 2 else -1
    spacing_ok = bool(si.size >= 2 and spacing >= MIN_INT_SPACING)
    enough = bool(si.size >= O2.MIN_BEATS)
    w = BW.IntegerEventWarp(si, n_time=n_time)
    warp_valid = bool(w.valid())
    non_identity = bool(not w.identity)
    core = float(w.core_offsets().max()) if non_identity else 0.0
    inv_mono = True
    if non_identity:
        tau = np.arange(n_time, dtype=np.float64)
        inv_mono = bool(np.all(np.diff(w.inverse(tau)) > 0))
    checks = {"integer": integer, "finite": finite, "increasing": increasing, "in_bounds": in_bounds,
              "spacing_ok": spacing_ok, "enough_beats": enough, "warp_valid": warp_valid,
              "inverse_monotone": inv_mono, "core_offset_ok": bool(core <= CORE_OFFSET_TOL)}
    if require_non_identity:
        checks["non_identity"] = non_identity
    return {"checks": checks, "passed": bool(all(checks.values())), "status": w.status, "M": int(si.size),
            "min_spacing": spacing, "max_core_offset": core, "identity": bool(w.identity)}


# ------------------------------------------------------------------------------------------ schedule quality
def schedule_quality(gt, s, fs: int = FS) -> dict:
    """Supplied schedule scored against GT R with the frozen one-to-one matcher. Not a generator metric."""
    gt = np.asarray(gt, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    n_ref = max(int(gt.size), 1)
    out = {"K": int(gt.size), "M": int(s.size), "beats_ratio": s.size / n_ref,
           "beats_ratio_dev": abs(s.size / n_ref - 1.0)}
    for tol in TOLS_MS:
        m, fp, fn = R.match_rpeaks(gt, s, fs, tol)
        p, rc, f1 = R.prf(len(m), fp, fn)
        out[f"f1_at_{int(tol)}"] = float(f1)
        if tol == 50.0:
            out["precision"] = float(p); out["recall"] = float(rc)
            out["missing"] = fn / n_ref; out["spurious"] = fp / n_ref
            err = np.asarray([abs(int(s[j]) - int(gt[i])) for i, j in m], dtype=np.float64) / fs * 1000.0
            out["matched_n"] = int(err.size)
            out["timing_median_ae_ms"] = float(np.median(err)) if err.size else np.nan
            out["timing_mae_ms"] = float(np.mean(err)) if err.size else np.nan
    return out


def adherence(supplied, generated_peaks, fs: int = FS) -> dict:
    """Generated events scored against the SUPPLIED schedule (diagnostic; no causal attribution)."""
    s = np.asarray(supplied, dtype=np.int64)
    g = np.asarray(generated_peaks, dtype=np.int64)
    n_ref = max(int(s.size), 1)
    out = {}
    for tol in (50.0, 100.0):
        m, fp, fn = R.match_rpeaks(s, g, fs, tol)
        _p, _r, f1 = R.prf(len(m), fp, fn)
        out[f"adherence_f1_at_{int(tol)}"] = float(f1)
        if tol == 50.0:
            out["adherence_missing"] = fn / n_ref
            out["adherence_spurious"] = fp / n_ref
    return out


# ------------------------------------------------------------------------------------------ per-beat primitives
def beat_primitives(y, centre: int, fs: int = FS) -> dict | None:
    """The exact O1 per-beat primitives (window_targets inner loop) evaluated at a SUPPLIED centre."""
    y = np.asarray(y, dtype=np.float64)
    r = int(centre)
    lo, hi = r - M1.CORE, r + M1.CORE
    if lo - 1 < 0 or hi + 2 > y.size:
        return None
    g = y[lo:hi + 1]
    dg = M1.d1(y[lo - 1:hi + 2])
    cg = M1.d2(y[lo - 1:hi + 2])
    w = R.qrs_width_ms(y, r, fs)
    return {"median_QRS_p2p": float(np.ptp(g)),
            "median_QRS_max_abs_derivative": float(np.abs(dg).max()),
            "median_QRS_curvature_energy": float(np.mean(cg ** 2)),
            "median_QRS_width_ms": float(w) if np.isfinite(w) else np.nan}


# ------------------------------------------------------------------------------------------ retention
def retention(cond: float, b: float, oracle: float, higher_better: bool) -> float:
    """Normalized performance-retention ratio. NOT an information-retention fraction. Never clipped."""
    denom = (oracle - b) if higher_better else (b - oracle)
    num = (cond - b) if higher_better else (b - cond)
    return float(num / denom) if denom != 0 else np.nan


# ------------------------------------------------------------------------------------------ gates and verdict
def joint_gates(res: dict) -> dict:
    """Preregistration section 15. `res` maps a metric to its paired-bootstrap dict (point/lo/hi/verdict)."""
    g = {"G1": bool(res["f1_excess"]["lo"] > 0 and res["f1_excess"]["point"] >= F1_EXCESS_MIN),
         "G2": bool(res["nAE_T6"]["lo"] > -NONINF_MARGIN),
         "G3": bool(res["nAE_T7"]["lo"] > -NONINF_MARGIN),
         "G4": bool(res["nAE_T6"]["verdict"] == "improves" or res["nAE_T7"]["verdict"] == "improves"),
         "G5": bool(res["qrs_deriv_rmse"]["verdict"] != "worsens" and res["qrs_curvature_err"]["verdict"] != "worsens"),
         "G6": bool(res["nAE_T4"]["verdict"] != "worsens" and res["nAE_T8"]["verdict"] != "worsens")}
    g["survives"] = bool(all(g[k] for k in ("G1", "G2", "G3", "G4", "G5", "G6")))
    return g


def bridge_gate(gates: dict, s1: dict, s2: dict) -> dict:
    """G7: source event stability of the R1 arm against B on the frozen 512-window / 8-source cohort."""
    g7a, g7b = bool(s1["lo"] > 0), bool(s2["lo"] > 0)
    return {**gates, "G7a": g7a, "G7b": g7b, "G7": bool(g7a and g7b),
            "supported": bool(gates["survives"] and g7a and g7b)}


def level_survives(rep_gates: list) -> bool:
    """A severity level survives only when ALL THREE perturbation replicates pass G1-G6."""
    return bool(len(rep_gates) == len(REPS) and all(bool(g["survives"]) for g in rep_gates))


def j_max(level_pass: dict) -> int | None:
    """Largest tested jitter level whose three replicates all survive. No interpolation between levels."""
    ok = [j for j in JITTER_LEVELS if level_pass.get(j)]
    return max(ok) if ok else None


VERDICT_REGRESSION = "FROZEN MODEL REGRESSION"
VERDICT_BRITTLE = "OPERATOR TOO BRITTLE TO ONE-SAMPLE SCHEDULE ERROR"
VERDICT_JITTER_INVALID = "JITTER PERTURBATION DESIGN INVALID"
VERDICT_EXTRA_INVALID = "EXTRA-BEAT PERTURBATION DESIGN INVALID"
VERDICT_A = "FROZEN R1 SCHEDULE BRIDGE SUPPORTED"
VERDICT_B = "SYNTHETIC TOLERANCE SUPPORTED, CURRENT R1 BRIDGE NOT SUPPORTED"
VERDICT_C = "TOLERANCE REGION TOO NARROW"
VERDICT_D = "NO ROBUST SCHEDULE-TOLERANCE REGION"


def decide_o3(level_pass: dict, r1_supported: bool, r1_precheck_ok: bool) -> dict:
    """Preregistration section 23. MISS/EXTRA never enter this tree; they characterise the failure mode."""
    if r1_precheck_ok and r1_supported:
        v = VERDICT_A
    elif any(bool(level_pass.get(j)) for j in (4, 6, 8)):
        v = VERDICT_B
    elif not bool(level_pass.get(2)):
        v = VERDICT_D
    else:
        v = VERDICT_C
    return {"verdict": v, "jitter_level_pass": {int(j): bool(level_pass.get(j)) for j in JITTER_LEVELS},
            "j_max": j_max(level_pass), "r1_precheck_ok": bool(r1_precheck_ok),
            "r1_supported": bool(r1_supported),
            "miss_extra_can_select_verdict": False}
