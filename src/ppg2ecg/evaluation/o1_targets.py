"""O1 — ECG component targets, frozen probe family, baselines and classification logic.

Frozen by docs/O1_ECG_COMPONENT_EXTRACTABILITY_PREREGISTRATION.md. DIAGNOSTIC ONLY: nothing here trains or
modifies a generator. Every target is derived with the project's existing frozen primitives (the R1/Q1/M1
R-peak detector, the M1 QRS-core geometry and derivatives, the frozen QRS-width and HF-fraction functions);
no new ECG delineator is introduced. ECG is a TRAINING LABEL only and never enters a probe input.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn

from ppg2ecg.evaluation import m1_structural as M1
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.metrics import hf_energy_ratio

FS, T_LEN = 128, 1024

# ------------------------------------------------------------------------------------------------ targets
TARGETS = ("beat_count", "median_RR_ms", "RR_IQR_ms", "median_QRS_p2p", "median_QRS_energy",
           "median_QRS_max_abs_derivative", "median_QRS_curvature_energy", "median_QRS_width_ms",
           "ECG_HF_fraction")
TARGET_IDS = {"beat_count": "T1", "median_RR_ms": "T2", "RR_IQR_ms": "T3", "median_QRS_p2p": "T4",
              "median_QRS_energy": "T5", "median_QRS_max_abs_derivative": "T6",
              "median_QRS_curvature_energy": "T7", "median_QRS_width_ms": "T8", "ECG_HF_fraction": "T9"}
RHYTHM_TARGETS = ("beat_count", "median_RR_ms", "RR_IQR_ms")
POSITIVE_CONTROLS = ("beat_count", "median_RR_ms")
UNITS = {"beat_count": "beats / 8 s", "median_RR_ms": "ms", "RR_IQR_ms": "ms",
         "median_QRS_p2p": "normalised ECG amplitude", "median_QRS_energy": "normalised ECG amplitude^2 (sum over 21 samples)",
         "median_QRS_max_abs_derivative": "normalised amplitude / sample", "median_QRS_curvature_energy": "normalised amplitude^2 / sample^4",
         "median_QRS_width_ms": "ms", "ECG_HF_fraction": "fraction of power >= 15 Hz"}


def window_targets(y: np.ndarray, fs: int = FS) -> dict:
    """All O1 scalar targets of ONE ground-truth ECG window, using the frozen primitives only.

    Per-beat quantities are aggregated with the MEDIAN over valid GT beats (preregistration section 5).
    A target is NaN when it is undefined (too few beats / no beat whose QRS core fits in the window).
    """
    y = np.asarray(y, dtype=np.float64)
    pk = R.detect_rpeaks(y, fs)                                   # frozen R1/Q1/M1 detector
    out = {t: np.nan for t in TARGETS}
    out["beat_count"] = float(len(pk))
    if len(pk) >= 2:
        rr = np.diff(np.asarray(pk, dtype=np.float64)) / fs * 1000.0
        out["median_RR_ms"] = float(np.median(rr))
        if len(rr) >= 2:
            q1, q3 = np.percentile(rr, [25, 75])
            out["RR_IQR_ms"] = float(q3 - q1)
    p2p, energy, mderiv, curv, width = [], [], [], [], []
    for r in np.asarray(pk, dtype=int):
        lo, hi = r - M1.CORE, r + M1.CORE                          # |tau| <= 80 ms  (M1.CORE = 10)
        if lo - 1 < 0 or hi + 2 > y.size:                          # M1 beat-validity rule, verbatim
            continue
        g = y[lo:hi + 1]                                           # 21 samples
        dg = M1.d1(y[lo - 1:hi + 2])                               # 22 first differences
        cg = M1.d2(y[lo - 1:hi + 2])                               # 21 second differences
        p2p.append(float(np.ptp(g)))
        energy.append(float(np.sum(g ** 2)))
        mderiv.append(float(np.abs(dg).max()))
        curv.append(float(np.mean(cg ** 2)))
        w = R.qrs_width_ms(y, int(r), fs)                          # frozen width implementation
        if np.isfinite(w):
            width.append(float(w))
    if p2p:
        out["median_QRS_p2p"] = float(np.median(p2p))
        out["median_QRS_energy"] = float(np.median(energy))
        out["median_QRS_max_abs_derivative"] = float(np.median(mderiv))
        out["median_QRS_curvature_energy"] = float(np.median(curv))
    if width:
        out["median_QRS_width_ms"] = float(np.median(width))
    out["ECG_HF_fraction"] = float(hf_energy_ratio(y[None], fs)[0])
    out["_rpeaks"] = np.asarray(pk, dtype=int)
    out["n_valid_qrs_beats"] = int(len(p2p))
    return out


# ------------------------------------------------------------------------------------------------ probe
CH, K, N_BLOCKS = 64, 5, 8
DILATIONS = tuple(2 ** i for i in range(N_BLOCKS))                 # 1..128, RF 2041 >= 1024
MAX_PARAMS = 500_000
SEEDS = (40, 42, 44)
LR, WEIGHT_DECAY, BATCH, MAX_EPOCHS, PATIENCE, HUBER_BETA = 1e-3, 1e-4, 128, 30, 5, 1.0
PREFLIGHT_STEPS = 100
BUDGET_GPU_HOURS = 4.0
BOOT_N, BOOT_SEED = 2000, 20260903
SS_SALT, XS_SALT = "o1-same-subject-shuffle-v1", "o1-cross-subject-shuffle-v1"
RIDGE_ALPHA = 1.0
MIN_VALID_FRACTION = 0.90
SKILL_R_STRONG, RHO_STRONG = 0.10, 0.30


class _Block(nn.Module):
    def __init__(self, ch: int, k: int, d: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding="same", dilation=d)
        self.c2 = nn.Conv1d(ch, ch, k, padding="same", dilation=d)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.c1(x))
        h = self.c2(h)
        return self.act(x + h)


class ComponentGlobalTCN(nn.Module):
    """R1's Global-TCN trunk + global temporal mean-pooling + a scalar head. PPG only, one output.

    The architecture is identical for every target; only the regression target differs.
    """

    def __init__(self, dilations=DILATIONS, ch: int = CH, k: int = K):
        super().__init__()
        self.stem = nn.Conv1d(1, ch, kernel_size=1)
        self.blocks = nn.Sequential(*[_Block(ch, k, int(d)) for d in dilations])
        self.head = nn.Linear(ch, 1)
        self.dilations = tuple(int(d) for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B,1,T] -> [B]
        h = self.blocks(self.stem(x))
        return self.head(h.mean(dim=2)).squeeze(-1)


def receptive_field(dilations=DILATIONS, k: int = K, convs_per_block: int = 2) -> int:
    return 1 + convs_per_block * (k - 1) * int(sum(dilations))


def build_probe(seed: int) -> ComponentGlobalTCN:
    torch.manual_seed(int(seed))
    return ComponentGlobalTCN()


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


# ------------------------------------------------------------------------------------------------ shuffles
def _rank_key(salt: str, subject: str, site: str, window_index: int) -> str:
    return hashlib.sha256(f"{salt}|{subject}|{site}|{int(window_index)}".encode()).hexdigest()


def same_subject_shuffle(subjects, sites, window_index, salt: str = SS_SALT) -> np.ndarray:
    """Deterministic fixed-point-free derangement WITHIN each (subject, site) stratum."""
    subjects, sites, window_index = np.asarray(subjects), np.asarray(sites), np.asarray(window_index)
    partner = np.full(len(subjects), -1, dtype=np.int64)
    for sub in np.unique(subjects):
        for site in np.unique(sites[subjects == sub]):
            m = np.flatnonzero((subjects == sub) & (sites == site))
            if m.size < 2:
                raise RuntimeError(f"SS-SHUFFLE undefined for stratum ({sub}, {site})")
            keys = [_rank_key(salt, str(sub), str(site), int(window_index[i])) for i in m]
            if len(set(keys)) != len(keys):
                raise RuntimeError(f"duplicate hash key inside ({sub}, {site})")
            order = m[np.argsort(keys, kind="stable")]
            partner[order] = np.roll(order, -1)
    if np.any(partner < 0):
        raise RuntimeError("SS-SHUFFLE incomplete")
    return partner


def cross_subject_shuffle(subjects, sites, window_index, salt: str = XS_SALT) -> np.ndarray:
    """Deterministic map to the OTHER validation subject, SAME site: rank both strata and pair by rank."""
    subjects, sites, window_index = np.asarray(subjects), np.asarray(sites), np.asarray(window_index)
    subs = sorted(set(subjects.tolist()))
    if len(subs) != 2:
        raise RuntimeError(f"XS-SHUFFLE needs exactly two subjects, got {subs}")
    partner = np.full(len(subjects), -1, dtype=np.int64)
    for site in np.unique(sites):
        idx = {}
        for sub in subs:
            m = np.flatnonzero((subjects == sub) & (sites == site))
            keys = [_rank_key(salt, str(sub), str(site), int(window_index[i])) for i in m]
            idx[sub] = m[np.argsort(keys, kind="stable")]
        a, b = idx[subs[0]], idx[subs[1]]
        n = min(len(a), len(b))
        if n == 0:
            raise RuntimeError(f"XS-SHUFFLE undefined for site {site}")
        partner[a[:n]] = b[:n]
        partner[b[:n]] = a[:n]
        for extra, other in ((a[n:], b), (b[n:], a)):             # ranks beyond the shorter stratum wrap
            for j, i in enumerate(extra):
                partner[i] = other[j % len(other)]
    if np.any(partner < 0):
        raise RuntimeError("XS-SHUFFLE incomplete")
    return partner


def assert_derangement(partner: np.ndarray) -> None:
    p = np.asarray(partner)
    if sorted(p.tolist()) != list(range(len(p))):
        raise RuntimeError("shuffle partner is not a bijection")
    if np.any(p == np.arange(len(p))):
        raise RuntimeError("shuffle partner has a fixed point")


def assert_cross_subject(partner: np.ndarray, subjects, sites) -> None:
    subjects, sites = np.asarray(subjects), np.asarray(sites)
    p = np.asarray(partner)
    if np.any(subjects[p] == subjects):
        raise RuntimeError("XS-SHUFFLE partner keeps the subject")
    if np.any(sites[p] != sites):
        raise RuntimeError("XS-SHUFFLE partner changes the site")


# ------------------------------------------------------------------------------------------------ skill + classification
def skill(mae_true: float, mae_ref: float) -> float:
    return float(1.0 - mae_true / mae_ref) if np.isfinite(mae_ref) and mae_ref > 0 else np.nan


CLASS_A = "STRONG WINDOW-SPECIFIC EXTRACTABILITY"
CLASS_B = "PARTIAL WINDOW-SPECIFIC EXTRACTABILITY"
CLASS_C = "RHYTHM / STATIC EXPLAINED"
CLASS_D = "NO CLEAR EXTRACTABILITY UNDER THIS PROBE"


def classify_component(skill_r: float, ss_ci_lo: float, rho_median: float,
                       beats_b0_all_seeds: bool, beats_b0b1: bool, beats_b2: bool) -> dict:
    """Preregistration section 19, evaluated in the frozen order A -> B -> C -> D."""
    ss_positive = bool(np.isfinite(ss_ci_lo) and ss_ci_lo > 0)
    strong = bool(np.isfinite(skill_r) and skill_r >= SKILL_R_STRONG) and ss_positive and \
        bool(np.isfinite(rho_median) and rho_median >= RHO_STRONG) and bool(beats_b0_all_seeds)
    partial = ss_positive and bool(np.isfinite(skill_r) and skill_r > 0)
    if strong:
        cls = CLASS_A
    elif partial:
        cls = CLASS_B
    elif bool(beats_b0b1) and ((not beats_b2) or (not ss_positive)):
        cls = CLASS_C
    else:
        cls = CLASS_D
    return {"class": cls, "skill_r": skill_r, "ss_ci_lo": ss_ci_lo, "rho_median": rho_median,
            "beats_b0_all_seeds": bool(beats_b0_all_seeds), "beats_b0_or_b1": bool(beats_b0b1),
            "beats_b2": bool(beats_b2), "ss_shuffle_ci_positive": ss_positive}


VERDICT_A = "COMPONENT-WISE EXTRACTABILITY HETEROGENEITY SUPPORTED"
VERDICT_B = "EXTRACTABILITY DOMINATED BY RHYTHM / STATIC INFORMATION"
VERDICT_C = "BROAD ECG-COMPONENT EXTRACTABILITY OBSERVED"
VERDICT_D = "EXTRACTABILITY MAP INCONCLUSIVE"


def decide_o1(classes: dict, positive_control_ok: bool, primary: tuple, morphology_ok: dict) -> dict:
    """Preregistration section 25. `classes` maps primary target -> class string."""
    high = [t for t in primary if classes.get(t) in (CLASS_A, CLASS_B)]
    low = [t for t in primary if classes.get(t) in (CLASS_C, CLASS_D)]
    morph = [t for t in primary if t not in RHYTHM_TARGETS]
    morph_high = [t for t in morph if classes.get(t) in (CLASS_A, CLASS_B)]
    morph_low = [t for t in morph if classes.get(t) in (CLASS_C, CLASS_D)]
    rhythm_high = [t for t in primary if t in RHYTHM_TARGETS and classes.get(t) in (CLASS_A, CLASS_B)]
    if not positive_control_ok:
        v = VERDICT_D
    elif len(morph_high) >= max(1, len(morph) - 1) and len(morph_high) > len(morph_low):
        v = VERDICT_C
    elif len(high) >= 2 and len(low) >= 2 and all(morphology_ok.get(t, True) for t in high):
        v = VERDICT_A
    elif len(rhythm_high) >= 1 and len(morph_low) > len(morph_high):
        v = VERDICT_B
    else:
        v = VERDICT_D
    return {"verdict": v, "high": high, "low": low, "morphology_high": morph_high,
            "morphology_low": morph_low, "rhythm_high": rhythm_high,
            "positive_control_ok": bool(positive_control_ok)}
