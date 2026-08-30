"""X3-G0 coupling cost-geometry primitives (docs/X3_G0_COUPLING_GEOMETRY_PREREGISTRATION.md).

ZERO-TRAINING analysis utilities: minibatch assignment under different target cost geometries, a cross-objective regret
manipulation check, and a cross-fitted linear source-to-residual dependence diagnostic.

Cost identity used throughout (sec. 7 of the pre-registration): for an exact one-to-one assignment
    sum_i ||x_i - y_pi(i)||^2 = sum_i ||x_i||^2 + sum_j ||y_j||^2 - 2 sum_i <x_i, y_pi(i)>
and the first two terms are permutation invariant, so every assignment here equals argmax_pi sum_i <x_i, y_pi(i)>.
Consequences: a single positive global scale on the target is an assignment NO-OP, and a large target norm does not by
itself dominate the assignment.

Nothing here is a generative model. The ridge map is a DIAGNOSTIC regression; its R^2 is
"source-to-residual linear predictive dependence", never mutual information / capacity / entropy.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

FS = 128
HF_CUT_HZ = 15.0          # X0's structural HF convention, reused for the cost/residual projections
QRS_HALF_MS = 100.0       # X0's frozen QRS half-width
PSD_FLOOR_REL = 1e-3      # frozen numerical floor, relative to median_{f>0} PSD_fit
TEST_SUBJECTS = ("kjd", "ssx")


class WildPPGTestFirewallError(RuntimeError):
    """Raised when X3-G0 code is asked to load a WildPPG test subject."""


def assert_no_test_subjects(subjects) -> None:
    """Fail loudly if any WildPPG test subject appears (pre-registration sec. 5)."""
    bad = sorted(set(map(str, subjects)) & set(TEST_SUBJECTS))
    if bad:
        raise WildPPGTestFirewallError(f"X3-G0 must never load WildPPG test subjects; got {bad}")


# ----------------------------------------------------------------------------------------------------------------------
# Target cost geometries (frozen transforms, pre-registration sec. 7)
# ----------------------------------------------------------------------------------------------------------------------
def demean(y: np.ndarray) -> np.ndarray:
    return y - y.mean(-1, keepdims=True)


def fit_whitener(y_fit: np.ndarray, fs: int = FS) -> np.ndarray:
    """Train-spectrum whitening weights w(f) from FIT-subject windows only.

    w(f) = 1/sqrt(max(PSD_fit(f), psd_floor)); w(0) = 0 (DC excluded); normalised so mean_{f>0} w = 1.
    The global normalisation is a conditioning convenience and cannot change an exact assignment.
    """
    Y = np.fft.rfft(demean(np.asarray(y_fit, dtype=np.float64)), axis=-1)
    psd = (np.abs(Y) ** 2).mean(0)
    floor = PSD_FLOOR_REL * np.median(psd[1:])
    w = 1.0 / np.sqrt(np.maximum(psd, floor))
    w[0] = 0.0
    return w / w[1:].mean()


def hf_mask(n_time: int, fs: int = FS, cut_hz: float = HF_CUT_HZ) -> np.ndarray:
    """Brick-wall rFFT mask, exactly f > cut_hz."""
    return (np.fft.rfftfreq(n_time, 1.0 / fs) > cut_hz).astype(np.float64)


def phi_raw(y: np.ndarray, **_) -> np.ndarray:
    return np.asarray(y, dtype=np.float64)


def phi_white(y: np.ndarray, w: np.ndarray, **_) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return np.fft.irfft(np.fft.rfft(demean(y), axis=-1) * w, n=y.shape[-1], axis=-1)


def phi_hf(y: np.ndarray, fs: int = FS, **_) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return np.fft.irfft(np.fft.rfft(demean(y), axis=-1) * hf_mask(y.shape[-1], fs), n=y.shape[-1], axis=-1)


def phi_resid(y: np.ndarray, m: np.ndarray, **_) -> np.ndarray:
    """SECONDARY arm: y - m_A6(c). No scalar rescaling (a positive global scale is an assignment no-op)."""
    return np.asarray(y, dtype=np.float64) - np.asarray(m, dtype=np.float64)


# ----------------------------------------------------------------------------------------------------------------------
# Assignment
# ----------------------------------------------------------------------------------------------------------------------
def assignment_cost(x0: np.ndarray, phi_y: np.ndarray, perm: np.ndarray) -> float:
    """Total squared-L2 assignment cost sum_i ||x0_i - phi_y_{perm[i]}||^2 (the full cost, not just the cross term)."""
    return float(((np.asarray(x0, dtype=np.float64) - np.asarray(phi_y, dtype=np.float64)[perm]) ** 2).sum())


def solve_assignment(x0: np.ndarray, phi_y: np.ndarray) -> np.ndarray:
    """Exact Hungarian assignment for squared-L2 cost; returns perm with source i matched to target perm[i].

    Uses the cross-term identity: argmin sum ||x - y_pi||^2 == argmax sum <x, y_pi>.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    phi_y = np.asarray(phi_y, dtype=np.float64)
    if len(x0) == 1:
        return np.zeros(1, dtype=int)
    row, col = linear_sum_assignment(-(x0 @ phi_y.T))
    assert np.array_equal(row, np.arange(len(x0)))
    return col


def random_permutations(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.permutation(n) for _ in range(k)]


def cross_objective_regret(x0: np.ndarray, phi_by_arm: dict, n_random: int, seed: int) -> dict:
    """Full regret matrix (pre-registration sec. 9).

    Regret(q <- p) = [Cost_q(pi_p) - Cost_q(pi_q)] / [E_rand Cost_q - Cost_q(pi_q) + eps].
    ~0 means pi_p is essentially optimal under q despite differing indices (near-tie churn);
    substantially positive means the geometries genuinely differ.
    """
    arms = list(phi_by_arm)
    perms = {a: solve_assignment(x0, phi_by_arm[a]) for a in arms}
    rand = random_permutations(len(x0), n_random, seed)
    out = {"perms": perms, "regret": {}, "overlap": {}}
    for q in arms:
        cq = assignment_cost(x0, phi_by_arm[q], perms[q])
        c_rand = float(np.mean([assignment_cost(x0, phi_by_arm[q], r) for r in rand]))
        denom = max(c_rand - cq, 1e-12)
        for p in arms:
            out["regret"][(q, p)] = (assignment_cost(x0, phi_by_arm[q], perms[p]) - cq) / denom
            out["overlap"][(q, p)] = float((perms[q] == perms[p]).mean())
        out.setdefault("cost", {})[q] = {"assigned": cq, "random": c_rand, "reduction": 1.0 - cq / max(c_rand, 1e-12)}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Residual representations (pre-registration sec. 10)
# ----------------------------------------------------------------------------------------------------------------------
def qrs_mask_from_rpeaks(rpeaks_per_window, n_time: int, fs: int = FS, half_ms: float = QRS_HALF_MS) -> np.ndarray:
    """Binary [N, T] mask: union of +/- half_ms around each GROUND-TRUTH R-peak."""
    half = int(round(half_ms / 1000.0 * fs))
    mask = np.zeros((len(rpeaks_per_window), n_time), dtype=np.float64)
    for i, rp in enumerate(rpeaks_per_window):
        for r in np.asarray(rp, dtype=int):
            mask[i, max(0, r - half) : min(n_time, r + half + 1)] = 1.0
    return mask


def residual_domains(r: np.ndarray, qmask: np.ndarray, fs: int = FS) -> dict:
    """FULL / QRS-masked / HF-projected residual representations."""
    r = np.asarray(r, dtype=np.float64)
    return {"FULL": r, "QRS": qmask * r, "HF": phi_hf(r, fs)}


def participation_ratio(mat: np.ndarray) -> dict:
    """Descriptive effective dimension of a centred [N, T] matrix: d_PR = (sum l)^2 / sum l^2, plus d90/d95."""
    x = np.asarray(mat, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    n = len(x)
    lam = np.linalg.eigvalsh(x.T @ x / max(n - 1, 1))[::-1]
    lam = np.clip(lam, 0.0, None)
    tot = lam.sum()
    if tot <= 0:
        return {"d_PR": float("nan"), "d90": 0, "d95": 0, "n_samples": int(n), "total_var": 0.0}
    csum = np.cumsum(lam) / tot
    return {"d_PR": float(tot**2 / (lam**2).sum()), "d90": int(np.searchsorted(csum, 0.90) + 1),
            "d95": int(np.searchsorted(csum, 0.95) + 1), "n_samples": int(n), "total_var": float(tot)}


# ----------------------------------------------------------------------------------------------------------------------
# Cross-fitted linear source -> residual diagnostic (pre-registration sec. 11)
# ----------------------------------------------------------------------------------------------------------------------
class PCABasis:
    """PCA basis fitted on FIT-subject residuals ONLY; 95% variance capped at max_components."""

    def __init__(self, y_fit: np.ndarray, var_target: float = 0.95, max_components: int = 128):
        y = np.asarray(y_fit, dtype=np.float64)
        self.mean_ = y.mean(0)
        yc = y - self.mean_
        lam, vecs = np.linalg.eigh(yc.T @ yc / max(len(yc) - 1, 1))
        order = np.argsort(lam)[::-1]
        lam, vecs = np.clip(lam[order], 0.0, None), vecs[:, order]
        k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), var_target) + 1) if lam.sum() > 0 else 1
        self.k_ = int(min(max(k, 1), max_components, vecs.shape[1]))
        self.explained_ = float(lam[: self.k_].sum() / max(lam.sum(), 1e-30))
        self.components_ = vecs[:, : self.k_].T

    def transform(self, y):
        return (np.asarray(y, dtype=np.float64) - self.mean_) @ self.components_.T

    def inverse_transform(self, z):
        return np.asarray(z, dtype=np.float64) @ self.components_ + self.mean_


class RidgeDiagnostic:
    """A = (X'X/n + lam I)^{-1} (X'Z/n).  Normalised objective (1/n)||Z - XA||^2 + lam||A||^2, lam fixed a priori.

    Caches the Cholesky factor of the Gram matrix so a permutation null (which leaves X'X unchanged) only recomputes X'Z.
    """

    def __init__(self, x: np.ndarray, lam: float = 1e-3):
        self.x = np.asarray(x, dtype=np.float64)
        self.n, self.d = self.x.shape
        self.lam = float(lam)
        g = self.x.T @ self.x / self.n + self.lam * np.eye(self.d)
        self.chol_ = np.linalg.cholesky(g)

    def fit(self, z: np.ndarray) -> np.ndarray:
        rhs = self.x.T @ np.asarray(z, dtype=np.float64) / self.n
        return np.linalg.solve(self.chol_.T, np.linalg.solve(self.chol_, rhs))


def r2_scores(z_true: np.ndarray, z_pred: np.ndarray, baseline_mean: np.ndarray) -> float:
    """R^2 against the FIT-subject mean baseline (never fitted on held-out data)."""
    z_true = np.asarray(z_true, dtype=np.float64)
    ss_res = ((z_true - np.asarray(z_pred, dtype=np.float64)) ** 2).sum()
    ss_tot = ((z_true - np.asarray(baseline_mean)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-30))


def dependence_with_null(x_fit, z_fit, x_ho, z_ho, lam=1e-3, n_perm=100, seed=0):
    """Held-out R^2 of the ridge diagnostic plus its permutation null.

    The null shuffles the fit-side source<->residual pairing (preserving both marginals), refits and re-evaluates on the
    UNPERMUTED held-out pairs. Returns observed R^2, null mean / 2.5-97.5 percentiles and delta = observed - null mean.
    """
    ridge = RidgeDiagnostic(x_fit, lam)
    base = np.asarray(z_fit, dtype=np.float64).mean(0)
    obs = r2_scores(z_ho, np.asarray(x_ho, dtype=np.float64) @ ridge.fit(z_fit), base)
    rng = np.random.default_rng(seed)
    z_fit = np.asarray(z_fit, dtype=np.float64)
    x_ho = np.asarray(x_ho, dtype=np.float64)
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = r2_scores(z_ho, x_ho @ ridge.fit(z_fit[rng.permutation(len(z_fit))]), base)
    return {"r2": obs, "null_mean": float(null.mean()), "null_lo": float(np.percentile(null, 2.5)),
            "null_hi": float(np.percentile(null, 97.5)), "delta_r2": float(obs - null.mean()), "n_perm": int(n_perm)}
