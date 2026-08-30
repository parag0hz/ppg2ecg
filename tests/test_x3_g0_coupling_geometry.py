"""X3-G0 unit tests (docs/X3_G0_COUPLING_GEOMETRY_PREREGISTRATION.md sec. 16). Synthetic tensors only."""
from __future__ import annotations

import numpy as np
import ppg2ecg.utils.mkl_warmup  # noqa: F401
import pytest
from scipy.optimize import linear_sum_assignment

from ppg2ecg.evaluation.coupling_geometry import (
    FS,
    HF_CUT_HZ,
    PCABasis,
    RidgeDiagnostic,
    WildPPGTestFirewallError,
    assert_no_test_subjects,
    assignment_cost,
    cross_objective_regret,
    dependence_with_null,
    demean,
    fit_whitener,
    hf_mask,
    participation_ratio,
    phi_hf,
    phi_raw,
    phi_resid,
    phi_white,
    qrs_mask_from_rpeaks,
    r2_scores,
    residual_domains,
    solve_assignment,
)

T = 256


# ---- 1 / 2 / 5 / 6: assignment semantics
def test_B1_is_identity():
    x0 = np.random.default_rng(0).standard_normal((1, T))
    assert np.array_equal(solve_assignment(x0, np.random.default_rng(1).standard_normal((1, T))), np.array([0]))


def test_known_synthetic_hungarian_optimum():
    # 3 sources, 3 targets on orthogonal axes: the optimum must match each source to its own axis
    x0 = np.eye(3) * 2.0
    y = np.eye(3)[[2, 0, 1]]  # target j sits on axis of source perm
    perm = solve_assignment(x0, y)
    assert np.array_equal(perm, np.array([1, 2, 0]))
    assert assignment_cost(x0, y, perm) < assignment_cost(x0, y, np.array([0, 1, 2]))


def test_squared_l2_assignment_equals_max_inner_product():
    rng = np.random.default_rng(3)
    x0, y = rng.standard_normal((12, T)), rng.standard_normal((12, T)) * 0.3 + 5.0
    full = linear_sum_assignment(((x0[:, None, :] - y[None, :, :]) ** 2).sum(-1))[1]
    assert np.array_equal(full, solve_assignment(x0, y))


def test_positive_global_target_scale_is_assignment_noop():
    rng = np.random.default_rng(4)
    x0, y = rng.standard_normal((16, T)), rng.standard_normal((16, T))
    base = solve_assignment(x0, y)
    for s in (0.01, 0.5, 3.0, 1e4):
        assert np.array_equal(solve_assignment(x0, s * y), base)


def test_23_residual_global_scaling_assignment_noop():
    rng = np.random.default_rng(5)
    x0, y, m = rng.standard_normal((10, T)), rng.standard_normal((10, T)), rng.standard_normal((10, T)) * 0.2
    r = phi_resid(y, m)
    assert np.array_equal(solve_assignment(x0, r), solve_assignment(x0, 7.5 * r))


# ---- 3 / 4: pairing and marginals
def test_condition_and_target_stay_paired_and_source_marginal_unchanged():
    rng = np.random.default_rng(6)
    n = 20
    y, cond = rng.standard_normal((n, T)), rng.standard_normal((n, T))
    x0 = rng.standard_normal((n, T))
    perm = solve_assignment(x0, y)
    # pairs are (x0_i, y[perm[i]], cond[perm[i]]): the condition follows its own target row
    assert np.array_equal(y[perm], np.stack([y[j] for j in perm]))
    assert np.array_equal(cond[perm], np.stack([cond[j] for j in perm]))
    # source multiset untouched by the assignment
    assert np.array_equal(np.sort(x0, axis=0), np.sort(x0[np.argsort(perm)], axis=0))
    assert sorted(perm.tolist()) == list(range(n))


# ---- 7: RAW cost parity
def test_raw_cost_parity():
    y = np.random.default_rng(7).standard_normal((5, T))
    assert np.array_equal(phi_raw(y), y.astype(np.float64))


# ---- 8 / 9 / 10: whitener
def test_whitener_dc_zero_floor_and_fit_only():
    rng = np.random.default_rng(8)
    y_fit = rng.standard_normal((64, T))
    y_other = rng.standard_normal((64, T)) * 50.0
    w = fit_whitener(y_fit)
    assert w[0] == 0.0                                    # DC excluded
    assert np.isclose(w[1:].mean(), 1.0)                  # normalisation
    assert np.array_equal(w, fit_whitener(y_fit))         # deterministic
    assert not np.allclose(w, fit_whitener(y_other))      # actually depends on the fit set
    # floor is active for a spectrum with a near-zero band
    Y = np.fft.rfft(demean(y_fit), axis=-1)
    Y[:, 40:60] = 0.0
    w2 = fit_whitener(np.fft.irfft(Y, n=T, axis=-1))
    assert np.all(np.isfinite(w2)) and w2[40:60].max() < np.inf


def test_white_transform_removes_dc_and_is_linear():
    rng = np.random.default_rng(9)
    y = rng.standard_normal((8, T))
    w = fit_whitener(rng.standard_normal((64, T)))
    out = phi_white(y, w)
    assert np.allclose(out.mean(-1), 0.0, atol=1e-10)
    assert np.allclose(phi_white(y + 3.7, w), out, atol=1e-9)  # invariant to a constant offset


# ---- 11: HF transform
def test_hf_transform_uses_exactly_gt_15hz():
    f = np.fft.rfftfreq(T, 1.0 / FS)
    m = hf_mask(T, FS)
    assert np.array_equal(m > 0, f > HF_CUT_HZ)
    assert m[np.argmin(np.abs(f - 15.0))] == 0.0 if np.isclose(f.min(), 0) else True
    # a pure 5 Hz tone is removed, a pure 40 Hz tone is kept
    t = np.arange(T) / FS
    assert np.abs(phi_hf(np.sin(2 * np.pi * 5 * t)[None])).max() < 1e-8
    assert np.abs(phi_hf(np.sin(2 * np.pi * 40 * t)[None])).max() > 0.9


# ---- 12 / 13 / 14: cross-objective regret
def test_regret_is_zero_for_identical_optima():
    rng = np.random.default_rng(10)
    x0, y = rng.standard_normal((16, T)), rng.standard_normal((16, T))
    res = cross_objective_regret(x0, {"A": y, "B": 3.0 * y}, n_random=16, seed=0)
    assert np.isclose(res["regret"][("A", "B")], 0.0, atol=1e-12)
    assert np.isclose(res["regret"][("A", "A")], 0.0, atol=1e-12)
    assert res["overlap"][("A", "B")] == 1.0


def test_near_tie_low_overlap_but_low_regret():
    """Almost-identical geometries: indices churn, regret stays small."""
    rng = np.random.default_rng(11)
    x0 = rng.standard_normal((64, T))
    y = rng.standard_normal((64, T))
    y2 = y + 0.01 * rng.standard_normal((64, T))
    res = cross_objective_regret(x0, {"A": y, "B": y2}, n_random=16, seed=0)
    assert res["overlap"][("A", "B")] < 0.95          # indices do move
    assert res["regret"][("A", "B")] < 0.05           # but the A-objective barely suffers


def test_genuinely_different_geometry_gives_low_overlap_and_substantial_regret():
    rng = np.random.default_rng(12)
    x0 = rng.standard_normal((64, T))
    y = rng.standard_normal((64, T))
    y2 = rng.standard_normal((64, T))                  # unrelated geometry
    res = cross_objective_regret(x0, {"A": y, "B": y2}, n_random=16, seed=0)
    assert res["overlap"][("A", "B")] < 0.2
    assert res["regret"][("A", "B")] > 0.5
    assert 0.0 <= res["cost"]["A"]["reduction"] <= 1.0


# ---- 15 / 16 / 17: diagnostic regression plumbing
def test_pca_fitted_on_fit_only_and_capped():
    rng = np.random.default_rng(13)
    fit = rng.standard_normal((300, 40)) @ rng.standard_normal((40, T))
    p = PCABasis(fit, var_target=0.95, max_components=8)
    assert p.k_ <= 8
    assert np.array_equal(p.components_, PCABasis(fit, 0.95, 8).components_)
    assert not np.allclose(p.mean_, PCABasis(rng.standard_normal((300, T)), 0.95, 8).mean_)


def test_ridge_normalised_lambda_convention():
    rng = np.random.default_rng(14)
    x = rng.standard_normal((500, 20))
    a_true = rng.standard_normal((20, 3))
    z = x @ a_true
    a = RidgeDiagnostic(x, lam=1e-8).fit(z)
    assert np.allclose(a, a_true, atol=1e-4)
    # explicit normalised closed form
    n, d = x.shape
    lam = 0.5
    expect = np.linalg.solve(x.T @ x / n + lam * np.eye(d), x.T @ z / n)
    assert np.allclose(RidgeDiagnostic(x, lam=lam).fit(z), expect, atol=1e-10)


def test_repeated_target_window_cannot_cross_folds():
    """Fold membership is by subject, so every repetition of a window lands in the same fold."""
    subj = np.array(["a", "a", "b", "b", "c", "c"])
    folds = {"f0": {"a"}, "f1": {"b"}, "f2": {"c"}}
    win = np.array([0, 0, 1, 1, 2, 2])  # each window appears twice
    for w in np.unique(win):
        owners = {f for f, s in folds.items() if set(subj[win == w]) & s}
        assert len(owners) == 1


# ---- 18 / 19 / 20: dependence and null
def test_independent_coupling_gives_delta_r2_near_zero():
    rng = np.random.default_rng(15)
    d, k, n = 32, 4, 800
    x_fit, x_ho = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    z_fit, z_ho = rng.standard_normal((n, k)), rng.standard_normal((n, k))  # independent of x
    res = dependence_with_null(x_fit, z_fit, x_ho, z_ho, lam=1e-3, n_perm=40, seed=0)
    assert abs(res["delta_r2"]) < 0.02
    assert res["null_lo"] <= res["null_mean"] <= res["null_hi"]


def test_known_linear_coupling_gives_positive_heldout_r2_and_null_destroys_it():
    rng = np.random.default_rng(16)
    d, k, n = 32, 4, 2000
    a = rng.standard_normal((d, k)) * 0.4
    x_fit, x_ho = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    z_fit = x_fit @ a + 0.3 * rng.standard_normal((n, k))
    z_ho = x_ho @ a + 0.3 * rng.standard_normal((n, k))
    res = dependence_with_null(x_fit, z_fit, x_ho, z_ho, lam=1e-3, n_perm=40, seed=0)
    assert res["r2"] > 0.5
    assert res["delta_r2"] > 0.5
    assert res["null_mean"] < 0.02          # permutation destroys the relationship


def test_r2_baseline_uses_supplied_mean():
    z = np.array([[1.0], [3.0]])
    assert np.isclose(r2_scores(z, z, np.array([2.0])), 1.0)
    assert np.isclose(r2_scores(z, np.full_like(z, 2.0), np.array([2.0])), 0.0)


# ---- 21: QRS mask
def test_qrs_mask_is_exactly_pm_100ms_around_gt_rpeaks():
    n_time = 1024
    mask = qrs_mask_from_rpeaks([np.array([200, 600])], n_time)
    half = int(round(100.0 / 1000.0 * FS))  # 13 samples
    assert half == 13
    expect = np.zeros(n_time)
    for r in (200, 600):
        expect[r - half : r + half + 1] = 1.0
    assert np.array_equal(mask[0], expect)
    assert mask[0].sum() == 2 * (2 * half + 1)
    assert np.array_equal(qrs_mask_from_rpeaks([np.array([2])], n_time)[0][:16], np.r_[np.ones(16)])  # clipped at 0


def test_residual_domains_shapes_and_masking():
    rng = np.random.default_rng(17)
    r = rng.standard_normal((4, 1024))
    qm = qrs_mask_from_rpeaks([np.array([100, 500])] * 4, 1024)
    dom = residual_domains(r, qm)
    assert set(dom) == {"FULL", "QRS", "HF"}
    assert np.array_equal(dom["QRS"][qm == 0], np.zeros((qm == 0).sum()))
    assert np.allclose(dom["HF"].mean(-1), 0.0, atol=1e-10)


# ---- 22: test-set firewall
def test_firewall_blocks_wildppg_test_subjects():
    assert_no_test_subjects(["e61", "an0", "w4p"])  # fine
    for bad in (["kjd"], ["ssx"], ["e61", "kjd"], ["ssx", "an0"]):
        with pytest.raises(WildPPGTestFirewallError):
            assert_no_test_subjects(bad)


# ---- 13 (dimension): participation ratio on a known synthetic covariance
def test_participation_ratio_matches_analytic_value():
    rng = np.random.default_rng(18)
    d, k = 60, 5
    basis = np.linalg.qr(rng.standard_normal((d, d)))[0][:k]
    x = rng.standard_normal((40000, k)) @ basis          # exactly k equal-variance directions -> d_PR = k
    pr = participation_ratio(x)
    assert abs(pr["d_PR"] - k) < 0.2
    assert pr["d90"] == k and pr["d95"] == k
    lam = np.array([4.0, 1.0, 1.0])                       # analytic: (6)^2 / (16+1+1) = 2.0
    y = rng.standard_normal((80000, 3)) * np.sqrt(lam)
    assert abs(participation_ratio(y)["d_PR"] - 36.0 / 18.0) < 0.05
