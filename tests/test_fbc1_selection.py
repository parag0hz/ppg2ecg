"""FBC1 pipeline validation on synthetic patient-loss matrices (preregistration section 13; not evidence).

A  one clearly optimal allocation, low noise      -> full-validation recovery increases with n
B  all allocations equal                          -> selection unstable (high entropy) but regret ~ 0
C  two near-tied best allocations                 -> exact recovery unstable, near-optimal rate high
D  small-n noisy false winner (high-variance arm) -> FBC-UCB no less stable and no worse than FBC-ERM
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fbc1_run as F  # noqa: E402

P, A, NS = 289, 6, 200
SIZES = (5, 10, 25, 50, 100)


def _subsets(seed=0):
    rng = np.random.default_rng(seed)
    M, groups, k = [], [], 0
    for n in SIZES:
        for _ in range(NS):
            row = np.zeros(P, bool); row[rng.choice(P, n, replace=False)] = True; M.append(row)
        groups.append(np.arange(k, k + NS)); k += NS
    return np.array(M), groups


def _run(L, delta=0.10):
    M, groups = _subsets()
    ev = F.evaluate_subsets(L, M)
    a_full = int(F.argmin_tie(L.mean(0)[None])[0])
    fixed = {"pure_width": 0, "pure_depth": A - 1, "balanced": 2, "historical": 0, "fullval": a_full}
    res, _, _ = F.summarise(ev, groups, fixed, delta, a_full)
    return res, ev, M


def _losses(offsets, noise, seed=1, arm_noise=None):
    rng = np.random.default_rng(seed)
    u = rng.normal(8.0, 2.0, P)[:, None]                      # patient difficulty shared by all arms
    sd = np.full(A, noise) if arm_noise is None else np.asarray(arm_noise)
    return np.abs(u + np.asarray(offsets)[None] + rng.normal(0, 1, (P, A)) * sd[None])


def test_tie_rule_prefers_lower_K():
    V = np.array([[5.0, 4.0, 4.0, 6.0, 4.0 + 5e-10, 7.0]])
    assert F.argmin_tie(V)[0] == 4                              # ties within 1e-9 -> the later arm (lower K)
    assert F.argmin_tie(np.array([[3.0, 4.0, 5.0]]))[0] == 0


def test_ucb_matches_manual_t_bound():
    L = _losses([0, 0.3, 0.6, 0.9, 1.2, 1.5], 1.0)
    M, _ = _subsets()
    ev = F.evaluate_subsets(L, M[:3])
    for r in range(3):
        x = L[M[r]]
        n = len(x)
        man = x.mean(0) + stats.t.ppf(0.90, n - 1) * x.std(0, ddof=1) / np.sqrt(n)
        assert np.allclose(ev["ucb"][r], man)
        held = L[~M[r]].mean(0)
        assert np.allclose(ev["RE"][r], held)


def test_A_clear_optimum_recovery_increases():
    res, _, _ = _run(_losses([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 0.5))
    for meth in ("UCB", "ERM"):
        rec = res[f"{meth}|fullval_recovery"]
        assert rec[0] < rec[-1] and rec[-1] > 0.95, rec
        assert res[f"{meth}|mean_regret"][-1] < res[f"{meth}|mean_regret"][0] + 1e-12


def test_B_equal_arms_unstable_but_harmless():
    res, _, _ = _run(_losses([0.0] * A, 0.5))
    for meth in ("UCB", "ERM"):
        assert res[f"{meth}|entropy"][0] > 0.8 * np.log(A)
        assert np.all(res[f"{meth}|mean_regret"] < 0.10)


def test_C_near_tie_exact_unstable_near_optimal_high():
    res, _, _ = _run(_losses([0.0, 0.02, 1.0, 1.5, 2.0, 2.5], 0.5), delta=0.10)
    i25 = SIZES.index(25)
    for meth in ("UCB", "ERM"):
        assert res[f"{meth}|fullval_recovery"][i25] < 0.90
        assert res[f"{meth}|near_rate"][i25] > 0.90


def test_D_noisy_false_winner_ucb_no_worse():
    L = _losses([0.0, 0.3, 1.5, 2.0, 2.5, 3.0], None, arm_noise=[0.2, 3.0, 0.2, 0.2, 0.2, 0.2])
    res, _, _ = _run(L)
    for i in (0, 1):                                             # n = 5, 10
        assert res["UCB|mean_regret"][i] <= res["ERM|mean_regret"][i]
        assert res["UCB|entropy"][i] <= res["ERM|entropy"][i]
        assert res["UCB|near_rate"][i] >= res["ERM|near_rate"][i]
