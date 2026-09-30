"""BF0 unit tests on synthetic data only (docs/BF0_BEAT_FIRST_FEASIBILITY_PREREGISTRATION.md §6)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.beatfirst import beats as BB
from ppg2ecg.beatfirst import model as BM
from ppg2ecg.beatfirst import render as BR
from ppg2ecg.beatfirst import timing as BT

ROOT = Path(__file__).resolve().parents[1]


def smooth_signal(n=512, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    return (x - x.mean()) / (np.abs(x).max() + 1e-9)


# ------------------------------------------------------------------------------------------ geometry
def test_segment_geometry():
    assert BB.SEG_LEN == 166 and BB.R_INDEX == 64
    assert BB.segment_bounds(100) == (36, 202)
    assert BB.inside(64, 512) and not BB.inside(63, 512)
    assert BB.inside(410, 512) and not BB.inside(411, 512)


def test_padded_segment_edge_values():
    s = np.arange(512, dtype=float)
    seg = BB.padded_segment(s, 10)
    assert seg.size == BB.SEG_LEN
    assert np.all(seg[: BB.R_INDEX - 10] == 0.0)             # clipped to index 0
    assert seg[BB.R_INDEX] == 10.0
    seg2 = BB.padded_segment(s, 500)
    assert np.all(seg2[BB.R_INDEX + 12:] == 511.0)


def test_neighbour_rr():
    prev, nxt = BB.neighbour_rr([100, 200, 320], fallback_s=0.9)
    np.testing.assert_allclose(prev, [np.median([100, 120]) / 128, 100 / 128, 120 / 128])
    np.testing.assert_allclose(nxt, [100 / 128, 120 / 128, np.median([100, 120]) / 128])
    p1, n1 = BB.neighbour_rr([200], fallback_s=0.9)
    assert p1[0] == 0.9 and n1[0] == 0.9
    assert BB.neighbour_rr([], 0.9)[0].size == 0


def test_extract_training_beats_only_inside_and_skips_single_peak_windows():
    X = np.stack([smooth_signal(seed=i) for i in range(3)])
    Y = np.stack([smooth_signal(seed=10 + i) for i in range(3)])
    peaks = [np.array([30, 150, 280, 450]), np.array([200]), np.array([100, 230])]
    E, P, RR, W = BB.extract_training_beats(X, Y, peaks)
    # window 0: 30 and 450 are too close to the edges; window 1 has one peak; window 2 keeps both
    assert list(W) == [0, 0, 2, 2]
    np.testing.assert_array_equal(E[0], Y[0][150 - 64:150 + 102].astype(np.float32))
    np.testing.assert_array_equal(P[2], X[2][100 - 64:100 + 102].astype(np.float32))
    np.testing.assert_allclose(RR[0], [120 / 128, 130 / 128], rtol=1e-6)
    assert E.shape == (4, 166) and RR.shape == (4, 2)


# ------------------------------------------------------------------------------------------ renderer
def test_assemble_reconstructs_true_slices_exactly():
    s = smooth_signal(seed=3)
    peaks = [70, 170, 280, 390, 480]
    y, info = BR.assemble([BB.padded_segment(s, p) for p in peaks], peaks, 512, empty_fill=-9.0)
    covered_from, covered_to = peaks[0] - BB.R_INDEX, min(511, peaks[-1] + 101)
    np.testing.assert_allclose(y[covered_from:covered_to + 1], s[covered_from:covered_to + 1], atol=1e-12)
    assert info["n_uncovered"] == covered_from               # samples before the first segment
    assert np.all(y[:covered_from] == y[covered_from])        # nearest-covered fill


def test_assemble_without_beats_uses_fill():
    y, info = BR.assemble([], [], 512, empty_fill=-0.7)
    assert info["empty"] and np.all(y == -0.7)


def test_template_render_equals_assemble_with_repeated_template():
    tmpl = np.sin(np.linspace(0, 3, BB.SEG_LEN))
    pos = [90, 200, 330]
    np.testing.assert_allclose(BR.render_template(tmpl, pos, 512, 0.0), BR.assemble([tmpl] * 3, pos, 512, 0.0)[0])


def test_weights_positive():
    assert BR.WEIGHTS.size == BB.SEG_LEN and np.all(BR.WEIGHTS > 0)


def test_mean_of_renders_equals_render_of_mean_beats():
    rng = np.random.default_rng(1)
    pos = [80, 190, 300, 420]
    reals = [rng.normal(size=(4, BB.SEG_LEN)) for _ in range(5)]
    renders = np.stack([BR.assemble(b, pos, 512, 0.0)[0] for b in reals])
    mean_render = BR.assemble(np.mean(reals, axis=0), pos, 512, 0.0)[0]
    np.testing.assert_allclose(renders.mean(axis=0), mean_render, atol=1e-12)


def test_beat_windows_and_diversity():
    x = np.arange(512, dtype=float)
    W = BR.beat_windows(x, [10, 100, 500])                   # 10 and 500 are too close to the edges
    assert W.shape == (1, 83) and W[0, 32] == 100.0
    beat = np.sin(np.linspace(0, 6, 83))
    sig = np.zeros(512)
    for p in (100, 220, 340):
        sig[p - 32:p + 51] = beat
    assert BR.within_window_diversity(sig, [100, 220, 340]) == pytest.approx(0.0, abs=1e-12)
    sig[220 - 32:220 + 51] += 0.5
    assert BR.within_window_diversity(sig, [100, 220, 340]) > 0.1
    assert np.isnan(BR.within_window_diversity(sig, [100]))
    assert BR.mean_pairwise_rms(np.stack([np.zeros(83), np.ones(83)])) == pytest.approx(1.0)


def test_boundary_statistic_excludes_peaks():
    x = np.zeros(512)
    x[100] = 5.0                                              # a spike at a placed peak: excluded
    assert BR.boundary_statistic(x, [100]) == 0.0
    x[300] = 1.0                                              # a jump away from any peak: counted
    assert BR.boundary_statistic(x, [100]) == 1.0


# ------------------------------------------------------------------------------------------ timing
def test_matched_residuals():
    r = BT.matched_residuals([100, 300], [110, 315], fs=128, match_ms=150.0)
    assert r[0] == pytest.approx(10 / 128 * 1000)
    assert r[1] == pytest.approx(15 / 128 * 1000)
    r2 = BT.matched_residuals([100], [130], fs=128, match_ms=150.0)
    assert np.isnan(r2[0])                                    # 234 ms away


def test_event_features_shape_and_padding():
    field = np.linspace(0, 1, 512)
    ppg = np.linspace(-1, 1, 512)
    F = BT.event_features(field, ppg, [5, 250])
    assert F.shape == (2, 65 + 193)
    assert F[0, 0] == pytest.approx(field[0])                # e - 32 clipped to 0
    assert F[1, 32] == pytest.approx(field[250], rel=1e-6)
    assert BT.event_features(field, ppg, []).shape == (0, 258)


def test_conformal_quantile_small_n_is_infinite():
    assert BT.conformal_quantile([0.1, 0.2, 0.3], 0.9) == float("inf")
    assert BT.conformal_quantile([0.1, 0.2, 0.3], 0.5) == 0.2


def test_conformal_fixes_an_overconfident_head():
    rng = np.random.default_rng(0)
    n = 20000
    mu, sigma = rng.normal(size=2 * n), rng.uniform(5, 30, size=2 * n)
    resid = mu + rng.normal(size=2 * n) * sigma * 2.0         # true spread is twice the predicted sigma
    cal, ev = slice(0, n), slice(n, 2 * n)
    for a in BT.ALPHAS:
        raw = BT.coverage(resid[ev], mu[ev], sigma[ev], BT.gaussian_z(a))
        q = BT.conformal_quantile(np.abs(resid[cal] - mu[cal]) / sigma[cal], a)
        cov = BT.coverage(resid[ev], mu[ev], sigma[ev], q)
        assert raw < a - 0.1                                  # overconfident before calibration
        assert abs(cov - a) < 0.02


# ------------------------------------------------------------------------------------------ model
def test_beatflownet_shapes_and_film_starts_as_identity():
    torch.manual_seed(0)
    net = BM.BeatFlowNet()
    x, p = torch.randn(4, BB.SEG_LEN), torch.randn(4, BB.SEG_LEN)
    t = torch.rand(4)
    v1 = net(x, p, t, torch.full((4, 2), 0.8))
    v2 = net(x, p, t, torch.full((4, 2), 1.3))
    assert v1.shape == (4, BB.SEG_LEN)
    torch.testing.assert_close(v1, v2)                        # zero-initialised FiLM ignores the condition at start


def test_cfm_loss_backward_and_sampler_determinism():
    torch.manual_seed(0)
    net = BM.BeatFlowNet()
    x1, p = torch.randn(8, BB.SEG_LEN), torch.randn(8, BB.SEG_LEN)
    rr = torch.full((8, 2), 0.9)
    loss = BM.cfm_loss(net, x1, p, rr, torch.randn(8, BB.SEG_LEN), torch.rand(8))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(q.grad is not None for q in net.parameters() if q.requires_grad)
    x0 = torch.randn(3, BB.SEG_LEN, generator=torch.Generator().manual_seed(7))
    a = BM.euler_sample(net, p[:3], rr[:3], x0)
    b = BM.euler_sample(net, p[:3], rr[:3], x0)
    torch.testing.assert_close(a, b)
    assert a.shape == (3, BB.SEG_LEN)
    assert BM.n_params(net) < 1_000_000


def test_deterministic_arm_depends_only_on_ppg_and_rr():
    torch.manual_seed(0)
    net = BM.BeatFlowNet()
    with torch.no_grad():                                    # make the conditioning matter, as after training
        for blk in net.blocks:
            blk.film.weight.normal_(0, 0.05)
    p, rr = torch.randn(4, BB.SEG_LEN), torch.full((4, 2), 0.9)
    y1, y2 = BM.deterministic_predict(net, p, rr), BM.deterministic_predict(net, p, rr)
    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(y1, net(torch.zeros_like(p), p, torch.zeros(4), rr))
    assert not torch.allclose(y1, BM.deterministic_predict(net, p, torch.full((4, 2), 1.4)))
    loss = BM.l1_loss(net, torch.randn(4, BB.SEG_LEN), p, rr)
    loss.backward()
    assert torch.isfinite(loss)


# ------------------------------------------------------------------------------------------ script guards
@pytest.fixture(scope="module")
def bf0():
    sys.path.insert(0, str(ROOT / "scripts"))
    import bf0_run
    return bf0_run


def test_load_role_refuses_test_split(bf0):
    with pytest.raises(ValueError):
        bf0.load_role("test")


def test_conformal_half_is_patient_level_and_deterministic(bf0):
    pid = np.repeat(np.arange(200), 5)
    h1, h2 = bf0.conformal_half(pid), bf0.conformal_half(pid)
    np.testing.assert_array_equal(h1, h2)
    for s in range(200):
        assert len(set(h1[pid == s])) == 1
    frac = h1.reshape(200, 5)[:, 0].mean()
    assert 0.35 < frac < 0.65


def test_shuffle_never_pairs_a_beat_with_its_own_patient(bf0):
    pid = np.repeat(np.arange(40), 7)
    d1, d2 = bf0.shuffle_donors(pid), bf0.shuffle_donors(pid)
    np.testing.assert_array_equal(d1, d2)
    assert np.all(pid[d1] != pid)
    with pytest.raises(ValueError):
        bf0.shuffle_donors(np.zeros(5, int))


def test_beat_seed_primary_realization_and_uniqueness(bf0):
    assert bf0.beat_seed(3, 0, 2) == 1_000_003 * 3 + 2
    seeds = {bf0.beat_seed(n, s, j) for n in range(50) for s in range(16) for j in range(12)}
    assert len(seeds) == 50 * 16 * 12


def test_timing_sd(bf0):
    ref = [np.array([100, 250, 400])]
    same = [[np.array([101, 250, 399])] * 16]
    assert bf0.timing_sd(ref, same) == (0.0, 3)
    rng = np.random.default_rng(0)
    jitter = [[np.array([100, 250, 400]) + rng.integers(-2, 3, size=3) for _ in range(16)]]
    med, n = bf0.timing_sd(ref, jitter)
    assert n == 3 and 0 < med < 3 / 128 * 1000
    sparse = [[np.array([100])] * 4 + [np.array([], int)] * 12]
    assert bf0.timing_sd(ref, sparse)[1] == 0                # fewer than 8 of 16 matched: excluded


def test_positions_are_corrected_sorted_and_clipped(bf0):
    pos, order = bf0._positions(np.array([300, 5, 505]), np.array([15.625, -100.0, 100.0]))
    np.testing.assert_array_equal(pos, [0, 302, 511])
    np.testing.assert_array_equal(order, [1, 0, 2])


# ------------------------------------------------------------------------------------------ pre-commit fidelity tests
def test_shuffle_rule_matches_a_manual_computation(bf0):
    import hashlib
    pid = np.array([0, 0, 1, 1, 2, 2, 3])
    M = pid.size
    key = [hashlib.sha256(f"bf0-shuffle-v1:{i}".encode()).hexdigest() for i in range(M)]
    order = sorted(range(M), key=lambda i: key[i])
    expected = []
    for b in range(M):
        k = (order.index(b) + M // 2) % M
        while pid[order[k]] == pid[b]:
            k = (k + 1) % M
        expected.append(order[k])
    np.testing.assert_array_equal(bf0.shuffle_donors(pid), expected)


def test_s_ppg_and_s_rr_replace_only_their_own_condition(bf0):
    ppg = np.arange(4 * BB.SEG_LEN, dtype=np.float32).reshape(4, BB.SEG_LEN)
    rr = np.arange(8, dtype=np.float32).reshape(4, 2)
    donor = np.array([2, 3, 0, 1])
    sp, sr = bf0.shuffled_conditions(ppg, rr, donor, "S_PPG")
    np.testing.assert_array_equal(sp, ppg[donor])
    np.testing.assert_array_equal(sr, rr)
    sp, sr = bf0.shuffled_conditions(ppg, rr, donor, "S_RR")
    np.testing.assert_array_equal(sp, ppg)
    np.testing.assert_array_equal(sr, rr[donor])
    with pytest.raises(ValueError):
        bf0.shuffled_conditions(ppg, rr, donor, "S_X")


def test_sample_stochastic_uses_the_frozen_noise_seeds(bf0):
    torch.manual_seed(0)
    net = BM.BeatFlowNet().eval()
    ppg = np.random.default_rng(0).normal(size=(3, BB.SEG_LEN)).astype(np.float32)
    rr = np.full((3, 2), 0.9, np.float32)
    own = np.array([[5, 0], [5, 1], [9, 0]])
    cpu = torch.device("cpu")
    out0 = bf0.sample_stochastic(net, ppg, rr, own, 0, cpu)
    x0 = torch.stack([torch.randn(BB.SEG_LEN, generator=torch.Generator().manual_seed(bf0.beat_seed(n, 0, j)))
                      for n, j in own])
    ref0 = BM.euler_sample(net, torch.from_numpy(ppg), torch.from_numpy(rr), x0).numpy()
    np.testing.assert_allclose(out0, ref0, atol=1e-6)
    np.testing.assert_array_equal(out0, bf0.sample_stochastic(net, ppg, rr, own, 0, cpu))   # reproducible
    assert not np.allclose(out0, bf0.sample_stochastic(net, ppg, rr, own, 1, cpu))          # realization 1 differs


def test_primary_fd_waveforms_use_realization_zero_and_equal_counts(bf0):
    rng = np.random.default_rng(0)
    A3 = rng.normal(size=(16, 7, 512)).astype(np.float16)
    R = {"A1": rng.normal(size=(7, 512)), "A2": rng.normal(size=(7, 512)), "A3": A3, "A3_primary": A3[0]}
    w = bf0.primary_waveforms(R)
    np.testing.assert_array_equal(w["A3"], A3[0])
    assert {k: v.shape for k, v in w.items()} == {"A1": (7, 512), "A2": (7, 512), "A3": (7, 512)}
    with pytest.raises(AssertionError):                       # the 16-sample mean is never the FD waveform
        bf0.primary_waveforms(R | {"A3_primary": A3.astype(np.float64).mean(0).astype(np.float16)})
    with pytest.raises(AssertionError):                       # no arm may contribute a different number of windows
        bf0.primary_waveforms(R | {"A1": R["A1"][:6]})


def test_fd_draw_uses_the_same_windows_for_both_arms(bf0):
    from ppg2ecg.evaluation import paper_metrics as PMX
    rng = np.random.default_rng(0)
    a, b, Y = (rng.normal(size=(60, 8)) for _ in range(3))
    bf0._FD.update(a=a, b=b, Y=Y)
    idx = np.array([0, 0, 3, 5, 7, 7, 7, 11] * 5)
    assert bf0._fd_draw(idx) == pytest.approx(PMX.kanflow_fd(a[idx], Y[idx]) - PMX.kanflow_fd(b[idx], Y[idx]))


def test_fd_bootstrap_refuses_a_regime_switch(bf0):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 16))
    with pytest.raises(AssertionError):                       # replicates below 3,000 windows would change kanflow_fd's regime
        bf0.fd_diff_ci(a, a, a, np.repeat(np.arange(8), 5))


def test_patient_bootstrap_takes_whole_patients(bf0):
    pid = np.repeat([3, 7, 11, 20], [1, 4, 2, 5])
    subs = np.unique(pid)
    reps = bf0.patient_bootstrap_indices(pid, 50, seed=1)
    draws = bf0.patient_resamples(len(subs), 50, seed=1)
    for r, d in zip(reps, draws):
        np.testing.assert_array_equal(r, np.concatenate([np.flatnonzero(pid == subs[k]) for k in d]))
        for s in subs:                                        # a drawn patient brings every one of its windows
            assert np.sum(pid[r] == s) == np.sum(d == np.searchsorted(subs, s)) * np.sum(pid == s)
    for r, r2 in zip(reps, bf0.patient_bootstrap_indices(pid, 50, seed=1)):
        np.testing.assert_array_equal(r, r2)


def test_cluster_ci_gives_equal_patient_weight(bf0):
    pid = np.array([0] * 9 + [1])
    d = np.array([1.0] * 9 + [0.0])
    pt, lo, hi = bf0.cluster_ci(d, pid, n_rep=200)
    assert pt == pytest.approx(0.5)                           # a window-weighted mean would be 0.9
    assert 0.0 <= lo <= pt <= hi <= 1.0
    d2 = d.copy()
    d2[0] = np.nan                                            # a nan window is skipped; its patient stays
    assert bf0.cluster_ci(d2, pid, n_rep=200)[0] == pytest.approx(0.5)
    assert all(np.isnan(bf0.cluster_ci(np.full(10, np.nan), pid, n_rep=10)))


def test_matched_pairs_are_shared_and_exclude_edges_and_unmatched_beats():
    ref, pos = [20, 150, 300, 470], [22, 155, 390, 472]
    # 20<->22 and 470<->472 match but their windows leave the signal; 300 is missed; 390 is extra
    assert BR.matched_pairs(ref, pos, 512) == [(150, 155)]
    g = smooth_signal(seed=4)
    a = np.zeros(512)
    a[155 - 32:155 + 51] = g[150 - 32:150 + 51]
    np.testing.assert_allclose(BR.pair_correlations(g, a, [(150, 155)]), [1.0])
    assert np.isnan(BR.pair_correlations(g, np.zeros(512), [(150, 155)])[0])      # flat arm window
    assert BR.matched_pairs(ref, ref, 512) == [(150, 150), (300, 300)]           # oracle: identity pairs


def test_window_pair_means_uses_pairs_finite_in_every_arm(bf0):
    corr = {"A1": [np.array([0.9, 0.5]), np.array([])], "A3": [np.array([np.nan, 0.7]), np.array([])]}
    out = bf0.window_pair_means(corr, ("A1", "A3"))
    assert out["A1"][0] == pytest.approx(0.5) and out["A3"][0] == pytest.approx(0.7)   # pair 0 dropped from both
    assert np.isnan(out["A1"][1]) and np.isnan(out["A3"][1])
    assert out["_n_pairs"] == 1


def test_g4b_diversity_uses_the_same_placed_beats_for_both_arms():
    pos = [70, 240, 410]                                      # 170 samples apart: segments do not overlap
    tmpl = np.sin(np.linspace(0, 6, BB.SEG_LEN))
    rng = np.random.default_rng(0)
    same = BR.assemble([tmpl] * 3, pos, 512, 0.0)[0]
    varied = BR.assemble([tmpl + 0.2 * rng.normal(size=BB.SEG_LEN) for _ in pos], pos, 512, 0.0)[0]
    assert BR.beat_windows(same, pos).shape == BR.beat_windows(varied, pos).shape == (3, 83)
    assert BR.within_window_diversity(same, pos) < 1e-12 < BR.within_window_diversity(varied, pos)


def test_seed_diversity_and_render_drift(bf0):
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 512))
    same = np.repeat(base[None], 16, axis=0)                  # [16, 1, 512]
    assert bf0.seed_diversity(same, [[100, 300, 500]]) == (0.0, 2)   # 500 + 51 > 512 is skipped
    noisy = same + 0.1 * rng.normal(size=same.shape)
    assert bf0.seed_diversity(noisy, [[100, 300]])[0] > 0.05
    d = bf0.render_drift([np.array([100, 300])], [np.array([101, 450])])
    assert d["n_placed"] == 2 and d["n_detected_within_50ms"] == 1 and d["fraction_detected"] == 0.5
    assert d["median_abs_offset_ms"] == pytest.approx(1000 / 128) and d["fraction_abs_offset_le_1_sample"] == 1.0


def _passing_detail():
    return {"G1": {"f1_diff": [-0.005, -0.019, 0.01], "rr_mae_diff_ms": [0.5, -0.2, 1.99]},
            "G3": {"A3-A1": [-5.0, -8.0, -0.1], "A3-A2": [-3.0, -6.0, -0.01]},
            "G4b": {"D(A3)-D(A2)": [0.02, 0.001, 0.04]},
            "G4a": {"A3_median_sd_ms": 1000.0 / 128},
            "G2": {"A3_mean-A1": [-0.01, -0.0199, 0.0]}}


@pytest.mark.parametrize("gate,section,key,value", [
    ("G1_rhythm_preservation", "G1", "f1_diff", [-0.005, -0.021, 0.01]),
    ("G1_rhythm_preservation", "G1", "rr_mae_diff_ms", [0.5, -0.2, 2.0]),
    ("G3_distributional_gain", "G3", "A3-A1", [-5.0, -8.0, 0.0]),
    ("G3_distributional_gain", "G3", "A3-A2", [0.5, -6.0, -0.01]),
    ("G4b_non_collapse", "G4b", "D(A3)-D(A2)", [0.02, 0.0, 0.04]),
    ("G4a_timing_invariance", "G4a", "A3_median_sd_ms", 7.9),
    ("G4a_timing_invariance", "G4a", "A3_median_sd_ms", float("nan")),
    ("G2_centre_consistency", "G2", "A3_mean-A1", [-0.01, -0.021, 0.0]),
])
def test_gate_rules_and_margins(bf0, gate, section, key, value):
    assert all(bf0.compute_gates(_passing_detail()).values())
    d = _passing_detail()
    d[section][key] = value
    g = bf0.compute_gates(d)
    assert not g[gate] and all(v for k, v in g.items() if k != gate)
    assert bf0.MARGIN["g4a_sd_ms"] == 1000.0 / 128           # exactly one sample at 128 Hz


def test_verdict_follows_the_frozen_gate_order(bf0):
    assert bf0.GATE_ORDER == ("G1_rhythm_preservation", "G3_distributional_gain", "G4b_non_collapse",
                              "G4a_timing_invariance", "G2_centre_consistency")
    all_pass = {g: True for g in bf0.GATE_ORDER}
    assert bf0.verdict_case(all_pass) == "F"
    for gate, case in zip(bf0.GATE_ORDER, "ABCDE"):
        assert bf0.verdict_case(all_pass | {gate: False}) == case
    assert bf0.verdict_case({g: False for g in bf0.GATE_ORDER}) == "A"          # the first failure decides
    assert bf0.verdict_case(all_pass | {"G3_distributional_gain": False, "G2_centre_consistency": False}) == "B"
    assert bf0.verdict_case(all_pass | {"G4a_timing_invariance": False, "G4b_non_collapse": False}) == "C"
    assert set(bf0.VERDICTS) == set("ABCDEF")


def test_training_never_reads_validation_or_oracle_positions(bf0):
    import inspect
    for fn in (bf0.stage_train_timing, bf0.stage_train_beat, bf0._train_beat_model, bf0._timing_chunk):
        src = inspect.getsource(fn)
        assert 'load_role("val")' not in src and "val_reference_peaks" not in src and "render_val" not in src
    for fn in (bf0.stage_train_timing, bf0.stage_train_beat):
        assert 'load_role("train")' in inspect.getsource(fn)


def test_json_cleaning(bf0):
    out = bf0.clean({"a": np.float64("nan"), "b": np.int64(3), "c": [np.float32(1.5), np.inf], "d": np.bool_(True)})
    assert out == {"a": None, "b": 3, "c": [1.5, None], "d": True}
