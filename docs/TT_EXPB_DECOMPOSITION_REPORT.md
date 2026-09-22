# EXP-B — Why width works: the pooling gain is governed by the **correlation of the samples' functional errors**, not by waveform diversity

Prereg `50948e2` (EXP-B), frozen before any number; theory note `6b23d25` written before these numbers were read. No training.
VitalDB V1 test (1,156 patients, 19,543 windows), seed-42 checkpoints for B1 / B3, seeds 42 / 1 / 2 for B2; samplers and noise
seeds identical to DW1 (`dw1_depth_width.make_sampler`). Per window c, sample k at depth S: `Y_{S,k} = HR(x_{S,k})`,
reference `T* = HR(target ECG)`, centre `m_S = median_k Y_{S,k}`. Windows with undefined reference HR excluded (Ω_HR);
samples with undefined HR enter as missing (`nanmedian`) and their fraction is reported. Patient-clustered bootstrap.
Raw: `artifacts/tt_expb_mechanism/{b1_fixed_K16_by_S.csv, b2_fixed_S_by_K.csv, b3_error_correlation.csv, result.json}`;
figures `expb_decomposition.png`, `expb_error_correlation.png`. Command: `.venv/bin/python scripts/tt_expb_decomposition.py`.

## B1 — K fixed at 16, S varied: what depth actually buys (seed 42)
`C` = centre error `|m_S − T*|` (= the K = 16 consensus error) · `I` = mean individual-sample error · `G = I − C` (median gain)
· `MAD` / `SD` = within-window dispersion of the sample functionals · `F1` = mean per-sample R-peak F1 · `wRMS` = pairwise
waveform RMS (2,000-window subset) · `FD` = mean per-sample KANFlow FD · `nan%` = share of samples with undefined HR.

| model | S | total NFE | **C** | I | **G** | MAD | SD | F1 | wRMS | FD | nan% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **iMF** | 1 | 16 | 6.675 | 10.121 | 3.446 | 3.488 | 10.23 | 0.675 | 0.477 | 2.41 | 0.80 |
| | **2** | 32 | **6.176** | 9.412 | 3.236 | 3.036 | 9.59 | 0.673 | 0.505 | 2.60 | 1.02 |
| | 4 | 64 | 6.221 | 8.974 | 2.752 | 2.809 | 8.61 | 0.677 | 0.490 | 2.47 | 2.39 |
| | 8 | 128 | 6.304 | 8.834 | 2.530 | 2.714 | 8.18 | 0.677 | 0.488 | 2.69 | 3.27 |
| **CD** | 1 | 16 | 6.229 | 8.249 | 2.020 | 2.733 | 6.92 | 0.675 | 0.422 | 3.03 | 1.78 |
| | 2 | 32 | 6.253 | 8.958 | 2.704 | 3.038 | 8.41 | 0.664 | 0.462 | 3.30 | 0.26 |
| | 4 | 64 | 6.203 | 8.639 | 2.436 | 2.919 | 7.80 | 0.669 | 0.474 | 5.20 | 0.68 |
| | 8 | 128 | 6.171 | 8.381 | 2.210 | 2.766 | 7.32 | 0.677 | 0.472 | 10.23 | 0.64 |
| **PENGUIN (Euler)** | 1 | 16 | 7.438 | 8.675 | 1.238 | 2.530 | 5.71 | **0.755** | **0.074** | 21.50 | 5.29 |
| | 2 | 32 | 6.349 | 8.069 | 1.720 | 2.627 | 6.18 | 0.703 | 0.219 | 14.92 | 2.23 |
| | **4** | 64 | 6.128 | 8.260 | 2.132 | 2.692 | 7.09 | 0.683 | 0.333 | 7.61 | 1.46 |
| | 8 | 128 | **6.028** | 8.463 | 2.434 | 2.683 | 7.71 | 0.669 | 0.405 | 4.48 | 0.15 |

Paired contrasts against S = 1 (patient-clustered CI):

| model | S = 2 − S = 1 | S = 4 − S = 1 | S = 8 − S = 1 |
|---|---|---|---|
| iMF | ΔC **−0.499** [−0.575, −0.427] · ΔI −0.709 · ΔG −0.210 | ΔC −0.453 · ΔI −1.147 · ΔG −0.694 | ΔC −0.371 · ΔI −1.287 · ΔG −0.916 |
| CD | ΔC **+0.025** [−0.059, +0.108] · ΔI **+0.709** · ΔG +0.684 | ΔC −0.026 [−0.101, +0.052] · ΔI +0.390 · ΔG +0.416 | ΔC −0.058 [−0.131, +0.016] · ΔI +0.133 · ΔG +0.190 |
| PENGUIN | ΔC **−1.089** [−1.226, −0.954] · ΔI −0.607 · ΔG **+0.483** | ΔC **−1.310** [−1.450, −1.169] · ΔI −0.416 · ΔG **+0.894** | ΔC −1.410 · ΔI −0.214 · ΔG +1.197 |

**Preregistered criteria — all three hold:**
- **B1-1 (PENGUIN, `C(S=4) < C(S=1)`): holds**, −1.310 [−1.450, −1.169].
- **B1-2 (iMF, `C(S=2) < C(S=1)`): holds**, −0.499 [−0.575, −0.427].
- **B1-3 (CD, no improvement S = 1 → 2): holds**, +0.025 [−0.059, +0.108] (interval contains 0).
So the DW1 / DW2-A optima are **not** an artefact of K shrinking as S grows: at a fixed K = 16 the same ordering appears.

**The preregistered reading of the decomposition (stated before the numbers) gives three different answers, one per model:**
1. **iMF — depth improves the samples themselves.** ΔI (−0.71) is larger than ΔC (−0.50) and the gain *shrinks* (−0.21):
   two steps make each individual sample better and less dispersed (MAD 3.49 → 3.04), leaving less for the median to remove.
   Beyond S = 2 the individual error keeps falling but C rises again — the interior optimum is real and is a property of the
   centre, not of K.
2. **CD — depth buys nothing for the functional.** The centre does not move (ΔC +0.03 at S = 2, −0.06 at S = 8, all intervals
   touching 0) while individual samples get *worse* (ΔI +0.71 at S = 2) and the median removes exactly that extra noise
   (ΔG +0.68). Meanwhile FD degrades 3.03 → 10.23. Re-noising a consistency model adds dispersion, not information.
3. **PENGUIN — depth buys poolability.** The centre improves by 1.1–1.4 bpm while the individual error improves by at most
   0.6 and *worsens* again beyond S = 2 (8.07 → 8.26 → 8.46): the improvement is carried by the gain (1.24 → 1.72 → 2.13 → 2.43).
   Its S = 1 samples are the best single samples in the programme on beat metrics (F1 0.755) and the worst to pool.
   Waveform realism moves in the opposite direction to F1 (FD 21.5 → 4.5).

## B2 — S fixed per model (CD 1, iMF 2, PENGUIN 4, chosen in DW1, not re-selected), K varied
HR error by K, partition estimate (32 draws split into 32/K disjoint groups, group medians averaged):

| model | seed | K = 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| iMF (S = 2) | 42 / 1 / 2 | 9.42 / 8.60 / 9.64 | 8.74 / 8.01 / 8.93 | 7.19 / 6.78 / 7.25 | 6.52 / 6.30 / 6.51 | 6.20 / 6.11 / 6.21 | 6.06 / 6.02 / 6.07 |
| CD (S = 1) | 42 / 1 / 2 | 8.27 / 9.02 / 9.34 | 7.70 / 8.38 / 8.67 | 6.78 / 7.12 / 7.39 | 6.41 / 6.56 / 6.83 | 6.24 / 6.32 / 6.57 | 6.14 / 6.20 / 6.44 |
| PENGUIN (S = 4) | 42 / 1 / 2 | 8.27 / 9.41 / 9.30 | 7.70 / 8.75 / 8.59 | 6.72 / 7.25 / 7.25 | 6.32 / 6.51 / 6.62 | 6.14 / 6.16 / 6.32 | 6.05 / 5.99 / 6.19 |

**B2-1 (marginal gains non-increasing in K): FAILS in 9/9**, always at the same place — the K = 1 → 2 gain (+0.56 … +0.71) is
**smaller** than the K = 2 → 4 gain (+0.92 … +1.68). The cause is the estimator, not the model: for K = 2 the median is the
**mean** of the two draws, and AB1 already measured that the mean is a much worse pooling rule than the median for this
error distribution (iMF K = 16: mean 8.35 vs median 6.68). From **K ≥ 2 the gains are strictly decreasing in 9/9 runs**
(e.g. iMF seed 42: 1.553 → 0.667 → 0.322 → 0.141). Reported as a failure of the criterion as written, with the K ≥ 2
statement given separately; no criterion was redefined after the fact.
Saturation under AB1's frozen rule (doubling K changes HR by < 0.1 bpm) is reached by K = 16 in 3 of 9 runs and not by
K = 32 in the other 6 — the width sweep is still paying at the end of the preregistered range.

## B3 — Functional-error correlation across samples (K = 16, seed 42)
Signed errors `e_{S,k}(c) = Y_{S,k}(c) − T*(c)`; correlation over test windows for each of the 120 sample pairs.

| model | S | mean Pearson ρ̄ [p5, p95] | mean Spearman | ICC(1) | median gain G |
|---|---|---|---|---|---|
| iMF | 1 | **0.558** [0.548, 0.568] | 0.366 | 0.558 | **3.446** |
| iMF | 2 | 0.586 | 0.366 | 0.586 | 3.236 |
| iMF | 4 | 0.645 | 0.375 | 0.645 | 2.752 |
| iMF | 8 | 0.671 | 0.382 | 0.671 | 2.530 |
| CD | 1 | 0.728 | 0.419 | 0.728 | 2.020 |
| CD | 2 | 0.628 | 0.403 | 0.628 | 2.704 |
| CD | 4 | 0.667 | 0.409 | 0.667 | 2.436 |
| CD | 8 | 0.678 | 0.424 | 0.678 | 2.210 |
| PENGUIN | 1 | **0.858** [0.847, 0.869] | 0.760 | 0.858 | **1.238** |
| PENGUIN | 2 | 0.789 | 0.438 | 0.789 | 1.720 |
| PENGUIN | 4 | 0.717 | 0.406 | 0.717 | 2.132 |
| PENGUIN | 8 | 0.652 | 0.400 | 0.652 | 2.434 |

**B3-1 (every condition with a positive gain has ρ̄ < 0.9): holds** — all 12 conditions have a gain whose CI excludes 0, and
the largest ρ̄ is 0.858.
**B3-2 (Spearman(ρ̄, G) < 0): holds** — **−0.965 (p = 3.9 × 10⁻⁷, n = 12)**.

**Post-hoc (defined after seeing B1/B3, labelled as such): which variable predicts the pooling gain?**

| candidate predictor | Spearman with G, 12 conditions | within iMF | within CD | within PENGUIN |
|---|---|---|---|---|
| **functional-error correlation ρ̄** | **−0.965** (p = 3.9e−7) | **−1.00** | **−1.00** | **−1.00** |
| functional dispersion MAD | +0.846 | +1.00 | +1.00 | +0.80 |
| waveform pairwise RMS (DW2-B's variable) | +0.867 | **−0.20** | **+0.40** | +1.00 |
| per-sample FD | −0.860 | −0.80 | +0.20 | −1.00 |

ρ̄ orders the gain **perfectly inside every model** and almost perfectly across all twelve conditions. The dispersion of the
sample functionals (MAD) is the second-best predictor and is nearly as consistent (+1.00 / +1.00 / +0.80) — expected, since a
wider conditional law leaves more for the median to remove; ρ̄ and MAD are the two quantities the decomposition names.
Waveform diversity and FD order the gain across models (the three models differ a lot on both) but are inconsistent *within*
a model — waveform RMS even runs the wrong way for iMF. **The mechanism variables are the correlation and the spread of the
samples' functional errors, not how different the waveforms look.**

## Consequences
1. **The working statement survives, with a mechanism that does not depend on waveform diversity.** Width reduces the
   finite-K term of the decomposition, and how much it can reduce is set by how correlated the samples' functional errors
   are. Depth acts on the other term, and what it does there is model-specific: it improves the samples (iMF), it makes them
   poolable (PENGUIN), or it does neither (CD).
2. **DW2-B's post-hoc waveform-diversity reading is superseded and partly contradicted.** DW2-B was right that PENGUIN's
   one-step samples are near-identical waveforms (wRMS 0.074, reproduced here) and right that its consensus gain is the
   smallest; it was wrong to read waveform diversity as *the* mechanism variable — within a model it does not predict the
   gain. The DW2-B tension ("nearly identical waveforms, yet a +1.32 bpm gain") is resolved: PENGUIN's S = 1 functional
   errors are correlated at 0.858, not at 1.0, because the detector is unstable on the collapsed waveform — so there is
   real, if small, error diversity to pool even when the waveforms are nearly the same.
3. **The DW1 optima are confirmed at fixed K** (B1-1/B1-2/B1-3), so they are properties of the depth axis, not of K.
4. Limits: seed 42 for B1 / B3 (three seeds for B2), VitalDB only, one functional (HR). `nan%` up to 5.3 % (PENGUIN S = 1)
   means part of that arm's pooling is over fewer than 16 usable samples. B2-1 failed as written.
