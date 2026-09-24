# M1 (pilot-gain prediction) — Do pilot-sample statistics predict the gain from disjoint future samples? **STRONG SIGNAL** under the amended rule (reliably non-zero, modest in size)

Preregistration **`7526201`** + dated amendment **`cff9388`** (`docs/M1_PILOT_GAIN_PREDICTION_AMENDMENT.md`, deviations
D1–D9), both pushed before any M1 predictive number. Frozen predictor **`672e0d4`** (validation only); informativeness gate
**`184755d`** (before any test feature table). Code `scripts/m1_run.py` as committed at `cff9388` (sha256 `a031d8f4…` in
`input_hashes.json`); the only later change is a figure-title edit (panel D), committed with this report, after which the
`figure` stage was re-run. Validation bank `scripts/m1_val_bank.py`. Results in `artifacts/m1_pilot_gain_prediction/`:
audit.json, split_manifest.json, input_hashes.json, bank_repro_check.json, feature_schema.json, validation_cv.csv,
extended_validation_cv.csv, frozen_predictor.json, sanity_informativeness.json, test_metrics.json, bootstrap.json,
model_specific.csv, extended_test_metrics.json, crossfit_mechanism.json, cost_accounting.json, test_predictions.csv
(condition × predicted-decile summary), posthoc_within_condition_quartiles.json, figure.png. Per-unit predictions stay in
`outputs/m1_pilot_gain_prediction/test_predictions_full.npz` (predictions are not committed). **No generator was trained.**
Test CIs: 5,000-replicate patient-clustered bootstrap (1,156 test patients, seed 20260924); every statistic is the mean over
the 32 pilot/future partitions, recomputed inside each replicate. *Post-hoc* items were not preregistered.

## 1. Question
Can observable statistics of a small set of stochastic generator samples (pilot A, 8 draws) predict the consensus gain of
**independent** future samples (B, 8 other draws), without the reference at test time? The target
**G_B = mean_{k∈B}|Y_k − y| − |median_B Y − y|** is the advantage of the 8-draw future median over a single future draw.
M1 tests only whether this signal exists under strict sample splitting and patient-level held-out evaluation; it is not an
allocator.

## 2. Audit
`docs/M1_PILOT_GAIN_PREDICTION_AUDIT.md` (at `7526201`) + corrections in amendment §2. V1 VitalDB, patient-disjoint splits:
validation 289 patients / 4,822 windows, test 1,156 patients / 19,543 windows. Three frozen seed-42 generators (iMF, CD,
PENGUIN-Euler) × S ∈ {1, 2, 4, 8} = 12 conditions, K = 16 draws (seeds 0–15). Test banks existed (DW1 / AB1 / EXP-B). The
validation bank was produced by `scripts/m1_val_bank.py` — written in a session that was later rewound, recovered
byte-identically from the editor backup and committed in `cff9388`; it calls only committed samplers. S = 1 of all three
models was regenerated **bitwise identically**; S = 2, 4, 8 were not regenerated and rest on the recorded sha256 values.
The rewound session's analysis modules were only byte-compiled, never executed (amendment §1). HR functional = neurokit
R-peaks → HR (the DW / EXP-B implementation). These test draws had already been analysed at condition level in
DW1 / EXP-B / B3, so the amendment requires within-condition evidence (D3).

| Model | Split | S | Available K | Same seeds across S? | HR available? | Waveforms available? | Usable |
|---|---|---:|---:|---|---|---|---|
| iMF | val / test | 1, 2, 4, 8 | 16 / ≥ 16 | yes (exact z0) | yes | S = 1 full splits (ed1_cache); S 1–8 on a 2,000-window test subset (EXP-B) | yes |
| CD | val / test | 1, 2, 4, 8 | 16 / ≥ 16 | z0 only — re-noise injected at depth-dependent times (the audit said "prefix-paired"; see amendment §4.8) | yes | as iMF | yes |
| PENGUIN | val / test | 1, 2, 4, 8 | 16 / ≥ 16 | yes (exact z0) | yes | S 1–8 on the 2,000-window test subset only | yes |

## 3. Frozen protocol (with the amendment's deviations)
Per condition, windows whose reference and 16 draws have a finite HR (Ω_HR); 32 balanced 8/8 partitions (seed 20260924).
Ten features from the A draws only (median, MAD, IQR, SD, range, mean / median pairwise |diff|, bootstrap and split-half
median instability, log S; `log1p` on the spread features); HR snapped to 1e-3 bpm (D7). StandardScaler + Ridge, α by
GroupKFold(5) over patients on validation (row-level OOF MAE); no model identity. **D1:** every statistic is computed
within a partition (pred(A_p) against G(B_p), each unit once), then averaged over partitions. The frozen "average over
partitions, then correlate" rule makes prediction and target functions of the same 16 draws (A_p and B_q overlap for
p ≠ q): with the frozen partition list, log1p MAD(A) as the only predictor and no window-level signal, the Spearman of
partition-averaged values was +0.173 / +0.295 (Gaussian / t3) against a mean per-partition Spearman of +0.003 / +0.007.
**D3:** STRONG also needs within-condition Spearman CI > 0. **D5 / D6:** FAILED routes (pooled ≤ 0; quartile contrast ≤ 0;
not better than the validation-mean constant on test; zero models positive) and the MAD-only downgrade (MAD works *and* B3
adds no within-condition value). Sign decision omitted (no τ).

## 4. Validation calibration (out-of-fold, patient-grouped)
| predictor | α | OOF MAE (rows) | OOF Spearman pooled | OOF Spearman within condition |
|---|---|---|---|---|
| **B3 full (10 features)** | 0.01 | **2.039** | **0.306** | **0.205** |
| B2 SD only | 0.01 | 2.070 | 0.291 | 0.195 |
| B1 MAD only | 0.01 | 2.117 | 0.266 | 0.163 |
| B5 log S only | 0.01 | 2.264 | 0.079 | −0.008 (≈ 0; log S is constant within a condition up to fold refits) |
| B0 constant (fold means) | — | 2.267 | — | — |
| B4 extended (+ depth probe; S ≤ 4 rows) / B3 on the same rows | 0.01 / 0.01 | 1.9705 / 1.9701 | 0.322 / 0.321 | 0.192 / 0.191 |

α sat on a grid edge in every fit: 0.01 everywhere except LOMO-without-PENGUIN (100, the upper edge); the CV MAE varies by
≤ 1.5e-4 across the grid, so the penalty is immaterial with ~1.7 M validation rows. The B3 coefficients are dominated by the
collinear spread pair SD (−2.55) and mean pairwise |diff| (+3.07) and are not interpretable individually.

## 5. Primary held-out test result (single evaluation)
Informativeness gate first (D4): the K = 16 consensus beats the validation-median constant (MAE 4.92 vs 13.12, difference
−8.20 [−8.59, −7.83]) and tracks the reference (Spearman 0.757 [0.734, 0.780]); every condition passes Gate A; on the same
units the consensus also beats PPG peak counting (−1.55 [−1.85, −1.31] bpm). **The HR functional is usable.**

| statistic (mean over 32 partitions) | value [95 % CI] |
|---|---|
| **pooled Spearman(predicted, observed G_B)** | **+0.303 [+0.292, +0.314]** |
| **within-condition Spearman** (mean of the 12 per-condition values) | **+0.203 [+0.192, +0.215]** |
| per-condition Spearman (12 conditions) | +0.161 … +0.242, every CI > 0 |
| **top − bottom predicted quartile, pooled cut-offs** | **+2.23 [+2.13, +2.33] bpm** (Q1 1.00 · Q2 1.40 · Q3 1.90 · Q4 3.23) |
| *post-hoc:* top − bottom quartile with condition-specific cut-offs (point) | +1.71 bpm (per condition 1.11 … 2.19) |
| Pearson / R² | +0.298 [+0.284, +0.312] / **+0.089** [+0.080, +0.097] |
| MAE: B3 vs B0 constant | 2.008 vs 2.232 → −0.224 [−0.231, −0.216] (not collapsed) |
| baselines, pooled Spearman: B2 SD / B1 MAD / B5 log S | +0.285 / +0.261 / +0.084 |
| B3 − B1 (MAD), within condition / pooled | **+0.047 [+0.041, +0.053]** / +0.043 [+0.037, +0.048] |
| B3 − B2 (SD), within condition / pooled | +0.017 [+0.015, +0.019] / +0.019 [+0.017, +0.020] |
| B3 − B5 (log S), pooled | +0.219 [+0.209, +0.229] |
| validation → test (pooled / within) | 0.306 → 0.303 / 0.205 → 0.203 (negligible shrinkage) |

The pooled quartile contrast mixes in between-condition structure already seen on these draws (e.g. most PENGUIN S = 1
windows, a low-gain condition, fall in the lowest pooled decile); the post-hoc condition-specific contrast (+1.71 bpm) shows
most of it is within-condition. Validation and test rows with the same index share the initial noise z0 per seed and the
same `bootMedInstab` resample indices (amendment §2.5); no labels are shared and the patients are disjoint.

Sensitivities (non-gating): common window set Ω_∩ pooled +0.284, within +0.170 (both CIs > 0). Patient-macro Spearman
+0.62 (point only; pooled across conditions, so it contains between-condition structure, and averaging a patient's windows
inflates rank correlation — not comparable with the window-level +0.30). Frozen average-then-metric statistics
**+0.635 [+0.616, +0.653]** (≈ 2.1× the disjoint +0.303) and +3.45 bpm (≈ 1.5× the disjoint +2.23): the gap is an **upper
bound** on reuse contamination, because averaging over 32 partitions also legitimately reduces noise; the two were not
separated. Secondary targets scored with the frozen G_B predictor (no refit): relative gain Grel_B +0.059 [+0.047, +0.072];
Δ_width = |median(A) − y| − |median(A ∪ B) − y| (shares A with the features, not cross-fitted) +0.077 [+0.073, +0.081].

## 6. Model-specific heterogeneity
| model | Spearman (its 4 depths pooled) | within-condition mean |
|---|---|---|
| iMF | +0.214 [+0.202, +0.226] | +0.193 [+0.181, +0.206] |
| CD | +0.236 [+0.223, +0.248] | +0.221 [+0.208, +0.233] |
| PENGUIN | +0.390 [+0.378, +0.402] | +0.196 [+0.184, +0.209] |

All three positive. Within condition the models are similar in magnitude (0.19–0.22); CD is somewhat higher than iMF
(marginal CIs do not overlap; no paired test was run). PENGUIN's larger pooled value comes from its wide between-depth spread
of gains (G_B 0.50 at S = 1 → 2.16 at S = 8).

## 7. Leave-one-model-out transfer (secondary)
Fit on the other two models' validation rows, tested on the held-out model: iMF +0.199 [+0.186, +0.212] (within +0.193),
CD +0.233 [+0.221, +0.245] (within +0.218), PENGUIN +0.363 [+0.351, +0.375] (within +0.194) — at most 0.03 below the
pooled model. The relation carries over to each held-out model **among this project's three in-house generators** (same
corpus, same HR functional, one training seed each); it says nothing about external generators, and a dispersion-dominated
predictor is expected to transfer when the dispersion→gain slope is similar.

## 8. Cross-fitted mechanism diagnostic (uses the reference; not deployable; descriptive)
Per partition, over the 12 conditions: Spearman(ρ_{A_p}, G_{B_p}) — ρ from the pilot draws, G from the disjoint future
draws — mean **−0.993** (test; validation −0.992); swap ρ_B → G_A −0.993; depth-adjusted −0.993; within each model −1.00
on 4 depths. Condition points: ρ_A 0.56 (iMF S = 1) … 0.86 (PENGUIN S = 1) against G_B 2.85 … 0.50 bpm. Cross-fitting
changes little here: ρ is estimated over ~17k windows per condition, ρ_A ≈ ρ_B to ≤ 0.002, and both ρ and G use the same
reference. The strong negative association is largely expected from Var(e_k − e_l) = 2σ²(1 − ρ) (more shared error, less
within-window spread, smaller gain). Within-model −1.00 rests on 4 points (1/24 under exchangeability); the permutation
p (< 2e-5, 100,000 permutations) is at its resolution floor and treats the 12 structured conditions as exchangeable; the
bootstrap intervals only describe the stability of a fixed set of conditions. It re-analyses draws already used in
B3 / B3-BOOT. Associational, not causal.

## 9. Cost of observable features
Primary family: the 8 pilot samples at depth S (8·S NFE per window, plus one HR extraction per draw). This is free only if
an 8-draw consensus would be run anyway; against a single-draw deployment the pilot costs 8×. Extended depth-probe: the same
8 seeds again at 2S → 8·(S + 2S) = 24·S NFE (3× the pilot); its added value is **negligible** — within condition
+0.0007 [+0.0005, +0.0009] (statistically non-zero, practically nil), pooled +0.0005, iMF + PENGUIN only +0.0005, per model
≤ +0.0012; on validation its OOF MAE was 1.9705 vs 1.9701 for B3 on the same rows. No adaptive policy was evaluated.

## 10. Failures / preregistered criteria not met
No criterion of the amended rule failed. Qualifications that bound the verdict:
- **STRONG = reliably non-zero, not large.** Every STRONG criterion is a CI-excludes-0 threshold with no magnitude floor; at
  1,156 patients the CIs are about ±0.01, so they certify a non-zero association. The verdict under the literal frozen rule
  (with its invalid averaging) was not computed.
- **The frozen §8 aggregation was invalid** (partition averaging re-couples A and B) and was replaced before any result (D1).
- **Effect size is modest:** the predictor weakly orders windows by future gain (within-condition Spearman 0.20, R² 0.09);
  it does not predict individual windows' gain.
- **Largely expected by construction.** By the triangle inequality G_B ≤ mean_{k∈B}|Y_k − median_B|, i.e. the target is
  bounded by the future draws' spread, and pilot and future draws are exchangeable, so pilot dispersion predicting G_B is
  expected. How much of the bound is realised depends on where the reference lies relative to the sample cloud, which
  reference-free features cannot see; accordingly the relative gain is barely predicted (Grel_B +0.059). M1 shows that
  per-window generative spread is heterogeneous and repeatable across draws; it does **not** show that pilots identify
  windows where the consensus is wrong (its bias). SD alone captures ~92 % of the within-condition signal (0.187 of 0.203);
  the full model adds +0.017 over SD and +0.047 over MAD.
- **Allocator relevance is limited.** The policy-relevant marginal benefit of adding the 8 future draws to the pilot,
  Δ_width, is only weakly associated with the prediction (+0.077, and that value is not cross-fitted).
- **Possible outlier driver, not analysed:** G_B is heavy-tailed (about −15 … +30 bpm) and SD beats MAD, so gross R-peak
  detection failures in some draws (HR doubling / halving, which the median rejects) may drive part of the signal — a
  property of the HR functional rather than of generator uncertainty. The mean-based quartile contrast is sensitive to these
  tails.
- **Evaluation population:** Ω_HR retains 85–98 % of test windows per condition (2–15 % excluded because the reference or
  any of the 16 draws has no finite HR). Among windows with finite pilots and reference, 1–4 % (mean over partitions) are
  excluded only because a future draw is non-finite — a selection on future outcomes unavailable at deployment. Windows where
  HR extraction fails, possibly the riskiest ones, are not covered.
- **What the CIs do not include:** they resample patients and condition on the frozen predictor, one training seed per
  generator, the fixed 16 draws per window and the fixed partition list.
- Only |A| = |B| = 8 was tested; the value of smaller pilots is unknown. The extended depth-probe family did not help.
  The test draws were not fresh at condition level; the within-condition result is the new evidence. Sign decision (τ)
  omitted by design.

## 11. What this establishes
Under strict pilot/future sample splitting, validation-only fitting and a single patient-clustered test evaluation on 1,156
held-out patients: **reference-free statistics of 8 pilot samples — chiefly their dispersion — carry reliable, modest
information about the consensus advantage (G_B) of 8 independent future samples**, within every one of the 12 (model, depth)
conditions and for all three in-house generators. Pooled over conditions (which includes between-condition structure already
seen on these draws), mean observed G_B is 3.23 bpm in the top predicted quartile vs 1.00 bpm in the bottom (difference
+2.23 [+2.13, +2.33]); with condition-specific cut-offs (post-hoc) the difference is 1.71 bpm. This is a **necessary, not a
sufficient,** condition for a functional-risk allocator.

## 12. What this does NOT establish
That any allocation policy helps (none was run; the policy-relevant Δ_width is only weakly predicted); that pilots detect
consensus bias or error (Grel_B +0.059); causality; that ρ is observable at test time (the mechanism diagnostic uses the
reference); generality beyond VitalDB ECG→HR and this project's three generators (EXP-D showed the RR and ABP functionals
are not usable); universal width superiority; that more diverse waveforms are intrinsically better; that the multi-feature
model is needed (SD alone is close); precise per-window gain prediction (R² 0.09); the value of pilots smaller than 8.

## 13. Recommendation for M2 / M3
The prerequisite association exists, but it is modest, dispersion-driven and only weakly tied to the policy-relevant
quantity. An allocation experiment **could test** whether it translates into a cost-matched benefit — e.g. draw 8 pilots and
spend a fixed extra budget on the highest predicted-gain windows versus uniformly, on held-out patients with the same
patient bootstrap, with (i) the one-feature SD rule as the primary candidate and the full model as comparator, (ii) Δ_width
(the marginal benefit of the extra draws) as the primary target, (iii) the usability gate retained, (iv) no depth-probe
features, and (v) an analysis of whether HR-detection outliers drive the gain. A synthetic phase diagram (shared vs
sample-specific error) would help set expectations but is secondary. Given the modest effect, starting the paper on the
existing ECG→HR results and treating the allocator as future work is also defensible. **HARD STOP — nothing further was run.**
