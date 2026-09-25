# M2 — Cost-matched width-vs-depth action prediction: **FAILED** (NO-GO for M3)

Preregistration **`51e9343`** (`docs/M2_ACTION_VALUE_PREREGISTRATION.md`, with the audit), frozen policy **`065ea63`**
(validation only, before any test row), single test evaluation. Code `scripts/m2_run.py`, synthetic tests
`tests/test_m2_action_null.py`. Results in `artifacts/m2_action_value/` (audit.json, partition_manifest.json,
prereg_manifest.json, validation_action_values.csv, validation_cv.csv, frozen_policy.json, test_metrics.json, bootstrap.json,
model_specific.json, condition_specific.json, missingness.json, outlier_sensitivity.json, cost_accounting.json, figure.png).
Per-window rows stay in `outputs/m2_action_value/` (not committed). Test CIs: 5,000-replicate patient-clustered bootstrap
(1,156 patients, seed 20260925); every metric is computed inside each of the 32 partitions and then averaged. M1 was not
modified. No generator was trained.

## 1. Research question
Given the same additional generative vector-field NFE, can reference-free statistics of a 4-sample pilot predict whether
WIDTH (8 more samples at depth S) or DEPTH (4 more samples at 2S) gives the lower HR error, well enough that routing on the
prediction beats the best static choice per (model, depth)?

## 2. Why M1 was insufficient
M1 showed that pilot dispersion predicts the consensus gain G_B of future samples (within-condition Spearman 0.20, R² 0.09),
but G_B is not a decision quantity: it is bounded by future spread and says nothing about whether spending the same compute
on depth would be better. Its association with the policy-relevant Δ_width was only +0.077. M2 targets the decision directly.

## 3. Audit (`docs/M2_ACTION_VALUE_AUDIT.md`)
9 conditions (iMF, CD, PENGUIN × S ∈ {1, 2, 4}; depth action at 2S). Validation 289 patients / 4,822 windows, test 1,156 /
19,543, patient-disjoint; reference HR 100 % finite; all 16 draws present at S and 2S on both splits; seed k = row k
(z0 aligned across depths for all arms; CD re-noising is depth-specific, and M2 needs no pairing). Draw-level non-finite HR
0.2–5.3 % per condition. No test label was used for any design decision.

## 4. Frozen 4 / 8 / 4 design
32 partitions of the 16 seed IDs into pilot A (4, at S), width W (8, at S) and depth D (4, at 2S), pairwise disjoint
(seed 20260925, manifest committed with the preregistration). The pilot never enters an action estimator. All metrics are
computed within a partition and then averaged (M1 lesson); a synthetic null (identical per-window distributions) gives a
within-condition Spearman < 0.03 through the full pipeline and a heterogeneous-dispersion positive control is detected
(both tests pass; the design's own signal in that control is ~0.085 because per-window ΔQ is very noisy).

## 5. Equal-NFE action definition
Width 8·S, depth 4·2S = 8·S, pilot 4·S, total 12·S vector-field NFE per window. Action estimate = median of the finite
action draws, else the frozen validation-median reference HR (73.317 bpm). ΔQ = L_D − L_W (> 0: width better).

## 6. Validation-only policy fitting (`frozen_policy.json`, `validation_cv.csv`)
| policy | frozen choice | validation policy MAE (patient-macro) |
|---|---|---|
| B0-W / B0-D | always width / always depth | 5.826 / 6.219 |
| B1 global static | WIDTH | 5.826 |
| **B2 condition static** | WIDTH in 8 conditions, **DEPTH for PENGUIN S = 1** (5.058 vs 4.971) | — |
| **P1 SD ridge** [log1p SD(A), log S] | α = 0.01 (all α identical); intercept +0.391, SD coefficient **+0.081**, log S +0.049 (standardised) | OOF 5.826 — **every OOF prediction > 0 → always width** |
| P2 SD threshold | t = −∞ (always width); every finite threshold was worse (5.837 … 6.187) | 5.826 (grouped OOF 5.826) |
| P3 full ridge | α = 100; ~99.99 % width | OOF 5.826 |
| oracle (descriptive) | min(L_W, L_D) | 4.763 |

The sign of the SD coefficient matches M1 (more pilot dispersion → width more favourable), but the slope is ~0.08 bpm per SD
of the feature against a mean width advantage of +0.39 bpm: no pilot moves the predicted advantage anywhere near zero.

## 7. Informativeness / deployability
Validation gate (per condition, 2,000-replicate patient bootstrap): both action estimators beat the constant by 6.7–8.1 bpm
and track the reference (Spearman 0.67–0.75) in all 9 conditions — **no condition flagged**. Deployable population: reference
and the 4 pilots finite (no future-draw selection); ~167,500 test units per partition. Future action sets with zero finite
draws (fallback used): ≤ 0.04 % per condition; partially finite 0.3–4.8 %.

## 8. Held-out test action-value prediction
| statistic (mean over 32 partitions) | value [95 % CI] |
|---|---|
| Spearman(pred ΔQ, ΔQ), **within condition** | **+0.0087 [+0.0064, +0.0110]** |
| Spearman, pooled | −0.0112 [−0.0137, −0.0089] |
| Pearson / R² | +0.013 [+0.010, +0.016] / +0.0001 [−0.0001, +0.0002] |
| top − bottom predicted quartile, mean ΔQ | +0.16 [+0.11, +0.20] bpm |
| decision accuracy (units with ΔQ ≠ 0; ties 1.5 %) | 0.562 [0.561, 0.564] — the share of windows where width is better, since P1 always picks width |
| P3 within-condition Spearman | +0.0051 [+0.0027, +0.0074] |

The within-condition association is statistically non-zero and in the expected direction but practically nil (compare M1's
+0.20 for G_B): the width-vs-depth action advantage is essentially not predictable from the pilot.

## 9. Adaptive vs condition-static policy (headline)
| policy | test MAE (patient-macro) | width decisions |
|---|---|---|
| always width (B0-W) = global static (B1) | 5.788 [5.420, 6.174] | 100 % |
| always depth (B0-D) | 6.145 [5.789, 6.524] | 0 % |
| **condition static (B2)** | **5.773 [5.408, 6.161]** | 89.6 % |
| SD threshold (P2) | 5.788 | 100 % |
| **SD ridge (P1)** | **5.788** | **100 %** |
| full ridge (P3) | 5.788 | 100 % |
| oracle (not attainable) | 4.726 [4.379, 5.088] | — |

**Δ_policy = MAE(P1) − MAE(B2) = +0.015 [+0.006, +0.024] bpm — the adaptive rule is significantly worse than the
condition-static baseline.** P1 − always depth −0.357 [−0.378, −0.336]; P1 − always width 0 (identical); P2 − P1 = 0.

## 10. Model / depth heterogeneity
| | P1 − B2 |
|---|---|
| iMF, CD (all depths), PENGUIN S = 2, 4 | 0 (both policies choose width) |
| **PENGUIN S = 1** (B2 = depth) | **+0.179 [+0.079, +0.283]** (P1 5.072 vs B2 4.893) |
| model level: iMF / CD / PENGUIN | 0 / 0 / +0.048 [+0.020, +0.077] — 0 of 3 models negative |

The whole difference is the one condition where knowing the (model, depth) already tells you to use depth; the pilot does
not recover that choice.

## 11. Oracle gap
Regret: P1 1.063 [1.019, 1.108] bpm, B2 1.047 [1.006, 1.091] bpm. Oracle-gap recovery **−0.014 [−0.023, −0.006]**: the
adaptive rule recovers none of the 1.05-bpm gap between condition-static and the per-window oracle. The oracle gap is
dominated by per-window noise in which action happens to win; it is not attainable.

## 12. Missing HR outcomes
Width sets: all finite 95–98 %, partial 1.6–4.8 %, none 0 %. Depth sets: all finite 95–99.7 %, partial 0.3–4.7 %, none
≤ 0.04 %. Sensitivity with the ≥ 50 %-finite rule: P1 − B2 = +0.014 [+0.005, +0.023] — unchanged.

## 13. HR-extractor outlier sensitivity (secondary)
Pilot flags (range [30, 200] bpm, or a pilot HR > 25 % from the pilot median): 22.7 % of test windows (23.3 % on validation).

| subset | MAE P1 | MAE B2 | P1 − B2 |
|---|---|---|---|
| all deployable | 5.788 | 5.773 | +0.015 [+0.006, +0.024] |
| pilot unflagged | 4.633 | 4.651 | −0.018 [−0.024, −0.012] |
| pilot flagged | 10.684 | 10.523 | +0.161 [+0.130, +0.190] |

Flagged windows are much harder (MAE ~10.6 vs 4.6), and PENGUIN S = 1's depth advantage — the one static decision that
beats always-width — comes from flagged windows; on clean windows width is better there too. This is a *post-hoc*
descriptive pattern of the preregistered secondary analysis (HR-extractor failures drive the depth advantage), not a rule
M2 may adopt.

## 14. Actual compute accounting (`cost_accounting.json`)
Pilot 4·S, width 8·S, depth 4·2S = 8·S, total 12·S vector-field NFE. Measured on the RTX 5090 (one window; measured while
the test stage was using the CPU, so individual values are noisy): **sequential batch-1** width and depth take about the same
time (e.g. PENGUIN S = 1: 302 vs 298 ms; iMF S = 4: 733 vs 755 ms); **batched**, width is ~2× faster than depth (e.g.
PENGUIN S = 4: 144 vs 296 ms; iMF S = 1: 22 vs 45 ms) because batched latency scales with the number of sequential steps,
not with the batch size. The precise term is "equal future vector-field NFE"; runtime is equal only in the sequential
batch-1 setting. No FLOPs were measured.

## 15. Failed preregistered criteria
- FAILED route (a): Δ_policy CI lower bound +0.006 > 0 — P1 significantly worse than condition-static.
- FAILED route (c): no pilot-conditioned rule improves — P1, P2, P3 all equal always-width (+0.015 vs B2).
- STRONG items not met: (1) CI below 0, (2) ≤ −0.10 bpm, (3) ≥ 2 models negative (0 of 3). Item (4), within-condition
  association CI > 0, holds but the association is +0.009.
- Route (b) (non-positive within-condition association) did not fire.

## 16. What M2 establishes
With equal future vector-field NFE, width (8 samples at S) beats depth (4 samples at 2S) in 8 of 9 conditions on validation
and on test (per-condition test values in `prereg_manifest.json`; PENGUIN S = 1 is the exception on both), by 0.35 bpm on average (always-width 5.788 vs always-depth 6.145), and **a 4-sample pilot does not predict when
depth would be better**: the within-condition association is +0.009, every pilot-conditioned rule collapses to always-width,
and the adaptive policy is slightly but significantly worse than the static (model, depth) rule (+0.015 bpm), whose one
depth choice (PENGUIN S = 1) the pilot cannot recover.

## 17. What M2 does NOT establish
That no pilot statistic could ever predict the action advantage (only the preregistered SD-based, full-feature and
threshold rules were tested, with 4 pilots); anything about other functionals, corpora or generators; that width and depth
cost the same wall-clock (batched width is ~2× faster); that routing on HR-extractor outlier flags would help (a post-hoc
pattern, §13); that the oracle gap is attainable.

## 18. Go / no-go for M3
**NO-GO.** M2 FAILED through routes (a) and (c): the width-vs-depth action advantage is not predictable from pilot
statistics beyond what the (model, depth) identity already gives. Per the brief, the adaptive functional-risk-allocator line
stops here; no neural controller, no new features. The usable conclusion for the paper is the static one: at equal future
NFE, sample wide (and for PENGUIN S = 1, deeper) — a condition-level recipe, not a per-window adaptive policy.
