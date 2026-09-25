# M2 — Cost-matched width-vs-depth action prediction: preregistration

Frozen and pushed **before any M2 validation or test predictive result** (no action loss, action advantage, policy loss or
predictive statistic has been computed). Audit: `docs/M2_ACTION_VALUE_AUDIT.md`. Head at freeze: `2d86f9e`. M1 is not
modified. Seed for everything new: **20260925**.

## 1. Question
Given the same additional generative vector-field NFE, can reference-free statistics of a 4-sample pilot predict whether
**WIDTH** (8 more samples at depth S) or **DEPTH** (4 more samples at depth 2S) yields the lower downstream HR error — and
does acting on that prediction beat the best static choice per (model, depth)? This tests the decision-theoretic
prerequisite of a functional-risk allocator; no adaptive allocator is built.

## 2. Data and conditions
V1 VitalDB, validation 289 patients / 4,822 windows (fitting only), test 1,156 patients / 19,543 windows (evaluated once).
**9 conditions**: iMF, CD, PENGUIN (seed-42) × base depth S ∈ {1, 2, 4}; depth action at 2S ∈ {2, 4, 8}. Banks: the M1
validation bank and the existing test banks (rows = noise seeds 0–15), HR functional `v1_evaluate._hr`. Every HR value
(banks, references) is rounded to 1e-3 bpm before any computation (as M1 D7). No generator training, no new HR extractor.

## 3. 4 / 8 / 4 disjoint partitions
For each of **32 partitions** p the 16 seed IDs are split into A_p (4 pilot), W_p (8 width), D_p (4 depth), pairwise
disjoint, union {0,…,15}. Rule: `rng = numpy.random.default_rng(20260925)`; repeat `perm = rng.permutation(16)`;
A = perm[0:4] (stored in this order), W = sorted(perm[4:12]), D = sorted(perm[12:16]); skip a triple already drawn; until 32.
Saved to `artifacts/m2_action_value/partition_manifest.json` in this commit. The same list is used for every condition,
window and split.
- Pilot: A_p at depth **S**. Width outcome: W_p at depth **S**. Depth outcome: the D_p seed IDs evaluated at depth **2S**.
- No trajectory pairing is required (for CD only z0 aligns across depths; recorded in the audit).

## 4. Equal future vector-field NFE; action estimators and losses
Width cost 8·S, depth cost 4·(2S) = 8·S, pilot 4·S (same for both), total 12·S NFE per window. The pilot is **never** part
of an action estimator (no sunk-compute advantage for width).
- Action estimate = median of the action's **finite** HR values; if **none** is finite, the frozen fallback
  **c_val = median of the validation reference HR over all 4,822 validation windows**. Sensitivity: an action estimate is
  used only if ≥ 50 % of its draws are finite (≥ 4 of 8 for width, ≥ 2 of 4 for depth), otherwise c_val.
- L_W(i) = |median(W_{i,S}) − y_i|, L_D(i) = |median(D_{i,2S}) − y_i|. **Primary target ΔQ(i) = L_D(i) − L_W(i)**
  (> 0 → width better). Never replaced by G_B, spread, relative gain or single-sample error.
- Oracle (descriptive only): min(L_W, L_D).

## 5. Population (deployable)
Per partition and condition: every window whose reference HR is finite and whose **4 pilot HRs are all finite** (known
before the decision). Future draws are never used for inclusion. Reported per condition and action: share of action sets
with all draws finite / partially finite / none finite.

## 6. Pilot features (from A_p at depth S only; no reference, no model identity, no depth probe)
- **Primary (P1):** z1 = log1p(SD(A)) (SD with ddof 1), z5 = log S.
- Secondary simple: z2 = log1p(MAD(A)) (median |Y − median|), z3 = log1p(IQR(A)) (numpy linear percentiles), z4 = median(A).
- **Full family (P3):** median, MAD, IQR, SD, range, mean and median of the 6 pairwise |differences|, bootstrap median
  instability (SD, ddof 1, of the medians of 64 resamples of size 4 with replacement;
  `default_rng([20260925, cond_idx, window_row_index, part_idx])`), split-half instability |median(A[0:2]) − median(A[2:4])|
  in stored order, log S; `log1p` on every spread feature (MAD … split-half).
- cond_idx order: iMF → CD → PENGUIN, S 1 → 2 → 4.

## 7. Policies (all fitting on validation only)
- **B0-W** always width; **B0-D** always depth.
- **B1-GLOBAL-STATIC:** the action with the lower validation policy MAE, applied everywhere (tie → width).
- **B2-CONDITION-STATIC:** per condition, the action with the lower validation policy MAE within that condition; 9 frozen
  decisions (tie → width). **The key baseline.**
- **P1-SD-RIDGE (primary):** StandardScaler + Ridge on [z1, z5] predicting ΔQ; action = width if prediction > 0 else depth.
  α ∈ {0.01, 0.1, 1, 10, 100}, GroupKFold(5) by patient over validation rows (rows = partition × condition × window; all rows
  of a patient in one fold); for each α the out-of-fold predictions are turned into actions and scored by the **policy MAE**
  (§8); smallest policy MAE wins, ties → smallest α; refit on all validation rows. Not selected by Spearman or MSE.
- **P2-SD-THRESHOLD:** width if z1 > t else depth. Candidate t: the validation quantiles of z1 (pooled validation rows) at
  probabilities 0.05, 0.10, …, 0.95, plus −∞ (always width) and +∞ (always depth). t = argmin of the validation policy MAE
  (ties → smallest t). The grouped out-of-fold policy MAE of this selection procedure (t chosen on training folds) is
  reported for comparison with P1.
- **P3-FULL-RIDGE:** as P1 with the full family (§6).
- No other model class.

## 8. Metrics — computed inside each partition, then averaged over the 32 partitions (never average first)
For each partition p: features from A_p, predictions, actions, L_W,p, L_D,p, ΔQ_p, and every metric are computed on that
partition's units (condition × window, each unit once); the headline is the mean of the 32 partition-level values. Averaging
predictions, targets or features over partitions before a metric is forbidden (M1 D1).
- **Policy MAE** (patient-macro): per patient, mean loss over the patient's units in all 9 conditions; mean over patients.
  Model-specific / condition-specific versions restrict the units.
- **Headline: Δ_policy = MAE(P1) − MAE(B2)** on test (< 0 = improvement). Paired differences of P1 against B0-W, B0-D, B1,
  B2; of P2 and P3 against B2; and P2 − P1.
- Width-decision fraction; decision accuracy P(sign(pred ΔQ) = sign(ΔQ)) over units with ΔQ ≠ 0 (descriptive; tie share
  reported); Spearman (pooled; **within-condition** = mean of the 9 per-condition values), Pearson, R² of pred ΔQ vs ΔQ;
  mean ΔQ in the top vs bottom quartile of pred ΔQ (inverted-CDF cut-offs per partition); regret = MAE(policy) − MAE(oracle);
  oracle-gap recovery = 1 − (MAE(P1) − MAE(oracle)) / (MAE(B2) − MAE(oracle)), reported only if the denominator > 0.
  Oracle performance is not called attainable.

## 9. Inference
Patient-clustered bootstrap on test: 5,000 replicates, `default_rng(20260925)`; each replicate draws the 1,156 test patients
with replacement; a patient's multiplicity weights all its units in every condition and partition; every statistic is
recomputed inside each partition and averaged over partitions within the replicate; 95 % percentile CIs; CIs condition on
the frozen policies. No multiplicity correction; §12 is the only decision rule.

## 10. Informativeness gate (validation, per condition, before test)
For each condition, the width and the depth action estimators (with fallback) are compared with c_val on validation:
MAE(action) − MAE(c_val) (paired patient bootstrap CI entirely below 0) and Spearman(action estimate, reference) (CI lower
bound > 0). A failing condition is flagged in `frozen_policy.json` before test; it stays in the primary analysis and the
verdict is additionally reported without flagged conditions.

## 11. HR-extractor outlier audit (secondary; rules fixed here)
- **Range flag:** a pilot HR outside **[30, 200] bpm** (the project convention, `v1_build_vitaldb.py:54`).
- **Pilot robust-outlier flag:** some pilot HR deviates from the pilot median by more than **25 % of the pilot median**
  (a design rule aimed at halving / doubling failures; not tuned).
- Flagged window = either flag. Validation flag rates are recorded in `frozen_policy.json` before test. On test, Δ_policy
  and the P1 − B2 comparison are reported for all deployable windows (primary), unflagged only, and flagged only.

## 12. Verdict (fixed; evaluated in this order on the primary population and fallback)
- Practical margin **δ_MAE = 0.10 bpm**, fixed now from the functional's resolution: HR = 60·fs·(n−1)/span; at ~75 bpm with
  ~5 beats in a 4-s window at 128 Hz, a one-sample change of the span moves HR by ≈ 0.2 bpm; a mean policy improvement below
  about half of that per-window resolution is treated as negligible.
- **FAILED** if (a) Δ_policy CI lower bound > 0 (P1 significantly worse than B2); or (b) the within-condition Spearman of
  P1's predicted vs observed ΔQ has point estimate ≤ 0; or (c) no pilot-conditioned rule improves on B2: the point estimates
  of MAE(P1) − MAE(B2), MAE(P2) − MAE(B2) and MAE(P3) − MAE(B2) are all ≥ 0.
- **STRONG** if not FAILED and all of: (1) Δ_policy CI upper bound < 0; (2) Δ_policy point ≤ −δ_MAE; (3) the model-specific
  Δ_policy point estimate is < 0 for ≥ 2 of 3 models; (4) within-condition Spearman CI lower bound > 0.
- **PARTIAL** otherwise, labelled with the failing items (direction favourable but CI crosses 0; one-model driven; below the
  practical margin; association positive but no policy improvement; only P2 / P3 improve). P3 never rescues P1.
- **P1 vs P2:** if MAE(P2) − MAE(P1) has a CI including 0, or |point| < δ_MAE / 2, the threshold rule is preferred.
- **GO to M3** only if STRONG, or PARTIAL where the only failing STRONG item is (1) (uncertainty), Δ_policy point ≤ −δ_MAE,
  and ≥ 2 models have negative model-specific Δ_policy. Otherwise NO-GO; if FAILED through (b) or (c), the adaptive-allocator
  line stops.

## 13. Cost accounting
Pilot 4·S, future action 8·S (width 8·S, depth 4·2S), total 12·S vector-field NFE. Measured latency of the width and depth
actions (sequential batch-1 and batched) per model and S. No FLOPs infrastructure exists. The precise term is **"equal
future vector-field NFE"**; equal runtime is not claimed unless measured.

## 14. Order of work and artifacts
This commit (audit, preregistration, partition manifest) → code + synthetic-null unit test (independent width / depth
outcomes must give per-partition statistics ≈ 0 and no policy advantage) → validation fitting → commit
`frozen_policy.json` (transforms, α, coefficients, threshold, static actions, fallback, gate, flag rates, partition-manifest
hash, validation CV, code / input hashes) → single test evaluation → figure, report → commit → **HARD STOP** (no M3).
Artifacts in `artifacts/m2_action_value/`: audit.json, partition_manifest.json, prereg_manifest.json,
validation_action_values.csv, validation_cv.csv, frozen_policy.json, test_metrics.json, bootstrap.json, model_specific.json,
condition_specific.json, missingness.json, outlier_sensitivity.json, cost_accounting.json, figure.png. Large per-window
arrays in `outputs/m2_action_value/` (not committed). Report `docs/M2_ACTION_VALUE_REPORT.md` (18 sections of the brief).
