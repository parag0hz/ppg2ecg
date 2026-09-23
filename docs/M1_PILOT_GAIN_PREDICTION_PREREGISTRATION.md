# M1 — Do pilot-sample statistics predict the gain from *future* independent samples? (preregistration)

Frozen and pushed **before any M1 predictive number exists**, and before the validation sample bank is generated.
No training of any generator, no new architecture, no new loss, no new functional. Audit: `docs/M1_PILOT_GAIN_PREDICTION_AUDIT.md`
(+ `artifacts/m1_pilot_gain_prediction/audit.json`). Head commit at freeze time: `b1eb66a`.

Completed work that must not be touched: DW1, DW2, EXP-A, EXP-B, RDDM, PPGFlowECG, EXP-D (respiration, ABP).

## 1. Question

> Can observable statistics of a small set of stochastic generator samples predict the benefit of **additional,
> independent** sampling, without using the reference at test time?

M1 is **not** an adaptive allocator. It tests only whether the prediction signal exists under strict sample splitting and
patient-level held-out evaluation. The deployable constraint: every predictor input is a function of generated
functional values alone.

## 2. Data, conditions, and the sample banks

- Corpus: **V1 VitalDB**, manifest `data/manifests/split_v1_vitaldb_seed42.json` (patient-level, split seed 20260917).
  Validation = 289 patients / 4,822 windows; test = 1,156 patients / 19,543 windows. Train windows are not used.
- Models (frozen seed-42 checkpoints, sha256 in the audit): **iMF** (`armI`), **CD** (`armD`), **PENGUIN (Euler)** (`armC`).
- Depths **S ∈ {1, 2, 4, 8}** → **12 conditions** (3 models × 4 depths).
- Draws: the **first 16** stochastic draws, noise seeds **0 … 15**, for every condition and both splits (even where the
  existing test bank has K = 32), so that all conditions are treated identically.
- Functional `Y_{i,S,k}` = HR of the generated ECG, `y_i` = HR of the target ECG, both by the unchanged DW/EXP-B
  implementation `scripts/v1_evaluate.py::_hr` = `R.hr_bpm(R.detect_rpeaks(row, 128, "neurokit"), 128)`.
- **Ω_HR inclusion rule (EXP-B's, unchanged):** a window enters a condition iff the reference HR and **all 16** draws of
  that condition are finite. Applied per condition, identically on validation and test.
- **Validation bank generation (executed after this document is pushed, before any predictor is fit):**
  `vm1_evaluate.load("val")` → `dw1_depth_width.make_sampler` (unchanged samplers) → seeds 0–15 → S ∈ {1,2,4,8} →
  `v1_evaluate.hr_batch`. Outputs `outputs/m1_pilot_gain_prediction/val_hr_{arm}_S{S}.npy` (16, 4822) and
  `val_ref.npz` (`ref_hr`, `pid`). Cost 1.16 M NFE per model. **No test window is touched during generation.**

## 3. Pilot / future partitions (frozen before any predictive metric)

From the 16 preregistered draws, **32 balanced 8/8 partitions**: `A` = pilot (|A| = 8), `B` = future (|B| = 8),
`A ∩ B = ∅`, `A ∪ B = {0..15}`. The 32 subsets `A` are drawn once, uniformly without replacement from the C(16,8) = 12,870
possible subsets, by `numpy.random.default_rng(20260924)`, and are **shared by every condition and every window** (so the
same partition structure is applied everywhere). Within a partition, `A` keeps the order produced by the generator; this
order defines the split-half feature (§5.9). The list is saved **before any predictive metric is computed** to
`artifacts/m1_pilot_gain_prediction/split_manifest.json`.

## 4. Targets

For condition j, window i, partition p:

**Primary — held-out consensus gain (uses only B and the reference):**

```
I_B = mean_{k∈B} |Y_{i,S,k} − y_i|
C_B = |median_{k∈B} Y_{i,S,k} − y_i|
G_B = I_B − C_B
```

**Secondary (descriptive):** `Grel_B = G_B / max(I_B, ε)`, with the preregistered stability floor **ε = 1.0 bpm**.
Grel is never the primary result.

**Secondary, policy-relevant (A appears in the baseline, therefore not primary):**
`E_A = |median(A) − y_i|`, `E_AB = |median(A ∪ B) − y_i|`, `Δ_width = E_A − E_AB`.
`Δ_width` may not replace `G_B` under any circumstance.

## 5. Pilot features (primary family — generated functionals in A only)

No reference, no error, no residual, no target-derived quantity, no patient ID, no model ID. With `m = median(A)`:

1. `median(A)` = m
2. `MAD(A)` = `median_{k∈A} |Y_k − m|`
3. `IQR(A)` = `percentile(A, 75) − percentile(A, 25)` (numpy linear interpolation)
4. `SD(A)` = `np.std(A, ddof=1)`
5. `range(A)` = `max(A) − min(A)`
6. `meanPair(A)` = mean over the 28 pairs k<l of `|Y_k − Y_l|`
7. `medPair(A)` = median over the same 28 pairs
8. `bootMedInstab(A)` = `np.std(medians, ddof=1)` over **R = 64** bootstrap resamples of size 8 drawn with replacement
   from A by `numpy.random.default_rng([20260924, cond_idx, win_idx, part_idx])` (counter-based, deterministic and
   reproducible; `cond_idx` is the index of the condition in the frozen order iMF→CD→PENGUIN × S 1→2→4→8, `win_idx` the
   row index within the split, `part_idx` the partition index 0–31)
9. `splitHalfInstab(A)` = `|median(A[0:4]) − median(A[4:8])|` in the partition's stored order
10. `log(S)` = natural log of the depth

**Transforms, fixed now:** features 2–9 are non-negative spread quantities and are transformed by `log1p`; features 1 and
10 are used as-is. Then `sklearn.preprocessing.StandardScaler`, **fit on validation rows only** and frozen for test.

## 6. Primary predictor and baselines

`sklearn.pipeline.Pipeline([StandardScaler(), Ridge(alpha)])` predicting `G_B`, pooled across all 12 conditions, with
**no model-identity feature** (log S is the only condition descriptor).

- Fitting rows are **partition-level** (condition × window × partition) from validation only.
- α selected by **GroupKFold(n_splits=5) grouped by patient** on validation; every partition and every condition of a
  window stays inside one fold. Grid frozen: **α ∈ {0.01, 0.1, 1, 10, 100}**; criterion = CV **MAE of G_B**; ties → smallest α.
- Baselines, identical pipeline and α search, differing only in inputs:
  **B0** constant = validation mean `G_B`; **B1** MAD-only; **B2** SD-only; **B3** full 10-feature model (primary).
  **B4** extended depth-probe model (secondary, §7).
- Frozen before test: feature list, transforms, scaler statistics, α, coefficients → `frozen_predictor.json`.

## 7. Extended depth-probe family (SECONDARY only)

Paired seeds are exact for iMF and PENGUIN and prefix-paired for CD (audit §6). For pilot seeds k ∈ A and the depth pairs
(S, 2S) ∈ {(1,2), (2,4), (4,8)}: `D_k = Y_{i,2S,k} − Y_{i,S,k}`, features `median(D)`, `MAD(D)`, `median(|D|)`,
`|median(Y_{2S,A}) − median(Y_{S,A})|`, appended to the primary 10. Reported separately as **EXTENDED / DEPTH-PROBE**,
never merged into the primary model, and always with its extra compute (§11).

## 8. Aggregation — no partition pseudoreplication

Headline statistics first **average predictions and targets over the 32 partitions** within each (condition, window),
then compute population-level metrics on those averaged values. Partition-level statistics appear only as a sensitivity
analysis. The bootstrap unit is always the **patient**.

## 9. Test procedure (executed once)

No refitting, no recalibration, no feature or α change. Metrics on VitalDB test:

1. **Primary: Spearman(predicted `G_B`, observed `G_B`)**, pooled over the 12 conditions on partition-averaged values.
2. Pearson (secondary), MAE of predicted `G_B`, R² (descriptive).
3. **Quartile contrast:** mean observed `G_B` in the top vs bottom quartile of predicted gain (quartiles computed on the
   pooled test predictions).
4. Per-model Spearman (iMF, CD, PENGUIN) as heterogeneity analysis; the pooled result stays primary.
5. **Sign decision is omitted**: no defensible practical τ for HR consensus gain exists anywhere in this project, and the
   brief forbids inventing one after seeing results.

**Uncertainty:** patient-clustered bootstrap, **5,000 replicates, seed 20260924**; a drawn patient contributes all of its
windows, conditions and partition observations; 95 % percentile CIs.

## 10. Secondary analyses

- **Leave-one-model-out:** fit on validation rows of the other two models, evaluate on test rows of the held-out model;
  all three directions; no model-ID feature. Cannot rescue a failed primary.
- **Cross-fitted mechanism diagnostic (not deployable — uses the reference):** per condition, `ρ_A` = mean over the 28
  pairs k<l ∈ A of `Pearson(e_k, e_l)` across the condition's Ω_HR windows, with `e_k = Y_{i,S,k} − y_i`; paired with the
  held-out `G_B` (patient-macro over the same windows); and the swap `ρ_B → G_A`. Report `Spearman(ρ_A, G_B)` over the 12
  conditions, pooled and within model, patient-clustered bootstrap. **Not causal.**
- **Informativeness sanity check (ECG→HR):** on the M1 test subset, compare the K = 16 consensus HR estimator against
  (a) the **validation-defined constant** baseline (median of validation reference HR, the project's training/validation-
  median convention from EXP-D Part B) and (b) the PPG-peak baseline `v1_evaluate.ppg_hr` (already aligned). No new
  regressor is trained. If the consensus estimator fails to beat the constant baseline on this subset, M1 is flagged
  **UNINTERPRETABLE** and stops.

## 11. Cost accounting

Primary family: `K_pilot × S` NFE per window (K_pilot = 8). Extended family additionally needs the paired `2S`
evaluations: `K_pilot × (S + 2S)`. No adaptive policy is evaluated in M1; an extended predictor is never called better
without reporting this overhead.

## 12. Preregistered verdicts

**STRONG SIGNAL** iff all four hold:
1. pooled primary Ridge Spearman(predicted, observed `G_B`) > 0 on test;
2. its patient-bootstrap 95 % CI lower bound > 0;
3. observed `G_B` in the top predicted quartile > bottom quartile, patient-bootstrap 95 % CI of the difference excluding 0;
4. at least **2 of 3** models show positive model-specific Spearman.

**PARTIAL SIGNAL** if the pooled direction is positive but its CI crosses 0, **or** the pooled result is significant but
only one model carries it, **or** B1 (MAD-only) matches B3 with no robust added value.

**FAILED** if the pooled association is ≤ 0, **or** top and bottom predicted quartiles do not differ, **or** prediction
collapses out of validation.

A failed M1 will be reported as failed. No target, feature, transform or α is changed after test evaluation, and no
feature is added afterwards. After M1 the work stops: no phase diagram, no allocator, no sequential policy, no new
domain, model or controller.

## 13. Seed register

| purpose | seed |
|---|---|
| partition list, bootstrap resampling, feature bootstrap | **20260924** |
| (inherited, unchanged) patient split | 20260917 |
| (inherited, unchanged) generator checkpoints | 42 |
| (inherited, unchanged) noise draws | 0 … 15 |

The project has no single global preregistration seed convention (DW1/AB1 used 20260911, B3-BOOT 20260923); the brief's
seed **20260924** is therefore adopted for everything new in M1 and recorded here before any result.
