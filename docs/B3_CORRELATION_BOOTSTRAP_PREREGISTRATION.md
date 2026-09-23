# B3-BOOT — Clustered bootstrap of the functional-error-dependence mechanism (preregistration)

Frozen and pushed before any number of this analysis is computed. **No training, no new sampling.** Every quantity is
recomputed from arrays already on disk (`outputs/tt_expb_raw/`, `outputs/dw1_raw/`, `outputs/dw2_raw/`,
`outputs/rd1_detector/test_rpeaks.npz`, `outputs/sr1_eval/arm_I_seed42.npz`). Head commit at freeze time: `4e2b4d5`.

## Question
EXP-B3 (report `6a2c8a8`) observed, across the 12 (model, S) conditions at K = 16 on the VitalDB test set:

| observed in EXP-B3 / EXP-B post-hoc (exploratory, **not** the criterion of this document) | value |
|---|---|
| Spearman(mean cross-sample functional-error correlation ρ̄, consensus gain G) | −0.965 |
| Spearman(waveform pairwise RMS, G) | +0.867 overall; within-model −0.20 / +0.40 / +1.00 |
| Spearman(functional dispersion MAD, G) | +0.846 |

Those were computed once, on the full test set, with an ordinary i.i.d. Spearman p-value. **That p-value is treated as
exploratory and is not used as inferential evidence here.** This document fixes what would count as the association being
stable under patient resampling, before it is measured.

## Interpretation under test
> Width helps because the *downstream functional errors* of the samples are non-redundant, not because the generated
> waveforms look different.

## 1. Bootstrap unit and paired structure
- The independent unit is the **VitalDB test patient** (n = 1,156). No window-level bootstrap.
- Each replicate draws 1,156 patients with replacement and takes **all** windows of each drawn patient (a patient drawn
  twice contributes its windows twice).
- **The same resampled patient set is used for all 12 conditions in that replicate**, preserving the pairing across
  conditions; every per-condition quantity is recomputed from scratch inside the replicate.
- **5,000 replicates**, bootstrap seed **20260923** (recorded in the result file). If the runtime exceeds ~30 min the run
  falls back to 2,000 replicates and the actual number is reported.
- Windows keep the Ω_HR rule of EXP-B (reference HR must be defined).

## 2. Per-condition quantities recomputed inside every replicate
Condition j = (model, S), model ∈ {iMF, CD, PENGUIN(Euler)}, S ∈ {1, 2, 4, 8}, K = 16, seed-42 checkpoints.
Signed functional error of sample k on window c: `e_{j,k}(c) = T(x_{j,k}(c)) − T*(c)`, T = HR (neurokit), T* from the
target ECG.

- **ρ̄_j — mean cross-sample functional-error correlation.** *Exactly EXP-B3's definition, unchanged*: the window set is
  those windows where the reference HR and **all 16** sample HRs are finite; `ρ_kl = Pearson(e_k, e_l)` over those windows;
  `ρ̄_j = mean over the 120 pairs k < l`. (Implementation note: inside the bootstrap this is evaluated from per-patient
  sufficient statistics — patient sums of `n`, `Σe_k`, `Σe_k e_l` — which reproduces the same estimator exactly.)
  A Fisher-z average may be reported as a supplementary column; the primary stays the plain mean of Pearson r.
- **G_j — consensus gain**, EXP-B's definition and aggregation: per window, `I = mean_k |e_{j,k}|` and
  `C = |median_k T(x_{j,k}) − T*|`; both are aggregated patient-macro (per-patient nanmean, then mean over patients),
  and `G_j = I_j − C_j`.
- **MAD_j / SD_j** — within-window dispersion of the sample functionals (`median_k |Y_k − m|`, `SD_k Y_k`), patient-macro.
- **RMS_j — waveform diversity**, DW2-B's measure on the same fixed 2,000-window `linspace` subset: mean pairwise RMS
  between the 16 stored waveforms, patient-macro over the patients present in that subset.
- **FEAT_j — feature-space diversity** (secondary): the same pairwise distance in `paper_metrics.default_feature_map`
  space, recomputed from the stored waveforms (no new sampling).

## 3. Association statistics (Phase 2)
In each replicate b, over the 12 conditions:
- **Primary:** `r_b^err = Spearman(ρ̄_j, G_j)`.
- **Secondary:** `r_b^MAD = Spearman(MAD_j, G_j)`, `r_b^wave = Spearman(RMS_j, G_j)`, `r_b^feat = Spearman(FEAT_j, G_j)`.

Reported for each: the original point estimate (full sample), the bootstrap median, the 2.5 / 50 / 97.5 percentiles, and the
sign-consistency rate (share of replicates with `r_b < 0` for the error-correlation association, `r_b > 0` for MAD / RMS /
FEAT).

### Success criteria (frozen now)
- **C1 — primary:** the 95 % percentile interval of `r_b^err` lies entirely below 0 **and** the sign-consistency rate is
  ≥ 0.95. Failure ⇒ the functional-error-dependence mechanism is reported as **not established**, and the central
  interpretation reverts to a descriptive statement about the DW1/DW2 optima.
- **C2 — comparative:** `r_b^err` is more consistent than `r_b^wave`, judged by two preregistered comparisons: (i) the
  sign-consistency rate of `r_b^err` exceeds that of `r_b^wave`; (ii) the per-replicate difference
  `|r_b^err| − |r_b^wave|` has a 95 % interval excluding 0. If (i) holds and (ii) does not, the conclusion is "no evidence
  that waveform diversity is the better predictor", not "functional-error dependence is better".
- **C3 — within-model (Phase 3):** for each model separately, the share of replicates in which the four tested depths are
  ordered with a **negative** rank relationship between ρ̄ and G (Spearman over the 4 points < 0) is reported per model.
  No significance test is claimed on n = 4; the wording is fixed now as *"perfect monotonic ordering over the four tested
  depths"* and never *"proof"* or *"perfect correlation"*.

## 4. Effective-sample-size diagnostic (Phase 4, explanatory only)
For the equal-correlation **mean** estimator, `Var(ē) = σ²/K · [1 + (K−1)ρ]`, hence `K_eff = K / (1 + (K−1)ρ̄)`.
Per condition report K, ρ̄, K_eff, G, the median error and the mean individual error, and `Spearman(K_eff, G)` with the same
patient bootstrap. Conditions with `ρ̄ ≤ −1/(K−1)` (the expression degenerating) are flagged and excluded from that
Spearman rather than forced. **This is intuition for a mean estimator; our estimator is the median, so K_eff is not
claimed to be the variance of the actual estimator, and this is not the primary mechanism test.**

## 5. K = 2 anomaly diagnostic (Phase 5, post-hoc, no promotion to headline)
EXP-B2's preregistered diminishing-return hypothesis failed 9/9 and that verdict stands unchanged. The post-hoc
explanation to be checked is that at K = 2 `median(a, b) = (a + b)/2`, so the estimator degenerates to the arithmetic mean,
which AB1 measured to be the worse pooling rule. Using the stored 32-draw matrices for the three B2 conditions
(CD S = 1, iMF S = 2, PENGUIN S = 4) × seeds 42 / 1 / 2, compute for K ∈ {1, 2, 3, 4, 8, 16, 32} (partition average over
disjoint groups of K consecutive noise seeds) the HR error of **median**, **mean** and **20 % trimmed mean** pooling.
Expected if the explanation holds: median and mean coincide at K = 2 and separate for K ≥ 3, and the median's marginal
gain becomes monotone once K ≥ 2. Any other outcome is reported as it is.

## 6. What this analysis may and may not change
- It may upgrade or downgrade the **mechanism** statement (STRONG / MODERATE / WEAK) and the wording of the central
  interpretation.
- It may **not** change any preregistered verdict already recorded (DW2-B's failed HR-SD criterion, EXP-B2's failed
  diminishing-return criterion, EXP-A's rejected A-H1/A-H3). Those stand; `DW2_MECHANISM_REPORT.md` receives an appended
  correction note and keeps its numbers.
- Single seed (42) for the 12 conditions, VitalDB only, one functional (HR), one aggregation rule (median). These limits
  are restated in the report whatever the outcome.

Script `scripts/b3_bootstrap.py`; raw `artifacts/b3_bootstrap/{conditions.csv, bootstrap.json, k2_operators.csv}`;
report `docs/B3_FUNCTIONAL_ERROR_MECHANISM_REPORT.md`.
