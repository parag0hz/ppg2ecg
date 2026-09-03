# O1 — ECG Component-wise Conditional Extractability Map — PREREGISTRATION

**What ECG information is actually recoverable from PPG?**

Frozen before any probe is trained and before any validation window is scored. Never edited afterwards.

| | |
|---|---|
| Base commit | `801bf6074591e9fa310a93ef2c999bdbbc065a27` (Q1 report), clean tree |
| Upstream pins | PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`; A4 md5 `31c042d291052fbb6dc15263ad316be2` unchanged; C2 still deferred |
| Companion audit | `docs/O1_COMPONENT_TARGET_AUDIT.md` (four-site identity, target definitions, validity, variability) — written and committed with this document |
| Test subjects | `kjd`, `ssx` — **never loaded** |
| Environment | RTX 5090, torch 2.11.0+cu130, numpy 2.3.5, scipy 1.16.3, neurokit2 0.2.12, Python 3.13.9 |

---

## 0. What O1 is and is not

O1 builds **no generator**. There is **NO** generator training, **NO** flow training, **NO** attention, **NO**
adapter, **NO** test-subject access, **NO** C2, and **NO** method-novelty claim. It trains a family of small
PPG-only regression probes to ask which ECG components are recoverable, and then compares that map with the
*already frozen* generator behaviour from Q1.

### 0.1 Status declarations (frozen)

1. **O1 was designed after V1/R1/R2/R3/Q1.** It is **problem-discovery / diagnostic** evidence and is **not**
   independent confirmatory evidence for any earlier verdict.
2. **ECG-derived targets are TRAINING LABELS only.** No ECG waveform and no ECG-derived variable ever enters a
   probe input; the probe input is the frozen preprocessed PPG window and nothing else.
3. **Validation ECG (an0, k2s) is evaluation truth only** — never used for target scaling, never for checkpoint
   selection, never seen before all probes are trained and selected.
4. **Terminology**: the object measured is **operational conditional extractability** — the performance of this
   frozen probe family under this protocol. It is *not* information-theoretic observability. **A finite-capacity
   probe failure does not prove absence of information**, and the word "unobservable" is not used for any result.
5. Two development-validation subjects only (an0/k2s; four windows previewed before X4-0 are excluded from the
   frozen X4-0 subsets and are irrelevant here because O1 uses the R1 cohort).

### 0.2 The four questions

**Q1** Which ECG components are directly extractable from PPG on unseen subjects? · **Q2** Which are predictable
only from coarse rhythm/site information? · **Q3** Which extractable components does the frozen ECG generator
appear to use? · **Q4** How does extractability change with PPG site and controlled condition degradation?

---

## 1. Population (frozen, reused — no new cohort)

Exact R1 subject split (`r1_cohort.internal_dev_split`) and exact R1 balanced cohort
(`r1_cohort.cohort_positions`, salt `r1-global-rhythm-observability-v1`):

| role | subjects | per subject × site | rows |
|---|---|---|---|
| `probe_train` | fex, l38, n31, ngh, p5d, p9p, qm9, trh, tz8, w4p | ≤ 2,048 | 81,920 |
| `internal_dev` | u7y, e61 | ≤ 2,048 | 16,384 |
| `validation` | an0, k2s | ≤ 1,024 | 8,192 |
| test | kjd, ssx | — | **never loaded** |

Total 106,496 rows over **65,251 unique ECG windows** (four-site duplication, §2).

## 2. Four-site duplication (established in the audit, binding here)

`(subject, window_index)` identifies one ECG waveform, one R-peak train and one value of every scalar target
across sites — verified on 85,830/85,830 groups (waveform hash) and 7,168/7,168 deep checks. Therefore:

- the **independent observation unit is the ECG window**, not the npz row;
- the cluster bootstrap (§9) resamples ECG windows and keeps all of that window's site rows together;
- target variability (§3) is computed once per unique ECG window.

## 3. Targets, validity, variability, scaling

Defined and audited in `docs/O1_COMPONENT_TARGET_AUDIT.md` (T1–T9; frozen primitives only; median over valid GT
beats; QRS core `|τ| ≤ 80 ms` = M1's `CORE = 10`). All nine targets have 100 % validation validity and enter the
**PRIMARY** map. Scaling is `y_z = (y − median_train)/IQR_train` with train-only statistics
(`target_scaling.json`); no IQR is near zero. Results are reported in both physical/project units and
normalised units. The **event-timing row reuses the frozen R1 diagnostic** (F1@50/100/150/200, RR MAE) and is
labelled *R1 existing diagnostic*, not a new O1 probe.

## 4. Baselines (three, non-neural, per target)

| id | definition |
|---|---|
| **B0 GLOBAL MEDIAN** | predict the train median of the target |
| **B1 SITE-ONLY** | predict the train median **per site** (sternum/head/wrist/ankle) — tests whether site-specific cohort composition alone explains performance, given that the ECG target is duplicated across sites |
| **B2 RHYTHM-ONLY** | Ridge (`alpha = 1.0`, no search) on PPG-derived rhythm features only: detected PPG pulse count, median PPI, PPI IQR, valid-pulse flag, site one-hot. Pulses come from the frozen V1 detector `s1_audit.dsp_ppg_peaks`. **No PPG amplitude, no waveform samples, no morphology features.** Features standardised with train-only statistics |

B2 is the decisive baseline for Q2: it tests whether apparent morphology prediction is explained by HR / rhythm / site.

## 5. Primary raw-PPG probe

`ComponentGlobalTCN` (`src/ppg2ecg/evaluation/o1_targets.py`): R1's Global-TCN trunk — `Conv1d(1→64, k=1)` stem,
8 residual blocks of two `Conv1d(64→64, k=5, dilation d)` with GELU, dilations 1…128 — followed by **global
temporal mean-pooling** and a scalar `Linear(64→1)` head. Receptive field **2,041 ≥ 1,024**; **328,897 trainable
parameters ≤ 500,000**, identical for every target (only the regression target differs).

Input: the frozen preprocessed green PPG window `[B, 1, 1024]`. **No ECG input, no GT R input, no detected PPG
peaks as network input, no site embedding.**

## 6. Positive control (gate on interpretation)

Positive-control targets are **T1 beat_count** and **T2 median_RR_ms**. At least one must reach, on validation:
Spearman ρ ≥ 0.70 **and** clearly beat B0. If neither does, the result is **PIPELINE POSITIVE CONTROL FAILED**,
morphology interpretation stops, and the verdict is D. The architecture is **not** redesigned after seeing
validation.

## 7. Training (frozen)

Seeds **40, 42, 44** (three per target). Identical architecture, optimizer, batch size, epoch budget, split and
cohort for every target × seed. `AdamW(lr = 1e-3, weight_decay = 1e-4)`, batch **128**, loss **SmoothL1/Huber
with `beta = 1.0` frozen before training**, on the standardised target. Maximum **30 epochs**; early stopping on
**INTERNAL_DEV standardised MAE** with **patience 5**; checkpoint selection on INTERNAL_DEV only.
`seed_everything(seed, deterministic=True)`; loader order from a per-run `torch.Generator`.

**Validation (an0/k2s) is not loaded until architecture, training and checkpoint selection are finished for all
target × seed probes** — enforced by construction (the training script never reads validation rows) and asserted
by test.

## 8. Runtime stop rule

Before full training: train T2 for exactly **100 optimizer steps**, measure wall time and peak VRAM, discard the
state, and project `n_primary_targets × 3 seeds × max-epoch cost`. **If the projection exceeds 4 GPU-hours,
STOP and report** — seeds, targets, cohort and architecture are never silently reduced.

## 9. Validation metrics and bootstrap

Per target × seed: MAE, median AE, RMSE, **normalised MAE = MAE / TRAIN_IQR**, Pearson r, Spearman ρ, R².
Primary regression metric **normalised MAE**; primary association metric **Spearman ρ**. Raw MAEs are never
compared across targets.

Bootstrap: **cluster** on the underlying ECG window (`subject + window_index`; all four site rows move
together), subject-stratified with equal an0/k2s weight, **2,000 replicates**, `default_rng(20260903)`. These
CIs describe uncertainty **within the frozen two-subject development cohort** and are **not**
population-generalisation intervals.

## 10. Controls (no retraining)

- **SS-SHUFFLE** (`o1-same-subject-shuffle-v1`): replace each validation PPG with another window of the **same
  subject × site**, deterministic rank-based derangement, no fixed points. Keeps subject, site, device and broad
  physiology; removes window-specific correspondence. Primary oriented effect `AE(SS-SHUFFLE) − AE(TRUE)`,
  positive = TRUE better.
- **XS-SHUFFLE** (`o1-cross-subject-shuffle-v1`): replace with a window of the **other validation subject, same
  site**, deterministic. Additionally breaks subject-specific physiology. Compare `AE(XS) − AE(SS)`; positive
  suggests the probe uses subject-level/static cues. **Secondary**, and never described as subject identification.

## 11. Extractability skill

`Skill_R = 1 − MAE_TRUE / MAE_B2` (raw-PPG probe vs rhythm/site baseline) and
`Skill_W = 1 − MAE_TRUE / MAE_SS-SHUFFLE` (window-specific). These are **operational skill scores, not
information fractions**.

## 12. Component classification (frozen, per target, three-seed mean with direction checked in all three seeds)

- **A. STRONG WINDOW-SPECIFIC EXTRACTABILITY** — all of: `Skill_R ≥ +0.10`; TRUE vs SS-SHUFFLE normalised-AE
  improvement with 95 % CI entirely > 0; median-across-seeds Spearman ρ ≥ 0.30; TRUE better than B0 in all three seeds.
- **B. PARTIAL WINDOW-SPECIFIC EXTRACTABILITY** — TRUE beats SS-SHUFFLE with CI > 0 **and** `Skill_R > 0`, but one or more A-thresholds fail.
- **C. RHYTHM / STATIC EXPLAINED** — TRUE beats B0/B1 but does **not** clearly beat B2, **or** TRUE vs SS-SHUFFLE is unresolved.
- **D. NO CLEAR EXTRACTABILITY UNDER THIS PROBE** — TRUE does not clearly beat the relevant baselines **and** TRUE vs SS-SHUFFLE is unresolved.

The label **UNOBSERVABLE is never used.** Evaluation order is A → B → C → D and is implemented in
`o1_targets.classify_component`, asserted by test to match this section.

## 13. Secondary analyses (do not alter the clean map)

- **§20 corruption transfer**: after the clean map is frozen, run every trained probe on the exact Q1 severe
  corruptions (`LP_1.25Hz`, `SNR_0dB`, `DROP_2.0s`, `SHUFFLED`, `NULL`) reusing `q1_corruption` byte-identically
  (same salts, same per-window seeds). No retraining. Report normalised MAE and Spearman ρ per component and condition.
- **§21 site-wise map**: per component and per site (sternum/head/wrist/ankle) report TRUE nMAE, SS-SHUFFLE nMAE,
  the TRUE−shuffle effect and Spearman ρ, with no pooling before reporting. Exploratory; no physiological causality
  is inferred from site differences.
- **§24 natural quality**: reuse Q1's PPG-only `periodicity_score` and `pulse_template_consistency` with the same
  within-(subject × site) quartile rule; report each probe's normalised MAE across Q1→Q4. No new quality metric.

## 14. Generator-utilization crosswalk (§22) — kept strictly separate

Reuse the **frozen Q1 arm-B CLEAN and SHUFFLED** per-window artifacts. No generator is trained or re-run.

| component | generator metric |
|---|---|
| rhythm / event (R1 diagnostic) | `f1_excess` (CLEAN − SHUFFLED) and `beats_ratio_dev` |
| T4 QRS p2p | `p2p_dev` |
| T5 QRS energy | `qrs_e_dev` |
| T6 max derivative | `qrs_deriv_rmse` (the frozen derivative metric; `qrs_maxderiv_dev` is not in the Q1 per-window artifact) |
| T7 curvature | `qrs_curvature_err` |
| T8 QRS width | **N/A unless an exact frozen per-window width-error metric exists in the Q1 artifacts** |
| T9 HF | `hf_err` |

`UtilizationEffect = Error_SHUFFLED − Error_CLEAN` (positive = correct PPG improves that generated component);
for F1 it is `F1_CLEAN − F1_SHUFFLED`. Paired clustered bootstrap where per-window artifacts permit. **No new
generator metric is invented to fill the table**; missing matches are marked N/A. Generator utilization is
**never** used to classify direct extractability (asserted by test).

## 15. Extractability × utilization quadrants (§23)

**Q-A** extractable + generator-sensitive · **Q-B** extractable + generator-insensitive → labelled **CANDIDATE
UNDERUTILIZATION** (never "causal underutilization") · **Q-C** weakly extractable + generator-sensitive
(interpret cautiously; possible proxy use) · **Q-D** weakly extractable + generator-insensitive (no
information-theoretic claim).

## 16. Final O1 verdict — exactly one

- **A. COMPONENT-WISE EXTRACTABILITY HETEROGENEITY SUPPORTED** — ≥ 2 primary components classified STRONG/PARTIAL
  **and** ≥ 2 others classified RHYTHM/STATIC or NO CLEAR, **and** the split is not explained solely by target
  validity or near-zero variance.
- **B. EXTRACTABILITY DOMINATED BY RHYTHM / STATIC INFORMATION** — most morphology targets fail to beat B2 or
  SS-SHUFFLE while rhythm targets succeed.
- **C. BROAD ECG-COMPONENT EXTRACTABILITY OBSERVED** — most primary morphology targets are STRONG/PARTIAL. No
  claim of exact waveform identifiability.
- **D. EXTRACTABILITY MAP INCONCLUSIVE** — positive controls fail, coverage is poor, results are dominated by low
  target variance, or seed directions are unstable.

Decision thresholds are **not** changed after results. The implementation (`o1_targets.decide_o1`) is asserted by
test to match this section.

## 17. Expected patterns (hypotheses only — none is assumed true)

P1 rhythm > amplitude/calibration > fine local shape · P2 if T4/T5 are extractable *and* the generator is
condition-sensitive there, Q1's S6/S7 behaviour has a direct conditional explanation · P3 extractable
derivative/curvature with an insensitive generator would be a strong **CANDIDATE UNDERUTILIZATION** · P4 if
derivative/curvature probes themselves fail the rhythm/shuffle controls, then R2/R3's attempt to force those
quantities from PPG was targeting weakly supported information.

## 18. Figures (§27)

(1) component × arm normalised-MAE heatmap (B0, B1, B2, TRUE, SS-SHUFFLE, XS-SHUFFLE); (2) component Spearman
heatmap; (3) component × site TRUE−shuffle skill heatmap; (4) component × corruption normalised-MAE heatmap;
(5) extractability × generator-utilization quadrant plot; (6) within- vs between-subject target variance plot.

## 19. Required tests (§29)

Test-subject firewall · exact R1 subject split · exact frozen cohorts · four-site ECG target identity · target
extraction determinism · targets use frozen primitives · validation never used for target scaling · validation
never used for checkpoint selection · no ECG in probe input · no GT R in probe input · identical architecture and
identical parameter count across targets · receptive field ≥ 1024 · exact seeds 40/42/44 · B2 uses rhythm/site
features only and contains no PPG amplitude/morphology feature · SS-SHUFFLE bijective without fixed points ·
XS-SHUFFLE cross-subject and same-site · cluster bootstrap resamples underlying ECG windows with all site rows
kept together · Q1 corruption functions reused code-equivalently · clean classification frozen before corruption
analysis · classification and verdict implement this document exactly · generator utilization never used to
classify direct extractability.

## 20. Artifacts (§28)

`docs/O1_COMPONENT_TARGET_AUDIT.md`, this file, `docs/O1_ECG_COMPONENT_EXTRACTABILITY_REPORT.md`, and
`artifacts/o1_component_extractability/`: `provenance.json`, `cohort_manifest.csv`, `four_site_target_audit.csv`,
`target_definitions.json`, `target_validity.csv`, `target_variability.csv`, `target_scaling.json`,
`baseline_metrics.csv`, `probe_training_manifest.csv`, `probe_training_logs/`, `probe_metrics.csv`,
`same_subject_shuffle_manifest.csv`, `cross_subject_shuffle_manifest.csv`, `shuffle_metrics.csv`,
`extractability_skill.csv`, `component_classification.csv`, `corruption_transfer.csv`,
`site_extractability.csv`, `natural_quality_extractability.csv`, `generator_utilization_crosswalk.csv`,
`extractability_utilization_map.csv`, `bootstrap_results.csv`, `decision.json`, `figures/`. Checkpoints go to
`outputs/o1_<target>_seed<seed>/` and remain gitignored; raw data, predictions and large artifacts are never committed.

## 21. Claim boundaries (§30)

**Allowed**: "Under a fixed family of PPG-only probes, ECG components exhibit different levels of conditional
extractability." · "Some ECG components contain window-specific predictive information beyond rhythm/site
baselines." · "The frozen generator's sensitivity to PPG conditioning differs from direct component extractability."

**Not allowed**: "this component is unobservable from PPG" · "PPG does not contain QRS morphology" · "the
generator ignores information causally" · "this is the information-theoretic limit" · "the probes establish
clinical reconstructability".

## 22. Next-step logic (§31) — recommendation only, nothing implemented

**A** → use the map to design a factorized generator (strongly extractable components condition-anchored, weakly
extractable ones stochastic / uncertainty-aware); candidates such as phase/event canonicalization or
component-aware conditional generation require a **new preregistration**. **B** → stop forcing fine morphology
from PPG; investigate the plausible conditional ECG distribution / uncertainty / downstream task framing.
**C** → deterministic/factorized conditioning is better justified than currently assumed. **D** → do not design a
method from O1; fix the probe/target issue first.

## 23. Commit order (§32)

1 integrity → 2 four-site audit → 3 component-target audit → 4 preregistration → **5 commit + push audit and
preregistration** → 6 implementation → 7 tests → **8 commit + push implementation** → 9 100-step runtime
preflight → 10 discard preflight state → 11 train all target × seed probes → 12 freeze checkpoints → 13
validation TRUE/B0/B1/B2 → 14 SS/XS shuffles → 15 **freeze the clean classification** → 16 corruption transfer →
17 site-wise map → 18 natural-quality secondary → 19 generator-utilization crosswalk → 20 final map and verdict →
21 report → 22 result commit + push → 23 STOP.
