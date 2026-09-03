# O1 — ECG Component-wise Conditional Extractability Map — REPORT

**What ECG information is actually recoverable from PPG?**

| | |
|---|---|
| Preregistration | `docs/O1_ECG_COMPONENT_EXTRACTABILITY_PREREGISTRATION.md` + `docs/O1_COMPONENT_TARGET_AUDIT.md`, frozen and pushed as **`b972dea`** before any probe existed |
| Amendment | `docs/O1_PREREGISTRATION_AMENDMENT_1.md` (verdict order), pushed with the implementation **`2fed313`**, before any probe was trained and before any validation window was scored. **It did not trigger** (`amendment_1_applied: false`) |
| Status | **problem-discovery / diagnostic**. Not independent confirmatory evidence. Two development-validation subjects. Terminology is **operational conditional extractability**; a finite-capacity probe failure is never called unobservability |
| Test subjects | `kjd`, `ssx` never loaded (`test_subjects_loaded: []`) |
| **FINAL O1 VERDICT** | **COMPONENT-WISE EXTRACTABILITY HETEROGENEITY SUPPORTED** |

---

## 1. Repository

| | |
|---|---|
| start SHA | `801bf607` (Q1 report), clean tree, PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`, A4 md5 `31c042d2…`, no C2 outputs |
| audit + prereg SHA | **`b972dea`** |
| implementation SHA | **`2fed313`** (24 O1 tests; full suite 365 passed) |
| result SHA | this commit |
| clean? | tracked tree clean at the two commit gates (`b972dea`, `2fed313`). Two stages ran with uncommitted files, both recorded in their own provenance: the **target build** at `801bf607` with `dirty_files: 3` (the three O1 files committed minutes later as `b972dea`) and the **evaluation** at `2fed313` with `dirty_files: 1` (the pickling fix of §16 item 2). Preflight and probe training both ran with `dirty_files: 0`. The evaluation re-run reproduced the classification and verdict **bit-identically** |
| test access? | none — no O1 script names or loads `kjd`/`ssx` |

## 2. Target audit (details in `docs/O1_COMPONENT_TARGET_AUDIT.md`)

Four-site ECG identity **established**: `(subject, window_index)` identifies one ECG waveform, one R-peak train
and one value of every target across sites — 85,830/85,830 window groups waveform-identical, 7,168/7,168 deep
checks identical. The independent unit is therefore the ECG window, and the O1 bootstrap clusters on it.

| id | target | definition (frozen primitives) | valid % (validation) | train IQR | within-subject variance / total |
|---|---|---|---|---|---|
| T1 | beat_count | `len(detect_rpeaks)` | 100.0 | 3.0 beats | 0.54 |
| T2 | median_RR_ms | median of `diff(peaks)` | 100.0 | 226.6 ms | 0.57 |
| T3 | RR_IQR_ms | IQR of the RR series | 100.0 | 46.9 ms | 0.76 |
| T4 | median_QRS_p2p | `ptp` of the 21-sample M1 QRS core | 100.0 | 0.505 | 0.70 |
| T5 | median_QRS_energy | `Σ y²` of the core | 100.0 | 5.814 | 0.67 |
| T6 | median_QRS_max_abs_derivative | `max\|M1.d1\|` | 100.0 | 0.230 | 0.69 |
| T7 | median_QRS_curvature_energy | `mean(M1.d2²)` | 100.0 | 0.0338 | 0.71 |
| T8 | median_QRS_width_ms | frozen `qrs_width_ms` | 100.0 | 31.25 ms | 0.68 |
| T9 | ECG_HF_fraction | frozen `hf_energy_ratio` | 100.0 | 0.0924 | **0.44** |

All nine are PRIMARY. No new delineator was written; the detector was not switched; per-window aggregation is
the median over valid GT beats.

## 3. Runtime

| item | value |
|---|---|
| targets × seeds | 9 × 3 = **27 probes**, 328,897 parameters each (identical), RF 2,041 |
| preflight (100 steps on T2, state discarded) | 12.83 ms/step, peak 1,575.5 MiB → **projected worst case 1.848 GPU-h** (budget 4.0), `stop: false` |
| actual training | **0.581 GPU-h** (34.9 min; 27 probes, 6–14 epochs each with internal-dev early stopping) |
| actual evaluation | 0.045 GPU-h (2.7 min) |
| **total** | **0.626 GPU-h**, max VRAM **1,578 MiB** |

## 4. Baselines and direct PPG extractability (validation an0 + k2s, 8,192 rows, 6,299 unique ECG windows)

All errors are **normalised MAE = MAE / train IQR** (lower is better); ρ is the median Spearman over seeds
40/42/44; the TRUE−SS effect is the equal-subject, ECG-window-clustered bootstrap (2,000, seed 20260903) of the
per-window normalised absolute-error improvement, positive = TRUE better.

| id | target | B0 | B1 site | B2 rhythm | **TRUE** | SS-SHUFFLE | XS-SHUFFLE | Skill_R | Skill_W | ρ | TRUE−SS [95 % CI] | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | beat_count | 0.3421 | 0.3421 | 0.3353 | **0.1935** | 0.4290 | 0.4382 | +0.423 | +0.549 | **+0.752** | +0.2355 [+0.2283, +0.2429] | **STRONG** |
| T2 | median_RR_ms | 0.3423 | 0.3418 | 0.3471 | **0.1462** | 0.4539 | 0.4700 | +0.579 | +0.678 | **+0.802** | +0.3077 [+0.2996, +0.3157] | **STRONG** |
| T3 | RR_IQR_ms | 0.4063 | 0.4063 | 1.1889 | **0.3961** | 0.4855 | 0.4987 | +0.667 | +0.184 | +0.336 | +0.0894 [+0.0813, +0.0973] | **STRONG** |
| T4 | median_QRS_p2p | 0.3337 | 0.3339 | 0.5134 | 0.3842 | 0.4204 | 0.4479 | +0.252 | +0.086 | +0.268 | +0.0362 [+0.0296, +0.0424] | PARTIAL |
| T5 | median_QRS_energy | 0.3563 | 0.3567 | 0.3782 | 0.3803 | 0.4023 | 0.4255 | −0.006 | +0.055 | +0.209 | +0.0219 [+0.0181, +0.0260] | NO CLEAR |
| T6 | median_QRS_max_abs_derivative | 0.3290 | 0.3291 | 0.5157 | 0.3884 | 0.4275 | 0.4497 | +0.247 | +0.092 | +0.263 | +0.0392 [+0.0323, +0.0456] | PARTIAL |
| T7 | median_QRS_curvature_energy | 0.3171 | 0.3172 | 0.3788 | 0.3584 | 0.3941 | 0.4075 | +0.054 | +0.091 | +0.203 | +0.0357 [+0.0299, +0.0414] | PARTIAL |
| T8 | median_QRS_width_ms | 0.2708 | 0.2708 | 0.4021 | 0.3557 | 0.3716 | 0.3896 | +0.115 | +0.043 | +0.199 | +0.0159 [+0.0109, +0.0208] | PARTIAL |
| T9 | ECG_HF_fraction | 0.3360 | 0.3360 | 0.3261 | 0.4008 | 0.3922 | 0.3873 | −0.229 | −0.022 | **−0.220** | −0.0085 [−0.0128, −0.0041] | NO CLEAR |

**Positive control PASSES**: T1 ρ = 0.752 and T2 ρ = 0.802, both ≥ 0.70 and both far better than B0 (0.194 /
0.146 vs 0.342). Morphology interpretation is therefore admissible.

**The single most important caveat, stated up front.** For every morphology target the raw-PPG probe is
**worse in absolute terms than the trivial global-median baseline B0** (T4 0.384 vs 0.334; T6 0.388 vs 0.329;
T7 0.358 vs 0.317; T8 0.356 vs 0.271; T5 0.380 vs 0.356; T9 0.401 vs 0.336). What the PARTIAL class certifies
is only that (i) the probe beats the **rhythm/site** baseline B2 and (ii) using the **correct window's** PPG
beats using another window of the same subject and site with a CI clear of zero. Those two facts establish the
presence of *some* window-specific morphology information; they do **not** establish that this probe family can
predict QRS morphology usefully. Only the rhythm targets T1–T3 beat every baseline.

**T3 caveat.** T3's very large `Skill_R` (+0.667) is inflated by a pathological B2: the rhythm Ridge extrapolates
badly on RR-IQR (nMAE 1.189, the only baseline above 1). T3 still beats B0 (0.396 vs 0.406, in all three seeds)
and has ρ = 0.336, so its STRONG class does not rest on B2 alone — but its margin over B0 is thin.

## 5. Seed stability (all directions reproduce in all three seeds)

| id | target | seed 40 | seed 42 | seed 44 |
|---|---|---|---|---|
| T1 | beat_count | 0.1954 / +0.754 | 0.1922 / +0.752 | 0.1930 / +0.741 |
| T2 | median_RR_ms | 0.1433 / +0.808 | 0.1449 / +0.802 | 0.1504 / +0.790 |
| T3 | RR_IQR_ms | 0.4048 / +0.339 | 0.3811 / +0.326 | 0.4024 / +0.336 |
| T4 | median_QRS_p2p | 0.3673 / +0.286 | 0.3902 / +0.241 | 0.3952 / +0.268 |
| T5 | median_QRS_energy | 0.3874 / +0.182 | 0.3675 / +0.209 | 0.3861 / +0.213 |
| T6 | median_QRS_max_abs_derivative | 0.3880 / +0.263 | 0.3889 / +0.213 | 0.3882 / +0.268 |
| T7 | median_QRS_curvature_energy | 0.3646 / +0.203 | 0.3693 / +0.175 | 0.3411 / +0.229 |
| T8 | median_QRS_width_ms | 0.3552 / +0.188 | 0.3549 / +0.199 | 0.3571 / +0.225 |
| T9 | ECG_HF_fraction | 0.3817 / −0.220 | 0.3642 / −0.226 | 0.4564 / −0.161 |

(nMAE / Spearman ρ. T9 is negatively correlated in every seed — the probe's ordering of validation windows is
systematically inverted relative to the target, i.e. this is a transfer failure, not noise.)

## 6. Static vs window-specific information (XS − SS, secondary)

| id | target | SS-SHUFFLE | XS-SHUFFLE | XS − SS [95 % CI] | reading |
|---|---|---|---|---|---|
| T1 | beat_count | 0.4290 | 0.4382 | +0.0092 [+0.0013, +0.0176] | subject-level cues used in addition |
| T2 | median_RR_ms | 0.4539 | 0.4700 | +0.0161 [+0.0073, +0.0252] | same |
| T3 | RR_IQR_ms | 0.4855 | 0.4987 | +0.0132 [+0.0049, +0.0211] | same |
| T4 | median_QRS_p2p | 0.4204 | 0.4479 | +0.0274 [+0.0203, +0.0348] | same |
| T5 | median_QRS_energy | 0.4023 | 0.4255 | +0.0232 [+0.0182, +0.0284] | same |
| T6 | max_abs_derivative | 0.4275 | 0.4497 | +0.0221 [+0.0152, +0.0295] | same |
| T7 | curvature_energy | 0.3941 | 0.4075 | +0.0134 [+0.0078, +0.0196] | same |
| T8 | QRS_width_ms | 0.3716 | 0.3896 | +0.0180 [+0.0127, +0.0233] | same |
| T9 | HF_fraction | 0.3922 | 0.3873 | −0.0049 [−0.0090, −0.0010] | no subject-level benefit |

For T1–T8 the cross-subject shuffle is worse than the same-subject shuffle, so the probes exploit subject-level
/ static PPG cues **in addition to** window-specific ones. For the morphology targets the static component
(SS ≈ 0.37–0.43 vs B0 ≈ 0.27–0.36) is a large part of what the probe does. This is **not** subject identification
and no such claim is made.

## 7. ECG timing anchor (R1 existing diagnostic — not a new O1 probe)

Frozen R1 Global-TCN on the same 8,192-window validation cohort: **F1@50 0.6199 · F1@100 0.7805 · F1@150 0.8582 ·
F1@200 0.8975 · F1@250 0.9223 · RR MAE 31.4 ms** (median 15.6 ms, corr 0.891), beats-ratio deviation 0.1166.

## 8. Component × site map (secondary, exploratory)

nMAE(TRUE) / Skill_W / ρ per site:

| id | target | sternum | head | wrist | ankle |
|---|---|---|---|---|---|
| T1 | beat_count | 0.193 / +0.53 / +0.76 | **0.151 / +0.66 / +0.87** | 0.224 / +0.49 / +0.64 | 0.206 / +0.51 / +0.74 |
| T2 | median_RR_ms | 0.148 / +0.66 / +0.78 | **0.096 / +0.80 / +0.92** | 0.184 / +0.60 / +0.71 | 0.157 / +0.65 / +0.79 |
| T3 | RR_IQR_ms | 0.413 / +0.19 / +0.27 | 0.351 / +0.27 / +0.43 | 0.409 / +0.11 / +0.36 | 0.412 / +0.16 / +0.28 |
| T4 | median_QRS_p2p | 0.395 / +0.07 / +0.23 | 0.374 / +0.09 / +0.31 | 0.407 / +0.09 / +0.26 | 0.361 / +0.09 / +0.28 |
| T5 | median_QRS_energy | 0.376 / +0.06 / +0.23 | 0.360 / +0.07 / +0.22 | 0.416 / +0.04 / +0.16 | 0.370 / +0.05 / +0.25 |
| T6 | max_abs_derivative | 0.394 / +0.08 / +0.28 | 0.374 / +0.09 / +0.27 | 0.414 / +0.10 / +0.23 | 0.371 / +0.10 / +0.27 |
| T7 | curvature_energy | 0.365 / +0.07 / +0.21 | 0.358 / +0.06 / +0.16 | 0.374 / +0.10 / +0.19 | 0.337 / +0.13 / +0.27 |
| T8 | QRS_width_ms | 0.344 / +0.05 / +0.27 | 0.371 / +0.02 / +0.20 | 0.357 / +0.03 / +0.13 | 0.350 / +0.08 / +0.22 |
| T9 | HF_fraction | 0.391 / −0.03 / −0.12 | 0.419 / −0.03 / −0.18 | 0.394 / −0.01 / −0.23 | 0.400 / −0.02 / −0.24 |

Rhythm extractability depends strongly on site (**head** best, **wrist** worst: T2 ρ 0.92 vs 0.71); morphology
skill is small and roughly site-independent. No physiological causality is inferred.

## 9. Controlled corruption transfer (secondary; the clean map was frozen first)

Probes are **not** retrained; the frozen `q1_corruption` module is reused unchanged. nMAE / ρ:

| id | target | CLEAN | LP_1.25Hz | SNR_0dB | DROP_2.0s | SHUFFLED | NULL |
|---|---|---|---|---|---|---|---|
| T1 | beat_count | 0.194 / +0.75 | 0.337 / +0.39 | 0.286 / +0.48 | 0.287 / +0.63 | 0.425 / +0.01 | 0.552 / n/a |
| T2 | median_RR_ms | 0.146 / +0.80 | 0.344 / +0.36 | 0.284 / +0.48 | 0.267 / +0.67 | 0.452 / +0.01 | 0.491 / n/a |
| T3 | RR_IQR_ms | 0.396 / +0.34 | 0.608 / +0.17 | 0.492 / +0.06 | 0.478 / +0.16 | 0.484 / +0.03 | 0.419 / n/a |
| T4 | median_QRS_p2p | 0.384 / +0.27 | 0.418 / +0.07 | 0.348 / +0.17 | 0.443 / +0.15 | 0.423 / +0.05 | 0.356 / n/a |
| T5 | median_QRS_energy | 0.380 / +0.21 | 0.345 / +0.18 | 0.425 / +0.16 | 0.419 / +0.16 | 0.405 / +0.02 | 0.412 / n/a |
| T6 | max_abs_derivative | 0.388 / +0.26 | 0.474 / +0.02 | 0.350 / +0.13 | 0.418 / +0.22 | 0.432 / +0.02 | 0.390 / n/a |
| T7 | curvature_energy | 0.358 / +0.20 | 0.373 / +0.04 | 0.359 / +0.12 | 0.394 / +0.15 | 0.397 / +0.00 | 0.379 / n/a |
| T8 | QRS_width_ms | 0.356 / +0.20 | 0.298 / +0.02 | 0.327 / +0.05 | 0.339 / +0.17 | 0.373 / +0.13 | 0.320 / n/a |
| T9 | HF_fraction | 0.401 / −0.22 | 0.408 / −0.26 | 0.444 / −0.14 | 0.370 / −0.19 | 0.391 / −0.04 | 0.344 / n/a |

Rhythm extractability collapses under every corruption (T1/T2 ρ 0.75/0.80 → 0.36–0.67) exactly as Q1's R1 support did, and under LP_1.25Hz their normalised MAE falls back to the B0 constant-baseline level (0.337 / 0.344 vs B0 0.342 / 0.342).
Morphology **ρ collapses to ≈ 0.02–0.22 while nMAE barely moves — and sometimes improves** (T8 0.356 → 0.298
under LP_1.25Hz, T4 0.384 → 0.348 under SNR_0dB): the error is dominated by the static/marginal component, so
destroying the window-specific signal costs little error but removes almost all ranking information. Under SHUFFLED — which replaces the window with **another window of the same subject and site**, so it is the
window-specific-information floor rather than a zero-information condition — ρ falls to |ρ| ≤ 0.05 for eight of
the nine targets; **T8 is the exception at +0.13**, two thirds of its clean ρ (0.199) and essentially its
SS-SHUFFLE level (ρ 0.122 / 0.125 / 0.136 across seeds 40/42/44), i.e. most of T8's already weak ranking skill
does not require this window's PPG — consistent with T8 having the smallest window-specific effect in §4
(TRUE−SS +0.0159, Skill_W +0.043). NULL produces
a constant prediction (ρ undefined).

## 10. Natural PPG quality (secondary, exploratory, frozen Q1 scores)

nMAE(TRUE) by quartile within (subject × site):

| id | target | periodicity Q1→Q4 | template consistency Q1→Q4 |
|---|---|---|---|
| T1 | beat_count | 0.232 → 0.212 → 0.179 → **0.151** | 0.243 → 0.201 → 0.172 → **0.155** |
| T2 | median_RR_ms | 0.194 → 0.163 → 0.127 → **0.101** | 0.199 → 0.148 → 0.128 → **0.107** |
| T3 | RR_IQR_ms | 0.460 → 0.429 → 0.378 → **0.318** | 0.455 → 0.444 → 0.368 → **0.318** |
| T4 | median_QRS_p2p | 0.406 → 0.395 → 0.380 → 0.356 | 0.394 → 0.402 → 0.377 → 0.364 |
| T5 | median_QRS_energy | 0.375 → 0.371 → 0.381 → **0.394** | 0.367 → 0.378 → 0.380 → **0.396** |
| T6 | max_abs_derivative | 0.416 → 0.405 → 0.379 → 0.353 | 0.405 → 0.410 → 0.379 → 0.360 |
| T7 | curvature_energy | 0.374 → 0.360 → 0.354 → 0.344 | 0.365 → 0.369 → 0.350 → 0.348 |
| T8 | QRS_width_ms | 0.364 → 0.360 → 0.353 → 0.346 | 0.376 → 0.359 → 0.343 → 0.347 |
| T9 | HF_fraction | 0.365 → 0.363 → 0.410 → **0.466** | 0.377 → 0.395 → 0.404 → **0.428** |

Rhythm gains most from good PPG (T2 −48 % error from Q1 to Q4); T4/T6/T7/T8 improve modestly; **T5 and T9 get
worse on the cleanest PPG**. That is consistent with their classification for different reasons: T9 carries no
window-specific benefit at all (SS effect −0.0085, CI entirely negative), while T5's window-specific benefit is
real (+0.0219 [+0.0181, +0.0260]) but does not beat the rhythm baseline (Skill_R −0.006) — in neither case is
the probe's error driven by window-specific PPG information.

## 11. Generator-utilization crosswalk (frozen Q1 arm-B artifacts; no generator was run)

`UtilizationEffect = Metric_SHUFFLED − Metric_CLEAN` for the lower-is-better generator metrics (seven rows, six
distinct metrics — T1 and T2 share `beats_ratio_dev`) and `F1_CLEAN − F1_SHUFFLED` for the one higher-is-better
metric, `f1_excess`, exactly as preregistered. The sign is orientation-adjusted throughout, so positive always
means the correct PPG improves that generated component. Paired **ECG-window-clustered**,
subject-stratified bootstrap (2,000, seed 20260903, 1,922 clusters) on Q1's frozen 2,048-window cohort.

| component | generator metric | CLEAN | SHUFFLED | utilization effect [95 % CI] | verdict |
|---|---|---|---|---|---|
| event timing (R1 diagnostic) | `f1_excess` | 0.3176 | 0.0082 | **+0.3093** [+0.2953, +0.3234] | improves |
| T1/T2 rhythm | `beats_ratio_dev` | 0.1067 | 0.1604 | **+0.0537** [+0.0475, +0.0598] | improves |
| T3 RR_IQR | — | — | — | — | **N/A** (no exact frozen metric) |
| T4 QRS p2p | `p2p_dev` | 0.2425 | 0.2986 | **+0.0562** [+0.0456, +0.0671] | improves |
| T5 QRS energy | `qrs_e_dev` | 0.6056 | 0.9074 | **+0.3019** [+0.2728, +0.3248] | improves |
| T6 max derivative | `qrs_deriv_rmse` | 0.3220 | 0.2941 | **−0.0279** [−0.0296, −0.0261] | **worsens** |
| T7 curvature | `qrs_curvature_err` | 0.2147 | 0.1920 | **−0.0227** [−0.0240, −0.0215] | **worsens** |
| T8 QRS width | — | — | — | — | **N/A** |
| T9 HF | `hf_err` | 0.0854 | 0.0860 | +0.0007 [−0.0016, +0.0030] | unresolved |

No generator metric was invented to fill the table; T3 and T8 are N/A. Note the population differs from O1's
validation cohort (Q1's frozen 2,048-window subset vs O1's 8,192-window R1 cohort), so this is a
component-level comparison, not a per-window join.

## 12. Extractability × utilization map

| id | component | direct extractability | generator effect | quadrant |
|---|---|---|---|---|
| T1 | beat_count | STRONG | +0.0537 improves | **Q-A** extractable + generator-sensitive |
| T2 | median_RR_ms | STRONG | +0.0537 improves | **Q-A** |
| T3 | RR_IQR_ms | STRONG | N/A | — |
| T4 | median_QRS_p2p | PARTIAL | +0.0562 improves | **Q-A** |
| T5 | median_QRS_energy | NO CLEAR | +0.3019 improves | **Q-C** weakly extractable + generator-sensitive |
| T6 | max_abs_derivative | PARTIAL | −0.0279 worsens | **Q-B CANDIDATE UNDERUTILIZATION** |
| T7 | curvature_energy | PARTIAL | −0.0227 worsens | **Q-B CANDIDATE UNDERUTILIZATION** |
| T8 | QRS_width_ms | PARTIAL | N/A | — |
| T9 | ECG_HF_fraction | NO CLEAR | unresolved | **Q-D** |

### Key mismatches

1. **Extractable but generator-insensitive (Q-B, candidate underutilization): T6 max-derivative and T7
   curvature.** Both carry window-specific PPG information that survives the SS-SHUFFLE control with CIs clear
   of zero (+0.0392 and +0.0357) and beat the rhythm baseline (Skill_R +0.247 / +0.054), yet the generator's
   matched metrics move the **wrong way**: `qrs_deriv_rmse` and `qrs_curvature_err` are *better* when the PPG is
   shuffled (−0.0279 [−0.0296, −0.0261] and −0.0227 [−0.0240, −0.0215]). This is Q1's structure-metric anomaly seen from the other side, and it is the strongest
   *candidate* underutilization signal in the map. No causal claim is made.
2. **Weakly extractable but generator-sensitive (Q-C): T5 QRS energy.** The direct probe fails to beat the
   rhythm baseline (Skill_R −0.006) while the generator's `qrs_e_dev` is strongly condition-sensitive (+0.3019 [+0.2728, +0.3248]).
   The generator's energy behaviour is therefore probably driven by something other than window-specific QRS
   energy information — most plausibly the rhythm/beat-placement channel, which is also what determines whether
   a QRS is generated at the right place at all. Interpret cautiously (possible proxy use).
3. **T9 HF is the one clear negative**: the only target with negative Spearman on validation in all three seeds,
   no window-specific benefit (CI entirely on the wrong side), no subject-level benefit, and an unresolved
   generator effect — and it is also the most subject-dominated target (within/total 0.44).

## 13. FINAL O1 VERDICT

**COMPONENT-WISE EXTRACTABILITY HETEROGENEITY SUPPORTED**

The preregistered criteria are met: ≥ 2 primary components are STRONG/PARTIAL (seven: T1, T2, T3, T4, T6, T7,
T8), ≥ 2 others are NO CLEAR (two: T5, T9), the positive control passes (T1 ρ 0.752, T2 ρ 0.802), and the split
is not an artefact of validity or near-zero variance (all nine targets are 100 % valid, all have non-trivial
within-subject variance). Amendment 1 did **not** trigger — the STRONG/PARTIAL set contains morphology targets,
so the frozen order A → B → C → D applied unchanged; the unamended rule would have returned the same verdict A.

The heterogeneity is, quantitatively, a **hierarchy**: beat schedule T1/T2 (ρ 0.75–0.80, beats every baseline)
≫ RR variability T3 (ρ 0.34, beats every baseline but only just — see the T3 caveat in §4) ≫ QRS amplitude /
sharpness / width (ρ 0.20–0.27, beats the rhythm baseline and the same-subject shuffle but **not** the global
median) ≫ QRS energy and HF fraction (no clear extractability). That is Pattern P1 of the
preregistration, with the amplitude tier much weaker than P1 anticipated, and Pattern P3 realised for T6/T7.

## 14. Interpretation — five things kept separate

1. **Target variability.** Every target has real within-subject variation (0.54–0.76 of total for T1–T8); T9 is
   the most subject-dominated (0.44) and is also the target that fails.
2. **PPG component extractability (absolute).** Only the rhythm targets are predicted better than a constant.
   All six morphology targets are worse than the global median in absolute normalised MAE.
3. **Window-specific information.** Independently of absolute accuracy, T1–T8 all carry information that
   requires *this* window's PPG: the SS-SHUFFLE effect is positive with CIs clear of zero for all eight, and it
   is 3–4× larger for T1/T2 (+0.24 / +0.31) than for T3 (+0.089) and an order of magnitude larger than for
   morphology (+0.016 … +0.039).
4. **Static / rhythm information.** A large part of every morphology probe's behaviour is static: XS − SS is
   positive for T1–T8, the B2 rhythm baseline already explains much of T5/T7/T9, and morphology nMAE hardly
   moves under corruption while its ρ collapses.
5. **Generator utilization.** Separate from all of the above, and measured only on frozen Q1 artifacts: the
   generator is condition-sensitive on events, beats-ratio, p2p and QRS energy, insensitive-in-the-wrong-
   direction on derivative and curvature, and unresolved on HF.

## 15. What this does NOT prove

- **A finite probe is not an information-theoretic limit.** T5/T9's failure means this probe family, this
  cohort and this protocol did not extract them — not that PPG lacks the information. The word "unobservable"
  is not used anywhere in this report.
- **Only an0/k2s validation.** Three probe seeds do not compensate for two validation subjects; the CIs are
  within-cohort, not population-generalisation intervals.
- **No test-subject evidence** (`kjd`/`ssx` never loaded) and **no new generator** — the crosswalk reuses frozen
  Q1 artifacts on a different (2,048-window) population.
- **Diagnostic probes are target-supervised.** They are trained on ECG-derived labels; nothing here is a
  deployable PPG→ECG system, and no clinical reconstructability is established.
- **The Q-B label is "candidate underutilization", not causal underutilization** — the generator was never
  intervened on.
- **Absolute accuracy is poor for every morphology component**, so "PARTIAL extractability" must not be read as
  "predictable".

## 16. Deviations and implementation notes

1. **Amendment 1** (verdict order; `docs/O1_PREREGISTRATION_AMENDMENT_1.md`) was frozen before any probe was
   trained and **did not trigger**.
2. **Evaluation re-run.** The first evaluation pass crashed in the natural-quality section (a `ProcessPoolExecutor`
   cannot pickle a function defined inside `main`); the classification, verdict and corruption transfer had
   already been written. The function was moved to module level and the whole script was re-run: the verdict and
   `component_classification.csv` came back **byte-identical** (sha256 `8a19c371…`), which is also the
   determinism check for the whole clean map. The fix was uncommitted while the evaluation ran
   (`provenance.json`: `dirty_files: 1`; `git diff --stat scripts/o1_evaluate.py` = 8 insertions, 5 deletions)
   and is committed together with this report.
3. **T7 definition**: `mean(d2²)` (curvature *energy*, matching the target name) while the generator crosswalk
   metric is `mean|d2_pred − d2_gt|`; both use the same frozen `M1.d2`, but they are not the same functional —
   recorded in the target audit before any result.
4. **B2 for T3 is pathological** (nMAE 1.189), which inflates T3's `Skill_R`; T3's classification also rests on
   beating B0 in all three seeds and ρ = 0.336 (§4).
5. **Crosswalk population differs** from the O1 validation cohort (§11).
6. **Crosswalk bootstrap corrected after review.** The first evaluation pass computed the §11 crosswalk CIs with
   the unclustered `paired_subject_bootstrap`, whereas preregistration §14 asks for a paired *clustered*
   bootstrap and §2 binds the cluster to the underlying ECG window. `scripts/o1_recompute_crosswalk.py` (pure
   post-processing of the frozen Q1 per-window artifacts) rebuilt the table with the ECG-window cluster
   bootstrap over the Q1 cohort's 1,922 clusters. Every point estimate and every verdict is unchanged; only the
   interval endpoints move in the third decimal.
7. **`runtime_preflight.json`** is written by the training script's `--preflight` mode; the preflight state is
   discarded, as preregistered.

## 17. Recommended next experiment (recommendation only — nothing implemented)

The preregistered mapping for verdict A is to use the map to design a **factorized generator**: strongly
extractable components condition-anchored, weakly extractable ones stochastic / uncertainty-aware. This run
sharpens that into a concrete, testable next step:

**Anchor the beat schedule, marginalise the shape.** T1/T2 (and the R1 timing diagnostic) are the only
components with both strong window-specific extractability *and* generator sensitivity, the natural-quality analysis shows a monotone gain
with better PPG (T2 −48 % error from Q1 to Q4), and under corruption they retain more rank information than any
morphology component (ρ 0.36–0.67 vs ≈ 0.02–0.22) even though their normalised MAE falls back to the B0 level
under LP_1.25Hz. The next preregistration
should test a generator whose beat placement is conditioned on an explicit rhythm estimate (the frozen R1
scaffold is already available and already validated as a support proxy) while QRS shape is drawn from a
*marginal* (or weakly conditioned) distribution rather than being forced from PPG — with T6/T7 (Q-B) as the
pre-declared components to watch, since they are precisely the ones where extractable information exists but the
current generator moves the wrong way. The comparison must include the Q1 SHUFFLED floor for every component
metric, and the design requires a **new preregistration**.

---

### Artifacts

`artifacts/o1_component_extractability/`: `provenance.json`, `target_build_provenance.json`,
`probe_training_provenance.json`, `runtime_preflight.json`, `cohort_manifest.csv`, `four_site_target_audit.csv`
(+ `_deep.csv`, `.json`), `target_definitions.json`, `target_validity.csv`, `target_variability.csv`,
`target_scaling.json`, `baseline_metrics.csv`, `probe_training_manifest.csv`, `probe_training_logs/` (27 CSVs),
`probe_metrics.csv`, `same_subject_shuffle_manifest.csv`, `cross_subject_shuffle_manifest.csv`,
`shuffle_metrics.csv`, `extractability_skill.csv`, `component_classification.csv`, `bootstrap_results.csv`,
`corruption_transfer.csv`, `site_extractability.csv`, `natural_quality_extractability.csv`,
`generator_utilization_crosswalk.csv` (rebuilt by `scripts/o1_recompute_crosswalk.py`, §16 item 6),
`extractability_utilization_map.csv`, `decision.json`, `figures/` (6).
Checkpoints in `outputs/o1_<target>_seed<seed>/` and all artifacts stay local (`artifacts/*` and `outputs/*` are
gitignored), as for R1–R3 and Q1.
