# E3 — Beat-Set-First PPG Event Geometry — PREREGISTRATION

**Can correcting or predicting the NUMBER of supplied events reduce R1's dominant overcount/spurious
failure without sacrificing recall, placement, or downstream morphology?**

## 0. Status, hypothesis and absolute rules

E3 does **not** build a large PPG phase/event network. It runs the cheapest available falsification first:
**Stage 0** supplies the frozen R1 timing scores together with the **oracle GT beat count**, and only if that
ceiling holds does **Stage 1** replace the oracle count with a **minimal PPG-derived count readout**. Both
are compared against a **train-only threshold control**, because an operating-point change is the cheap
explanation that must be ruled out before any count representation is claimed.

The motivating statement is a **hypothesis, not a conclusion**: *much of the current failure may be
reducible by getting the event set / beat count right before refining timing.*

### Absolute rules

**NO** O2c retraining · generator weight update · R1 backbone weight update · large Transformer · phase
network · set Transformer · event decoder · candidate-level neural classifier · count CNN · auxiliary
morphology loss · timing regression head · peak shifting · site-specific delay correction · GT correction at
inference · validation threshold search · test subjects · C2 · SOTA or novelty claim.

**Exactly one learned new object is permitted: a minimal LINEAR count readout (StandardScaler + Ridge) on
frozen R1 features.** If it is insufficient, E3 **STOPS**. Architecture is never escalated inside E3.

## 1. Repository and frozen components

`HEAD == origin/main == 153bba6`, clean tree, PENGUIN `6cd70cde`, iMeanFlow `bf60cd7c`, A4 md5
`31c042d291052fbb6dc15263ad316be2`, C2 deferred, `kjd` / `ssx` never loaded.

| component | path | identity | change |
|---|---|---|---|
| R1 Global-TCN | `outputs/r1_global_tcn_seed42/checkpoint_best.pt` | file `bfe76ea6…`, state `0986a7af…` | **frozen, no weight update** |
| O2c generator | `outputs/o2c_canon_oracle_seed42/checkpoint_final.pt` | file `5aab09be…`, state `f1cc44b3…` | **frozen, no retraining** |
| O2b operator | `o2b_warp.py` / `o2_warp.py` | `cb4d1866…` / `046becfb…` | **unmodified** |

R1's frozen baseline operating point is threshold **0.35** with refractory **32** samples.

## 2. The E2 contract is imported, never restated

`artifacts/e2_evaluation_contract/contract_v1.json`, sha256
**`06e869412114e1efb9ab6624540aebc141495e2690150f4e512df7676c5a9115`**, git blob **`d52b8953…`**, contract
version `e2-event-geometry-contract-v1`, source E1 SHA `a5af4af`. E3 imports
`ppg2ecg.evaluation.event_geometry_contract` and **must not** copy a metric formula, redefine T0/T1/T2/T3,
change the ± 150 ms `S → G` identity, the ± 50 ms `P → S` adherence, the `[-10, +15]` morphology support,
the coverage requirements, the aggregation order or the bootstrap. If the E2 evaluator cannot be imported
cleanly, E3 **STOPS**.

## 3. Cohorts

**Validation / primary development cohort:** the exact frozen E2/O3 cohort — 2,048 rows (`an0`, `k2s`),
19,834 GT beats, 1,922 ECG-window clusters. No new validation windows, no test subjects.

**Count-readout training population:** the original train12 only — `e61`, `fex`, `l38`, `n31`, `ngh`,
`p5d`, `p9p`, `qm9`, `trh`, `tz8`, `u7y`, `w4p`. **No validation feature may enter the scaler fit, the ridge
fit, the threshold-control selection or any other decision.**

## 4. Frozen R1 score sequence

The score is the **sigmoid probability** `sigmoid(RhythmTCN(ppg))`, the exact quantity from which the
threshold-0.35 events are currently obtained (`o3_common.r1_schedules`). **No recalibration, no temperature
scaling, no smoothing, no per-window normalisation, no preprocessing change.**

## 5. Threshold-free candidate extraction

`rhythm_tcn.extract_events` computes local maxima, filters them at `score >= threshold`, then applies greedy
NMS by **descending** score with a 32-sample refractory. The threshold-free variant is that function with
**only the amplitude-filter line removed**; the peak definition, boundary behaviour, refractory logic and
tie behaviour are unchanged.

**Bit-exactness requirement.** Filtering the threshold-free candidates at `score >= 0.35` must reproduce the
frozen R1 event list **bit-exactly**. This holds structurally — NMS is greedy by descending score, so a
sub-threshold peak is processed after every supra-threshold peak and can never suppress one, and the stable
sort preserves the relative order of the supra-threshold peaks — and it is asserted by a regression test and
re-asserted on all 2,048 validation windows at audit time. If it cannot be achieved, E3 stops with
**CANDIDATE EXTRACTION NOT IDENTICAL TO R1**; that is an implementation fault with no scientific reading.

## 6. Candidate capacity audit (before Stage 0)

For all 2,048 validation windows report the threshold-free candidate count, the GT count, their difference,
the fraction of windows with `candidate_count < GT_count`, and **candidate coverage @150 ms** (the fraction
of GT events with at least one threshold-free candidate within ± 150 ms).

**STOP** if `candidate_count < GT_count` in more than **0.5 %** of validation windows — count-only selection
would then have no fair ceiling. **No candidate is ever added.**

## 7. Top-K event selector (one deterministic rule)

Given the threshold-free candidates, their scores and an integer `K`: sort by **score descending**, break
ties by **lower sample index first**, take the first `K`, return the selected positions **sorted ascending
in time**. The refractory constraint belongs to candidate extraction and is **never reapplied differently**;
any subset of the candidates already satisfies it. If `K` exceeds the available candidates, return all
candidates and flag `candidate_shortage = true`. **No fallback, no GT correction, no timing movement.**

## 8. Stage 0 — oracle-count ceiling (no training)

`K_oracle` = the GT ECG beat count of the window; `S_ORACLE_COUNT = topK(K_oracle)`. **Event locations come
only from frozen R1 scores. GT supplies the COUNT ONLY** — not timing, not identity, not peak location, not
phase, not offset. This is an **ORACLE COUNT DIAGNOSTIC and is not deployable.**

### Stage-0 schedule gates (`ORACLE-COUNT-R1` vs `R1-0.35`, all six E2 blocks reported)

| id | requirement |
|---|---|
| **OC1** | A5 exact-set fraction: CI entirely > 0 **and** point gain **≥ +0.10** absolute |
| **OC2** | T3 overcount fraction decreases by **≥ 0.15** absolute, CI favourable |
| **OC3** | A4 spurious decreases by **≥ 0.05** absolute, CI favourable |
| **OC4** | A3 missing not worse than R1 by more than **0.020** absolute |
| **OC5** | B5 T0-only timing MAE not worse by more than **8.0 ms** |
| **OC6** | `S → G` F1@50: CI entirely favourable (no minimum beyond > 0) |

### Stage-0 generator gates (only if OC1–OC6 all pass; frozen O2c, NFE 4, source seed 0)

The same supplied schedule warps the PPG and drives the inverse warp; **no GT timing enters inference.**

| id | requirement |
|---|---|
| **OG1** | `P → G` F1@50 vs `O2C-R1-SCHEDULE`: CI > 0 **and** point gain **≥ +0.05** |
| **OG2** | own-centre T6 non-inferior, margin **0.020** |
| **OG3** | own-centre T7 non-inferior, margin **0.020** |
| **OG4** | at least one of T6 / T7 improves with CI > 0 |
| **OG5** | GT-anchored J2 derivative error not worse by more than **0.020** |
| **OG6** | GT-anchored J3 curvature error not worse by more than **0.020** |

**If any OC or OG gate fails: FINAL E3 VERDICT `COUNT-ONLY CEILING NOT SUPPORTED`, STOP, and the ridge
readout is never fitted.**

## 9. Train-only threshold control (only after the Stage-0 ceiling passes)

Using **train12 only** and the exact original frozen R1 detector, evaluate the fixed grid
**0.05, 0.10, …, 0.95** (19 thresholds, no other values) with the E2 schedule metrics, and select one
threshold by the frozen lexicographic objective: (1) maximise A5 exact-set fraction; (2) ties within 1e-12
→ minimise A4 spurious; (3) then minimise A3 missing; (4) then minimise `|threshold − 0.35|`; (5) then the
lower numerical threshold. The selection is written to `train_only_threshold_control.json` and **hashed and
committed before any validation evaluation**. The threshold never sees `an0` / `k2s`. Arm name:
**`R1-TRAIN-THRESH`**.

## 10. Minimal count feature

The frozen R1 forward is `stem → 8 residual blocks → head`; the feature tensor immediately **before** the
final event-logit projection is the last block's output `H ∈ R^{64×1024}` — unambiguous, since `head` is the
only layer after it. Per window:

```
z = concat( mean_t(H), max_t(H) )        dim(z) = 2C = 128
```

**No extra handcrafted feature, no event scores appended, no site id, no PPG-quality feature, no subject
id.** R1 is completely frozen. If the architecture had not exposed one unambiguous pre-logit tensor, E3
would stop rather than redesign.

## 11. Count target and readout

Target: `K` = the number of GT ECG R events in the window from the exact frozen detector. Training is
supervised; **at inference no ECG is available.**

Model, fitted exactly once with no search of any kind: `StandardScaler` (mean/std from train12 only,
`std = 1` for zero-variance dimensions) followed by `Ridge(alpha=1.0, fit_intercept=True)`. **No alpha
search, no cross-validation, no validation selection, no seed** — the fit is deterministic. Saved: scaler
mean, scaler std, ridge coefficients, intercept, sklearn version, feature dimension. Arm:
**`E3-RIDGE-COUNT`**.

Integer conversion: `K_hat = round_half_to_even(y_hat)` clipped **only** to the structural range
`[0, 32]` implied by the 32-sample refractory over 1,024 samples. **Never** clipped to a validation range, a
train quantile or a subject-specific range.

`S_E3 = topK(K_hat)`. No threshold, no timing regression, no peak shift, no GT. If `K_hat < 3` the predicted
schedule is retained as-is — the inherited O2b behaviour may then produce an identity warp — and the
frequency is recorded. **No fallback to the R1 threshold events.**

## 12. Freeze before validation

Before any `an0` / `k2s` window is loaded for Stage 1, `artifacts/e3_beat_set_first/` must contain
`count_readout_manifest.json`, `count_readout_coefficients.npz`, `train_feature_manifest.json` and
`train_only_threshold_control.json`, all hashed. `artifacts/` is git-ignored because it holds results, so
the small **hash-bearing JSON manifests** (not the coefficient array, which is model weights) are tracked by
a narrow `.gitignore` exception and **committed before validation runs**, giving a timestamped freeze; the
validation script re-verifies every hash and refuses to proceed if one changed.

## 13. Diagnostics and arms

Count-prediction diagnostics on validation (diagnostic only; primary success remains the E2 event-set
metrics): count MAE, count median AE, exact-count fraction, the predicted-count distribution, bias
`mean(K_hat − K)`, undercount fraction and overcount fraction.

Four schedule arms, differing **only** in the event-selection rule and sharing the frozen R1 backbone,
preprocessing, score sequence and refractory: **`R1-0.35`**, **`R1-TRAIN-THRESH`**, **`ORACLE-COUNT-R1`**,
**`E3-RIDGE-COUNT`**.

## 14. Stage-1 schedule gates (`E3-RIDGE-COUNT` vs `R1-0.35`)

| id | requirement |
|---|---|
| **PC1** | A5 exact-set: CI > 0 **and** point **≥ +0.05** absolute |
| **PC2** | T3 overcount decreases by **≥ 0.10** absolute, CI favourable |
| **PC3** | A4 spurious decreases by **≥ 0.04** absolute, CI favourable |
| **PC4** | A3 missing not worse by more than **0.020** absolute |
| **PC5** | T2 undercount fraction not worse by more than **0.020** absolute |
| **PC6** | B5 T0-only timing MAE not worse by more than **8.0 ms** |
| **PC7** | `S → G` F1@50: CI favourable vs R1 |

**If any PC gate fails: STOP before the generator stage** — the E3-RIDGE schedule is never run through O2c.
This preserves the cheap falsification.

## 15. Must beat the train-only threshold control

| id | requirement (`E3-RIDGE-COUNT` vs `R1-TRAIN-THRESH`) |
|---|---|
| **TC1** | A5 exact-set: CI entirely > 0 **and** point difference **≥ +0.03** absolute |
| **TC2** | A3 missing non-inferior, margin **0.020** |
| **TC3** | A4 spurious non-inferior, margin **0.020** |

**If PC1–PC7 pass but TC1–TC3 fail, STOP before the generator stage** with the final verdict
`TRAIN-ONLY THRESHOLD CONTROL SUFFICIENT`, and no count-specific benefit is claimed.

## 16. Downstream evaluation (only if PC1–PC7 and TC1–TC3 all pass)

Frozen O2c at NFE 4 with source seed 0 for `R1-0.35`, `R1-TRAIN-THRESH`, `E3-RIDGE-COUNT` and
`ORACLE-COUNT-R1`, evaluated with the exact E2 contract. No retraining.

| id | requirement (`E3-RIDGE-COUNT` vs `O2C-R1-SCHEDULE`) |
|---|---|
| **DG1** | `P → S` adherence F1@50 **≥ 0.90** |
| **DG2** | `P → G` F1@50: CI favourable **and** point gain **≥ +0.05** |
| **DG3** | own-centre T6 non-inferior, margin **0.020** |
| **DG4** | own-centre T7 non-inferior, margin **0.020** |
| **DG5** | at least one of T6 / T7 improves with CI entirely favourable |
| **DG6** | GT-anchored J2 not worse by more than **0.020** |
| **DG7** | GT-anchored J3 not worse by more than **0.020** |

| id | requirement (`E3-RIDGE-COUNT` vs `R1-TRAIN-THRESH`, downstream) |
|---|---|
| **DC1** | `P → G` F1@50 not worse by more than **0.02** |
| **DC2** | neither own-centre T6 nor T7 worse by more than **0.02** |
| **DC3** | at least one of `P → G` F1@50, T6, T7, J2, J3 clearly improves with CI favourable |

## 17. Reporting, bootstrap and site secondary

**All six E2 blocks are mandatory for every validation schedule and generator arm** — A event set /
topology, B placement, C joint event, D adherence, E own-centre morphology with all coverages, F GT-anchored
joint structure with D1–D3. **No axis may be omitted.**

Bootstrap: the exact E2 unit — the underlying ECG-window cluster with all four site rows together,
`an0` / `k2s` stratified with equal subject weight, **2,000 replicates**, `default_rng(20260904)`. Positive
always means the NEW arm is better, except in tables explicitly labelled as damage tables. Window-fraction
metrics (A5, T0–T3) enter the bootstrap as per-window 0/1 indicators.

Site secondary (sternum / head / wrist / ankle, **no site causality claim**): schedule-side exact-set
fraction, T3 fraction, missing, spurious and T0-only timing MAE for `R1`, `TRAIN-THRESH` and `E3-RIDGE`;
generator-side `P → G` F1@50 and own-centre T6 / T7 if that stage is reached.

## 18. Final verdict tree — exactly one

- **A. BEAT-SET-FIRST COUNT CONSTRAINT SUPPORTED** — OC1–OC6, OG1–OG6, PC1–PC7, TC1–TC3, DG1–DG7 and
  DC1–DC3 all pass. Licenses a fresh confirmatory / richer event-existence experiment; proves no architecture.
- **B. COUNT CORRECTION CEILING SUPPORTED, MINIMAL PPG COUNT READOUT INSUFFICIENT** — Stage 0 passes but
  E3-RIDGE fails a PC gate. Do not modify O2c; a richer representation may be justified under a new
  preregistration.
- **C. TRAIN-ONLY THRESHOLD CONTROL SUFFICIENT** — PC gates may pass but TC1–TC3 fail. Do not build a larger
  count model.
- **D. COUNT-CONSTRAINED SCHEDULE IMPROVES, DOWNSTREAM BENEFIT NOT SUPPORTED** — PC and TC pass but a DG/DC
  gate fails. Do not expand the schedule model automatically.
- **E. COUNT-ONLY CEILING NOT SUPPORTED** — a Stage-0 OC or OG gate fails. Kill the count-only direction and
  never fit the ridge readout.
- **PRECHECK STOP — CANDIDATE EXTRACTION NOT IDENTICAL TO R1** — implementation fault, no scientific reading.

## 19. Interpretation boundary

If E3 succeeds, **never** say "beat count solves PPG-to-ECG", "count is the causal bottleneck", or "timing
is unimportant". The single permitted sentence is: *"Under the frozen development protocol, constraining the
event set using a PPG-derived count estimate improves the dominant overcount failure while preserving
separately measured timing and morphology."* Timing remains a separate E2 axis.

## 20. Artifacts

`docs/E3_BEAT_SET_FIRST_EVENT_GEOMETRY_PREREGISTRATION.md`, `docs/E3_BEAT_SET_FIRST_EVENT_GEOMETRY_REPORT.md`
and `artifacts/e3_beat_set_first/`: `provenance.json`, `frozen_component_manifest.json`,
`e2_contract_identity.json`, `candidate_extraction_audit.json`, `candidate_capacity.csv`,
`oracle_count_schedule_metrics.csv`, `oracle_count_generator_metrics.csv`, `oracle_count_bootstrap.csv`,
`train_threshold_grid.csv`, `train_only_threshold_control.json`, `train_feature_manifest.json`,
`count_readout_manifest.json`, `count_readout_coefficients.npz`, `validation_count_predictions.csv`,
`schedule_metrics.csv`, `schedule_bootstrap.csv`, `generator_metrics.csv`, `generator_bootstrap.csv`,
`site_metrics.csv`, `gates.csv`, `decision.json`, `figures/`.

**Never committed:** raw PPG, raw ECG, large hidden feature matrices, full generated prediction tensors,
and the ridge coefficient array. The hash-bearing JSON manifests of §12 are the exception described there.

## 21. Tests

**Repository** firewall, pins, A4 md5, C2 untouched. **E2 contract** exact `contract_v1` sha256, definitions
imported, no local metric duplication. **R1** exact checkpoint hashes, `requires_grad = False`, `eval()`
mode, the original 0.35 output reproduced. **Candidates** threshold-free extraction deterministic,
thresholding at 0.35 reproduces R1 bit-exactly, refractory unchanged at 32, no timing movement,
deterministic score/index tie break. **Oracle count** GT contributes count only, selected locations come
exclusively from R1 candidates, no GT location lookup. **Threshold control** exact grid 0.05:0.05:0.95,
train12 only, frozen lexicographic selection, validation never touched. **Feature** exact pre-logit frozen
R1 tensor, mean + max only, no site / subject / quality feature. **Ridge** scaler from train12 only,
`alpha = 1.0`, deterministic, no CV, no validation, exact rounding, structural clip `[0, 32]`.
**Evaluation** exact 2,048 cohort, the six mandatory E2 blocks, clustered bootstrap, all coverages.
**Stop logic** a Stage-0 failure prevents ridge fitting, a PC failure prevents O2c inference, a TC failure
prevents O2c inference, and the exact final verdict tree.

## 22. Execution and commit order

1 repository integrity → 2 freeze the E2 contract identity → 3 inspect and freeze the R1 pre-threshold score
and candidate path → 4 write this preregistration → **5 commit + push the preregistration** → 6 implement
threshold-free candidate extraction → 7 tests → **8 implementation commit + push** → 9 candidate-capacity
audit (**fail ⇒ report, result commit, STOP**) → 10 Stage-0 oracle-count schedule evaluation → 11 Stage-0
frozen O2c evaluation → 12 freeze the OC/OG gates (**any failure ⇒ report, result commit, STOP**) →
13 extract train12 frozen R1 features → 14 run the fixed train-only threshold grid → 15 freeze the selected
threshold → 16 fit the scaler + `Ridge(alpha=1.0)` → 17 freeze the count coefficients and hashes →
**18 only now load validation for the predicted-count evaluation** → 19 evaluate R1 / TRAIN-THRESH /
ORACLE-COUNT / E3-RIDGE → 20 freeze PC1–PC7 → 21 freeze TC1–TC3 (**failure ⇒ final verdict, report, STOP**)
→ 22 run the frozen O2c downstream arms → 23 DG1–DG7 → 24 DC1–DC3 → 25 site secondary → 26 freeze the final
verdict → 27 figures → 28 report → 29 full tests → **30 result commit + push** → 31 clean-tree verification
→ 32 STOP.
