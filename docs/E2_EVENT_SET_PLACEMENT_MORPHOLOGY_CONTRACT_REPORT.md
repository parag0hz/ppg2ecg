# E2 — Event-Set / Placement / Morphology Evaluation Contract — REPORT

**Freeze the measurement before building the next PPG predictor.**

## 0. Status

E2 is **measurement-contract construction after E1**. It is not independent confirmation, not a scientific
result by itself, not a deployable model, not a claim about PPG observability, not causal evidence, not
information-theoretic evidence, and not novelty or SOTA evidence. **The E1 formal verdict
`MIXED TOPOLOGY AND PLACEMENT LIMITATION` is unchanged and was not reinterpreted here.**

Preregistration and the frozen `contract_v1.json` were pushed as **`14a26a7`** *before* any validation
number existed; the implementation is **`9a1c75d`**. **NO TRAINING, NO WEIGHT UPDATE, NO NEW PPG PREDICTOR,
NO GENERATOR RETRAINING, NO THRESHOLD TUNING, NO TEST SUBJECT, NO C2** — every network was loaded frozen in
`eval()` with `requires_grad = False` and no optimizer was constructed. Validation cost 1.7 min and
1,763 MiB.

## FINAL E2 VERDICT

**A. EVENT-GEOMETRY EVALUATION CONTRACT ACCEPTED** — V1–V13 all pass.

The suite separates event-set correctness, event placement, own-centre beat morphology and GT-anchored joint
structure on the frozen diagnostic arms. This licenses a **new preregistered** predictor experiment; it does
not license training anything under E2.

## 1. Repository

| item | value |
|---|---|
| start SHA | `a5af4af` (E1 result commit) |
| prereg + contract SHA | **`14a26a7`** |
| implementation SHA | **`9a1c75d`** |
| result SHA | this commit |
| clean? | yes — `git status` empty, `HEAD == origin/main` |
| test? | **no test subject** — `kjd` / `ssx` in no E2 source; `test_subjects_loaded: []` |
| C2? | **still deferred** |

Pins `6cd70cde` / `bf60cd7c` unchanged; A4 md5 `31c042d2…` unchanged. Full suite at the result commit:
**510 passed**, of which 32 are the E2 contract tests.

## 2. E1 source integrity

Every E1 artifact was hashed into `source_artifact_manifest.json` before use (18 files, e.g.
`own_center_window_metrics.csv` `3d7c5b38…`, `coverage_metrics.csv` `3d54afaf…`,
`r1_topology_strata.csv` `cc0a83a0…`).

**Reused without rerun:** the own-centre and GT-anchored morphology reference rows, the chain coverages and
the R1 reference-card inputs. **Rerun required** for three contract quantities that E1's artifacts cannot
supply: per-window topology class and per-window schedule placement (E1 wrote only per-arm aggregates —
these need no generator), joint event fidelity `P → G` at 100/150/200 ms, and the full detected event list
`P` (E1 stored only chained beats). E2 therefore ran a **deterministic frozen rerun of exactly the five
validation arms** at NFE 4, source seed 0.

**Reproduction gate: PASS at `max |Δ| = 0.000e+00` across 99 checks** — bit-exact. The four synthetic arms
were checked against E1's per-arm morphology and coverage rows; R1, for which E1 never wrote a per-arm
morphology row, was checked against E1's **per-stratum** T0/T1/T2/T3 rows (own T6/T7, GT-anchored T6/T7 and
schedule MAE per stratum) plus its coverage row.

## 3. Frozen taxonomy

| family | question | metrics |
|---|---|---|
| **AXIS A — EVENT SET / TOPOLOGY** | did the schedule supply the correct number/set of events? | A1 abs count error · A2 count deviation · A3 missing · A4 spurious · A5 exact-set fraction · A6 T0/T1/T2/T3 |
| **AXIS B — EVENT PLACEMENT** | for corresponding events, how accurately placed? | B1 median AE · B2 MAE · B3 p90 · B4 p95 · **B5/B6 T0-only** |
| **AXIS C — OWN-CENTRE BEAT MORPHOLOGY** | ignoring placement, is the beat's shape right? | C1–C4 own-centre T4/T6/T7/T8 · C5–C8 local raw RMSE / deriv RMSE / curvature / correlation |
| **JOINT — GT-ANCHORED JOINT STRUCTURE** | is the right structure at the right GT event? | J1–J4, plus D1–D3 same-functional alignment sensitivity |
| *JOINT EVENT FIDELITY* | existence + count + placement together | F1@50/100/150/200, precision, recall for `S → G` and `P → G` |
| *GENERATOR ADHERENCE* | did the generator follow what it was given? | AD_F1@50/100, missing, spurious, timing MAE |

38 metrics, each with a formula, direction, tier, unit and normaliser in `contract_v1.json`. **Naming rules
are frozen**: the GT-anchored family is never "pure morphology", the F1 family is never "pure timing
accuracy", and T4/T6/T7/T8 are never placement evidence.

## 4. Matching contract

| pair | tolerance | matcher |
|---|---|---|
| `S → G` | **± 150 ms** | exact construction identity for synthetic arms; frozen **monotonic one-to-one DP** for predicted schedules (maximise cardinality → minimise total absolute error → deterministic tie break) |
| `P → S` | **± 50 ms** | the frozen greedy one-to-one matcher, unchanged |
| `P → G` | 50 / 100 / 150 / 200 ms | the same frozen matcher |

The `S → G` assignment for predicted schedules is **TARGET-DERIVED DIAGNOSTIC MATCHING** and may never
shift, repair, insert into or delete from the schedule, nor touch inference. A test asserts the schedule
array is byte-identical before and after matching.

## 5. Metric contract — the complete mandatory block, on all five arms

### A. Event set

| arm | A5 exact-set | A1 abs count err | A3 missing | A4 spurious | T0 | T1 | T2 | T3 |
|---|---|---|---|---|---|---|---|---|
| ORACLE | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 2048 | 0 | 0 | 0 |
| JITTER_8 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 2048 | 0 | 0 | 0 |
| MISS1 | 0.0000 | 1.0000 | 0.1051 | 0.0000 | 0 | 0 | 2048 | 0 |
| EXTRA1 | 0.0000 | 1.0000 | 0.0000 | 0.1051 | 0 | 0 | 0 | 2048 |
| **R1-SCHEDULE** | **0.4966** | 1.1372 | 0.1062 | 0.2157 | 1017 | 132 | 73 | **826** |

### B. Placement

| arm | B2 MAE | B3 p90 | **B5 T0-only MAE** | **B6 T0-only p90** | n T0 windows |
|---|---|---|---|---|---|
| ORACLE | 0.00 ms | 0.00 | 0.00 | 0.00 | 2048 |
| JITTER_8 | 33.04 ms | 54.39 | 33.04 | 54.39 | 2048 |
| MISS1 | 0.00 ms | 0.00 | — | — | **0** |
| EXTRA1 | 0.00 ms | 0.00 | — | — | **0** |
| **R1-SCHEDULE** | **43.82 ms** | 70.93 | **28.63 ms** | 47.33 | 1017 |

B5/B6 are undefined for MISS1/EXTRA1 because those arms have **no** T0 window by construction; the sample
size is printed rather than the absence hidden. For R1 the T0-only figure (28.63 ms) is markedly better than
the all-window figure (43.82 ms) — exactly the isolation of timing from count error that B5 exists for.

### C. Joint event fidelity · D. Generator adherence

| arm | SG F1@50/100/150 | PG F1@50/100/150 | AD F1@50 | AD MAE | label |
|---|---|---|---|---|---|
| ORACLE | 1.0000 / 1.0000 / 1.0000 | 0.9840 / 0.9886 / 0.9900 | 0.9840 | 1.63 ms | HIGH ADHERENCE |
| JITTER_8 | 0.7666 / 1.0000 / 1.0000 | 0.7628 / 0.9842 / 0.9856 | 0.9816 | 1.22 ms | HIGH ADHERENCE |
| MISS1 | 0.9445 / 0.9445 / 0.9445 | 0.8509 / 0.8693 / 0.8850 | 0.8917 | 2.55 ms | — |
| EXTRA1 | 0.9501 / 0.9501 / 0.9501 | 0.9396 / 0.9425 / 0.9445 | 0.9473 | 1.53 ms | HIGH ADHERENCE |
| R1-SCHEDULE | 0.6157 / 0.7788 / 0.8598 | 0.6045 / 0.7635 / 0.8405 | 0.9357 | 2.05 ms | HIGH ADHERENCE |

### E. Own-centre morphology · F. GT-anchored joint structure · alignment sensitivity

| arm | C1 T4 | C2 T6 | C3 T7 | C4 T8 | C6 deriv | J1 raw | J2 deriv | J3 curv | J4 corr | D2 | D3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ORACLE | 0.4653 | 0.4833 | 0.4861 | 0.5350 | 0.1103 | 0.2473 | 0.1153 | 0.0995 | 0.9405 | +0.0009 | +0.0006 |
| **JITTER_8** | 0.5367 | 0.5482 | 0.4877 | 0.5375 | **0.1147** | 0.6170 | **0.3809** | **0.2458** | **−0.0354** | **+0.2607** | **+0.1431** |
| **MISS1** | **1.0401** | **1.0451** | **0.7146** | 0.6901 | 0.1507 | 0.3532 | 0.1558 | 0.1237 | 0.8776 | **+0.0004** | **+0.0003** |
| **EXTRA1** | 0.8320 | 0.8209 | 0.6009 | 0.6173 | 0.1326 | 0.3114 | 0.1360 | 0.1112 | 0.9140 | **+0.0001** | **+0.0001** |
| R1-SCHEDULE | 0.7480 | 0.7483 | 0.5861 | 0.6804 | 0.1307 | 0.5584 | 0.3437 | 0.2289 | 0.0962 | +0.2064 | +0.1148 |

Coverage (pooled cohort ratios, required beside every morphology number):

| arm | C1 `S→G` | C2 `P→S` | C3 full chain | C4 GT excluded | C5 generated excluded |
|---|---|---|---|---|---|
| ORACLE | 1.0000 | 0.9832 | 0.9753 | 0.0247 | 0.0231 |
| JITTER_8 | 1.0000 | 0.9792 | 0.9711 | 0.0289 | 0.0241 |
| MISS1 | 1.0000 | 0.9157 | 0.9098 | 0.1841 | 0.1228 |
| EXTRA1 | 0.9064 | 0.9659 | 0.8683 | 0.0420 | 0.0906 |
| R1-SCHEDULE | 0.8025 | 0.9327 | 0.7427 | 0.1775 | 0.1867 |

## 6. The 2 × 2 dissociation — why the contract is accepted

This is the point of the whole exercise, and the numbers make it unusually clean.

| perturbation | AXIS A responds? | AXIS B responds? | AXIS C responds? | JOINT responds? |
|---|---|---|---|---|
| **JITTER_8** (timing only) | **no** (100 % T0) | **yes** (0 → 33.04 ms) | **barely** (C6 0.1103 → 0.1147) | **yes** (J2 0.1153 → 0.3809, J4 0.94 → −0.04) |
| **MISS1 / EXTRA1** (count only) | **yes** (all T2 / all T3) | **no** (MAE stays 0.00 ms) | **yes** (C2 0.4833 → 1.0451 / 0.8209) | **barely** (J2 0.1153 → 0.1558 / 0.1360) |

Each axis moves for its own perturbation and stays put for the other. The same-functional alignment
sensitivity makes the separation explicit: **D2 is +0.2607 under pure timing corruption and +0.0004 /
+0.0001 under pure count corruption.**

### Contract validation — JITTER_8 (paired, ECG-window clustered, 2,000 replicates, `default_rng(20260904)`; positive = more damage)

| quantity | ORACLE | JITTER_8 | effect [95 % CI] |
|---|---|---|---|
| topology | 100 % T0 | 100 % T0 | — (V1) |
| schedule timing MAE | 0.00 ms | 33.04 ms | **+33.0363** [+32.7808, +33.3126] |
| own-centre local deriv RMSE (C6) | 0.1103 | 0.1147 | (reference, not a gate) |
| GT-anchored local deriv RMSE (J2) | 0.1153 | 0.3809 | **+0.2656** [+0.2626, +0.2687] |
| GT-anchored local curvature (J3) | 0.0995 | 0.2458 | **+0.1463** [+0.1444, +0.1482] |
| adherence `P → S` F1@50 | 0.9840 | 0.9816 | HIGH ADHERENCE |

### Placement excess (same functional vs same functional — this replaces E1's ambiguous P3/P4)

| quantity | effect [95 % CI] | verdict |
|---|---|---|
| **DerivativePlacementExcess** | **+0.2598** [+0.2566, +0.2631] | damage confirmed |
| **CurvaturePlacementExcess** | **+0.1425** [+0.1405, +0.1445] | damage confirmed |
| DerivativePlacementExcess, window-level variant *(secondary)* | +0.2612 [+0.2581, +0.2645] | damage confirmed |
| CurvaturePlacementExcess, window-level variant *(secondary)* | +0.1443 [+0.1424, +0.1464] | damage confirmed |
| own-centre T6 vs GT-anchored derivative RMSE | **not computed** | forbidden by `contract_v1.prohibited_comparisons` |

The contract defines D1–D3 as **per-beat** differences that are then aggregated by the frozen order, so the
gate uses `median(J − C)`. Because `median(J − C) ≠ median(J) − median(C)` in general, the window-level
variant is published beside it; the two agree to within 0.0019 here, and only the contracted form gates.

### Contract validation — topology (positive = more damage)

| contrast | own-centre T6 | own-centre T7 |
|---|---|---|
| MISS1 vs ORACLE | **+0.5617** [+0.5339, +0.5902] | **+0.2285** [+0.2075, +0.2501] |
| EXTRA1 vs ORACLE | **+0.3378** [+0.3177, +0.3593] | **+0.1147** [+0.0984, +0.1321] |
| MISS1 vs JITTER_8 | **+0.4969** [+0.4712, +0.5222] | **+0.2269** [+0.2082, +0.2465] |
| EXTRA1 vs JITTER_8 | **+0.2730** [+0.2559, +0.2908] | **+0.1132** [+0.0989, +0.1280] |
| *(reference)* JITTER_8 vs ORACLE | +0.0648 [+0.0538, +0.0758] | +0.0016 [−0.0083, +0.0116] unresolved |

These are **measurement sanity checks that the axes respond**, not a new causal result — E1 already
established the underlying contrast, and E2 reproduces it bit-exactly.

## 7. Gates

| gate | requirement | result |
|---|---|---|
| **V1** | JITTER_8 topology 100 % T0 | **PASS** (2048/2048) |
| **V2** | schedule timing MAE clearly increases | **PASS** (+33.04 ms) |
| **V3** | GT-anchored local derivative RMSE clearly worsens | **PASS** (+0.2656) |
| **V4** | GT-anchored local curvature clearly worsens | **PASS** (+0.1463) |
| **V5** | `P → S` adherence stays HIGH | **PASS** (0.9816 ≥ 0.90) |
| **V6** | DerivativePlacementExcess CI > 0 | **PASS** (+0.2598) |
| **V7** | CurvaturePlacementExcess CI > 0 | **PASS** (+0.1425) |
| **V8** | MISS1 own-centre T6 worse than ORACLE | **PASS** (+0.5617) |
| **V9** | MISS1 own-centre T7 worse than ORACLE | **PASS** (+0.2285) |
| **V10** | EXTRA1 T6 or T7 worse than ORACLE | **PASS** (both) |
| **V11** | MISS1 excess over JITTER_8, both T6 and T7 | **PASS** |
| **V12** | EXTRA1 excess over JITTER_8, at least one | **PASS** (both) |
| **V13** | all six mandatory blocks computable for R1 with finite values and coverage | **PASS** |

Sanity checks: MISS1 is 2048/2048 T2 and EXTRA1 is 2048/2048 T3, as constructed.

## 8. R1 reference card (every value loaded from a frozen artifact, none typed)

| quantity | value |
|---|---|
| exact-set fraction | **0.49658** |
| topology distribution | T0 0.4966 · T1 0.0645 · T2 0.0356 · **T3 0.4033** |
| dominant error | **T3 overcount** |
| schedule F1@50 / F1@150 | 0.61567 / 0.85984 |
| schedule timing MAE (O3 ± 50 ms matched) | 22.804 ms |
| schedule missing / spurious | 0.36842 / 0.47800 |
| `P → S` adherence F1@50 | 0.93571 |
| exact-set own-centre T6 / T7 | 0.45706 / 0.45088 |

Provenance hashes for the three source files are stored in `r1_reference_card.json`.

**A caution for anyone using both numbers.** The card's 22.80 ms comes from O3's `S → G` timing over pairs
matched at **± 50 ms**; the contract's B2 = 43.82 ms comes from the **± 150 ms** monotonic identity. They are
different quantities — a tighter matcher conditions the average on the easy pairs — and the contract's
B-axis figure is the one future experiments must report. The T0-only B5 = 28.63 ms is the like-for-like
number to compare across schedules.

## 9. What is now frozen for future experiments

Event-set metrics (A1–A6, with F1 explicitly **excluded** as the primary event-set measure) · placement
metrics (B1–B6, including the T0-only pair) · joint-event metrics (F1@50/100/150/200, precision, recall, for
both `S → G` and `P → G`) · generator adherence (AD_*) · own-centre morphology (C1–C8) · GT-anchored joint
structure (J1–J4) and same-functional alignment sensitivity (D1–D3) · the ± 150 ms / ± 50 ms matching and
the monotonic DP identity · the `[-10, +15]` support and its eligibility rule · the five coverage quantities
and the rule that morphology without coverage is invalid · beat → window → subject → equal-subject-macro
aggregation · the ECG-window clustered bootstrap at 2,000 replicates and seed 20260904 · the universal
effect orientation and the separate damage convention · the six-block mandatory reporting set, which no
future experiment may shorten because a result is unfavourable.

A future predictor experiment must preregister its reference arms, margins, seed count and compute budget
**before training**, and may not change any of the above.

## 10. What E2 does NOT prove

- **No new model, no training, no fresh test.** Post-E1 measurement design, not confirmation.
- **No causal result**, no PPG-observability claim, no deployability, no novelty or SOTA claim.
- The validation shows the axes *respond to controlled perturbations of a frozen generator*; it does not show
  that they are sufficient for every future failure mode, only that they separate the four E1 exposed.
- Two development subjects, one cohort, one generator, five arms.
- The R1 arm's beat identity comes from target-derived diagnostic matching; a different assignment rule
  would move the R1 strata (the synthetic arms, which carry V1–V12, use exact construction identity).
- Own-centre morphology still does not penalise missing beats — that is deliberate, it is why coverage is
  mandatory, and AXIS A is where those penalties live.

## 11. Recommended next step (recommendation only — nothing implemented)

The verdict is A, so the licensed next move is a **new preregistered experiment**:

> **E3 — Beat-Set-First PPG Event Geometry.** Primary engineering target: **reduce the R1 overcount /
> spurious-event rate without collapsing recall** — the reference card puts 40.3 % of windows in T3 overcount
> against 3.6 % in T2 undercount, with spurious 0.478 against missing 0.368 — while evaluating timing
> **separately** under this contract, never merged into one score.

E3 must preregister its schedule-side and generator-side reference arms, its improvement and
non-inferiority margins, seed count and compute budget before any training, and must report all six blocks.
**E3 is not implemented here**: no Transformer, no phase network, no uncertainty model, no set decoder, no
count head and no new detector was built, and C2 remains deferred.

## Artifacts

Version-controlled definitions: `artifacts/e2_evaluation_contract/contract_v1.json`, `metric_taxonomy.json`,
`matching_contract.json`, `aggregation_contract.json`, `coverage_contract.json`.

Un-committed validation results (this report is their durable record): `provenance.json`,
`source_artifact_manifest.json`, `frozen_metric_manifest.json`, `r1_reference_card.json`,
`e1_reproduction.csv`, `contract_validation_metrics.csv`, `contract_validation_windows.csv`,
`contract_validation_beats.csv`, `contract_validation_bootstrap.csv`, `placement_excess.csv`,
`topology_validation.csv`, `validation_gates.json`, `decision.json`, `figures/` (FIG 1–4).
**No checkpoint, no training log, no predictor output, no new model directory.**

Code: `src/ppg2ecg/evaluation/event_geometry_contract.py` (the module future experiments import),
`scripts/e2_validate.py`, `scripts/e2_figures.py`, `tests/test_e2_contract.py` (32 tests).
