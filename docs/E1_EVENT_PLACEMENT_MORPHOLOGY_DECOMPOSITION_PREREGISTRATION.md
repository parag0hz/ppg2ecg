# E1 — Event Placement vs Beat-Morphology Decomposition — PREREGISTRATION

**Should the next PPG-side module primarily improve beat-set / count correctness, or fine event timing?**

## 0. Scientific status — read first

**E1 was designed AFTER the O3 results were known.** O3 itself was designed after O2c. E1 is a
**post-O3 problem-decomposition diagnostic**, and every threshold, stratum boundary, gate and interpretation
below is therefore a **frozen post-hoc diagnostic criterion**, *not* independent preregistered confirmation.
Nothing in E1 is confirmatory evidence, a deployable method, a new generator, proof of a causal mechanism,
proof of PPG observability, or an information-theoretic result.

**NO TRAINING. NO WEIGHT UPDATE. NO NEW PREDICTOR. NO GENERATOR RETRAINING. NO NEW ATTENTION. NO NEW LOSS.
NO TEST SUBJECT. NO C2. NO THRESHOLD TUNING.** Every network is loaded frozen, in `eval()`, with
`requires_grad = False`; no optimizer is constructed anywhere in E1.

E1 decomposes O3's `TOLERANCE REGION TOO NARROW` into three axes: **A event topology / count**, **B event
placement**, **C beat morphology measured at each waveform's own centre**.

## 1. Frozen components (identities asserted by test)

| component | file sha256 | state sha256 |
|---|---|---|
| B | `557c7054…` | `47d7ccb9…` |
| O2c (step 10,046, 4,568,707 params) | `5aab09be…` | `f1cc44b3…` |
| R1 Global-TCN | `bfe76ea6…` | `0986a7af…` |
| `o2b_warp.py` / `o2_warp.py` | `cb4d1866…` / `046becfb…` | — |

Operator configuration imported unchanged (`W = 10`, `MIN_INT_SPACING = 21`, `CORE_OFFSET_TOL = 1e-6`,
`round_half_to_even`, bilinear `grid_sample`). R1 runs at its frozen operating point: threshold **0.35**,
NMS refractory **32** samples, no correction of any kind. **No modification, no threshold tuning.**

## 2. Cohort

The exact O3 primary cohort: **2,048 rows** (`an0`, `k2s`), **19,834 GT beats**, **1,922 underlying
ECG-window clusters**, asserted at run time. No new cohort, no validation expansion, no test subject.

## 3. Primary arms — the minimum diagnostic set

**B**, **O2c ORACLE**, **JITTER_2 rep0** (last accepted timing level), **JITTER_4 rep0** (first rejected
level), **JITTER_8 rep0** (severe timing-only corruption), **MISS1 rep0**, **EXTRA1 rep0**,
**R1-SCHEDULE**. **No arm may be added after E1 results are seen.** MISS2 / EXTRA2 may be quoted from the
frozen O3 summaries but **do not enter E1 primary classification**.

B has no external supplied schedule, so it participates in GT-fidelity comparisons only and never in the
S-adherence decomposition.

## 4. Prediction source

O3 deliberately saved no per-window supplied schedule, no generated waveform and no detected event list
(predictions are never committed in this project). The frozen O3 artifacts therefore **cannot** supply the
beat-level data E1 needs, so E1 uses a **DETERMINISTIC FROZEN RERUN** of the eight primary arms:
NFE 4, source seed 0, the exact O3 cohort, the asserted frozen model hashes, and supplied schedules
reconstructed by the frozen `o3_schedule` generators from the same salts (`o3-jitter-v1`, `o3-miss-v1`,
`o3-extra-v1`) and by the frozen R1 probe. **No training.**

Two integrity gates run before any E1 interpretation:

1. **Schedule reconstruction gate** — the reconstructed schedules must reproduce the frozen O3
   `perturbation_manifest.csv`, `schedule_precheck.csv` and `schedule_quality_metrics.csv` rows for the six
   synthetic arms, and the frozen `r1_schedule_manifest.csv` / `r1_schedule_quality.csv` rows for R1,
   at `|Δ| ≤ 1e-9`.
2. **Generator reproduction gate** — B and O2c ORACLE must reproduce their frozen O3 / O2c summary rows
   (B F1 excess 0.3175618683, O2c ORACLE F1 excess 0.8592510053 and T4/T6/T7/T8
   0.4071540678 / 0.4019096984 / 0.4169922852 / 0.4138793945), and every rerun arm must reproduce its frozen
   O3 `synthetic_generator_metrics.csv` / `r1_generator_metrics.csv` row, at `|Δ| ≤ 1e-6`.

Any failure ⇒ report, **STOP**. `prediction_reuse_manifest.json` records `REUSED O3 PREDICTIONS` vs
`DETERMINISTIC FROZEN RERUN` and both gate results.

## 5. Three event sets — never collapsed

`G` = GT ECG R events · `S` = the schedule supplied to O2c · `P` = events detected in the generated ECG with
the frozen detector. For every O2c arm, **S vs G**, **P vs S** and **P vs G** are reported separately and are
never combined into a single number.

## 6. Event identity

**Synthetic arms.** ORACLE / JITTER_2 / JITTER_4 / JITTER_8: every supplied beat carries the exact
originating `gt_beat_id` (index k ↔ index k). MISS1: retained beats keep their exact identity and the deleted
GT beat has **no** supplied counterpart. EXTRA1: the original supplied beats keep their identity and the
inserted beat has `gt_beat_id = NONE`; **no GT morphology target is ever assigned to an inserted beat.**

**R1 arm.** R1 has no known beat identity, so **one frozen diagnostic assignment** is used: a **monotonic
one-to-one** match of `S_R1` to `G` at a maximum tolerance of **± 150 ms**. The existing frozen matcher
(`rpeaks.match_rpeaks`) is greedy-nearest and can produce crossing assignments, so it does **not** qualify;
E1 therefore adds one explicit dynamic-programming matcher, written and tested **before** any metric, with
the frozen objective hierarchy

1. maximise the number of matches within ± 150 ms,
2. among equal-cardinality solutions, minimise the total absolute timing error,
3. deterministic tie break by transition preference `match → skip-ref → skip-pred`, scanning indices in
   ascending order.

This is **TARGET-DERIVED DIAGNOSTIC MATCHING**. It is used only to attach a GT identity for analysis and
**never** to modify, correct or re-time the supplied R1 schedule. Unmatched supplied beats are *spurious
schedule events*; unmatched GT beats are *missing schedule events*.

## 7. Generated-to-supplied chaining

`P → S` is matched within **± 50 ms** with the **exact frozen O3 adherence matcher**
(`rpeaks.match_rpeaks`, greedy one-to-one, unchanged). The chain is `P → S → G`: for synthetic arms the second
link is the exact identity of §6, for R1 it is the diagnostic assignment. **Only successfully chained beats
are eligible for the own-centre analysis.** No GT correction of a schedule and no event shift is applied
anywhere.

## 8. Coverage — reported on every morphology row

**C1** schedule-to-GT identity coverage (fraction of supplied beats with a GT identity) · **C2**
generated-to-supplied adherence coverage (fraction of supplied beats matched by a generated beat) · **C3**
full-chain coverage `P → S → G` · **C4** fraction of GT beats excluded from the own-centre comparison ·
**C5** fraction of generated beats excluded. **No morphology-only metric may appear without its coverage on
the same row**, so that "good morphology because the hard beats disappeared" cannot hide.

**Adequacy precondition (frozen now):** every one of the six synthetic O2c arms must reach full-chain
coverage **C3 ≥ 0.80**. If any does not, the verdict is **DECOMPOSITION INCONCLUSIVE**. R1 coverage is
reported descriptively and governed instead by the stratum-size rules of §14–§16.

## 9. Axis A — event topology / count

Per window: GT beat count `K`, supplied count `M`, signed count error `M − K`, absolute count error, missing
and extra beat counts, missing fraction and spurious fraction (both normalised by `K`). Every window is
classified into exactly one stratum, using the § 6 matching at **± 150 ms**:

| id | definition |
|---|---|
| **T0** EXACT EVENT SET | `M == K` **and** every supplied event matches a unique GT event within ± 150 ms **and** no unmatched GT **and** no unmatched supplied event |
| **T1** COUNT CORRECT, EVENT SET IMPERFECT | `M == K` but at least one unmatched event at ± 150 ms |
| **T2** UNDERCOUNT | `M < K` |
| **T3** OVERCOUNT | `M > K` |

For the synthetic arms this classification is a sanity check with a known expected answer.

## 10. Axis B — event placement

For identity-matched schedule/GT pairs: signed timing error, absolute timing error, median AE, MAE, p90 AE,
p95 AE. For generated beats chained through `S` to `G`: generated-to-GT timing AE and generated-to-supplied
timing AE. This separates **schedule placement error** from **generator adherence error**.

## 11. Axis C — own-centre beat morphology

For every chained beat with generated event `p`, supplied event `s` and originating GT event `g`, the
generated window is centred on **`p`** and the GT window on **`g`**, and they are compared in **local beat
coordinates**, so global placement error is removed by construction.

Fixed support **`[-10, +15]` samples** around each signal's own event — it spans the frozen QRS-width search
and contains the O1/M1 QRS primitives, and it was fixed before any E1 result. **A beat is eligible iff
`c - 11 ≥ 0` and `c + 15 ≤ 1023` for its own centre `c`** (the O1 validity rule `c-11 ≥ 0`, `c+12 ≤ T`
extended to the width support). **No local cross-correlation shift, no DTW, no oracle shift or amplitude
optimisation, no normalisation beyond the frozen preprocessing.**

### Primary own-centre metrics (exact O1 primitives, exact O1 train IQRs)

| id | quantity | normaliser |
|---|---|---|
| **M1** | `|p2p_gen − p2p_gt|` | O1 T4 train IQR 0.50532 |
| **M2** | `|max|d1|_gen − max|d1|_gt|` | O1 T6 train IQR 0.22995 |
| **M3** | `|mean(d2²)_gen − mean(d2²)_gt|` | O1 T7 train IQR 0.03380 |
| **M4** | `|width_gen − width_gt|` | O1 T8 train IQR 31.25 |

Aggregation: **per-window median over eligible beats, then equal-subject macro.** These are **OWN-CENTRE
SHAPE DIAGNOSTICS**; they do not penalise missing beats directly, which is exactly why §8 coverage travels
with them.

### Secondary local waveform diagnostics

Over the same local windows: **W1** local raw RMSE, **W2** local derivative RMSE, **W3** local curvature
error, **W4** local correlation. Secondary only.

## 12. GT-anchored comparator and alignment sensitivity

For the **same chained beats**, the generated waveform is instead taken around **`g`** (not `p`). Two
GT-anchored families are computed, and the distinction is stated because §22 of the task is ambiguous
between them:

- **waveform-style** (the §17 comparator): GT-anchored local derivative RMSE and local curvature error;
- **same-functional** GT-anchored **T6** and **T7**, i.e. exactly M2 and M3 with the generated window
  centred on `g` instead of `p`.

`AlignmentSensitivity = GT-anchored error − own-centre error` for each family. Positive means the metric
worsens when global event placement is enforced. This is an **alignment-sensitivity diagnostic only** and is
never called a causal placement penalty.

## 13. Bootstrap and effect orientation

Per-beat values are **first aggregated to the window** (median over eligible beats), and only windows are
bootstrapped. Unit: the **underlying ECG window cluster** (all four site rows move together),
subject-stratified with equal `an0` / `k2s` weight, **2,000 replicates**, `default_rng(20260904)`. Beats are
never bootstrap units. Windows with no eligible beat in either arm of a contrast are dropped and the dropped
count is reported.

**Damage(A, B) = Error_A − Error_B**, so a **positive effect always means the FIRST arm is worse**. This
orientation is never reversed between tables. Named contrasts:
`JITTER4_DAMAGE = Damage(JITTER_4, ORACLE)`, `JITTER8_DAMAGE = Damage(JITTER_8, ORACLE)`,
`MISS1_DAMAGE = Damage(MISS1, ORACLE)`, `EXTRA1_DAMAGE = Damage(EXTRA1, ORACLE)`.

**TopologyExcessDamage_MISS = Damage(MISS1, ORACLE) − Damage(JITTER_8, ORACLE)** (equivalently
`Error_MISS1 − Error_JITTER8`), and the same for EXTRA1. Positive means one count/topology error damages
own-centre morphology **more than the most severe tested timing jitter**. This is the main E1 contrast.

## 14. Placement-decomposition gates (descriptive pattern, paired bootstrap, no absolute threshold)

| id | requirement (all on JITTER_4 unless stated) |
|---|---|
| **P1** | GT-anchored **local derivative RMSE** damage vs ORACLE has CI entirely > 0 |
| **P2** | GT-anchored **local curvature error** damage vs ORACLE has CI entirely > 0 |
| **P3** | own-centre **T6** damage < GT-anchored **T6** damage (same functional, same units, same beats), paired CI of the difference entirely > 0 |
| **P4** | own-centre **T7** damage < GT-anchored **T7** damage, paired CI of the difference entirely > 0 |

If P1/P2 reproduce but P3/P4 do not, the O3 "placement failure" reading is **weakened**, and the report must
say so.

## 15. Topology-damage gates

Topology priority is supported if **all** hold:

| id | requirement |
|---|---|
| **C1** | MISS1 own-centre **T6** clearly worse than ORACLE (CI entirely > 0) |
| **C2** | MISS1 own-centre **T7** clearly worse than ORACLE |
| **C3** | at least one of EXTRA1 own-centre T6 / T7 clearly worse than ORACLE |
| **C4** | `TopologyExcessDamage_MISS > 0` with CI entirely > 0 for **both** T6 and T7 |
| **C5** | `TopologyExcessDamage_EXTRA > 0` with CI entirely > 0 for **at least one** of T6 / T7 |

No percentage threshold anywhere.

## 16. R1 decomposition (secondary but decision-relevant)

Whole-arm: T0/T1/T2/T3 distribution, schedule timing AE on matched events, `P → S` adherence, own-centre
T4/T6/T7/T8, GT-anchored derivative/curvature, alignment sensitivity.

**Stratum A — `R1-SET-CORRECT`**: windows with `M == K`, every supplied beat matched to a unique GT beat
within ± 150 ms, no unmatched GT and no unmatched supplied beat (i.e. exactly T0). Report rows, unique ECG
windows and the per-subject split. **Minimum for interpretation: ≥ 100 unique ECG windows in total AND ≥ 30
unique ECG windows from each validation subject.** If not met, label **INSUFFICIENT STRATUM COVERAGE**; the
threshold is never lowered.

**Timing bins inside `R1-SET-CORRECT`**, by per-window schedule matched-timing MAE, fixed before any E1
metric (16 ms ≈ the O3 `J_MAX` boundary of 15.625 ms; 32 ms ≈ twice that scale):
**BIN A ≤ 16 ms · BIN B > 16 and ≤ 32 ms · BIN C > 32 ms.** Never altered. Per bin: N windows, own-centre
T6/T7, GT-anchored derivative/curvature, alignment sensitivity, and the arm's F1 excess where available.

**Stratum B — `R1-TOPOLOGY-WRONG`**: any missing or extra schedule event, `M ≠ K`, or an incomplete
one-to-one match at ± 150 ms. Same own-centre T6/T7. UNDERCOUNT and OVERCOUNT are reported separately only if
each reaches ≥ 100 unique ECG windows in total and ≥ 30 per subject; otherwise they stay pooled.

**Timing-controlled comparison (§17 of the task).** If powered, compare `R1-TOPOLOGY-WRONG` against
`R1-SET-CORRECT` **within** each of the three timing bins, using **all** windows in the bin (no
nearest-neighbour optimisation), with the same subject-stratified clustered bootstrap. This remains
**observational**: the permitted phrasing is *"within coarse matched-timing strata…"*, never *"count error
causes…"*. The synthetic MISS/EXTRA arms remain the cleaner intervention.

## 17. Generator adherence label

For every O2c supplied-schedule arm, `P → S` F1@50 is reported. **`HIGH ADHERENCE` is defined now as
≥ 0.90.** The label is descriptive, **does not enter the global verdict**, and exists only to locate whether
the failure is upstream of the generator. If R1's `P → S` stays ≥ 0.90 while `S → G` is poor, the permitted
sentence is *"the frozen generator generally follows the supplied geometry; schedule error remains
upstream"* — no causal statement.

## 18. Source-stability reuse

The frozen O3 multi-source results are **reused, not rerun**: ORACLE, JITTER8, MISS1, EXTRA1, R1 —
beat-count SD, pairwise event F1@50, waveform SD. The question asked is descriptive: does source-driven event
instability associate more strongly with topology/count corruption than with timing jitter? **No causal model
is fitted.**

## 19. Verdicts — exactly one

- **A. EVENT-TOPOLOGY / COUNT PRIORITY SUPPORTED** — P1–P4 supported **and** C1–C5 all pass.
- **B. FINE PLACEMENT PRIORITY SUPPORTED** — not A, the topology excess-damage gate C4 fails, and for
  **both** T6 and T7 the JITTER_8 damage point estimate is ≥ the MISS1 damage point estimate, with at least
  one of them having CI entirely > 0.
- **C. MIXED TOPOLOGY AND PLACEMENT LIMITATION** — not A and not B, while **both** at least one JITTER_8
  own-centre damage (T6 or T7) and at least one MISS1 own-centre damage have CI entirely > 0.
- **D. DECOMPOSITION INCONCLUSIVE** — the §8 adequacy precondition fails, or either integrity gate of §4
  fails, or none of A/B/C applies.

**R1 observational strata can never override conflicting synthetic intervention evidence**: the verdict is
selected by the synthetic contrasts alone, and the R1 strata are reported alongside it.

## 20. Claim limits

If verdict A, the allowed sentences are exactly: *"The controlled O3 perturbations indicate that event-set
correctness is a higher-priority schedule failure mode than fine timing for preserving own-centre QRS
morphology."* and *"Timing jitter primarily exposes placement sensitivity in GT-anchored structure
metrics."*

**Never allowed**: "Beat count is the causal bottleneck" · "PPG lacks fine timing information" · "A beat
detector will solve PPG-to-ECG" · "Shape is independent of timing". Also never: a deployability, novelty,
SOTA, clinical or information-theoretic claim.

## 21. Figures

FIG 1 three-axis decomposition (schedule timing error vs GT-anchored derivative error and own-centre T6, for
ORACLE / J2 / J4 / J8) · FIG 2 the same for curvature and T7 · FIG 3 topology comparison (ORACLE, J8, MISS1,
EXTRA1) of own-centre T6 and T7 with 95 % CIs · FIG 4 coverage (S→G, P→S, P→S→G) per arm · FIG 5 R1 window
topology-category distribution · FIG 6 R1 set-correct timing bins, own-centre vs GT-anchored · FIG 7 site
secondary, topology-error fraction vs own-centre T6/T7. No cherry-picked waveform figure is required.

## 22. Tests

**Repository** firewall, pins, A4 md5, C2 untouched. **Frozen** B / O2c / R1 / operator hashes,
`requires_grad = False`, no optimizer reachable from any E1 source. **Cohort** the exact O3 2,048 cohort,
19,834 beats, 1,922 clusters, no test subject. **Identity** synthetic jitter preserves exact GT identity, the
MISS-deleted beat has no supplied event, the EXTRA-inserted beat has `gt_beat_id = NONE`, the R1 assignment
is deterministic and monotonic with tolerance exactly ± 150 ms and the frozen objective hierarchy.
**Chain** generated→supplied at exactly ± 50 ms with the frozen matcher, no GT correction of a schedule, no
event shift. **Morphology** own-centre uses the generated waveform's own event and the GT's own event, fixed
`[-10, +15]` support, exact O1 `d1`/`d2`/`p2p`/`width` primitives, no local shift optimisation, missing beats
excluded only with coverage reported. **Bootstrap** per-beat → window aggregation first, ECG-window cluster
unit, four site rows together, equal subject weighting, 2,000 replicates, seed 20260904. **R1 strata** exact
T0–T3 definitions, exact timing bins, minimum coverage thresholds, no post-hoc bin change. **Verdict** the
exact A/B/C/D tree and the rule that R1 observational strata cannot override the synthetic evidence.

## 23. Artifacts

`docs/E1_EVENT_PLACEMENT_MORPHOLOGY_DECOMPOSITION_PREREGISTRATION.md`,
`docs/E1_EVENT_PLACEMENT_MORPHOLOGY_DECOMPOSITION_REPORT.md`, and
`artifacts/e1_event_morphology_decomposition/`: `provenance.json`, `frozen_component_manifest.json`,
`cohort_manifest.csv`, `prediction_reuse_manifest.json`, `beat_identity_manifest.csv`,
`chain_matching_metrics.csv`, `coverage_metrics.csv`, `topology_metrics.csv`, `placement_metrics.csv`,
`own_center_beat_metrics.csv`, `own_center_window_metrics.csv`, `gt_anchored_local_metrics.csv`,
`alignment_sensitivity.csv`, `synthetic_contrasts.csv`, `topology_excess_damage.csv`,
`paired_bootstrap.csv`, `r1_topology_strata.csv`, `r1_timing_bins.csv`, `r1_stratum_metrics.csv`,
`r1_timing_controlled_comparison.csv`, `source_stability_reuse.csv`, `gates.csv`, `decision.json`,
`figures/`.

**No checkpoint, no training log and no new model output directory.** Artifacts, predictions and raw data are
never committed.

## 24. Commit order

1 repository integrity → 2 freeze O3 prediction/artifact availability → 3 write this preregistration →
**4 commit + push preregistration** → 5 implement identity chaining → 6 implement own-centre morphology →
7 implement the GT-anchored comparator → 8 implement the R1 strata → 9 tests → **10 implementation commit +
push** → 11 B / O2c / O3 regression reproduction (**fail ⇒ STOP**) → 12 build synthetic identities →
13 compute synthetic own-centre metrics → 14 freeze the placement contrasts → 15 freeze the topology
contrasts → 16 compute topology-excess damage → 17 **only then** compute the R1 diagnostic assignment →
18 R1 strata → 19 source-stability reuse → 20 freeze gates → 21 freeze the verdict → 22 figures → 23 report →
24 full tests → **25 result commit + push** → 26 clean-tree verification → 27 STOP. **NO TRAINING.**
