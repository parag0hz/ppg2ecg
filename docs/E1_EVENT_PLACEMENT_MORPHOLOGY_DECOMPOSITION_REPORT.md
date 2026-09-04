# E1 — Event Placement vs Beat-Morphology Decomposition — REPORT

**Should the next PPG-side module primarily improve beat-set / count correctness, or fine event timing?**

## 0. Status

**E1 was designed AFTER the O3 results were known.** Every threshold, stratum boundary and gate is a **frozen
post-hoc diagnostic criterion**, not independent preregistered confirmation. Preregistration
`docs/E1_EVENT_PLACEMENT_MORPHOLOGY_DECOMPOSITION_PREREGISTRATION.md`, pushed as **`2d5bb54`** before any E1
decomposition result; implementation **`f815478`**. **NO TRAINING, NO WEIGHT UPDATE, NO NEW PREDICTOR, NO
GENERATOR RETRAINING, NO NEW LOSS, NO TEST SUBJECT, NO C2, NO THRESHOLD TUNING** — every network was loaded
frozen in `eval()` with `requires_grad = False` and no optimizer was constructed (asserted by test).

## FINAL E1 VERDICT

**C. MIXED TOPOLOGY AND PLACEMENT LIMITATION**

The frozen tree returns C. **Read §6 before reading that letter**: gates C1–C5 (topology) all pass and the
topology effect dominates the timing effect by roughly 8.7× on own-centre T6, while the placement gates P1/P2
pass and P3/P4 fail for a *metric-degeneracy* reason that §6 sets out in full. A single sentence summary that
the data support without any ambiguity: **one missing or one extra beat damages own-centre QRS morphology far
more than 62.5 ms of timing jitter does, and timing jitter instead shows up in placement-sensitive
GT-anchored metrics.**

## 1. Repository

| item | value |
|---|---|
| start SHA | `d003bd7` (O3 result commit) |
| prereg SHA | **`2d5bb54`** |
| implementation SHA | **`f815478`** (+ two fixes, §2) |
| result SHA | this commit |
| clean? | yes — `git status` empty, `HEAD == origin/main` |
| test? | **no test subject** — `kjd` / `ssx` in no E1 source; `test_subjects_loaded: []` |
| C2? | **still deferred** |

Pins `6cd70cde` / `bf60cd7c` unchanged; A4 md5 `31c042d2…` unchanged. Full suite at the result commit:
**478 passed**, of which 24 are the E1 tests. Runtime 3.6 min, peak 1,763 MiB, no GPU training.

## 2. Prediction source

**DETERMINISTIC FROZEN RERUN.** O3 committed no per-window supplied schedule, generated waveform or detected
event list — predictions are never committed in this project — so the beat-level decomposition could not
reuse them. The eight primary arms were regenerated at NFE 4, source seed 0, on the exact O3 cohort with the
asserted frozen hashes, and two gates guarded the rerun:

- **Schedule-reconstruction gate** (tolerance 1e-9): all six synthetic arms reproduce the frozen O3
  `perturbation_manifest.csv` / `schedule_quality_metrics.csv` rows, and R1 reproduces
  `r1_schedule_quality.csv`. **PASS.**
- **Generator-reproduction gate** (tolerance 1e-6): B, every synthetic arm and the R1 arm reproduce their
  frozen O3 / O2c rows at **`max |Δ| = 0.00e+00`** — bit-exact. **PASS.**

Two implementation defects were found and fixed *before* any result was produced, and both are recorded here
because the second was caught by a preregistered safeguard rather than by inspection:

1. The reconstruction gate compared `max_abs_shift_samples`, which O3 leaves undefined (NaN) for MISS/EXTRA;
   the comparison treated NaN-vs-NaN as a mismatch and stopped the run. Fixed with a NaN-aware comparison,
   and at the same time the quantity is now **recomputed with O3's exact expression for every family**
   instead of being copied from O3 for the non-jitter families — the original check was vacuous there.
2. The chaining code passed a GT **beat index** where a GT **sample position** was required. Full-chain
   coverage collapsed to **0.013** and ORACLE own-centre T6 came out at 1.95 against O3's 0.52 — the
   preregistered §8 coverage requirement made the defect impossible to miss. Fixed, with a regression test
   pinning index-versus-position. The run that produced the wrong numbers was discarded; its verdict was
   `DECOMPOSITION INCONCLUSIVE`, produced by the coverage precondition doing exactly its job.

## 3. Three event sets

`G` = GT ECG R events · `S` = the schedule supplied to O2c · `P` = the events detected in the generated ECG
with the frozen detector. `S → G`, `P → S` and `P → G` are reported separately and never collapsed. B has no
external `S` and therefore appears only in the reproduction gate.

Identity: for the synthetic arms every supplied beat carries its exact originating GT beat (an inserted
EXTRA beat carries **none** and is excluded from every morphology comparison); for R1 a **frozen monotonic
one-to-one DP assignment at ± 150 ms** is used. That matcher was necessary: the frozen greedy matcher is
distance-first and can lose matches — on the constructed case `G = [100, 118]`, `S = [110, 128]` it finds one
match where the DP finds two. The assignment is **TARGET-DERIVED DIAGNOSTIC MATCHING** and never modifies,
corrects or re-times the supplied schedule.

## 4. Coverage

| arm | C1 `S→G` identity | C2 `P→S` adherence | C3 full chain `P→S→G` | C4 GT beats excluded | C5 generated excluded |
|---|---|---|---|---|---|
| ORACLE | 1.0000 | 0.9832 | **0.9753** | 0.0247 | 0.0231 |
| JITTER_2 | 1.0000 | 0.9868 | **0.9784** | 0.0216 | 0.0201 |
| JITTER_4 | 1.0000 | 0.9857 | **0.9770** | 0.0230 | 0.0211 |
| JITTER_8 | 1.0000 | 0.9792 | **0.9711** | 0.0289 | 0.0241 |
| MISS1 | 1.0000 | 0.9157 | **0.9098** | 0.1841 | 0.1228 |
| EXTRA1 | 0.9064 | 0.9659 | **0.8683** | 0.0420 | 0.0906 |
| R1-SCHEDULE | 0.8025 | 0.9327 | 0.7427 | 0.1775 | 0.1867 |

Every synthetic arm clears the frozen adequacy floor of 0.80, so the precondition holds. EXTRA1's C1 is 0.906
because one supplied beat per window is the inserted one and by construction has no GT identity. MISS1's C4
is 0.184 because the deleted beat's GT counterpart has no supplied event — that is the intended behaviour,
and it is exactly why coverage travels with every morphology number below.

## 5. Synthetic placement decomposition

Own-centre = generated window centred on its own detected event, GT window on the GT event. GT-anchored =
generated window centred on the **GT** event. Same beats, same support, same primitives.

| arm | schedule timing MAE | GT-anchored local deriv RMSE | own-centre local deriv RMSE | own-centre T6 | GT-anchored T6 | GT-anchored local curvature | own-centre local curvature | own-centre T7 | GT-anchored T7 |
|---|---|---|---|---|---|---|---|---|---|
| ORACLE | 0.00 ms | 0.1153 | 0.1103 | 0.4833 | 0.4833 | 0.0995 | 0.0962 | 0.4861 | 0.4864 |
| JITTER_2 | 9.36 ms | **0.2827** | 0.1105 | 0.4871 | 0.4871 | **0.2063** | 0.0961 | 0.4807 | 0.4807 |
| JITTER_4 | 17.41 ms | **0.3904** | 0.1111 | 0.4951 | 0.4952 | **0.2578** | 0.0962 | 0.4707 | 0.4687 |
| JITTER_8 | 33.04 ms | **0.3809** | 0.1147 | 0.5482 | 0.5482 | **0.2458** | 0.0982 | 0.4877 | 0.4912 |

Generator adherence is **HIGH ADHERENCE** everywhere (`P→S` F1@50 0.9840 / 0.9873 / 0.9863 / 0.9816), and the
generated-to-supplied timing error is 0.03–0.12 ms while the generated-to-GT error tracks the schedule error
almost exactly (0.12 / 9.74 / 17.42 / 32.83 ms). The generator reproduces the geometry it is handed to within
a small fraction of a sample.

### Alignment sensitivity (GT-anchored − own-centre) — the cleanest result in E1

| arm | T6 | T7 | local deriv RMSE | local curvature | local raw RMSE |
|---|---|---|---|---|---|
| ORACLE | −0.00005 | +0.00030 | +0.0049 | +0.0033 | +0.0046 |
| JITTER_2 | −0.00004 | +0.00001 | **+0.1723** | **+0.1102** | **+0.1485** |
| JITTER_4 | +0.00006 | −0.00204 | **+0.2793** | **+0.1616** | **+0.2802** |
| JITTER_8 | −0.00000 | +0.00347 | **+0.2662** | **+0.1477** | **+0.3613** |
| MISS1 | −0.00003 | −0.00025 | +0.0051 | +0.0034 | +0.0054 |
| EXTRA1 | −0.00011 | +0.00041 | +0.0034 | +0.0025 | +0.0031 |

The scalar O1 QRS primitives (T4/T6/T7/T8) have **essentially zero** alignment sensitivity at every jitter
level: re-centring by up to 8 samples leaves the R peak inside the ± 10-sample core, so p2p, max |d1|,
curvature energy and QRS width barely change. The **waveform** functionals are strongly alignment-sensitive
and scale with the jitter. Count errors, which introduce no timing error, show no alignment sensitivity at
all. The two axes are cleanly separable — and the scalar primitives are blind to one of them.

### Placement gates

| gate | requirement | result |
|---|---|---|
| **P1** | GT-anchored local derivative RMSE, J4 vs ORACLE, CI > 0 | **PASS** (+0.2752 [+0.2713, +0.2789]) |
| **P2** | GT-anchored local curvature error, J4 vs ORACLE, CI > 0 | **PASS** (+0.1582 [+0.1561, +0.1602]) |
| **P3** | own-centre T6 damage < **same-functional** GT-anchored T6 damage | **FAIL** (+0.0001 [−0.0002, +0.0004]) |
| **P4** | own-centre T7 damage < **same-functional** GT-anchored T7 damage | **FAIL** (−0.0023 [−0.0036, −0.0011]) |

## 6. Why P3/P4 fail, and the disclosure that goes with it

The task specification's §22 wording is *"own-center T6 damage is smaller than GT-anchored **derivative**
deterioration"*. That pairs a scalar-primitive quantity against a waveform quantity, which are on different
scales and are different functionals. The preregistration (§14) recorded that ambiguity explicitly and
resolved it the conservative way — **same functional, same units, same beats** — so P3 compares own-centre T6
damage against GT-anchored **T6** damage. §5 shows why that comparison is near-degenerate: GT-anchored T6 and
own-centre T6 differ by ~1e-4, so the contrast has essentially no discriminating power and P3/P4 cannot pass.

The task's literal reading is reported here as a clearly-labelled **post-hoc, non-decisional** secondary
contrast (`paired_bootstrap.csv`, rows prefixed `POSTHOC_`):

| J4 contrast | own-centre damage | GT-anchored **waveform** damage |
|---|---|---|
| T6 vs local derivative RMSE | **+0.0118** | **+0.2752** (23× larger) |
| T7 vs local curvature error | **−0.0153** | **+0.1582** |

Under that reading P3/P4 would pass and the frozen tree would return **A**. **It is not adopted.** The
preregistration was frozen before the numbers existed, and switching operationalisation after seeing them is
the exact failure mode the whole protocol exists to prevent. The verdict stands at **C**, and the reader now
has both numbers.

The methodological finding is worth stating in its own right: **the frozen O1 scalar QRS primitives cannot
express event placement.** Any future evaluation that wants to separate shape from placement must use
waveform-level functionals for the placement axis; the scalar primitives measure shape and only shape.

## 7. Synthetic topology decomposition

| arm | missing fraction | spurious fraction | own-centre T4 | T6 | T7 | T8 |
|---|---|---|---|---|---|---|
| ORACLE | 0.0000 | 0.0000 | 0.4653 | 0.4833 | 0.4861 | 0.5350 |
| JITTER_8 | 0.0000 | 0.0000 | 0.5367 | 0.5482 | 0.4877 | 0.5375 |
| **MISS1** | 0.1051 | 0.0000 | **1.0401** | **1.0451** | **0.7146** | **0.6901** |
| **EXTRA1** | 0.0000 | 0.1051 | **0.8320** | **0.8209** | **0.6009** | **0.6173** |

Topology classification is a clean sanity check: all four jitter arms are 2,048/2,048 **T0 EXACT EVENT SET**,
MISS1 is 2,048/2,048 **T2 UNDERCOUNT**, EXTRA1 is 2,048/2,048 **T3 OVERCOUNT**.

### Damage vs ORACLE (positive = the arm is worse; ECG-window clustered, subject-stratified, 2,000 replicates, `default_rng(20260904)`)

| contrast | T4 | T6 | T7 | T8 |
|---|---|---|---|---|
| JITTER_4 | +0.0175 [+0.0107, +0.0240] | +0.0118 [+0.0038, +0.0200] | −0.0153 [−0.0233, −0.0075] | −0.0059 [−0.0204, +0.0091] |
| **JITTER_8** | +0.0715 [+0.0613, +0.0813] | **+0.0648** [+0.0538, +0.0758] | +0.0016 [−0.0083, +0.0116] | +0.0025 [−0.0133, +0.0184] |
| **MISS1** | +0.5748 [+0.5462, +0.6046] | **+0.5617** [+0.5339, +0.5902] | **+0.2285** [+0.2075, +0.2501] | +0.1551 [+0.1300, +0.1806] |
| **EXTRA1** | +0.3670 [+0.3462, +0.3890] | **+0.3378** [+0.3177, +0.3593] | **+0.1147** [+0.0984, +0.1321] | +0.0830 [+0.0623, +0.1040] |

At J4 the own-centre T7 damage is *negative* — the arm's own-centre curvature error is slightly **lower** than
ORACLE's. That is reported as measured; it is one more sign that timing jitter does not damage shape.

### Topology excess damage — the main E1 contrast

`TopologyExcessDamage = Damage(count error, ORACLE) − Damage(JITTER_8, ORACLE)`; positive means one count
error damages own-centre morphology more than the most severe tested timing jitter.

| contrast | T6 | T7 |
|---|---|---|
| **MISS1 vs JITTER_8** | **+0.4969** [+0.4712, +0.5222] | **+0.2269** [+0.2082, +0.2465] |
| **EXTRA1 vs JITTER_8** | **+0.2730** [+0.2559, +0.2908] | **+0.1132** [+0.0989, +0.1280] |

One missing beat damages own-centre T6 **8.7×** more than ± 8 samples (62.5 ms) of jitter (0.5617 vs 0.0648);
one extra beat **5.2×** more.

### Topology gates

| gate | requirement | result |
|---|---|---|
| **C1** | MISS1 own-centre T6 worse than ORACLE | **PASS** (+0.5617) |
| **C2** | MISS1 own-centre T7 worse than ORACLE | **PASS** (+0.2285) |
| **C3** | EXTRA1 T6 or T7 worse than ORACLE | **PASS** (both) |
| **C4** | TopologyExcessDamage_MISS > 0, CI > 0, both T6 and T7 | **PASS** |
| **C5** | TopologyExcessDamage_EXTRA > 0, CI > 0, at least one | **PASS** (both) |

**All five topology gates pass.**

## 8. Generator adherence

| arm | `P→S` F1@50 | label |
|---|---|---|
| ORACLE | 0.9840 | HIGH ADHERENCE |
| JITTER_2 / _4 / _8 | 0.9873 / 0.9863 / 0.9816 | HIGH ADHERENCE |
| MISS1 | 0.8917 | — |
| EXTRA1 | 0.9473 | HIGH ADHERENCE |
| R1-SCHEDULE | 0.9357 | HIGH ADHERENCE |

The label is descriptive and entered no gate. Since R1's `P→S` stays ≥ 0.90 while its `S→G` quality is poor,
the permitted sentence applies: *the frozen generator generally follows the supplied geometry; schedule error
remains upstream.* No causal statement is made.

## 9. R1 topology

| category | rows | unique ECG windows | an0 | k2s | sufficient | own-centre T6 | own-centre T7 | schedule MAE |
|---|---|---|---|---|---|---|---|---|
| **T0 exact event set** | 1,017 (49.7 %) | 981 | 481 | 500 | yes | **0.4571** | **0.4509** | 28.60 ms |
| T1 count correct, set imperfect | 132 (6.4 %) | 130 | 76 | 54 | yes | 0.7291 | 0.6348 | 69.89 ms |
| T2 undercount | 73 (3.6 %) | 72 | 53 | 19 | **no** | 0.5088 | 0.4842 | 45.68 ms |
| **T3 overcount** | 826 (40.3 %) | 800 | 369 | 431 | yes | **1.1182** | **0.7480** | 58.37 ms |

R1's dominant failure mode is **over-detection** (40.3 % of windows), and the undercount stratum does not meet
the frozen minimum (53 / 19 unique windows per subject against a floor of 30), so it is labelled
**INSUFFICIENT STRATUM COVERAGE** and is not interpreted separately.

Pooled: **R1-SET-CORRECT** own T6 **0.4571**, essentially the ORACLE level of 0.4833, against
**R1-TOPOLOGY-WRONG** own T6 **1.0385** — a 2.3× gap. Their F1 excess differs correspondingly
(+0.7099 vs +0.2541).

## 10. R1 set-correct timing bins

| bin | range | rows | unique windows | own-centre T6 | own-centre T7 | GT-anchored deriv RMSE | GT-anchored curvature | F1 excess |
|---|---|---|---|---|---|---|---|---|
| **A** | ≤ 16 ms | 261 | 260 | 0.4597 | 0.4582 | 0.2968 | 0.2138 | **+0.8712** |
| **B** | > 16, ≤ 32 ms | 424 | 414 | 0.4607 | 0.4678 | 0.4021 | 0.2614 | +0.8146 |
| **C** | > 32 ms | 332 | 326 | 0.4428 | 0.4235 | 0.3866 | 0.2506 | **+0.4567** |

All three bins meet the frozen minimum coverage. Within event-set-correct windows, **own-centre shape is flat
across a 2× span of timing error** (0.4597 / 0.4607 / 0.4428 — bin C is if anything the *best*), while the
placement-sensitive GT-anchored derivative error rises and event fidelity collapses from +0.87 to +0.46. This
is the R1-side echo of §5, on real PPG-derived schedules rather than synthetic jitter.

## 11. R1 topology-wrong, within matched timing strata

| bin | n topology-wrong / set-correct | unique windows | powered | own T6 wrong vs correct | own T7 wrong vs correct |
|---|---|---|---|---|---|
| A ≤ 16 ms | 41 / 261 | 41 / 260 | **no** | 0.5309 vs 0.4597 | 0.5497 vs 0.4582 |
| B 16–32 ms | 139 / 424 | 136 / 414 | yes | **0.6508 vs 0.4607** | 0.5662 vs 0.4678 |
| C > 32 ms | 846 / 332 | 823 / 326 | yes | **1.1288 vs 0.4428** | 0.7570 vs 0.4235 |

*Within coarse matched-timing strata*, windows whose event set is wrong carry substantially worse own-centre
morphology than windows whose event set is correct, in both powered bins. **This is observational** — R1's
topology errors are not randomly assigned, and windows where R1 miscounts are plausibly harder in other ways.
The synthetic MISS/EXTRA arms remain the cleaner intervention, and bin A is under-powered by the frozen rule.

Site secondary (no site causality claim): topology-error fraction 0.365 (head), 0.458 (sternum), 0.592
(wrist), 0.595 (ankle); own-centre T6 0.611 / 0.686 / 0.928 / 0.782 — the ordering of morphology damage
follows the ordering of topology error, not the ordering of timing error.

## 12. Source-stability reuse (frozen O3 numbers, not rerun)

| arm | beat-count SD | pairwise event F1@50 | pointwise waveform SD |
|---|---|---|---|
| ORACLE | 0.2118 | 0.9748 | 0.2256 |
| JITTER_8 | 0.2969 | 0.9720 | 0.2403 |
| MISS1 | 1.0167 | 0.8167 | 0.3005 |
| EXTRA1 | 0.7333 | 0.9216 | 0.2723 |
| R1-SCHEDULE | 0.5185 | 0.9116 | 0.2435 |

Source-driven event instability associates with **count** corruption, not with timing jitter: 62.5 ms of
jitter leaves stability at oracle level while one missing beat nearly restores the baseline's instability.
Descriptive; no causal model was fitted.

## 13. Interpretation — seven things kept strictly separate

1. **Event-set correctness.** The dominant factor in every own-centre morphology comparison. One missing beat
   costs 8.7× what 62.5 ms of jitter costs; one extra beat 5.2×. All five topology gates pass.
2. **Event placement.** Real and large, but it lands on *placement-sensitive* metrics: GT-anchored local
   derivative RMSE rises from 0.1153 to 0.3904 between ORACLE and J4, with alignment sensitivity growing from
   +0.005 to +0.279, while own-centre shape moves by ~0.001.
3. **Generator adherence.** High everywhere (0.89–0.99); generated-to-supplied timing error is a fraction of a
   sample. The generator draws what it is told to draw, where it is told to draw it.
4. **Own-centre morphology.** Nearly invariant to timing, strongly damaged by count errors.
5. **GT-anchored morphology.** Strongly damaged by timing, barely moved by count errors' own timing (they
   introduce none) — it is the placement channel.
6. **Source stochasticity.** Tracks count accuracy, not timing accuracy.
7. **R1 observational evidence.** Corroborates the synthetic picture — set-correct windows sit at ORACLE-level
   own-centre shape regardless of timing bin, topology-wrong windows are 2.3× worse — but it is observational
   and by preregistration it **cannot** override the synthetic contrasts, and it did not: the verdict is
   selected by the synthetic contrasts alone.

## 14. What this does NOT prove

- **Post-O3 diagnostic**, designed after the results were known; frozen post-hoc criteria, not confirmation.
- Development cohort only, **two validation subjects**, **no fresh test**, **no new training**.
- R1 beat identity comes from **target-derived diagnostic matching**; a different assignment rule could shift
  the R1 strata (the synthetic arms, which carry the verdict, use exact construction identity instead).
- Own-centre metrics **do not penalise missing beats**; that is their point, and it is why coverage is
  reported on every row. A method that emitted fewer, easier beats could look good on them alone.
- **No causal proof.** "Beat count is the causal bottleneck" is not a licensed statement, and neither is
  "PPG lacks fine timing information", "a beat detector will solve PPG-to-ECG", or "shape is independent of
  timing" — §5 shows shape and timing are separable *under these metrics*, not independent in general.
- No information-theoretic conclusion, no clinical claim, no deployability, novelty or SOTA claim.
- The R1 timing-controlled comparison is observational and its ≤ 16 ms bin is under-powered by the frozen rule.

## 15. Recommended next experiment (recommendation only — nothing implemented)

The frozen verdict is **C**, whose preregistered recommendation is a **joint set + timing representation**.
The measurements sharpen what "joint" has to mean, and the two components are not symmetric:

1. **Event existence is the larger lever for morphology.** Under these metrics a count error costs 5–9× what
   severe jitter costs, R1's own set-correct windows already reach ORACLE-level own-centre shape at 28.6 ms
   mean timing error, and R1's dominant error is over-detection (40.3 % of windows). Any next schedule module
   should be specified and evaluated on **beat-set correctness first**, with over-detection treated as the
   primary error mode.
2. **Timing still has to be modelled, because it controls the deployable metric.** O3's binding constraint was
   G5, a GT-anchored metric, and §10 shows event fidelity falling from +0.87 to +0.46 across timing bins even
   when the event set is exactly right. Set correctness alone would not have passed O3's gate.
3. **The evaluation itself needs fixing before the next model.** §6 shows the frozen O1 scalar primitives are
   blind to placement. A joint representation cannot be assessed with metrics that measure only one of the two
   axes, so the criterion — a placement-tolerant shape term *and* a placement term — must be agreed **before**
   any such model is trained.

Nothing here is implemented, no predictor was built, and C2 remains deferred.

## Artifacts

`artifacts/e1_event_morphology_decomposition/`: `provenance.json`, `frozen_component_manifest.json`,
`cohort_manifest.csv`, `prediction_reuse_manifest.json`, `beat_identity_manifest.csv`,
`chain_matching_metrics.csv`, `coverage_metrics.csv`, `topology_metrics.csv`, `placement_metrics.csv`,
`own_center_beat_metrics.csv`, `own_center_window_metrics.csv`, `gt_anchored_local_metrics.csv`,
`alignment_sensitivity.csv`, `synthetic_contrasts.csv`, `topology_excess_damage.csv`,
`paired_bootstrap.csv`, `r1_topology_strata.csv`, `r1_timing_bins.csv`, `r1_stratum_metrics.csv`,
`r1_timing_controlled_comparison.csv`, `r1_site_topology.csv`, `source_stability_reuse.csv`, `gates.csv`,
`decision.json`, `figures/` (FIG 1–7). **No checkpoint, no training log, no new model output directory.**

Code: `src/ppg2ecg/evaluation/e1_decompose.py`, `scripts/e1_decompose.py`, `scripts/e1_figures.py`,
`tests/test_e1_decomposition.py` (24 tests).
