# E3 — Beat-Set-First PPG Event Geometry — REPORT

**Can correcting or predicting the NUMBER of supplied events reduce R1's dominant overcount/spurious failure
without sacrificing recall, placement, or downstream morphology?**

## FINAL E3 VERDICT

**E. COUNT-ONLY CEILING NOT SUPPORTED** — gate **OC4** fails.

Supplying the **correct beat count** while keeping the frozen R1 timing scores removes overcount and
undercount **entirely** (T3 0.4033 → 0.0000, T2 0.0356 → 0.0000) and lifts the exact-set fraction from
0.4966 to 0.6255, but it makes **missing worse by 0.0332** (0.1062 → 0.1394), against a preregistered
non-inferiority margin of 0.020, with the CI entirely on the wrong side. The count-only direction is
therefore killed at its cheapest test: **the ridge readout was never fitted, the train-only threshold
control was never run, and no schedule was ever sent through O2c.**

Preregistration **`20f890b`** was pushed before any E3 number existed; implementation **`c8b1eee`**.
**NO TRAINING happened anywhere in E3** — the one permitted learned object was never reached.

## 1. Repository

| item | value |
|---|---|
| start SHA | `153bba6` (E2 result commit) |
| prereg SHA | **`20f890b`** |
| implementation SHA | **`c8b1eee`** |
| result SHA | this commit |
| clean? | yes — `git status` empty, `HEAD == origin/main` |
| test? | **no test subject** — `kjd` / `ssx` in no E3 source; `test_subjects_loaded: []` |
| C2? | **still deferred** |

Pins `6cd70cde` / `bf60cd7c` unchanged; A4 md5 `31c042d2…` unchanged. Full suite at the result commit:
**526 passed**, of which 16 are the E3 tests.

## 2. E2 contract

| item | value |
|---|---|
| `contract_v1.json` sha256 | `06e869412114e1efb9ab6624540aebc141495e2690150f4e512df7676c5a9115` |
| git blob | `d52b8953…` |
| version | `e2-event-geometry-contract-v1` |
| source E1 SHA | `a5af4af` |
| **modified?** | **NO** |

E3 imports `ppg2ecg.evaluation.event_geometry_contract`; a test asserts that no E3 source defines
`classify_topology`, `placement_metrics`, `assign_schedule_to_gt`, `joint_event_fidelity`,
`own_center_morphology` or `gt_anchored_joint_structure` locally, and the run stops if the contract's
sha256 has moved. Matching (± 150 ms `S → G`, ± 50 ms `P → S`), the `[-10, +15]` support, the coverage
requirements, the aggregation order and the bootstrap are all the frozen E2 ones.

## 3. Candidate extraction

| item | value |
|---|---|
| exact R1 reproduction? | **YES — bit-exact on 2,048 / 2,048 windows** |
| candidate shortage fraction | **0.0000 %** (threshold 0.5 %) |
| candidate coverage @150 ms | **0.9590** |
| mean candidates vs mean GT | 20.50 vs 9.68 (median surplus +11, minimum 12 candidates) |

The threshold-free extractor is `rhythm_tcn.extract_events` with **only** the amplitude-filter line
removed. Filtering its output at `score >= 0.35` reproduces the frozen R1 event list exactly, and this is
structural rather than incidental: the NMS is greedy by **descending** score, so a sub-threshold peak is
processed after every supra-threshold one and can never suppress it, and the stable sort preserves their
relative order. The refractory stays at 32 samples, no peak is moved, and the score is the exact frozen
`sigmoid(RhythmTCN(ppg))` with no recalibration, smoothing or per-window normalisation.

Capacity is ample — there is no window where the candidates cannot supply K events — so the oracle-count
ceiling below is a **fair** ceiling and its failure cannot be blamed on missing candidates. The 4.1 % of GT
beats with **no** candidate within ± 150 ms is the part of the ceiling that no selection rule can recover.

## 4. Stage 0 — oracle-count ceiling

`K_oracle` = the GT beat count. **GT supplied the count only**: every event location came from the frozen R1
candidate scores by top-K selection, and a test asserts that no GT location is ever read in the selection
path. This is an **ORACLE COUNT DIAGNOSTIC and is not deployable.**

| arm | exact-set | T1 | T2 | T3 | missing | spurious | B5 T0-only timing | n T0 | SG F1@50 |
|---|---|---|---|---|---|---|---|---|---|
| **R1-0.35** | 0.4966 | 132 | 0.0356 | 0.4033 | **0.1062** | 0.2157 | 28.63 ms | 1017 | 0.6157 |
| **ORACLE-COUNT-R1** | **0.6255** | **767** | **0.0000** | **0.0000** | **0.1394** | **0.1394** | 30.50 ms | 1281 | 0.6190 |

Full A/B/joint-event block for both arms: A1 absolute count error 1.1372 → **0.0000**; A2 count deviation
0.1183 → **0.0000**; B1 median AE 41.65 → 41.52 ms; B2 MAE 43.82 → 43.81 ms; B3 p90 70.93 → 69.99 ms;
B4 p95 77.23 → 76.07 ms; B6 T0-only p90 47.33 → 50.98 ms; SG F1@100 0.7788 → 0.7813; SG F1@150
0.8598 → 0.8606; SG F1@200 0.8973 → 0.8996; SG precision 0.6041 → 0.6190; SG recall 0.6316 → 0.6190.

**Blocks C (`P → G`), D, E and F were never computed for ORACLE-COUNT**, because the preregistration runs
the generator only after OC1–OC6 pass and OC4 failed. That is the designed cheap stop, not an omitted axis.

### Paired effects (frozen E2 bootstrap; positive = ORACLE-COUNT better)

| metric | R1 | ORACLE-COUNT | effect [95 % CI] |
|---|---|---|---|
| A5 exact-set | 0.4966 | 0.6255 | **+0.1289** [+0.1143, +0.1435] |
| T3 overcount | 0.4033 | 0.0000 | **+0.4033** [+0.3820, +0.4251] |
| T2 undercount | 0.0356 | 0.0000 | **+0.0356** [+0.0278, +0.0442] |
| A1 abs count error | 1.1372 | 0.0000 | **+1.1372** [+1.0593, +1.2222] |
| A4 spurious | 0.2157 | 0.1394 | **+0.0764** [+0.0702, +0.0832] |
| **A3 missing** | **0.1062** | **0.1394** | **−0.0332** [−0.0371, −0.0296] |
| B5 T0-only timing MAE | 28.63 ms | 30.50 ms | **+0.0000** [+0.0000, +0.0000] (1,031 windows dropped) |
| B2 timing MAE | 43.82 ms | 43.81 ms | −0.0610 [−0.3137, +0.2106] |
| SG F1@50 | 0.6157 | 0.6190 | +0.0034 [+0.0017, +0.0050] |

### OC / OG gates

| gate | requirement | result |
|---|---|---|
| **OC1** | exact-set CI > 0 and point ≥ +0.10 | **PASS** (+0.1289) |
| **OC2** | T3 decreases ≥ 0.15, CI favourable | **PASS** (+0.4033) |
| **OC3** | spurious decreases ≥ 0.05, CI favourable | **PASS** (+0.0764) |
| **OC4** | missing not worse by more than 0.020 | **FAIL** (−0.0332 [−0.0371, −0.0296]) |
| **OC5** | B5 T0-only MAE not worse by more than 8.0 ms | **PASS** (+0.0000) |
| **OC6** | `S → G` F1@50 CI favourable | **PASS** (+0.0034) |
| **OG1–OG6** | — | **not evaluated** — the generator stage is gated on OC and was never run |

## 5. Why it fails, precisely

**Forcing `M == K` makes precision and recall identical by construction.** With the count exactly right,
every unmatched supplied event is paired with an unmatched GT event, so `missing == spurious` — visible in
the table as SG precision = SG recall = 0.6190 and A3 = A4 = 0.1394 for ORACLE-COUNT.

R1 at threshold 0.35 emits about 11.8 % more events than GT (A2 = 0.1183), and **that surplus buys recall**:
its missing rate (0.1062) is less than half its spurious rate (0.2157). Constraining the count trades the
surplus away — spurious falls by 0.076, missing rises by 0.033 — and the two meet at 0.1394. The exact-set
fraction still improves by +0.129, but 767 windows (37.5 %) land in **T1: the count is right and the set is
still wrong.** So the correct count converts an *asymmetric* overcount error into a *symmetric* missing /
spurious error at a rate the ranking cannot beat.

A structural check confirms the mechanism and validates the implementation: on the 1,017 windows where R1
already emits exactly `K` events, top-K selection returns the **identical** set — a candidate below 0.35
cannot outrank one above it, so when exactly `K` candidates clear the threshold they *are* the top K. That
is why the paired B5 effect on the T0 intersection is exactly `+0.0000` with a zero-width CI. The two arms
differ only where R1's count is wrong, and the arm-level B5 gap (28.63 vs 30.50 ms) comes from
ORACLE-COUNT's T0 set being larger (1,281 vs 1,017 windows) and therefore including harder windows — the
paired and marginal readings of a T0-only metric are different quantities, exactly as the E2 contract warns.

The binding constraint is **which** candidates rank highest, not how many are emitted. The 4.1 % of GT beats
with no candidate within ± 150 ms puts a further hard floor under any selection rule.

## 6. Site-wise secondary (no site causality claim)

| site | arm | exact-set | T3 | T2 | missing | spurious |
|---|---|---|---|---|---|---|
| sternum | R1-0.35 | 0.5400 | 0.3588 | 0.0413 | 0.0973 | 0.1774 |
| sternum | ORACLE-COUNT | 0.6755 | 0.0000 | 0.0000 | **0.1184** | 0.1184 |
| head | R1-0.35 | 0.6341 | 0.2900 | 0.0328 | 0.0489 | 0.1117 |
| head | ORACLE-COUNT | 0.7831 | 0.0000 | 0.0000 | **0.0639** | 0.0639 |
| wrist | R1-0.35 | 0.4026 | 0.4899 | 0.0273 | 0.1414 | 0.3205 |
| wrist | ORACLE-COUNT | 0.4902 | 0.0000 | 0.0000 | **0.2015** | 0.2015 |
| ankle | R1-0.35 | 0.4046 | 0.4809 | 0.0400 | 0.1386 | 0.2608 |
| ankle | ORACLE-COUNT | 0.5458 | 0.0000 | 0.0000 | **0.1774** | 0.1774 |

The missing-rate penalty appears at **every** site, and it is largest exactly where R1 is weakest (wrist
+0.060, ankle +0.039) — the sites whose surplus events were doing the most recall work.

## 7. Stages that never ran

| stage | status |
|---|---|
| train-only threshold control | **not run** — gated on the Stage-0 ceiling |
| train12 feature extraction | **not run** |
| ridge count readout | **NEVER FITTED** — `ridge_fitted: false` in `decision.json` |
| validation count predictions | **not produced** |
| `E3-RIDGE-COUNT` schedule arm | **does not exist** |
| any O2c generator run | **not executed** |
| PC1–PC7, TC1–TC3, DG1–DG7, DC1–DC3 | **not evaluated** |

FIG 4 (count prediction) and FIG 5/6 (downstream) are correspondingly absent; the figures manifest records
which stages ran. Nothing was computed and then withheld.

## 8. Interpretation — nine things kept separate

1. **R1 candidate quality.** Ample capacity (20.5 candidates for 9.7 beats, zero shortage) but 4.1 % of GT
   beats have no candidate within ± 150 ms at all — a hard floor for any selection rule.
2. **Oracle count ceiling.** Removes T2/T3 completely and adds +0.129 exact-set, but fails the missing gate.
3. **PPG count prediction.** Never attempted; the ceiling did not license it.
4. **Threshold control.** Never run; it is only meaningful once a ceiling exists to compare against.
5. **Event-set correctness.** Improved but far from solved: 37.5 % of windows are T1 — right count, wrong set.
6. **Placement.** Essentially unchanged (paired T0-intersection effect exactly zero; B2 −0.06 ms, CI spanning
   zero). Count constraints do not move timing, exactly as E1/E2 predicted.
7. **Generator adherence.** Not measured for the new arm — no generator was run.
8. **Own-centre morphology.** Not measured for the new arm.
9. **GT-anchored joint structure.** Not measured for the new arm.

**The hypothesis that motivated E3 — that getting the count right first would fix much of the failure — is
falsified at its cheapest test, in the specific sense that the count is not the binding constraint: ranking
is.** This says nothing about whether timing matters; timing remains a separate E2 axis and was not the
subject of this test.

## 9. What this does NOT prove

- Development cohort only, **two validation subjects**, **no fresh test**, one frozen backbone.
- **No training happened**, so nothing here bounds what a *trained* event-existence model could do — only
  what count information alone buys **given this frozen ranking**.
- The ceiling was measured with **one** selection rule (top-K by score). A different, still count-only rule
  is not excluded by these numbers; what is excluded is that the count alone, with the frozen R1 ranking,
  clears the preregistered bar.
- `O2c` was itself trained with GT-R canonical coordinates, and R1 was supervised with ECG R labels.
- **No causal proof.** "Beat count solves PPG-to-ECG", "count is the causal bottleneck" and "timing is
  unimportant" are all unlicensed — and so is the converse claim that count never matters, since the count
  constraint did remove T2/T3 entirely and did improve the exact-set fraction.
- No deployability, novelty or SOTA claim. The oracle-count arm is not deployable by construction.

## 10. Recommended next step (recommendation only — nothing implemented)

Per the frozen tree, verdict E means: **kill the count-only direction.** Do not fit the ridge readout, do not
build a larger count model, and do not modify O2c.

What the measurement actually points at, as questions rather than builds:

1. **The binding constraint is candidate ranking, not candidate count.** 37.5 % of windows have the right
   count and the wrong set, and the surplus events R1 emits are doing real recall work that a count
   constraint destroys. Any future event-geometry work should be specified against **ranking / scoring
   quality at a fixed count**, which the E2 contract already measures (A5 with T2 = T3 = 0 isolates it
   exactly).
2. **4.1 % of GT beats have no candidate at all.** That fraction is invisible to every selection rule and
   bounds the achievable ceiling; a future experiment should measure it first and decide whether the
   detector front-end, not the selector, is what needs changing.

Both are recommendations for a **new preregistration**. Nothing was implemented: no Transformer, no phase
network, no set decoder, no count head, no new detector, and C2 remains deferred.

## Artifacts

`artifacts/e3_beat_set_first/`: `provenance_stage0.json`, `frozen_component_manifest.json`,
`e2_contract_identity.json`, `candidate_extraction_audit.json`, `candidate_capacity.csv`,
`candidate_capacity_summary.json`, `oracle_count_schedule_metrics.csv`, `oracle_count_bootstrap.csv`,
`stage0_gates.json`, `site_metrics.csv`, `gates.csv`, `decision.json`, `figures/` (FIG 1–3).
Artifacts for stages that never ran do not exist. **No checkpoint, no training log, no predictor output, no
new model directory, and no raw signal or feature matrix is committed.**

Code: `src/ppg2ecg/evaluation/e3_beat_set.py`, `scripts/e3_common.py`, `scripts/e3_stage0.py`,
`scripts/e3_figures.py`, `tests/test_e3_beat_set.py` (16 tests).
