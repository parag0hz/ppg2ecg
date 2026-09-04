# E2 — Event-Set / Placement / Morphology Evaluation Contract — PREREGISTRATION

**Freeze the measurement before building the next PPG predictor.**

## 0. Status and claim boundary

E2 is **measurement-contract construction after E1**. It is **not** independent confirmation, not a new
scientific result by itself, not a deployable model, not a claim about PPG observability, not causal
evidence, not information-theoretic evidence, and not novelty or SOTA evidence.

**NO TRAINING. NO WEIGHT UPDATE. NO NEW PPG PREDICTOR. NO GENERATOR RETRAINING. NO THRESHOLD TUNING.
NO NEW ATTENTION. NO NEW LOSS. NO TEST SUBJECT. NO C2.** Every network is loaded frozen, in `eval()`, with
`requires_grad = False`; no optimizer is constructed anywhere in E2.

**The E1 formal verdict remains `MIXED TOPOLOGY AND PLACEMENT LIMITATION`.** E2 must not relabel E1 as
verdict A, must not reinterpret E1's gates post hoc, must not change the E1 report, and must not replace the
frozen E1 verdict. E2 fixes the metric ambiguity E1 exposed **for future experiments only**.

### Why a contract is needed (established in E1, restated, not re-derived)

One missing or extra beat damages own-centre QRS morphology far more than severe timing jitter; timing
jitter barely moves own-centre morphology but strongly moves GT-anchored waveform metrics; the frozen O1
scalar targets T4/T6/T7/T8 are almost invariant to event re-centring and therefore cannot measure placement;
and GT-anchored derivative/curvature mix shape with placement by construction. No single "structure metric"
can adjudicate all of this.

## 1. Frozen taxonomy — four families that are never merged

| family | question | label to use |
|---|---|---|
| **AXIS A** | did the schedule supply the correct number/set of events? | **EVENT SET / TOPOLOGY** |
| **AXIS B** | for events that correspond, how accurately are they placed? | **EVENT PLACEMENT** |
| **AXIS C** | ignoring global placement, is the corresponding beat's shape right? | **OWN-CENTRE BEAT MORPHOLOGY** |
| **JOINT** | is the right structure at the right GT event? | **GT-ANCHORED JOINT STRUCTURE** |

Two auxiliary families are reported but never merged into the four: **JOINT EVENT FIDELITY** (F1-family) and
**GENERATOR ADHERENCE** (`P → S`).

**Naming rules, frozen.** The GT-anchored family is **never** called "pure morphology" or "morphology-only".
The F1 family is **never** called "pure timing accuracy". T4/T6/T7/T8 are **own-centre / local shape summary
metrics** and are never presented as placement evidence.

## 2. Repository and frozen components

`HEAD == origin/main == a5af4af`, clean tree, PENGUIN `6cd70cde`, iMeanFlow `bf60cd7c`, A4 md5
`31c042d291052fbb6dc15263ad316be2`, C2 deferred, `kjd` / `ssx` never loaded. Frozen model identities are the
E1/O3 ones: B `557c7054…` / `47d7ccb9…`, O2c `5aab09be…` / `f1cc44b3…`, R1 `bfe76ea6…` / `0986a7af…`,
operator `cb4d1866…` / `046becfb…`.

## 3. Source of E2 data

E1 artifacts are the first source and their sha256 hashes are frozen in `source_artifact_manifest.json`.
E1's per-window and per-beat tables supply own-centre and GT-anchored morphology, alignment sensitivity and
chain coverage directly.

Three contract quantities **cannot** be reconstructed from E1 artifacts and therefore require a rerun:

1. **per-window topology class and per-window schedule placement** — E1 wrote only per-arm aggregates
   (these need only `G` and `S`, so they involve **no generator inference**);
2. **joint event fidelity `P → G` at 100 / 150 / 200 ms** — E1 and O3 wrote `P → G` only at 50 ms;
3. **the full detected event list `P`** — E1's beat table holds only successfully chained beats, so
   `P → S` adherence over *all* matched pairs is not recoverable (it under-covers EXTRA1 and R1).

E2 therefore performs a **deterministic frozen rerun of exactly the five validation arms** — ORACLE,
JITTER_8 rep0, MISS1 rep0, EXTRA1 rep0, R1-SCHEDULE — at **NFE 4**, **source seed 0**, on the exact E1/O3
cohort (2,048 windows, 19,834 GT beats, 1,922 ECG-window clusters) with the asserted frozen hashes.
**No training under any circumstance.**

**Reproduction gate, before any interpretation:** every rerun arm must reproduce its E1 per-arm rows —
own-centre T4/T6/T7/T8, own-centre and GT-anchored local raw RMSE / derivative RMSE / curvature error, and
the chain coverages C1/C2/C3 — at **`max |Δ| ≤ 1e-6`**. Any failure ⇒ report, **STOP**.

## 4. Three event sets — never collapsed

`G` = GT ECG R events · `S` = supplied event schedule · `P` = events detected in the generated ECG with the
frozen detector. Reported separately, always: **`S → G` schedule quality**, **`P → S` generator adherence to
the supplied geometry**, **`P → G` final generated event fidelity**.

## 5. Matching contract

- **`S → G`**: for synthetic arms the exact construction identity from O3/E1; for R1 and any future
  predicted schedule the frozen **monotonic one-to-one dynamic-programming assignment** at **± 150 ms**,
  with the objective hierarchy (1) maximise matched cardinality, (2) minimise total absolute timing error,
  (3) deterministic tie break by transition preference `match → skip-ref → skip-pred` scanning indices
  ascending. This is **TARGET-DERIVED DIAGNOSTIC MATCHING**: it may never shift, repair, insert into, delete
  from or otherwise modify `S`, and never touches inference.
- **`P → S`**: the exact frozen greedy one-to-one matcher at **± 50 ms** (`rpeaks.match_rpeaks`), unchanged.
- **`P → G`**: the same frozen matcher, reported at 50 / 100 / 150 / 200 ms.

## 6. AXIS A — event set / topology (primary)

| id | metric |
|---|---|
| **A1** | absolute beat-count error `|M − K|` |
| **A2** | beat-count deviation `|M/K − 1|` |
| **A3** | missing fraction (unmatched GT events / `K`, at ± 150 ms) |
| **A4** | spurious fraction (unmatched supplied events / `K`, at ± 150 ms) |
| **A5** | exact-event-set window fraction (share of windows in T0) |
| **A6** | window topology category T0 / T1 / T2 / T3 |

**T0** `M == K` **and** every supplied event assigned to a unique GT event within ± 150 ms **and** no
unmatched GT **and** no unmatched supplied event · **T1** `M == K` but the set is imperfect · **T2**
`M < K` undercount · **T3** `M > K` overcount.

**F1@50 is NOT the primary event-set metric**, because it mixes topology with placement. F1 metrics belong
to §8 joint event fidelity.

## 7. AXIS B — event placement (primary)

Measured on `S → G` identity-matched pairs only, with coverage always attached:
**B1** matched timing median AE · **B2** matched timing MAE · **B3** matched timing p90 AE · **B4** matched
timing p95 AE. Plus, computed on **T0 windows only** so that timing is isolated from count error:
**B5** exact-set placement MAE · **B6** exact-set placement p90 AE. The T0 sample size is always reported.

## 8. Joint event fidelity (reported, never called timing accuracy)

F1@50, F1@100, F1@150, F1@200, precision and recall for **`S → G`** and for **`P → G`**. These jointly
reflect existence, count and placement, and carry the label **JOINT EVENT FIDELITY**.

## 9. Generator adherence

`P → S`: F1@50, F1@100, missing, spurious, matched timing MAE. Frozen descriptive label **HIGH ADHERENCE**
when `P → S` F1@50 ≥ 0.90. The label is descriptive only and never by itself determines a future method's
success.

## 10. AXIS C — own-centre beat morphology (primary)

For every valid chain `P → S → G`: the generated beat window is centred on its own detected event `P`, the
GT beat window on `G`, support **`[-10, +15]` samples**, eligibility `c − 11 ≥ 0` and `c + 15 ≤ 1023` for
each own centre. **No local cross-correlation shift, no DTW, no oracle shift, no amplitude optimisation, no
waveform renormalisation.**

Primary: **C1** own-centre T4 p2p nAE · **C2** T6 max-|derivative| nAE · **C3** T7 curvature-energy nAE ·
**C4** T8 QRS-width nAE, each the exact O1 per-beat primitive normalised by the frozen O1 train IQRs
(T4 0.50532, T6 0.22995, T7 0.03380, T8 31.25).
Secondary: **C5** local raw RMSE · **C6** local derivative RMSE · **C7** local curvature error ·
**C8** local correlation.

## 11. Morphology coverage — mandatory

Every morphology row carries **`S → G` identity coverage**, **`P → S` adherence coverage**, **full
`P → S → G` chain coverage**, **fraction of GT beats excluded** and **fraction of generated beats excluded**.
**Morphology reported without coverage is invalid.** Own-centre morphology does not penalise missing or
unmatched extra beats — those penalties live in AXIS A and are never silently combined with AXIS C.

## 12. JOINT — GT-anchored joint structure

The same matched beats, but the generated window is centred on **`G`**, support `[-10, +15]`:
**J1** GT-anchored local raw RMSE · **J2** local derivative RMSE · **J3** local curvature error ·
**J4** local correlation. These intentionally combine shape and placement and are labelled exactly
**GT-ANCHORED JOINT STRUCTURE**.

## 13. Alignment sensitivity — same functional only

`AlignmentSensitivity(m) = GT-anchored(m) − own-centre(m)` for the **same functional** only:
**D1** local raw RMSE · **D2** local derivative RMSE · **D3** local curvature error. Comparing own-centre
T6 against GT-anchored derivative RMSE is a **cross-functional comparison and is prohibited by this
contract** — that ambiguity is exactly what E1 exposed.

## 14. Scalar-metric warning, frozen into the contract

T4/T6/T7/T8 are **own-centre / local shape summary metrics**, not standalone placement metrics. E1 measured
that under jitter, re-centring changed these scalar primitives by ≤ 0.0035 at every level while the
waveform-level alignment-sensitive metrics changed by up to +0.28. **A future report may not infer "placement
is correct" from good T4/T6/T7/T8 alone.**

## 15. Future mandatory reporting block

Every future PPG schedule experiment must report at least: **A** exact-set fraction, absolute count error,
missing, spurious, T0/T1/T2/T3 · **B** matched timing MAE, p90 timing AE, T0-only timing MAE, T0-only p90 ·
**C** F1@50, F1@100, F1@150 · **D** `P → S` F1@50 and `P → S` timing MAE · **E** own-centre T4, T6, T7, T8,
local derivative RMSE, local curvature, chain coverage · **F** GT-anchored local raw RMSE, derivative RMSE,
curvature, local correlation. **No future experiment may omit an axis because the result is unfavourable.**

## 16. Aggregation and bootstrap

Per-beat values are aggregated **within a window by the median** (unless a metric's own definition says
otherwise), then within subject, then by **equal-subject macro** across `an0` and `k2s`. Bootstrap unit: the
**underlying ECG-window cluster** — all four site rows sharing one target ECG move together —
subject-stratified, **2,000 replicates**, `default_rng(20260904)`. **Beats are never bootstrapped
independently.**

## 17. Effect orientation, frozen universally

Higher-is-better: `effect = NEW − REFERENCE`. Lower-is-better: `effect = REFERENCE − NEW`. **Positive always
means NEW is better.** For damage analyses only: `Damage = Error_condition − Error_reference`, and such a
table's title **must** say *positive = more damage*. Conventions are never mixed within one table.

## 18. Validation arms

Exactly five, no additions: **ORACLE** (clean reference) · **JITTER_8 rep0** (topology correct, placement
degraded) · **MISS1 rep0** (undercount, no retained-beat timing shift) · **EXTRA1 rep0** (overcount, no
retained-beat timing shift) · **R1-SCHEDULE** (natural PPG schedule, mixed errors).

## 19. Validation gates

A contrast "clearly" moves when its paired clustered-bootstrap 95 % CI lies entirely above 0 in the stated
orientation.

| id | requirement |
|---|---|
| **V1** | JITTER_8 topology is 100 % T0 |
| **V2** | JITTER_8 schedule timing MAE clearly increases vs ORACLE |
| **V3** | JITTER_8 GT-anchored local derivative RMSE clearly worsens vs ORACLE |
| **V4** | JITTER_8 GT-anchored local curvature error clearly worsens vs ORACLE |
| **V5** | JITTER_8 `P → S` adherence remains HIGH ADHERENCE (≥ 0.90) |
| **V6** | `DerivativePlacementExcess` CI entirely > 0 |
| **V7** | `CurvaturePlacementExcess` CI entirely > 0 |
| **V8** | MISS1 own-centre T6 clearly worse than ORACLE |
| **V9** | MISS1 own-centre T7 clearly worse than ORACLE |
| **V10** | at least one of EXTRA1 own-centre T6 / T7 clearly worse than ORACLE |
| **V11** | MISS1 excess over JITTER_8 > 0 with CI > 0 for **both** T6 and T7 |
| **V12** | EXTRA1 excess over JITTER_8 > 0 with CI > 0 for **at least one** of T6 / T7 |
| **V13** | all six mandatory reporting blocks computable for R1 with finite values and coverage |

with, per window and then bootstrapped,

```
DerivativePlacementExcess = [deriv_GTanchored(J8) − deriv_owncentre(J8)]
                          − [deriv_GTanchored(ORACLE) − deriv_owncentre(ORACLE)]
```

and the same construction for curvature. **This is same-functional versus same-functional and it replaces
E1's ambiguous P3/P4 formulation.** The prohibited cross-functional comparison (own-centre T6 against
GT-anchored derivative RMSE) is never used as a gate.

MISS1 must be 100 % T2 and EXTRA1 100 % T3; these are reported as measurement sanity checks. V11/V12 are
**measurement sanity checks, not a new causal result** — E1 already established the underlying contrast.

## 20. Contract acceptance tree — exactly one verdict

- **A. EVENT-GEOMETRY EVALUATION CONTRACT ACCEPTED** — V1–V13 all pass. The suite separates event-set
  correctness, placement, own-centre morphology and GT-anchored joint structure on the frozen diagnostic
  arms. This licenses a **new preregistered** predictor experiment.
- **B. EVALUATION CONTRACT INVALID** — any of V1–V12 fails. The decomposition does not reproduce the known
  controlled contrasts. **Do not train a predictor**; diagnose the measurement implementation only.
- **C. CONTRACT INCOMPLETE FOR NATURAL SCHEDULES** — V1–V12 pass but V13 fails because R1 coverage or
  matching cannot support a required metric. **Do not train a predictor** until fixed under a new
  preregistration.

## 21. The machine-readable contract

`artifacts/e2_evaluation_contract/contract_v1.json` is written **before** any validation result and carries:
`contract_version`, the source E1 SHA, the metric definitions and directions, matching tolerances, window
support, normalisation constants, aggregation order, bootstrap unit, primary/secondary labels, coverage
requirements and terminology restrictions. **Future PPG schedule experiments must import this contract
rather than restate metric definitions.** The R1 reference card (§22) is generated separately, **loaded from
the frozen artifacts and hashed — never typed by hand**.

## 22. R1 development reference card

`r1_reference_card.json` is built from the frozen E1/O3 artifacts with provenance hashes and holds at least:
exact-set fraction, the dominant error category, R1 schedule F1@50, matched timing MAE, `P → S` adherence and
exact-set own-centre T6. It is the baseline for the next schedule-predictor experiment.

## 23. Future method gate policy

E2 freezes **what must be measured**, not the success threshold for a future predictor. A future predictor
experiment must preregister, **before training**: which frozen R1 arm is the schedule-side reference, which
B / O2c arm is the generator reference, exact improvement margins, exact non-inferiority margins, seed count
and compute budget. It **may not** change E2's metric definitions, remove a metric family, redefine
coverage, change matching, change support, change aggregation, or replace the event-set metrics with F1
alone.

## 24. Future predictor priority (research priority, not an architecture prescription)

If E2 passes, the next predictor experiment should be specified **EVENT-SET FIRST, TIMING SECOND**, with
schedule-side objectives in the order: reduce spurious events → reduce or preserve missing events → improve
the exact-event-set fraction → then improve timing on correctly represented events. **Nothing is implemented
under E2** — no Transformer, no phase network, no uncertainty model, no set decoder, no count head, no new
detector.

## 25. Figures

FIG 1 four-axis schematic showing which metric belongs to which family · FIG 2 ORACLE vs JITTER_8 (timing
MAE, own-centre derivative RMSE, GT-anchored derivative RMSE) · FIG 3 ORACLE / J8 / MISS1 / EXTRA1
own-centre T6 and T7 · FIG 4 R1 error decomposition (T0–T3, matched timing MAE, adherence, own-centre T6).
No waveform cherry-picking.

## 26. Artifacts

`docs/E2_EVENT_SET_PLACEMENT_MORPHOLOGY_CONTRACT_PREREGISTRATION.md`,
`docs/E2_EVENT_SET_PLACEMENT_MORPHOLOGY_CONTRACT_REPORT.md`, and
`artifacts/e2_evaluation_contract/`: `contract_v1.json`, `provenance.json`, `source_artifact_manifest.json`,
`frozen_metric_manifest.json`, `metric_taxonomy.json`, `matching_contract.json`, `aggregation_contract.json`,
`coverage_contract.json`, `r1_reference_card.json`, `contract_validation_metrics.csv`,
`contract_validation_bootstrap.csv`, `placement_excess.csv`, `topology_validation.csv`,
`validation_gates.json`, `decision.json`, `figures/`. **No checkpoint, no training log, no predictor output,
no new model directory.**

**Version control of the contract.** `artifacts/` is git-ignored in this repository because it holds results,
predictions and large files. The contract is not a result: it is a definition that future experiments must
import, so exactly five **definitional** JSON files — `contract_v1.json`, `metric_taxonomy.json`,
`matching_contract.json`, `aggregation_contract.json` and `coverage_contract.json` — are un-ignored by a
narrow `.gitignore` exception and committed. **Every validation result stays un-committed as usual**
(CSV tables, `validation_gates.json`, `decision.json`, `r1_reference_card.json`, `provenance.json`,
figures); the durable record of those numbers is the committed report.

## 27. Tests

**Repository** firewall, pins, A4 md5, C2 untouched. **Contract** `contract_v1` immutable fields, version
string, source E1 SHA, exact taxonomy. **Matching** ± 150 ms `S → G`, ± 50 ms `P → S`, deterministic
monotonic assignment, no schedule modification. **Topology** exact T0–T3 definitions, J8 all T0, MISS1 all
T2, EXTRA1 all T3. **Placement** matched-timing calculations, T0-only calculations, coverage attached.
**Morphology** own-centre support `[-10, +15]`, exact O1 primitives, same-event centring, no optimisation
shift. **Joint** GT-centred support `[-10, +15]`, same functionals, the correct "joint structure" label.
**Alignment** same-functional subtraction only, the T6-versus-derivative cross-functional gate prohibited.
**Aggregation** beat → window first, ECG-cluster bootstrap, equal subjects, 2,000 replicates, seed 20260904.
**Validation** exact V1–V13, and the contract cannot pass with a missing axis.

## 28. Commit order

1 repository integrity → 2 freeze E1 source artifact hashes → 3 write this preregistration → 4 write the
`contract_v1` draft → **5 commit + push preregistration and contract definition (metric definitions are
frozen here, before any validation result)** → 6 implement the reusable evaluator → 7 implement the
validation script → 8 tests → **9 implementation commit + push** → 10 reproduce the source E1 rows
(**fail ⇒ report, STOP**) → 11 run the contract validation → 12 freeze V1–V13 → 13 freeze the E2 decision →
14 figures → 15 report → 16 full tests → **17 result commit + push** → 18 verify clean tree → 19 STOP.
**NO TRAINING.**
