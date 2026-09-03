# O3 — Event-Schedule Error Tolerance & Frozen R1 Bridge — PREREGISTRATION

**How accurate must a supplied event schedule be before the O2c joint event–morphology advantage disappears,
and does the already-frozen PPG-only R1 Global-TCN schedule fall inside that region?**

## 0. Status, oracle disclosure and absolute rules

O3 is **not** a generator or method development experiment. **NO TRAINING. NO WEIGHT UPDATE. NO NEW EVENT
PREDICTOR. NO O2c RETRAINING. NO B RETRAINING. NO ATTENTION. NO ADAPTER. NO LOSS CHANGE. NO TEST SUBJECT.
NO C2. NO HYPERPARAMETER SEARCH.** Every generator and every event probe is loaded frozen, in `eval()`, with
`requires_grad = False`, and no optimizer is constructed anywhere in O3.

1. Designed **after** the O2c verdict A. Exploratory **mechanism bridge**, not confirmatory evidence.
2. Every synthetic schedule is derived by perturbing the **GT ECG R schedule**, so all synthetic arms remain
   **ORACLE DIAGNOSTICS**.
3. The R1 arm is **secondary**. The R1 Global-TCN was trained long before O3 (stage R1) and is used here
   exactly as frozen. It is PPG-only at inference, but it was **supervised with ECG R labels during training**.
4. Two development-validation subjects (`an0`, `k2s`), **no fresh test**, no new predictor training, one seed.
5. `O2C-CANON-ORACLE` was itself **trained with GT-R-derived canonical coordinates**. Nothing in O3 changes that.

### Allowed terminology

schedule-error tolerance · event-geometry tolerance · frozen R1 schedule bridge · end-to-end inference diagnostic

### Never allowed

"exact R timing is observable from PPG" · "phase is solved" · "deployability established" ·
"calibrated uncertainty" · "clinical validity" · "causal bottleneck proof" · "information-theoretic limit" ·
"SOTA" · "novelty".

## 1. Frozen components (identities asserted by test before any O3 number is produced)

| component | path | identity |
|---|---|---|
| **B** baseline generator | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` | file sha256 `557c7054…`, state `47d7ccb9…` |
| **O2c** canonical generator | `outputs/o2c_canon_oracle_seed42/checkpoint_final.pt` | file sha256 `5aab09be…`, state `f1cc44b3…`, step 10,046, 4,568,707 params |
| **O2b integer warp** | `src/ppg2ecg/evaluation/o2b_warp.py` | sha256 `cb4d1866…` |
| **O2 warp dependency** | `src/ppg2ecg/evaluation/o2_warp.py` | sha256 `046becfb…` |
| **R1 Global-TCN** | `outputs/r1_global_tcn_seed42/checkpoint_best.pt` | file sha256 `bfe76ea6…`, state `0986a7af…` |
| **R3 GTF-ORACLE** | not used in O3 | — |

Operator configuration is imported unchanged: `W = ANCHOR_W = 10`, `MIN_BEATS = 3`, `EPS = 1e-3`,
`MIN_INT_SPACING = 21`, `CORE_OFFSET_TOL = 1e-6`, `round_half_to_even`, bilinear
`grid_sample(align_corners=True, padding_mode="border")`, identity rows bit-exact, no renormalisation, no
amplitude Jacobian. **No operator edit of any kind is permitted; if any change were required, O3 stops.**

R1 event extraction is the frozen R1 path: `sigmoid(RhythmTCN(ppg))` → `rhythm_tcn.extract_events(prob,
threshold = 0.35, refractory = REFRACTORY_SAMPLES = 32)`. The threshold 0.35 was frozen on R1's internal dev
split before any `an0`/`k2s` window was loaded (`docs/R1_PPG_GLOBAL_RHYTHM_OBSERVABILITY_REPORT.md` §4).
**No threshold tuning, no NMS change, no preprocessing change.**

## 2. Primary cohort

The exact frozen O2c primary evaluation cohort, resolved through the same loader
(`scripts/o2_stage0_roundtrip.load_cohort`, salt `x4-event-nfe-v2`, 1,024 windows per subject, four sites):
**2,048 windows**, **19,834 GT beats**, **1,922 underlying ECG-window clusters**. The run asserts these three
numbers and stops otherwise. No approximate reconstruction.

All primary generator evaluations use **NFE = 4** and **Gaussian source seed 0** (frozen bank sha256
`86808579…`). No NFE sweep. `cohort_manifest.csv` is written before any generator call.

Feasibility metadata of the frozen GT schedules of this cohort, inspected before freezing this document (GT
properties only — no perturbation was generated, no schedule metric computed, no generator run): K ∈ [7, 16],
minimum GT RR 41 samples, first beat ≥ 39, last beat ≤ 1019, every window has ≥ 6 GT RR intervals of ≥ 42
samples, and the uniform canonical spacing stays ≥ 54 samples for K − 2 … K + 2 beats. The perturbation levels
below are the ones specified for O3 and were **not** chosen using this metadata; it is recorded only so that the
STOP rules of §5–§8 are stated in full knowledge of what they guard.

## 3. Frozen-model regression (before any O3 interpretation)

Reproduce on the primary cohort, at NFE 4 with source seed 0:

| arm | quantity | frozen value | tolerance |
|---|---|---|---|
| B | F1 excess | 0.3175618683270061 | \|Δ\| ≤ 1e-6 |
| B | raw F1 / chance F1 / missing / spurious / beats dev | 0.4367299151 / 0.1191680467 / 0.5661580180 / 0.5153896464 / 0.1066636102 | \|Δ\| ≤ 1e-6 |
| O2c ORACLE (J0) | F1 excess | 0.8592510052638713 | \|Δ\| ≤ 1e-6 |
| O2c ORACLE (J0) | T4 / T6 / T7 / T8 nAE | 0.40715406781869296 / 0.40190969840528823 / 0.41699228519117515 / 0.41387939453125 | \|Δ\| ≤ 1e-6 |

Any mismatch ⇒ verdict **FROZEN MODEL REGRESSION**, report, stop. Nothing else is interpreted.

## 4. Supplied-schedule notation and the O3 inference pipeline

GT schedule `R = [r_1 … r_K]`. A supplied schedule (perturbed or predicted) is `S = [s_1 … s_M]`. For every
O3 O2c arm, in this exact order:

1. `S` builds the O2b integer-grid canonical warp `W_S` (and its inverse).
2. The **raw PPG** is warped to canonical coordinates with `W_S`.
3. The frozen O2c generator produces a canonical ECG (NFE 4, source seed 0).
4. The **same** `W_S` is used for the inverse warp.
5. The generated raw-coordinate ECG is evaluated against the original paired **GT ECG / GT R**.

**GT R is never substituted back after `S` is created, and no schedule correction, alignment, offset removal,
peak snapping or smoothing is applied to the generated waveform at any point.**

## 5. Synthetic family A — integer per-beat jitter

Levels in samples: **J ∈ {0, 1, 2, 4, 6, 8}**; at 128 Hz, 1 sample = 7.8125 ms, so
J1 = 7.8125 ms, J2 = 15.625 ms, J4 = 31.25 ms, J6 = 46.875 ms, J8 = 62.5 ms.

For every GT beat `k` (0-based) of a row identified by `(subject, site, window_index)`:

```
key    = f"o3-jitter-v1|{rep}|{subject}|{site}|{window_index}|{k}|{J}"
u      = int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big")
delta  = u % (2*J + 1) - J                      # J = 0 gives delta = 0 with no special case
s_k    = clip(r_k + delta, 0, 1023)
```

No global RNG is consulted anywhere. Exactly **three perturbation replicates**, `rep ∈ {0, 1, 2}`.

Every resulting schedule is prechecked (§8). **Requirement: `s_{k+1} − s_k ≥ 21` for all beats.** If any jitter
schedule violates it, O3 stops **before** the generator sweep with verdict **JITTER PERTURBATION DESIGN
INVALID**. No reordering, merging, relative clipping, repair or window dropping is permitted.

## 6. Synthetic family B — beat miss

Conditions **MISS0** (= GT schedule), **MISS1**, **MISS2**: delete exactly one / exactly two **interior** GT
beats. The first and last beats are **never** deleted. Selection is a deterministic SHA256 rank over the
interior beat index `k ∈ {1 … K−2}`:

```
rank_k = int.from_bytes(sha256(f"o3-miss-v1|{rep}|{subject}|{site}|{window_index}|{k}".encode()).digest()[:8], "big")
```

ascending, ties broken by `k` ascending; the first `n` ranked interior beats are deleted. Three replicates
`rep ∈ {0, 1, 2}`. No result-dependent selection, no fallback. Every window must have enough interior beats;
this is verified explicitly at run time and any shortfall stops the family.

## 7. Synthetic family C — extra beat

Conditions **EXTRA0** (= GT schedule), **EXTRA1**, **EXTRA2**: insert one / two synthetic beats at integer
midpoints of existing GT RR intervals. For interval `i` between `r_i` and `r_{i+1}`:

```
m_i = round_half_to_even((r_i + r_{i+1}) / 2)
```

Interval `i` is **eligible** iff `m_i − r_i ≥ 21` **and** `r_{i+1} − m_i ≥ 21`. Eligible intervals are ranked by

```
rank_i = int.from_bytes(sha256(f"o3-extra-v1|{rep}|{subject}|{site}|{window_index}|{i}".encode()).digest()[:8], "big")
```

ascending, ties broken by `i` ascending; the first `n` eligible intervals receive an insertion. Insertions into
distinct intervals do not affect each other's eligibility. Three replicates `rep ∈ {0, 1, 2}`.

If any window lacks enough eligible intervals for the requested condition, O3 stops with verdict **EXTRA-BEAT
PERTURBATION DESIGN INVALID**. The inserted beat is never moved, no other rule is chosen post hoc, and no window
is excluded.

## 8. Schedule precheck (before any generator inference)

For **every** supplied schedule (synthetic and R1) verify: integer-valued positions · strictly increasing ·
all within `[0, 1023]` · minimum spacing ≥ 21 · all finite · **M ≥ 3** · the O2b warp is valid **and
non-identity** · the inverse is monotone · the maximum protected-core fractional coordinate ≤ 1e-6.
`schedule_precheck.csv` is written before any generator call. Any failure in a synthetic family ⇒ the
corresponding **PERTURBATION DESIGN INVALID** stop. (For R1, §21 applies instead.)

## 9. Schedule quality metrics — the x-axis of O3

For every condition / replicate the **supplied schedule itself** is scored against GT R, before the generator
runs: F1@50, F1@100, F1@150, F1@200, precision, recall, missing (`n_missing / n_ref`), spurious
(`n_spurious / n_ref`), beat-count deviation `|M/K − 1|`, matched-timing median AE and MAE in ms. Matching is
the frozen one-to-one `rpeaks.match_rpeaks` with `ref = GT R`, `pred = S`. Aggregation is the project's
equal-subject macro. These describe **schedule quality**, not generator quality.

## 10. Operator-floor curve and the J1 early-falsification gate

The O2b operator was morphology-preserving when its anchors were GT R. An imperfect `S` moves the protected
core away from the true QRS, so for **every** supplied schedule the round trip of the **GT ECG** is measured:
`x_can = W_S(ECG_GT)`, `x_rt = W_S^{-1}(x_can)`, reporting raw RMSE, T4/T6/T7/T8 nAE and original-vs-round-trip
event F1@50 with the exact O2 round-trip metric code.

**This floor is never subtracted from any generator error, never used to correct a generator metric and never
used to adjust a confidence interval.**

**Early falsification.** If **J1** (± 1 sample) produces a **median** operator-floor `T6 > 0.020` **or**
`T7 > 0.020` in **any** of the three replicates, O3 stops before the generator sweep with the pre-generator
verdict **OPERATOR TOO BRITTLE TO ONE-SAMPLE SCHEDULE ERROR**, because a predicted schedule cannot reasonably
be expected to be sub-sample exact. The threshold is not loosened.

For MISS / EXTRA an operator-floor failure does **not** stop the experiment; instead the morphology
interpretation of that condition is labelled **OPERATOR-CONFOUNDED**, while its count-error sensitivity remains
meaningful end-to-end.

## 11. Primary frozen O2c sweep

Only after the prechecks and the J1 operator gate pass. For every jitter × rep, MISS × rep and EXTRA × rep, the
frozen O2c runs at NFE 4 with source seed 0 and the metrics are the frozen ones:

**Event** raw F1 · chance floor · F1 excess · precision · recall · missing · spurious · beat-count deviation.
**O1-aligned morphology** T4 / T6 / T7 / T8 normalised AE at the frozen O1 train IQRs (T4 0.50532,
T6 0.22995, T7 0.03380, T8 31.25). **Secondary structure** raw RMSE · raw correlation · QRS RMSE · QRS-core
derivative RMSE · QRS-core curvature error · `qrs_e_dev` · `p2p_dev` · HF error.

## 12. Schedule adherence

Every generated ECG is additionally scored against the **supplied** schedule `S`: adherence F1@50, F1@100,
missing relative to `S`, spurious relative to `S` (frozen detector on the generated waveform, one-to-one
matching with `ref = S`). This separates "the generator stopped following the geometry" from "the generator
faithfully followed a wrong geometry". It is a diagnostic and **no causal attribution is claimed**.

## 13. Three perturbation replicates

The three replicates are **not** pooled as independent observations. Every severity level reports each
replicate separately plus a descriptive mean across replicates. A level counts as PASS only when the required
gate direction holds in **all three** replicates. The best replicate is never selected.

## 14. Bootstrap

Paired at the exact window, clustered on the **underlying ECG window** (all four site rows move together),
subject-stratified with equal `an0` / `k2s` weight, **2,000 replicates**, `default_rng(20260904)`.
Orientation: for higher-is-better metrics `effect = O2c_condition − B`; for lower-is-better metrics
`effect = Error_B − Error_O2c_condition`. **Positive always means the corrupted-schedule O2c arm is better
than B.** A metric "improves" iff the 95 % CI lies entirely above 0 and "worsens" iff it lies entirely below 0.

## 15. Joint-benefit survival gate

For a condition / replicate, **JOINT BENEFIT SURVIVES** iff all of:

| id | requirement |
|---|---|
| **G1** | F1 excess effect vs B: CI entirely > 0 **and** point ≥ **+0.10** |
| **G2** | T6 O1-aligned nAE effect: lower 95 % CI > **−0.020** |
| **G3** | T7 O1-aligned nAE effect: lower 95 % CI > **−0.020** |
| **G4** | at least one of T6 / T7 has CI entirely > 0 |
| **G5** | neither QRS-core derivative RMSE nor curvature error is clearly worse than B |
| **G6** | neither T4 nor T8 is clearly worse than B |

The margins are the frozen O2/O2c ones and are **not** changed. A severity level survives only when **all three**
perturbation replicates pass G1–G6.

## 16. Primary tolerance readout

`J_MAX` is the largest J ∈ {0, 1, 2, 4, 6, 8} for which all three replicates satisfy G1–G6, reported in samples
and milliseconds. **No threshold is interpolated between tested levels.** The actual schedule quality at
`J_MAX` (F1@50, F1@150, timing MAE, timing median AE) is reported alongside. For MISS and EXTRA the report
states whether MISS1 / MISS2 and EXTRA1 / EXTRA2 survive.

## 17. Retained-oracle-gain curves (descriptive only)

`EventGainRetention = (F1ex_condition − F1ex_B) / (F1ex_ORACLE − F1ex_B)` and, for a lower-is-better target `T`,
`MorphGainRetention_T = (Err_B − Err_condition) / (Err_B − Err_ORACLE)` for T4, T6, T7, T8. These are
**normalized performance-retention ratios only** and are never called information-retention fractions. Values
may be < 0 or > 1 and **nothing is clipped**.

## 18. Shape-vs-placement secondary diagnostic

The O2c report stated that GT-anchored morphology metrics cannot fully separate morphology from event
placement, so a **beat-identity** diagnostic is added. For JITTER every supplied event keeps its originating GT
beat identity; for MISS the retained supplied events keep theirs; for EXTRA the inserted events have **no** GT
morphology identity and are excluded. For every retained original beat: match a generated event to the supplied
event within ± 50 ms (one-to-one, frozen matcher); then compute the exact O1 per-beat primitives for T4, T6, T7,
T8 using **each waveform's own event centre** (generated beat at its detected event, GT beat at `r_k`) and
compare, normalised by the frozen O1 train IQRs. Report matched coverage and the per-beat nAEs.

This metric gives **no** penalty for missing generated beats or inserted schedule beats except through the
separately reported coverage, so it is a **SHAPE-ONLY SECONDARY DIAGNOSTIC**, not a waveform fidelity metric,
and it **cannot enter G1–G6**.

## 19. Multi-source analysis on selected conditions

Not run at every severity. The exact frozen Q1 512-window uncertainty subcohort, source seeds 0…7, NFE 4, arms:
**B**, **O2c ORACLE**, **JITTER_4 rep0**, **JITTER_8 rep0**, **MISS1 rep0**, **EXTRA1 rep0**, and
**R1-SCHEDULE** if stage B runs. Reported: beat-count SD, pairwise event F1@50, pairwise event F1@150,
pointwise waveform SD, pairwise waveform RMSE, and generated-event adherence to the supplied schedule across
sources. Secondary for the synthetic curves; part of **G7** only for the R1 arm.

## 20. The synthetic curve freezes first

Before **any** R1 schedule is extracted, the synthetic sweep completes and `synthetic_curve_frozen.json` is
written, containing `J_MAX`, the MISS and EXTRA tolerances, the full G1–G6 table, the schedule-quality table and
sha256 hashes of every synthetic artifact. **Stage B may not begin until that file exists**, so R1 performance
cannot influence the synthetic tolerance definition. The R1 script asserts the file's presence and records its
hash.

## 21. Stage B — the frozen R1 PPG-only schedule

No training. For each of the 2,048 primary PPG windows the frozen R1 Global-TCN is run with the exact frozen
preprocessing, threshold 0.35, NMS/refractory 32 and output conversion, giving `S_R1`. **No GT-based phase
correction, no offset correction, no beat insertion or deletion correction, no site-specific timing correction,
no oracle alignment, no schedule smoothing.** The exact frozen R1 output is the schedule.

**Validity.** Before O2c inference, `S_R1` is checked as in §8. If `M < 3`, the inherited O2b identity behaviour
is used — it is mathematically defined for any supplied schedule, since the identity map does not depend on the
schedule's provenance — and the fraction is recorded. **GT R is never used as a fallback.** If any *other*
invalid schedule exists (non-integer, non-monotone, out of bounds, spacing < 21, non-monotone inverse), the
R1 arm reports **R1 BRIDGE FAILS PRECHECK**, is not repaired and is stopped; the synthetic O3 result remains
valid.

**Quality (§9 metrics) and the operator-floor diagnostic (§10 metrics)** are computed for `S_R1` before the
generator runs. The R1 operator floor does **not** gate the end-to-end arm, but if its T6 or T7 exceeds 0.020 the
R1 morphology interpretation is labelled **schedule/operator coupled** and no R1 morphology degradation is
attributed purely to the generator. Nothing is corrected or subtracted.

**End-to-end arm `O2C-R1-SCHEDULE`**: raw PPG → frozen R1 → `S_R1` → integer-grid canonicalization → frozen O2c
→ inverse warp with `S_R1` → generated ECG. **No GT ECG R enters inference.** The name does not imply the R1
schedule is accurate. Evaluated with the same event, T4/T6/T7/T8, legacy structure and adherence metrics at
source seed 0 and NFE 4.

**R1 bridge gate.** G1–G6 as in §15, applied to `O2C-R1-SCHEDULE` vs B, **plus**

| id | requirement |
|---|---|
| **G7a** | beat-count SD across the 8 sources lower than B, paired CI favourable |
| **G7b** | pairwise event F1@50 across the 8 sources higher than B, paired CI favourable |

on the frozen 512-window / 8-source cohort, with the same clustered bootstrap. No percentage threshold. The R1
bridge counts as supported only if **G1–G7 all pass**.

## 22. Tolerance overlap and site-wise secondary

The observed R1 point is placed on the synthetic schedule-quality curves against schedule F1@50, schedule
F1@150, matched timing MAE and beat-count deviation, for the jitter, MISS and EXTRA levels. **These four axes
are never collapsed into one composite signal-quality index**, and it is **not** claimed that the synthetic
corruption model matches the R1 error distribution. The question asked is only whether the current R1 schedule
quality falls inside, near, or outside the region where the joint benefit survives.

Site-wise (secondary, no site causality claim): for sternum / head / wrist / ankle, the R1 schedule F1@50,
F1@150 and beat-count deviation, and `O2C-R1-SCHEDULE` F1 excess, T6 nAE and T7 nAE, against B and O2c oracle.

## 23. Final verdict tree — exactly one

- **PRETRAIN STOP — FROZEN MODEL REGRESSION** — the frozen B / O2c reproduction of §3 fails.
- **PRE-GENERATOR STOP — OPERATOR TOO BRITTLE TO ONE-SAMPLE SCHEDULE ERROR** — the J1 operator-floor median T6
  or T7 exceeds 0.020. Interpretation: the accepted coordinate operator itself lacks the minimum timing-error
  robustness a predicted schedule would need. Do not build a schedule predictor.
- **A. FROZEN R1 SCHEDULE BRIDGE SUPPORTED** — synthetic tolerance curve completed, R1 precheck valid, and
  `O2C-R1-SCHEDULE` passes G1–G7.
- **B. SYNTHETIC TOLERANCE SUPPORTED, CURRENT R1 BRIDGE NOT SUPPORTED** — JITTER_4 or a more severe level
  survives G1–G6 in all three replicates, but R1 fails G1–G7 or its precheck.
- **C. TOLERANCE REGION TOO NARROW** — J1 and/or J2 may survive, but J4 and every more severe level fail
  G1–G6, and the R1 bridge fails.
- **D. NO ROBUST SCHEDULE-TOLERANCE REGION** — J2 already fails G1–G6 in all three replicates even though the
  J1 operator-floor precheck passed.

**MISS and EXTRA never select the global verdict.** They characterise the failure mode, and the report states
explicitly whether one missed beat or one extra beat destroys the joint benefit.

Additional stop conditions that end O3 before the sweep: **JITTER PERTURBATION DESIGN INVALID** and
**EXTRA-BEAT PERTURBATION DESIGN INVALID** (§5, §7).

## 24. Claim boundaries for verdict A

If A occurs, the single allowed claim is: *"A frozen PPG-only event probe provides an inference-time schedule
that preserves the preregistered joint advantage of the oracle-trained canonical generator over the frozen
baseline on the development cohort."*

Not allowed: "PPG-derived phase solves PPG-to-ECG" · "the method is deployable clinically" · "exact R timing is
determined by PPG" · "the factorization is confirmed generally". The O2c generator was still **trained** with
GT-R-derived canonical coordinates, and that distinction stays explicit in every sentence.

## 25. Runtime preflight

Before the full sweep, 100 windows are run through all planned single-source synthetic conditions for one
perturbation replicate; wall time and VRAM are measured and the total is projected including three replicates,
the R1 arm and the selected 8-source arms. **If the projection exceeds 2.0 GPU-hours, O3 stops and reports the
projected cost.** The cohort, severity levels, replicate count and source count are never silently reduced.

## 26. Figures

FIG 1 schedule timing MAE vs generator F1 excess (jitter curve, R1 point, B line) · FIG 2 schedule timing vs T6
and T7 nAE · FIG 3 retained oracle gain (event, T4, T6, T7, T8) vs jitter level · FIG 4 count-error sensitivity
(ORACLE, MISS1, MISS2, EXTRA1, EXTRA2) · FIG 5 supplied-schedule F1@50 vs GT against generated-event F1@50 vs
supplied schedule · FIG 6 the synthetic tolerance region with the R1 point overlaid · FIG 7 site-wise R1 bridge.
Any waveform visualisation reuses the frozen V1 windows only; no cherry-picked waveform figure is required.

## 27. Tests

**Repository** test-subject firewall, PENGUIN / iMeanFlow pins, A4 md5, C2 untouched. **Frozen models** B, O2c,
R1 and operator hashes, `eval()` mode, `requires_grad = False`, no optimizer constructed anywhere.
**Perturbations** exact jitter levels, integer-only jitter, exact salts, exact reps 0/1/2, deterministic
reproducibility, no jitter repair, MISS1/MISS2 exact counts, first/last never deleted, EXTRA midpoint rule and
eligibility exact, no result-dependent selection. **Schedule** sorted, unique, spacing ≥ 21, integer, in bounds,
no GT correction after creation. **Operator** accepted O2b source hash, no operator edit, the same supplied
schedule used for the PPG warp and for the inverse warp. **Evaluation** exact 2,048 cohort, NFE 4, source seed
0, B and O2c regression, exact O1 T4/T6/T7/T8, ECG-window clustered bootstrap, three replicates not treated as
independent samples. **R1** exact frozen threshold 0.35, exact NMS/refractory, no phase correction, no
site-specific delay, no GT fallback, R1 evaluated only after `synthetic_curve_frozen.json` exists. **Verdict**
exact G1–G7 and the exact final verdict tree, MISS/EXTRA cannot alter the global verdict directly.

## 28. Artifacts

`docs/O3_SCHEDULE_ERROR_TOLERANCE_PREREGISTRATION.md`, `docs/O3_SCHEDULE_ERROR_TOLERANCE_REPORT.md` and
`artifacts/o3_schedule_tolerance/`: `provenance.json`, `frozen_component_manifest.json`, `cohort_manifest.csv`,
`runtime_preflight.json`, `perturbation_manifest.csv`, `schedule_precheck.csv`, `schedule_quality_metrics.csv`,
`operator_floor_metrics.csv`, `synthetic_generator_metrics.csv`, `synthetic_paired_bootstrap.csv`,
`schedule_adherence.csv`, `retained_gain.csv`, `shape_only_diagnostic.csv`, `synthetic_curve_frozen.json`,
`multisource_metrics.csv`, `multisource_bootstrap.csv`, `r1_schedule_manifest.csv`, `r1_schedule_quality.csv`,
`r1_operator_floor.csv`, `r1_generator_metrics.csv`, `r1_paired_bootstrap.csv`, `r1_site_metrics.csv`,
`joint_benefit_gates.csv`, `tolerance_summary.json`, `decision.json`, `figures/`.

**No new checkpoint, no training log and no model output directory is created.** Checkpoints, predictions, raw
data and large artifacts are never committed.

## 29. Commit order

1 repository integrity → 2 freeze component identities → 3 write this preregistration → **4 commit + push
preregistration** → 5 implement perturbations → 6 implement operator-floor audit → 7 implement evaluator and
bootstrap → 8 implement the R1 bridge path → 9 tests → **10 commit + push implementation** → 11 frozen B / O2c
regression (**fail ⇒ report, STOP**) → 12 perturbation prechecks (**fail ⇒ report, STOP**) → 13 operator-floor
curve (**J1 fail ⇒ report, STOP**) → 14 runtime preflight (**> 2 h ⇒ report, STOP**) → 15 synthetic
single-source sweep → 16 synthetic bootstrap → 17 freeze G1–G6 synthetic tolerance → 18 write
`synthetic_curve_frozen.json` → 19 selected multi-source synthetic analysis → 20 **only now** run the frozen R1
schedule extraction → 21 R1 schedule quality and precheck → 22 R1 operator-floor diagnostic → 23
`O2C-R1-SCHEDULE` evaluation → 24 R1 multi-source evaluation → 25 R1 G1–G7 → 26 tolerance-overlap analysis →
27 site secondary → 28 freeze the final verdict → 29 figures → 30 report → 31 full tests → **32 result commit +
push** → 33 clean-tree verification → 34 STOP. **NO TRAINING.**
