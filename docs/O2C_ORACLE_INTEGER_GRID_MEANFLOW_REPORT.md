# O2c — Oracle Integer-Grid Event-Canonicalized MeanFlow — REPORT

**ORACLE TARGET-LEAKAGE DIAGNOSTIC. The ground-truth ECG R schedule builds the temporal coordinate at TRAINING
and at INFERENCE. `O2C-CANON-ORACLE` IS NOT DEPLOYABLE and is not a model proposal.**

Preregistration: `docs/O2C_ORACLE_INTEGER_GRID_MEANFLOW_PREREGISTRATION.md`, committed and pushed as `d458895`
**before** the train-corpus audit, before the preflight and before any weight update. Implementation commit
`39aa184`, visualisation commit `e5060aa`. Every number below was produced after those pushes; no threshold, no
gate and no verdict rule was touched afterwards.

Status: **problem-discovery / mechanism diagnostic**. One training seed (42), two development-validation
subjects (`an0`, `k2s`), **no test subjects loaded anywhere**, no C2 work. Not confirmatory evidence.

## FINAL O2c VERDICT

**A. ORACLE EVENT-CANONICALIZATION JOINTLY SUPPORTED** — J1–J7 all pass
(`artifacts/o2c_oracle_integer_grid/decision.json`, produced by the frozen `o2_warp.decide_o2`).

The executed verdict string is the frozen O2 one, `ORACLE EVENT-CANONICALIZATION JOINTLY SUPPORTED`. Section 15
of the O2c preregistration fixes precedence: where its wording differs from the frozen O2 preregistration, O2
wins. (The only such difference found is in the restated verdict **D** label — O2c §16 writes "NO MATERIAL ORACLE
FACTORIZATION BENEFIT", the frozen code writes "NO MATERIAL ORACLE CANONICALIZATION BENEFIT". Verdict D did not
occur; the discrepancy is recorded, not resolved post hoc, and is asserted by test.)

**Read §7 before reading the event numbers.** Under an oracle coordinate a large event gain is close to
definitional, and the verdict tree was written knowing that. The informative content of this run is the
*morphology* behaviour and the *source-stability* behaviour, not the event score.

## 1. Repository

`external/PENGUIN` at `6cd70cde…`, `external/iMeanFlow` at `bf60cd7c…` — unchanged and asserted by test. A4
checkpoint md5 `31c042d291052fbb6dc15263ad316be2` unchanged. No `outputs/*c2*` exists; C2 remains deferred. The
WildPPG test subjects `kjd` / `ssx` appear in no script and were never loaded (`test_subjects_loaded: []` in
`provenance.json`). Working tree clean at evaluation time (`provenance.json`: commit `e5060aa`, `dirty_files: 0`).
Full suite at the result commit: **420 passed**, of which 25 are the O2c tests.

## 2. Operator — imported, not edited

`artifacts/o2c_oracle_integer_grid/operator_identity.json` freezes the accepted O2b implementation by git blob
and sha256: `src/ppg2ecg/evaluation/o2b_warp.py` blob `ef6ca832…` / sha256 `cb4d1866…`, and
`src/ppg2ecg/evaluation/o2_warp.py` blob `ef474e7e…` / sha256 `046becfb…`. Configuration `W = 10`,
`MIN_BEATS = 3`, `EPS = 1e-3`, `MIN_INT_SPACING = 21`, `CORE_OFFSET_TOL = 1e-6`, banker's rounding, bilinear
`grid_sample(align_corners=True, padding_mode="border")`, identity rows bit-exact, no renormalisation, no
amplitude Jacobian. `"modification": "none"`. The test suite re-hashes both files and asserts that no O2c script
redefines the schedule, the anchors, the rounding, the inverse or the resampler.

## 3. Pre-training gates — all passed before any weight update

### 3.1 Train-corpus integer-warp audit (preregistration §5)

All **293,271** training windows of the 12 training subjects, **2,834,207** GT beats. No validation or test
subject was loaded.

| quantity | value | rule |
|---|---|---|
| K range (median) | 0 … 20 (10.0) | — |
| `K < 3` identity windows | **696 = 0.2373 %** | STOP B if > 0.5 % → pass |
| integer spacing violations | **0** | STOP C if > 0 → pass |
| other invalid warps | **0** | STOP A if > 0 → pass |
| minimum integer spacing (non-identity) | **47** samples | ≥ 21 required |
| maximum protected-core fractional coordinate | **0.0** | ≤ 1e-6 required |
| maximum \|q_int − q_real\| | **0.5** samples | rounding bound |

No window was dropped, no schedule was repaired and `W` was not changed.

### 3.2 Mandatory O2b Stage-0 regression guard (preregistration §6)

Re-run of the accepted operator's round trip on the exact frozen 2,048-window cohort with the exact O2 metric
code and the exact frozen O2 gate: R0-1 … R0-6 **all pass**, and the medians reproduce the frozen O2b reference
**bit-exactly** — `max |Δ| = 0.000e+00` across raw RMSE, raw corr, QRS-core RMSE, F1@50, beat-count difference
and T4/T6/T7/T8 (`stage0_regression_guard.json`, tolerance 1e-12).

### 3.3 GPU preflight (preregistration §11)

100 steps: **555.18 ms/step**, peak 19,187 MiB, projected **1.549 GPU-h** against the frozen 3.0 h budget → no
STOP. Preflight state was discarded (no checkpoint and no output directory existed afterwards).

## 4. Training

Exactly **10,046 optimizer steps** — the compute-matched count resolved in `baseline_step_resolution.json`
(`batch_rounds(loader, 220)`, 4,583 batches/epoch = 20 × 220 + 183, best round 46; cross-checked against the C2
preregistration's 14,409 steps for 66 rounds). Wall clock 92.7 min = **1.544 GPU-h**, peak 19,187 MiB.

Model, objective, optimizer and RNG are the C1 arm-B replay's, asserted field by field against
`outputs/c1_imf_baseline_replay_seed42/train_meta.json`: 4,568,707 parameters, `cond_mode="h_only"`,
`h_scale=1.0`, AdamW 1e-3 / 0.01, batch 64 as 2 × 32 micro-batches, `sample_tr_c1(arm="B")` with
logit-normal(−0.4, 1.0) and `data_proportion=0.5`, forward-mode JVP, seed 42 with loader generator 42 and (t, r)
generator 43, CUDA source stream. The **only** difference is the coordinate: per window, PPG and ECG are warped
by the **same** integer-grid map, and the Gaussian source is drawn directly in canonical coordinates and never
warped. 696 identity rows (K < 3) pass through bit-exactly.

Fresh seed-42 construction — never initialised from B. Initial state sha256 `026d90b3…`, final `f1cc44b3…`,
checkpoint file sha256 `5aab09be…`, 10,046/10,046 steps. `validation_loaded: false`, `validation_selection:
false`, `early_stopping: false`; the run wrote **one** checkpoint (`checkpoint_final.pt`) and no fractional or
best checkpoints exist. No historical C1 initialization hash exists, so historical initialization identity could
not be independently verified; the constructor, seed and initial hash are recorded instead
(`initialization_hash.json`).

## 5. Baseline regression gate

The frozen B row was reproduced **exactly** on the frozen 2,048-window cohort — `max |Δ| = 0.000e+00` against
raw F1 0.4367, chance F1 0.1192, F1 excess 0.3176, missing 0.5662, spurious 0.5154, beat-count deviation 0.1067
(preregistered tolerance 1e-6). The evaluation population is unchanged: 2,048 windows, **19,834** GT beats,
1,922 underlying ECG-window clusters, source bank seed 0 (sha256 `86808579…`), NFE 4 uniform. All 2,048 windows
have K ≥ 3, so **no identity warp** occurs in the evaluation cohort.

## 6. Results

### 6.1 Event metrics (subject-macro)

| arm | F1 | chance F1 | **F1 excess** | precision | recall | missing | spurious | beat-count dev |
|---|---|---|---|---|---|---|---|---|
| B (deployable baseline) | 0.4367 | 0.1192 | **0.3176** | 0.4435 | 0.4338 | 0.5662 | 0.5154 | 0.1067 |
| **O2C-CANON-ORACLE** | 0.9840 | 0.1247 | **0.8593** | 0.9846 | 0.9844 | 0.0156 | 0.0154 | 0.0140 |
| GTF-ORACLE (secondary) | 0.9384 | 0.1220 | **0.8164** | 0.9544 | 0.9295 | 0.0705 | 0.0380 | 0.0490 |

The chance floor itself rises slightly for O2c (0.1192 → 0.1247, effect −0.0056 [−0.0072, −0.0039]) because the
arm emits more detectable beats; F1 excess already subtracts it.

### 6.2 O1-aligned morphology (normalised absolute error, O1 train IQRs, lower is better)

| target | B | O2c | GTF-ORACLE | paired effect (positive = O2c better) |
|---|---|---|---|---|
| T4 median QRS p2p | 0.7467 | **0.4072** | 0.5596 | **+0.3395** [+0.3149, +0.3643] improves |
| T6 median QRS max abs derivative | 0.7513 | **0.4019** | 0.5519 | **+0.3493** [+0.3238, +0.3737] improves |
| T7 median QRS curvature energy | 0.5856 | **0.4170** | 0.5400 | **+0.1686** [+0.1477, +0.1899] improves |
| T8 median QRS width | 0.6504 | **0.4139** | 0.5455 | **+0.2365** [+0.2108, +0.2642] improves |

Scaling is the frozen O1 train IQR (T4 0.50532, T6 0.22995, T7 0.03380, T8 31.25). These functionals are
**self-referenced**: each is evaluated at the peaks detected *in the signal being scored*, so an arm with better
event timing is also measured at better-placed windows. §7 separates that from shape.

### 6.3 Secondary structure metrics (subject-macro)

| metric | B | O2c | GTF-ORACLE | paired effect | verdict |
|---|---|---|---|---|---|
| raw RMSE | 0.4233 | 0.2576 | 0.3648 | +0.1657 [+0.1628, +0.1687] | improves |
| raw corr | 0.1040 | 0.8410 | 0.4372 | +0.7369 [+0.7291, +0.7451] | improves |
| raw QRS RMSE | 0.5462 | 0.2600 | 0.4654 | +0.2863 [+0.2818, +0.2907] | improves |
| **QRS-core derivative RMSE** | 0.3220 | 0.1329 | 0.3131 | +0.1891 [+0.1861, +0.1920] | improves |
| **QRS-core curvature error** | 0.2147 | 0.1095 | 0.2151 | +0.1052 [+0.1032, +0.1071] | improves |
| QRS-energy deviation | 0.6056 | 0.3516 | 0.3958 | +0.2539 [+0.2319, +0.2756] | improves |
| p2p deviation | 0.2425 | 0.1350 | 0.1699 | +0.1075 [+0.0989, +0.1158] | improves |
| **HF fraction error** | 0.0854 | **0.0924** | 0.0957 | **−0.0070** [−0.0100, −0.0038] | **worsens** |

The high-frequency-fraction error is the single metric that clearly worsens. It sits in no J gate, it is reported
here rather than omitted, and it is consistent with the resampled canonical target carrying slightly different
high-frequency content than the raw target.

### 6.4 Multi-source factorization test (frozen Q1 512-window subcohort, source seeds 0…7)

| quantity | B | O2c |
|---|---|---|
| beat-count SD across sources | 1.2309 | **0.2118** |
| pairwise event F1@50 across sources | 0.3859 | **0.9748** |
| pairwise event F1@150 across sources | 0.6560 | 0.9837 |
| GT-beat timing SD across sources | 70.76 ms | **3.39 ms** |
| pointwise waveform SD | 0.2981 | 0.2256 |
| pairwise waveform RMSE | 0.4391 | 0.3003 |

- **S1** beat-count SD, B − O2c: **+1.0191 [+0.9541, +1.0880]** → CI entirely > 0, pass.
- **S2** pairwise event F1@50, O2c − B: **+0.5889 [+0.5682, +0.6098]** → CI entirely > 0, pass.

Waveform diversity did **not** collapse (pointwise SD 0.2981 → 0.2256, a 24 % reduction, not a collapse), which
the preregistration explicitly did not require. The stochasticity that remains has moved out of the event channel
and stayed in the waveform channel — that is the shape a factorization would predict.

## 7. What the oracle makes definitional, and what it does not

**The event result is largely definitional.** The canonical schedule `q_int` is a function of the GT R schedule
(`r_1`, `r_K`, K). In canonical coordinates the beats sit on a near-uniform grid, and the inverse warp puts
whatever the model produced back onto the true R positions. A model that only learns "emit a QRS on the canonical
grid" already scores near the event ceiling. **F1 excess 0.8593 is therefore not evidence that PPG carries R
timing, and it is not a performance claim.** J1 was written with a +0.10 floor precisely so that the event side
could not by itself carry the verdict; it is J2–J7 that carry the content.

**The GT-anchored morphology metrics also benefit from correct placement.** `qrs_deriv_rmse` and
`qrs_curvature_err` compare prediction and ground truth inside a QRS core centred on the **GT** R positions, so a
correctly placed QRS of mediocre shape scores better than a mistimed QRS of good shape. Neither these metrics nor
the self-referenced O1 functionals can separate "shape quality" from "shape placement", and this run does not
attempt to.

**What is genuinely informative** is the *direction* of the morphology result. The preregistered risk — the one
R2 and R3 actually realised — was that handing event geometry to the model damages QRS morphology. R3's frozen
verdict was **EVENT GAIN WITH STRUCTURE TRADE-OFF PERSISTS**. Here, with the event burden removed by a change of
coordinates rather than by an added feature, **no structure trade-off appears**: J2 and J3 non-inferiority pass
with large improvements, all four O1 targets improve, and both frozen QRS-core structure metrics improve. The
factorization hypothesis predicted exactly this asymmetry and it was not falsified.

The second informative comparison is §8.3: at a *comparable* event gain, the coordinate-level interface leaves
morphology far better than the feature-level interface — subject to the capacity/compute confound stated there.

## 8. Secondary analyses (they cannot alter the verdict)

### 8.1 Canonical-domain diagnostic — the inverse warp costs essentially nothing

| domain | F1@50 | F1 excess | F1@150 | raw RMSE | QRS-deriv RMSE | T4 | T6 | T7 | T8 |
|---|---|---|---|---|---|---|---|---|---|
| canonical (before inverse warp) | 0.9824 | 0.8569 | 0.9888 | 0.2577 | 0.1330 | 0.4063 | 0.4012 | 0.4182 | 0.4168 |
| raw (after inverse warp) | 0.9840 | 0.8593 | — | 0.2576 | 0.1329 | 0.4072 | 0.4019 | 0.4170 | 0.4139 |

The residual error is present already in canonical generation; the inverse mapping introduces none of it. The
canonical F1@50 is marginally *lower* than the raw one, so the inverse warp is not hiding a canonical failure.

### 8.2 Site map (no site causality claim)

| arm | sternum | head | wrist | ankle | spread |
|---|---|---|---|---|---|
| B, F1 excess | 0.3720 | 0.4453 | 0.2322 | 0.2177 | 0.2276 |
| O2c, F1 excess | 0.8633 | 0.8629 | 0.8582 | 0.8526 | **0.0107** |
| B, T6 nAE | 0.7179 | 0.6485 | 0.8049 | 0.8350 | 0.1864 |
| O2c, T6 nAE | 0.3815 | 0.3932 | 0.3995 | 0.4332 | 0.0517 |

The site ordering that B shows (head and sternum better than wrist and ankle) nearly vanishes under the oracle
coordinate. This is expected — the event coordinate no longer depends on PPG quality — and is not evidence about
sites.

### 8.3 Oracle interface comparison — **TARGET LEAKAGE DIAGNOSTIC**, both arms

| arm | oracle interface | F1 excess | QRS-deriv RMSE | QRS-curvature err | T4 | T6 | T7 | T8 |
|---|---|---|---|---|---|---|---|---|
| B | none (deployable) | 0.3176 | 0.3220 | 0.2147 | 0.7467 | 0.7513 | 0.5856 | 0.6504 |
| GTF-ORACLE | GT R as a conditioning field | 0.8164 | 0.3131 | 0.2151 | 0.5596 | 0.5519 | 0.5400 | 0.5455 |
| **O2c** | GT R as a time coordinate | 0.8593 | **0.1329** | **0.1095** | 0.4072 | 0.4019 | 0.4170 | 0.4139 |

At broadly comparable event gain, feature-level GT event injection leaves both frozen QRS-core structure metrics
essentially at the baseline (0.3131 vs 0.3220; 0.2151 vs 0.2147) while coordinate-level normalization more than
halves them.

**Confound, stated plainly:** the two arms are **not** capacity- or compute-matched. GTF-ORACLE is 12,849
trainable parameters trained for 2,200 steps on top of the frozen 4,568,707-parameter B; O2c is all 4,568,707
parameters trained from scratch for 10,046 steps. This comparison is therefore suggestive of an interface
difference, **not** a controlled test of one, and no claim that "a coordinate method is a superior architecture"
is made or licensed.

### 8.4 Operator floor (reported, never subtracted)

Median / p90 / p95 / max over the same 2,048 windows: raw RMSE 0.0017 / 0.0041 / 0.0053 / 0.0324; QRS-core RMSE
1.05e-06 / 1.73e-06 / 1.94e-06 / 2.70e-06; T4 0.0 / 1.27e-05 / 1.92e-05 / 0.0443; T6 0.0 / 2.01e-05 / 3.26e-05 /
0.0390; T7 1.09e-05 / 6.67e-05 / 1.14e-04 / 0.0144; T8 0.0 / 0.0 / 0.0 / 0.25.

The floor is roughly 0.7 % of O2c's raw RMSE (0.0017 vs 0.2576) and five orders of magnitude below its
QRS-core RMSE (1.05e-06 vs 0.2600), so
the operator is not the source of the effect. It was **never** subtracted from any generator error, never used
to correct outputs and never used to adjust a confidence interval.

## 9. Gates

| id | requirement | result |
|---|---|---|
| **J1** | F1 excess CI > 0 and point ≥ +0.10 | **PASS** (+0.5417 [+0.5278, +0.5554]) |
| **J2** | T6 non-inferior, margin 0.020 | **PASS** (lower bound +0.3238 > −0.020) |
| **J3** | T7 non-inferior, margin 0.020 | **PASS** (lower bound +0.1477 > −0.020) |
| **J4** | at least one of T6 / T7 improves | **PASS** (both improve) |
| **J5** | QRS-deriv RMSE and curvature error not clearly worse | **PASS** (both improve) |
| **J6** | T4 and T8 not clearly worse | **PASS** (both improve) |
| **J7** | S1 and S2 both pass | **PASS** |

All seven pass → verdict **A**, by `o2_warp.decide_o2`, unchanged since the O2 stage.

Bootstrap for every contrast: paired at the exact window, clustered on the underlying ECG window (all four site
rows move together), subject-stratified with equal `an0` / `k2s` weight, 2,000 replicates, `default_rng(20260903)`,
orientation such that positive always means O2c is better.

## 10. What this does NOT show

- It does **not** show that PPG-derived phase solves the problem, that event timing is deterministic from PPG,
  or that PPG contains exact R timing. **No PPG-derived schedule was evaluated anywhere in O2c.**
- It does **not** license "WHAT should be stochastic and WHEN deterministic" as a claim.
- It is **not** a deployability, novelty or SOTA claim. `O2C-CANON-ORACLE` cannot be run without the GT ECG.
- It does **not** separate morphology shape quality from morphology placement (§7).
- It is **not** confirmatory: one seed, two development-validation subjects, no test subjects, one architecture,
  one operator, one step budget.
- It does **not** establish that the coordinate interface beats the feature interface as an architecture — §8.3
  is confounded by capacity and compute.
- The high-frequency-fraction error worsened; nothing here claims uniform structural improvement.

## 11. Recommended next experiment (recommendation only — nothing implemented)

The honest next question is whether **any** of this survives when the schedule is not an oracle. Two candidates,
in the order that keeps the leakage boundary intact:

1. **Schedule-quality sweep on the same frozen operator.** Keep O2c's training and evaluation code fixed and
   replace the GT schedule at *inference only* with progressively degraded schedules (jitter of a preregistered
   set of magnitudes, dropped beats, inserted beats), measuring how fast the morphology gain decays. This costs
   no training, uses the already-frozen checkpoint, and quantifies how much schedule accuracy the factorization
   actually needs before anyone tries to predict one.
2. **Capacity- and compute-matched interface comparison.** If §8.3 is to become a claim rather than a hint, the
   feature-level arm must be retrained at O2c's parameter count and step budget. That is a new experiment and
   needs its own preregistration.

Neither is implemented, and C2 remains deferred.

## Artifacts

`artifacts/o2c_oracle_integer_grid/`: `operator_identity.json`, `baseline_step_resolution.json`,
`train_corpus_warp_audit.csv` / `.json`, `stage0_regression_guard.json`, `stage0_regression_metrics.csv`,
`gpu_preflight.json`, `initialization_hash.json`, `training_manifest.json`, `training_log.csv`,
`checkpoint_manifest.json`, `frozen_component_manifest.json`, `baseline_regression.json`, `event_metrics.csv`,
`o1_aligned_component_metrics.csv`, `structure_metrics.csv`, `paired_bootstrap.csv`, `multisource_metrics.csv`,
`multisource_bootstrap.csv`, `canonical_domain_metrics.csv`, `site_metrics.csv`,
`oracle_interface_comparison.csv`, `operator_floor_summary.csv`, `decision.json`, `provenance.json`,
`figures/` (64 frozen V1 atlas windows with GT R and `q_int` marked, R-centred −300…+500 ms overlays, primary
contrast forest). Model output: `outputs/o2c_canon_oracle_seed42/checkpoint_final.pt`. Artifacts, checkpoints,
predictions and raw data are not committed.

Scripts: `scripts/o2c_train_corpus_audit.py`, `scripts/o2c_stage0_regression.py`, `scripts/o2c_train.py`,
`scripts/o2c_evaluate.py`, `scripts/o2c_figures.py`. Tests: `tests/test_o2c_oracle_integer_grid.py` (25).
