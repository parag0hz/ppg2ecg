# R2 — Frozen Global Rhythm Scaffold → ECG Generator Transfer Probe — REPORT

**Preregistration:** `docs/R2_RHYTHM_SCAFFOLD_TRANSFER_PREREGISTRATION.md` (frozen at `f954e07`, untouched).
**Type:** minimal transfer / falsification probe. No new attention, transformer or flow objective; no C2;
no test; single seed; no full generator retraining; no hyperparameter search. **No generator block is
implemented.**

## FINAL R2 VERDICT

**SCAFFOLD INFORMATIVE, MINIMAL INTERFACE INSUFFICIENT** (verdict C of the frozen decision tree, §22).

| gate item (NFE 4, frozen 2,048 windows) | requirement | observed | pass |
|---|---|---|---|
| 1 | TRUE vs B F1 excess CI > 0 **and** point ≥ +0.02 | +0.0194 [+0.0160, +0.0227], CI > 0 | **no** (0.0006 below the minimal effect) |
| 2 | TRUE vs SHUFFLE F1 excess CI > 0 | +0.0102 [+0.0078, +0.0128] | yes |
| 3 | one of beats-dev / missing / spurious improves | missing +0.0196, spurious +0.0159 (beats-dev unresolved) | yes |
| 4 | fewer than two of S1–S5 clearly degrade | QRS-core derivative RMSE and curvature error worsen (2 of 5) | **no** |
| 5 | TRUE macro beats-ratio deviation < 0.20 | 0.1047 | yes |
| ORACLE | (v_OB, v_TB, v_OT) | (improves, improves, improves) → case 1 | — |

Decision path: not A (items 1 and 4 fail); not B (item 1 fails — "clearly improves" is item 1 with its
+0.02 magnitude); C because ORACLE beats B and beats TRUE on F1 excess with CIs > 0. Had item 1 passed,
item 4 would have made the verdict **B (event gain with structure trade-off)**, never A. No qualifier
applies (beats-ratio deviation and spurious do not worsen). `decision.json`.

Permitted reading (prereg §26 / §1): rhythm / event information *can* help the frozen generator, but the
PPG-derived scaffold and this minimal additive interface are together insufficient (possibility P-B);
extractor and interface are not separated causally, and the train/val scaffold reliability shift (§0 of the
preregistration) is an unseparated contributor.

## Repository

| item | value |
|---|---|
| start SHA | `71aefb44a180fe319c5343306bb35789a42ba78f` |
| prereg SHA | `f954e07c1353ed0d07d17b8dd4ab7d6a54c73787` (docs only) |
| implementation SHA | `5f3a3997fbb5584293d206e17a01658572c837cc` (code + 28 tests, no number computed) |
| result SHA | this commit |
| origin/main | == HEAD at every push; working tree clean at every stage |
| clean? | yes; submodules PENGUIN `6cd70cd`, iMeanFlow `bf60cd7` unchanged; C2 still deferred with zero weight updates; A4 checkpoint untouched |
| test access? | none — `kjd`/`ssx` never loaded (`assert_no_test_subjects` in every entry point; `provenance.json` `test_subjects_loaded: []`) |

All caches and manifests were (re)built at `5f3a399` (`build_provenance.json`); a first build at `f954e07`
with the then-untracked implementation was overwritten by the deterministic rebuild (identical sha256).

## Frozen components

| component | value |
|---|---|
| generator | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt`, round 45, C1 arm B; file sha256 `557c7054…`, state_dict sha256 `47d7ccb9…` = the A4 weights (`generator_checkpoint_manifest.json`) |
| Global-TCN | `outputs/r1_global_tcn_seed42/checkpoint_best.pt`; file sha256 `bfe76ea6…`, state_dict sha256 `0986a7af…`; dilations 1…128, RF 2,041, best epoch 12 (`rhythm_checkpoint_manifest.json`) |
| frozen parameter counts | generator 4,568,707; Global-TCN 328,897 — all `requires_grad = False`, asserted to receive no gradient after step 1 in every process |
| trainable adapter | `rhythm_adapter.proj.weight` only: Conv1d(1 → 128, k = 1, no bias), zero-initialised, **128 weights** (`trainable_parameters.json`) |
| arm-B parity | `RhythmMeanFlowS5` with a zero adapter is `torch.equal` to the frozen `MeanFlowS5` on the full 2,048-window NFE-4 prediction, for a zero scaffold **and** for the real scaffold; B's 14 macro metrics reproduce `artifacts/c1_interval_exposure/stage1_metrics.csv` row (B, 4) with **max |Δ| = 0** |

Scaffold: dense pre-NMS `sigmoid(GlobalTCN(PPG))`, detached, no threshold / NMS / shift, computed in
32-window batches at training and evaluation. Per-window max of the field: training-visited windows mean
0.847 (p10 0.634, p90 0.969); validation 0.854 (p10 0.652, p90 0.963) — the amplitude distribution does not
differ between the in-sample and out-of-sample scaffolds; the quality shift (R1: F1@50 = 0 in 5.4 % of
validation windows) is not visible in amplitude.

## Runtime

| item | value |
|---|---|
| preflight (100 TRUE steps, discarded) | 381 ms/step (step 1: 808 ms), peak 6,268 MiB allocated / 6,508 reserved → projected 3 × 2,200 steps = **0.70 GPU-h** (budget 6.0; no STOP) |
| TRUE | 2,200 steps (asserted), 846.6 s = 14.1 min, 384 ms/step, peak 6,268 MiB, final adapter L2 7.70 |
| SHUFFLE | 2,200 steps, 848.0 s = 14.1 min, 385 ms/step, peak 6,270 MiB, final L2 5.84 |
| ORACLE | 2,200 steps, 841.6 s = 14.0 min, 382 ms/step, peak 7,414 MiB (+1.2 GB field cache), final L2 12.04 |
| paired-randomness probe hash (first 4 micro-batches of idx, t, r, e) | `04aad6ae5ec41798…` identical across preflight, TRUE, SHUFFLE, ORACLE (asserted at the fourth micro-batch of each trained arm) |
| loader | first 2,200 batches of the seed-42 epoch-1 order; 140,800 windows visited, none twice; fex + p5d share 17.1 % |
| ORACLE cache | 293,271 GT-R fields built on CPU in 43 s (sha256 `2e6c548c…`, verified in the ORACLE process with 32 recomputed rows); 24 rows without a detected beat |
| evaluation | 5.7 min (13 arm × NFE generations, C0 scoring, bootstraps); site-wise secondary 134 s (projected 95 s against a 7,200 s budget) |
| adapter magnitude at step 2200 on the validation population | RMS(rhythm_e) / RMS(ppg_e): TRUE 0.062, SHUFFLE 0.047, ORACLE 0.112 |

Training curves: the weighted loss is 0.9999 in every arm (as pre-disclosed); the unweighted MSE over the
last 100 steps is 0.1225 (TRUE), 0.1224 (SHUFFLE), 0.1213 (ORACLE) against 0.1200 over the first 100 —
batch composition dominates the training curves and arm differences are within it; arms are compared on the
validation metrics only.

## NFE 4 — Event correspondence (frozen 2,048 windows, 19,834 GT beats)

| arm | raw F1 | chance floor | **F1 excess** | precision | recall | missing | spurious | beats ratio | beats dev | median \|Δt\| | mean Δt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 0.4367 | 0.1192 | 0.3176 | 0.4435 | 0.4338 | 0.5662 | 0.5154 | 0.9492 | 0.1067 | 23.4 ms | −6.9 ms |
| TRUE | 0.4562 | 0.1193 | **0.3369** | 0.4628 | 0.4534 | 0.5466 | 0.4995 | 0.9530 | 0.1047 | 23.4 ms | −6.3 ms |
| SHUFFLE | 0.4457 | 0.1189 | 0.3267 | 0.4524 | 0.4429 | 0.5571 | 0.5077 | 0.9506 | 0.1074 | 23.4 ms | −6.5 ms |
| ORACLE (GT-R leakage; diagnostic only) | 0.4798 | 0.1197 | 0.3601 | 0.4870 | 0.4767 | 0.5233 | 0.4770 | 0.9536 | 0.1006 | 23.4 ms | −5.6 ms |
| TRUE adapter, +2.0 s shifted scaffold (§18) | 0.4353 | 0.1193 | 0.3161 | 0.4418 | 0.4326 | 0.5674 | 0.5179 | 0.9505 | 0.1070 | 23.4 ms | −6.8 ms |

Macro = equal-subject mean; missing / spurious are fractions of GT beats; no window had an empty prediction
or zero GT beats. The matched timing error is unchanged by conditioning (median 23.4 ms in every arm): the
adapter changes *which* beats are found, not how precisely a found beat is placed.

## TRUE vs B (paired, subject-stratified, 2,000 replicates, seed 20260902; positive = TRUE better)

| metric | effect | 95 % CI | verdict |
|---|---|---|---|
| **F1 excess** | +0.0194 | [+0.0160, +0.0227] | improves (point < +0.02 → item 1 fails) |
| beats-ratio deviation | +0.0020 | [−0.0002, +0.0043] | unresolved |
| missing fraction | +0.0196 | [+0.0163, +0.0229] | improves |
| spurious fraction | +0.0159 | [+0.0123, +0.0195] | improves |
| precision / recall | +0.0193 / +0.0196 | both CI > 0 | improves |

+0.0194 is +6.1 % of B's F1 excess and 1.8× B's own NFE 2 → 4 gain on this population (+0.011); it is
0.0006 short of the pre-specified minimal effect. At the secondary NFE the same contrast is +0.0181
[+0.0152, +0.0210] (NFE 1) and +0.0207 [+0.0175, +0.0238] (NFE 2): the gain is present from the first step
of the sampler and does not grow with steps.

## TRUE vs SHUFFLE (specificity) and SHUFFLE vs B

| comparison | F1 excess | beats dev | missing | spurious |
|---|---|---|---|---|
| TRUE vs SHUFFLE | +0.0102 [+0.0078, +0.0128] improves | +0.0028 [+0.0008, +0.0048] improves | +0.0105 improves | +0.0082 improves |
| SHUFFLE vs B | +0.0092 [+0.0070, +0.0113] improves | −0.0008 unresolved | +0.0091 improves | +0.0077 improves |

Reading rule (§16): SHUFFLE − B is "improves", not "worsens", so item 2 is read as window-specific
transfer — but with the disclosure that the unaligned scaffold itself carries **about half (0.0092 / 0.0194)
of TRUE's gain**: a generic, rate-distributed scaffold from the same subject × site improves event
correspondence through this interface without any window-specific alignment. The adapter norms (TRUE 7.70,
SHUFFLE 5.84) are not far apart, so SHUFFLE ≈ B does not hold. At NFE 1 and 2 TRUE vs SHUFFLE is +0.0102 and
+0.0111 (CIs > 0).

## Structural protection (NFE 4)

| arm | S1 raw RMSE | S2 raw corr | S3 QRS RMSE | S4 deriv RMSE | S5 curvature | S6 QRS-energy dev | S7 p2p dev | S8 HF ratio (GT 0.194) | whole-window RMSE / corr | QRS-core RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 0.4233 | 0.1040 | 0.5462 | 0.3220 | 0.2147 | 0.6056 | 0.2425 | 0.2305 | 0.4113 / 0.1061 | 0.5858 |
| TRUE | 0.4214 | 0.1129 | 0.5449 | 0.3233 | 0.2157 | 0.5925 | 0.2365 | 0.2322 | 0.4095 / 0.1145 | 0.5847 |
| SHUFFLE | 0.4222 | 0.1086 | 0.5453 | 0.3223 | 0.2149 | 0.6007 | 0.2413 | 0.2315 | 0.4103 / 0.1107 | 0.5850 |
| ORACLE (leakage) | 0.4199 | 0.1198 | 0.5449 | 0.3253 | 0.2172 | 0.5732 | 0.2287 | 0.2331 | 0.4078 / 0.1213 | 0.5850 |

TRUE − B paired effects (positive = TRUE better): S1 +0.0019 [+0.0016, +0.0022] improves; S2 +0.0089
[+0.0076, +0.0103] improves; S3 +0.0013 [+0.0009, +0.0018] improves; **S4 −0.0013 [−0.0017, −0.0010]
worsens; S5 −0.0010 [−0.0013, −0.0008] worsens.** Both clear degradations are in the QRS-core family
(S3–S5); the segment family (S1–S2) improves. The degradation scales with the injection magnitude
(S4 effect vs B: SHUFFLE −0.0003, TRUE −0.0013, ORACLE −0.0033), i.e. adding a per-channel copy of a smooth
100 ms-wide field to the PPG stem output buys beat placement at the cost of QRS-core sharpness. S6/S7
(calibration ratios) improve, as M1 warned they can while direct structure worsens; HF fraction and its
error are unchanged.

## Oracle diagnostic (F1 excess, NFE 4; ORACLE = GT-R leakage, diagnostic only)

| contrast | effect | 95 % CI | verdict |
|---|---|---|---|
| ORACLE − B | +0.0425 | [+0.0379, +0.0474] | improves |
| TRUE − B | +0.0194 | [+0.0160, +0.0227] | improves |
| ORACLE − TRUE | +0.0231 | [+0.0197, +0.0266] | improves |

Case 1: the generator / interface can exploit rhythm information, and the PPG-derived scaffold is still a
bottleneck (TRUE recovers 46 % of the ORACLE gain). Two further readings, pre-disclosed: (i) ORACLE's gain
is bounded by the interface and partly detector-circular — even a field centred exactly on every scored R
location raises F1 excess by only +0.043 (13 % of B) through this path, so the additive 1×1 interface is
itself a ceiling; (ii) ORACLE degrades S4/S5 more than TRUE (−0.0033 / −0.0025 vs B), so perfect event
information through this path also does not come with QRS structure. ORACLE's missing 0.523 / spurious 0.477
remain far from zero.

## Phase ablation (TRUE adapter, scaffold rolled by +256 samples = +2.0 s; positive = TRUE better)

| stratum (φ = frac(256 / mean GT RR)) | n | F1 excess | missing | spurious |
|---|---|---|---|---|
| all | 2,048 | +0.0209 [+0.0173, +0.0244] | +0.0208 | +0.0183 |
| in-phase (φ < 0.1 or ≥ 0.9) | 216 | +0.0064 [+0.0007, +0.0123] | +0.0069 | +0.0063 |
| anti-phase (0.4 ≤ φ ≤ 0.6) | 559 | +0.0334 [+0.0246, +0.0419] | +0.0324 | +0.0313 |
| rest | 1,273 | +0.0181 [+0.0141, +0.0221] | +0.0185 | +0.0151 |

The phase-destroyed scaffold gives **no** benefit (its own F1 excess 0.3161 is below B's 0.3176), and the
TRUE-minus-shifted effect grows monotonically from the residual in-phase stratum to the anti-phase stratum,
exactly the signature of phase dependence; the overall figure is a lower bound, as pre-stated.

## NFE event persistence (B vs TRUE, ±250 ms greedy one-to-one; ms)

| arm | match fraction NFE 1 / 2 / 4 | mean \|δ\| matched 1 / 2 / 4 | sign(δ1) = sign(δ4) | δ4 − δ1 (mean) | \|δ4\| − \|δ1\| | NFE 4 strictly closer | ties |
|---|---|---|---|---|---|---|---|
| B | 0.821 / 0.823 / 0.816 | 75.1 / 73.1 / 71.5 | 0.911 | +3.9 | −2.8 | 0.186 | 0.680 |
| TRUE | 0.829 / 0.830 / 0.825 | 72.8 / 70.6 / 69.2 | 0.908 | +3.8 | −2.8 | 0.187 | 0.679 |

On the intersection of beats matched at NFE 1 and 4 in both arms (15,075 beats; B-only 215, TRUE-only 389):
mean |δ1| 72.0 (B) vs 69.9 (TRUE), |δ4| 69.2 vs 67.2; "NFE 4 closer" 0.184 in both. Rhythm conditioning
shifts detected events by ~2–3 ms on average at every NFE and adds ~0.8 % matched beats; the trajectory
refinement from NFE 1 to 4 is the same in both arms (68 % of beats do not move at all at integer-sample
resolution). Detector-dependent diagnostic; no solver claim.

## Site-wise (R1 8,192-window validation cohort, B vs TRUE at NFE 4; exploratory, 16 uncorrected CIs)

| site | B F1 excess | TRUE F1 excess | TRUE − B F1 excess | beats dev | QRS RMSE (S3) | deriv RMSE (S4) |
|---|---|---|---|---|---|---|
| sternum | 0.3784 | 0.4105 | +0.0321 [+0.0286, +0.0354] | +0.0057 improves | +0.0015 improves | −0.0021 worsens |
| head | 0.4239 | 0.4541 | +0.0302 [+0.0269, +0.0335] | +0.0032 improves | +0.0012 improves | −0.0018 worsens |
| wrist | 0.2625 | 0.2954 | +0.0329 [+0.0292, +0.0365] | +0.0033 improves | +0.0005 unresolved | −0.0026 worsens |
| **ankle** | 0.2165 | 0.2102 | **−0.0063 [−0.0090, −0.0036] worsens** | +0.0015 unresolved | +0.0010 improves | +0.0007 improves |
| contrast (wrist + ankle) − (sternum + head) | | | −0.0178 [−0.0211, −0.0146] proximal gains more | −0.0020 unresolved | −0.0006 | +0.0010 |

The answer to the exploratory question is no: rhythm conditioning does **not** help the distal sites more.
It helps sternum, head and wrist by a similar +0.03 and **hurts ankle**, the site where the R1 scaffold was
weakest (F1@50 0.45); on ankle the adapter also leaves S4 intact, i.e. it injects less and mis-places what it
injects. The contrast uses an independent subject-stratified two-group bootstrap (windows are unpaired
across sites; adapted from the C1 difference-of-improvement idiom; rng re-seeded per metric).

Scaffold-quality strata (§17, exploratory; terciles of the scaffold's own F1@50, edges 0.405 / 0.889):
TRUE − B F1 excess +0.0058 [+0.0010, +0.0106] (low), +0.0273 [+0.0208, +0.0340] (mid), +0.0244 [+0.0192,
+0.0298] (high). The gain is concentrated where the scaffold is right and vanishes where it is wrong — the
1×1 adapter has no way to know which is which.

## Visual observations (V1 validation VIZ cohort, 64 windows, 619 GT beats; counts only, no selection)

| arm | missing | spurious | windows with ≥ 1 spurious | windows with ≥ 1 missing |
|---|---|---|---|---|
| B | 359 | 327 | 59 | 57 |
| TRUE | 348 | 327 | 58 | 55 |
| SHUFFLE | 352 | 325 | 58 | 56 |
| ORACLE (leakage) | 338 | 312 | 57 | 56 |

TRUE scaffold value at the GT R samples: median 0.72, p10 0.31, p90 0.93; 12.9 % of GT beats sit under a
scaffold value below the R1 event threshold 0.35. On these 64 windows the arms differ by 11 missing beats
and 0 spurious beats between B and TRUE — the same order as the population effect. Atlas:
`visual_atlas/{sub}_{site}_w{idx}.png` (rows PPG / TRUE scaffold with its value at each GT R / GT ECG /
B / TRUE / SHUFFLE / ORACLE at NFE 4, missing × and spurious ▲ annotated) and `…_zoom.png` (GT-R-centred
−300 … +500 ms, first eligible beat), index in `visual_atlas/atlas_index.csv`; B rows reproduce the V1
construction (32-row stratum batch, seed-0 bank).

## What this does NOT prove

- **Stronger-supervision caveat.** The Global-TCN was trained on GT ECG R-peaks; TRUE conditioning carries
  target-derived training supervision the baseline never had. Any comparison with other methods must give
  them equivalent event supervision. ORACLE is target leakage by design and appears only as a diagnostic.
- **Single seed** (42) for the adapters and one loader / source / (t, r) realisation shared by the arms; the
  CIs quantify validation-window sampling within an0 / k2s only, not training-seed or subject variance.
- **No test evidence** (`kjd` / `ssx` never loaded); **no SOTA claim**; **no novelty claim** (a zero-init
  1×1 adapter is an instrument, not a method); **no full generator retraining** and no claim about what a
  retrained or richer interface would do; the ORACLE ≈ ceiling of +0.04 is a property of this path within
  2,200 steps, not of attention.
- E1–E4 are detector-dependent; the detector-independent S3–S5 show the QRS-core trade-off.
- Gate items 3 and 4 are disjunctive / tolerant by preregistration and uncorrected; the 16 site-wise CIs are
  exploratory.
- The item-1 miss is by 0.0006 on a point estimate whose CI excludes zero: the frozen rule is applied as
  written, and the report does not argue it away. With item 1 passed the verdict would still not have been A.

## Deviations, corrections and disclosures

1. **Implementation review before the implementation commit** (recorded in the commit message of
   `5f3a399`): a heterogeneous-row CSV writer that would have crashed the evaluator after all computation
   and before `provenance.json`, the missing cross-process probe-hash assertion, the missing site-wise
   budget / isolation guard, training-window scaffold statistics, C0 text identity of `_peaks` /
   `_chance_chunk`, a contiguous channel split, and tiny-backbone test gates (CPU conv kernels differ by
   3e-8 when weights require grad; the protocol compares frozen vs frozen and is bit-exact).
2. **`provenance.json` `tests_ran.summary`** captured the pytest documentation line instead of the
   "28 passed" line; the recorded exit code 0 and empty skip list are the meaningful fields. The parser is
   fixed in this commit (post-run; no number depends on it).
3. **Evaluation determinism flags.** The evaluator and the atlas ran under the same non-deterministic cuDNN
   defaults as C0 / C1 / V1 (recorded: `cudnn_deterministic: false`); arm-B parity and the stage-1
   regression (Δ = 0) hold under them.
4. **Site contrast** implemented as an independent two-group bootstrap (windows unpaired across sites), an
   adaptation of the named idiom; stated in `site_metrics.csv` and `provenance.json`.
5. **Scaffold batch.** Evaluation-time scaffolds are computed in 32-window batches to match training; the
   SHUFFLE partner field at evaluation is gathered from the population-order field (1-ULP class difference
   at most, no metric consequence).
6. The optional secondary NFE grid was run for all four arms (event family and S1–S8 rows in
   `event_metrics.csv` / `structure_metrics.csv`); paired effects at NFE 1 / 2 are reported for the four
   event metrics only, as frozen.

## Recommended next architecture (recommendation only — NOT IMPLEMENTED)

Verdict C keys, by the preregistration, to *cross-attention / temporal fusion becomes justified*. The R2
evidence sharpens what that fusion must do and must not do:

- **Separate WHEN from WHAT.** The additive stem injection trades QRS-core sharpness for beat placement in
  proportion to its magnitude (S4 / S5 worsen for SHUFFLE < TRUE < ORACLE). A rhythm path should therefore
  enter where it can shift *where* beats are placed without re-shaping the QRS core — a temporal-fusion path
  (cross-attention or a gated temporal mixer over the scaffold) feeding the target stream, not a per-channel
  add to the PPG embedding.
- **Gate by confidence.** The gain is +0.006 in the low-quality tercile, negative on ankle, and +0.024 to
  +0.027 where the scaffold is right; the 1×1 adapter cannot tell them apart. The scaffold's own field value
  (12.9 % of GT beats sit under 0.35) is a free confidence signal that the fusion should consume.
- **Do not retrain the generator yet, and do not build a large transformer.** The ORACLE ceiling of +0.04
  shows that even perfect event information barely moves this generator through an additive path; the
  first step is the interface, evaluated under a new preregistration with the same frozen population, bank,
  and gate, with the stronger-supervision caveat carried forward.

## Artifacts

`artifacts/r2_rhythm_transfer/`: `provenance.json`, `generator_checkpoint_manifest.json`,
`rhythm_checkpoint_manifest.json`, `trainable_parameters.json`, `shuffle_manifest.csv` (train / eval /
viz), `shuffle_eval_descriptive.json`, `cache_build.json`, `loader_order_provenance.json`,
`build_provenance.json`, `runtime_preflight.json`, `training_log_{true,shuffle,oracle}.csv`,
`train_provenance_{true,shuffle,oracle}.json`, `metrics_by_window.csv` (26,624 rows: 13 arm × NFE
configurations × 2,048), `event_metrics.csv`, `structure_metrics.csv`, `paired_bootstrap.csv`,
`oracle_gap.csv`, `phase_ablation.csv` + `phase_ablation_summary.csv`, `nfe_event_persistence.csv`
(39,668 beat rows) + `nfe_event_persistence_summary.json`, `scaffold_quality_strata.csv`,
`site_metrics.csv`, `decision.json`, `visual_atlas/` (64 + 64 zoom PNGs, `atlas_index.csv`,
`atlas_summary.json`). Adapter weights (128 floats × 4 steps × 3 arms) in
`outputs/r2_{true,shuffle,oracle}_adapter_seed42/`. Nothing in `outputs/`, `artifacts/`,
`data/processed/` enters git. Full suite: 291 passed.

**R2 ends at its verdict and this recommendation. STOP.**
