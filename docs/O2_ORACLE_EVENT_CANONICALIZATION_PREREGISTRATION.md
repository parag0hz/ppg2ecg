# O2 — Oracle Event-Canonicalized MeanFlow — PREREGISTRATION

**Can event geometry be separated from morphology generation?**

Frozen before any O2 result. Once committed and pushed, never edited.

| | |
|---|---|
| Base commit | `8cb8c76c214116967be4ee523b65bf05ab38f6b2` (O1 report), clean tree |
| Companion | `docs/O2_CANONICAL_WARP_AUDIT.md` (the frozen warp operator), committed together with this file |
| Upstream pins | PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`; A4 md5 `31c042d291052fbb6dc15263ad316be2` unchanged; C2 still deferred, no C2 outputs |
| Test subjects | `kjd`, `ssx` — **never loaded** |
| Environment | RTX 5090, torch 2.11.0+cu130, numpy 2.3.5, scipy 1.16.3, neurokit2 0.2.12, Python 3.13.9 |

---

## 0. What O2 is and is not

**O2 is an ORACLE TARGET-LEAKAGE DIAGNOSTIC, not a deployable method.** The GT ECG R schedule is used to build
a temporal coordinate at **training and at inference**.

There is **NO** test-subject access, **NO** C2, **NO** novelty claim, **NO** SOTA claim, **NO** deployability
claim, **NO** new attention, **NO** new rhythm network and **NO** R-event auxiliary loss.

### 0.1 Status declarations (frozen)

1. **O2 was designed after O1.** It is problem-discovery evidence and **not** independent confirmatory evidence.
2. **GT ECG R peaks are used at inference.** O2 is not deployable and no result may be read as a deployable method.
3. It tests an **architectural / factorization hypothesis only**. It does **not** show that PPG itself supplies
   exact R timing, and it **cannot** establish identifiability.
4. Development-validation subjects only (`an0`, `k2s`); a single training seed.
5. Terminology: results speak about *beat schedule* and *morphology*, never "WHAT vs WHEN"; see §12.

### 0.2 Hypothesis

**H.** If the beat schedule is represented as event geometry (a temporal coordinate) instead of being jointly
generated inside the waveform stream, the generator can improve event correspondence **without** paying the
derivative/curvature penalty seen in R2/R3.

O1 motivates this: beat schedule is by far the most extractable component (T1 ρ 0.752, T2 ρ 0.802), while T6
(max derivative) and T7 (curvature energy) are only partially extractable **and** the frozen generator's matched
metrics move the wrong way under PPG shuffling (Q-B *candidate underutilization*).

## 1. Frozen references (never retrained)

| role | path | identity |
|---|---|---|
| **B** — deployable baseline, raw coordinate | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` | file sha256 `557c7054…`, state_dict `47d7ccb9…`, round 46, 4,568,707 params |
| **R3-GTF-ORACLE** — GT-R feature-injection oracle (**TARGET LEAKAGE**) | `outputs/r3_gtf_true_seed42/…` → the oracle arm is `outputs/r3_gtf_oracle_seed42/module_step2200.pt` on the same frozen generator, with the R2 GT-R field cache | resolved from `artifacts/r3_rhythm_fusion/train_provenance_gtf_oracle.json` |
| **O2-CANON-ORACLE** — new coordinate oracle (**TARGET LEAKAGE**) | `outputs/o2_canon_oracle_seed42/` | trained by this stage |

## 2. Data and split

Generator training uses the **exact** frozen A4/C1 train subjects (e61, fex, l38, n31, ngh, p5d, p9p, qm9, trh,
tz8, u7y, w4p; 293,271 windows of `data/processed/wildppg_8s` under `split_a4_wildppg_seed42.json`) and the exact
C1 baseline-replay window pipeline. No new train/dev split is introduced. Evaluation uses `an0`, `k2s` only.
`kjd`/`ssx` are never loaded.

## 3. Compute matching (resolved, not guessed)

`artifacts/o2_oracle_canonicalization/baseline_step_resolution.json` resolves the baseline's exact optimizer-step
count from the frozen `batch_rounds` semantics and the recorded run facts:

- `n_train = 293,271`, `batch_size = 64` → `ceil(293271/64) = 4,583` batches per epoch;
- `val_every_steps = 220` → each epoch is `20 × 220 + 1 × 183` rounds;
- best round = 46 (1-based; `best_epoch = 45`) → **exactly 10,046 optimizer steps**;
- the same derivation gives 14,409 steps for the full 66 rounds, which **matches the independently recorded C2
  preregistration figure exactly**.

O2 trains for **exactly 10,046 optimizer steps**, with **no early stopping, no validation evaluation during
training and no checkpoint selection**.

## 4. The model

`O2-CANON-ORACLE` uses the **exact** frozen iMeanFlow/PENGUIN architecture of C1 arm B (`MeanFlowS5` over
`build_penguin_backbone(n_step=1, sample_rate=128, h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0)`,
`cond_mode="h_only"`, `h_scale=1.0`), **4,568,707 parameters — identical to B**, no extra input channel and no new
parameter. Objective: the existing `imeanflow_loss` (`norm_p 1.0`, `norm_eps 0.01`, forward JVP,
`sample_tr` with `p_mean −0.4`, `p_std 1.0`, `data_proportion 0.5`). Optimizer: `AdamW(lr = 1e-3,
weight_decay = 0.01)`, batch 64 as 2 micro-batches of 32, seed 42, loader/`(t,r)`/CUDA-source RNG streams exactly
as in the C1 arm-B replay (`seed_everything(42)`, loader generator 42, `tr_gen` 43, `e` from the CUDA stream).

**The only difference from B is that the paired PPG and ECG are event-canonicalized before entering iMeanFlow.**

Initialization: the same seed-42 construction path as C1 arm B; the resulting `state_dict` hash is recorded in
`initialization_hash.json`. If no historical initialization hash exists for B, the report says so explicitly and
does not claim verified initialization identity. O2 is **not** initialised from the trained B checkpoint.

## 5. The warp

Exactly as frozen in `docs/O2_CANONICAL_WARP_AUDIT.md`: QRS-preserving piecewise-linear event warp,
`W = 10` samples, identity for `K < 3`, bilinear `grid_sample(align_corners=True)`, no renormalisation, no
amplitude Jacobian, Gaussian source drawn **in canonical coordinates**, inverse warp with the same `f_r`, and
**no event feature channel of any kind**.

## 6. Stage 0 — round-trip falsification (before any training)

On the frozen 2,048-window cohort, round-trip the **GT ECG** and evaluate raw RMSE, raw correlation, QRS-core
RMSE, the O1-aligned T4/T6/T7/T8 normalised absolute errors (scaled by the **exact O1 train IQRs**), the
detector F1@50 between original and round-trip ECG, and the beat-count difference. The gate (R0-1 … R0-6) and its
thresholds are in the warp audit §6. **Any failure ⇒ verdict `CANONICALIZATION OPERATOR REJECTED`, no generator
is trained, report and stop.** Thresholds are not loosened.

Warp validity is checked for every window (strictly increasing anchors in both coordinates, monotone map and
inverse, finite values, all slopes > 0). `K < 3` fraction and any other fallback fraction are recorded; **> 0.5 %
non-`K<3` fallbacks ⇒ STOP**. A diagnostic **centre-only** warp is round-tripped for comparison but never trained
and never enters a verdict.

## 7. Runtime preflight

100 optimizer steps of O2-CANON-ORACLE, measuring ms/step and peak VRAM; the state is discarded. **If the
projected full training exceeds 3.0 GPU-hours, STOP** — step count, batch and training subjects are not reduced
without a new preregistration.

## 8. Evaluation

Primary cohort: the frozen **2,048-window** an0/k2s development cohort (C0/C1/R2/R3/Q1/O1), 19,834 GT beats,
asserted against `artifacts/x4_0_event_reliability/nfe_subset.json`. Same frozen ECG detector, same matching,
**source seed 0**, **NFE 4 only** — no NFE sweep before the primary verdict.

Arms: **B** (deployable baseline), **R3-GTF-ORACLE** (target leakage), **O2-CANON-ORACLE** (target leakage). The
two oracle arms are never compared as deployable methods.

**Event metrics**: raw F1, count-matched random-phase floor, **F1 excess** (primary), precision, recall, missing,
spurious, beats-ratio deviation. No waveform shifting; no matching beyond the frozen detector tolerance.

**O1-aligned morphology metrics (primary morphology test)** — for every generated window compute the *same
scalar functional as O1* (`o1_targets.window_targets`) and the absolute error against the GT window's value,
normalised by the **O1 train IQR**: M1 = T4 `median_QRS_p2p`, M2 = T6 `median_QRS_max_abs_derivative`,
M3 = T7 `median_QRS_curvature_energy` (`mean(M1.d2²)` per beat, median per window), M4 = T8
`median_QRS_width_ms`. This prevents the O1-scalar / generator-waveform-functional conflation of the earlier
crosswalk.

**Secondary structure metrics** (frozen R2/R3 definitions): raw RMSE, raw correlation, fixed-coordinate QRS RMSE,
derivative RMSE, curvature error, QRS energy deviation, p2p deviation, HF metric.

**Bootstrap**: paired at the exact validation window, **clustered on the underlying ECG window** (all site rows
of a window move together), subject-stratified with equal an0/k2s weight, **2,000 replicates**,
`default_rng(20260903)`. Positive orientation: for F1 `A − B`; for error metrics `Error_B − Error_A`; positive
always means the first-named arm is better.

## 9. Source-stability test

On the Q1 frozen 512-window uncertainty subset (`q1-uncertainty-v1`, rebuilt metadata-only if needed), for **B**
and **O2** generate 8 samples with source seeds **0…7** at NFE 4 and report U1 beat-count SD, U2 pairwise
generated-event F1@50, U3 pairwise F1@150, U4 pointwise waveform SD, U5 pairwise waveform RMSE.

Gates: **S1** O2's beat-count SD lower than B with the paired CI entirely > 0 under the positive orientation, and
**S2** O2's pairwise event F1@50 higher than B with CI entirely > 0. Waveform diversity is **not** required to
collapse.

## 10. Secondary analyses (never alter the primary verdict)

- **§26 oracle-interface comparison** O2 vs R3-GTF-ORACLE on F1 excess, T4/T6/T7/T8, derivative RMSE and
  curvature error. No novelty claim, no causal-mechanism claim.
- **§29 site-wise** (sternum/head/wrist/ankle) F1 excess and T6/T7/T8 normalised AE. Exploratory; no site
  causality.
- **§30 canonical-domain diagnostic** for O2 only: score the canonical prediction against the canonical GT at the
  canonical `q_k` (F1@50, T6, T7) *before* the inverse warp. Primary claims use the inverse-warped output.

## 11. Primary success gate (frozen)

O2 joint factorization is supported only if **all** hold:

| id | requirement |
|---|---|
| **J1** | O2 vs B **F1 excess**: 95 % CI entirely > 0 **and** point estimate **≥ +0.10** |
| **J2** | T6 O1-aligned normalised AE **non-inferior** to B, margin **0.020** (CI lower bound > −0.020) |
| **J3** | T7 O1-aligned normalised AE non-inferior to B, margin **0.020** |
| **J4** | at least one of T6 / T7 **improves** vs B with CI entirely > 0 |
| **J5** | neither frozen `qrs_deriv_rmse` nor `qrs_curvature_err` is clearly worse than B |
| **J6** | neither T4 p2p nor T8 QRS width is clearly worse than B |
| **J7** | source-stability gates **S1 and S2** both pass |

### Final verdict — exactly one

- **CANONICALIZATION OPERATOR REJECTED** — Stage 0 fails; no generator trained.
- **A. ORACLE EVENT-CANONICALIZATION JOINTLY SUPPORTED** — J1–J7 all pass.
- **B. EVENT ANCHOR HELPS BUT MORPHOLOGY REMAINS UNRESOLVED** — J1 passes, one or more of J2–J6 fails.
- **C. MORPHOLOGY IMPROVES WITHOUT MATERIAL EVENT BENEFIT** — T6/T7 materially improve but J1 fails.
- **D. NO MATERIAL ORACLE CANONICALIZATION BENEFIT** — neither event nor morphology evidence is convincing.

Implemented in `o2_warp.decide_o2`, asserted by test to match this section. Nothing is modified after results.

## 12. Claim boundaries

**Allowed only if the J gates support it**: "Oracle event-coordinate factorization separates a substantial part
of beat-schedule variability from morphology generation under the frozen development protocol." · "This provides
evidence that coordinate-level event anchoring can avoid the event–morphology trade-off observed with direct
rhythm-feature injection."

**Never allowed**: "PPG-derived phase solves the problem" · "event timing is deterministic from PPG" · "WHAT
should be stochastic and WHEN deterministic". The permitted phrasing is: *"Beat schedule may be strongly
condition-constrained, while morphology retains residual uncertainty."*

## 13. Visualization

The exact frozen V1 64 validation windows, rows: raw PPG · canonical PPG · raw GT ECG · canonical GT ECG · B
NFE 4 · R3-GTF-ORACLE NFE 4 · O2 canonical output (before inverse warp) · O2 final output (after inverse warp),
marking original `r_k` and canonical `q_k`; plus R-centred −300…+500 ms overlays for GT/B/GTF-ORACLE/O2. No
output shifting; visuals are secondary and yield no metric conclusion.

## 14. Required tests

Firewall; upstream pins; C2 untouched; exact generator train subjects; exact evaluation cohort; GT detector
unchanged. **Warp**: identity bit-exact, monotone map and inverse, exact boundary anchors, `f(r_k) = q_k`,
`f^{-1}(q_k) = r_k`, local ±10-sample slope 1, no amplitude Jacobian, no post-warp renormalisation, same warp
applied to PPG and ECG, `K < 3` identity. **Leakage**: GT R used only in the warp builder / inverse warp, GT R
never in a model tensor, no R field, no phase feature, no event loss. **Model**: parameter count and architecture
identical to B, exact iMeanFlow objective, same optimizer configuration, same source and `(t,r)` RNG policy,
fixed optimizer-step count, no validation model selection. **Evaluation**: O1 T4/T6/T7/T8 functionals and O1
train-IQR scaling reused exactly, ECG-window cluster bootstrap keeps a window's site rows together, source seeds
exactly 0…7, verdict code matches §11.

## 15. Artifacts

`docs/O2_CANONICAL_WARP_AUDIT.md`, this file, `docs/O2_ORACLE_EVENT_CANONICALIZATION_REPORT.md`, and
`artifacts/o2_oracle_canonicalization/`: `provenance.json`, `frozen_checkpoint_manifest.json`,
`baseline_step_resolution.json`, `warp_manifest.csv`, `warp_slope_distribution.csv`,
`warp_roundtrip_metrics.csv`, `center_only_roundtrip_metrics.csv`, `training_manifest.json`, `training_log.csv`,
`initialization_hash.json`, `event_metrics.csv`, `o1_aligned_component_metrics.csv`, `structure_metrics.csv`,
`paired_bootstrap.csv`, `multisource_event_stability.csv`, `site_metrics.csv`,
`oracle_interface_comparison.csv`, `decision.json`, `figures/`. Checkpoints go to
`outputs/o2_canon_oracle_seed42/` and stay gitignored; predictions, raw data and large artifacts are never committed.

## 16. Commit order

1 integrity → 2 resolve baseline step → 3 warp utilities → 4 warp specification → 5 preregistration →
**6 commit + push** → 7 Stage-0 round-trip (**fail ⇒ report, result commit, STOP**) → 8 training integration →
9 tests → **10 commit + push implementation** → 11 preflight → 12 discard preflight state → 13 fresh seed-42
init → 14 train exactly 10,046 steps → 15 freeze checkpoint → 16 load validation → 17 primary NFE-4 evaluation →
18 O1-aligned T4/T6/T7/T8 → 19 multi-source stability → 20 site secondary → 21 GTF-ORACLE comparison →
22 visualization → 23 decision → 24 report → **25 result commit + push** → 26 STOP.
