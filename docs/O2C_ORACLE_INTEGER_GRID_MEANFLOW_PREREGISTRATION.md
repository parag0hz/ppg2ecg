# O2c — Oracle Integer-Grid Event-Canonicalized MeanFlow — PREREGISTRATION

**Factorization hypothesis test.** Frozen before any O2c training or result; never edited afterwards.

| | |
|---|---|
| Base commit | `5f776ae0cef8cd5068b25e5e7c962a67050fd52e` (O2b acceptance report), clean tree |
| Upstream pins | PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`; A4 md5 `31c042d291052fbb6dc15263ad316be2`; C2 deferred, no C2 outputs; `outputs/o2c_canon_oracle_seed42/` does not exist |
| Test subjects | `kjd`, `ssx` — **never loaded** |
| Operator | the **accepted O2b integer-grid operator, imported unchanged** (identity frozen in `artifacts/o2c_oracle_integer_grid/operator_identity.json`) |

---

## 0. Status, oracle disclosure and claim boundary

**O2c is an ORACLE TARGET-LEAKAGE DIAGNOSTIC. The GT ECG R schedule is used at TRAINING and at INFERENCE to
build the temporal coordinate. O2c IS NOT DEPLOYABLE.**

1. Designed after O2b; **problem-discovery / mechanism diagnostic**, not independent confirmatory evidence.
2. **One training seed** (42), **two development-validation subjects** (an0, k2s), **no test subjects**, **no C2**.
3. O2b validated the *operator* only. O2c tests the **coordinate-level event/morphology factorization
   principle** under oracle event geometry. **No PPG-derived schedule is evaluated anywhere in O2c.**
4. No SOTA claim, no novelty claim, no claim that PPG provides exact R timing, no information-theoretic claim.

### Absolute rules

No new attention, adapter, event embedding, R-event or phase input channel, auxiliary event/QRS/morphology/
contrastive loss, architecture change, flow-objective change, inference peak snapping, waveform shifting, oracle
post-processing, hyperparameter search, early stopping, validation checkpoint selection, test subject or C2 work.
Exactly **one** new model is trained: `O2C-CANON-ORACLE seed42`. Anything beyond this needs a new preregistration.

## 1. Operator (frozen, imported, not edited)

O2c imports `ppg2ecg.evaluation.o2b_warp` — the implementation accepted by
`docs/O2B_INTEGER_GRID_CANONICALIZATION_REPORT.md` — and asserts its identity at runtime: `W = 10`,
`MIN_BEATS = 3`, `EPS = 1e-3`, `MIN_INT_SPACING = 21`, `CORE_OFFSET_TOL = 1e-6`, explicit
`round_half_to_even`, boundaries `0→0` / `1023→1023`, unchanged inverse, unchanged
`grid_sample(bilinear, align_corners=True, padding_mode=border)`, identity rows bit-exact, **no post-warp
renormalisation and no amplitude Jacobian**. Recorded source hashes: `o2b_warp.py` git blob
`ef6ca832…` / sha256 `cb4d1866…`; `o2_warp.py` git blob `ef474e7e…` / sha256 `046becfb…`. **If any modification
to the accepted operator were required, O2c stops** (that would be O2d with a new preregistration).

## 2. Compute matching (reused, not recomputed)

`artifacts/o2_oracle_canonicalization/baseline_step_resolution.json` is reused verbatim: baseline best round 46
(0-based 45) ⇒ **exactly 10,046 optimizer steps**, with the cross-check that 66 rounds ⇒ 14,409 steps, matching
the C2 preregistration. O2c trains for **exactly 10,046 optimizer steps — no more, no less**.

## 3. Reference baseline

Frozen **B** = `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` (file sha256 `557c7054…`, state_dict
`47d7ccb9…`, best_epoch 45 = round 46). Never retrained.

## 4. Training population and pipeline

The exact original generator TRAIN split (e61, fex, l38, n31, ngh, p5d, p9p, qm9, trh, tz8, u7y, w4p; 293,271
windows of `data/processed/wildppg_8s` under `split_a4_wildppg_seed42.json`) and the exact C1 window pipeline.
No new split, filtering, quality/site exclusion, filtering change or new normalisation.

## 5. Train-corpus integer-warp audit (before any GPU training)

For **every** training pair: frozen detector → accepted `q_int` schedule → verify monotone map and inverse,
finite values, integer `q_int`, protected-core fractional offset ≤ 1e-6, minimum integer event spacing ≥ 21.
`K < 3` inherits the identity warp. Report total windows, K distribution, `K<3` count/fraction, spacing
violations, invalid-warp count, minimum spacing, max core fractional coordinate, max `|q_int − q_real|`.

**STOP rules** (report, result commit, stop; no repair, no dropped windows, no `W` change, no dynamic fix):
**A** any non-`K<3` invalid warp · **B** `K<3` identity fraction > 0.5 % · **C** any integer spacing violation.

## 6. Mandatory O2b Stage-0 regression guard (before training)

Re-run the accepted O2b Stage-0 on the exact frozen 2,048-window cohort and require the same frozen gates:
R0-1 raw RMSE ≤ 0.020 · R0-2 T6 ≤ 0.020 · R0-3 T7 ≤ 0.020 · R0-4a T4 ≤ 0.020 · R0-4b T8 ≤ 0.020 · R0-5 F1@50 ≥ 0.98
· R0-6 median beat-count difference = 0, and compare against the frozen O2b reference numbers (raw RMSE 0.001696,
T4 0.000000, T6 0.000000, T7 0.0000109, T8 0.000000, F1@50 1.000000, beat-count difference 0). **Any failure ⇒
verdict `OPERATOR REGRESSION DETECTED`, no training, no repair under O2c.**

## 7. Canonical training transform

Per paired window: `q_int = O2b_integer_schedule(r)` from the GT ECG R schedule; build `f_r` and `f_r^{-1}`;
warp **both** modalities with the **same** map — `PPG_can(τ) = PPG_raw(f^{-1}(τ))`, `ECG_can(τ) = ECG_raw(f^{-1}(τ))`.
The Gaussian source `e ~ N(0, I)` is sampled **directly in canonical coordinates** and is never warped.

## 8. GT-R leakage boundary

GT R is permitted **only** in the warp construction and the inverse warp. The network must never receive an R
binary sequence, R Gaussian field, phase field, beat-count or RR scalar, `q_int` scalar/vector, event token or R
embedding, and **no loss may use R positions, beat counts, QRS masks or event labels**. Runtime assertions are
added where practical; static tests enforce the rest.

## 9. Model, objective, optimizer, RNG

Architecture: the exact C1 arm-B iMeanFlow generator (`MeanFlowS5` over
`build_penguin_backbone(n_step=1, sample_rate=128, h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0)`,
`cond_mode="h_only"`, `h_scale=1.0`); **parameter count exactly equal to B (4,568,707, asserted)**; no new or
deleted parameter. Objective: the existing `imeanflow_loss` unchanged (`norm_p 1.0`, `norm_eps 0.01`, forward
JVP; `x = ECG_can`, `c = PPG_can`, `e ~ N(0,I)` in canonical coordinates); no additional loss term. Optimizer and
RNG resolved from the C1 provenance and reused exactly: `AdamW(lr = 1e-3, weight_decay = 0.01)`, batch 64 as two
micro-batches of 32 with `(loss·Bc/B).backward()` and one step per batch, `seed_everything(42, deterministic=True)`,
loader generator 42 with `shuffle=True`, `(t,r)` generator 43 via `sample_tr_c1(arm="B")`, source `e` from the
CUDA global stream, no scheduler, no AMP, no gradient clipping. All values are recorded in
`training_manifest.json`. Initialization: fresh seed-42 construction on the same code path — **never** initialised
from trained B; the initial `state_dict` hash is recorded, and because no historical C1 initialization hash exists
the report will state that exact historical initialization identity could not be independently verified.

## 10. No validation model selection

`an0`/`k2s` are **not loaded during training**. Training runs exactly 10,046 steps; the **final** step is the only
primary checkpoint. Intermediate checkpoints at 0 / 25 % / 50 % / 75 % / final may be stored, but there is no
validation scoring, no best-checkpoint selection and no early stopping.

## 11. GPU preflight

Exactly 100 optimizer steps of the real O2c pipeline; measure wall time, ms/step and peak VRAM; discard the state;
project the 10,046-step cost. **If the projection exceeds 3.0 GPU-hours, STOP** — batch, steps, architecture and
dataset are not altered to fit runtime.

## 12. Evaluation

Cohort: the exact frozen **2,048-window** an0/k2s development cohort (19,834 GT beats asserted). **NFE 4**,
**source seed 0**, no NFE sweep before the verdict. Inference: GT R → accepted integer warp → warp PPG → sample
canonical ECG → **inverse warp to original coordinates**; the primary evaluation uses **only** the inverse-warped
original-coordinate ECG. No post-processing, no peak snapping, no shifting, and generated peaks never alter the warp.

**B regression gate** (before interpreting O2c): frozen B evaluated through the same evaluator on raw
coordinates must reproduce the frozen baseline row — raw F1 0.4367, chance 0.1192, F1 excess 0.3176, missing
0.5662, spurious 0.5154, beats dev 0.1067 — and the frozen structure metrics, to a tolerance fixed here of
**|Δ| ≤ 1e-6** on every macro value. Material failure ⇒ STOP, O2c is not interpreted.

**Event metrics**: raw F1, chance floor, **F1 excess** (primary), precision, recall, missing, spurious,
beats-ratio deviation. **Primary morphology**: the exact O1 functionals T4 `median_QRS_p2p`, T6
`median_QRS_max_abs_derivative`, T7 `median_QRS_curvature_energy`, T8 `median_QRS_width_ms`, each as
`|pred − GT|` normalised by the **O1 train IQR**; no functional substitution. **Secondary structure metrics**:
raw RMSE, raw correlation, fixed-coordinate QRS RMSE, derivative RMSE, curvature error, `qrs_e_dev`, `p2p_dev`,
HF metric.

**Operator floor**: the O2b round-trip distortion (median, p90, p95, max for T4/T6/T7/T8 and raw RMSE) is carried
into the report as a diagnostic. It is **never** subtracted from generator error, never used to correct outputs
and never used to adjust CIs.

**Bootstrap**: paired at the exact window, **clustered on the underlying ECG window** (all site rows together),
subject-stratified with equal an0/k2s weight, **2,000 replicates**, `default_rng(20260903)`. Orientation: for
event metrics `O2c − B`; for error metrics `Error_B − Error_O2c`; **positive always means O2c is better**.

## 13. Multi-source factorization test

The exact Q1 512-window uncertainty cohort (`q1-uncertainty-v1`, 64 per subject × site), arms **B** and
**O2C-CANON-ORACLE**, source seeds **0…7**, NFE 4. Report S-M1 beat-count SD, S-M2 pairwise generated-event
F1@50, S-M3 pairwise F1@150, S-M4 mean pointwise waveform SD, S-M5 pairwise waveform RMSE, and optionally S-M6
per-GT-beat timing SD under the frozen matching definition. The factorization prediction is that **source-driven
event variation decreases**; waveform diversity need not vanish, and reduced event variance is never called
"calibration".

## 14. Source-stability gates (copied verbatim from the frozen O2 preregistration §9)

> Gates: **S1** O2's beat-count SD lower than B with the paired CI entirely > 0 under the positive orientation, and
> **S2** O2's pairwise event F1@50 higher than B with CI entirely > 0. Waveform diversity is **not** required to
> collapse.

(with `O2` read as `O2C-CANON-ORACLE`.)

## 15. Primary success gate — copied verbatim from the frozen O2 preregistration §11

> | id | requirement |
> |---|---|
> | **J1** | O2 vs B **F1 excess**: 95 % CI entirely > 0 **and** point estimate **≥ +0.10** |
> | **J2** | T6 O1-aligned normalised AE **non-inferior** to B, margin **0.020** (CI lower bound > −0.020) |
> | **J3** | T7 O1-aligned normalised AE non-inferior to B, margin **0.020** |
> | **J4** | at least one of T6 / T7 **improves** vs B with CI entirely > 0 |
> | **J5** | neither frozen `qrs_deriv_rmse` nor `qrs_curvature_err` is clearly worse than B |
> | **J6** | neither T4 p2p nor T8 QRS width is clearly worse than B |
> | **J7** | source-stability gates **S1 and S2** both pass |

(with `O2` read as `O2C-CANON-ORACLE`.) The gates are **not** relaxed, reinterpreted or redesigned; the frozen
implementation `o2_warp.decide_o2` (margins `NONINF_MARGIN = 0.020`, `F1_EXCESS_MIN = 0.10`) is reused verbatim
and asserted by test. **Where any wording here differs from the frozen O2 preregistration, the frozen O2
preregistration wins.** No discrepancy was found when copying.

## 16. Verdicts — exactly one

- **PRETRAIN STOP — OPERATOR REGRESSION DETECTED** (§6 fails)
- **PRETRAIN STOP — TRAIN-CORPUS CANONICALIZATION INVALID** (§5 fails)
- **A. ORACLE EVENT-CANONICALIZATION JOINTLY SUPPORTED** — J1–J7 all pass
- **B. EVENT ANCHOR HELPS BUT MORPHOLOGY REMAINS UNRESOLVED** — J1 passes, one or more of J2–J6 fails
- **C. MORPHOLOGY IMPROVES WITHOUT MATERIAL EVENT BENEFIT** — T6/T7 clearly improve but J1 fails
- **D. NO MATERIAL ORACLE FACTORIZATION BENEFIT** — neither joint event nor morphology evidence supports the hypothesis

Because GT R defines the coordinate at inference, a large event improvement is **not** oversold: event gain,
T6/T7 morphology behaviour and source-driven event stability are reported separately, and the evidence for
factorization is the **joint** pattern (event fidelity materially improves **and** T6/T7 do not reproduce the
R2/R3 damage, preferably with decreased source-driven event variability). If event improves but T6/T7 worsen,
the central hypothesis **fails its intended purpose**.

## 17. Secondary analyses (never alter the primary verdict)

Canonical-domain diagnostic (canonical prediction vs canonical GT: F1@50, F1@150, T4/T6/T7/T8 nAE — does a
failure exist already in canonical generation or is it introduced by the inverse mapping?); site-wise map
(sternum/head/wrist/ankle: F1 excess and T4/T6/T7/T8 nAE for B and O2c, with the effect where the bootstrap
permits; no site causality claim); the frozen **R3 GTF-ORACLE** comparison (feature-level GT event injection vs
coordinate-level GT event normalization), never retrained, prominently labelled **TARGET LEAKAGE DIAGNOSTIC**,
and never a claim that a coordinate method is a superior architecture.

## 18. Visualization

The exact frozen V1 64 validation windows, rows: raw PPG · canonical PPG · raw GT ECG · canonical GT ECG · B NFE 4
· O2c canonical output · O2c inverse-warped output · GTF-ORACLE output, marking original `r` and `q_int`; plus
R-centred −300…+500 ms overlays. No shifting, no cherry-picking, no perceptual conclusion.

## 19. Tests

Repository (firewall, PENGUIN and iMeanFlow pins, A4 md5, C2 untouched); operator (imports the exact O2b
implementation, source hash asserted, `q_int`/`W`/rounding/inverse/resampler unchanged, no modification);
train audit (full corpus checked, `K<3` behaviour, spacing ≥ 21, no invalid warp, integer protected-core
coordinates); leakage (GT R only in warp construction, never in a model input, no phase input, no event map, no
event or QRS auxiliary loss); model (exact B architecture and parameter count, exact iMeanFlow objective, fresh
seed-42 init, exact optimizer/batch/RNG policy, exactly 10,046 steps, no validation access during training, final
checkpoint only); evaluation (exact frozen 2,048 cohort, source seed 0, NFE 4, B regression reproduction, exact
O1 functionals and train IQRs, no prediction shift, no oracle post-processing, ECG-window clustered bootstrap,
exact Q1 512 cohort, source seeds 0…7); decision (J1–J7 and S1/S2 copied from O2, verdict code exact, secondary
metrics cannot alter the primary verdict).

## 20. Artifacts

`docs/O2C_ORACLE_INTEGER_GRID_MEANFLOW_PREREGISTRATION.md`, `docs/O2C_ORACLE_INTEGER_GRID_MEANFLOW_REPORT.md`,
and `artifacts/o2c_oracle_integer_grid/`: `provenance.json`, `frozen_component_manifest.json`,
`operator_identity.json`, `baseline_step_resolution.json`, `train_corpus_warp_audit.csv`,
`stage0_regression_guard.json`, `stage0_regression_metrics.csv`, `training_manifest.json`,
`initialization_hash.json`, `training_log.csv`, `checkpoint_manifest.json`, `baseline_regression.json`,
`event_metrics.csv`, `o1_aligned_component_metrics.csv`, `structure_metrics.csv`, `paired_bootstrap.csv`,
`multisource_metrics.csv`, `multisource_bootstrap.csv`, `canonical_domain_metrics.csv`, `site_metrics.csv`,
`oracle_interface_comparison.csv`, `operator_floor_summary.csv`, `decision.json`, `figures/`. Model output goes
to `outputs/o2c_canon_oracle_seed42/`. Checkpoints, raw predictions, raw data and large artifacts are never committed.

## 21. Commit order

1 integrity → 2 freeze operator identity → 3 preregistration → 4 copy/check O2 J1–J7 and S1/S2 → **5 commit +
push preregistration** → 6 training integration → 7 train-corpus audit → 8 evaluator → 9 tests → **10 commit +
push implementation** → 11 train-corpus audit run (**fail ⇒ report, result commit, STOP**) → 12 Stage-0
regression guard (**fail ⇒ report, result commit, STOP**) → 13 100-step preflight (**> 3 h ⇒ STOP**) → 14 discard
preflight state → 15 fresh seed-42 init → 16 train exactly 10,046 steps → 17 freeze checkpoint identity → 18 load
validation → 19 B regression → 20 O2c NFE-4 primary → 21 O1-aligned T4/T6/T7/T8 → 22 secondary structure →
23 clustered paired bootstrap → 24 Q1 512-window multi-source → 25 canonical-domain diagnostic → 26 site
secondary → 27 GTF-ORACLE secondary → 28 freeze J/S results → 29 freeze verdict → 30 visualisations → 31 report →
32 full test suite → **33 result commit + push** → 34 verify clean tree → 35 STOP.
