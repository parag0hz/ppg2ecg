# R1 — RhythmField-WW (final architecture experiment of this branch) — PREREGISTRATION (frozen)

**Status: FROZEN (2026-10-01).** Committed and pushed with the R1 implementation **before any R1 model is trained and
before any R1 metric exists**. It is never edited afterwards; amendments are dated and separate.
Hashes: `artifacts/r1_rhythmfield_ww/prereg_manifest.json`.

## 0. Context and evidence status

- **Closed lines:** BF0, D0, C0, C0-A and E0. None of their files, checkpoints, metrics or verdicts is modified.
- **Hypothesis:** thresholded event locations discard useful subthreshold rhythm information. A whole-window ECG
  predictor conditioned on a **dense rhythm field** may keep WW-DET's waveform realism while reaching CoherentBeat-level
  event fidelity.
- **Final experiment of the branch.** If R1 fails under the rules below, the PPG→ECG architecture search stops; there is
  no R2.
- **Populations:**
  - ARCH-TRAIN: training only.
  - ARCH-VAL: qualification and selection. It was analysed by C0, C0-A and E0, but the two new models have never been
    evaluated on it.
  - ARCH-HOLDOUT: **not used**; it appears only as historical context.
  - Old VitalDB V1 TEST (1,156 patients): opened only under §9.

## 1. Frozen references (never retrained)

| arm | source |
|---|---|
| placed events | the frozen C0 timing detector (RD1 RhythmTCN, 328,897 params): sigmoid(logits) → `extract_events(threshold 0.35, refractory 32)` |
| CoherentBeat-C0 | `outputs/c0_coherentbeat/c0.pt` (592,770), rendered by C0's `render_c0` at the placed events |
| WW-DET | `outputs/c0a_coherentbeat_ablation/ww.pt` (593,577), rendered by C0-A's `render_ww` (Gaussian raster, σ = 20 ms) |

- **Reproduction checks (ARCH-VAL):** checkpoint sha256 equal the C0 / C0-A records. The placed events re-extracted from
  the pre-threshold field equal C0's stored events.
- **STOP conditions:** the run stops unless C0 renders equal C0's stored renders, and the C0 / WW-DET / placed-event
  metrics equal the C0-A stored values within 1e-9. The metrics checked are F1, precision, recall, FP, FN, RR-MAE,
  HR-MAE, MAE, PCC, S4, S5, spectral deviation and FD.

## 2. The two new models (exactly two; parameter counts fixed before training)

**Dense rhythm field (SOFT).** The frozen detector's pre-threshold output sigmoid(logits).

- It is the exact tensor that C0's event extraction thresholds.
- The detector is fully convolutional at 128 Hz, so its output has the 512-sample ECG grid. **No interpolation.**
- No temperature, recalibration, sharpening or threshold.

| model | definition | trainable params |
|---|---|---|
| **SOFT-RHYTHM-WW** | WW-DET's class, unchanged (C0 encoder; decoder width 72, depth 5). Its single conditioning channel carries the dense field instead of the Gaussian raster. No hard channel | 593,577 (= WW-DET); full pipeline incl. the frozen detector 922,474 |
| **JOINT-RHYTHMFIELD-WW** | h = E(PPG) with C0's encoder; z_R = RhythmHead(h), a 1 × 1 64→64 conv, GELU, 1 × 1 64→1 conv giving [B, 512]; p_R = σ(z_R), **not detached, never thresholded**; ECG = D(h, p_R) with WW-DET's decoder | 597,802 = 593,577 waveform + 4,225 rhythm head (+0.71% vs WW-DET); no external detector |

**Excluded from both models:** beat-local generation, event-cell rendering, event guard, global-local decomposition,
stochastic latent, diffusion and flow matching.

## 3. Training (ARCH-TRAIN only, 3,470 patients; seed 42; one run each; last checkpoint)

- **Protocol (WW-DET's):** 20,000 steps × 64 windows, epoch-permutation batches, AdamW lr 1e-3, weight decay 0.01,
  gradient clip 1.0.
- **Not done:** early stopping, ARCH-VAL monitoring, seed selection, tuning.
- **SOFT:** L = full-window L1. The input field is the frozen detector's pre-threshold field on ARCH-TRAIN PPG
  (inference only, the same as at evaluation; the detector was trained on ARCH-TRAIN).
- **JOINT:** L = full-window L1 + 1.0 × L_rhythm.
  - L_rhythm uses the frozen C0 detector's convention: `BCEWithLogitsLoss` (mean) against the max-of-Gaussians field
    (σ = 20 ms) at ARCH-TRAIN reference R peaks (RD1 neurokit cache).
  - λ = 1.0, no tuning.
  - Waveform-loss gradients flow through p_R into the rhythm head and the shared encoder.
- **No other loss:** no spectral, PCC, peak, FD, HR, event-consistency, morphology, adversarial or stochastic loss.
- **Recorded:** time, params, peak GPU memory, sha256, NaN steps, optimizer, lr, batch, FLOPs.

## 4. Metrics (C0 / C0-A implementations)

**Event metrics.** One-to-one ±50 ms matching of the R peaks detected by neurokit on each waveform against the reference
R. The patient-macro quantities are computed **per patient by pooling the patient's windows**, then averaged with equal
patient weight:

| quantity | per-patient definition |
|---|---|
| FP rate | FP_p / windows_p |
| precision | TP_p / (TP_p + FP_p); undefined for a patient without detections, skipped |
| recall | TP_p / (TP_p + FN_p) |
| F1 | 2 TP_p / (2 TP_p + FP_p + FN_p) |

Also reported: pooled precision / recall / F1, and the historical window-mean F1 (continuity only), RR-MAE and HR-MAE.

**Waveform metrics:**

- FD (`kanflow_fd`, primary);
- beat-aligned correlation on matched pairs (reference R ↔ placed event), restricted to pairs finite in all four
  waveform arms;
- MAE, PCC, S4, S5, spectral ratio deviation.

## 5. Statistics

- 2,000 patient-clustered bootstrap replicates, seed 20261001.
- Paired: identical patient resamples for every arm and metric.
- FD is recomputed inside every replicate on the matched windows (`C0.fd_diff_ci`).
- Windows are never resampled independently.

## 6. ARCH-VAL gates (each candidate separately; differences = candidate − reference)

| gate | rule |
|---|---|
| V1 reduce WW false events | FP rate − WW: CI upper < 0 |
| V2 approach C0 event fidelity | FP rate − C0: CI upper < +0.03 |
| V3 preserve WW recall | patient-macro recall − WW: CI lower > −0.005 |
| V4 preserve WW realism | FD − WW: CI upper < +1.0 **and** FD − C0: CI upper < 0 |
| V5 morphology | beat correlation − WW: CI lower > −0.02 |

**QUALIFIED** = V1 ∧ V2 ∧ V3 ∧ V4 ∧ V5.

## 7. Descriptive analyses (no gates)

- **E0 taxonomy on every waveform arm:** A / B / C / D, with placed = the frozen detector events, by region.
- **Rhythm-field values** (a rhythm score, not a calibrated probability), for the frozen detector field and JOINT's p_R:
  - at reference R;
  - at correctly placed events;
  - at missed reference R;
  - at 20,000 random event-free positions (≥ 150 ms from every reference R and placed event; numpy seed 20261001);
  - at each candidate's rendered A / B / C / D detections.
- **JOINT rhythm head as a detector:** the frozen convention (0.35, 32) without tuning, scored with the same event
  metrics as the frozen detector.
- **FP-vs-FD Pareto plot:** BF0-DET and PM-BF0 (C0-A ARCH-VAL values, historical), C0, WW-DET, SOFT, JOINT. No combined
  score.
- **Compute:** parameters (waveform / head / full pipeline incl. the frozen detector for SOFT, WW-DET and C0), FLOPs,
  batch-1 GPU and CPU (4-thread) latency, training time and memory.

## 8. Candidate selection (frozen)

- **Neither qualifies:** R1 FAILED. TEST is not opened, and the architecture branch terminates.
- **Exactly one qualifies:** that candidate is selected.
- **Both qualify:**
  1. the lower FD point estimate wins;
  2. if |ΔFD| < 0.25, the lower FP rate wins;
  3. if |ΔFP| < 0.01, the simpler SOFT-RHYTHM-WW wins.
- **Interpretation of SOFT vs JOINT** follows the specification §26:
  - SOFT only → dense frozen rhythm conditioning; joint training not headlined.
  - JOINT only → the learned rhythm representation appears necessary.
  - Both → descriptive.
  - Neither → not supported.

## 9. Old V1 TEST — access, freshness audit, freeze

**Access.** `load_test()` refuses unless all of these hold:

- a committed, unmodified `final_test_freeze_manifest.json` exists;
- that manifest names a selected candidate;
- the freshness audit is CLEAN;
- every frozen hash matches.

**Freshness audit.** It runs only after a candidate qualifies, and is metadata-only: docs, artifacts, git history,
scripts and tracked logs. No TEST waveform or outcome is loaded.

- **Disclosed now:** project documents from the earlier program phases (8/25–9/25) report V1 TEST metrics for earlier
  models. The audit decides whether they meet the rule below.
- **Rule, frozen now.** TEST is **NOT CLEAN** if committed records show V1 TEST outcomes that were:
  - (a) computed for any model, detector, threshold or convention that R1 reuses, namely the RD1 / C0 timing-detector
    family and its threshold / refractory / target convention, WW-DET or CoherentBeat-C0; or
  - (b) cited as motivation or evidence for a design choice or hypothesis of this line (Beat-First design note, BF0,
    D0, C0, C0-A, E0, R1).
- **Otherwise CLEAN.** The audit writes `test_freshness_audit.md` ending in `VERDICT: CLEAN` or `VERDICT: NOT CLEAN`.
- **If NOT CLEAN:** stop before TEST, write the limitation, and do not call any later use a fresh final test.

**Freeze, if a candidate qualifies and the audit is CLEAN.** Commit `final_test_freeze_manifest.json`, which hashes:

- the selected checkpoint, the frozen detector, C0 and WW-DET;
- this preregistration and the R1 code;
- the shared metric, renderer and matching code;
- the qualification and selection files;
- the audit;
- the V1 manifest.

**TEST evaluation.** Exactly once on all windows of the 1,156 TEST patients.

- Arms: placed events, C0, WW-DET and the selected candidate only.
- Reference R is detected by neurokit on the TEST ECG, the same function as the cache.
- No retraining, threshold change or added metric afterwards.

## 10. Final TEST gates and verdict

**Gates:** T1 – T5 = V1 – V5 with the same margins, applied to the selected candidate on TEST.

| verdict | rule |
|---|---|
| STRONG | all of T1 – T5 pass |
| PARTIAL | T1 and T4 pass, and exactly one of T2 / T3 / T5 fails narrowly — within twice its margin (T2 upper < +0.06; T3 lower > −0.01; T5 lower > −0.04) |
| FAILED | otherwise, including T1 or T4 failing, or major recall degradation (T3 lower ≤ −0.01) |

**Consequences:**

- STRONG: only `docs/R1_RHYTHMFIELD_PAPER_ARCHITECTURE_SUMMARY.md`; no new modules, no stochastic generation.
- PARTIAL: write the remaining trade-off; no rescue, no R2.
- FAILED, ARCH-VAL failure, or a NOT CLEAN audit before TEST: "The preregistered final RhythmField-WW architecture did
  not resolve the deterministic event-fidelity / waveform-realism trade-off" (or the audit limitation). The architecture
  search is closed.

## 11. Tests, artifacts, claim boundaries

- **Tests:** `tests/test_r1_rhythmfield.py`, 15 cases covering the 23 required items:
  - split reuse without the holdout;
  - TEST seal;
  - frozen checkpoint hashes;
  - the pre-threshold soft field and no thresholded input to SOFT;
  - SOFT = WW family and parameters;
  - JOINT structure (one encoder, [B, 512] head, field into the decoder);
  - waveform and rhythm gradients;
  - no guard / renderer / stochastic path / extra loss;
  - the TRAIN-only rhythm target;
  - clustered patient-macro metrics;
  - paired comparisons;
  - TEST evaluation behind the freeze;
  - deterministic selection, gates and verdict.
- **Synthetic dry run:** every stage up to selection and the figure ran on synthetic data in a scratch directory, with
  real frozen checkpoints applied to synthetic PPG. The numbers mean nothing.
- **Artifacts:** `artifacts/r1_rhythmfield_ww/`, the specification §40 list (TEST files only if TEST is opened).
- **Not committed:** checkpoints and caches, in `outputs/r1_rhythmfield_ww/`.
- **Report:** `docs/R1_RHYTHMFIELD_WW_FINAL_ARCHITECTURE_REPORT.md` (23 sections).
- **Never claimed, even if STRONG:**
  - calibrated uncertainty or calibrated confidence of the field;
  - stochastic generation;
  - patient-specific morphology;
  - clinical validity or clinical R-peak recovery;
  - external or multi-seed robustness;
  - superiority to all methods, state of the art, or a first dense rhythm-field model;
  - a causal explanation of hallucinations.
