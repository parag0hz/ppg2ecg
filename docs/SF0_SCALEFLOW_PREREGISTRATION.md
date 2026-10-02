# SF0 — ScaleFlow-ECG Feasibility — PREREGISTRATION (frozen)

**Status: FROZEN (2026-10-02).** Committed and pushed with the split and the implementation **before the timing detector
or any waveform model is trained and before any SF-VAL outcome exists**. It is never edited afterwards; amendments are
dated and separate. Hashes: `artifacts/sf0_scaleflow/prereg_manifest.json`.

## 0. Status, scope, evidence

- **Closed line.** BF0 / D0 / C0 / C0-A / E0 / R1 are closed and unmodified.
- **What SF0 is.** A new architecture line with a different formulation: whole-window conditional generation by flow
  matching, with a fixed multiresolution (Haar) representation and coarse → mid → fine coupling. It is not R2.
- **No novelty claim.** No "first", "novel" or "state of the art" wording; a literature review is required before any
  such claim.
- **SF-VAL** is new for this line but **not project-naive**. Its patients were training data of earlier project models
  (the C0 / C0-A / R1 detectors and waveform models were trained on the former ARCH-TRAIN). They were never a primary
  evaluation population.
- **Never used:** ARCH-VAL and ARCH-HOLDOUT.
- **Old V1 TEST** (1,156 patients): unopened unless §11 allows it.

## 1. Split (`split_manifest.json`, `split_hashes.json`; created before this commit)

- **Source:** the former ARCH-TRAIN (3,470 patients, C0's split manifest).
- **Rule:** `default_rng(20261002).permutation` of the sorted patient IDs. The first 433 become **SF-VAL** (29,223
  windows); the remaining 3,037 become **SF-TRAIN** (201,997 windows).
- **Checks passed:**
  - the two sets are disjoint and their union is ARCH-TRAIN;
  - no ARCH-VAL or ARCH-HOLDOUT patient;
  - no old V1 validation or test patient;
  - never reshuffled.
- **Window identifiers:** wid = index in the frozen ARCH-TRAIN concatenation.
- **Reference R:** the RD1 neurokit cache, as in C0.

## 2. Timing detector (retrained)

- **Training:** the RD1 / C0 RhythmTCN protocol, unchanged, on **SF-TRAIN only**.
  - Gaussian target, σ = 20 ms, at reference R; BCE with logits.
  - 14,000 × 64, AdamW 1e-3 / 0.01, seed 42, last checkpoint.
- **Events:** threshold 0.35, refractory 32. One event sequence per window.
- **Evaluation:** every waveform model receives the **same** Gaussian event raster (σ = 20 ms) of these events.
- **Training raster:** all four models are trained on the reference-R raster (WW-DET convention).

## 3. The four models (exactly four; widths by parameter count only — `parameter_match.json`)

| model | definition | params |
|---|---|---|
| **A WW-L1** | C0-A WWDet (C0 encoder 64 ch × 8; decoder 1 × 1 on [h, raster] → 5 residual blocks, dilations 1–16, width 72 → 1 × 1), retrained from scratch; L1 | 593,577 |
| **B WW-FM** | WW family + x_t + time: C0 encoder on PPG; 1 × 1 on [h, raster, x_t] → 5 time-conditioned residual blocks (dilations 1–16) → 1 × 1 → 512-sample velocity. Decoder width 63 | 597,664 (−0.11% vs D) |
| **C SCALE-FM-INDEPENDENT** | Haar(x_t), Haar(PPG), Haar(raster) → three independent branches: coarse 128, mid 128, fine 256. Each branch: input [x_t, PPG, raster] coefficient of its scale → 1 × 1 stem → 6 time-conditioned residual blocks (dilations 1–32) → 1 × 1 head. No hidden feature crosses scales. Width 50 | 593,485 (−0.81% vs D) |
| **D SCALEFLOW-COUPLED** | As C, plus hierarchical coupling via 1 × 1 projections (16 channels) of the final branch features: coarse → mid input (same length); mid and coarse → fine input (nearest ×2 upsampling). No fine → coarse or fine → mid path. Width 50 | 598,333 (−0.28% vs 600k) |

- **Velocity reconstruction (C and D):** v = Haar⁻¹(v_c, v_m, v_f), one whole-window velocity.
- **Time conditioning (B, C, D):** a sinusoidal embedding (64-d, argument 1000 t), then Linear 64→128, GELU,
  Linear 128→128. The embedding is added through a per-block Linear projection after the first convolution of every
  residual block.
- **Parameter-matching rule:**
  - Depths are fixed structurally: WW-FM uses the 5 blocks of WW-DET; each branch uses 6 blocks.
  - Widths are searched over 16…160. D is closest to 600,000; C and B are closest to D's count.
  - All are within ±1% of D.
- **Haar transform (fixed, no parameters, orthonormal):**
  - low = (x[2k] + x[2k+1]) / √2, high = (x[2k] − x[2k+1]) / √2, applied twice;
  - coarse = low₂ (128), mid = high₂ (128), fine = high₁ (256);
  - applied to x_t, PPG and the raster;
  - reconstruction error < 1e-6.
- **Excluded:** beat stitching, event cells, global / local addition, event guard, peak suppression, attention, SSM,
  transformer, recurrence, auxiliary losses, adversarial training.

## 4. Flow matching and training

- **Path:** x₀ ~ N(0, I); t ~ U(0, 1); x_t = (1 − t) x₀ + t x₁; target u = x₁ − x₀.
- **Loss:** MSE(v_θ(x_t, t, PPG, raster), u), in waveform space (C and D invert the Haar transform **before** the loss).
  No other loss.
- **Training (all four):**
  - SF-TRAIN, seed 42;
  - 20,000 steps × 64 windows, epoch-permutation batches;
  - AdamW lr 1e-3, weight decay 0.01, clip 1.0;
  - last checkpoint, no early stopping;
  - FM noise and t from a CUDA generator seeded 42.
  - Normalization: the stored ECG / PPG units, as WW-L1.
- **Recorded:** time, GPU, peak memory, batch, optimizer, lr, parameters, sha256, NaN steps; FLOPs in the compute stage.

## 5. Inference

- **Solver:** Euler, **NFE = 8**, t = 0 → 1, steps at t = i / 8.
- **Noise:** one N(0, I) 512-vector per window, from `default_rng(int(sha256("patient:window:20261002")[:16], 16))`.
  The same noise is used by WW-FM, C and D.
- **Primary metrics** use exactly this one sample per window. No best-of-K, no median-of-K.
- **Secondary only, after the primary verdict:** NFE 4 / 16, and K = 16 samples (noise key suffix ":k1"…":k15", with
  k = 0 the primary).

## 6. Metrics (C0 / C0-A / R1 implementations)

**Waveform metrics:**

- FD (`kanflow_fd`);
- beat-aligned correlation on matched pairs (reference R ↔ the SF detector's events), restricted to pairs finite in
  all six evaluated waveforms (four arms + two shuffle arms);
- MAE, PCC, S4, S5, spectral deviation.

**Event metrics** (neurokit R on each waveform, one-to-one ±50 ms):

- patient-macro FP / window, precision, recall and F1 (pooled within patient, then averaged over patients — R1
  definition);
- pooled precision / recall / F1;
- RR-MAE and HR-MAE.

## 7. Controls

- **PPG-SHUFFLE (SCALEFLOW):** `default_rng(20261002).permutation` over the evaluated windows applied to the PPG
  windows. Noise, raster and model are kept.
- **EVENT-SHUFFLE (descriptive):** the same permutation applied to the rasters. PPG, noise and model are kept.

## 8. Statistics

- 2,000 patient-clustered bootstrap replicates, **seed 20261002**.
- All paired comparisons share the resamples.
- FD is recomputed inside every replicate on matched windows.
- Windows are never resampled independently.

## 9. Gates (SF-VAL; differences are left − right)

| gate | rule |
|---|---|
| G1 (context only) | WW-FM − WW-L1. **FM BETTER** if FD CI upper < 0, correlation CI lower > −0.02 and FP CI upper < +0.05. **NO BENEFIT** if FD CI lower ≥ 0 and correlation CI upper ≤ 0. Otherwise **MIXED** |
| **G2** | FD(SF) − FD(WW-FM): CI upper < 0 |
| **G3** | FD(SF) − FD(best baseline): CI upper < 0. The best baseline is the lower FD point estimate of WW-L1 and WW-FM |
| **G4** | corr(SF) − corr(best-correlation baseline among WW-L1 / WW-FM): CI lower > −0.02 |
| **G5** | FP(SF) − FP(WW-L1): CI upper < +0.05, **and** recall(SF) − recall(WW-L1): CI lower > −0.01 (patient-macro) |
| **G6** | FD(SF) − FD(IND): CI upper < 0, **and** corr(SF) − corr(IND): CI lower > −0.02 |
| **G7** | FD(PPG-shuffled SF) − FD(SF): CI lower > 0, **and** corr(shuffled) − corr(SF): CI upper < 0 |

- **QUALIFIED** = G2 ∧ G3 ∧ G4 ∧ G5 ∧ G6 ∧ G7.
- **If not qualified,** the failure categories are recorded (specification §47).
- **No rescue:** no other wavelet, backbone, NFE, loss, width, scale, direction or fusion.

## 10. Secondary analyses (never gates; only after `val_gates.json` is written)

- **NFE:** 4 / 8 / 16 for WW-FM and SF — FD, correlation, FP, recall, batch-1 GPU latency.
- **Stochastic diversity:** a fixed 2,000-window SF-VAL subset (salted rank `sf0-stoch-v1`), K = 16, for WW-FM, IND and
  SF:
  - within-window waveform diversity (pairwise RMS across samples);
  - beat-aligned diversity at placed events (`seed_diversity`);
  - generated / real beat-to-beat diversity ratio;
  - R-time seed SD (`timing_sd`);
  - K = 16 median-HR MAE vs single-sample HR MAE.
- **Scale diagnostics (descriptive):**
  - coarse, mid and fine band components of each waveform; band RMS errors vs the reference, overall and within
    ±10 samples of reference R;
  - PPG-shuffle and coupling (SF − IND) effects per band;
  - cumulative coarse → coarse + mid → full error of SF.
- **Compute:**
  - parameters and pipeline parameters (+ detector);
  - FLOPs per vector-field evaluation and total at NFE 8 (+ detector);
  - batch-1 GPU and CPU (4-thread) latency of the full pipeline;
  - training time and memory.

## 11. Old V1 TEST

**Access** is allowed only if all of these hold:

- SF0 qualifies on SF-VAL;
- a metadata-only freshness audit ends `VERDICT: CLEAN`;
- `final_test_freeze_manifest.json` is committed and unchanged, and hashes:
  - the split, the detector and all four checkpoints;
  - the code, including Haar, flow and the noise map;
  - the metric and bootstrap code;
  - the gates and the audit.

`load_test()` refuses otherwise.

**Audit rule, frozen now.** TEST is NOT CLEAN if committed records show V1 TEST outcomes that were:

- (a) computed for any model, detector or convention SF0 reuses: the RD1 / C0 detector family with its threshold /
  refractory / target convention, the WW-DET family, the C0 encoder; or
- (b) cited as motivation for a design choice or hypothesis of SF0 or of the closed line it follows.

Earlier-phase reporting of V1 TEST metrics is disclosed here as known.

**If opened:**

- Evaluated exactly once on all TEST windows: the same detector protocol, all four models, NFE 8, the same noise map,
  the same PPG-shuffle rule.
- G2 – G7 are repeated.

**TEST verdict:**

| verdict | rule |
|---|---|
| STRONG | all of G2 – G7 pass on TEST |
| FAILED | the SF-VAL gates failed, or the TEST G2 FD point estimate is ≥ 0 (major reversal) |
| PARTIAL | otherwise (SF-VAL qualified but some TEST gates fail) |

- STRONG allows only `docs/SF0_SCALEFLOW_PAPER_DRAFT_NOTES.md`.
- Otherwise: no rescue, and SF0 closes.

## 12. Tests and implementation

- **Code:** `scripts/sf0_scaleflow.py` (stages `split`, `audit`, `manifest`, `train_detector`, `train_wwl1`,
  `train_wwfm`, `train_ind`, `train_sf`, `evaluate_val`, `nfe`, `stochastic`, `compute`, `figure_val`, `freeze`,
  `evaluate_test`) and `src/ppg2ecg/scaleflow/model.py`.
- **Tests:** `tests/test_sf0_scaleflow.py`, 16 tests covering the 28 required items:
  - split (counts, disjointness, exclusions, determinism);
  - detector on SF-TRAIN and one shared raster;
  - Haar (lengths, inverse < 1e-6, no parameters, orthonormality);
  - FM path, target and t range;
  - WW-FM without multiresolution;
  - independent with no cross-scale flow (perturbation test);
  - coupled coarse → mid → fine only (perturbation test);
  - inverse before the waveform loss, and the same objective for B / C / D;
  - no auxiliary loss, renderer or guard;
  - deterministic noise shared by the FM arms;
  - Euler with exactly 8 evaluations;
  - count-only parameter selection;
  - TEST seal;
  - clustered bootstrap with seed 20261002;
  - PPG shuffle keeps the other inputs;
  - gate and G1 rules.
- **Synthetic dry run:** every stage from detector training to the figure ran on synthetic data in a scratch directory
  (30 training steps, an FD bootstrap stub). It found and fixed NaN handling in the figure and table.

## 13. Claim boundaries

- **Never claimed:** novelty / first / state of the art; calibrated uncertainty; clinical validity; patient-specific
  morphology; external or multi-seed robustness; anything about TEST unless it is legitimately opened.
- **"PPG-conditioned morphology modelling"** is claimed only if G7 passes.
- **Report:** `docs/SF0_SCALEFLOW_FEASIBILITY_REPORT.md` (28 sections).
- **HARD STOP.**
