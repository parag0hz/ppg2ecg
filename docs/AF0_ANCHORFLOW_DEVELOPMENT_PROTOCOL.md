# AF0 — AnchorFlow-ECG — DEVELOPMENT PROTOCOL (frozen)

**Status: FROZEN (2026-10-02).** Committed and pushed with the split and the implementation **before any AF-DEV metric,
before the detector, anchor or any flow is trained**. It is never edited afterwards; deviations are dated amendments.
Hashes: `artifacts/af0_anchorflow/protocol_manifest.json`.

## 0. What kind of study this is

- **AF0 is adaptive development, not a single-shot confirmatory experiment.** On **AF-DEV** the proposed model may be
  changed repeatedly within the bounded, predeclared search space of §6, and every result is kept. AF-DEV results are
  development evidence only.
- **AF-LOCK** (337 patients) is the **only locked confirmation population in AF0**. It is opened once, for exactly one
  frozen development winner, after a committed freeze. No search happens after it.
- **The old V1 TEST** (1,156 patients) stays **closed for all of AF0**, even if AF-LOCK confirms. A separate AF1
  preregistration would be needed.
- **Closed lines are untouched:** BF0, D0, C0, C0-A, E0, R1, SF0.
- **No novelty claim** (first, novel, state of the art).

## 1. Data (`split_manifest.json`, `split_hashes.json`; created before this commit)

- **Source and rule:** SF-TRAIN (3,037 patients), permuted with `default_rng(20261002)`:
  - [0:300] → **AF-DEV** (300 patients, 20,319 windows);
  - [300:637] → **AF-LOCK** (337, 22,133);
  - the rest → **AF-TRAIN** (2,400, 159,545).
- **Checks:**
  - the roles are disjoint and their union is SF-TRAIN;
  - no SF-VAL, ARCH-VAL or ARCH-HOLDOUT patient;
  - no old V1 validation or test patient.
- **Evidence status:** these patients trained earlier project models (C0 / C0-A / R1 / SF0), so AF-DEV and AF-LOCK are
  new to AF0 but not project-naive.
- **Reference R:** the RD1 neurokit cache.

## 2. Frozen components (trained once on AF-TRAIN, seed 42, then frozen)

- **Timing detector:**
  - RD1 / C0 RhythmTCN protocol (14,000 × 64, BCE to a Gaussian σ = 20 ms target);
  - events at threshold 0.35, refractory 32;
  - one event raster (Gaussian, σ = 20 ms) per window, shared by every model.
- **Anchor AF-WW-ANCHOR:**
  - architecture: SF0's WW-L1 (C0-A WWDet 72 × 5, 593,577 parameters);
  - training: L1, 20,000 × 64, AdamW 1e-3 / 0.01, clip 1.0; reference-R raster in training (SF0 protocol);
  - frozen after training (sha256 in `anchor_checkpoint_hash.json`); never updated during the search, with no joint
    fine-tuning.
- **Anchor outputs:** μ is computed with the frozen detector's events, on AF-TRAIN (for residual training; in-sample
  detector) and on AF-DEV.
- **Residual:** r = x − μ (stop-gradient by construction: μ is a stored array).
- **Residual statistics:** from AF-TRAIN residuals only (`residual_stats.json`):
  - A0: per-band mean / std (coarse / mid / fine Haar coefficients);
  - A1: per-band median / IQR;
  - A2: one global mean / std;
  - "waveform": global mean / std, for the vanilla baseline.
- **Haar transform:** SF0's fixed two-level orthonormal Haar (128 / 128 / 256; inverse error < 1e-6).

## 3. Models (residual generators about 600k parameters by count; anchor reported separately)

| id | model | params |
|---|---|---|
| B0 | AF-WW-ANCHOR (point estimate μ) | 593,577 |
| B1 | FULL-SIGNAL-SCALEFLOW: SF0 SCALEFLOW-COUPLED retrained on AF-TRAIN with the SF0 protocol | 598,333 |
| B2 | ANCHOR + VANILLA RESIDUAL FM: waveform-domain residual (global normalization). C0 encoder on PPG; 1 × 1 on [h, raster, μ, y_t]; 5 time-conditioned residual blocks | 597,727 |
| B3 | ANCHOR + MULTISCALE INDEPENDENT RESIDUAL FM: A0-normalized Haar residual; per-scale branches with input [z_s, PPG_s, event_s, μ_s], 6 blocks; no cross-scale features | 593,635 |
| B4 / c01 | ANCHORFLOW-SCALECOUPLED (base): as B3 plus coupling C1 (16-channel projections of coarse → mid, mid and coarse → fine, concatenated at the receiving stem; ×2 nearest upsampling); anchor injection B0 (concat μ_s); A0 normalization | 598,483 |

**Flow matching** (B1 – B4):

- y₀ ~ N(0, I), t ~ U(0, 1), y_t = (1 − t) y₀ + t y₁, target u = y₁ − y₀, MSE.
- y₁ is the normalized residual coefficients (B1: the full ECG).
- No auxiliary loss.
- Training: AF-TRAIN only; 20,000 × 64; AdamW 1e-3 / 0.01; clip 1.0; seed 42; last checkpoint.
- Inference: Euler (NFE 8 unless stage E).
- Noise: one N(0, I) 512-vector per window, `scaleflow.window_noise` (sha256 of "patient:window:20261002"; ":k{k}" for
  k = 1 … 15), shared across B1 – B4.

**Generated ECG:** x = μ + decode(y).

**Two outputs:**

- the **point output** is μ;
- the **conditional samples** are μ + r⁽ᵏ⁾.

A single stochastic sample is never scored as the point estimator.

## 4. Metrics (AF-DEV; same code on AF-LOCK)

- **Point estimate (anchor only):**
  - beat-aligned correlation on matched pairs (reference R ↔ detector events), as a per-window mean;
  - patient-macro FP / window, recall, precision and F1;
  - RR-MAE, HR-MAE, MAE.
- **Population generative metrics** (one sample per window):
  - FD (`kanflow_fd`) of x vs ECG;
  - residual FD of r_gen = x − μ vs r_real = ECG − μ;
  - diversity ratio = std(r_gen) / std(r_real);
  - per-band residual mean / variance / RMS;
  - residual spectral discrepancy (mean |log PSD ratio|).
- **K = 16 conditional set** (2,000 AF-DEV windows, salted rank `af0-k16-v1`):
  - sample mean and median waveforms;
  - within-condition diversity, beat-aligned diversity, per-band diversity;
  - center − anchor distance;
  - R-time seed SD;
  - event counts;
  - **mean16 vs the anchor on the same windows:** correlation, MAE, FP, recall.
- **Condition controls** (B4; permutation `default_rng(20261002)` over windows):
  - **S1:** PPG permuted inside the residual-flow condition only; μ kept.
  - **S2:** PPG permuted for the whole pipeline; μ recomputed.
  - **Anchor shuffle:** μ permuted in the flow condition; PPG kept, the output still adds the true μ.

## 5. Statistics and gates

**Statistics:**

- 2,000 patient-clustered bootstrap replicates, seed 20261002, paired.
- FD differences are recomputed inside every replicate. They use the exact sufficient-statistics bootstrap of
  `kanflow_fd` (`anchorflow/fastfd.py`):
  - point estimates come from `kanflow_fd` itself;
  - one replicate and the full-set FD are re-checked against `kanflow_fd` at run time (relative tolerance 1e-6).

**Development gates (all required):**

| gate | rule |
|---|---|
| D1 | FD(cand) − FD(anchor) CI upper < 0 **and** FD(cand) − FD(B1) CI upper < +1.0 |
| D2 | corr(mean16) − corr(anchor) CI lower > −0.02 |
| D3 | FP(mean16) − FP(anchor) CI upper < +0.05 **and** recall(mean16) − recall(anchor) CI lower > −0.01 |
| D4 | residual FD(cand) − residual FD(B2) CI upper < 0 |
| D5 | residual FD(cand) − residual FD(B3) CI upper < 0 |
| D6 | residual FD(S1) − residual FD(cand) CI lower > 0 **and** residual FD(anchor shuffle) − residual FD(cand) CI lower > 0 |
| D7 | 0.5 ≤ diversity ratio ≤ 1.5 |

## 6. Bounded adaptive search (AF-DEV only)

**Budget:** at most **10 proposed configurations** (c01…c10, excluding B0 – B3) and at most **30 training jobs** in
total (detector, anchor, B1 – B3 and candidates count; ledger in `training_jobs.json`).

**Search space** (one major factor changed at a time, in this mechanistic order):

1. **A — normalization:** A0 / A1 / A2.
2. **B — anchor injection:** B0 concat / B1 FiLM / B2 concat + FiLM.
3. **C — coupling:**
   - C0: additive projected features;
   - C1: concatenative (base);
   - C2: gated additive, with learned channel-wise sigmoid gates initialized at 0.
4. **D — width multiplier:** 0.75 / 1.0 / 1.25 of the parameter-matched width (≤ 1.5 × base).
5. **F — residual amplitude:** F0 none / F1 global learned α / F2 per-band learned α. Implemented only if invoked.
6. **G — PPG-condition dropout:** 0 / 0.05 / 0.10 at training. Only if the flow ignores PPG.
7. **E — NFE:** 4 / 8 / 16, Euler.
   - Evaluated only for a configuration that already passes D1 – D7 at NFE 8. The lowest passing NFE is kept.
   - NFE cannot rescue a failing architecture.

**Width rule:** every structural variant is re-matched to about 600k parameters by count (width 16–160). Stage D then
multiplies that width.

**Per iteration:** hypothesis, exact change, train, AF-DEV evaluation, all metrics. The next move is decided on AF-DEV
only. Nothing is deleted. The scoreboard is `search_history.csv`.

**Not allowed:** transformer / S5 / Mamba / attention / cross-attention, adversarial or perceptual losses, FD /
morphology / event / peak / HR / spectral losses, learned wavelets, other datasets, TEST or AF-LOCK feedback, metric
redefinition.

## 7. Winner, freeze, AF-LOCK

**Stopping:** the search stops at the first configuration that passes D1 – D7. If several pass in the same round, the
winner is chosen lexicographically:

1. lowest residual FD;
2. lowest FD;
3. highest mean16 correlation;
4. lowest NFE;
5. fewest parameters.

**Development failure:** no winner within the budget means **AF0 DEVELOPMENT FAILED**; AF-LOCK stays closed.

**Freeze:** `lock_freeze_manifest.json` hashes:

- the detector, the anchor, the winner, and B1 – B3;
- the normalization and the configuration;
- the code (Haar, flow, noise map, K16 protocol, metrics, bootstrap, controls);
- the split.

It is committed and pushed. `load_role("af_lock")` refuses until it is committed and unchanged.

**AF-LOCK:**

- Evaluated once: the winner and the frozen B0 – B3.
- No retraining, threshold or NFE change.
- D1 – D7 are repeated. **CONFIRMED** only if all pass; otherwise FAILED, with no rescue and no return to AF-DEV.

## 8. After AF-LOCK

**If confirmed:**

- Training-seed robustness (seeds 42 / 43 / 44, the same architecture), reported on AF-DEV only, mean ± SD.
- `docs/AF1_FINAL_TEST_PREREGISTRATION_DRAFT.md`, not run.
- `docs/ANCHORFLOW_PAPER_METHOD_DRAFT.md`.

**In every case:** the old TEST stays closed.

## 9. Claim boundaries

- **Allowed if supported:** "AnchorFlow separates deterministic paired reconstruction from stochastic residual
  generation and uses scale-coupled conditional transport to model the remaining ECG residual distribution."
- **Never claimed:** first / novel / state of the art; clinical validity; calibrated uncertainty; exact patient-specific
  morphology; external generalization; final TEST performance.

## 10. Tests and implementation

- **Code:**
  - `scripts/af0_anchorflow.py` (stages `split`, `audit`, `manifest`, `train_detector`, `train_anchor`, `prep`,
    `train <job>`, `eval_baselines`, `eval_cand <id>`, `nfe <id>`, `winner <id>`, `freeze <id>`, `eval_lock`);
  - `src/ppg2ecg/anchorflow/{model,fastfd}.py`.
- **Tests:** `tests/test_af0_anchorflow.py`, 19 tests covering the 32 required items:
  - split (counts, disjointness, exclusions, determinism);
  - the AF-LOCK seal, and no TEST or val / holdout loader;
  - detector and anchor trained on AF-TRAIN;
  - the frozen anchor and residual = x − μ;
  - TRAIN-only statistics and an exact normalizer round trip;
  - Haar, the flow target, and output = μ + residual;
  - vanilla without Haar; independent with no coupling; the coupled topology (C0 / C1 / C2, no fine → coarse);
  - μ in the condition (B0 / B1 / B2 and vanilla);
  - no prohibited loss or block;
  - deterministic noise, K16 and mean;
  - residual FD on residuals;
  - S1 / S2 / anchor-shuffle definitions;
  - budget enforcement and AF-DEV-only search;
  - an exact fast-FD bootstrap;
  - the gate rules.
- **Synthetic dry run:** all stages from detector training to candidate evaluation ran in a scratch directory. The
  numbers mean nothing.
