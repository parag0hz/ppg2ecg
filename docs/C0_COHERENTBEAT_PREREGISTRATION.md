# C0 — CoherentBeat Deterministic Feasibility — PREREGISTRATION (frozen)

**Status: FROZEN (2026-10-01).** This file is committed and pushed with the C0 split and implementation **before any C0
model is trained and before any ARCH-VAL outcome exists**. It is never edited afterwards; amendments are dated and
separate. Hashes: `artifacts/c0_coherentbeat/prereg_manifest.json`. Audit: `artifacts/c0_coherentbeat/audit.md`.

## 0. Scope

- **What C0 is.** A new, prospective, **deterministic** architecture experiment. It is not BF1 and not a D0
  continuation. BF0 (`504395d` / `d380e3b`, Case A) and D0 (`d85f3ef` / `a54ec7e`, NOT SUPPORTED) are closed.
- **Hypothesis.** A local beat generator should not independently determine the absolute ECG waveform context. A global
  continuous field carries baseline, slow morphology and shared context. Event-local branches add compact residuals that
  vanish smoothly (value and slope) at their support boundary.
- **Not in C0:**
  - stochastic latents, noise, flow matching, diffusion or hierarchical latents;
  - tangent projection or QRS-topology constraints;
  - crossfading between full beats;
  - auxiliary losses;
  - the old VitalDB validation or test split, WildPPG, or any other corpus;
  - starting C1.

## 1. Split (frozen; `split_manifest.json`, `split_hashes.json`)

- **Construction:** the V1 VitalDB TRAIN patient pool (4,337 patients), sorted and permuted with
  `numpy.random.default_rng(20261001)`.

| role | patients | windows | use |
|---|---|---|---|
| ARCH-TRAIN | 3,470 | 231,220 | every learned component |
| ARCH-VAL | 433 | 28,649 | Stage A qualification |
| ARCH-HOLDOUT | 434 | 28,531 | Stage B only, after a committed freeze |

- **Integrity:**
  - patient-level, with all cases of a patient kept together;
  - the overlap checks are all 0, including against the old validation (289) and test (1,156) patients;
  - the split is never reshuffled.
- **ARCH-HOLDOUT history:** these patients were training data of earlier project models and part of TRAIN-only
  diagnostics. No C0 component uses them, and no outcome on them informed C0. They are "untouched by C0", not "never
  seen".
- **Holdout seal in code:** `load_arch("holdout")` and `reference_peaks("holdout", …)` refuse to run unless all of these
  hold:
  - a holdout-freeze manifest exists and is committed to git;
  - every frozen file still matches its hash;
  - the ARCH-VAL verdict is QUALIFIED.

## 2. Components (all trained on ARCH-TRAIN only, seed 42, last checkpoint, no early stopping)

| component | definition |
|---|---|
| timing detector | RD1 `RhythmTCN`, retrained: Gaussian targets σ = 20 ms at reference R; BCE; 14,000 steps; batch 64 windows; AdamW lr 1e-3, weight decay 0.01. Events: `extract_events(0.35, refractory 32)`. **One event sequence per window, shared by every waveform arm.** No timing head |
| BF0-DET-RETRAIN | BF0's deterministic `BeatFlowNet` (x_t = 0, t = 0), L1 on R-aligned 166-sample ARCH-TRAIN beats; 20,000 steps × 256; AdamW 1e-3 / 0.01; clip 1.0. Rendered with BF0's renderer and conditioning at the placed events (template and singleton RR from ARCH-TRAIN) |
| CoherentBeat-C0 | x̂(t) = g(t) + Σ_i m_i(t − r_i) q_i(t − r_i) (§3) |
| C0-LOCAL-ONLY | the same network with g ≡ 0 (diagnostic ablation) |

**C0 / LOCAL-ONLY training:**

- full-window L1 (mean |x̂ − y| over 512 samples);
- events = the window's reference R peaks;
- 20,000 steps, batch 64 windows (RD1's window-level convention), AdamW 1e-3 / 0.01, clip 1.0;
- no other loss term.

## 3. CoherentBeat-C0 architecture (frozen)

- **Shared encoder.** A 1 × 1 stem (1 → 64), then 8 residual dilated conv blocks (kernel 5, dilations
  1, 2, 4, 8, 16, 32, 1, 2, GELU) over the 512-sample PPG.
- **Global field.** 17 coefficients from the encoder features averaged over ±16 samples around control points
  t = 0, 32, …, 512 (1 × 1 conv → GELU → 1 × 1 conv). g(t) = Σ_k c_k β₃((t − 32k)/32), a centred cardinal cubic B-spline
  with 250 ms spacing over the whole window. Deterministic and differentiable.
- **Local residual.**
  - For each event r_i: encoder features at τ = −38 … +57 samples (zero outside the window), a 1 × 1 conv, then four
    FiLM blocks (dilations 1, 2, 4, 8).
  - FiLM conditioning: an MLP on [window encoder mean, RR_prev, RR_next in seconds].
  - A 1 × 1 head gives q_i(τ).
- **Support, in physical time.**
  - L_i = min(300 ms = 38.4 samples, 0.45 · RR_prev) and R_i = min(450 ms = 57.6 samples, 0.45 · RR_next).
  - A missing neighbour uses the ARCH-TRAIN median reference RR (100 samples).
  - No phase normalization, warping or stretching: q_i lives on absolute offsets τ.
- **Envelope.** m_i(τ) = b(−τ/L_i) for τ < 0 and b(τ/R_i) for τ ≥ 0, with b(u) = exp(1 − 1/(1 − u²)) for u < 1 and 0
  otherwise. So m_i(0) = 1, and m_i and all its derivatives are 0 at −L_i and +R_i. Not tuned.
- **Final waveform.** x̂ = g + Σ local. No renormalization, no fill, no stitching of absolute-level beats. Outside the
  supports, x̂ = g.
- **Parameters.** C0 592,770 (1.23 × BF0-DET's 482,049; within the 1.5× limit); LOCAL-ONLY 588,545. The global-only
  output g of the trained C0 is reported descriptively.

## 4. Evaluation population and metrics

- **Population:** all windows of the role, with the same windows for the placed events, BF0-DET-RETRAIN, C0-LOCAL-ONLY
  and C0. Missingness is reported.
- **Reference R:** neurokit, from the RD1 cache sliced per role and re-verified on 200 salted windows of the role.
- **Events** (per window, neurokit on each render, ±50 ms one-to-one):
  - F1 (over evaluable windows), precision, recall;
  - FP rate (unmatched detections per window, patient mean) and FN rate;
  - RR-MAE, HR MAE.
- **Waveform:**
  - FD (`kanflow_fd`);
  - matched-pair beat-aligned correlation (reference R ↔ placed events within ±50 ms, both 83-sample windows inside;
    pairs finite in every arm of {BF0-DET, LOCAL, C0});
  - S4, S5;
  - MAE, PCC;
  - spectral ratio deviation (band mean).
- **Coherence (C0, LOCAL; descriptive):**
  - boundary-near FP: an FP within **±2 samples** of any support boundary r_i − L_i or r_i + R_i;
  - the local residual at the boundary;
  - the value jump and first-difference jump at boundaries, compared with the reference ECG at the same places;
  - global vs local RMS;
  - the QRS-region (±10 samples of placed events) energy share of the local part, and the non-QRS energy share of the
    global part, with g taken about its window mean.
- **Statistics:**
  - patient-clustered bootstrap, 2,000 replicates, seed 20261001, equal patient weight;
  - paired arms share the resamples;
  - FD is recomputed inside every replicate on the resampled windows, and every replicate must keep ≥ 3,000 windows;
  - windows are never resampled independently.

## 5. Stage A — ARCH-VAL qualification (frozen gates)

| gate | rule |
|---|---|
| G1 rhythm preservation | ΔF1 = F1(C0) − F1(placed events): lower CI > −0.02; ΔRR = RR-MAE(C0) − RR-MAE(placed): upper CI < +2 ms |
| G2 false-event improvement | FP rate(C0) − FP rate(BF0-DET-RETRAIN): CI entirely < 0 |
| G3 distributional improvement | FD(C0) − FD(BF0-DET-RETRAIN): CI entirely < 0 |
| G4 morphology non-inferiority | beat-aligned correlation(C0) − (BF0-DET-RETRAIN): lower CI > −0.02 |

**Verdict:**

- **QUALIFIED:** all four gates pass. Only this opens Stage B.
- **PARTIAL:** G1, G2 and G3 pass but G4 fails. **Stop; no holdout.**
- **FAILED:** G1, G2 or G3 fails. **Stop; no holdout.**

No redesign happens in this run. LOCAL-ONLY and global-only results are reported and never gate.

## 6. Stage B — ARCH-HOLDOUT (only if QUALIFIED)

1. **Freeze** checkpoints, detector, configs, preprocessing, metric and bootstrap code, thresholds and eligible-window
   logic in `holdout_freeze_manifest.json` (sha256).
2. **Commit and push** the freeze. Print `## C0 HOLDOUT FREEZE COMMIT`.
3. **Evaluate ARCH-HOLDOUT exactly once** with the same G1–G4. No training, checkpoint, threshold or metric change.
4. **Verdict:** STRONG (all pass), PARTIAL (G1–G3 pass, G4 fails), FAILED (any of G1–G3 fails).

STRONG permits only `docs/C1_SHARED_LATENT_DESIGN_DRAFT.md`, a design note. Its first stochastic comparison would be
independent beat noise vs shared window noise at matched noise dimension, NFE, data and compute. C1 is not trained.

## 7. Interpretation rules
- **Gate failures:**
  - G1 failure: the factorization does not preserve observable rhythm; the architecture line stops.
  - G2 failure: no event-artifact improvement over the retrained beat baseline; stop.
  - G3 failure: event or continuity gains are paid for with realism; stop.
- **LOCAL-ONLY:**
  - If LOCAL-ONLY ≈ C0, the global field is not claimed as the mechanism.
  - The g = 0 ablation also removes the window baseline (about −0.5 on this data), so a C0 advantage over LOCAL-ONLY
    cannot separate time-varying global context from a constant offset (audit §5).
- **Claims never made, even if STRONG:**
  - stochastic generation;
  - a complete PPG → ECG generative model;
  - calibrated uncertainty;
  - patient-specific morphology;
  - best method;
  - clinical validity;
  - cross-dataset generalization;
  - superiority to iMF or PENGUIN on all metrics;
  - a first global-local or coherent physiological generator.
- **After a failure:** the model size, spline basis, support, losses, stochasticity and data are not changed, and the
  old test split is not opened.

## 8. Tests, run order, artifacts

- **Tests:** `tests/test_c0_coherentbeat.py`, 16 synthetic and manifest-metadata tests:
  - split counts, disjointness and exclusion of old validation / test patients; determinism; window offsets;
  - holdout seal: no freeze manifest, failed qualification or unknown role all refuse; Stage A never loads the holdout;
  - spline shape and determinism;
  - bump centre / boundary / slope;
  - physical-time supports with 0.45 RR clipping;
  - edge RR from ARCH-TRAIN only;
  - local output equals the envelope exactly (no renormalization or warping) and is exactly zero outside support;
    x̂ = g there;
  - LOCAL-ONLY g = 0;
  - batch-size invariance and finite gradients;
  - parameter budget;
  - one shared event sequence from the retrained detector;
  - patient-clustered bootstrap;
  - the gate / verdict table;
  - boundary points and ±50 ms FP matching.
- **Run order** (`scripts/c0_coherentbeat.py`):
  1. `split`, `audit`, then this commit and push;
  2. `train_detector`, `train_bf0det`, `train_c0`, `train_localonly`;
  3. `evaluate_val`, `compute`, `figure`;
  4. ARCH-VAL result commit;
  5. only if QUALIFIED: `freeze`, commit, `evaluate_holdout`, `figure`, result commit;
  6. report `docs/C0_COHERENTBEAT_FEASIBILITY_REPORT.md`, then **HARD STOP**.
- **Artifacts** (`artifacts/c0_coherentbeat/`):
  - `audit.md`, `audit.json`
  - `split_manifest.json`, `split_hashes.json`
  - `prereg_manifest.json`, `input_hashes.json`
  - `model_config.json`, `timing_detector_config.json`, `training_manifest.json`
  - `checkpoint_*.json`, `checkpoint_hashes.json`
  - `val_metrics.json`, `val_bootstrap.json`, `qualification.json`
  - `boundary_metrics.json`, `decomposition_metrics.json`
  - `compute_accounting.json`
  - `figure.png`
  - if opened: `holdout_freeze_manifest.json`, `holdout_metrics.json`, `holdout_bootstrap.json`
- **Not committed:** checkpoints and renders, kept in `outputs/c0_coherentbeat/`.
