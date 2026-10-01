# C0-A — CoherentBeat Architecture-Attribution Ablation Pack — PREREGISTRATION (frozen)

**Status: FROZEN (2026-10-01).** This file is committed and pushed with the C0-A implementation **before any C0-A model
is trained and before any C0-A metric exists**. It is never edited afterwards; amendments are dated and separate.
Hashes: `artifacts/c0a_coherentbeat_ablation/prereg_manifest.json`. Audit: `artifacts/c0a_coherentbeat_ablation/audit.md`.

## 0. Purpose and evidence status

- **C0 stays as it is.** C0 (`3b6ea98` / `dddaa4b` / `f9e25b5`) is complete, frozen and STRONG, and is not modified or
  retrained.
- **Three explanations C0 left confounded:**
  - **capacity:** C0 has 1.23 × BF0-DET's parameters;
  - **time-varying global field vs any shared absolute context:** LOCAL-ONLY removed both together;
  - **global-local decomposition vs window-level event-conditioned modelling.**
- **What C0-A is:** a preregistered architecture-attribution analysis on the frozen C0 split.
  - ARCH-VAL is development and qualification evidence.
  - The **previously opened** ARCH-HOLDOUT is a **frozen secondary replication population**. It is not fresh, not
    untouched, and not independent prospective validation for C0-A.
- **Not in C0-A:**
  - C1, stochastic latents, flow matching, diffusion, new losses;
  - old validation or test, WildPPG, other datasets;
  - tuning after outcomes, multiple seeds.

## 1. Frozen inputs

- **Split:** C0's ARCH-TRAIN / VAL / HOLDOUT (3,470 / 433 / 434 patients); the hashes are re-verified.
- **Detector and events:** C0's frozen timing detector and its stored event sequences, re-verified bit for bit at run
  time. **Every waveform arm uses the same events.**
- **Frozen reference arms:** BF0-DET-RETRAIN (482,049 parameters), C0-LOCAL-ONLY (588,545) and CoherentBeat-C0
  (592,770), all re-rendered with C0's functions. The run **stops** unless:
  - the renders equal C0's stored renders exactly;
  - the frozen arms' event, waveform and FD metrics reproduce C0's stored results to 1e-9.

## 2. New ablation models (ARCH-TRAIN only, seed 42, last checkpoint, one run each)

| name | definition | params (vs C0) |
|---|---|---|
| **PM-BF0-DET** | BF0's deterministic BeatFlowNet with channel width **73**. All other BF0 settings unchanged: 166-sample absolute-level beat, L1, FiLM conditioning, BF0 renderer and conditioning, ARCH-TRAIN template. Width chosen from 32 … 128 by parameter count only | 599,445 (+1.13%) |
| **CONST-GLOBAL-LOCAL** | C0 with g(t) = a_global: one scalar per window from the window encoder mean (Linear 64→64 → GELU → Linear 64→1). Encoder, local residual branch, supports, bump and forward inherited from C0 unchanged. No time-varying global path | 592,770 (0.00%) |
| **WW-DET** | C0's encoder (64 channels, 8 blocks) → 1 × 1 conv on [features, event raster] → 5 residual blocks (dilations 1, 2, 4, 8, 16) → 1 × 1 head → 512 samples. Raster: max of Gaussians, σ = 20 ms (detector target convention). No decomposition, supports or stitching. Decoder width 72 and depth 5 chosen from {32 … 96 step 8} × {1 … 12} by parameter count only | 593,577 (+0.14%) |

**Training:**

- **PM-BF0:** the BF0-DET protocol: 20,000 × 256 ARCH-TRAIN beats at reference R, AdamW 1e-3 / 0.01, clip 1.0.
- **CONST and WW-DET:** the C0 protocol: full-window L1, 20,000 × 64 windows, AdamW 1e-3 / 0.01, clip 1.0; training
  events (or raster) at reference R.
- **Not done:** early stopping, per-model learning-rate tuning, or choosing among seeds.

Models are parameter-matched, not compute-matched. Parameters, training time, peak memory, batch-1 GPU / CPU latency and
FLOPs (`torch.utils.flop_counter`) are reported.

## 3. Metrics and statistics (C0's implementations)

- **Events:** F1 (evaluable windows), precision, recall, FP and FN per window, RR-MAE, HR MAE.
- **Waveform:** FD (`kanflow_fd`), matched-pair beat-aligned correlation (pairs finite in all six waveform arms), MAE,
  PCC, S4, S5, spectral ratio deviation.
- **Coherence:** boundary-near FP for C0 and CONST (±2 samples of a support boundary).
- **Statistics:**
  - patient-clustered bootstrap, 2,000 replicates, seed 20261001, equal patient weight;
  - paired comparisons use identical patient resamples;
  - FD is recomputed inside each replicate on the matched windows;
  - windows are never resampled independently.

## 4. Claims (differences are C0 − ablation unless stated)

| claim | definition | decision |
|---|---|---|
| **M1 capacity** (C0 vs PM-BF0) | FP CI < 0 **and** FD CI < 0 **and** correlation lower CI > −0.02 **and** C0 inside the placed-event rhythm margins (ΔF1 lower > −0.02, ΔRR upper < +2 ms) | SUPPORTED / NOT SUPPORTED |
| **M2 time-varying field** (C0 vs CONST) | FD CI < 0 **and** no meaningful event / morphology disadvantage (F1 lower CI ≤ −0.02 or correlation lower CI ≤ −0.02 counts as a disadvantage). FP, MAE and whether each CI excludes 0 are reported | SUPPORTED / NOT SUPPORTED |
| **M3 decomposition** (C0 vs WW-DET) | STRONG: FP CI < 0 **and** FD CI < 0, correlation non-inferior (−0.02). PARTIAL: one of FP / FD has CI < 0, the other's CI not entirely > 0, correlation non-inferior. Otherwise NOT SUPPORTED | STRONG / PARTIAL / NOT SUPPORTED |
| **shared absolute context carrier** (CONST − LOCAL-ONLY) | FD CI < 0 **and** FP CI < 0 | SUPPORTED / NOT SUPPORTED |

**Notes on the claims:**

- M1 SUPPORTED means the capacity-only explanation is disfavored, not that capacity has no effect.
- M2 NOT SUPPORTED means the claim is restated as "shared absolute context carrier + compact local residuals". In that
  case the spline is not headlined.
- M3 NOT SUPPORTED means the decomposition is not claimed as the mechanism; window-level event-conditioned modelling may
  explain the gain.
- LOCAL-ONLY is context only. It is never used as proof of the time-varying field.

## 5. Order

1. ARCH-VAL: all claims, recorded as `claim_matrix_val.json`.
2. Freeze: new checkpoints, configs, C0-A and C0 code, metric and bootstrap code → `holdout_ablation_freeze.json`.
3. Commit and push.
4. **Only then**, and once: the previously opened ARCH-HOLDOUT with the same code. The C0-A loader refuses the holdout
   unless that freeze is committed and unchanged.

Nothing is modified after either evaluation.

## 6. Replication and final matrix

**Primary effects per claim:** M1 (FP, FD); M2 (FD); M3 (FP, FD); carrier (FD, FP).

| class | rule |
|---|---|
| REPLICATED | every primary point estimate has the same sign on ARCH-VAL and ARCH-HOLDOUT, and the decision is the same |
| DIRECTIONALLY CONSISTENT | same signs, but the holdout decision differs (its CI crosses the claim boundary) |
| NOT REPLICATED | any primary point estimate reverses sign |

**Final matrix** (from the ARCH-VAL decisions; any claim whose replication is NOT REPLICATED is downgraded):

| entry | rule |
|---|---|
| capacity-only explanation | DISFAVORED if M1 is SUPPORTED, else REMAINS PLAUSIBLE |
| shared absolute context carrier | carrier decision |
| time-varying global field | M2 decision |
| explicit global-local decomposition | M3: STRONG → SUPPORTED, PARTIAL → PARTIAL, otherwise NOT SUPPORTED |

VAL and HOLDOUT are never pooled after the fact.

**Case reading:**

| case | condition |
|---|---|
| A | M1, M2 and M3 (STRONG or PARTIAL) all supported |
| B | M1 and M3 supported, M2 not |
| C | M3 not supported |
| D | M1 not supported |
| E | WW-DET clearly dominates: architecture escalation stops |

## 7. C1 rule and hard stop

- **C1 GO** (experimental design only) **iff** both hold:
  - M1 is SUPPORTED and M3 is STRONG or PARTIAL on ARCH-VAL;
  - neither M1 nor M3 is NOT REPLICATED on ARCH-HOLDOUT.
- **Otherwise NO-GO.** M2 need not pass; without it, C1 would use the simplest supported shared-context carrier.
- **Not done in this run:** training C1, adding a latent, enlarging CoherentBeat, adding losses, adding rescue
  ablations, opening old test or external data, or adding seeds.

## 8. Claim boundaries

- **Never claimed:**
  - that the holdout is fresh;
  - causality;
  - that parameter count has no effect;
  - that the spline is necessary (unless M2 supports it) or the decomposition necessary (unless M3 supports it);
  - stochastic generation or uncertainty;
  - patient-specific morphology;
  - cross-dataset or multi-seed robustness;
  - state of the art, or a first global-local architecture.
- **Permitted claims** follow the M1 / M2 / M3 / carrier outcomes exactly.

## 9. Implementation, tests, artifacts

- **Code:** `scripts/c0a_ablation.py` (stages `params`, `audit`, `manifest`, `train_pm`, `train_const`, `train_ww`,
  `evaluate_val`, `freeze`, `evaluate_holdout`, `compute`, `figure`) and `src/ppg2ecg/coherentbeat/ablation.py`.
- **Tests:** `tests/test_c0a_ablation.py`, 15 tests:
  - C0 split reuse with no old patients; frozen data, detector, events and preprocessing reused; one event sequence for
    every arm; C0-A holdout seal;
  - PM-BF0 family kept and chosen by parameter count; selection rule;
  - CONST exactly constant with no time-varying path, local branch equal to C0's;
  - WW-DET uses the raster, outputs 512 samples, has no local renderer, chosen by parameter count;
  - no stochastic path;
  - paired bootstrap and matched-window FD;
  - strict C0 reproduction check;
  - M1, M2, M3, carrier, replication, final matrix and C1 rules.
- **Artifacts** (`artifacts/c0a_coherentbeat_ablation/`):
  - `audit.md`, `audit.json`
  - `prereg_manifest.json`, `parameter_match.json`, `model_configs.json`
  - `checkpoint_*.json`, `checkpoint_hashes.json`, `train_manifest.json`
  - `val_metrics.json`, `val_bootstrap.json`, `claim_matrix_val.json`
  - `holdout_ablation_freeze.json`, `holdout_metrics.json`, `holdout_bootstrap.json`, `claim_matrix_holdout.json`
  - `replication_summary.json`, `compute_accounting.json`
  - `figure.png`, `table.csv`
- **Not committed:** checkpoints, kept in `outputs/c0a_coherentbeat_ablation/`.
- **Report:** `docs/C0A_COHERENTBEAT_ARCHITECTURE_ABLATION_REPORT.md` (20 sections).
