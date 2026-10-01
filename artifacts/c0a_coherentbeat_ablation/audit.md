# C0-A audit (2026-10-01) — before any C0-A training or metric

## 1. Frozen C0 references (`audit.json`)

- **C0 commits:** prereg `3b6ea98`, holdout freeze `dddaa4b`, result `f9e25b5`. Verdict: STRONG. Nothing from C0 is
  modified. The C0 model file and run script are in C0's freeze manifest, so the new models live in a new module,
  `src/ppg2ecg/coherentbeat/ablation.py`, which imports and subclasses the C0 modules.
- **Checkpoints:** detector, BF0-DET-RETRAIN, C0 and C0-LOCAL-ONLY have sha256 equal to C0's `checkpoint_hashes.json`.
- **Split:** C0's split hashes recomputed from `split_manifest.json` are identical. ARCH-TRAIN 3,470 / ARCH-VAL 433 /
  ARCH-HOLDOUT 434 patients, with no old validation or test patient.
- **Events:**
  - The stored C0 ARCH-VAL events (`outputs/c0_coherentbeat/val_renders.npz`) are reproduced exactly by re-running the
    frozen detector (28,649 / 28,649 windows).
  - Every C0-A arm uses these events.
  - The holdout events are re-verified the same way at holdout time.
- **Frozen arms re-rendered at evaluation:** BF0-DET, LOCAL-ONLY and C0 are re-rendered with C0's own functions. The run
  **stops** unless:
  1. the renders equal C0's stored float32 renders exactly;
  2. the event, waveform and FD metrics of PLACED, BF0, LOCAL and C0 reproduce C0's stored results to 1e-9.
  The beat-aligned correlation population covers six arms here and three in C0. Any deviation is reported, not gated.

## 2. Parameter matching (`parameter_match.json`; parameter count is the only criterion)

| model | grid | selected | params | vs C0 592,770 |
|---|---|---|---|---|
| PM-BF0-DET | BeatFlowNet channel width 32 … 128, everything else BF0 | **ch = 73** | 599,445 | +1.13% (within ±2%) |
| WW-DET | decoder width {32 … 96 step 8} × depth 1 … 12 on the C0 encoder | **dec_ch = 72, 5 blocks** | 593,577 | +0.14% (within ±5%) |
| CONST-GLOBAL-LOCAL | none (C0 with the spline replaced) | — | 592,770 | 0.00% |

- **PM-BF0 neighbour:** ch = 72 gives 585,761 (−1.18%). Only the channel width changes; BF0's dilations, kernel, FiLM
  conditioning, segment, renderer and conditioning are unchanged.
- **CONST:** its scalar head (Linear 64→64 → GELU → Linear 64→1 on the window encoder mean) has exactly the layers of
  C0's spline-coefficient head. A 1 × 1 conv over 17 positions is the same linear map, so the counts are identical with
  no padding. The encoder, local residual branch, supports, bump and `forward` are inherited from C0 unchanged; only
  `global_field` is overridden.

## 3. Design decisions the specification leaves open (frozen in the preregistration)

1. **WW-DET.**
   - Encoder: C0's (64 channels, 8 dilated residual blocks, unchanged).
   - Decoder: a 1 × 1 conv on [encoder features, event raster], then 5 residual blocks with dilations 1, 2, 4, 8, 16,
     then a 1 × 1 head giving 512 samples.
   - The raster is a max of Gaussians with σ = 20 ms at the window's events, the RD1 / C0 detector target convention.
   - There is no additive decomposition, no compact support and no stitching.
2. **Training events.**
   - CONST and WW-DET train like C0: full-window L1, 20,000 × 64 windows, AdamW 1e-3 / 0.01, clip 1.0, seed 42; events
     (or raster) at reference R.
   - PM-BF0 trains like BF0-DET: 20,000 × 256 beats.
   - Every arm is evaluated at C0's frozen events.
3. **M2's "meaningful event / morphology disadvantage":** F1(C0 − CONST) lower CI ≤ −0.02, or correlation lower
   CI ≤ −0.02. These are the C0 G1 and G4 margins.
4. **M3 PARTIAL:** exactly one of FP and FD is lower (CI < 0), the other is not significantly worse (its CI does not lie
   entirely above 0), and morphology is non-inferior.
5. **"Shared absolute context carrier":** SUPPORTED if CONST − LOCAL-ONLY has a CI below 0 for both FD and FP.
6. **Replication:** see the preregistration, §6.

## 4. Evidence status

ARCH-HOLDOUT was opened once by C0, so it is not fresh for C0-A. C0-A's holdout analysis is a frozen secondary
replication of preregistered ablation comparisons on that previously opened population. The primary prospective C0
result stays C0's own.

## 5. Synthetic dry run

A synthetic C0 world was built in a scratch directory, then every C0-A stage ran end to end, including the C0-A holdout
seal (`PermissionError` before the freeze) and a forced holdout path. The numbers mean nothing.

Fixed before the commit:
- figure panel F now draws CIs as vertical lines, because a percentile CI can exclude the point estimate;
- shorter panel labels;
- a NaN-safe check that the frozen C0 results are reproduced.
