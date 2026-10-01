# R1 audit (2026-10-01) — before any R1 training or metric

## 1. Frozen references (`input_hashes.json`)

- **Checkpoints:** the timing detector, CoherentBeat-C0 and WW-DET match the sha256 recorded by C0 / C0-A.
- **Reuse:** the C0 split, ARCH-VAL stored events and renders, and the C0-A model configuration and metric files are
  reused unchanged. No C0, C0-A or E0 file is modified.

## 2. Timing detector (RD1 RhythmTCN, as frozen by C0)

| property | value |
|---|---|
| input | PPG [B, 1, 512] at 128 Hz |
| output | per-sample logits [B, 1, 512], fully convolutional |
| dense field | sigmoid(logits) on the ECG grid; no interpolation is needed |
| events | `extract_events(sigmoid, threshold 0.35, refractory 32)`, exactly C0's `detect_events` |
| training target | max of Gaussians at reference R, σ = 20 ms |
| training loss | `BCEWithLogitsLoss` |

These are the conventions R1 reuses for the SOFT input and for JOINT's rhythm target and loss.

## 3. WW-DET (C0-A)

- **Architecture:** C0 Encoder (64 ch, 8 blocks) → 1 × 1 conv on [features, raster] → 5 residual blocks (dilations
  1, 2, 4, 8, 16) → 1 × 1 head → 512 samples. 593,577 parameters.
- **Training:** 20,000 × 64 windows, AdamW 1e-3 / 0.01, clip 1.0, seed 42, raster at reference R.
- **R1 reuse:** this protocol and architecture family for both R1 models.

## 4. R1 models (`model_configs.json`)

| model | trainable params | relation to WW-DET |
|---|---|---|
| SOFT-RHYTHM-WW | 593,577 | the WW-DET class unchanged; conditioning channel = dense detector field |
| JOINT-RHYTHMFIELD-WW | 597,802 (waveform 593,577 + rhythm head 4,225) | +0.71% |

No width or depth was chosen from any outcome.

## 5. Decisions the specification leaves open (frozen in the preregistration)

- **SOFT training input:** the frozen detector's field on ARCH-TRAIN (inference only), the same input type as at
  evaluation. WW-DET, by contrast, was trained on the reference-R raster.
- **Rhythm head:** two 1 × 1 convolutions; no upsampling is needed.
- **Patient-macro event metrics:** pooled within patient, then averaged over patients.
- **Morphology correlation population:** pairs of reference R and placed events, finite in all four waveform arms.
- **Random event-free positions:** ≥ 150 ms from every reference R and placed event; 20,000 positions; seed 20261001.
- **TEST-verdict PARTIAL:** "narrowly" means within twice the margin. "Major recall degradation" means the T3 lower
  bound ≤ −0.01.
- **TEST freshness rule:** preregistration §9, written before the audit.

## 6. Synthetic dry run

Every stage from training through selection and the figure ran on synthetic data in a scratch directory, with 40
training steps and an FD bootstrap stub (the frozen `fd_diff_ci` needs ≥ 3,000 windows). The numbers mean nothing.

Before the commit, the model-config key clash (the rhythm-head description was overwritten by the parameter count) was
fixed.
