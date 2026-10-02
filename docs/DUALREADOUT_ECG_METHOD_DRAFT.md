# DualReadout-ECG — method draft

**Status: draft for paper development, written after DP0 CONFIRMED** (`docs/DP0_DUALREADOUT_REPORT.md`).
- It describes only what DP0 tested.
- No novelty or priority language is used. A current literature review is required before positioning against prior
  multi-task, dual-head or reconstruction / generation work.
- The final test (DP1) has not been run.

## Motivation

- PPG-to-ECG translation is used in two different ways:
  - a single paired representative ECG for a given PPG segment (paired estimation);
  - a set of plausible ECGs for that segment (conditional generation).
- **Earlier experiments in this project separated the two behaviours:**
  - deterministic whole-window regression (WW-L1) gave better paired morphology and fewer false R detections;
  - multiresolution conditional flow matching (ScaleFlow) gave much better distributional fidelity (FD), but its
    single samples had poorer paired morphology and more false R.
- **Forcing one output to do both failed:** adding a stochastic residual to the deterministic estimate (AnchorFlow)
  did not keep the sample centre at the estimate.
- **DualReadout keeps one shared condition representation** and gives each use its own readout.

## Shared representation

H = E(c), c = (PPG window, event raster).

- **Event raster:**
  - events come from a frozen PPG timing detector (threshold 0.35, refractory 32 samples at 128 Hz);
  - they are rendered as Gaussians with σ = 20 ms;
  - in training, the reference R peaks are rendered the same way.
- **E (in the confirmed configuration):**
  - a 1 × 1 stem on [PPG, raster] (2 → 64 channels);
  - six residual dilated-convolution blocks (64 channels, kernel 5, dilations 1, 2, 4, 8, 16, 32; receptive field 505
    of 512 samples).
- **The noisy ECG x_t never enters E.**

## Point readout

μ = D_point(H).

- **Task-specific part:**
  - adapter: 1 × 1 conv + GELU;
  - two residual blocks (dilations 1, 2);
  - a decoder: 1 × 1 to width 71, five residual blocks (dilations 1, 2, 4, 8, 16), then 1 × 1 to one channel.
- **Training:** L1 against the ECG window.
- **Inputs:** no noise and no time input. The output is deterministic.

## Generative readout

v_θ = D_flow(x_t, t, H).

- **Task-specific part:** an adapter (1 × 1 conv + GELU) and two residual blocks (dilations 1, 2), giving H_512.
- **Multiscale condition:** H_256 and H_128 come from H_512 by the fixed orthonormal Haar low-pass.
- **Vector field:** a coupled multiresolution field on the fixed two-level Haar decomposition of x_t (coarse 128, mid
  128, fine 256):
  - three branches of six time-conditioned residual blocks (width 30);
  - branch inputs [Haar_s(x_t), H_s] (coarse and mid read H_128, fine reads H_256);
  - coarse → mid and coarse / mid → fine 1 × 1 projections (16 channels);
  - the inverse Haar gives one 512-sample velocity.
- **Objective:** linear-path flow matching: x_0 ~ N(0, I), x_t = (1 − t) x_0 + t x_1, target u = x_1 − x_0, MSE.
- **Samples are generated from noise by the flow alone.** They are never formed as μ + residual.

## Training

- **Round-robin multi-objective optimization with one AdamW optimizer:** lr 1e-3, weight decay 0.01, gradient clip
  1.0, constant LR.
- **Schedule:**
  - each cycle performs one POINT update (L1) and one FLOW update (flow-matching MSE), in alternating order: odd cycles
    POINT → FLOW, even cycles FLOW → POINT;
  - 20,000 cycles in total, so each readout receives exactly as many updates as a standalone model.
- **Masking:** on each substep, only the shared parameters and that task's private parameters receive gradients. The
  other task's private parameters are left untouched, including weight decay.
- **Not used:** loss weights, adaptive task weighting or gradient surgery.

## Inference

- **Point mode:** one forward pass, E then D_point.
- **Generative mode:**
  - E runs once;
  - the flow-specific features are cached;
  - Euler sampling with 8 steps calls only D_flow.
- **Dual mode:**
  - the shared trunk runs once;
  - its output feeds the point branch and the cached flow condition.

## Architecture variants

Three preregistered sharing depths on the natural encoder stages (stem | blocks 1–6 | blocks 7–8). Every readout path
had the same architecture and parameter count in all three.

| variant | shared | parameters | development outcome (DP-DEV) |
|---|---|---|---|
| full | stem + blocks 1–8 | 861,196 | point readout exceeded the event-safety margin (+0.144 FP/window); FD margin narrowly exceeded |
| middle (selected) | stem + blocks 1–6 | 943,372 | all gates passed; confirmed on AF-LOCK |
| stem only | stem | 1,189,900 | performance gates passed; no parameter saving (0.17 %) |

## Efficiency

**Parameters:**
- the middle-sharing model has 943,372 waveform parameters;
- two separate specialists have 1,191,910 (WW-L1 593,577 + ScaleFlow 598,333);
- the saving is 20.85 %. The detector (328,897) is common to both pipelines and is not counted.

**Computation:**

| | separate specialists | middle-sharing model |
|---|---|---|
| FLOPs, both outputs, one window | 1.85 G | 1.16 G |
| FLOPs, generation only | 1.25 G | 0.81 G |
| GPU batch-1, both outputs | 13.3 ms | 11.1 ms |
| CPU 4 threads, both outputs | 29.3 ms | 23.9 ms |

- The computation gain comes mostly from encoding the condition once per generation instead of at every step.
- Training one dual model took about 15 % longer than training the two specialists sequentially.

## Supported mechanism (DP0 only)

- **One shared stem + first dilation cycle supports both readouts.** On a locked population (AF-LOCK, 337 patients):
  - the point readout was non-inferior to the point specialist: morphology −0.005, FP/window +0.042 against a +0.05
    margin, recall +0.001;
  - the generative readout was non-inferior to the generative specialist: FD −0.61 against a +1.0 margin;
  - the generative readout used the PPG condition: shuffling PPG raised FD by 11.5 and lowered sample correlation by
    0.33.
- **Sharing the full encoder harmed the point readout's event fidelity on the development population.** The mechanism
  is not established: shared-gradient cosines were similar for full and middle sharing.
- **Not shown:**
  - seed robustness;
  - calibrated uncertainty (the samples are under-dispersed and have no timing variability);
  - external or final-test performance;
  - any superiority over the specialists beyond non-inferiority.
