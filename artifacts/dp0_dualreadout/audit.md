# DP0 conditioning-path audit (before any DP0 training or DP-DEV outcome)

## WW-L1 (SPECIALIST P; C0-A WWDet 72 x 5, 593,577 parameters)
- PPG [B, 512] -> C0 encoder: stem 1x1 (1 -> 64) and eight residual blocks (64 channels, kernel 5, dilations 1, 2, 4, 8, 16,
  32, 1, 2; GELU) at the full 512-sample resolution -> h [B, 64, 512] (328,832 parameters).
- The event raster [B, 512] is concatenated AFTER the encoder: 1x1 on [h, raster] (65 -> 72) -> five residual blocks
  (72 channels, dilations 1, 2, 4, 8, 16) -> 1x1 -> ECG (264,745 parameters).
- PPG conditioning is a deep learned single-resolution encoder; event conditioning is a raw raster channel at the decoder input.

## SCALEFLOW-COUPLED (SPECIALIST G; ScaleFM(50, coupled), 598,333 parameters)
- No learned condition encoder. Haar(x_t), Haar(PPG), Haar(raster) are computed by the fixed orthonormal two-level Haar
  transform (coarse 128 / mid 128 / fine 256) and stacked as three raw channels at each branch input.
- Each branch (width 50, six time-conditioned residual blocks, dilations 1..32) processes the noisy ECG and the condition
  jointly from its first 1x1 stem; coarse -> mid and coarse / mid -> fine 1x1 projections (16 channels each).
- PPG / event conditioning is therefore entangled with x_t processing at every scale; there is no separable conditioning
  sub-network to share.

## Consequence for DP0
- The only learned PPG conditioning path in either specialist is the WW-L1 encoder. The DP0 shared encoder is that family,
  with the event raster moved into its input (stem 2 -> 64) so that H = E(PPG, raster) carries all condition information.
- Multiscale features: the closest exact resolutions naturally supported by the repository are those of the fixed Haar
  transform: H_512 (encoder output), H_256 = Haar low-pass of H_512, H_128 = Haar low-pass of H_256 (parameter-free).
  Coarse / mid flow branches read H_128, the fine branch reads H_256, the point decoder reads H_512.
- Natural encoder stages: E0 stem (1x1), E1 blocks 1-6 (first dilation cycle 1..32; receptive field grows from 1 to 505
  samples, i.e. the whole 512-sample window), E2 blocks 7-8 (second cycle 1, 2; refinement at full context). There are no
  further natural boundaries (a half-depth split after block 4 falls inside the first dilation cycle).
- x_t never enters the shared encoder; it is private to the flow decoder.
