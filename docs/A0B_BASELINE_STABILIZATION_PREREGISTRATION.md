# A0-b Pre-registration — baseline checkpoint-selection stabilisation

Written 2026-08-25 **before** any A0-b training. Frozen at the commit that launches the run (recorded in
`outputs/a0b_penguin_otcfm_ppgdalia_8s_seed42/provenance.json`). Nothing below may be changed after results are seen.

## 1. Why
A0 (`docs/A0_PENGUIN_REPRODUCTION_REPORT.md`) selected its checkpoint by the *stochastic* validation MAE of one 50-NFE
sample per window per epoch; the metric fluctuated by ±0.1 between epochs, patience 10 fired at epoch 21 while the
training CFM loss was still decreasing. We must know whether the A0 50-NFE quality (HR error 10.99 bpm, template
correlation 0.662) is the OT-CFM + S5 ceiling on this protocol or an under-training artefact.

## 2. Identical to A0 (frozen)
PPG-DaLiA · 8 s @ 128 Hz (1024 samples) · preprocessing v0 (per-window filters/z-score/min-max) · processed data
`data/processed/v0_8s` (MANIFEST sha256 `57ce09d8…`) · split `split_p0_holdout_seed42.json` (sha256 `11c154e4…`; train 13,
val S11, test S2) · seed 42 · upstream PENGUIN model class unchanged (`6cd70cd`; h_dim 128, 4 blocks, S5 state 256,
mlp_ratio 2, 4,568,707 params) · PPG conditioning unchanged · OT-CFM objective unchanged (`t ~ U(0,1)`, `x_t = (1−t) z + t x₁`,
target `x₁ − z`, MSE) · AdamW lr 1e-3, weight-decay 0.01, batch 64, fp32 · max 300 epochs · training loop
`ppg2ecg.training.train_a0` (same shuffling generator seed) · evaluation script and paired noise seed 0.

## 3. The only change: checkpoint selection / early stopping
**Primary selection metric = deterministic validation CFM loss on fixed (t, z) banks** (`val_cfm_fixed`).
- Banks: `n_banks = 4`, bank `b ∈ {0,1,2,3}` drawn once from `torch.Generator().manual_seed(1000 + b)`:
  `t_b ~ U(0,1)` of shape `[n_val, 1]`, `z_b ~ N(0, I)` of shape `[n_val, 1, 1024]`, in the fixed validation window order
  (subject S11, temporal order). Bank hash = sha256 over the float32 bytes of `(t_0, z_0, …, t_3, z_3)`; stored in provenance.
- Per epoch, for every bank: `x_t = (1 − t_b) z_b + t_b x₁`, `v* = x₁ − z_b`, loss = MSE(`v_θ(x_t, PPG, t_b)`, `v*`) averaged over
  windows (window-weighted); `val_cfm_fixed = mean over the 4 banks`. Cost ≈ 4 forward passes over 1131 windows (≈ 6 s).
- Selection: checkpoint is saved when `val_cfm_fixed < best − min_delta`; **`min_delta = 1e-4` (absolute)** — above the GPU
  run-to-run noise of a deterministic forward pass (≈ 1e-6–1e-5) and well below the late-A0 epoch-to-epoch changes (≈ 1e-3).
- **Early stopping: patience 20** epochs without such an improvement; max 300 epochs. Training loss may still be decreasing.
- The stochastic 50-NFE validation MAE is **not** computed every epoch. Diagnostic only, **every 5 epochs** (cost checked in the
  smoke test): 50-NFE Heun generation on the first 128 validation windows with fixed noise (`z_0[:128]`) → HR abs error,
  beat-template correlation, amplitude ratio. These never influence selection.

## 4. Evaluation (identical to A0)
`scripts/eval_a0_nfe_curve.py` on `checkpoint_best.pt`: Heun 25/10/5/2/1 steps (50/20/10/4/2 NFE) and Euler 1 step (1 NFE),
paired noise seed 0, PPG-shuffle derangement seed 1, seed-diversity 4 seeds × 256 windows, latency batch 64.
Primary metrics: corrected HR abs error (bpm), beat-template correlation (matched beats), **amplitude ratio**
(mean over windows of std(pred)/std(target); added for A0-b and computed post hoc for A0 from its saved predictions, no
re-sampling), conditioning gain (PPG-shuffle), RMSE/MAE. Secondary diagnostics: R-peak F1, RR MAE, PCC (absolute beat-level
correspondence is unreliable under the current PPG-DaLiA protocol, especially under motion), upstream HR error
(corrected / as-shipped), HF-energy ratio, seed diversity, latency, NFE.

## 5. Pre-specified analysis
Table A0 vs A0-b at 50 NFE and 1 NFE (HR error, morph corr, amplitude ratio, conditioning gain, RMSE). Questions and rules:
1. *Was A0 under-trained?* — YES if A0-b's best epoch > 21 **and** `val_cfm_fixed(A0-b best) < val_cfm_fixed(A0 checkpoint) − 1e-4`
   (A0's checkpoint is scored on the same fixed banks post hoc).
2. *How much did 50-NFE quality change?* — report Δ with bootstrap CIs; "changed" = HR error Δ > 1.0 bpm or morph corr Δ > 0.05.
3. *Does the 1-NFE collapse persist?* — YES if A0-b Euler-1 HR error > A0-b 50-NFE + 1.0 bpm or morph corr < 50-NFE − 0.05
   or amplitude ratio < 0.5 or conditioning gain < 50 % of the 50-NFE gain.
4. *Checkpoint artefact vs objective/sampler limitation?* — artefact if the 50→1 NFE gap shrinks below all margins in A0-b;
   otherwise objective/sampler limitation.

## 6. iMeanFlow gate (mechanical)
**GO** if rule 3 says the collapse persists (any one criterion). **NO-GO** if A0-b Euler-1 is within all margins of A0-b 50 NFE
(HR error ≤ +1.0 bpm, morph corr ≥ −0.05, amplitude ratio ≥ 0.5, conditioning gain ≥ 50 %). NO-GO ⇒ no iMeanFlow implementation.

## 7. Not allowed in this stage
seeds 43/44, hyper-parameter changes, architecture/loss/conditioning changes, data re-synchronisation, any post-hoc change of
the selection criterion, min_delta, patience or the margins above.
