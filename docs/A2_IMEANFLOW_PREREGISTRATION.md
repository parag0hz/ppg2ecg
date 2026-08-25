# A2 Pre-registration — Improved MeanFlow on the identical S5 backbone (one-step PPG→ECG)

Written 2026-08-25 after the A0-b gate returned **GO** (`docs/A0B_BASELINE_STABILIZATION_REPORT.md`) and **before** any A2
training. Frozen at the commit that launches the run (recorded in `outputs/a2_imeanflow_s5_ppgdalia_8s_seed42/provenance.json`).

## 1. Question
> Can Improved MeanFlow make the long noise→ECG transport jump in one network evaluation while preserving the physiological
> structure (beat existence, beat rate, sharp QRS morphology, waveform amplitude, PPG-conditioning dependence) that OT-CFM needs
> many evaluations to generate?

## 2. Identical to A0-b (frozen)
PPG-DaLiA · 8 s @ 128 Hz · preprocessing v0 · `data/processed/v0_8s` · split `split_p0_holdout_seed42.json` (train 13 / val S11 /
test S2) · seed 42 · upstream PENGUIN backbone class unchanged (`6cd70cd`, 4,568,707 parameters) · PPG conditioning path unchanged ·
AdamW lr 1e-3, weight-decay 0.01, batch 64, fp32, no EMA, no schedule · max 300 epochs · patience 20, min_delta 1e-4 ·
evaluation code, paired noise seed 0, PPG-shuffle derangement seed 1, quantile example windows from A0.

## 3. The only change: training objective / flow parameterisation (`src/ppg2ecg/flow/imeanflow.py`, `docs/IMEANFLOW_AUDIT.md`)
- Convention: t = 1 noise, t = 0 data; `z_t = (1−t)x + t e`, `v = e − x`.
- Network `u_θ(z, PPG, t, h = t − r)`: the unmodified backbone with conditioning vector `E(t) + E(1000·h)` from its single existing
  timestep embedder (shared weights; **parameter count unchanged**). *Amended before any result (see §9): with `E(t) + E(h)` the
  interval is almost invisible to the network (review finding); scaling `h` by 1000 — the DiT integer-timestep convention for the same
  sinusoidal embedder — makes t, h and r fully decodable without new parameters.*
- Loss (iMF Eq. 12 / Alg. 1, official `imf.py` L347-393 without CFG/labels/aux-head):
  `V = u_θ(z_t, r, t) + (t − r)·sg[JVP(u_θ; (v_θ, 0, 1))]`, `v_θ = u_θ(z_t, t, t)` (boundary condition, gradient-free),
  `loss = mean_b[ Σ_T (V − (e − x))² · sg(1/(Σ_T(V − (e−x))² + 0.01)^1) ]`.
- `(t, r)`: i.i.d. logit-normal(−0.4, 1.0), `t = max, r = min`, first 50 % of every batch `r = t`; noise `e ~ N(0, I)`.
- JVP: `torch.func.jvp` (forward-mode AD; verified vs finite differences on the backbone). Fallback `double_vjp` only if
  forward-mode fails at full size — recorded if used.
- Sampling: 1-NFE `x̂ = e − u_θ(e, r = 0, t = 1)` (**primary**); 2- and 4-step MeanFlow sampling (`z_r = z_t − (t − r)u`) as diagnostics only.

## 4. Checkpoint selection / early stopping (mirrors A0-b)
Primary metric = **deterministic fixed-bank iMF validation MSE** (`val_imf_mse_fixed`): 4 banks, bank b drawn once with
`torch.Generator().manual_seed(1000 + b)`: `(t_b, r_b)` via `sample_tr` **with the rows randomly permuted (so the r = t half is a
random subset of the temporally ordered validation windows, not always the first half — review finding)**, `e_b ~ N(0, I)`; metric = per-element mean of
`(V − (e − x))²` (no adaptive weight), averaged over banks; hash of `(t, r, e)` recorded. Best checkpoint when
`val_imf_mse_fixed < best − 1e-4`; patience 20; max 300 epochs. Diagnostic **every epoch**: **1-NFE** generation on the first 128
validation windows with fixed `e_0[:128]` → HR error, template correlation, amplitude ratio, beats/reference (never used for
selection; the selection metric is the validation counterpart of the training objective and is deliberately *not* a pointwise
1-NFE reconstruction error, which would favour conditional-mean outputs). Non-finite unweighted MSE or JVP term aborts training
(the adaptive-weighted loss itself saturates near 1 and is uninformative).

## 5. Evaluation and comparison (identical metric code)
Arms: A0-b OT-CFM Heun 25 (50 NFE), Heun 2 (4 NFE), Euler 1 (1 NFE) from `outputs/a0b_…/nfe_curve.csv`; **A2 iMF 1 NFE**
(+ 2/4-step diagnostics) on the same test windows with the **same paired noise tensor** (seed 0) as `e`.
Metrics: corrected HR error, beat-template correlation, amplitude ratio (std(pred)/std(target)), conditioning gain (PPG-shuffle,
same derangement), RMSE, MAE, latency, actual NFE; secondary: PCC, R-peak F1, RR MAE, HF ratio, seed diversity, upstream HR error.
**Recovery score** for metric m (auxiliary summary; raw metrics are always shown):
`rec(m) = (iMF₁ − OTCFM₁)/(OTCFM₅₀ − OTCFM₁)` for higher-is-better (morph corr, conditioning gain),
`rec(m) = (OTCFM₁ − iMF₁)/(OTCFM₁ − OTCFM₅₀)` for lower-is-better (HR error, RMSE), and for amplitude
`rec = 1 − |amp_iMF − amp₅₀| / |amp₁ − amp₅₀|`.

## 6. Verdict rules (frozen)
Physiological set P = {HR error, template correlation, amplitude ratio, conditioning gain}.
- **SUCCESS**: rec(m) ≥ 0.5 for **all** m ∈ P at 1 NFE, and the 1-NFE output has beats (mean predicted beats per window ≥ 0.7 ×
  reference) — i.e. beat existence, rate, morphology, amplitude and conditioning recover simultaneously.
- **PARTIAL**: rec ≥ 0.5 for at least one m ∈ P but not all, or SUCCESS on P with beats < 0.7 × reference.
- **FAIL**: rec < 0.25 for all m ∈ P, or training instability (non-finite loss / divergence that cannot be attributed to a bug),
  or conditioning gain ≤ 0.25 × A0-b 50-NFE gain together with rec(HR) < 0.25.
- Any outcome not covered above (e.g. all recoveries in [0.25, 0.5)) is **PARTIAL**.
- RMSE/MAE improvements without recovery on P are explicitly **not** success (conditional-mean collapse).
Thresholds may not be changed after results are seen. If PARTIAL/FAIL: write the failure taxonomy (F1–F6) — no new method.

## 7. Not allowed
New architecture/loss/conditioning, CFG, EMA, LR/optimizer changes without a logged literature reason, seeds 43/44,
hyper-parameter search, re-synchronisation of PPG-DaLiA, cherry-picked examples.

## 8. Implementation details fixed before launch (not protocol changes)
- **Memory**: forward-mode JVP through the S5 scan at T = 1024 needs ≈ 0.51 GiB per sample (measured: B = 8/16/32/48 → 4.1/8.2/16.3/24.4 GiB;
  B = 64 → OOM on the 31.4 GiB card; double-VJP is worse at 0.82 GiB/sample). The **effective batch stays 64**: each optimiser step
  accumulates two micro-batches of 32 (`--micro-batch 32`), each with its own `(t, r)` draw (50 % r = t per micro-batch) and loss
  scaled by 32/64 — mathematically identical to a single batch of 64 (per-sample adaptive weights, mean over the batch, one AdamW
  step per 64 windows). The fixed-bank validation metric uses `--val-batch 32` (a pure implementation detail; values unchanged).
- **MKL warm-up** (`ppg2ecg.utils.mkl_warmup`) imported before torch in all entry points (docs/ENVIRONMENT.md incident of 19:00).
- Numerics: fp32, `torch.func.jvp`, deterministic-algorithms warn-only, same shuffle generator seed as A0/A0-b; `(t, r)` from a
  separate CPU generator (seed 43); `e` from the CUDA RNG (seed 42).

## 9. Amendments (all before any A2 result was examined)
- **2026-08-25 19:35 — conditioning `E(t)+E(h)` → `E(t)+E(1000·h)`.** The adversarial implementation review showed that with the
  shared embedder and `h ∈ [0,1]` the conditioning vector is 99.3 % explained by `t + h` alone (linear decodability of `r`: R² = 0.18;
  official h-only design: 1.00), i.e. the MeanFlow interval would be nearly invisible to the network during training — a confound
  against the objective under test. Measured on the backbone's embedder (fresh and A0-b weights): scaling `h` by 1000 gives
  R² = 1.00 for t, h and r with **no added parameters**. The first A2 run (h_scale = 1) was stopped after epoch 1 (its log is kept in
  `outputs/aborted/a2_hscale1_aborted_epoch1/`; epoch-1 diagnostic HR 24.5 / morph 0.41 / amp 0.81 — no test-set result was
  produced or examined). Alternative considered and rejected for the frozen parameter count: a second embedder for h (+49 k params,
  the MeanFlow paper's design).
