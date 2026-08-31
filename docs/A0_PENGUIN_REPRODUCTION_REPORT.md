# A0 PENGUIN Reproduction Report

> **Metric semantics note (added 2026-08-31).** In this document `morph` / `morphology_corr` is
> **matched-beat morphology correlation** and `rr_mae_ms` is **matched-consecutive-beat RR MAE**. Both are
> conditional on successful ≤50 ms one-to-one R-peak matching and are averaged only over matched beats, so
> neither can fall when a beat is missed. Values and specifications here are unchanged; only the labels and
> their scope are made explicit. See [METRIC_SEMANTICS.md](METRIC_SEMANTICS.md).


Experiment `a0_penguin_otcfm_ppgdalia_8s_seed42` — generated 2026-08-25T16:09:55 from `outputs/a0_penguin_otcfm_ppgdalia_8s_seed42/`. Single seed (42). No new method; baseline gate only.

## Frozen protocol
- Model: PENGUIN (upstream class, unmodified) Flow-SSM/S5 + OT-CFM; cfg `{'n_step': 25, 'sample_rate': 128, 'h_dim': 128, 'ssm_block_num': 4, 'ssm_ratio': 2.0, 'mlp_ratio': 2.0}`; params total 4,568,707 (effective 4,304,513; dead `cross_attn`/`revin` excluded)
- Objective: OT-CFM (Lipman conditional-OT path, σ=0, independent coupling), t ~ U(0,1), MSE on velocity — upstream `train_flow`/`optimize` unchanged
- Train/val sampler: Heun 25 steps = **50 NFE**
- Optimiser AdamW(betas=(0.9,0.999)), lr 0.001, weight decay 0.01, batch 64, ≤ 300 epochs, early stopping on val_mae_batchmean (full n_step Heun samples) (patience 10), checkpoint = best val MAE
- Precision: float32, AMP False, BF16 False, TF32 matmul False; deterministic: {'cudnn_deterministic': True, 'use_deterministic_algorithms': 'True (warn_only)'}
- Pre-registration: `docs/PREREGISTRATION_V0.md` (§6 window/split/seed decisions fixed before training)

## Git / environment provenance
- Repo commit `6e5a4f1da2408077e7a809e969a3bf93c7f74c8d` (branch main, 0 dirty files at preflight)
- Upstream PENGUIN `6cd70cdefb91f10efeb8dce34019b5067cb25344` (expected `6cd70cdefb91f10efeb8dce34019b5067cb25344`, dirty 0) — https://github.com/Neurogica/PENGUIN.git
- GPU NVIDIA GeForce RTX 5090 (31.4 GiB, sm_120), torch 2.11.0+cu130 / CUDA 13.0 / cuDNN 91900, Python 3.13.9
- Full provenance: `outputs/a0_penguin_otcfm_ppgdalia_8s_seed42/provenance.json`

## Dataset
- PPG-DaLiA (UCI #495, CC BY 4.0); raw zip sha256 `5772387956e34e2e…` (`data/raw/CHECKSUMS.sha256`)
- Processed `8 s` windows @ 128 Hz = 1024 samples, 16181 windows total, built 2026-08-25T15:10:00; per-file sha256 in provenance
- Preprocessing (PENGUIN-faithful, bit-exact vs upstream `preprocess.py` at 8 s for 15/15 subjects): PPG {'bandpass': True, 'freq_range': [0.5, 4], 'zscore': True, 'normalize': True}, ECG {'bandpass': True, 'freq_range': [0.5, -1], 'zscore': True, 'normalize': True}, all statistics per window

## Split
- Manifest `data/manifests/split_p0_holdout_seed42.json` (sha256 `11c154e471b65e15…`), protocol P0-holdout seed 42
- train (13): S1, S3, S4, S5, S6, S7, S8, S9, S10, S12, S13, S14, S15; val: S11; test: S2
- windows: train 14025, val 1131, test 1025
- Leakage checks at preflight: subject-disjoint True {'train∩val': [], 'train∩test': [], 'val∩test': []}; window-hash-disjoint True {'train∩val': 0, 'train∩test': 0, 'val∩test': 0}; window-local normalisation PPG True / ECG True
- Limitation: upstream's own split is glob-order dependent (on this machine it would be val S4 / test S10); the paper's held-out subject is unknown, so this is **not** guaranteed to be the paper's split.

## Window-length decision
8 s windows were fixed **before training** from the audit (`docs/PENGUIN_AUDIT.md` §5/§20, `PREREGISTRATION_V0.md` §6): with the shipped 4 s config the upstream HR metric compresses the 8 s evaluation window 2× (true 60 bpm → 119.7), masks high-HR windows and zero-fills failures; upstream's `sample_num=16181` equals the number of 8 s windows exactly; the paper does not state the window length (a figure caption mentions 4 s). The 4 s configuration remains a documented ambiguity and was not trained in this stage.

## Training
- Epochs run: 21 (max 300); best epoch **11**; early stopped: True; total training time 0.83 h; peak GPU memory 18.0 GiB
- Best validation MAE (batch-mean, 50-NFE samples, val subject S11): 0.2989; first-epoch val MAE 0.6704; last-epoch val MAE 0.3167
- Train CFM loss: epoch 1 0.3393 → final 0.1717; val CFM loss (fixed noise): 0.2568 → 0.1744; LR constant 0.001
- Per-epoch time ≈ 143 s; full log `outputs/a0_penguin_otcfm_ppgdalia_8s_seed42/training_log.csv`

## Paper-vs-code discrepancies
See `docs/PENGUIN_AUDIT.md` §22/§25. Consequential for this run: (1) window length ambiguity (handled: 8 s); (2) paper claims a 6:1:1 subject split, code does 13/1/1 with an unlogged, filesystem-dependent test subject (handled: deterministic manifest, single test subject S2); (3) upstream HR metric pathology (handled: corrected + our own HR error; as-shipped reported only as diagnostic); (4) PPG conditioning is MLP-based, not a linear projection; dead `cross_attn`; (5) no seeds/variance in the paper.

## Main result
Reference arm Heun 25 steps (**50 NFE**) on test subject S2 (1025 × 8 s windows):
- **corrected HR error (ours) = 10.99 bpm**, R-peak F1 0.141, RR MAE 34.9 ms, QRS-width error 33.7 ms, beat morphology corr 0.662, MAE 0.400, RMSE 0.472, PCC 0.002
- upstream `HeartRateError` corrected 11.74 bpm; **as-shipped (diagnostic only) 25.14 bpm**
- windows with no detected predicted beats: 0.0 %

## NFE-quality curve
| Solver | Steps | Actual NFE | HR Error (bpm) | R-F1 | RR MAE (ms) | RMSE | PCC | QRS Error (ms) | Morph corr | Latency (ms / batch 64) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Heun | 25 | 50 | 10.99 | 0.141 | 34.9 | 0.472 | 0.002 | 33.7 | 0.662 | 4175 |
| Heun | 10 | 20 | 11.59 | 0.140 | 35.2 | 0.471 | 0.002 | 33.7 | 0.664 | 1652 |
| Heun | 5 | 10 | 12.25 | 0.138 | 36.0 | 0.466 | 0.002 | 33.1 | 0.639 | 825 |
| Heun | 2 | 4 | 11.17 | 0.146 | 34.6 | 0.444 | 0.001 | 33.9 | 0.475 | 329 |
| Heun | 1 | 2 | 36.32 | 0.087 | 37.3 | 0.479 | -0.001 | 33.9 | 0.230 | 165 |
| Euler | 1 | 1 | 39.22 | 0.087 | 33.9 | 0.295 | 0.007 | 32.8 | 0.136 | 82 |

Pre-registered non-inferiority margins vs the 50-NFE reference (§5: F1 drop > 0.02, HR error rise > 1.0 bpm, morph-corr drop > 0.05, QRS-width error rise > 10 ms):
| Solver | Steps | NFE | ΔF1 | ΔHR (bpm) | Δmorph | ΔQRS (ms) | metrics failing |
|---|---:|---:|---:|---:|---:|---:|---|
| Heun | 25 | 50 | +0.000 | +0.00 | +0.000 | +0.0 | — |
| Heun | 10 | 20 | -0.001 | +0.60 | +0.001 | +0.0 | — |
| Heun | 5 | 10 | -0.003 | +1.27 | -0.023 | -0.6 | hr_abs_err_bpm |
| Heun | 2 | 4 | +0.004 | +0.18 | -0.187 | +0.2 | morph_corr |
| Heun | 1 | 2 | -0.055 | +25.33 | -0.432 | +0.2 | rpeak_f1, hr_abs_err_bpm, morph_corr |
| Euler | 1 | 1 | -0.055 | +28.23 | -0.526 | -0.9 | rpeak_f1, hr_abs_err_bpm, morph_corr |

Figure: `outputs/a0_penguin_otcfm_ppgdalia_8s_seed42/figures/nfe_curve.png`; per-window metrics in `predictions/*.npz`.

## Morphology analysis
| Solver | Steps | NFE | HF-energy ratio pred / target | seed std (mean) | seed pairwise corr | cond. gain (bpm) | HR err vs right target (shuffled PPG) | HR err vs wrong target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heun | 25 | 50 | 0.321 / 0.323 | 0.218 | 0.000 | 3.84 | 11.10 | 14.95 |
| Heun | 10 | 20 | 0.327 / 0.323 | 0.217 | 0.001 | 3.76 | 11.48 | 15.24 |
| Heun | 5 | 10 | 0.343 / 0.323 | 0.215 | 0.001 | 3.56 | 12.36 | 15.92 |
| Heun | 2 | 4 | 0.418 / 0.323 | 0.212 | 0.001 | 3.56 | 11.13 | 14.69 |
| Heun | 1 | 2 | 0.475 / 0.323 | 0.292 | 0.004 | -0.40 | 36.95 | 36.55 |
| Euler | 1 | 1 | 0.636 / 0.323 | 0.062 | 0.064 | 0.40 | 38.49 | 38.89 |

**What degrades first (50 → 4 NFE).** Rate-level metrics are essentially flat down to 4 NFE (HR error 10.99 → 11.59 → 12.25 →
11.17 bpm; the +1.27 bpm at 10 NFE nominally crosses the +1.0 bpm margin but is non-monotonic and its 95 % CI (11.6–12.9) barely
separates from the 50-NFE CI (10.4–11.6) — single noise seed, treat as borderline; RR MAE 34.9 → 34.6 ms; QRS-width proxy
33.7 → 33.9 ms), while the **beat-aligned template correlation is the first primary metric to cross its margin decisively**:
0.662 (50) → 0.664 (20) → 0.639 (10) → **0.475 (4 NFE, −0.19, CI 0.463–0.486)**.
Qualitatively (`figures/example_*.png`), the 4-NFE traces keep sharp QRS spikes at the right rate but the baseline between beats
becomes noisy (HF-energy ratio 0.32 → 0.42) — residual noise, not yet averaging.

**Collapse at ≤ 2 NFE.** Heun 1 step (2 NFE) and Euler 1 step (1 NFE) fail every rate/morphology criterion: HR error 36–39 bpm,
template correlation 0.23 / 0.14, predicted beats per window 4.9 / 4.3 vs 10.5 true (spurious detections, not real beats),
conditioning gain of the PPG-shuffle test ≈ 0 (−0.4 / +0.4 bpm vs 3.8 bpm at 50 NFE).

**1-NFE Euler = conditional-mean averaging, plus residual noise.** The 1-NFE output is `x0 + v(x0, t=0) ≈ E[x1 | PPG]`: amplitude
collapses (per-window std 0.046 vs 0.236 for the target and 0.196 at 50 NFE), the waveform is a flat line near −0.3 with small
ripples (`example_*` bottom rows), seed-to-seed diversity vanishes (std 0.062 vs 0.218), and the remaining energy is
high-frequency (HF ratio 0.64 vs 0.32) — i.e. the mean prediction still contains un-cancelled noise. This is why **RMSE improves
at 1 NFE (0.472 → 0.295) while every clinical metric fails**: global waveform error rewards the averaged, beat-free output.
RMSE/MAE/PCC must not be used as quality criteria for this task.

**Conditional fidelity of the baseline itself.** Even at 50 NFE the model tracks heart rate only coarsely: predicted vs
reference HR correlate at r = 0.40, with regression to the training mean (ref HR < 70 bpm: +11 bpm bias; 90–110: −6; > 110:
−28 bpm; stairs/cycling windows have 18–26 bpm error). The PPG-shuffle test confirms the conditioning carries rate information
(3.8 bpm gain) but not much more. **Beat-level alignment is absent at every NFE** (R-peak F1 0.14 at 50 ms, PCC ≈ 0, prediction-to-
target cross-correlation lag uniformly spread over ±0.5 s, no gain from a global shift). The data-level diagnostic
(`figures/dalia_sync.png`, `DATA_PROTOCOL.md` §6) shows the wrist-PPG/chest-ECG streams of PPG-DaLiA are only second-level
synchronised with ~20 ms/min relative drift, so beat-level phase is not a learnable target on the raw windows: the low F1 is a
property of the dataset, not evidence about the model, and F1/PCC/RR-MAE (beat-matched) are reported only for completeness.

Example figures (deterministic selection: fixed positions [0, 256, 512, 768, 1024] and 10/50/90 % HR-error quantiles of the 50-NFE arm [880, 482, 824]): `outputs/a0_penguin_otcfm_ppgdalia_8s_seed42/figures/example_*.png`

## Efficiency
| Solver | Steps | Actual NFE | Latency (ms / batch 64, median) | samples / s | peak GPU mem (MiB) |
|---|---:|---:|---:|---:|---:|
| Heun | 25 | 50 | 4174.6 | 15.3 | 1767 |
| Heun | 10 | 20 | 1652.1 | 38.7 | 1767 |
| Heun | 5 | 10 | 825.0 | 77.6 | 1767 |
| Heun | 2 | 4 | 328.9 | 194.6 | 1767 |
| Heun | 1 | 2 | 165.4 | 386.9 | 1766 |
| Euler | 1 | 1 | 82.2 | 778.9 | 1766 |

Measured on the same GPU, fp32, no compile, 3 warm-ups, fixed batch of 64 test windows.

## Upstream HR metric pathology
| Solver | Steps | NFE | ours corrected HR err | upstream corrected | upstream as-shipped (4 s path, diagnostic) |
|---|---:|---:|---:|---:|---:|
| Heun | 25 | 50 | 10.99 | 11.74 | 25.14 |
| Heun | 10 | 20 | 11.59 | 11.97 | 25.74 |
| Heun | 5 | 10 | 12.25 | 12.50 | 27.14 |
| Heun | 2 | 4 | 11.17 | 20.10 | 26.16 |
| Heun | 1 | 2 | 36.32 | 47.84 | 30.01 |
| Euler | 1 | 1 | 39.22 | 44.65 | 33.45 |

The as-shipped column reproduces upstream's 4 s-config code path on the 8 s windows (2× time compression, high-HR masking, 0.0 fallback). It is reported only to show whether that pathology lands near the paper's number; it is **not** a reproduction metric.

## Paper number comparison
- Paper (arXiv:2602.03858, Table 1, PPG-DaLiA): HR Error **15.64 bpm** (RDDM 16.43, CycleGAN 23.61, RespDiff 22.75, PaPaGei-S 40.89; w/o PPG conditioning 24.40)
- Ours, 50 NFE, corrected HR error: **10.99 bpm** → **PASS** (PASS ≤ 17.2, BORDERLINE ≤ 20, FAIL > 20)
- Upstream corrected: 11.74 bpm; upstream as-shipped: 25.14 bpm (|Δ paper| = 9.50)
- Caveat: single test subject, single seed, unknown paper split/window length → no binary 'paper reproduced' claim; the verdict is a feasibility gate.

## Limitations
- Single seed (42), single test subject (S2, 1025 windows) and single validation subject (S11); early stopping is driven by a
  noisy stochastic validation MAE (patience 10 fired at epoch 21 while the train loss was still falling) — the checkpoint is
  probably under-trained relative to a smoother criterion.
- Window length (8 s) and split are audit-based choices; the paper's held-out subject and window length are unknown, so the
  paper comparison is a feasibility gate, not a reproduction claim.
- Beat-aligned metrics are invalid on raw PPG-DaLiA (device synchronisation, §6 of DATA_PROTOCOL); QRS width is a QS-trough proxy
  on min-max-normalised signals; no absolute amplitude is available.
- Efficiency numbers are for the unmodified upstream S5 implementation (vmap + associative scan, fp32, no compile) on one GPU.
- The A0 loop is ours (mirrors upstream step-for-step, verified bit-exact for sampler/objective); the official upstream
  `train.py` with its glob-order split was not run.

## GO / NO-GO
Pre-registered questions (PREREGISTRATION_V0 §5): **H1 confirmed** — at NFE ≤ 2 (Euler 1, Heun 1) every primary metric that is
interpretable on this dataset fails by a wide margin (HR error +25–28 bpm, template correlation −0.43/−0.53, QRS proxy within
margin only because no real beats are detected); the first failure appears at 4 NFE (template correlation −0.19). The
hypothesis that "OT-CFM is already fine at 1–2 NFE" is **rejected**: the NO-GO condition for the one-step line does not hold.

Failure mode (H2 vs H2′): at 1 NFE the output is the conditional mean — beat morphology, rhythm and conditioning are lost
together (amplitude collapse, zero diversity, conditioning gain → 0). This is closer to **H2 (averaging/blur)** than to a pure
conditioning failure, but the distinction is moot because the mean waveform has no beats; the primary target for a one-step
objective is therefore "produce a beat-bearing, rate-correct waveform in one evaluation", measured by HR error, template
correlation on matched beats, amplitude (per-window std ratio) and the PPG-shuffle conditioning gain.

Verdict: **GO for the objective-swap line (A1′/A2)**, with two caveats that must be handled before A2 is meaningful:
1. the baseline's own conditional fidelity is weak (HR r = 0.40, regression to the mean) — a one-step model that merely
   matches 50-NFE OT-CFM inherits a weak ceiling; the comparison must be relative (recover the 50-NFE level at 1 NFE), not absolute;
2. beat-level metrics require a re-synchronised evaluation protocol on DaLiA (or a beat-synchronised dataset), otherwise
   "morphology preserved" can only be claimed at the template/rate level.

## Recommended next experiment
Not a new method yet. In order:
1. **Ceiling & variance of the baseline (cheap, no new code):** re-run A0 with a smoother model-selection criterion
   (val CFM loss or val MAE averaged over ≥ 3 noise draws) and seeds 43/44, to know whether HR r = 0.40 / 11 bpm is the
   OT-CFM+S5 ceiling on DaLiA or an early-stopping artefact. Pre-register as A0-b before running.
2. **Re-synchronised beat-level protocol (data work):** estimate the per-segment PPG↔ECG lag (cross-correlation of the PPG
   derivative with the R-peak train, smoothed over minutes) and re-cut windows; verify on the dataset R-peaks that the
   pulse-arrival delay becomes physiologically stable (≈ 250–400 ms, IQR < 60 ms in rest); re-evaluate A0 on the re-synchronised
   test windows. Only then are R-peak F1 / RR MAE meaningful.
3. **A1′ control on the same protocol**, then **A2 = iMeanFlow on the identical S5 backbone at 1 NFE**, judged against the 50-NFE
   reference with the margins of §5 on HR error, template correlation, amplitude ratio and conditioning gain.
