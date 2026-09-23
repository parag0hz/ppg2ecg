# EXP-D — Functional generalisation of fixed-budget allocation: preregistration

Frozen and pushed **before any EXP-D sample is generated** on a test input and before any test functional is computed.
Audit: `docs/EXP_D_FUNCTIONAL_GENERALIZATION_AUDIT.md` (ABP_PHYSICAL_SCALE = VALID). Head at freeze: `eb3f2d7`.
This design **replaces** the unexecuted EXP-D section of `docs/TOP_TIER_COMPLETION_PREREGISTRATION.md` (`50948e2`); that
frozen text is not edited.

**Question.** Does the fixed-budget allocation principle found for ECG → HR ("refine enough, then sample wide") hold for
qualitatively different physiological functionals: respiratory rate from a generated respiration waveform, and SBP / DBP /
MAP from a generated ABP waveform? No training, no new architecture / loss / backbone, no test-set tuning, no new external
model, no dataset change. Budget wording: **fixed generative vector-field evaluation budget** B = K × S (Euler NFE); for
PENGUIN this is also its total network compute (no encoder / decoder outside the vector field).

## 1. Models and data (fixed)
| | Respiration (Part A) | ABP (Part B) |
|---|---|---|
| checkpoint | `outputs/u1_upstream/PENGUIN_BIDMC_u1/ckpt/pretrain_ckpt.pth` (sha256 `fa8b3d96…`, epoch 6); exploratory: `PENGUIN_WESAD_u1` (`946dd2b1…`, epoch 9) | `outputs/u1_upstream/PENGUIN_MIMIC-BP_u1/ckpt/pretrain_ckpt.pth` (`02dd37c1…`, epoch 17) |
| training | upstream `train.py` as shipped (U1), seed 42, upstream split | same |
| test split | upstream split (`outputs/u1_upstream/split_<dataset>.json`, `test`) | same |
| test size | **BIDMC 6 subjects, 720 windows, 48 blocks of 60 s**; WESAD 1 subject, 1,605 windows, 107 blocks | **MIMIC-BP 190 subjects, 39,900 windows, 19,950 blocks of 8 s** |
| preprocessing | upstream (PPG 0.5–4 Hz band-pass, z-score, [−1, 1]; resp label 1 Hz low-pass, z-score, [−1, 1]; 128 Hz, 4 s) | upstream (same PPG; ABP raw mmHg, resampled to 128 Hz only) |
UCI-BP is not used (test subject duplicated in training). Loading: upstream `initialize_model(ckpt["cfg"])` +
`load_state_dict(strict=True)` (audited: no missing / unexpected keys).

## 2. Generation (DW1 protocol, functional changed only)
- Test stream = the split's `test` file order, windows in file order. Draw d uses `torch.Generator().manual_seed(d)` to
  draw z_d ~ N(0, I) of shape [N_test, 1, 512] once for the whole stream; **the same z_d for every S** (common noise).
- Sampler: `ppg2ecg.flow.samplers.euler_sample` on upstream's grid, velocity = upstream `forward_step`, float32, batches
  of 512 windows. S ∈ {1, 2, 4, 8, 16, 32} — vector-field NFE = S (hook-verified).
- Draws generated: S = 1: d = 0…31; S = 2, 4, 8: d = 0…15; S = 16: d = 0, 1; S = 32: d = 0.
- Shipped-quality reference: upstream Heun, 25 steps = **50 NFE**, K = 1, noise z_0 (`heun_sample`, bitwise = `PENGUIN.sample`).
  It is not a cell of any budget.

## 3. Functionals (upstream definitions; frozen)
Blocks: consecutive test windows of one subject — **60 s = 15 windows** (respiration), **8 s = 2 windows** (ABP), as
upstream `train.py:41-47` (no block crosses a subject). A sample's block = concatenation of that draw's windows.
- **RR** (breaths / min): 60 × the positive FFT frequency of maximum magnitude of the block; the generated block is first
  low-passed (order-8 Butterworth, 1 Hz, `filtfilt`), the reference block is not — upstream `RespRateError`, via
  `ppg2ecg.evaluation.penguin_metrics` (unit-tested port).
- **SBP** = block maximum, **DBP** = block minimum (mmHg; upstream `SBPError` / `DBPError`).
- **MAP** = block time-average (mmHg; **ours** — not in the upstream protocol; no parameters).
- Reference functional = the same definition on the reference block (RR: unfiltered, as upstream). A non-finite generated
  sample makes its functionals undefined; consensus = median over the defined sample functionals of a group.
  No beat detection, no threshold, nothing tuned.

## 4. Cells, estimator, contrasts
- **Estimator (DW1):** cell (K, S) uses draws 0 … K − 1 at depth S; consensus = median over them; error |T̂ − T\*|.
- **B = 32 (primary):** (32,1) (16,2) (8,4) (4,8) (2,16) (1,32). **B = 16 (secondary):** (16,1) (8,2) (4,4) (2,8) (1,16).
- **Headline contrasts (fixed; negative = width better):**
  B32: **(32,1) − (1,32)** (width extreme) and **(8,4) − (1,32)** (the ECG rule's cell). B16: (16,1) − (1,16) and (4,4) − (1,16).
- Reported, not selected: all other cells and their contrast with the budget's depth cell; the test-best cell (labelled
  exploratory); (32,1) and (8,4) vs the shipped Heun-50 reference; consensus checks (16,S) − (1,S) for S ∈ {1, 2, 4, 8}.
- Aggregation: block errors → subject mean → mean over subjects (subject macro).
- Inference: paired subject-clustered bootstrap, **5,000 replicates, seed 20260924**, percentile 95 % CI; subject win
  rate (share of subjects whose mean difference is < 0; ties reported). No multiplicity correction; §8 is the only
  decision rule. BIDMC has 6 test subjects: its bootstrap is coarse and the respiration result is labelled low-power.
- WESAD (1 subject): same cells, block-level bootstrap within the subject (107 blocks), **exploratory, no verdict role**.
- Sanity baseline (no training): constant predictor = median reference functional over **training-subject** blocks.

## 5. Waveform / event metrics (secondary; draw 0 at each S and Heun-50)
Per 4-s window vs reference: MAE, RMSE, Pearson r (resp in normalised units, ABP in mmHg), subject macro with CI;
upstream raw-signal FD (`fid_features_to_statistics` / `fid_statistics_to_metric`, all test windows). No respiratory-event
or pulse-timing metric exists upstream; none is added. Latency per cell: K sequential batch-1 calls and one call with the
K samples batched (GPU). **Width-condition collapse** (resp verdict): a width condition's depth (S = 1 for (32,1),
S = 4 for (8,4)) has Pearson r < 0.5 × that of S = 32, or RMSE > 1.5 × that of S = 32, or FD > 3 × that of S = 32, or
> 1 % non-finite samples.

## 6. Fixed-K mechanism (K = 16, S ∈ {1, 2, 4, 8}, draws 0–15, per functional)
On blocks with every one of the 64 sample functionals defined: individual error I_S = mean_k |Y_k − T\*|, consensus error
C_S = |median_k Y_k − T\*|, **G_S = I_S − C_S**, functional SD (ddof 1), MAD, waveform pairwise RMS (mean over the 120
draw pairs of the block-waveform RMS difference) — subject macro; **ρ̄_S** = mean over the 120 draw pairs of the Pearson
correlation, across blocks, of the signed errors Y_k − T\*; K_eff = 16 / (1 + 15 ρ̄_S).
Reported: ordering over S; Spearman(ρ̄_S, G_S) over the four depths (also G_S with waveform RMS, SD, MAD; ρ̄_S with
G_S / I_S) with a subject bootstrap (5,000, seed 20260924; ρ̄ and G recomputed from resampled subjects) giving the share
of negative replicates. **"Directionally consistent with HR"** = Spearman(ρ̄_S, G_S) point estimate < 0. Four points:
descriptive. ABP only (190 subjects): subject-level Spearman(ρ̄_p, G_p) at each S over subjects with ≥ 8 blocks, with a
subject bootstrap. If ρ̄ is undefined (a constant error series) that is reported, not repaired.

## 7. Latency / compute accounting
Vector-field NFE per cell = K × S (Heun-50 = 50); no encoder / decoder overhead; sequential and batched wall-clock reported
as measured, including any overhead of width.

## 8. Verdicts (fixed)
**Respiration (BIDMC).** A budget *succeeds* if at least one of its two width contrasts has a 95 % CI entirely below 0 and
that width condition is not collapsed (§5). *Depth clearly better at B32* = both B32 contrasts have CI entirely above 0.
*No useful consensus gain* = (16,S) − (1,S) has CI upper bound ≥ 0 for every S ∈ {1, 2, 4, 8}.
- **FAILED** — depth clearly better at B32; or no useful consensus gain; or neither B32 point estimate is < 0 and no budget
  succeeds.
- **STRONG** — B32 succeeds **and** B16 succeeds **and** the mechanism is directionally consistent (§6).
- **PARTIAL** — everything else (e.g. width direction with CI crossing 0; only one budget succeeds; a significant width
  contrast whose condition is collapsed; mechanism not negative).
- WESAD never enters the verdict.

**ABP (MIMIC-BP), per functional f ∈ {SBP, DBP, MAP}:** *success_f* = at B32, (32,1) − (1,32) or (8,4) − (1,32) has CI
entirely below 0; *depth-clear_f* = both B32 contrasts have CI entirely above 0; *directional_f* = at least one B32 point
estimate < 0.
- **FAILED** — depth-clear for ≥ 2 functionals; or extraction invalid (> 1 % non-finite generated samples); or no
  functional succeeds and ≤ 1 is directional.
- **STRONG** — success for ≥ 2 of the 3 functionals.
- **PARTIAL** — everything else (1 success, or ≥ 2 directional without 2 successes).
Each functional's result is reported separately; one success is never called an ABP success. B16 is reported and has no
verdict role. MAP is labelled as a functional of ours.

## 9. Order of work and stops
Part A (BIDMC + WESAD exploratory) → `docs/EXP_D_RESPIRATION_REPORT.md` → commit / push → **HARD STOP A**. Part B only
after that, and only because ABP_PHYSICAL_SCALE = VALID → `docs/EXP_D_ABP_REPORT.md` → **HARD STOP B**. The
cross-functional synthesis and `docs/EXP_D_FINAL_REPORT.md` only if both are valid. No direct RR / BP regressor is trained.
No functional is redefined after results; failed cells are not removed.

## 10. Claim boundaries
Allowed if supported: "Across multiple physiological outputs and downstream functionals, allocating a fixed generative
vector-field budget entirely to trajectory refinement is often suboptimal"; the stronger "the same width-heavy allocation
principle extends from rate estimation to blood-pressure amplitude functionals" only if ABP is STRONG. Never: width always
wins; one step is always optimal; one S suits every functional; consensus beats direct specialised predictors;
generation is necessary; the mechanism is causal; waveform fidelity is irrelevant.

## 11. Outputs
`artifacts/exp_d_functional_generalization/{respiration, abp, cross_functional}/`: audit.json, prereg_manifest.json,
fixed_budget.csv, bootstrap.json, mechanism.csv, latency.json, per_patient.csv, figure.png. Large arrays:
`outputs/exp_d_functional_generalization/` (gitignored). Script: `scripts/expd/expd_run.py`.
