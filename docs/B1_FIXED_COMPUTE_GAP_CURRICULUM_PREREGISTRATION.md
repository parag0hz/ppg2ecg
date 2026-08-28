# B1-v2 Pre-registration — Fixed-Compute Paired Progressive Temporal-Gap iMeanFlow

Written 2026-08-28 **before any scientific B1 training**; frozen by the commit that introduces it. First method-intervention
experiment of the programme (the A-line established the phenomenon; B1 tests one training-dynamics intervention).

## Research question
> Under identical architecture, data, optimization, random streams, training compute, and one-step inference cost, does progressive
> temporal-gap weighting improve iMeanFlow's one-step conditional ECG morphology relative to vanilla iMeanFlow?
Secondary (mechanistic): does staged temporal-gap supervision change the optimisation dynamics consistently with improved long-gap
one-step transport?

## Why B1-v1 was revised — the early-stopping confound
Our frozen iMF protocol early-stops (patience 20); historically iMF stopped at rounds 81/36/66 of 300. A curriculum whose horizon is
the full budget would then be evaluated at s ≈ 0.7–0.9, with the h ≈ 1 gap that 1-NFE inference uses still down-weighted — a negative
result could mean "curriculum ineffective" or "curriculum never completed", indistinguishably. B1-v2 removes the confound:
**fixed-compute paired training** (early stopping never terminates; it runs as a diagnostic only), and the **primary comparison is the
FINAL checkpoint at exactly T_train steps in both arms**.

## Source audit (docs/B1_GAP_CURRICULUM_SOURCE_AUDIT.md; arXiv:2511.19065 v2 / CVPR 2026; no official code found)
Exact source equation, adopted verbatim: **β(h, s) = 1 − s + λ·s·(1 − h)**, h = t − r; **s = 1 − i/T (k = 1 linear, the paper's
choice in all experiments)** with T = the total planned iterations; **λ = 1/E[1 − h]** (mean β = 1 at initialisation); **β multiplies
the adaptive-weighted non-boundary u-loss only and is applied strictly AFTER the adaptive normalisation** (App. B Eq. 8/9, verbatim
"we strictly apply these weightings after the adaptive normalization"); the boundary (r = t) term carries no β.

## Isolated intervention and controlled deviations from the source
The source's full method = (1) accelerated v-learning (α(t) boundary weighting or p_acc(t) sampling) + (2) progressive L_u weighting.
**B1-v2 implements (2) only** — a *source-inspired isolated progressive-gap intervention*. Not implemented: α(t), p_acc(t)/DTD, the
DiT/ImageNet stack. Baseline objective is our frozen **Improved** MeanFlow (V-parameterisation, h-only conditioning, 50 % r = t;
identical adaptive-weight form p = 1, c = 0.01) rather than original MeanFlow. Boundary β ≡ 1 in both arms (matches Eq. 9's unweighted
boundary; the α component is deliberately excluded). The paper's headline gains (FID 3.43→2.87 with the combination) are NOT the
expected effect size; the isolated-component ablation (11.58→10.98 on DiT-B/4) is the closer analogue.

## Fixed training budgets (artifacts/b1_gap_curriculum/fixed_budget.json; frozen)
T_train = the optimizer-update count implied by each frozen config's ORIGINAL 300-round budget; T_schedule = T_train (source: T = total
iterations):
| Dataset | rounds | steps/epoch | **T_train = T_schedule** |
|---|---|---|---|
| DaLiA S2 (A2 protocol) | 300 epochs | 220 | **66,000** |
| DaLiA S1 (A3 protocol) | 300 epochs | 218 | **65,400** |
| WildPPG (A4 protocol) | 300 × min(epoch, 220) | 4,583 | **65,482** |
Early stopping (frozen criterion: fixed-bank iMF MSE, 4 banks seed 1000, min_delta 1e-4, patience 20) runs unchanged as a
**diagnostic**: the best-validation checkpoint and the round where the historical rule would have fired are recorded; training always
completes T_train unless a hard numerical failure (NaN/Inf/OOM) occurs, which is then recorded as a failure (no rescue, no tuning).

## λ calibration (artifacts/b1_gap_curriculum/curriculum_calibration.json; frozen before training)
Deterministic MC over the frozen `sample_tr(p_mean −0.4, p_std 1.0, data_proportion 0.5)` **non-boundary** distribution: seed 12345,
1,000,000 samples → E[h] = 0.233504, E[1 − h] = 0.766496, **λ = 1.304639**; mean β at s = 1 = 1.000000 (unit-tested, tolerance 0.01).
No alternative λ is trained.

## Paired randomness
Both arms run the SAME new driver (`src/ppg2ecg/training/train_b1_fixed_compute.py`, derived from the frozen A2 trainer) with the same
seed 42: identical initialisation, dataloader order, (t, r) stream, boundary masks and Gaussian e. β is a deterministic function of
(h, step) and consumes no RNG (unit-tested). A probe hashes the first 64 micro-batches (data, targets, t, r, mask, e); the two arms'
hashes must match (`paired_randomness_probe.json`; verified in the integration test and asserted per run).

## Frozen model / optimizer
PENGUIN S5 (4,568,707 / 4,304,513 effective), h-only conditioning, boundary v_θ, p = 1, c = 0.01, logit-normal(−0.4, 1), 50 % r = t,
forward-mode JVP, micro-batch 32 × 2, AdamW 1e-3 / wd 0.01 / batch 64 / seed 42. The ONLY difference between arms is β.

## Runs (6, all mandatory; no result-dependent gating)
`b1v2_{vanilla,curriculum}_fixed_dalia_s2_seed42` (A2 protocol), `…_dalia_s1_seed42` (A3), `…_wildppg_seed42` (A4 original
window-normalised representation — NOT A9 global-z). Historical A2/A3/A4 iMF, OT-1/OT-50, MSE-proxy artefacts are context only;
the primary causal pair is always new-vanilla vs new-curriculum.

## Checkpoints and evaluation
PRIMARY: the FINAL checkpoint at exactly T_train (both arms). SECONDARY: the frozen-criterion best-validation checkpoint. Fraction
checkpoints at 0/10/25/50/75/100 % of T_train for diagnostics. Evaluation = the frozen A2/A4 pipeline (`scripts/eval_a2.py`) on the
frozen test windows, paired noise seed 0, derangement seed 1 (WildPPG: 4,096-subset); primary inference NFE = 1 (2/4 diagnostic only).
Metrics: the frozen set (HR, morphology, amplitude ratio + fidelity −|amp−1|, conditioning gain, beats/ref, HF, RMSE/MAE, latency,
params, NFE; WildPPG + R-peak P/R/F1, RR MAE; QRS ±100 ms analyses). Test data never selects checkpoints.

## Temporal-gap diagnostics (mandatory)
Per-round training-batch statistics in the frozen h bins [0,.1) [.1,.3) [.3,.5) [.5,.7) [.7,1] + boundary: count, h/β/w means,
effective weight β·w, δ², ‖u‖, ‖du/dt‖, ‖h·du/dt‖ (`gap_bins_train.csv`); per-round schedule state (s, β at h = .05/.25/.5/.75/.95)
(`schedule_state.csv`); fixed-checkpoint diagnostics at the 0/10/25/50/75/100 % checkpoints, the best-validation and the final
checkpoint via `scripts/imf_diagnostics_a8.py` on the fixed 512-window bank.

## Success criteria (frozen; FINAL checkpoints; ΔX = curriculum − vanilla)
**SUCCESS** requires ALL of: (A) ΔMorph ≥ +0.03 on ≥ 2/3 conditions; (B) no condition with ΔMorph < −0.03; (C) HR not >5 % worse than
vanilla on any dataset unless (Morph gain ≥ +0.05 and HR ≤ 1.10 × historical OT-50 HR); (D) amplitude fidelity −|amp−1| not degraded by
> 0.05; (E) conditioning gain not decreased by >10 % relative (unless ill-conditioned — then raw differences, judged separately);
(F) WildPPG ΔF1 ≥ −0.03; (G) latency within +10 % of vanilla; (H) identical params and NFE.
**STRONG FRONTIER WIN** (per dataset): curriculum-final Morph ≥ OT50 Morph − 0.03, HR ≤ 1.10 × OT50 HR, amp fidelity ≥ vanilla − 0.05,
Cond ≥ 0.90 × vanilla, and (WildPPG) F1 ≥ vanilla − 0.03; strong overall if on ≥ 2/3 conditions.
**SELECTION-SENSITIVE**: best-validation meets SUCCESS but final does not (not a clean success).
**DATASET-SPECIFIC** (only one condition materially benefits) / **TRADE-OFF** (morphology up but a safety threshold violated) /
**NULL** (ΔMorph < +0.03 on ≥ 2/3 and no compelling paired diagnostic advantage) / **HARMFUL** / **UNINTERPRETABLE**.

## Statistics
Single training seed 42 — no between-seed significance claims. Paired per-window deltas with bootstrap CIs (WildPPG: subject/site-aware
cluster bootstrap); stated limitation: bootstrap does not measure between-training-seed variance. Effect-size thresholds are primary.

## Hypotheses
H-B1.1 ΔMorph ≥ +0.03 on ≥ 2/3 at equal fixed compute. H-B1.2 gain at identical params/inference. H-B1.3 no material
HR/amplitude/conditioning sacrifice. H-B1.4 WildPPG morphology gain without ΔF1 < −0.03. H-B1.5 diagnostics show early small-gap
emphasis that decays per the schedule. H-B1.6 curriculum-final moves toward the multistep structural reference without reverting toward
MSE/OT-1 attenuation. H-B1.7 the conclusion is free of the early-stop-time confound (paired fixed compute).

## No-tuning rule
After any result: no change of λ, T_schedule, k, sampler, boundary fraction, LR/WD/batch, budget, checkpoint rule, seed,
representation; no loss/architecture additions; no seed-43/44 rescue; bug fixes only for objectively demonstrated implementation bugs,
documented before rerun, with affected results invalidated wholesale. After B1-v2: **STOP** and report (no DTD/AlphaFlow/CMT/
Re-MeanFlow/KAN/global-z-B1/multi-seed without a new pre-registration).

## Limitations (declared)
Single seed; the vanilla arm's fixed-budget endpoint may differ from the historical early-stopped iMF (that is the point of retraining
it); ~81 GPU-hours total; curriculum wall-clock overhead expected ≈ 1 (β is a scalar multiply), verified in compute logs.
