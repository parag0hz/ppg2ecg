# FBC1 — Functional Budget Calibration: **PARTIAL** — calibration selection works, uncertainty-aware variant unsupported

**Commits.** Preregistration **`bf49397`** (with the audit, candidate grid, calibration-subset manifest, code and synthetic
tests). Frozen method **`b27b67f`**: gate, full-validation reference and δ_near, committed before any calibration-subset
result and before any test read.

**Primary evidence** comes from the VitalDB **validation** split: 289 patients and 200 calibration subsets per size. Each
allocation is chosen on the subset and evaluated on the remaining patients. **Secondary** evidence is a frozen
legacy-test confirmation.

**Files.** Code `scripts/fbc1_run.py` and `scripts/fbc1_val_bank.py`. Artifacts are in
`artifacts/fbc1_calibration_allocation/`, all 16 files of the brief. Banks and selections are in
`outputs/fbc1_calibration_allocation/` (not committed).

**Scope.** No generator was trained. M1 and M2 are unchanged. M3 stays closed.

## 1. Research question
Take a frozen stochastic generator G, the ECG→HR functional and a fixed generative **vector-field evaluation budget**
B = K × S. Can a single allocation (K\*, S\*), selected on a small labeled calibration set and then used for every future
window, reach near-optimal held-out patient-macro HR MAE? How many calibration patients does that take?

This is static, calibration-time allocation, with no per-window routing. Two selectors were preregistered:
- **FBC-UCB (primary):** argmin_a of mean + t₀.₉₀,ₙ₋₁·SD/√n of the calibration patients' losses.
- **FBC-ERM (baseline):** argmin_a of the calibration mean.

Ties go to the lower K.

## 2. Why M2 ended instance adaptation
M2 (`77764da`) found three things at equal future vector-field NFE:
- A 4-sample pilot did not predict whether width or depth would win on a given window (within-condition Spearman +0.009).
- Every pilot-conditioned rule collapsed to always-width.
- The adaptive policy was significantly worse than the static (model, depth) rule, by +0.015 bpm.

The only usable information was **static**: which allocation is best for a given generator. FBC1 asks whether that static
choice can be *learned from a few labeled patients*, rather than read off a large legacy test set as DW1 did.

## 3. Audit (`docs/FBC1_CALIBRATION_ALLOCATION_AUDIT.md`, `audit.json`)
**Generators.** iMF, CD and PENGUIN-Euler, seed-42 checkpoints (sha256 recorded), with the DW1 samplers. Draw k is noise
seed k, and each allocation uses seeds 0…K−1 (prefix reuse).

**Validation bank.**
- The M1 bank covers S = 1–8 with 16 draws.
- FBC1 generated the missing draws: S = 1 seeds 16–31, S = 16 seeds 0–1, and S = 32 seed 0.
- Seed 15 was regenerated and matched the existing bank bit-for-bit for all three models.

**Test bank.** Every cell already existed in the DW1 grid, so no test sample was generated.

**Coverage.** Every B = 32 and B = 16 cell is exactly reconstructable on both splits.

**Missing HR.**
- Depth-heavy cells have windows with no finite draw: iMF (1,32) 3.6 % (test 3.8 %), iMF (2,16) 1.3 %, CD (1,32) 1.4 %.
- These windows receive the frozen fallback **73.143 bpm**, the median reference HR of the training split, so no
  validation or test label is used.
- DW1 dropped such windows instead. As a result the legacy-test risk of iMF pure depth is 8.60 here versus 8.10 under the
  DW1 convention (both are reported).

## 4. Candidate allocation grid
- **B = 32 (primary):** (32,1), (16,2), (8,4), (4,8), (2,16), (1,32).
- **B = 16 (secondary):** (16,1), (8,2), (4,4), (2,8), (1,16).

Fixed baselines:

| baseline | B = 32 | B = 16 |
|---|---|---|
| pure depth | (1,32) | (1,16) |
| pure width | (32,1) | (16,1) |
| balanced | (8,4) | (4,4) |
| historical static (frozen DW1 best, derived on legacy test) | iMF (16,2), CD (32,1), PENGUIN (8,4) | iMF (8,2), CD (16,1), PENGUIN (8,2) |

## 5. Functional Budget Calibration
- **Window loss:** |median of finite draw HRs − reference HR|.
- **Patient loss:** L_{p,a}, the mean of the window losses.
- **Risk:** the mean over patients (patient-macro, primary). Window-micro is secondary.
- **Calibration subsets:** 200 per n ∈ {5, 10, 25, 50, 100, 200}, drawn without replacement (seed 20260925, manifest
  committed). The same subsets are used for every model and budget, so all comparisons are paired.
- **Held-out evaluation:** selection sees only C_r and is evaluated on E_r (the other patients).
- **Regret** = R_{E_r}(selected) − min_a R_{E_r}(a). The held-out oracle is descriptive only.
- **Uncertainty:** 2,000-replicate patient population bootstrap. Each replicate resamples the 289 patients and reruns the
  whole procedure.
  - Known caveat: a duplicated patient can land in both C_r and E_r, which biases selection metrics slightly optimistic.
  - Consequently a few percentile CIs do not centre on the point estimate. Point estimates come from the actual data.

## 6. Informativeness gate (`validation_full_grid.csv`)
The gate passes for **every model, budget and cell**:
- The consensus HR beats the validation-median constant (73.317 bpm, patient-macro MAE 13.684) by **4.53–7.67 bpm**. All
  CI upper bounds are < 0.
- The window-level Spearman with the reference is **0.569–0.722**, with every CI lower bound > 0.

All three models therefore count as usable HR functionals. Unlike respiration and ABP in EXP-D, there is a real functional
to calibrate.

## 7. Full-validation risk landscape (`fullval_reference.json`, frozen at `b27b67f`)
Patient-macro HR MAE (bpm) on all 289 validation patients:

| model | (32,1) | (16,2) | (8,4) | (4,8) | (2,16) | (1,32) | a_fullval | best − 2nd (paired 95 % CI) | δ_near |
|---|---|---|---|---|---|---|---|---|---|
| iMF | 6.465 | **6.117** | 6.479 | 7.164 | 8.265 | 8.365 | (16,2) | −0.348 [−0.488, −0.209] vs (32,1) | 0.225 |
| CD | **6.010** | 6.263 | 6.479 | 6.825 | 7.767 | 8.123 | (32,1) | −0.253 [−0.407, −0.099] vs (16,2) | 0.211 |
| PENGUIN | 7.114 | **6.233** | 6.245 | 6.687 | 8.214 | 9.152 | (16,2) | **−0.011 [−0.141, +0.118]** vs (8,4) | 0.250 (ceiling) |

| model, B = 16 | (16,1) | (8,2) | (4,4) | (2,8) | (1,16) | a_fullval | best − 2nd | δ_near |
|---|---|---|---|---|---|---|---|---|
| iMF | 6.670 | **6.420** | 7.020 | 8.309 | 8.396 | (8,2) | −0.250 [−0.412, −0.087] | 0.198 |
| CD | **6.069** | 6.578 | 7.011 | 7.729 | 8.144 | (16,1) | −0.509 [−0.699, −0.320] | 0.208 |
| PENGUIN | 7.195 | **6.399** | 6.678 | 7.741 | 8.732 | (8,2) | −0.279 [−0.461, −0.098] | 0.233 |

- **Shape of the landscape.** Pure depth is worst by 2.0–2.9 bpm. The optimum is width-heavy but interior for iMF and
  PENGUIN, and pure width for CD. This matches DW1, but here it was derived on validation.
- **Near-tie.** At B = 32, PENGUIN's (16,2) and (8,4) are statistically indistinguishable.
- **Window-micro.** The best cell is the same under window-micro risk in all 6 model × budget cases.

## 8. Calibration sample efficiency (B = 32; `regret_curves.csv`, `bootstrap.json`)
Mean held-out regret in bpm, with the patient-bootstrap 95 % CI. Near-optimal rate is P(regret ≤ δ_near) over the 200
subsets.

| n | iMF UCB | iMF ERM | CD UCB | CD ERM | PENGUIN UCB | PENGUIN ERM |
|---|---|---|---|---|---|---|
| 5 | 0.351 [0.250, 0.488] · 44 % | 0.211 [0.161, 0.281] · 53 % | 0.403 [0.270, 0.493] · 45 % | 0.221 [0.154, 0.291] · 57 % | 0.475 [0.323, 0.691] · 59 % | 0.265 [0.173, 0.379] · 73 % |
| 10 | 0.140 · 67 % | 0.123 · 70 % | 0.170 · 60 % | 0.107 · 70 % | 0.373 · 72 % | 0.127 · 83 % |
| **25** | **0.085 [0.032, 0.098] · 78.5 %** | **0.062 [0.021, 0.082] · 84.5 %** | **0.095 [0.042, 0.146] · 76.5 %** | **0.042 [0.023, 0.083] · 88 %** | **0.076 [0.037, 0.153] · 89.5 %** | **0.050 [0.028, 0.099] · 95 %** |
| 50 | 0.020 · 95.5 % | 0.016 · 96.5 % | 0.038 · 90 % | 0.022 · 94 % | 0.042 · 96.5 % | 0.023 · 100 % |
| 100 | 0.003 · 99.5 % | 0.003 · 99.5 % | 0.010 · 97.5 % | 0.004 · 99 % | 0.034 · 100 % | 0.034 · 100 % |
| 200 | 0.000 · 100 % | 0.000 · 100 % | 0.000 · 100 % | 0.000 · 100 % | 0.078 · 100 % | 0.077 · 100 % |

Pooled over the three models:

| n | 5 | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| mean regret, UCB | 0.410 | 0.228 | 0.085 | 0.033 | 0.016 | 0.026 |
| mean regret, ERM | 0.232 | 0.119 | 0.051 | 0.020 | 0.014 | 0.026 |
| near-optimal rate, UCB | 49 % | 66 % | 81.5 % | 94 % | 99 % | 100 % |
| near-optimal rate, ERM | 61 % | 74 % | 89 % | 97 % | 99.5 % | 100 % |

**Reading the curve.**
- Regret drops by roughly 5× from n = 5 to n = 25, and to ≈ 0 by n = 100. Median regret is 0 from n = 10 for iMF and CD.
- The n = 5 mean is carried by a tail: the 97.5th percentile of regret across subsets is 2.1–3.0 bpm for UCB and
  1.1–2.0 bpm for ERM, meaning some 5-patient subsets pick a depth-heavy cell.
- **Smallest n with near-optimal rate ≥ 80 % or mean regret ≤ δ_near:**

  | | iMF | CD | PENGUIN |
  |---|---|---|---|
  | UCB | 10 | 10 | 25 |
  | ERM | 5 | 10 | 10 |

- **PENGUIN's rise at n = 200 (0.034 → 0.078) is not a failure of selection.** It is the preregistered held-out-oracle
  optimism: E_r holds only 89 patients, and (16,2) and (8,4) are tied, so the oracle picks whichever is luckier. Median
  regret at n = 200 is 0.068, and the near-optimal rate is 100 %.
- **Window-micro regret** tracks macro to within 0.01 bpm in every cell of the table.

## 9. FBC-UCB vs FBC-ERM (paired on identical subsets)
Pooled mean of Regret(UCB) − Regret(ERM), B = 32:

| n | 5 | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| point estimate | +0.178 | +0.109 | **+0.034** | +0.013 | +0.002 | +0.000 |
| 95 % CI | [0.106, 0.250] | [0.036, 0.143] | **[0.007, 0.051]** | [0.001, 0.029] | [−0.001, 0.013] | [−0.003, 0.007] |

**UCB never beats ERM.** It is significantly worse through n = 50, and the two converge once both almost always pick
a_fullval. The same holds per model at n = 25: iMF +0.023 [−0.005, 0.032], CD +0.053 [0.007, 0.076], PENGUIN +0.026
[−0.000, 0.075]. It also holds at B = 16, pooled +0.032 [0.007, 0.074] at n = 25.

**Why, descriptively and post hoc** (`paired_risk.csv`; no new selector is built from this):
- **The UCB penalty is far larger than the gaps it has to resolve.** Between-patient SD of L is ≈ 6.0–7.3 bpm for every
  cell, so at n = 25 the penalty t·SD/√n is ≈ 1.7 bpm. The winner-minus-runner-up *paired* SE on the same patients is only
  0.22–0.25 bpm, against full-validation gaps of 0.25–0.35 bpm (iMF, CD).
- **What the SD measures.** Patient difficulty is shared by all cells, so the marginal SD mostly reflects that common
  difficulty rather than which allocation is better. Sampling noise in the SD estimate therefore enters the choice as extra
  noise.
- **Systematic bias for CD.** The best CD cell (32,1) has a larger spread than its nearest competitors (6.73 vs 6.56 for
  (16,2) and 6.49 for (8,4)). In the 24 of 200
  n = 25 subsets where UCB and ERM disagree, UCB always moves to a lower-K cell, with +0.44 bpm mean extra regret in those
  subsets.
- **The synthetic control where UCB was designed to win (§13 of the preregistration).** That control had a single
  high-variance false winner and *independent* arm noise. It does not describe these data. Here the losses are strongly
  shared across arms, and the variances are nearly equal.

## 10. Allocation stability (`selection_frequency.csv`)
Selection frequencies at n = 25, B = 32:

| model | method | (32,1) | (16,2) | (8,4) | (4,8) | (2,16) | (1,32) |
|---|---|---|---|---|---|---|---|
| iMF | UCB | 12.5 % | **78.5 %** | 9.0 % | 0 | 0 | 0 |
| iMF | ERM | 8.0 % | **84.5 %** | 7.5 % | 0 | 0 | 0 |
| CD | UCB | **76.5 %** | 13.5 % | 8.5 % | 1.5 % | 0 | 0 |
| CD | ERM | **88.0 %** | 9.0 % | 3.0 % | 0 | 0 | 0 |
| PENGUIN | UCB | 1.5 % | **44.0 %** | 45.5 % | 9.0 % | 0 | 0 |
| PENGUIN | ERM | 1.5 % | **48.0 %** | 47.0 % | 3.5 % | 0 | 0 |

**Selection entropy (nats), UCB / ERM:**

| n | iMF | CD | PENGUIN |
|---|---|---|---|
| 5 | 1.38 / 1.15 | 1.50 / 1.19 | 1.60 / 1.39 |
| 25 | 0.67 / 0.54 | 0.75 / 0.43 | 1.00 / 0.89 |
| 100 | 0.03 / 0.03 | 0.12 / 0.06 | 0.69 / 0.69 |

- **At B = 32, neither (2,16) nor pure depth (1,32) is ever selected from n = 25 on.** At B = 16, pure depth is picked
  once (0.5 %, UCB, PENGUIN, n = 25).
- **PENGUIN's entropy stays at ≈ ln 2 = 0.69.** It is split between two tied cells, so for n ≥ 25 exact recovery of a_fullval is
  only 44–60 % while near-optimality is 89.5–100 %. This is the §21 case: exact-cell instability that costs almost nothing.
- **Paired-risk structure on the calibration set.** At n = 25, the calibration winner is *significantly* better than the
  runner-up in only 2.5–7.5 % of subsets (B = 32). Yet the top two calibration cells contain a_fullval in 91.5–98.5 % of
  subsets.
  Small calibration sets rank the right region reliably but cannot certify the winner. On all 289 patients the paired
  difference is significant for iMF and CD, and not for PENGUIN.

## 11. Held-out regret of the fixed baselines (validation, n = 25 subsets, B = 32)

| baseline | iMF | CD | PENGUIN |
|---|---|---|---|
| pure depth | 2.244 [1.959, 2.555] | 2.114 [1.773, 2.467] | 2.924 [2.505, 3.378] |
| pure width | 0.351 [0.200, 0.488] | 0.000 | 0.886 [0.657, 1.166] |
| balanced (8,4) | 0.365 | 0.469 | 0.016 |
| historical static | 0.000 | 0.000 | 0.016 [0.000, 0.138] |
| full-validation selector (reference) | 0.000 | 0.000 | 0.004 |
| **FBC-UCB** | 0.085 | 0.095 | 0.076 |
| **FBC-ERM** | 0.062 | 0.042 | 0.050 |

- **Against the simple static rules,** both FBC variants beat pure depth by about 2–3 bpm for every model. They beat pure
  width for iMF and PENGUIN. For CD, pure width *is* the optimum, so no selector can have lower regret there (criterion B
  fails for CD by construction).
- **Against the historical static cells,** FBC-UCB is worse by 0.06–0.10 bpm (point estimates) at n = 25:
  - iMF +0.085 [0.032, 0.098];
  - CD +0.095 [0.042, 0.146];
  - PENGUIN +0.060 [−0.045, 0.144].

  These cells were derived from 1,156 legacy-test patients, so this is small-n selection versus a large-n prior. The
  excess is below δ_near for every model, so F3 does not fire.

## 12. B = 32 primary result (`near_optimality.json`)
| criterion | FBC-UCB | FBC-ERM (for the PARTIAL label) |
|---|---|---|
| A: near-optimal ≥ 80 % at n = 25 for ≥ 2/3 models | **no** (iMF 78.5 %, CD 76.5 %, PENGUIN 89.5 %; 1 of 3) | yes (84.5 %, 88 %, 95 %) |
| B: regret < pure depth and < pure width for ≥ 2/3 models | yes (iMF, PENGUIN; CD no) | yes (iMF, PENGUIN) |
| C: pooled UCB − ERM point < 0 and upper CI < 0.025 | **no** (+0.034 [+0.007, +0.051]) | — |
| D: by n = 50, near ≥ 80 % or regret ≤ δ_near for all 3 | yes (95.5 %, 90 %, 96.5 %) | yes |

None of the FAILED routes fires:

| route | result |
|---|---|
| F1 (no learning) | no |
| F2 (n = 50 material suboptimality) | no; P(regret > 2δ) ≤ 3.5 % |
| F3 (worse than historical by > δ_near) | no |
| F4 (unstable at n = 25) | no |

**Verdict: PARTIAL**, with three labels:
1. "calibration selection works, uncertainty-aware variant unsupported";
2. "only 1 of 3 models meets the n = 25 near-optimality criterion" (for UCB);
3. "sample efficiency improves with n but the n = 25 criteria are not met" (for UCB).

**GO / NO-GO:** CONDITIONAL GO for the FBC methodology, **FBC-ERM only**. FBC-UCB is not supported.

## 13. B = 16 secondary result (descriptive; cannot rescue or change B = 32)
- **At n = 25**, near-optimal rates are:

  | | iMF | CD | PENGUIN |
  |---|---|---|---|
  | UCB | 67.5 % | 95 % | 64.5 % |
  | ERM | 78.5 % | 97.5 % | 73 % |

  Criterion A therefore fails for *both* selectors at B = 16 (ERM meets it only for CD). B and D hold for both.
- **UCB vs ERM:** pooled UCB − ERM is +0.032 [0.007, 0.074].
- **Pooled mean regret by n, B = 16:**

  | n | 5 | 10 | 25 | 50 | 100 | 200 |
  |---|---|---|---|---|---|---|
  | UCB | 0.422 | 0.266 | 0.099 | 0.023 | 0.006 | 0.001 |
  | ERM | 0.270 | 0.144 | 0.067 | 0.016 | 0.004 | 0.001 |

- **Required n by budget** (smallest n with near-optimal rate ≥ 80 % or mean regret ≤ δ_near):

  | | B | iMF | CD | PENGUIN |
  |---|---|---|---|---|
  | UCB | 32 | 10 | 10 | 25 |
  | UCB | 16 | 10 | 25 | 25 |
  | ERM | 32 | 5 | 10 | 10 |
  | ERM | 16 | 10 | 10 | 10 |

  The requirement is roughly unchanged. B = 16 is, if anything, slightly harder at n = 25, for two reasons:
  - iMF's winning margin is narrower (0.25 vs 0.35 bpm at B = 32).
  - PENGUIN's runner-up (4,4) is 0.28 bpm behind (8,2), which exceeds δ_near = 0.233. At B = 32 the tied runner-up still
    counted as near-optimal; here picking it does not.

  By n = 50 every model at B = 16 is ≥ 89.5 % near-optimal with both selectors.

## 14. Compute accounting (`compute_accounting.json`)
**Budget statement.** Every candidate at budget B uses exactly **B generative vector-field evaluations per window**
(K samples × S steps). The sampler asserts NFE = S per sample. CD also draws S − 1 re-noise tensors, which involve no
network evaluation. B = K·S is *not* called equal total compute.

**Costs outside the vector field.**
- The HR functional runs once per sample on CPU (neurokit R-peak detection), so a window costs **K extractions**. That is
  32 for (32,1) and 1 for (1,32) at B = 32, and it is where width pays extra.
- The consensus median itself is negligible.

**Measured (RTX 5090, one validation window, 10 warm-up + 100 timed runs per cell and mode, median).** The run was
made after the GPU had been idle for 2 minutes, and `nvidia-smi` showed no other compute process at start or at end. An
earlier attempt had started while an unrelated training job of another project held the GPU; it was stopped within
~2 minutes and wrote nothing. Host-side sampler overhead (noise generation, transfer) is included. The HR functional is
timed separately on CPU.

| B = 32 cell | NFE | sequential batch-1 ms (iMF / CD / PENGUIN) | batched ms (iMF / CD / PENGUIN) | peak GPU MiB (batched) | HR functional ms / window (K × ~1.04, CPU) |
|---|---|---|---|---|---|
| (32,1) | 32 | 638 / 648 / 648 | 20 / 21 / 20 | 461 | 33 |
| (16,2) | 32 | 625 / 636 / 632 | 40 / 40 / 40 | 245 | 17 |
| (8,4) | 32 | 623 / 632 / 627 | 78 / 79 / 78 | 137 | 8 |
| (4,8) | 32 | 619 / 630 / 626 | 155 / 158 / 156 | 83 | 4 |
| (2,16) | 32 | 618 / 629 / 625 | 310 / 316 / 308 | 56 | 2 |
| (1,32) | 32 | 621 / 630 / 624 | 619 / 629 / 651 | 42 | 1 |

| B = 16 cell | NFE | sequential batch-1 ms (iMF / CD / PENGUIN) | batched ms (iMF / CD / PENGUIN) | peak GPU MiB (batched) | HR functional ms / window (K × ~1.04, CPU) |
|---|---|---|---|---|---|
| (16,1) | 16 | 322 / 324 / 388 | 20 / 20 / 24 | 245 | 17 |
| (8,2) | 16 | 314 / 317 / 358 | 39 / 39 / 39 | 137 | 8 |
| (4,4) | 16 | 309 / 316 / 313 | 78 / 79 / 77 | 83 | 4 |
| (2,8) | 16 | 310 / 315 / 311 | 155 / 158 / 155 | 56 | 2 |
| (1,16) | 16 | 309 / 315 / 310 | 309 / 316 / 310 | 42 | 1 |

- **Latency per NFE.** One vector-field evaluation costs **≈ 19.4–19.7 ms regardless of batch size** (1 to 32 windows'
  worth). The fit T_seq/K = c0 + c1·S gives c1 = 19.4 / 19.7 / 19.4 ms per NFE and a negligible per-sample overhead c0 of
  0.1 / 0.3 / 1.9 ms. PENGUIN's c0 is inflated by two noisy B = 16 cells, (16,1) and (8,2), whose p10–p90 range is
  314–401 ms.
- **Sequential batch-1 wall-clock is equal across allocations at fixed B** (≈ B × 19.5 ms): 618–648 ms at B = 32 and
  309–388 ms at B = 16. This is the only regime where "fixed NFE" also means "fixed time".
- **Batched wall-clock scales with the number of sequential steps S, not with K.** At B = 32, pure width (32,1) takes
  20 ms and pure depth (1,32) takes 619–651 ms, a ≈ 30× difference.
  - The validation-selected cells cost 40 ms for (16,2) (iMF, PENGUIN) and 20 ms for (32,1) (CD). The historical
    PENGUIN (8,4) costs 78 ms.
  - The FBC choice is therefore also the fast choice when samples are batched.
- **Peak GPU memory (batched) scales with K:** 461 MiB for K = 32 and 42 MiB for K = 1. Sequential runs use 42 MiB.
- **HR functional:** ≈ 1.04 ms per sample on CPU, i.e. 33 ms per window for (32,1) against 1 ms for (1,32). With
  batched sampling, width's total cost (20 + 33 ms) is still ≈ 12× below pure depth (≈ 620 ms).
- **FLOPs:** `torch.utils.flop_counter` counts **3.23 GFLOP per NFE per window**, identical for the three models. They
  share one PENGUIN backbone architecture: 4,568,707 parameters, SSM blocks. That makes **103 GFLOP per window at B = 32
  for every allocation**. The counter covers matmul / conv / attention only; custom ops such as the SSM scan may be
  missed, so treat this as a lower bound.

**Conclusion.** Equal NFE means equal FLOPs and equal sequential batch-1 time. In batched deployment, width-heavy
allocations are much faster but need more memory and more CPU-side HR extractions.

## 15. Legacy-test confirmation — **not fresh prospective validation** (`legacy_test_confirmation.json`)
This section re-uses VitalDB TEST, which DW1, DW2, EXP-B, M1 and M2 already analysed. Each validation-selected allocation
(200 subsets per n) was applied unchanged to all 1,156 test patients. Values are patient-macro test MAE in bpm with a
5,000-replicate test-patient bootstrap.

B = 32:

| | iMF | CD | PENGUIN |
|---|---|---|---|
| pure depth | 8.604 [8.224, 8.995] | 8.287 [7.919, 8.670] | 9.286 [8.944, 9.643] |
| pure width | 6.454 [6.096, 6.832] | 6.138 [5.764, 6.533] | 7.366 [6.964, 7.797] |
| balanced (8,4) | 6.478 | 6.464 | 6.304 |
| historical static | 6.176 (16,2) | 6.138 (32,1) | 6.304 (8,4) |
| **FBC-UCB, n = 25 (expected over subsets)** | **6.238 [5.876, 6.614]** | **6.192 [5.822, 6.579]** | **6.377 [6.002, 6.763]** |
| **FBC-ERM, n = 25** | **6.221 [5.857, 6.597]** | **6.158 [5.787, 6.550]** | **6.356 [5.979, 6.745]** |
| full-validation selector | 6.176 (16,2) | 6.138 (32,1) | 6.349 (16,2) |
| test oracle (descriptive) | 6.176 (16,2) | 6.138 (32,1) | 6.304 (8,4) |

**Test regret at n = 25** (FBC minus test oracle):

| | iMF | CD | PENGUIN |
|---|---|---|---|
| UCB | 0.062 [0.052, 0.072] | 0.054 [0.036, 0.071] | 0.072 [0.051, 0.105] |
| ERM | 0.045 | 0.020 | 0.052 |

**UCB − ERM on test:** +0.017 / +0.033 / +0.021, all CIs > 0.

**FBC-UCB versus the fixed rules on test, n = 25:**
- Pure depth: better by 2.1–2.9 bpm.
- Pure width: better by 0.22 (iMF) and 0.99 (PENGUIN); worse by 0.054 for CD, where pure width is the optimum.

**Test regret by n:**
- UCB, pooled: n = 5 0.37, n = 10 0.20, n = 50 0.023, n = 100 0.009.
- ERM at n = 50: ≤ 0.027 for every model.

**Check on the validation-selected reference.** The full-validation selector matches the test oracle for iMF and CD. For
PENGUIN it picked (16,2), which is +0.044 [0.000, 0.114] above (8,4) on test, the same near-tie.

**B = 16:**
- n = 25: UCB 6.571 / 6.249 / 6.628 and ERM 6.546 / 6.236 / 6.599, against test oracles 6.502 / 6.229 / 6.510.
- The full-validation selector equals the test oracle for all three models.

**Conclusion.** The legacy test reproduces the validation pattern: small-n selection is near-optimal, UCB is slightly
worse than ERM, and pure depth is never competitive. The test was not used for any decision.

## 16. Failed preregistered criteria
- **STRONG item A (FBC-UCB):** only 1 of 3 models reaches 80 % near-optimality at n = 25 (iMF 78.5 %, CD 76.5 %).
- **STRONG item C:** UCB is significantly *worse* than ERM, pooled +0.034 [+0.007, +0.051] bpm. The point estimate does not
  favour UCB and the upper bound exceeds the 0.025 margin.
- **Descriptive B = 16:** item A fails for both selectors.
- **Not failed:** A, B and D hold for FBC-ERM. B and D hold for FBC-UCB. None of F1–F4 fires.

## 17. What FBC1 establishes
For ECG-derived HR with three frozen generators (iMF, CD, PENGUIN-Euler) at a fixed budget of 32 (and 16) vector-field
evaluations per window:
- **A static depth/width allocation can be selected from a small labeled calibration set.** Plain empirical-risk
  selection on 25 patients is within δ_near of the held-out optimum in 84.5–95 % of calibration draws. Its mean regret is
  0.04–0.06 bpm, against 2.1–2.9 bpm for pure depth and up to 0.9 bpm for pure width.
- **Near-certainty takes more patients.** ERM near-optimality reaches ≥ 94 % by n = 50 and ≥ 99 % by n = 100.
- **Exact-cell instability is harmless where cells are tied** (PENGUIN (16,2) vs (8,4)).
- **The simple uncertainty-aware variant does not help.** A marginal one-sided t upper bound is significantly worse than
  ERM at n ≤ 50, because between-patient loss variance is shared across allocations.
- **The same picture holds on the legacy test set**, which is a confirmation, not a fresh test.

## 18. What FBC1 does NOT establish
- **That FBC-UCB works.** The primary method as preregistered is not supported.
- **That the allocation is optimal for true compute.** B = K·S counts vector-field NFE, not FLOPs or wall-clock (see §14).
- **Generality beyond this setting.** Nothing is shown for other functionals (respiration and ABP failed EXP-D's
  usefulness gate), other corpora (WildPPG, PPGFlowECG), other generators or other budgets.
- **Anything about instance adaptation.** M2 stands.
- **Novelty of the selections.** The legacy-test numbers are not fresh evidence. The selected cells coincide with the DW1
  test-derived static cells, so FBC mostly *recovers* a known recipe.
- **That n = 25 suffices in general.** The required n depends on the gap-to-noise ratio of the top cells, which is specific
  to these generators.
- **That a paired or other selector would fix UCB.** This is a descriptive hypothesis only; it was not tested.

## 19. Recommendation: continue FBC vs stop the methodology line
**Continue only in its ERM form, and only as a narrow, honest claim.** The preregistered verdict is PARTIAL with a
CONDITIONAL GO for FBC-ERM.

The defensible methodological statement is: *validate that the functional is informative, then select the static
(K, S) by empirical patient-macro risk on ≥ 25–50 labeled patients.*

**Do not present FBC-UCB as the method.** Its failure is informative: a marginal-SE penalty is the wrong uncertainty for
choosing among allocations evaluated on the same patients.

Any further FBC work would need its own preregistration, and **none is started here (HARD STOP)**:
- a paired-comparison selector;
- a fresh-corpus test, where the historical static recipe is unknown and FBC's value as a *discovery* procedure could be
  measured.
