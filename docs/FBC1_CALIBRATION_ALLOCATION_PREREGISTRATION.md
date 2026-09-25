# FBC1 — Functional Budget Calibration: small-calibration static allocation — preregistration

Frozen and pushed **before any FBC1 risk, selection, regret or calibration-subset result is computed** on validation or
test (only label-free audit quantities — shapes, non-finite rates, sha256 — have been looked at). Audit:
`docs/FBC1_CALIBRATION_ALLOCATION_AUDIT.md`. Head at freeze: `d721a0b`. M1 and M2 are not modified; M3 stays closed.
Seed: **20260925**. Code: `scripts/fbc1_run.py` (stages audit → grid → fullval → nested → test → latency → figure),
`scripts/fbc1_val_bank.py`; pipeline tests `tests/test_fbc1_selection.py`, `tests/test_fbc1_pipeline_smoke.py`.

## 1. Question
Given a frozen stochastic conditional generator G, the ECG→HR functional T, a fixed generative **vector-field evaluation
budget** B = K × S (NFE per window) and a small labeled calibration set, can one **single allocation (K\*, S\*)**, chosen on
the calibration set and then used for every future window, achieve near-optimal held-out risk — and how many labeled
calibration patients does that take? This is static, calibration-time allocation. There is no per-instance adaptation:
M2 (`77764da`) closed that line, since pilot observables do not predict the per-window width-vs-depth advantage beyond the
static model/depth information.

## 2. Data-use disclosure
VitalDB **TEST** (1,156 patients) is **legacy**: DW1, DW2, EXP-B, M1 and M2 analysed it, and the candidate grid, the
historical static recipes and the width-over-depth finding all come from it. FBC1 therefore separates:
- **A. Primary methodological evidence:** patient-level calibration → held-out evaluation **inside VitalDB VALIDATION**
  (289 patients, 4,822 windows) by prospective resampling (§6). The validation split was used before for M1/M2 fitting at
  S ≤ 8 with ≤ 16 draws and for VM1 CFG choice. The B = 16 / B = 32 allocation risks of FBC1 have not been computed on it.
- **B. Secondary frozen confirmation:** after `frozen_method.json` is committed, the validation-selected allocations are
  applied unchanged to legacy TEST. That section is titled **"Legacy-test confirmation, not fresh prospective
  validation"** and changes nothing.

## 3. Generators, grid, draws
- iMF, CD and PENGUIN Euler, seed-42 checkpoints, with the unchanged DW1 samplers (`dw1_depth_width.make_sampler`).
- HR = `v1_evaluate._hr`, snapped to 1e-3 bpm.
- **B = 32 (primary):** (32,1), (16,2), (8,4), (4,8), (2,16), (1,32).
- **B = 16 (secondary):** (16,1), (8,2), (4,4), (2,8), (1,16).
- Draws: noise seeds 0 … K−1 at depth S (prefix reuse; row k = `torch.Generator().manual_seed(k)` over the whole split).
- Validation bank: the M1 bank (seeds 0–15 at S = 1, 2, 4, 8) plus FBC1 generation of the missing draws: S = 1 seeds 16–31,
  S = 16 seeds 0–1, S = 32 seed 0. Frozen checkpoint and sampler, no training. Seed 15 at S = 1 regenerated bit-identically.
- Test: the DW1 bank. Every cell exists, so there is no new test generation.

## 4. Risk
- Window loss l = |median of the finite draw HRs − reference HR|.
- If no draw is finite, the estimate is the frozen fallback **c_train = 73.143 bpm**: the median reference HR of the VitalDB
  **training** windows (DB1 labels, `outputs/db1_hr_regressor/train_hr_labels.npz`). It uses no validation or test label.
- Patient loss L_{p,a} = mean over the patient's windows.
- **Primary risk: patient-macro** R_a = mean_p L_{p,a}. **Secondary: window-micro.** Macro is never replaced by micro.
- DW1's historical convention (windows with no finite draw dropped) is reported descriptively for the full-validation and
  legacy-test landscapes only.

## 5. Informativeness gate (validation; computed with the full-validation landscape, before any subset result)
The gate is applied to every model, budget and cell, with a 2,000-replicate patient bootstrap, `default_rng([20260925, 2])`.
- **Gate A:** MAE(consensus) − MAE(constant), where the constant is the validation median reference HR (calibration-derived,
  in-sample, i.e. conservative). The CI upper bound must be < 0.
- **Gate B:** window-level Spearman(consensus HR, reference HR) with a patient-clustered bootstrap. The CI lower bound must be
  > 0.

A model passes a budget if every cell passes both gates. A model that fails at B = 32 is reported but cannot count as
meeting criteria A, B or D (§12).

## 6. Calibration subsets (manifest committed with this document)
- Sizes: n ∈ {5, 10, 25, 50, 100} is the primary curve; 200 is secondary. n = 289 is never a small-calibration result.
- **200 subsets per n**, drawn without replacement from the 289 validation patients.
- Rule: `rng = default_rng(20260925)`; for n in (5, 10, 25, 50, 100, 200) in this order, repeat
  `sorted(rng.choice(289, n, replace=False))` over the sorted patient IDs, skipping duplicates, until 200.
- Saved as `artifacts/fbc1_calibration_allocation/calibration_subsets.json`. The same subsets are used for every model and
  budget, so all comparisons are paired.
- For subset C_r, the held-out set is **E_r = validation patients \ C_r**. Selection sees only C_r; E_r labels never enter
  the selector.

## 7. Selectors (frozen)
- **FBC-ERM:** argmin_a mean_{p∈C} L_{p,a}.
- **FBC-UCB (primary method):** argmin_a [mean_{p∈C} L_{p,a} + t_{0.90, n−1} · SD_{p∈C}(L_{p,a}) / √n], with SD ddof 1.
  The one-sided 90 % Student-t level is frozen and is not tuned.
- **Ties:** |R_a − R_min| ≤ 1e-9 bpm → the lower K.
- The method is not renamed after the results.

## 8. Reference, held-out oracle, regret, baselines
- **Full-validation reference** a_fullval = argmin over all 289 patients. It is an achievable reference for "a large
  calibration set", not an oracle and not a test oracle.
- **Held-out oracle** (descriptive, not deployable): a_oracle,r = argmin_a R_{E_r}(a).
- **Regret_r = R_{E_r}(a_selected,r) − R_{E_r}(a_oracle,r) ≥ 0.** This is the primary sample-efficiency metric.
- Known property, stated before the results: the held-out oracle exploits noise in E_r. E_r shrinks from 284 patients
  (n = 5) to 89 (n = 200), so a perfect selector has positive regret that can *grow* with n. The synthetic controls show
  this. FBC1 therefore also reports R_{E_r}(a_selected) − R_{E_r}(a_fullval), which carries no oracle optimism.
- **Baselines per budget:**

  | baseline | B = 32 | B = 16 |
  |---|---|---|
  | pure depth | (1,32) | (1,16) |
  | pure width | (32,1) | (16,1) |
  | balanced | (8,4) | (4,4) |
  | historical static | iMF (16,2), CD (32,1), PENGUIN (8,4) | iMF (8,2), CD (16,1), PENGUIN (8,2) |

  The historical static cells are the frozen DW1 best cells (`artifacts/dw1_depth_width/result.json`; DW1 prereg
  `cc902b3`; seed-stable in DW2 at B = 32). They were derived on legacy TEST, so they are an external prior, not a
  validation selection. The full-validation selector is a reference only.

## 9. Near-optimality margin (frozen at the fullval stage, before any subset result)
Per model and budget:
- Gap_scale = R_val(pure depth) − R_val(a_fullval). If pure depth is not worse, Gap_scale = max_a |R_val(a) − R_val(a_fullval)|.
- **δ_near = clip(0.10 × Gap_scale, 0.05, 0.25) bpm.**

This is an algorithmic tolerance, not a clinical threshold. δ_near, a_fullval and the gate are committed in
`frozen_method.json` (with `fullval_reference.json` and `validation_full_grid.csv`) **before** the nested stage runs. The
code refuses to run the nested stage otherwise.

## 10. Metrics (per model × budget × n × selector)
- Regret: mean, median, and the 2.5–97.5 % range across subsets.
- Near-optimal rate P(Regret ≤ δ_near).
- Materially suboptimal rate P(Regret > 2 δ_near).
- Exact recovery P(a_sel = a_oracle,r).
- Full-validation recovery P(a_sel = a_fullval), and P(R_{E}(a_sel) − R_{E}(a_fullval) ≤ δ_near).
- Mean held-out risk; window-micro regret (secondary).
- Selection frequency of every cell, and selection entropy H = −Σ p log p (nats).
- The same regret, near-optimal and risk quantities for the fixed baselines.
- Paired differences in regret: UCB − ERM, and UCB − each baseline.
- **Paired-risk structure (descriptive only):** in each subset, d_p = L_{p,a1} − L_{p,a2} for the two lowest calibration means,
  with mean, paired SE, 95 % t-CI and the share of subsets whose CI excludes 0. No paired selector is built from this.
- **Patient-clustered uncertainty:** a 2,000-replicate population bootstrap, `default_rng([20260925, 1])`. Each replicate
  resamples the 289 validation patients with replacement into 289 slots, applies the frozen subset manifest to slot positions,
  and reruns selection, held-out risk, held-out oracle and the full-validation selector; δ_near is held fixed. Percentile 95 %
  CIs are reported. In a replicate, copies of one patient can land in both C_r and E_r; this is disclosed.

## 11. Legacy-test confirmation (after `frozen_method.json` is committed)
- Each subset's validation-selected allocation is applied unchanged to all 1,156 test patients.
- Reported: the expected test risk over subsets (macro, with micro secondary), and the test regret against the best test
  cell.
- Compared with pure depth, pure width, balanced, historical static, the full-validation selector and the **test oracle**
  (best test cell; descriptive only).
- CIs: 5,000-replicate test-patient bootstrap, `default_rng([20260925, 3])`, with the selections held fixed.
- Nothing is changed afterwards.

## 12. Criteria (fixed; B = 32, headline n = 25, primary method FBC-UCB)
**STRONG** requires all of:
- **A.** FBC-UCB near-optimal rate ≥ 80 % at n = 25 for ≥ 2 of 3 models.
- **B.** FBC-UCB mean held-out regret at n = 25 is below both the pure-depth and the pure-width regret (point estimates) for
  ≥ 2 of 3 models.
- **C.** Pooled over the 3 models, the paired mean of Regret(UCB) − Regret(ERM) at n = 25 has a point estimate **< 0**
  (favours UCB), and its patient-bootstrap 95 % upper bound is **< 0.025 bpm** (non-inferiority margin = half the smallest
  admissible δ_near).
- **D.** At n = 50, every model has a near-optimal rate ≥ 80 % or a mean regret ≤ its δ_near.

**FAILED** if any of these holds (FBC-UCB, B = 32):
- **F1 (no learning):** for ≥ 2 models, the mean regret at n = 100 is > δ_near and ≥ 0.75 × the mean regret at n = 5.
- **F2 (n = 50 still materially suboptimal):** for ≥ 2 models, P(Regret > 2 δ_near) at n = 50 is > 20 %.
- **F3 (systematically worse than historical static):** for ≥ 2 models at n = 25, the mean Regret(UCB) − Regret(historical)
  is > δ_near with a bootstrap CI lower bound > 0.
- **F4 (too unstable):** for ≥ 2 models at n = 25, the near-optimal rate is < 50 % and the mean regret is > δ_near. Exact-cell
  instability alone, with negligible regret, is not failure.

**PARTIAL** otherwise, labelled with every label that applies:
- "ERM works but UCB does not add value": A, B and D hold for UCB, but C fails.
- "calibration selection works, uncertainty-aware variant unsupported": A, B and D fail for UCB but hold with FBC-ERM in
  place of UCB. FBC-UCB is then not called successful.
- "only 1 of 3 models meets the n = 25 criterion".
- "sample efficiency improves with n but the n = 25 criteria are not met".

**GO / NO-GO for the FBC methodology:**
- GO if STRONG.
- CONDITIONAL GO if PARTIAL and A, B and D hold for UCB, or for ERM. It names the variant that met them.
- NO-GO otherwise.

**B = 16** is evaluated with the same criteria, descriptively. It cannot rescue B = 32. The report states whether the
required n (the smallest n with near-optimal rate ≥ 80 % or mean regret ≤ δ_near) changes with the budget.

## 13. Pipeline validation (synthetic; not evidence)
`tests/test_fbc1_selection.py` covers:
- the tie rule and the UCB formula;
- **A** (clear optimum): recovery rises with n;
- **B** (all equal): entropy is high, regret ≈ 0;
- **C** (near-tie): exact recovery is unstable, near-optimal rate is high;
- **D** (noisy false winner): UCB regret, entropy and near-optimal rate are no worse than ERM.

`tests/test_fbc1_pipeline_smoke.py` runs fullval → nested → test → figure end-to-end on synthetic banks. All pass at freeze.

## 14. Compute accounting
- The primary statement is a **fixed generative vector-field evaluation budget**. B = K·S is not called equal total compute.
- Measured per model and cell:
  - total NFE;
  - sequential batch-1 wall-clock (K calls of S steps);
  - batched wall-clock (one call, batch K);
  - peak GPU memory;
  - the per-sample fixed overhead c0 from the fit T_seq(K, S)/K = c0 + c1·S;
  - the CPU cost of the HR functional per sample (K extractions per window).
- Protocol: one validation window, 10 warm-up runs and 100 timed runs, with no concurrent GPU job.
- FLOPs are recorded only if PyTorch's built-in `torch.utils.flop_counter` works. No FLOP counter is written for FBC1.

## 15. Order of work, artifacts, stop
- **This commit:** audit (doc and `audit.json`), this preregistration, `candidate_grid.json`, `calibration_subsets.json`,
  `prereg_manifest.json`, code and tests.
- **Then:** the `fullval` stage, followed by a commit of `frozen_method.json`, `fullval_reference.json` and
  `validation_full_grid.csv`.
- **Then:** `nested` (primary), `test` (legacy confirmation), `latency`, `figure`, the report, and a commit.
- **HARD STOP.** None of the following follow: a new FBC variant, a paired selector, PPGFlow / WildPPG FBC, other domains, or
  new generators.

Artifacts go in `artifacts/fbc1_calibration_allocation/` (the 16 files of the brief). Banks and selections go in
`outputs/fbc1_calibration_allocation/` and are not committed. The report is `docs/FBC1_CALIBRATION_ALLOCATION_REPORT.md`
(19 sections).

## 16. What FBC1 can establish at most
A small labeled calibration set can or cannot reliably identify a near-optimal static depth/width allocation for
ECG-derived HR under a frozen generator and a fixed vector-field evaluation budget. It makes no claim about:
- cross-domain generality or a universal functional allocation;
- causal mechanisms or instance adaptation;
- optimal FLOP allocation;
- universal width superiority.

The respiration and ABP boundaries from EXP-D are context only. They motivate the usefulness gate and are not FBC evidence.
