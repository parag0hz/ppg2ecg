# EXP-D Part B (ABP) — prospective preregistration amendment: usability / informativeness gate

Dated **2026-09-23**. Frozen and pushed **before any ABP test-set sample is generated or any ABP test result is computed
or inspected** (no MIMIC-BP test output exists at this commit). References:
- original preregistration **`eddbe60`** (`docs/EXP_D_FUNCTIONAL_GENERALIZATION_PREREGISTRATION.md`) — **not modified**;
- respiration result that revealed the failure mode **`a4d66e5`** (`docs/EXP_D_RESPIRATION_REPORT.md` §9).

## Reason
The respiration experiment revealed a failure mode not anticipated in `eddbe60`: a width-heavy consensus can lower MAE by
shrinking the variance of an estimator that carries essentially no information about the reference (BIDMC: no estimator
correlated with the reference RR; a training-median constant beat every cell). "Width beats depth" alone is therefore not
sufficient evidence of functional generalisation.

## What is unchanged
Every element of `eddbe60` for Part B: checkpoint, data, split, cells, budgets, K / S allocations, functional definitions
(SBP = block max, DBP = block min, MAP = block mean, 8-s blocks), waveform metrics, mechanism analyses, bootstrap
(subject-clustered, 5,000 replicates, seed 20260924), the preregistered training-median sanity baseline, and the original
ABP verdict categories and rules (§8). They are computed and reported exactly as frozen. The only addition is a gate that
governs **interpretation**.

## Usability gate (SBP, DBP, MAP evaluated separately)
Unit: the preregistered 8-s test block (valid = reference and consensus both finite).
- **Consensus used by the gate:** the preregistered B = 32 intermediate cell **(K = 8, S = 4)** (nested draws 0–7 at S = 4,
  median) — preregistered before any ABP result, not the possibly collapsed S = 1 extreme, and the "refine enough, then
  sample wide" allocation.
- **Training-only constant:** b_F = median of F over the 8-s blocks of the **training subjects only** (blocks formed within
  subject, trailing windows dropped — the constant already preregistered in `eddbe60` §4). The test set is not used to
  choose it.
- **Gate A — baseline superiority.** Δ_const(F) = MAE_(8,4)(F) − MAE_const(F), paired, subject-clustered bootstrap
  (as every contrast of `eddbe60`). **PASS only if the upper 95 % CI bound of Δ_const(F) < 0.**
- **Gate B — target association.** Spearman correlation, across valid test blocks, between the (8,4) consensus F and the
  reference F\*; subject-clustered bootstrap (subjects resampled with replacement, all of each drawn subject's blocks,
  Spearman recomputed; 5,000 replicates, seed 20260924, percentile CI). **PASS only if the lower 95 % CI bound > 0.**
  Pearson is reported as secondary and does not replace the Spearman gate. Waveform correlation is **not** used (phase
  alignment is not required for SBP / DBP / MAP to carry amplitude information).
- **Functional usability:** **USABLE** = A and B pass; **PARTIALLY INFORMATIVE** = exactly one passes; **UNINFORMATIVE** =
  both fail. Only a USABLE functional can count as evidence (for or against). Thresholds are not changed after results.

## Two separate verdicts per functional
1. **Allocation verdict** — the original frozen rule of `eddbe60` §8: *success* = at B = 32, (32,1) − (1,32) or (8,4) − (1,32)
   has a CI entirely below 0; *depth-clear* = both B = 32 contrasts have CIs entirely above 0; otherwise *no clear
   difference*.
2. **Evidentiary verdict** — the allocation verdict read through the gate:
   | usability | allocation | evidentiary verdict |
   |---|---|---|
   | USABLE | success | **SUPPORTS FUNCTIONAL GENERALISATION** |
   | USABLE | depth-clear | **GENUINE COUNTEREXAMPLE** to the width-heavy principle |
   | USABLE | no clear difference | **NO EFFECT on a usable functional** |
   | PARTIALLY INFORMATIVE or UNINFORMATIVE | success | **NOT EVIDENCE FOR GENERALISATION** (respiration failure mode) |
   | PARTIALLY INFORMATIVE or UNINFORMATIVE | depth-clear | **NOT EVIDENCE** (not counterevidence either) |
   | PARTIALLY INFORMATIVE or UNINFORMATIVE | no clear difference | **UNINTERPRETABLE / NON-INFORMATIVE TASK** |
   The two verdicts are always reported side by side, never merged.

## Cross-functional evidentiary criterion (in addition to the original ABP categories, which stay as frozen)
Evaluated in this order (first match):
1. **COUNTEREVIDENCE** — ≥ 2 USABLE functionals are depth-clear.
2. **STRONG CROSS-FUNCTIONAL SUPPORT** — ≥ 2 functionals are USABLE **and** allocation-success.
3. **PARTIAL CROSS-FUNCTIONAL SUPPORT** — exactly 1 USABLE functional is allocation-success, **or** ≥ 2 USABLE functionals
   have a B = 32 width point estimate < 0 without a success (consistent direction, uncertain CIs).
4. **NO SUPPORT** — otherwise (no USABLE functional shows the predicted effect).
A functional failing the gate counts neither for support nor for counterevidence.

## Respiration: retrospective label (interpretation only)
The frozen respiration statistical verdict is **not altered: STRONG** (`a4d66e5`). Applying this gate retrospectively, for
interpretation only, with the numbers already committed in `a4d66e5` (`respiration/posthoc.json`, BIDMC, (8,4) median):
Gate A: (8,4) − constant = **+0.781 [+0.406, +1.188]** → FAIL; Gate B: Spearman −0.112 [−0.309, +0.027] → FAIL →
**UNINFORMATIVE**. Evidentiary label: **usability gate FAIL — respiration / RR is not evidence for cross-functional
generalisation.** This retrospective gate does not change the preregistered statistical result.

## Not changed / not added
No new ABP cell, no K / S change, no change to SBP / DBP / MAP definitions, no extraction threshold (none exists), no direct
BP model, no new baseline (the constant is the one preregistered in `eddbe60`), no removal of failed functionals.
Implementation of the gate: `scripts/expd/expd_abp_gates.py`, committed with this amendment. After this commit: Part B as
preregistered, then **HARD STOP B**.
