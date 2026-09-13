# N7 — A calibrated timing posterior over candidate beats: REPORT

Preregistration `docs/N7_CALIBRATED_MARKED_POSTERIOR_PREREGISTRATION.md` (`648838e`, pushed before any N7
number existed). 448 s, 12 runs. `kjd` / `ssx` never loaded. R1's Global-TCN frozen.

---

## 1. Stage verdict: **NOT SUPPORTED** — 3 of 12 runs calibrated

All four preregistered conditions except calibration pass in **every one of the 12 runs**. The coverage
gate fails in 9.

| condition | runs passing |
|---|---|
| (ii) `CONST − FM` on CRPS, CI > 0 | **12 / 12** |
| (iii) sharpness Spearman ≥ 0.20, CI > 0 | **12 / 12** |
| (iv) `FM − FM-SHUFFLE` on CRPS, CI > 0 | **12 / 12** |
| **(i) coverage@50 and @80 within ±0.10** | **3 / 12** |

Nine runs are **SHARP BUT OVERCONFIDENT**; three clear the gate marginally. Stage rule: SUPPORTED needs
≥ 9 → **NOT SUPPORTED**.

## 2. The sharpness result replicates — this is the first replicated finding in this line

Across **4 folds of unseen subjects × 3 seeds**:

| FM metric | mean | sd | range |
|---|---|---|---|
| **sharpness Spearman** | **+0.445** | 0.021 | [+0.395, +0.479] |
| CRPS | 29.630 | 0.539 | [28.827, 30.694] |
| coverage @ 50 % | 0.424 | 0.017 | [0.403, 0.459] |
| coverage @ 80 % | 0.688 | 0.019 | [0.666, 0.733] |
| median sample SD | 39.9 ms | 2.2 | [36.6, 44.3] |

N5 measured +0.475 on `an0`/`k2s` with one seed. N7 gets **+0.445 ± 0.021 on twelve runs, every
evaluation subject unseen by its own model.** Per-beat timing uncertainty being predictable from PPG is
now a replicated result, not a single observation — and it survives the move to the full candidate set,
which N5 and N6 did not evaluate.

`FM − FM-SHUFFLE` is **+5.130** CRPS on 12/12 runs: the sharpness is read from that candidate's own
features, not learned as a marginal.

## 3. Calibration fails the same way everywhere, and the NFE grid was exhausted

Coverage is under nominal in all 12 runs and by a consistent amount — **0.424 against 0.50** and
**0.688 against 0.80**. The spread is systematically ~0.11 too narrow at the 80 % level.

**The preregistered NFE rule hit its fallback in 12 of 12 runs.** The rule was "the smallest NFE whose dev
coverage@80 is within 0.10 of 0.80, else the largest in the grid". No NFE qualified anywhere, so all 12
runs used **NFE 32**, the grid maximum. The dev curve shows why (fold 0, seed 42):

| NFE | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| dev coverage@80 | 0.123 | 0.383 | 0.550 | 0.626 | 0.660 | **0.677** |

Coverage is still rising at 32 and has not reached 0.70, let alone 0.80. **The grid {1…32} was too small,
and the preregistered fallback made that visible rather than hiding it.** Whether coverage converges to
nominal at 64, 128 or never is not answerable from these runs and is not guessed at here.

## 4. The generative form is not better than the Gaussian head

`FM − HEAD` on CRPS: mean **−0.212**, with CI > 0 in only **2 of 12** runs. At the pinned NFE the sampler
is, on the proper score, indistinguishable from — or slightly behind — N5's heteroscedastic head.

**And the one preregistered justification for the generative machinery is absent again.** Fraction of
candidates whose 32 samples are significantly bimodal (Hartigan dip, Holm-corrected): **0.0000 in every
one of the 12 runs.** N6 found the same at NFE 1 and it was dismissed as an artefact of the 7.1 ms point
mass; at NFE 32 with a 39.9 ms spread that explanation is gone. The posterior is unimodal everywhere.

On this evidence **N7's recommendation is N5's Gaussian head**, not the flow-matching sampler: same score,
same sharpness, no sampling budget, and no multimodality to justify the extra machinery.

## 5. The marked point process: what handling the full candidate set showed

N5 and N6 evaluated only candidates that matched a GT beat. N7 emits `p_valid` for **every** candidate.

| | mean over 12 runs |
|---|---|
| candidate validity rate (precision at threshold 0.35) | **0.779** |
| GT recall within ±150 ms | **0.869** |
| validity-head Brier | **0.1252** |
| base-rate Brier (predicting the constant rate) | 0.1715 |
| improvement | **+0.0463** |

The validity head is informative — it beats the base rate by 0.046 Brier — so the 22 % of candidates that
correspond to nothing are, to a useful degree, identifiable from the same PPG features.

**Two numbers must not be confused.** The 0.869 recall here is at the **±150 ms** matching window the
preregistration fixed; R1's headline **F1@50 = 0.62** is at ±50 ms. N7 did not improve R1's detector and
does not claim to: 13 % of GT beats have no candidate at all within ±150 ms and are outside every number
in this report.

## 6. Which of N6's five requirements are now met

| N6 §6 requirement | status |
|---|---|
| 1. Pin the NFE before running | **met** — and the fallback fired 12/12, exposing that the grid was too small |
| 2. Calibration a primary endpoint | **met** — and it is what fails |
| 3. Handle unmatched detections and misses | **partly** — `p_valid` handles false positives (Brier +0.046 over base rate); **misses are untouched** |
| 4. Multiple seeds and fresh subjects | **met** — 4 folds × 3 seeds, every subject evaluated once unseen |
| 5. Multimodality at a spread-bearing NFE | **met** — and the answer is 0.0000 |

## 7. Where this leaves the method line

**Established, replicated:** per-beat timing uncertainty is predictable from PPG (+0.445 ± 0.021 across 12
unseen-subject runs), the signal is candidate-specific (shuffle costs 5.1 CRPS), and false candidates are
identifiable (Brier +0.046).

**Not established, and now measured rather than suspected:**

1. **Calibration.** Every configuration tested is ~0.11 too narrow at the 80 % level, and more sampling
   steps were still helping when the preregistered grid ran out. This is now the single blocking problem.
2. **The generative form earns nothing here.** Equal CRPS to a Gaussian head, no multimodality at any NFE,
   and a sampling cost. The honest current recommendation is the head.
3. **R1's misses.** 13 % of GT beats at ±150 ms — and far more at the ±50 ms tolerance the field uses —
   never get a posterior at all.

**A successor must therefore decide whether it is a method paper about calibration** — the failure is
consistent, large, and the obvious remedies (a wider NFE grid, a variance-correcting objective, a
post-hoc conformal recalibration) are each testable in under an hour — **or whether the finding is simply
that a small Gaussian head on a frozen detector's field predicts per-beat timing uncertainty well, which
N5 and N7 §2 already support and which does not need a generative model at all.**

The second reading is cheaper, replicated, and currently better supported by the evidence than the first.
It is also a much smaller paper. That choice is not made here.

## 8. Limits

- "Fresh" means unseen by that fold's model. `kjd` / `ssx` remain unloaded; there is still no
  corpus-level held-out test.
- WildPPG only. One detector, one threshold for the verdict (the 0.20 secondary was not run — declared
  scope, not a result).
- Everything is conditional on R1's candidate set (§5).
- A bootstrap defect was found and fixed before any number was produced: the sharpness CI drew
  **separate** resample indices for the spread and the residual, which would have paired unrelated beats.
  It failed loudly on a length mismatch; had the lengths matched it would have silently reported a
  meaningless correlation.

## 9. Provenance

Folds fixed by `numpy.random.default_rng(20260913).permutation` over the 12 non-`an0`/`k2s` WildPPG
subjects; per-fold train/dev/eval asserted disjoint. Seeds 42/43/44. R1 Global-TCN frozen, state sha256
`0986a7af…` asserted, `requires_grad=False` verified; no ECG array touched at inference. One shared
validity head per (fold, seed) used identically by every timing arm. All arms scored from K = 32 samples
by one ensemble CRPS estimator. Subject-clustered paired bootstrap, 2,000 replicates, seed 20260911;
sharpness CI 500 replicates. Hartigan dip via `diptest` 0.11.0, Holm-corrected.

Pins `6cd70cd` / `bf60cd7c` unchanged; A4 md5 `31c042d291052fbb6dc15263ad316be2` unchanged; C2 deferred.
