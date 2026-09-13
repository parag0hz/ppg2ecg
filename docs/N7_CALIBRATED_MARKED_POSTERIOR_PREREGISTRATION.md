# N7 — A calibrated timing posterior over *candidate* beats: fixing what N6 named

**Preregistration. Frozen on commit and push; never edited afterwards.**

Start HEAD `3cd024d`. Pins `6cd70cd` / `bf60cd7c` unmodified.

---

## 1. What N6 required, and how N7 fixes each

N6's frozen verdict was DOES NOT PRESERVE, and its declared post-hoc sweep showed the cause was the
**sampler's NFE**, not the generative form. §6 of that report named five defects. N7 fixes all five and
nothing else changes.

| N6 §6 requirement | N7 |
|---|---|
| 1. **Pin the NFE before running** | §4: chosen from a fixed grid by a fixed rule **on internal dev only**, before any evaluation number exists |
| 2. **Calibration a primary endpoint, not a diagnostic** | §6: coverage is a **gate**, and no arm passes on sharpness alone |
| 3. **Handle R1's 31 % unmatched detections and its misses** | §3: a **validity head** makes the output a marked point process — is this candidate real, and if so when — plus a declared lower-threshold arm that trades recall against precision |
| 4. **Multiple seeds and a real held-out test** | §2: **4 folds × 3 seeds**; every one of the 12 pool subjects is evaluated exactly once on a model that never saw it. `an0` / `k2s` are reported separately and labelled as already-seen |
| 5. **Re-run multimodality where samples have spread** | §5: run at the pinned NFE, not at NFE 1 where the samples were a point mass |

## 2. Subjects — fresh evaluation without touching the forbidden test

`kjd` / `ssx` remain **never loaded**. The fresh evaluation comes from a 4-fold partition of the 12
WildPPG subjects that are not `an0` / `k2s`, fixed here by
`numpy.random.default_rng(20260913).permutation`:

| fold | evaluation (fresh) | internal dev | train |
|---|---|---|---|
| 0 | `p9p trh u7y` | fold 1 | folds 2, 3 |
| 1 | `e61 p5d tz8` | fold 2 | folds 3, 0 |
| 2 | `l38 n31 w4p` | fold 3 | folds 0, 1 |
| 3 | `fex ngh qm9` | fold 0 | folds 1, 2 |

Six training subjects, three dev, three fresh evaluation per fold; **every subject is evaluated exactly
once by a model that never saw it in training or selection**. Seeds **42, 43, 44**. Twelve (fold, seed)
runs per arm.

**`an0` / `k2s` are the development set that designed N1–N6.** They are evaluated and reported, always
labelled *already-seen*, and **never** enter a verdict.

## 3. The output object: a marked point process over candidates

R1's detector at its frozen threshold produces candidates of which ~31 % match no GT beat, and misses
~38 % of GT beats (F1@50 = 0.62). A timing posterior defined only on matched candidates, as in N5 and N6,
is a posterior over the wrong event set. N7's model emits, for every candidate:

| output | trained with |
|---|---|
| `p_valid` — does this candidate correspond to a real beat (GT within ±150 ms)? | BCE on all candidates |
| the residual distribution, **conditional on validity** | on valid candidates only |

`p_valid` is produced by **one shared validity head per (fold, seed)**, used identically by every timing
arm, so the arms differ *only* in the timing posterior and the detection comparison is held fixed.

**Candidate thresholds.** Primary: R1's frozen **0.35**, so the population is comparable to N5/N6.
Declared secondary: **0.20**, which raises recall and hands more false positives to the validity head —
the mechanism by which a marked posterior can address misses. Both are reported; the primary verdict uses
0.35 only.

## 4. The NFE, pinned by a rule that never sees evaluation data

Grid **{1, 2, 4, 8, 16, 32}**. For each (fold, seed), compute FM's coverage at the 80 % level on that
fold's **internal dev** set and take:

> the **smallest** NFE whose dev coverage@80 lies within **0.10** of 0.80; if none qualifies, the
> **largest** NFE in the grid.

The chosen NFE is recorded per (fold, seed) **before** any evaluation-set number is computed, and the
distribution of chosen values is reported. No evaluation result may influence it.

## 5. Arms

| arm | timing posterior | role |
|---|---|---|
| **CONST** | one global `(μ, σ)` from that fold's train split | the bar |
| **HEAD** | Gaussian heteroscedastic head (N5) | discriminative reference |
| **FM** | conditional MeanFlow, sampled at the pinned NFE (§4) | the method |
| **FM-SHUFFLE** | FM on a partner candidate's features | control |

All arms scored from **K = 32 samples** by the one ensemble CRPS estimator; the Gaussian arms are sampled,
not scored in closed form. Everything else — features, optimizer, batch, 6,000 steps — is N5/N6's,
unchanged.

**Multimodality (N6 §6.5):** Hartigan dip, Holm-corrected at 0.05, run **at the pinned NFE**, with the
per-candidate CRPS of HEAD compared on bimodal versus unimodal candidates.

## 6. Decision rule — calibration is a gate, fixed here

Per fold and seed, on the **fresh** evaluation subjects at threshold 0.35:

| verdict | rule |
|---|---|
| **CALIBRATED PER-BEAT POSTERIOR** | (i) `|coverage@80 − 0.80| ≤ 0.10` **and** `|coverage@50 − 0.50| ≤ 0.10`; **and** (ii) `CONST − FM` on CRPS CI entirely > 0; **and** (iii) FM sharpness Spearman ≥ **0.20** with CI > 0; **and** (iv) `FM − FM-SHUFFLE` on CRPS CI entirely > 0 |
| **SHARP BUT OVERCONFIDENT** | (ii)–(iv) hold, (i) fails | 
| **NOT SUPPORTED** | (ii) fails |

**Stage verdict** over the 12 (fold, seed) runs: **SUPPORTED** if CALIBRATED in ≥ 9; **PARTIAL** if 5–8;
**NOT SUPPORTED** otherwise. Reported alongside the per-fold spread, which is the first seed-and-subject
variance estimate this method line has had.

Bars (ii)–(iv) are N5/N6's, reused unchanged. The 0.10 coverage tolerance is N4's, reused unchanged.

**Detection, reported and gated separately:** `p_valid` calibration (Brier score, reliability curve) and
the precision/recall of the candidate set at both thresholds. A timing posterior on a candidate set whose
validity is not calibrated is not a usable output, and the report must say so if that is the case.

## 7. What N7 still cannot conclude

- Nothing about `kjd` / `ssx`; they remain unloaded. "Fresh" here means subjects unseen *by that fold's
  model*, not a pristine corpus-level test set.
- Nothing about waveform reconstruction (N2/N3 closed that).
- Nothing about other datasets; WildPPG only.
- A positive result is a working component with subject-level and seed-level variance, on one corpus.
  It is not a paper.

## 8. Firewalls

- `kjd` / `ssx` never loaded; `assert_no_test_subjects` at entry; per-fold train / dev / eval asserted
  disjoint.
- R1 Global-TCN frozen, state sha256 `0986a7af…` asserted, `requires_grad=False` verified.
- No ECG array touched at inference; ground truth forms targets only.
- A4 md5 `31c042d2…` re-checked; C2 deferred; no checkpoint, prediction or raw data enters git.

## 9. Cost

Two small heads plus one validity head per (fold, seed); 12 runs. Estimated under 1 GPU-hour.

## 10. Reporting

`docs/N7_CALIBRATED_MARKED_POSTERIOR_REPORT.md`: the per-fold, per-seed verdict table with the stage
tally; coverage, CRPS, sharpness and the pinned NFE distribution; `p_valid` calibration and the
threshold-0.20 secondary; the multimodality result at the pinned NFE; the already-seen `an0` / `k2s`
numbers reported separately; and an explicit statement of which N6 §6 requirements are now met and which
remain.
