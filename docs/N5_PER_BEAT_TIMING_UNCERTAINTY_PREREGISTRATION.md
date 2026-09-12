# N5 — Can per-beat timing uncertainty be predicted from PPG at all?

**Preregistration. Frozen on commit and push; never edited afterwards.**

Start HEAD `09ad999`. Pins `6cd70cd` / `bf60cd7c` unmodified.

---

## 1. Question

N4 established the premise of a timing-posterior method and measured exactly what is missing from the
current models:

| N4 finding | value |
|---|---|
| ensemble spread is the right **size** | coverage 0.587 / 0.833 / 0.902 vs nominal 0.50 / 0.80 / 0.90 |
| ensemble centre is **wrong** | median bias **−15.5 ms** (beats placed early), PIT KS 0.194 |
| spread carries **no per-beat information** | Spearman(spread, \|bias\|) = **+0.016 / +0.048** |

The third line is the research problem. But it has two possible causes and N4 cannot separate them:

- **(a) the generator**: it never learns to modulate its uncertainty, though the information is in the PPG;
- **(b) the signal**: PPG says nothing about *which particular beats* are timed uncertainly, so no model
  could modulate.

> **Q.** Given PPG only, can a model predict **per-beat** timing uncertainty that tracks its own actual
> error — or is per-beat uncertainty simply not observable from PPG?

If **(b)**, the timing-posterior method direction closes exactly as the shape direction closed in N2/N3,
and this program's honest output is the measurement paper. If **(a)**, the method target is real and
achievable, and N5 is its first working component.

## 2. Design — the cheapest decisive test

The PPG-only timing extractor this program already has is R1's frozen **Global-TCN** (state sha256
`0986a7af…`, 328,897 parameters), which maps PPG to a per-sample R-probability field. N5 asks a
heteroscedastic head, reading **only PPG-derived quantities**, to predict the distribution of the residual
`GT_time − detected_time` at each event R1 detects.

That is precisely a per-beat timing posterior: a centre (correcting R1's bias) and a width (its per-beat
uncertainty).

| arm | predicts | role |
|---|---|---|
| **CONST** | one global `(μ, σ)` fitted on train only | **the bar** — N4's situation: right width, no per-beat information |
| **HEAD** | per-beat `(μ_i, σ_i)` from a local patch of the R1 field and the PPG | the test |
| **HEAD-SHUFFLE** | HEAD's weights fed a **partner beat's** features | control — a gain that survives shuffle is not feature-derived |

`CONST` is not a strawman: it is exactly what a correctly-sized, per-beat-uninformative posterior looks
like, which is what N4 measured the generator to be. Beating it is the whole question.

## 3. Data

- Frozen R1 split: train `fex l38 n31 ngh p5d p9p qm9 trh tz8 w4p`, internal dev `u7y e61`, evaluation
  `an0 k2s`. `kjd` / `ssx` never loaded.
- Windows: `select_subset(salt="n2-beat-v1", n_take=1024)` per subject (N2/N3's rule).
- Events: R1's frozen extractor — `extract_events(threshold 0.35, refractory 32)`, unchanged.
- A detected event enters the analysis only if a GT R peak lies within **±150 ms** (`GT_ANCHOR_MS`, N4's
  window). **Unmatched detections — R1's false positives — are excluded and their rate is reported.**
  N5 is a statement about the beats R1 finds, never about the ones it invents or misses.
- Features (PPG-derived only, no ECG at inference): a ±32-sample patch of the R1 probability field around
  the event (65 values) and the co-located ±96-sample PPG patch. Fixed here.

## 4. Training

HEAD is a small MLP trained with **Gaussian negative log-likelihood** on the train subjects; checkpoint
selection on internal-dev NLL; seed 42; AdamW 1e-3 / wd 0.01; batch 256; **6,000 steps** — the N2/N3
budget, reused. CONST's `(μ, σ)` are the train-set residual mean and SD, computed once, never fitted to
evaluation data.

## 5. Metrics

Per matched event, with residual `y_i = GT_i − detected_i` in ms:

| metric | |
|---|---|
| **NLL** (primary) | Gaussian negative log-likelihood of `y_i` under that arm's `(μ_i, σ_i)` |
| **CRPS** | closed-form Gaussian CRPS — a proper scoring rule that does not reward overconfident tails |
| **sharpness Spearman** (co-primary) | Spearman between `σ_i` and `\|y_i − μ_i\|` |
| coverage @ 50 / 80 / 90 % | as N4, against the predicted interval |
| median `\|μ_i − y_i\|` | does the head also fix N4's −15.5 ms bias? |

Subject-macro aggregation, paired subject-clustered bootstrap, 2,000 replicates, seed 20260911.

## 6. Decision rule — fixed here

| verdict | rule |
|---|---|
| **PER-BEAT TIMING UNCERTAINTY IS PREDICTABLE** | `CONST − HEAD` on NLL has paired CI entirely > 0, **and** HEAD's sharpness Spearman ≥ **0.20** with CI entirely > 0, **and** `HEAD − HEAD-SHUFFLE` on NLL has CI entirely > 0 |
| **MARGINAL** | NLL improves over CONST with CI > 0 but the sharpness or shuffle condition fails |
| **NOT PREDICTABLE** | `CONST − HEAD` on NLL has CI including 0 or entirely < 0 |

**0.20** is fixed now and justified now: N4 measured the frozen generator at **+0.03**. A head reaching
0.20 is a six-fold improvement and the smallest correlation at which per-beat widths would visibly
separate confident from unconfident beats. Below it, a method could claim "per-beat uncertainty" while
being, in practice, a global sigma.

## 7. What a positive result licenses

- That per-beat timing uncertainty **is observable from PPG**, which no measurement in this program or,
  as far as the N-series literature review found, in this literature has shown.
- It is the first component of a timing-posterior method, **not the method** — it operates on R1's detected
  events, inherits R1's 0.62 F1 and its false-positive and miss rates, and says nothing about the 30 % of
  beats N4 found the generator cannot reliably produce.
- No deployment claim: `an0` / `k2s` are development validation with four pre-viewed windows.

## 8. Firewalls

- `kjd` / `ssx` never loaded; train / dev / eval subjects asserted disjoint.
- The R1 Global-TCN is **frozen** — loaded in `eval()`, `requires_grad=False`, state sha256 asserted. The
  only trained object is the head.
- Ground truth is used to form the training target and never as an inference-time input; asserted by the
  feature construction, which touches no ECG array.
- A4 md5 `31c042d2…` re-checked; C2 deferred; nothing enters git but code, docs and small tables.

## 9. Cost

One small MLP on ~100k events. Minutes.

## 10. Reporting

`docs/N5_PER_BEAT_TIMING_UNCERTAINTY_REPORT.md`: NLL / CRPS against CONST with intervals, the sharpness
Spearman, the shuffle control, coverage, the residual bias before and after, the excluded-detection rate,
the §6 verdict, and an explicit statement of whether the timing-posterior direction is open or closed.
