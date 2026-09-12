# N4 — Is the generator's timing variability a defect, or a posterior? REPORT

Preregistration `docs/N4_TIMING_CALIBRATION_PREREGISTRATION.md` (`4d14865`, pushed before any N4 number
existed). **No training, no weight update, no optimizer.** 111 s.

---

## 1. Verdict: **TIMING POSTERIOR CALIBRATED** — with two qualifications of equal weight

By the frozen §5 rule, at both NFE 1 and NFE 4, all three coverage levels land inside the ±0.10 tolerance.

| | nominal | NFE 1 | Δ | NFE 4 | Δ |
|---|---|---|---|---|---|
| coverage @ 50 % | 0.50 | 0.587 | **+0.087** | 0.543 | +0.043 |
| coverage @ 80 % | 0.80 | 0.833 | +0.033 | 0.795 | **−0.005** |
| coverage @ 90 % | 0.90 | 0.902 | **+0.002** | 0.869 | −0.031 |
| | | **CALIBRATED** | | **CALIBRATED** | |

**The variability X4-0 scored as a defect is the right order of magnitude for the uncertainty.** Spread is
51.6 ms (NFE 1) and 48.1 ms (NFE 4) against a mean absolute per-beat error of 28.7 / 29.4 ms. This is not
noise of arbitrary size.

The two preregistered secondary readings both qualify it, and §5 required them to be reported in these
words.

### 1.1 The centre is wrong: a systematic ~15 ms early bias

The PIT is **strongly non-uniform** — KS 0.194 (NFE 1) and 0.189 (NFE 4), `p ≈ 1e−117` and `1e−110`.
The histogram is monotonically increasing:

| PIT decile | 0–.1 | .1–.2 | .2–.3 | .3–.4 | .4–.5 | .5–.6 | .6–.7 | .7–.8 | .8–.9 | .9–1 |
|---|---|---|---|---|---|---|---|---|---|---|
| NFE 1 | **0.010** | 0.043 | 0.064 | 0.094 | 0.098 | 0.153 | 0.138 | 0.136 | 0.139 | 0.123 |
| NFE 4 | **0.021** | 0.048 | 0.066 | 0.081 | 0.096 | 0.134 | 0.136 | 0.120 | 0.146 | **0.153** |

PIT is the rank of the true time among the 32 predicted times, so mass at the top means the predictions
sit **below** the truth. Median bias is **−15.6 ms** (NFE 1) and **−15.4 ms** (NFE 4): the model places
beats about 15 ms **early**, consistently.

Central coverage still lands in tolerance because a ~50 ms spread is wide enough to bracket zero despite a
15 ms shift. **The ensemble is the right width and the wrong centre.**

### 1.2 The spread carries no per-beat information

Spearman correlation between a beat's spread and its `|bias|`: **+0.016** (NFE 1), **+0.048** (NFE 4).
Essentially zero.

The model is **not more uncertain where it is more wrong.** Its spread is calibrated *marginally* — right
on average across beats — and uninformative *conditionally*. §5 fixed the wording for exactly this:
a spread that is the right size on average but carries no per-beat information is a materially weaker
result.

## 2. Censoring: this describes 70 % of beats

| | NFE 1 | NFE 4 |
|---|---|---|
| GT beats | 5,004 | 5,004 |
| eligible (≥ 16 / 32 sources detect) | 3,547 | 3,503 |
| **detection rate** | **0.709** | **0.700** |

**Three in ten GT beats are produced by fewer than half the sources and are excluded from every number
above.** Those are the beats the model is least able to find, so the calibration statement is about the
beats it does produce, never about the ones it misses. Any method built on this premise inherits that
30 % as an open problem, not a solved one.

## 3. What this changes, and what it does not

**Changed.** X4-0's `PERSISTENT SOURCE SENSITIVITY` stays a frozen verdict on its own terms, but the
reading it could not test is now measured: the source-induced timing variability is **not** arbitrary
noise — it is within ±0.09 of nominal coverage at three levels. The existing generators have been sampling
a timing distribution of roughly the right width, and no evaluation in this literature — including every
stage of this program before N4 — has ever scored them as doing so. Beat F1 at a 50 ms tolerance treats a
correctly-wide posterior as a failure.

**Not changed.** Calibration relates spread to error; it does not bound error. Median spread is ~50 ms and
median `|bias|` ~15 ms on a task where R1's detector achieves F1@50 = 0.62. Nothing here says the timing is
*accurate*, and nothing here is deployable: `an0` / `k2s` are development validation with four pre-viewed
windows.

## 4. Consequence: the timing-posterior direction is supported, and its method target is now specific

§1 of the preregistration named two futures. N4 selects the first — with the shape of the remaining work
now measured rather than guessed:

| deficiency | size | nature |
|---|---|---|
| systematic early bias | **−15.5 ms** | a constant offset; correctable outside the model |
| no per-beat sharpness | Spearman **+0.03** | **the research problem** |
| 30 % of beats undetected by half the sources | 0.70 detection | inherited, unsolved |

The method question is therefore not "can a model produce timing variability" — it already does, at the
right width — but **"can a model make that variability per-beat informative, and centre it correctly?"**
That is a sharper and more defensible target than anything the shape side offered, because N2 and N3
showed the shape side has no signal to work with at all, while N4 shows the timing side has signal that is
currently being spent uniformly.

It also implies the evaluation: a method of this kind must be scored with proper scoring rules and
calibration curves, not with an F1 at a fixed tolerance — which N1 independently showed is won by a method
incapable of morphology.

**Honest limits on the claim.** N4 establishes a premise, not a result. It does not show that a
per-beat-informative timing posterior is achievable, only that the quantity being modelled is real and the
current models are the right order of magnitude on it. The next step is the cheapest test of *achievability*,
not a method build.

## 5. Provenance

Frozen iMeanFlow generator `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt`, state sha256
`47d7ccb9…` asserted with the repository's canonical digest (a hand-rolled hash was written first and
rejected by that assertion — the check did its job). X4-0's frozen 512-window source subset asserted
element-wise; X4-0's 32 source seeds, `GT_ANCHOR_MS = 150`, and the `≥ 16/32` eligibility filter reused
unchanged. Per-source offsets recomputed raw rather than taken from `gt_anchored_presence`, which collapses
them to mean and SD. Subject-macro aggregation, paired subject-clustered bootstrap, 2,000 replicates,
seed 20260911.

`kjd` / `ssx` never loaded (`test_subjects_loaded: []`); pins `6cd70cd` / `bf60cd7c` unchanged; A4 md5
`31c042d291052fbb6dc15263ad316be2` unchanged; C2 deferred.
