# X3-G0 — Is Minibatch Coupling Strong Enough to Test?

> **Metric semantics note (added 2026-08-31).** In this document `morph` / `morphology_corr` is
> **matched-beat morphology correlation** and `rr_mae_ms` is **matched-consecutive-beat RR MAE**. Both are
> conditional on successful ≤50 ms one-to-one R-peak matching and are averaged only over matched beats.
> A missed beat therefore incurs no explicit penalty in either metric — it is excluded from the denominator
> rather than scored — so neither metric is monotonic in event coverage: both may rise or fall when the
> matched set changes. Values and specifications here are unchanged; only the labels and their scope are
> made explicit. See [METRIC_SEMANTICS.md](METRIC_SEMANTICS.md).


Pre-registration: `docs/X3_G0_COUPLING_GEOMETRY_PREREGISTRATION.md` (commit `a9707e7`, **pushed before any G0 metric was computed**).
Pre-preregistration disclosure: `docs/X3_G0_PREPREREG_DESIGN_AUDIT.md`. Artefacts: `artifacts/x3_g0_coupling_geometry/`;
cached residuals/masks in `outputs/x3_g0_coupling_geometry/` (559 MB, git-ignored).
**Zero deep training.** Frozen-checkpoint inference only (A6 / iMF-1 / OT-CFM-50). No checkpoint or historical artefact was written.

## Executive verdict

**INCONCLUSIVE** (frozen rule, §14 of the pre-registration).

The gate did not fail for lack of a manipulation. Standard raw-waveform minibatch OT **does** create real, permutation-null-corrected
source dependence in the QRS-band conditional residual — ΔR²_QRS reaches **0.062** at B = 512, above the 0.05 bar. What fails is the
**structural-translation leg**: the pre-registered rule requires ≥ 20 % recovery of the A6→iMF-1 gap in **morphology** *and* in one
other structural metric, and morphology recovery is **negative (−0.33)** for a measurement reason documented below. Because RAW is not
"weak", neither COST-GEOMETRY LIMITED nor WEAK FINITE-BATCH LEVER can fire either, so the frozen rule set lands on INCONCLUSIVE.

The scientifically informative result is the answer to the question the experiment was built for, and it is **neither** of the two
outcomes that were anticipated.

## The central comparison: raw-waveform OT vs spectrally reweighted costs

| Residual domain | RAW | WHITE (train-spectrum whitened) | HF (brick-wall > 15 Hz) |
|---|---:|---:|---:|
| ΔR²_FULL (max over B ≤ 512) | **0.339** | 0.010 | 0.001 |
| **ΔR²_QRS** | **0.062** | 0.006 | 0.0005 |
| **ΔR²_HF** | 0.010 | 0.010 | **0.050** |

Both anticipated outcomes are wrong. It is **not** "both weak" (RAW is strong on FULL/QRS), and it is **not** "only the reweighted
cost is strong" (the reweighted costs are *worse* for QRS by one to two orders of magnitude).

What is true is that **cost geometry demonstrably controls where the dependence goes** — the HF cost redirects 5× more dependence into
the HF band than RAW does — but **the QRS-relevant dependence is already maximised by the plain raw-waveform L2 cost**. Spectral
reweighting moves the permutation budget *away* from the modes that carry QRS energy, not toward them.

The mechanism is visible in the effective dimensions (pooled over folds):

| Representation | d_PR | d90 | d95 |
|---|---:|---:|---:|
| GT waveform | 3.86 | 231 | 300 |
| FULL residual `r = y − m_A6(c)` | **4.16** | 245 | 311 |
| QRS-masked residual | **143.9** | 311 | 361 |
| HF-projected residual | **176.9** | 165 | 194 |

The conditional residual is dominated by ~4 effective modes, and those dominant modes are where the QRS energy lives (the QRS complex
is the highest-variance feature of an ECG window). Raw L2 assignment spends its budget exactly there. Whitening and HF projection
deliberately down-weight those modes and spread the same ≤ 512-element permutation across a 144–177-dimensional subspace, where it
buys far less per direction. This is descriptive geometry, not a feasibility theorem; **no conclusion here follows from d_PR alone.**

## Manipulation check: the geometries genuinely differ (not near-tie churn)

Cross-objective regret, pooled over folds (regret ≈ 0 means the other cost's assignment is essentially optimal under this cost
despite differing indices; substantial positive regret means the geometry really differs):

| B | cost q ← assignment p | regret | index overlap |
|---:|---|---:|---:|
| 64 | RAW ← WHITE | 0.411 | 0.187 |
| 64 | RAW ← HF | 0.646 | 0.087 |
| 64 | **RAW ← RESID** | **0.050** | **0.535** |
| 64 | WHITE ← HF | 0.482 | 0.129 |
| 512 | RAW ← WHITE | 0.381 | 0.118 |
| 512 | RAW ← HF | 0.639 | 0.037 |

This vindicates the pre-preregistration design decision. The residualised cost differs from RAW on ~47 % of assignment **indices** yet
costs only 5 % of the random-assignment gap — i.e. **near-tie churn, not a different geometry**, exactly as the design audit predicted
from corr(y, r) ≈ 0.96. Its dependence numbers duly track RAW's (ΔR²_QRS 0.058 vs 0.051 at B = 64). WHITE and HF, by contrast, are
genuinely different geometries. Note also that the assignment barely moves the transport cost at all: the mean cost reduction versus a
random permutation is 4.0 % (RAW), 1.8 % (HF) and 0.1 % (WHITE) at B = 512 — consistent with distance concentration in 1024 dimensions.

**Null sanity check:** B = 1 (independent coupling, no assignment) gives ΔR²_FULL = −0.0004, ΔR²_QRS = −0.0001, ΔR²_HF = 0.0000 —
the permutation-null correction behaves exactly as it must.

## Dose response (primary: 12 train subjects, 4-fold subject-grouped cross-fitting)

| B | Cost | Cost reduction | ΔR² Full | ΔR² QRS | ΔR² HF | QRS-relative | HF-relative |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | independent | 0.0000 | −0.0004 | −0.0001 | 0.0000 | N/A | N/A |
| 8 | RAW | 0.0183 | 0.1522 | 0.0259 | 0.0012 | 0.170 | 0.008 |
| 8 | WHITE | 0.0005 | 0.0014 | 0.0008 | 0.0017 | N/A | N/A |
| 8 | HF | 0.0078 | −0.0007 | −0.0001 | 0.0087 | N/A | N/A |
| 64 | RAW | 0.0313 | 0.2898 | 0.0507 | 0.0052 | 0.175 | 0.018 |
| 64 | WHITE | 0.0008 | 0.0058 | 0.0032 | 0.0057 | 0.552 | 0.983 |
| 64 | HF | 0.0138 | 0.0013 | 0.0003 | 0.0281 | N/A | N/A |
| 64 | RESID *(secondary)* | 0.0345 | 0.3395 | 0.0578 | 0.0057 | 0.170 | 0.017 |
| 256 | RAW | 0.0377 | 0.3329 | **0.0608** | 0.0085 | 0.183 | 0.026 |
| 256 | RESID *(secondary)* | 0.0417 | 0.3907 | 0.0687 | 0.0092 | 0.176 | 0.024 |
| 512 | RAW | 0.0404 | 0.3388 | **0.0624** | 0.0101 | 0.184 | 0.030 |
| 512 | WHITE | 0.0011 | 0.0084 | 0.0055 | 0.0105 | N/A | N/A |
| 512 | HF | 0.0184 | 0.0004 | 0.0005 | **0.0497** | N/A | N/A |

QRS-/HF-relative are reported **N/A** wherever ΔR²_FULL is not clearly positive; the ratio is meaningless there (the shipped CSV
contains the raw quotient, including the degenerate values, and must be read with this caveat). RAW's dose response rises roughly
logarithmically in B and saturates by B ≈ 256. These are **source-to-residual linear predictive dependence** measurements, never
mutual information.

## Structural translation — where the gate actually fails

Cross-fitted linear endpoint proxy `F_proxy(x0, c) = m_A6(c) + A_FULL x0`, K = 32 sources on 64 held-out windows per fold, scored per
generated sample with the frozen X0 primitives. **This is not an upper bound and not a population flow optimum.**

| B | Cost | Morph | Amp | QRS energy | Slope | HF | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | independent | 0.4864 | 0.2928 | 0.0750 | 0.1202 | 0.0087 | 0.3872 |
| 64 | RAW | 0.4838 | 0.3135 | 0.1984 | 0.1240 | 0.0086 | 0.3778 |
| 512 | RAW | 0.4790 | 0.3291 | 0.2230 | 0.1266 | 0.0084 | 0.3689 |
| 512 | WHITE | 0.4841 | 0.3127 | 0.0798 | 0.1253 | 0.0089 | 0.3798 |
| 512 | HF | 0.4866 | 0.2942 | 0.0752 | 0.1209 | 0.0089 | 0.3854 |
| — | ref A6 | 0.5135 (n=46) | 0.2478 | 0.0597 | 0.1117 | 0.0074 | 0.4196 |
| — | ref iMF-1 | 0.6197 (n=56) | 0.9943 | 0.8215 | 1.2402 | 0.2145 | 0.3989 |
| — | ref OT-50 | 0.7739 (n=55) | 0.9259 | 0.5976 | 1.0607 | 0.2633 | 0.4515 |

Recovery of the A6→iMF-1 gap (denominators: morph +0.106, amp +0.747, QRS energy +0.762, slope +1.129, HF +0.207):

| B | Cost | Morph | Amp | **QRS energy** | **Slope** | **HF** |
|---:|---|---:|---:|---:|---:|---:|
| 64 | RAW | −0.280 | 0.088 | 0.182 | 0.011 | 0.006 |
| 256 | RAW | −0.305 | 0.102 | 0.207 | 0.013 | 0.005 |
| 512 | RAW | −0.325 | 0.109 | **0.214** | 0.013 | 0.005 |
| 256 | RESID | −0.294 | 0.104 | 0.234 | 0.013 | 0.005 |
| 512 | WHITE | −0.277 | 0.087 | 0.026 | 0.012 | 0.007 |
| 512 | HF | −0.253 | 0.062 | 0.020 | 0.008 | 0.007 |

Two things matter here.

**(a) The morphology leg is measurement-confounded, and it is what makes the verdict INCONCLUSIVE.** Morphology is computed on
*matched beats only*. The A6 reference is so flat that the frozen detector finds beats in just **46 of 256** proxy windows, and its
0.5135 is the average over that biased, easiest subset — compare X0's test-set A6 morphology of 0.316 over 3,907 windows. The proxies
produce detectable beats in ~1,430 of 8,192 sample-windows, a much larger and harder population, so their 0.479–0.490 is not
comparable with A6's 0.5135 and the recovery fraction comes out negative. This is precisely the "the denominator moves between arms"
risk; it was flagged before the run, and the pre-registered rule nonetheless made morphology a mandatory leg. **The rule is not being
changed after the fact**: the verdict stands as INCONCLUSIVE, and the confound is reported.

**(b) On the detector-independent metrics, which do not suffer that bias, the translation is real but small and lands in the wrong
place.** RAW at B = 512 recovers **21 % of the QRS-energy gap** and 11 % of amplitude — but only **1.3 % of the max-slope gap** and
**0.5 % of the HF-energy gap**. X0 identified sharpness and QRS/HF energy as the actual one-step failure; the coupling-induced
dependence buys back some QRS *energy* while leaving *sharpness* essentially untouched. The spectral arms translate worse still
(QRS energy 2–3 %), despite the HF arm's 5× higher HF dependence.

## Secondary design-informed validation (an0, k2s)

Run under the identical frozen protocol with the whitener fitted on the 12 training subjects only. **This is design-informed, not
independent confirmation**, because validation ground truth was used to select the WHITE and HF arms in the pre-preregistration audit.

| B | RAW ΔR²_FULL / QRS / HF | WHITE | HF |
|---:|---|---|---|
| 1 | 0.0019 / 0.0002 / 0.0002 | — | — |
| 64 | 0.0828 / 0.0225 / 0.0056 | 0.0108 / 0.0050 / 0.0057 | 0.0016 / 0.0005 / 0.0314 |
| 512 | 0.1094 / 0.0321 / 0.0103 | 0.0147 / 0.0090 / 0.0106 | −0.0029 / 0.0001 / 0.0543 |

Same ordering, roughly **half the magnitude** (RAW ΔR²_QRS 0.032 vs 0.062). Two subjects, one held-out set, so the number is a
direction check rather than an estimate; it does show the dependence shrinks appreciably when the diagnostic map has to transfer to
subjects outside the fitting pool.

## Data, folds, firewall

Frozen A4 split. Pool = deterministic stride subsample to ≤ 4,096 windows per subject: **53,358 windows, 516,470 GT beats** over the
12 train + 2 validation subjects. 4 folds × 3 held-out subjects by manifest order (fold 0 e61/fex/l38, 1 n31/ngh/p5d, 2 p9p/qm9/trh,
3 tz8/u7y/w4p); ~34.3 k fit / ~11.5 k held-out windows per fold. Whitener, residual PCA (95 %, all folds hit the 128 cap), ridge map
and R² baseline were fitted on fit subjects only. `m(c)` = frozen A6c (round 33); **A6 was itself trained on these subjects, so the
residuals are in-sample and are not an out-of-sample estimate of true conditional-mean error** — G0 is a coupling-geometry resource
gate, not a population uncertainty estimate.

**Test firewall held.** `provenance.json` records `subjects_loaded` = the 14 train+val subjects and `test_subjects_loaded` = `[]`;
no `kjd`/`ssx` file exists in the G0 cache and no G0 artefact mentions them. The pre-preregistration design audit **did** read
already-published frozen test arrays (fully disclosed in `docs/X3_G0_PREPREREG_DESIGN_AUDIT.md`); G0 itself did not.

## What the result supports

- Standard raw-waveform minibatch OT induces **genuine, null-corrected source-to-residual linear dependence** on WildPPG, rising
  logarithmically with B and saturating near B ≈ 256 (ΔR²_FULL 0.34, ΔR²_QRS 0.062 at B = 512).
- **The raw waveform cost and the spectrally preconditioned costs induce materially different finite-batch assignment geometries**
  (cross-objective regret 0.38–0.65, index overlap 0.04–0.19), and the difference is not near-tie churn.
- **Only the raw cost induces substantial dependence in QRS-band residual structure**; the HF cost redirects dependence into the HF
  band (5× RAW) but leaves QRS and FULL near null.
- The residualised cost is a near-duplicate of the raw cost (regret 0.05, overlap 0.53) — the pre-preregistration demotion was correct.
- The measured dependence translates into only ~21 % of the A6→iMF-1 QRS-energy gap and ~1 % of the max-slope gap in the cross-fitted
  linear endpoint proxy.

## What the result does NOT support

- It does **not** show that minibatch coupling is impossible, mathematically or otherwise. `log₂(B!)/B` (4.6 bits per pair at B = 64)
  and `d_PR` are descriptive context only and entered no decision rule.
- It does **not** show that a trained coupling-modified flow model would fail: the linear proxy is a first-order diagnostic, **not an
  upper bound**, and a nonlinear network could extract more.
- It does **not** establish that spectral cost geometry is useless — only that, for QRS-band residual structure on this dataset, it is
  worse than the plain cost. No claim is made that whitening is the right or wrong coupling for anything else.
- It says nothing about C²OT, which was not run, and nothing about a raw-PPG cosine condition cost.
- The morphology comparison is confounded by the matched-beat denominator and should not be read as evidence either way.
- Single A6 checkpoint, single protocol, two validation subjects, in-sample residuals: no training-seed or population uncertainty is
  estimated.

## Should any coupling training now be run?

**No.** The reasons are the measured ones, not a theorem:

1. The dependence exists but lands in the wrong place. X0 established that the one-step ECG failure is a **sharpness/QRS-energy
   attenuation**; the coupling buys ~21 % of the QRS-energy gap and ~1 % of the slope gap and ~0.5 % of the HF gap.
2. There is no better cost to train. The two spectrally reweighted arms — the ones added precisely to test the cost-misalignment
   hypothesis — are an order of magnitude *worse* for QRS structure than the plain cost already in use.
3. The assignment barely moves the transport cost (4.0 % at B = 512) and the dependence saturates by B ≈ 256, so a larger coupling
   pool is not a promising lever.
4. Held-out-subject transfer roughly halves the effect.

## Recommended next experiment — DO NOT EXECUTE

Nothing here indicts the *objective-level* lever, which this project has already shown works: iMF-1 reaches amplitude 0.99 and
QRS-energy 0.82 at one evaluation on these same windows, versus the proxy's 0.33 and 0.22. The natural pre-registered follow-up is
therefore **not** a coupling experiment but a question about what finite-interval supervision supplies that an instantaneous velocity
field at t = 0 does not — for which X2 already isolated a concrete, cheap handle: the endpoint carries essentially no explicit training
mass under uniform timestep sampling. A single pre-registered run with timestep mass placed at t ≈ 0 (or an explicit endpoint loss),
holding everything else fixed, would test whether the measured cancellation is the population identity or an artefact of an
unsupervised boundary. **Not started.**
