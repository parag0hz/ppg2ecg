# N5 — Can per-beat timing uncertainty be predicted from PPG? REPORT

Preregistration `docs/N5_PER_BEAT_TIMING_UNCERTAINTY_PREREGISTRATION.md` (`caa3034`, pushed before any N5
number existed). 28.3 s. The R1 Global-TCN stayed frozen; the only trained object is the head.

---

## 1. Verdict: **PER-BEAT TIMING UNCERTAINTY IS PREDICTABLE**

All three preregistered conditions pass, two of them by a wide margin.

| condition | required | observed | |
|---|---|---|---|
| `CONST − HEAD` on NLL | CI entirely > 0 | **+0.1998** [+0.1916, +0.2082] | pass |
| sharpness Spearman(σ, \|residual\|) | ≥ **0.20**, CI > 0 | **+0.475** [+0.450, +0.501] | **2.4× the bar** |
| `HEAD − HEAD-SHUFFLE` on NLL | CI entirely > 0 | **+0.6236** [+0.5920, +0.6571] | pass |

| arm | NLL ↓ | CRPS ↓ | median σ | median \|residual\| | sharpness Spearman |
|---|---|---|---|---|---|
| **CONST** (one global μ, σ from train) | 5.4163 | 30.095 | 55.2 ms | 29.4 ms | undefined † |
| **HEAD** | **5.2164** | **27.881** | **42.7 ms** | 29.0 ms | **+0.475** |
| HEAD-SHUFFLE | 5.8400 | 31.911 | 42.7 ms | 30.4 ms | **+0.002** |

† Spearman is undefined for a constant σ — CONST has no per-beat width to correlate. That is exactly what
it is the bar for.

## 2. The shuffle control is the result

HEAD-SHUFFLE has the **same weights** and the **same marginal σ distribution** (median 42.7 ms, identical
to HEAD) — it is fed a partner beat's features. Its sharpness Spearman collapses from **+0.475 to +0.002**
and its NLL becomes *worse than CONST* (5.8400 vs 5.4163).

So the per-beat width is not a learned marginal, not a rate effect, and not an artefact of the σ range.
**It is read out of that beat's own PPG-derived features**, and scrambling which beat those features
belong to destroys it completely while leaving the width distribution untouched.

## 3. Against N4: 16× more per-beat information

| | sharpness Spearman |
|---|---|
| N4, frozen generator's source-ensemble spread | **+0.016 / +0.048** |
| N5, head on R1's field | **+0.475** |

N4 left two candidate explanations for the generator's flat uncertainty: **(a)** the model never learns to
modulate, or **(b)** PPG carries no per-beat uncertainty information. **N5 selects (a).** The information
is in the PPG; the generator simply does not use it.

That is the opposite shape of result from N2/N3, and the difference is the whole point:

| direction | question | answer |
|---|---|---|
| **shape** (N2, N3) | is beat morphology in the PPG? | **no** — +0.0039 single view, +0.0041 four views, against a template |
| **timing uncertainty** (N4, N5) | is per-beat timing uncertainty in the PPG? | **yes** — +0.475, and current models spend it uniformly |

## 4. Censoring: this describes the beats R1 finds

| | train | internal dev | eval |
|---|---|---|---|
| R1 detections | 112,560 | 22,490 | 21,791 |
| matched to a GT beat within ±150 ms | 73,842 | 14,999 | 14,998 |
| **unmatched (R1 false positives), excluded** | **0.344** | **0.333** | **0.312** |

**Roughly a third of R1's detections do not correspond to any GT beat and are dropped**, and R1's misses
(F1@50 = 0.62) are outside this analysis entirely. N5 is a statement about the timing of beats R1 finds.
A method built on it inherits the false-positive and miss problems untouched; they are not made easier by
anything measured here.

## 5. What the head does and does not fix

It **sharpens**: median σ falls from 55.2 to 42.7 ms and CRPS from 30.095 to 27.881, while the residual is
covered — the width is smaller *and* the score is better, which is the definition of a useful sharpening.

It **barely re-centres**: median |residual| moves only 29.4 → 29.0 ms. Almost all of the gain is in the
*width*, not the *location*. N4's −15.5 ms systematic bias is a property of the generator's sampler, not of
R1's detector, whose train-set residual mean is only −1.82 ms — so N5 did not have that bias to fix and
does not claim to have fixed it.

## 6. What this licenses — and what it does not

**Licensed:** per-beat timing uncertainty **is observable from PPG**. No measurement in this program, and
none found in the N-series literature review, had shown this. It is the first working component of a
timing-posterior method and it clears its bar by 2.4×.

**Not licensed:**
- That a *generator* can be made to do this. N5 puts a head on a frozen detector's field; nothing here
  shows the same information survives inside a waveform sampler.
- Any claim about R1's 31 % false positives or its misses (§4).
- Any deployment or accuracy claim. Median |residual| is 29 ms and median σ 43 ms; the timing is not
  accurate, it is *honestly described*.
- `an0` / `k2s` are development validation with four pre-viewed windows. One seed. Not a test-set result.

## 7. Where this leaves the method paper

The three probes have now decided the shape of the contribution, each by a preregistered rule:

- **N1** — the event metric this literature uses is won by a detector-plus-template that cannot express
  morphology at all, and most of a generator's score on it is beat *rate*.
- **N2 / N3** — beat morphology is not in the PPG, under one view or four, with or without oracle timing.
- **N4 / N5** — per-beat timing uncertainty *is* in the PPG (+0.475), current models spend it uniformly
  (+0.03), and their centre is 15.5 ms off.

The defensible method contribution is therefore **a PPG-conditioned model whose output is a calibrated,
per-beat-sharp distribution over beat times**, evaluated with proper scoring rules and calibration curves
rather than a fixed-tolerance F1 — with N1 supplying the reason the usual metric must be abandoned, N2/N3
supplying the reason the waveform must not be the output object, and N4/N5 supplying the evidence that the
remaining quantity is real, currently wasted, and learnable.

**The next question is the one N5 explicitly does not answer** (§6): whether that per-beat information
survives inside a generative sampler, or only in a discriminative head on a frozen field. That is the
first experiment of the method itself rather than of its premise, and it needs its own preregistration.

## 8. Provenance

R1 Global-TCN `outputs/r1_global_tcn_seed42/checkpoint_best.pt`, state sha256 `0986a7af…` asserted by
`RT.load_rhythm_tcn`, `eval()`, `requires_grad=False`, verified with an explicit assertion. Frozen R1
split; `select_subset(salt="n2-beat-v1", n_take=1024)`; R1's frozen `extract_events(0.35, 32)`;
`GT_ANCHOR_MS = 150` matching. Features are a ±32-sample patch of the probability field and the
co-located ±96-sample PPG — **no ECG array is touched at inference**; ground truth forms the target only.
Head: 2-layer MLP, Gaussian NLL, AdamW 1e-3 / wd 0.01, batch 256, 6,000 steps, seed 42, internal-dev
checkpoint selection. Subject-macro aggregation; paired subject-clustered bootstrap, 2,000 replicates,
seed 20260911; the sharpness CI is a subject-clustered bootstrap of the Spearman itself.

`kjd` / `ssx` never loaded; pins `6cd70cd` / `bf60cd7c` unchanged; A4 md5
`31c042d291052fbb6dc15263ad316be2` unchanged; C2 deferred.
