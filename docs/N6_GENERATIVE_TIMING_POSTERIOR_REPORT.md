# N6 — Does per-beat timing uncertainty survive inside a generative sampler? REPORT

Preregistration `docs/N6_GENERATIVE_TIMING_POSTERIOR_PREREGISTRATION.md` (`cd30878`, pushed before any N6
number existed). 43.5 s + 33.0 s. R1's Global-TCN stayed frozen.

---

## 1. Frozen verdict: **DOES NOT PRESERVE**

| arm | CRPS ↓ | median sample SD | median \|residual\| | sharpness Spearman | cov 50 / 80 / 90 |
|---|---|---|---|---|---|
| CONST (global μ, σ) | 31.100 | 54.7 ms | 29.9 ms | −0.019 † | 0.559 / 0.779 / 0.848 |
| **HEAD** (N5, discriminative) | **28.704** | 42.6 ms | 29.5 ms | **+0.455** | 0.457 / 0.753 / 0.861 |
| **FM** (the method) | **35.501** | **7.1 ms** | **27.2 ms** | **+0.052** | **0.090 / 0.182 / 0.225** |
| FM-SHUFFLE | 39.061 | 7.1 ms | 30.8 ms | −0.026 | 0.080 / 0.158 / 0.201 |

† CONST's σ is constant; its Spearman is noise around zero by construction.

| §5 condition | required | observed | |
|---|---|---|---|
| `CONST − FM` on CRPS | CI > 0 | **−4.4012** [−4.6732, −4.1224] | **fails, on the wrong side** |
| FM sharpness Spearman | ≥ 0.20, CI > 0 | +0.052 [+0.032, +0.069] | fails |
| `FM − FM-SHUFFLE` on CRPS | CI > 0 | +3.5599 [+3.2437, +3.8817] | passes |

**The sampler collapsed.** Median sample SD is **7.1 ms** against a residual SD of 53.7 ms — coverage at
the 90 % level is **0.225**. FM learned an excellent *point* estimate (median |residual| **27.2 ms**, the
best of all four arms, better than HEAD's 29.5) and threw the distribution away.

`FM − HEAD` on CRPS is **−6.7972** [−7.0409, −6.5482]: against the discriminative reference the generative
form, as run, is far worse.

**Multimodality — the only preregistered justification for the generative machinery — is absent.** Zero of
14,998 beats have significantly bimodal samples (Hartigan dip, Holm-corrected at 0.05). With a 7.1 ms
spread the sample set is a point mass; there is nothing to be bimodal about.

## 2. This is the program's own collapse, reproduced at 1-D scale

A5/A6 showed OT-CFM at 1 NFE **collapses onto an MSE conditional-mean regressor** at window scale; X2
identified the endpoint barycenter degeneracy; X0 showed the resulting object is a temporally-correct,
amplitude-destroyed envelope. N2 reproduced it at beat scale (IMF −0.2306 behind plain regression).

N6 reproduces it again with the output object reduced to **a single scalar**. Whatever this is, it is not
a property of the waveform: one-step flow matching converges on the conditional mean and discards the
conditional spread, at 1024 samples, at 83 samples, and at 1.

## 3. Declared post-hoc: the collapse is a **one-step** phenomenon, not a generative one

**The preregistration did not pin the sampler's NFE.** The implementation used NFE 1 — the endpoint form,
consistent with this program's one-step subject — and that is what the §5 verdict was computed on. The
sweep below was written **after** seeing that result and **cannot change the frozen verdict**. It is
reported because it identifies the cause.

Same weights, same seed, same 32 samples per beat, only the sampling schedule varied:

| NFE | CRPS ↓ | median SD | \|residual\| | sharpness | cov 50 / 80 / 90 |
|---|---|---|---|---|---|
| **1** (the frozen verdict) | 36.165 | **5.1 ms** | 26.8 ms | **−0.067** | 0.041 / 0.109 / 0.170 |
| 2 | 30.638 | 17.5 ms | 28.0 ms | **+0.491** | 0.201 / 0.369 / 0.453 |
| 4 | 28.830 | 27.4 ms | 28.2 ms | +0.486 | 0.314 / 0.546 / 0.654 |
| 8 | 28.335 | 32.4 ms | 28.3 ms | +0.484 | 0.373 / 0.627 / 0.738 |
| **16** | **28.192** | 35.1 ms | 28.4 ms | +0.483 | 0.400 / 0.663 / 0.774 |

Three things follow, and the first two would have **flipped the verdict**:

1. **Sharpness recovers completely at NFE ≥ 2**: +0.49, *above* the discriminative head's +0.455 and 10×
   the 0.20 bar. The per-beat information is in the weights the whole time; NFE 1 does not express it.
2. **CRPS at NFE 16 (28.192) beats HEAD (28.704) and CONST (31.100).** The generative form is not
   inherently worse — it is better, given enough steps.
3. **Overconfidence persists at every NFE.** Coverage at 16 steps is 0.400 / 0.663 / 0.774 against nominal
   0.50 / 0.80 / 0.90, and the spread is still growing at the largest budget tested. Sharp but too narrow
   is *still miscalibrated*, and this is the remaining research problem.

**This is exactly why the NFE had to be preregistered and was not.** Had I pinned NFE 16, N6 would read
"GENERATIVE SAMPLER PRESERVES PER-BEAT UNCERTAINTY". Had I pinned NFE 1 — which I effectively did, in
code — it reads DOES NOT PRESERVE. The verdict stands as the frozen rule computed it; choosing the
schedule after seeing the sweep is the thing preregistration exists to stop, and it is not done here.

## 4. What this means for the method

The method is **not dead and not demonstrated.** What N6 establishes, across its frozen verdict and its
declared post-hoc sweep:

| | status |
|---|---|
| per-beat timing information exists in PPG | **established** (N5, +0.475) |
| it survives inside a generative sampler | **yes, at NFE ≥ 2** (+0.49) — post-hoc, not preregistered |
| a **one-step** generative posterior | **collapses** (frozen verdict; SD 7.1 ms, coverage 0.225) |
| the generative form beats the discriminative head | **at NFE 16** (CRPS 28.192 vs 28.704) — post-hoc |
| the posterior is **calibrated** | **no, at any NFE tested** — coverage 0.400 / 0.663 / 0.774 at best |
| multimodality justifies the generative form | **no evidence** — 0 of 14,998 beats bimodal at NFE 1 |

The honest headline is that the program's one-step framing — the thing named in `RESEARCH_QUESTION.md`
and pursued since A0 — is **the** obstacle to a calibrated posterior, and it is an obstacle the same
weights do not have at two steps.

## 5. Limits

1. **Population unchanged from N5**: only R1 events matching a GT beat within ±150 ms. About 31 % of R1's
   detections are unmatched and excluded; its misses (F1@50 = 0.62) are outside the analysis. A method
   built here inherits both.
2. Nothing about waveform reconstruction; N6 moved the output object away from it and does not revisit
   N2/N3.
3. `an0` / `k2s` are development validation with four pre-viewed windows. **One seed.** Not a test result.
4. The multimodality test was run only at the frozen NFE 1, where the samples are a point mass, so it is
   uninformative rather than negative. At NFE ≥ 2 it was not run and is not claimed either way.
5. The post-hoc sweep retrained FM under the identical recipe rather than reloading N6's checkpoint; its
   NFE-1 row (CRPS 36.165, SD 5.1 ms) differs slightly from the frozen run's (35.501, 7.1 ms) because
   checkpoint selection differed — the frozen run selected on internal-dev CRPS, the sweep used the final
   step. Both show the same collapse; the discrepancy is reported, not smoothed.

## 6. What a successor preregistration must fix

1. **Pin the NFE before running.** §3 shows the verdict is entirely determined by it.
2. **Make calibration a primary endpoint, not a diagnostic.** Every configuration tested is overconfident;
   sharpness without coverage is not a posterior.
3. Address the 31 % unmatched detections and R1's misses — a timing posterior over the wrong event set is
   not a timing posterior.
4. Multiple seeds and a real held-out test before any claim.
5. Re-run the multimodality test at an NFE where the samples have spread, since it is the only argument
   for preferring the generative form over N5's head.

## 7. Provenance

R1 Global-TCN frozen, state sha256 `0986a7af…` asserted, `requires_grad=False` verified. N5's population,
features, split, optimizer, batch, seed and 6,000-step budget reused unchanged; only the objective and
output differ. All four arms scored from **K = 32 samples by one ensemble CRPS estimator** — the Gaussian
arms sampled rather than scored in closed form, so no arm gets an estimator advantage. Subject-macro
aggregation; paired subject-clustered bootstrap, 2,000 replicates, seed 20260911; the sharpness CI is a
subject-clustered bootstrap of the Spearman. Hartigan dip via `diptest` 0.11.0, Holm-corrected.

`kjd` / `ssx` never loaded; no ECG array touched at inference; pins `6cd70cd` / `bf60cd7c` unchanged;
A4 md5 `31c042d291052fbb6dc15263ad316be2` unchanged; C2 deferred.
