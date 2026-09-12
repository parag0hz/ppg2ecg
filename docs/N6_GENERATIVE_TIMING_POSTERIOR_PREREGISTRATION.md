# N6 — Does per-beat timing uncertainty survive inside a generative sampler?

**Preregistration. Frozen on commit and push; never edited afterwards.**

Start HEAD `999589a`. Pins `6cd70cd` / `bf60cd7c` unmodified.

---

## 1. Question — this is the method's first experiment, not another premise probe

N5 showed per-beat timing uncertainty **is** observable from PPG: a Gaussian head on R1's frozen
probability field predicts per-beat σ with sharpness Spearman **+0.475**, against the frozen waveform
generator's **+0.03** (N4). But N5's head is **discriminative** — it *outputs* a σ. A generative model
does not output its uncertainty; it produces **samples**, and the uncertainty has to be visible in their
spread. That is exactly what N4 measured the waveform sampler failing to do.

> **Q.** If the output object is the beat's **timing residual** rather than a waveform, does a conditional
> **flow-matching sampler** produce samples whose *empirical* per-beat spread is calibrated and sharp —
> or is the loss of per-beat information a property of generative sampling itself?

- **Survives** — the method exists: PPG → a calibrated, per-beat-sharp generative posterior over beat
  times, and N4's finding is localised to the *waveform* output object rather than to generative sampling.
- **Does not survive** — the honest method is the discriminative head of N5, the generative framing buys
  nothing on this task, and the paper says so.

**Why a generative form at all, if a Gaussian head already works?** One reason only, and it is tested here:
a Gaussian head cannot represent a **multimodal** timing posterior, and R1's field is frequently ambiguous
between two candidate beat positions. §5's multimodality reading is the only thing that would justify the
extra machinery; without it, N6's honest recommendation is the simpler model.

## 2. Arms

Every arm predicts the same scalar: the residual `y = GT_time − R1_detected_time` in ms, at every R1 event
that matches a GT beat within ±150 ms — N5's exact population, features and split.

| arm | form | role |
|---|---|---|
| **CONST** | one global `(μ, σ)` from train | the bar (N4's situation) |
| **HEAD** | N5's Gaussian heteroscedastic head | the discriminative reference (+0.475) |
| **FM** | conditional **MeanFlow** on the 1-D residual, sampled | **the method** |
| **FM-SHUFFLE** | FM fed a partner beat's features | control |

**All four are scored from samples by one estimator.** CONST and HEAD are *sampled* from their Gaussians
(K draws each) rather than scored in closed form, so no arm gets an estimator advantage. `K = 32`, matching
N4's source count.

## 3. Training

FM shares HEAD's conditioning features (±32-sample R1 field patch, ±96-sample PPG patch), the frozen R1
Global-TCN, the frozen R1 split, seed 42, AdamW 1e-3 / wd 0.01, batch 256 and **6,000 optimizer steps** —
N5's budget, reused unchanged. Only the objective and the output differ. Checkpoint selection is FM's own
deterministic internal-dev metric (fixed-noise sample CRPS), never an evaluation metric.

## 4. Metrics — one sample-based estimator for every arm

With `K = 32` samples `{x_k}` per beat and truth `y`:

| metric | definition |
|---|---|
| **CRPS** (primary) | `(1/K)Σ|x_k − y| − (1/2K²)ΣΣ|x_k − x_j|` — the standard proper ensemble estimator |
| **sharpness Spearman** (co-primary) | Spearman between the sample SD and `\|y − sample mean\|` |
| coverage @ 50 / 80 / 90 % | is `y` inside the central empirical interval of `{x_k}`? |
| PIT | rank of `y` among `{x_k}`; KS distance from uniform |
| median sample SD, median `\|y − mean\|` | sharpness and accuracy in ms |

Subject-macro aggregation; paired subject-clustered bootstrap, 2,000 replicates, seed 20260911.

## 5. Decision rule — fixed here

| verdict | rule |
|---|---|
| **GENERATIVE SAMPLER PRESERVES PER-BEAT UNCERTAINTY** | `CONST − FM` on CRPS CI entirely > 0, **and** FM sharpness Spearman ≥ **0.20** with CI > 0, **and** `FM − FM-SHUFFLE` on CRPS CI entirely > 0 |
| **PARTIAL** | CRPS beats CONST with CI > 0 but the sharpness or shuffle condition fails |
| **DOES NOT PRESERVE** | `CONST − FM` on CRPS includes 0 or is entirely < 0 |

`0.20` is N5's bar, reused unchanged so the generative and discriminative forms are held to one standard.

**Two secondary readings, fixed now, neither able to change the verdict:**

1. **Cost of the generative form.** `FM − HEAD` on CRPS. If FM is not worse (CI not entirely < 0), the
   generative framing is free. If it is clearly worse, that is the paper's honest headline for this
   component and the recommendation is the simpler head.
2. **Multimodality — the only justification for the extra machinery.** Fraction of beats whose 32 samples
   are significantly bimodal (Hartigan dip test, p < 0.05, Holm-corrected within the eval set), and
   whether HEAD's per-beat CRPS is worse on that subset than on the rest. A generative posterior that is
   never multimodal, or whose multimodal beats are not ones the Gaussian struggles with, has no reason to
   exist here and the report must say so.

## 6. What N6 cannot conclude

- Nothing about R1's ~31 % unmatched detections or its misses (N5 §4); the population is unchanged.
- Nothing about waveform reconstruction. N6 changes the output object *away* from the waveform; it does
  not revisit N2/N3.
- Nothing deployable: `an0` / `k2s` are development validation with four pre-viewed windows, one seed.
- A positive result is one component working on one dataset, not a method paper. The remaining pieces —
  false positives, misses, a real test set, seeds, an external baseline — are named in the report, not
  claimed.

## 7. Firewalls

- R1 Global-TCN **frozen**: `eval()`, `requires_grad=False`, state sha256 `0986a7af…` asserted.
- No ECG array is touched at inference; ground truth forms targets only.
- `kjd` / `ssx` never loaded; train / dev / eval subjects asserted disjoint.
- A4 md5 `31c042d2…` re-checked; C2 deferred; no checkpoint, prediction or raw data enters git.

## 8. Cost

A 1-D conditional sampler on ~74k training events. Minutes.

## 9. Reporting

`docs/N6_GENERATIVE_TIMING_POSTERIOR_REPORT.md`: the four-arm CRPS/sharpness/coverage/PIT table, the §5
verdict, both secondary readings with the multimodality analysis, and an explicit statement of whether the
generative form is justified over the discriminative head.
