# B3-BOOT — The consensus gain tracks the non-redundant part of the functional errors, not waveform diversity (patient-clustered bootstrap)

Prereg `590d191` (`docs/B3_CORRELATION_BOOTSTRAP_PREREGISTRATION.md`), frozen and pushed before any number here was
computed. **No training, no new sampling**: every quantity is recomputed from arrays saved by EXP-B / DW1 / DW2
(`outputs/tt_expb_raw`, `outputs/dw1_raw`, `outputs/dw2_raw`, `outputs/rd1_detector/test_rpeaks.npz`,
`outputs/sr1_eval/arm_I_seed42.npz`). Before bootstrapping, the script reproduces every EXP-B point estimate
(ρ̄, G, C, I, MAD, SD, waveform RMS, feature diversity) — maximum absolute difference **9.4 × 10⁻⁷**.
Command: `.venv/bin/python scripts/b3_bootstrap.py` (runtime 10 s). Raw: `artifacts/b3_bootstrap/{bootstrap.json,
conditions.csv, k2_operators.csv, b3_bootstrap.png}`; draws `outputs/b3_bootstrap_raw/bootstrap_draws.npz`.

**Setting.** VitalDB V1 test, 1,156 patients, 12 conditions = {iMF, CD, PENGUIN (Euler)} × S ∈ {1, 2, 4, 8}, K = 16,
seed-42 checkpoints, T = HR. **Bootstrap:** patients with replacement, all windows of a drawn patient, **the same draw
for all 12 conditions**, every per-condition quantity recomputed inside the replicate; 5,000 replicates, seed 20260923.
The ordinary i.i.d. Spearman p-value reported in EXP-B (p = 3.9 × 10⁻⁷) is **not** used as evidence anywhere below.

---

## 1. Preregistered results

### C1 — primary: Spearman(ρ̄, G) across the 12 conditions — **HOLDS**
| association with the consensus gain G | point (full sample) | bootstrap median | 95 % percentile interval | replicates with the stated sign |
|---|---|---|---|---|
| **functional-error correlation ρ̄** (primary) | **−0.965** | −0.972 | **[−0.986, −0.930]** | **100.0 %** negative |
| ρ̄, Fisher-z average (supplementary) | −0.965 | −0.972 | [−0.986, −0.930] | 100.0 % negative |

Criterion: interval entirely below 0 **and** sign consistency ≥ 0.95 → both met.

### C2 — comparative against waveform diversity — **split: (i) fails, (ii) holds**
| association with G | point | bootstrap median | 95 % interval | replicates positive |
|---|---|---|---|---|
| waveform pairwise RMS | +0.867 | +0.839 | [+0.804, +0.867] | 100.0 % |
| feature-space diversity | +0.720 | +0.706 | [+0.671, +0.720] | 100.0 % |
| functional MAD (secondary) | +0.846 | +0.839 | [+0.769, +0.909] | 100.0 % |
| functional SD (secondary) | **+1.000** | +1.000 | **[+0.993, +1.000]** | 100.0 % |

- **(i) sign consistency of ρ̄ exceeds that of waveform RMS — FAILS.** Both are sign-consistent in 100 % of replicates;
  across the 12 conditions waveform diversity is *also* a stable predictor of the gain. Sign consistency cannot separate them.
- **(ii) `|r_err| − |r_wave|` excludes 0 — HOLDS:** +0.105, 95 % [+0.070, +0.175]. The functional-error association is
  reliably **stronger in magnitude**.
- The preregistration named the outcome "(i) holds, (ii) does not" but not this one. Read literally: across conditions,
  **waveform diversity is a real but weaker correlate of the gain; it is not refuted as a correlate.** What separates the two
  is C3.

### C3 — within-model ordering over the four tested depths — **negative in every replicate, for every model**
| model | ρ̄ vs G: point · replicates < 0 · replicates with perfect negative ordering | waveform RMS vs G: point · sign in replicates | feature diversity vs G | functional SD vs G |
|---|---|---|---|---|
| iMF | −1.00 · 100 % · 100 % | **−0.20 · negative in 100 %** | −0.40 · negative in 100 % | +1.00 · perfect positive in 100 % |
| CD | −1.00 · 100 % · 100 % | +0.40 · positive in 100 % | +0.20 · positive in 100 % | +1.00 · perfect positive in 100 % |
| PENGUIN (Euler) | −1.00 · 100 % · 100 % | +1.00 · positive in 100 % | +1.00 · positive in 100 % | +1.00 · perfect positive in 100 % |

Wording fixed in the preregistration: **perfect monotonic ordering over the four tested depths**, retained in every
bootstrap replicate for each model. With four points per model this is an ordering statement, not a significance test,
and not proof.
The discriminating observation: **the functional quantities (ρ̄, SD) keep the same sign in every model; waveform and
feature diversity change sign between models** — negative within iMF, positive within CD and PENGUIN. A variable whose
relation to the gain reverses depending on the generator cannot be the general explanation of the gain.

## 2. Post-hoc observations (labelled; none promoted to a preregistered result)

### 2.1 ρ̄ and functional SD measure the same thing from two sides
Across the 12 conditions Spearman(ρ̄, SD) = −0.965 [−0.972, −0.916]. Both are consequences of one decomposition of each
sample's error into a part **shared by all samples of the same window** and a **sample-specific** part (derivation in the
theory note, §7.2): ρ̄ is the shared share of the error variance, the within-window SD is the absolute size of the
sample-specific part. SD's near-perfect association with G (+1.000) is **partly mechanical** — the median gain
`G = mean_k |e_k| − |median_k e_k|` cannot be large unless the sample functionals spread — so SD is the less informative
of the two predictors even though it is numerically stronger. **The claim is therefore about the non-redundant
(sample-specific) component of the functional errors, of which low ρ̄ and high SD are two measurements; it is not a claim
that correlation specifically, rather than dispersion, is the mechanism.**

### 2.2 Effective sample size (mean-estimator intuition; explanatory only)
`K_eff = K / (1 + (K−1) ρ̄)`, K = 16. No condition was flagged (all ρ̄ > −1/15).

| model | S | ρ̄ [95 %] | K_eff | G [95 %] | median error C | mean individual error I | observed C / I | √(1 / K_eff) |
|---|---|---|---|---|---|---|---|---|
| iMF | 1 | 0.558 [0.531, 0.582] | 1.71 | 3.446 [3.355, 3.537] | 6.675 | 10.121 | 0.660 | 0.765 |
| iMF | 2 | 0.586 [0.558, 0.612] | 1.64 | 3.236 [3.147, 3.321] | 6.176 | 9.412 | 0.656 | 0.782 |
| iMF | 4 | 0.645 [0.614, 0.673] | 1.50 | 2.752 [2.675, 2.829] | 6.221 | 8.974 | 0.693 | 0.817 |
| iMF | 8 | 0.671 [0.639, 0.700] | 1.45 | 2.530 [2.455, 2.603] | 6.304 | 8.834 | 0.714 | 0.832 |
| CD | 1 | 0.728 [0.702, 0.752] | 1.34 | 2.020 [1.952, 2.089] | 6.229 | 8.249 | 0.755 | 0.863 |
| CD | 2 | 0.628 [0.605, 0.649] | 1.54 | 2.704 [2.625, 2.785] | 6.253 | 8.958 | 0.698 | 0.807 |
| CD | 4 | 0.667 [0.642, 0.690] | 1.45 | 2.436 [2.358, 2.511] | 6.203 | 8.639 | 0.718 | 0.829 |
| CD | 8 | 0.678 [0.654, 0.701] | 1.43 | 2.210 [2.135, 2.285] | 6.171 | 8.381 | 0.736 | 0.836 |
| PENGUIN | 1 | 0.858 [0.839, 0.874] | 1.15 | 1.238 [1.170, 1.308] | 7.438 | 8.675 | 0.857 | 0.931 |
| PENGUIN | 2 | 0.789 [0.765, 0.811] | 1.25 | 1.720 [1.661, 1.784] | 6.349 | 8.069 | 0.787 | 0.896 |
| PENGUIN | 4 | 0.717 [0.692, 0.740] | 1.36 | 2.132 [2.065, 2.198] | 6.128 | 8.260 | 0.742 | 0.857 |
| PENGUIN | 8 | 0.652 [0.629, 0.673] | 1.49 | 2.434 [2.363, 2.504] | 6.028 | 8.463 | 0.712 | 0.821 |

- `Spearman(K_eff, G)` = +0.965, 95 % [+0.930, +0.986]. **This is not independent evidence**: K_eff is a strictly
  decreasing function of ρ̄, so the rank correlation is the mirror of C1 by construction.
- **Sixteen samples are worth only 1.2–1.7 "independent" samples** under this intuition: most of the variance of the
  per-sample HR error is shared by all samples of a window (it is set by the PPG window, not by the noise draw). Pooling
  can remove only the sample-specific part; the shared part is the floor that width cannot lower. This is consistent with
  B2's fast saturation (most of the width gain is realised by K ≈ 8–16).
- In all 12 conditions the observed error ratio C / I is **below** √(1/K_eff) (e.g. 0.660 vs 0.765): the median removes
  more than the equal-correlation mean model predicts. The two ratios are not strictly comparable (MAE of a median vs SD
  of a mean) — this is a descriptive direction, consistent with AB1's finding that the median beats the mean on these
  heavy-tailed sample errors, and is not interpreted further.

### 2.3 The K = 2 anomaly is an aggregation-operator effect
EXP-B2's preregistered diminishing-return criterion **failed 9/9 and remains failed.** The post-hoc explanation — at
K = 2 the median of two values is their mean — was checked on the stored 32-draw matrices (3 models × 3 seeds,
partition average over disjoint noise-seed groups):

| check | result |
|---|---|
| median ≡ mean at K = 2 | exactly equal in 9/9 (by construction; patient-bootstrap interval [0, 0]) |
| median − mean at K = 3 | **−0.43 to −1.13 bpm in 9/9, every 95 % interval excludes 0** (e.g. iMF seed 42: −0.996 [−1.037, −0.957]) |
| median − mean at K = 4 / 8 | −0.44 to −1.15 / −0.50 to −1.50 bpm, 9/9 |
| **mean** operator: doubling gains non-increasing from K = 1 | **9/9** (e.g. iMF seed 42: 0.684 → 0.534 → 0.346 → 0.223 → 0.130) |
| median operator: doubling gains non-increasing from K = 2 | 9/9 |
| where B2's large "2 → 4" step sits once K = 3 is inserted | almost entirely in **2 → 3**, the first K at which the median is a true middle value (iMF seed 42: 8.738 → **7.404** → 7.185) |
| 20 % trimmed mean | identical to the mean for K ≤ 4 (nothing trimmed), approaches the median from K = 8 |

So the anomaly is a finite-K property of the median operator, not of the generators: the operator that is well-behaved
from K = 1 (the mean) is uniformly worse (by 0.4–1.5 bpm at every K ≥ 3), and the operator that is better (the median)
only becomes robust from K = 3. **Practical consequence (post-hoc):** K = 2 is a wasted configuration for median pooling
— it costs two samples and buys only mean pooling. This does not rescue B2-1; it explains it.

## 3. What was wrong before, stated explicitly
1. **DW2-B (post-hoc) read waveform diversity as the mechanism of the consensus gain.** Waveform diversity is a stable
   correlate of the gain *across* models, but its relation to the gain reverses sign between models (negative within iMF,
   positive within CD and PENGUIN), so it is not the explanation. The DW2-B report keeps its numbers and receives an
   appended correction note.
2. **The EXP-B report stated that ρ̄ was "the" mechanism variable and quoted an i.i.d. Spearman p-value.** The p-value is
   withdrawn as evidence (it ignores patient clustering and treats 12 conditions from the same windows as independent), and
   the statement is narrowed: ρ̄ and functional SD are two measurements of one quantity — the sample-specific part of the
   functional errors — and SD predicts the gain even more closely, partly for mechanical reasons.
3. **"Spearman = −1.0 within each model"** is reworded as *perfect monotonic ordering over the four tested depths*, retained
   in every bootstrap replicate.

## 4. Revised mechanism statement
> At a fixed inference budget, width reduces the error of a functional estimate only through the **sample-specific,
> non-redundant part of the samples' functional errors**; the part of the error shared by all samples of the same
> condition is untouched by pooling. The amount of non-redundant functional error — low cross-sample error correlation,
> equivalently wide within-condition functional dispersion — orders the consensus gain across all 12 tested
> model-depth conditions and, within each model, over all four tested depths, in every patient-bootstrap replicate.
> Diversity of the generated waveforms is a weaker correlate whose direction depends on the generator.

**Rating: MODERATE.**
- For STRONG it holds: preregistered primary criterion met with the whole bootstrap interval far from 0; sign-stable within
  every model; more strongly associated than waveform diversity (interval of the difference excludes 0); the one place
  where the two explanations disagree (within-model direction) goes the functional way in 100 % of replicates.
- Against STRONG: (a) it is an association over 12 conditions varied only through model and depth — no intervention changes
  the error dependence while holding everything else fixed; (b) single training seed, one dataset (VitalDB, regular
  rhythm), one functional (HR), one aggregator (median); (c) the bootstrap resamples patients, not training seeds or
  noise draws; (d) the within-model evidence rests on four depths per model; (e) SD's even tighter association is partly
  mechanical, so "correlation" is not singled out over "dispersion".
- The **negative** half — waveform diversity is not the mechanism — is the best-supported part.

## 5. Caveats that stay attached to every use of this result
- VitalDB only, seed 42 only for the 12 conditions (B2 has three seeds, but only at one S per model), HR only.
- Associational; the "mechanism" is an accounting identity plus an ordering, not a causal experiment.
- ρ̄ is computed over the windows where all 16 samples have a defined HR (EXP-B3's definition): 16,639–19,116 of 19,543
  windows depending on the condition (smallest: iMF S = 8, 16,639; largest: PENGUIN S = 8, 19,116).
- K_eff and the C/I comparison are mean-estimator intuition for a median estimator.
- No preregistered verdict was changed: DW2-B's HR-SD criterion (failed), EXP-B2's diminishing-return criterion (failed),
  EXP-A's A-H1 / A-H3 (rejected) all stand.
