# S1.2–S1.6 — Metric-Validity Audit — REPORT

**G1 is reported separately and is not restated or reinterpreted here:** `docs/S1_G1_METRIC_VALIDITY_REPORT.md`
(commit `f27234f`, PASS, T-B macro F1 0.9993 at 50 ms). That file is untouched by this work.

Protocol: `docs/S1_METRIC_VALIDITY_PREREGISTRATION.md` (`b749339`), Amendment 1 (`dc75079`).
Implementation-only commit, pushed before any number below existed: **`374e104`**.

**Population for everything below:** the frozen X4-0 stage-B subset — 2,048 development windows
(`an0` 1,024 + `k2s` 1,024), **19,834 GT beats**, asserted element-for-element against the committed
`nfe_subset.json`. Gaussian source **seed 0**. **These are seed-0 values and are NOT comparable to the
recorded 4-seed-pooled X4-0 table.** Every arm's anchor is recomputed here.

**NO TRAINING.** Frozen-checkpoint forward inference only; no checkpoint written. `kjd`/`ssx` never loaded.

---

## S1.2 — Zero-parameter DSP floor

`nk.ppg_findpeaks(nk.ppg_clean(ppg))` at library defaults, no tuning, delayed by a constant and
**template-stamped with the frozen Template A** so it is scored through the identical ECG detector path.
Train-only delay: median **296.9 ms** over 17,210 GT-R→next-PPG-peak pairs from the 10 template training
subjects. Per-val-subject medians `an0` 304.7 / `k2s` 289.1 ms (used only for the LOSO variant).

| regime | tol | F1 macro | 95 % CI | chance floor | excess | precision | recall | beats ratio | an0 | k2s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| δ = 0 | 50 | 0.0879 | [0.0838, 0.0923] | 0.1330 | **−0.0451** | 0.0834 | 0.0957 | 1.2165 | 0.0801 | 0.0958 |
| δ = 0 | 75 | 0.1256 | [0.1207, 0.1306] | — | — | 0.1189 | 0.1370 | 1.2165 | 0.1160 | 0.1352 |
| δ = 0 | 150 | 0.2639 | [0.2572, 0.2705] | — | — | 0.2493 | 0.2883 | 1.2165 | 0.2595 | 0.2684 |
| δ = train-global | 50 | **0.2066** | [0.1990, 0.2149] | 0.1325 | **+0.0741** | 0.1890 | 0.2328 | 1.2254 | 0.2482 | 0.1650 |
| δ = train-global | 75 | 0.3057 | [0.2970, 0.3145] | — | — | 0.2798 | 0.3441 | 1.2254 | 0.3547 | 0.2566 |
| δ = train-global | 150 | 0.5447 | [0.5366, 0.5531] | — | — | 0.5017 | 0.6095 | 1.2254 | 0.5820 | 0.5074 |
| δ = LOSO | 50 | 0.1991 | [0.1918, 0.2071] | 0.1324 | +0.0668 | 0.1827 | 0.2237 | 1.2247 | 0.2480 | 0.1503 |
| δ = LOSO | 75 | 0.2982 | [0.2895, 0.3070] | — | — | 0.2737 | 0.3347 | 1.2247 | 0.3545 | 0.2418 |
| δ = LOSO | 150 | 0.5472 | [0.5391, 0.5555] | — | — | 0.5044 | 0.6119 | 1.2247 | 0.5773 | 0.5171 |

All nine cells reported; none selected. **This is a zero-parameter DSP floor and a lower bound on simple
PPG-only methods** — one untuned detector plus a constant delay. It is not an upper bound on PPG
information, not a competitor, and not a learned PPG event predictor.

At 50 ms the floor's excess over chance is **+0.0741** (train-global), against **+0.3201** for iMF-8 and
**+0.3613** for OT-CFM-50 on the same windows. With no delay correction the arm scores *below* its own
chance floor (−0.0451): a systematically wrong constant is worse than random phase.

---

## S1.3 — Joint-fidelity census

Same 2,048 windows, seed 0. Matched-beat coverage = matched GT beats / valid GT beats.

| arm | F1 | 95 % CI | chance | **F1 excess** | coverage | matched morph | same_coord | oracle_corr | oracle QRS-E | QRS width err | beats ratio | HF |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iMF-1 | 0.4144 | [0.4016, 0.4268] | 0.1201 | +0.2943 | 0.4362 | 0.6679 | 0.0959 | 0.5922 | 0.4368 | 37.68 | 0.9810 | 0.2230 |
| iMF-4 | 0.4367 | [0.4229, 0.4504] | 0.1163 | +0.3204 | 0.4568 | 0.7743 | 0.1040 | 0.6677 | 0.5331 | 28.29 | 0.9492 | 0.2305 |
| iMF-8 | 0.4341 | [0.4202, 0.4480] | 0.1141 | +0.3201 | 0.4502 | 0.7795 | 0.1021 | 0.6631 | 0.5580 | 29.70 | 0.9161 | 0.2384 |
| iMF-50 | 0.4261 | [0.4120, 0.4400] | 0.1113 | +0.3148 | 0.4392 | 0.7775 | 0.1004 | 0.6475 | 0.5712 | 31.95 | 0.8863 | 0.2480 |
| OT-CFM-50 | 0.4828 | [0.4687, 0.4973] | 0.1215 | +0.3613 | 0.5129 | 0.8168 | 0.1239 | 0.7160 | 0.7012 | 23.30 | 1.0038 | 0.2687 |
| **MSE** | **0.4871** | [0.4723, 0.5017] | 0.1013 | **+0.3858** | 0.4899 | 0.5788 | **0.2425** | 0.5193 | **0.0105** | **98.59** | **0.7720** | **0.0073** |

Secondary columns:

| arm | oracle_absent | matched RR MAE | zero-contrib windows | precision | recall | missing | spurious |
|---|---:|---:|---:|---:|---:|---:|---:|
| iMF-1 | 0.3785 | 25.37 | 0.0908 | 0.4188 | 0.4137 | 0.5863 | 0.5673 |
| iMF-4 | 0.3298 | 23.08 | 0.1118 | 0.4435 | 0.4338 | 0.5662 | 0.5154 |
| iMF-8 | 0.3366 | 22.65 | 0.1240 | 0.4459 | 0.4278 | 0.5722 | 0.4882 |
| iMF-50 | 0.3537 | 22.04 | 0.1475 | 0.4420 | 0.4174 | 0.5826 | 0.4689 |
| OT-CFM-50 | 0.2505 | 21.28 | 0.0903 | 0.4829 | 0.4846 | 0.5154 | 0.5192 |
| MSE | **0.7465** | 17.22 | 0.1636 | 0.5351 | 0.4632 | 0.5368 | 0.3088 |

### The preregistered joint-fidelity gate rule, applied exactly as frozen

Multipliers were committed in the preregistration before any of these numbers existed, relative to the
OT-CFM Heun-25 reference `R`. **Reported only because the rule was pre-committed. S1 does not use it to
select a method, and no multiplier was changed.**

| arm | morph ≥ 0.80·R | coverage ≥ 0.80·R | QRS-E ≥ 0.60·R | width ≤ 1.50·R | beats ∈ [0.90,1.10] | ALL |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| iMF-1 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| iMF-4 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| iMF-8 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| iMF-50 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| OT-CFM-50 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| MSE | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |

**Interpretation rules held to:** no single column is called overall quality; high matched morph alone is
not high reconstruction fidelity; low F1 alone is not bad placement; MSE's high F1 alone is not a good ECG.

---

## S1.4 — Failure decomposition, oracle null, chance floor

### S1.4a — DISPLACED / WEAK / ABSENT (±150 ms search, preregistered threshold)

Threshold = 5th percentile of GT matched-beat `amp_rel`, computed per arm on this population.

| arm | threshold | unmatched | DISPLACED | WEAK | ABSENT | contested | an0 / k2s DISPLACED |
|---|---:|---:|---:|---:|---:|---:|---|
| iMF-1 | 7.261 | 11,863 | 0.4799 | 0.0335 | 0.4866 | 0 | 0.4847 / 0.4747 |
| iMF-4 | 7.236 | 11,486 | 0.4758 | 0.0182 | 0.5060 | 0 | 0.4814 / 0.4698 |
| iMF-8 | 7.242 | 11,607 | 0.4502 | 0.0143 | 0.5355 | 0 | 0.4540 / 0.4463 |
| iMF-50 | 7.317 | 11,808 | 0.4267 | 0.0107 | 0.5626 | 0 | 0.4342 / 0.4188 |
| OT-CFM-50 | 6.843 | 10,461 | 0.5198 | 0.0153 | 0.4649 | 0 | 0.5236 / 0.5159 |
| MSE | 7.285 | 10,881 | 0.2985 | 0.0014 | 0.7001 | 0 | 0.3559 / 0.2327 |

`contested` = 0 for every arm: the one-to-one matcher never denies an unmatched GT beat a peak that was
inside tolerance. WEAK is ≤ 3.4 % everywhere, so the split is essentially two-way.

**Estimator caveat, disclosed rather than smoothed:** the fractions above are **beat-pooled**
(total displaced / total unmatched) while the bootstrap intervals in
`s1_4_event_failure_classes.csv` are on the **window-macro** estimator (equal weight per window, then per
subject). They are different quantities and the CI therefore does not bracket the pooled point estimate
(e.g. iMF-8 pooled 0.4502 vs window-macro CI [0.5385, 0.5687]). Both are reported; neither is corrected
into the other.

**These numbers must not be compared to the exploratory 65.9 / 24.0 / 10.1 split** disclosed in
preregistration §3. That split searched ±100 ms and therefore counted beats displaced further as absent;
it remains labelled confounded and exploratory.

### S1.4b — Oracle-shift null calibration

| arm | same_coord | oracle_corr | true gain | null gain | **excess over null** |
|---|---:|---:|---:|---:|---:|
| iMF-1 | 0.0959 | 0.5922 | +0.4963 | +0.4962 | **+0.0001** |
| iMF-4 | 0.1040 | 0.6677 | +0.5637 | +0.5633 | +0.0004 |
| iMF-8 | 0.1021 | 0.6631 | +0.5610 | +0.5606 | +0.0004 |
| iMF-50 | 0.1004 | 0.6475 | +0.5471 | +0.5468 | +0.0003 |
| OT-CFM-50 | 0.1239 | 0.7160 | +0.5921 | +0.5919 | +0.0002 |
| MSE | 0.2425 | 0.5193 | +0.2768 | +0.2775 | **−0.0007** |

20 draws, `default_rng(20260901)`, partner drawn uniformly from the other GT beats of the same window.

**The oracle's entire gain is reproduced by pairing a beat with an unrelated beat.** Excess over the null
is +0.0001 to +0.0004 (MSE −0.0007) against gains of +0.28 to +0.59 — three orders of magnitude smaller.
The preregistration's CASE D condition is met: **displacement claims resting on `oracle_corr` are
withdrawn.** `oracle_corr` measures how well *any* beat-shaped segment of a prediction can be shift-fitted
to *any* beat-shaped reference segment, not correspondence with the specific GT beat.

Note that the null gain is itself arm-dependent (0.2775 for MSE, 0.5919 for OT-CFM-50), so `oracle_corr`
differences across arms are partly differences in each prediction's own local structure.

### S1.4c — Count-matched event chance floor

| arm | observed F1 | random-phase | circular-shift | excess (RP) | excess (CS) |
|---|---:|---:|---:|---:|---:|
| iMF-1 | 0.4144 | 0.1201 | 0.1202 | +0.2943 | +0.2941 |
| iMF-4 | 0.4367 | 0.1163 | 0.1182 | +0.3204 | +0.3185 |
| iMF-8 | 0.4341 | 0.1141 | 0.1159 | +0.3201 | +0.3182 |
| iMF-50 | 0.4261 | 0.1113 | 0.1133 | +0.3148 | +0.3129 |
| OT-CFM-50 | 0.4828 | 0.1215 | 0.1229 | +0.3613 | +0.3599 |
| MSE | 0.4871 | 0.1013 | 0.1033 | +0.3858 | +0.3838 |

Both constructions agree to ≤ 0.002. Roughly a quarter of every arm's raw F1 at 50 ms is chance.

---

## S1.5 — C2 perturbation re-analysis

`source` = PPG fixed, Gaussian source re-drawn. `ppg_shuffle` = Gaussian source **bit-identical**, PPG
deranged. Permutation floor: 20 draws, `default_rng(20260901)`.

| NFE | pair | arm | event F1 | perm floor | excess | wave PCC | timing all | timing excl-0 | timing uncensored | matched pairs | exact zeros | after excl. |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0→1 | source | 0.3664 | 0.1230 | +0.2434 | 0.0783 | 23.69 | 25.85 | 173.1 | 1750 | 164 (9.4 %) | 1586 |
| 1 | 0→1 | ppg_shuffle | 0.3042 | 0.1262 | +0.1780 | 0.5622 | **6.65** | **17.57** | 192.5 | 1487 | **995 (66.9 %)** | 492 |
| 1 | 2→3 | source | 0.3678 | 0.1236 | +0.2442 | 0.0827 | 23.38 | 25.65 | 160.8 | 1745 | 180 (10.3 %) | 1565 |
| 1 | 2→3 | ppg_shuffle | 0.3097 | 0.1233 | +0.1864 | 0.5757 | 6.82 | 16.03 | 195.7 | 1483 | 994 (67.0 %) | 489 |
| 8 | 0→1 | source | 0.3863 | 0.1131 | +0.2732 | 0.0925 | 22.46 | 24.65 | 205.6 | 1784 | 186 (10.4 %) | 1598 |
| 8 | 0→1 | ppg_shuffle | 0.2665 | 0.1185 | +0.1480 | 0.4371 | 6.73 | 18.78 | 216.5 | 1182 | **857 (72.5 %)** | 325 |
| 8 | 2→3 | source | 0.3831 | 0.1142 | +0.2689 | 0.0933 | 22.53 | 24.43 | 184.8 | 1772 | 178 (10.0 %) | 1594 |
| 8 | 2→3 | ppg_shuffle | 0.2814 | 0.1127 | +0.1687 | 0.4529 | 6.74 | 19.13 | 226.2 | 1202 | 872 (72.5 %) | 330 |

**The comparability defect is confirmed and quantified.** The `ppg_shuffle` arm's headline timing figure of
~6.7 ms rests on **66.9–72.5 % exact-zero pairs**, against 9.4–10.4 % in the `source` arm — the structural
consequence of holding the Gaussian source bit-identical. Excluding exact zeros moves `ppg_shuffle` from
6.65 → 17.57 ms and `source` from 23.69 → 25.85 ms, narrowing the ratio from ~3.6× to ~1.5×. Under an
uncensored matcher the ordering **reverses**: `ppg_shuffle` 192.5 ms vs `source` 173.1 ms at NFE 1.

The matched denominators also differ substantially (1,182–1,784 pairs), so even the all-matched statistics
are not denominator-comparable between arms.

**The X4-0 §10 prohibition remains in force.** These two perturbations are not on a common scale and their
magnitudes are **not** calibrated causal effects. No statement of the form "the source matters more than
the PPG" or its converse is made or supported here. The band-limited and QRS-masked pred-vs-pred
correlations that would separate the two mechanisms were **not** computed; they are new metrics requiring
their own preregistration.

---

## S1.6 — Development-side GT reliability

Two preregistered detectors on the **reference** ECG. Both reported; neither selected.
Detector A = `neurokit` (the project's frozen detector), B = `pantompkins1985`.

| subject | A/B F1 | 95 % CI | precision B vs A | recall B vs A | beats A | beats B | mean count diff | RR out A | RR out B | count implausible A / B |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| an0 | 0.1962 | — | 0.1913 | 0.2024 | 10,036 | 10,539 | 0.587 | 0.0008 | 0.0178 | 0.0000 / 0.0000 |
| k2s | 0.0930 | — | 0.0916 | 0.0945 | 9,798 | 10,078 | 0.357 | 0.0001 | 0.0073 | 0.0000 / 0.0000 |
| **MACRO** | **0.1446** | [0.1394, 0.1500] | 0.1415 | 0.1485 | 19,834 | 20,617 | 0.472 | 0.0005 | 0.0126 | 0.0000 / 0.0000 |

Every window scores below F1 0.9. Both detectors find **almost the same number of beats** (19,834 vs
20,617, mean per-window difference 0.47) and both produce physiologically plausible trains: 0 % of windows
have an implausible beat count under either detector, and RR intervals outside [333, 1500] ms are 0.05 %
(A) and 1.26 % (B).

### POST-HOC diagnostic — added after seeing the above, labelled and additive

Near-equal beat counts with near-zero positional agreement is the signature of a systematic offset, not of
ambiguity about where the beats are. Measuring the signed nearest-partner offset (B − A, no tolerance cap,
19,834 pairs; `s1_6_posthoc_detector_offset.json`):

- **median offset = 11.0 samples = 85.94 ms, IQR [11.0, 11.0]** — a constant, deterministic lag.
- only 15.2 % of pairs fall within the 6-sample match tolerance.
- removing that **single global offset** raises agreement from **0.1446 → 0.7861** macro
  (`an0` 0.7273, `k2s` 0.8448).

This is **POST-HOC and additive**: it replaces nothing, and the preregistered S1.6 values above stand as
the record. Its consequence is that the 0.1446 figure is dominated by a fixed implementation lag in the
`pantompkins1985` pipeline rather than by genuine reference ambiguity — and equally, that **residual
disagreement at 0.786 after the lag is removed is real and is not negligible.**

**Nothing here says anything about `kjd`/`ssx`.** That audit is a separate protocol whose QA rules must be
frozen before any new model's test performance is seen.

---

## Evidence classification (preregistration §13)

Multiple cases hold; no single story is forced.

- **CASE D — HOLDS, decisively.** Oracle gain equals its null to within 0.0007. Displacement claims based
  on `oracle_corr` are withdrawn.
- **CASE F — HOLDS.** The MSE regressor has the highest event F1 (0.4871) and the highest excess over
  chance (+0.3858) while retaining oracle QRS energy 0.0105, HF ratio 0.0073 against a GT HF of ~0.19, a
  QRS width error of 98.59 ms, beats ratio 0.7720 and oracle-absent 0.7465. *Event F1 alone admits a
  degenerate smooth solution and cannot serve as a standalone reconstruction objective.*
- **CASE C — HOLDS in weakened form, via a different instrument.** 42.7–52.0 % of unmatched GT beats have a
  **detected** predicted peak at 50–150 ms. This is detector-based and is untouched by the S1.4b failure.
  It is weakened by two things: no chance floor for the DISPLACED class was preregistered or computed, and
  a ±150 ms band around ~9.7 beats per 8 s window is not a small target.
- **CASE B — HOLDS.** All-GT-beat fidelity is low for every arm on the one uncontaminated statistic
  (`same_coord_corr` 0.096–0.243), ABSENT is the largest or joint-largest class for every arm after the
  ±150 ms repair, and every arm's F1 excess over chance is ≤ +0.386.
- **CASE E — PARTIALLY HOLDS.** Preregistered agreement is 0.1446, but the POST-HOC diagnostic shows this
  is dominated by a constant 85.94 ms lag; residual disagreement is 0.786, which is material but far from
  the preregistered reading.
- **CASE A — NOT SUPPORTED as stated.** It requires high all-GT waveform fidelity, which no arm has
  (`same_coord_corr` ≤ 0.243).

---

## Answers to the five questions (preregistration §14)

**1. Is there evidence for a real joint event-fidelity / waveform-fidelity gap after measurement confounds
are controlled?** Yes. After removing chance (a quarter of every raw F1) and discarding the oracle-inflated
statistic, no arm reaches both competent event fidelity and competent waveform fidelity: the two arms with
real QRS structure (OT-CFM-50, iMF-8) sit at F1 excess +0.32 to +0.36 with `same_coord_corr` 0.10–0.12,
while the arm with the best event score and the best `same_coord_corr` (MSE, +0.3858 and 0.2425) has 1 % of
GT oracle QRS energy and a 98.59 ms QRS width error. The gap is real on this development population.

**2. How much of the low model F1 is attributable to each cause?** For iMF-8, of 19,834 GT beats: 42.8 %
matched, and of the 11,607 unmatched, 45.0 % have a detected peak at 50–150 ms (DISPLACED), 1.4 % WEAK,
53.6 % ABSENT, 0 lost to the matching rule. Chance matching accounts for 0.1141 of the raw 0.4341, i.e.
26.3 %. Detector/morphology interaction is bounded from G1 rather than measured here: a realistic full-beat
stamp at exactly correct positions already costs 0.14 F1, so it is non-zero, and it is not separated from
true absence by anything that ran.

**3. Does MSE's high F1 survive when detector-independent waveform fidelity is considered?** No. It has the
highest F1 and the highest `same_coord_corr`, and it fails four of the five preregistered gate criteria —
oracle QRS energy 0.0105 (1.5 % of OT-CFM-50's 0.7012), HF ratio 0.0073 against a GT HF of ~0.19, QRS width
error 98.59 ms, beats ratio 0.7720 — with the highest oracle-absent rate (0.7465). Its high
`same_coord_corr` is consistent with a smooth predictor tracking low-frequency structure, which makes
`same_coord_corr` unusable as a standalone quality metric too.

**4. Does increasing NFE move models toward a better joint frontier, or trade axes?** Mostly trade. From
NFE 1 → 50 the iMF F1 excess moves +0.2943 → +0.3148 (peaking at NFE 4, +0.3204), matched morph 0.6679 →
0.7775, and oracle QRS energy rises monotonically 0.4368 → 0.5712 — while beats ratio falls monotonically
0.9810 → 0.8863, oracle-absent worsens after NFE 4 (0.3298 → 0.3537), and ABSENT rises 48.7 % → 56.3 %.
Under the frozen gate rule iMF-4 and iMF-8 pass and iMF-50 fails on beats ratio.

**5. Are the current metrics trustworthy enough to support a next-method preregistration?** Partly, with
two named repairs. Trustworthy: event F1 with a count-matched chance floor (G1 validated the instrument;
S1.4c calibrates it), the structural axes, coverage, and beats ratio. **Not trustworthy as used:**
`oracle_corr` and any statistic derived from the same ±150 ms shift maximisation, including
`oracle_qrs_energy_median` and `oracle_absent`, which inherit the shift selection and were never null-
calibrated. Also unresolved: no chance floor exists for the DISPLACED class, and the reference-detector
question is only bounded by a POST-HOC diagnostic on development data.

---

## What is supported

- The frozen event metric, with a chance floor attached, distinguishes arms on this population.
- Roughly a quarter of every arm's raw event F1 at 50 ms is quasi-periodic coincidence.
- A zero-parameter DSP floor reaches +0.0741 excess over chance at 50 ms — far below every model.
- The one-to-one matcher denies no unmatched GT beat a within-tolerance peak: `contested` = 0 everywhere.
- `oracle_corr`'s elevation above `same_coord_corr` is entirely a shift-selection effect.
- No arm achieves competent event fidelity and competent waveform fidelity simultaneously.
- Event F1 alone admits a degenerate smooth solution that fails four of five frozen structural criteria.
- The published ~6.7 ms `ppg_shuffle` timing figure rests on 67–73 % exact-zero pairs and reverses under an
  uncensored matcher.

## What is NOT supported

- Any displacement claim resting on `oracle_corr`, `oracle_qrs_energy_median` or `oracle_absent`.
- Any statement that the source matters more than the PPG, or the converse.
- Any conclusion about `kjd`/`ssx` reference quality, or about the test set at all.
- That the reference ECG is unreliable: the preregistered 0.1446 is dominated by a constant 85.94 ms lag.
- That the DISPLACED class exceeds chance — no floor for it was preregistered or computed.
- Any population-level inference: n = 2 development subjects, seed 0, previously visually inspected.
- Any comparison of these numbers to the recorded 4-seed-pooled X4-0 table.
- Any method recommendation. S1 produces none, by design.

## Artifacts

`artifacts/s1_metric_validity/` (gitignored): `s1_2_dsp_floor.csv`, `s1_3_joint_fidelity.csv`,
`s1_4_event_failure_classes.csv`, `s1_4_oracle_null.csv`, `s1_4_chance_floor.csv`,
`s1_5_c2_reanalysis.csv`, `s1_6_gt_reliability.csv`, `s1_6_posthoc_detector_offset.json`,
`bootstrap_summary.json`, `provenance_s1_remaining.json`, and figures
`joint_fidelity_frontier.png`, `event_failure_decomposition.png`, `matched_vs_all_gt_morphology.png`,
`oracle_true_vs_null.png`, `gt_detector_agreement.png`. No prediction dumps.
