# M1 — C1 Structural Mechanism Audit — PREREGISTRATION

**Status:** frozen at this commit, pushed **before any M1 prediction or result is generated**.
**Type:** existing-checkpoint forward inference and analysis only.

**NO NEW TRAINING. NO MULTI-SEED. NO TEST. NO NEW SAMPLER. NO NEW LOSS. NO NEW MODEL.
NO HYPERPARAMETER TUNING.**

---

## 0. What this study is, and is not

**M1 was designed AFTER seeing the C1 result.** It is a **mechanism-generating diagnostic / falsification
audit**, not a confirmatory discovery, and it is **not independent of C1's M1/M2 finding** — it exists
precisely because of it. No claim of independent confirmation may be made from M1, and M1 alone cannot
establish novelty.

> **Does targeted h = 0.5 exposure improve ECG reconstruction specifically around high-curvature QRS
> structure, or does it merely improve global waveform fitting / calibration?**

Three candidate readings of the C1 H50 effect are to be separated: **(A)** genuine QRS/event-local
waveform-structure improvement; **(B)** broad, generic fitting improvement across the whole waveform;
**(C)** an aggregate QRS-energy / amplitude **calibration** effect with no local structural support.

## 1. The confound that M1 cannot remove

**The C1 arms are not compute-matched:** B ran 66 rounds, H25 68, H50 **101**. M1 reuses those exact
checkpoints, so **even verdict A does not establish an interval-exposure effect.** Throughout the report,
these must be kept apart:

- **OBSERVATION** — the H50 checkpoint exhibits X.
- **CAUSE** — h = 0.5 exposure caused X. **Unresolved, because H50 trained longer.**

M1 decides only whether a potentially interesting structural phenomenon exists and is worth a
compute-matched follow-up.

## 2. Provenance and checkpoints — resolved, not guessed

Start HEAD `877841d8e90e679d3f4e4d491ad3c51747f62a7f`; submodules PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`;
frozen A4 checkpoint md5 `31c042d291052fbb6dc15263ad316be2`. **C2 training never started** — see
`docs/C2_DEFERRED_BEFORE_TRAINING.md`.

Paths resolved from `scripts/eval_c1_arms.py:41-43` and verified against
`artifacts/c1_interval_exposure/stage2_result.json` (md5 match):

| arm | path | kind | rounds | best round | selection metric | sha256 (48) | size |
|---|---|---|---:|---:|---|---|---:|
| B | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` | BEST | 66 | 45 | 0.11945885431656277 | `557c70541f5cdd07819a3da04bb53477ac98827285507380` | 20,446,965 B |
| H25 | `outputs/c1_imf_h25_seed42/checkpoint_best.pt` | BEST | 68 | 47 | 0.12780840157960571 | `c1c1b09bd84843dd61e4bb6cefd887edacd4760a8d96bb00` | 20,446,965 B |
| H50 | `outputs/c1_imf_h50_seed42/checkpoint_best.pt` | BEST | 101 | 80 | 0.11824402330613042 | `e9eb78f37726dd157dbe987432c8571d2328fa1d00ea0342` | 20,446,965 B |

**No checkpoint is overwritten.**

## 3. Population and source

The frozen C0/C1 development subset: `an0` 1,024 + `k2s` 1,024 = **2,048 windows**
(`select_subset("x4-event-nfe-v2", …, 1024)`). **`kjd`/`ssx` are never loaded** — no test metadata, no test
visualisation, no test QA.

Evaluation Gaussian source **seed 0**, **the identical source tensor for every arm and every NFE**
(bank sha256 `868085798050102e…`, as in C1). Arms evaluated at **NFE 2** (primary) and **NFE 4**
(NFE-specificity).

## 4. GT event reference — coordinates only

GT R-peaks come from the frozen `rpeaks.detect_rpeaks` at its existing configuration, and are used
**only to define local coordinates**. **Predictions are never shifted or aligned.**

**Prohibited everywhere:** predicted-peak alignment, oracle translation, max-correlation shift,
`oracle_corr`, `oracle_qrs_energy`, `oracle_absent`. Every quantity is computed at original fixed
coordinates.

## 5. Question A — where does the improvement occur?

For every sample, τ = signed distance to the nearest GT R-peak. Regions, with the sample conversion frozen
at `fs = 128` using `round(ms/1000 · fs)`:

| region | definition | samples |
|---|---|---|
| **QRS-core** | \|τ\| ≤ 80 ms | \|τ\| ≤ **10** |
| **peri-QRS** | 80 ms < \|τ\| ≤ 250 ms | 10 < \|τ\| ≤ **32** |
| **background** | \|τ\| > 250 ms | \|τ\| > **32** |

Per region and arm, at original coordinates with no translation:

**A1** mean \|pred − GT\| · **A2** mean (pred − GT)² · **A3** mean \|D(pred) − D(GT)\| with
`D x[n] = x[n+1] − x[n]` · **A4** local amplitude error, \|ptp(pred) − ptp(GT)\| over the region's samples
within each beat neighbourhood · **A5** local signal-energy error, \|Σpred² − ΣGT²\| normalised by ΣGT².

## 6. Event-centred error profile

Around each GT R-peak over τ ∈ [−250, +250] ms (**±32 samples, 65 points**), fixed coordinates:
`E_arm(τ)` = equal-subject-weighted mean error.

Reported curves at NFE 2: the three arms' absolute-error profiles; `Δ_B(τ) = E_B(τ) − E_H50(τ)`;
`Δ_25(τ) = E_H25(τ) − E_H50(τ)`. Positive = H50 better. The same three for derivative error.

**No smoothing is applied to the reported curves.** Raw per-τ values are written to
`event_error_profiles.csv` regardless.

## 7. QRS-localisation test

Per bootstrap replicate, with `R` the **relative** error reduction of H50 vs H25:

```
R_core = (E_H25(core) − E_H50(core)) / E_H25(core)
R_bg   = (E_H25(bg)   − E_H50(bg))   / E_H25(bg)
L      = R_core − R_bg
```

Positive `L` = the H50 advantage is relatively more concentrated near QRS. Computed for **both** waveform
squared error and derivative absolute error. **This is not a causal statement.**

## 8. Question B — frequency structure

One frozen spectral method for all arms: `scipy.signal.welch(x, fs=128, nperseg=256, noverlap=128,
window="hann", detrend="constant")`, applied identically to GT, prediction and residual. Bands:
**F1** 0.5–4 Hz · **F2** 4–8 Hz · **F3** 8–15 Hz · **F4** 15–64 Hz (Nyquist).

Per band: reconstruction-error spectral energy, GT spectral energy, predicted spectral energy, and
\|pred/GT energy ratio − 1\|. Compared H25 vs B, H50 vs B, H50 vs H25 at NFE 2 (NFE 4 secondary). The
existing historical HF > 15 Hz statistic is retained for comparability, but **HF is not treated as
synonymous with QRS.**

## 9. Question C — QRS-scale morphology, at GT-fixed coordinates only

Within QRS-core: local RMSE · local ptp · QRS-energy ratio deviation · slope ratio deviation · derivative
RMSE · maximum absolute derivative · curvature, using the fixed second finite difference
`D2 x[n] = x[n+1] − 2x[n] + x[n−1]`, reported as mean \|D2 pred − D2 GT\|.

**No peak alignment. No metric conditional on predicted-peak matching may be primary evidence.**

## 10. Statistics

Subject-stratified paired bootstrap, equal `an0`/`k2s` weight, **2,000 resamples,
`default_rng(20260901)`**, positive oriented difference = H50 better.
Primary at NFE 2: `H50 vs H25`, `H50 vs B`. Secondary: `H25 vs B`, and all three at NFE 4.
Point estimate, paired difference and 95 % CI for every metric.

**Only one training seed exists per C1 arm. The bootstrap reflects uncertainty over this frozen
development-window population only, and must not be presented as training-run uncertainty.**

## 11. Frozen visual atlas

Reuses the already-frozen metadata-only C2 cohort: salt **`c2-visual-atlas-v1`**, 64 windows, 8 strata
(`an0`/`k2s` × sternum/head/wrist/ankle), 8 per stratum. **The cohort is not redefined from outputs**, and
its indices are verified before any prediction is loaded.

Panels per window: PPG · GT ECG · B@2 · H25@2 · H50@2 · B@4 · H25@4 · H50@4. Views: **(A)** full 8 s;
**(B)** GT-R-centred ±250 ms zoom; **(C)** QRS-relevant component, using the filter frozen here —
zero-phase Butterworth band-pass 8–40 Hz, order 4, `scipy.signal.filtfilt`; **(D)** first derivative;
**(E)** local energy envelope, squared signal smoothed by a 25-sample (≈195 ms) moving average;
**(F)** prediction − GT residual.

GT R locations appear as vertical reference lines only. **Predictions are never translated.** Contact
sheets by subject and by PPG site. **No "best" or post-hoc "representative" examples.**

## 12. Site-wise analysis — exploratory

At NFE 2, split by `sternum` / `head` / `wrist` / `ankle` for B/H25/H50: QRS-core RMSE, background RMSE,
derivative QRS-core error, QRS-energy deviation, p2p deviation, raw correlation, F1 excess, beats-ratio
deviation. **No measurement-information causality may be inferred.**

## 13. NFE 2 vs NFE 4 localisation

`E2` = H50 improvement vs B at NFE 2, `E4` = at NFE 4, `D = E2 − E4`, for QRS-core waveform error, QRS
derivative error, QRS-energy deviation, p2p deviation and background waveform error. `D > 0` = larger
benefit at NFE 2; unresolved = broad model improvement; `D < 0` = benefit at least as strong at NFE 4.
**No equivalence claims.**

## 14. Diagnostic verdicts

### A — `QRS-LOCALIZED STRUCTURAL SIGNAL` — only if **all four** hold
1. H50 vs H25 clearly improves at least one direct fixed-coordinate QRS-core waveform metric
   (QRS-core squared error **or** QRS-core derivative error);
2. for at least one of those two families, the localisation contrast `L` has a 95 % CI entirely > 0;
3. H50 vs H25 clearly improves at least one additional structure-sensitive quantity (QRS-energy deviation,
   p2p deviation, derivative/curvature error, or upper-band spectral reconstruction error);
4. the visual atlas reveals no obvious metric pathology such as globally rescaled but misplaced complexes.

Permitted: *"Under the frozen single-seed development protocol, the H50 advantage is not purely global:
part of the improvement is localized around GT-defined QRS structure."*
**Not permitted:** *"h = 0.5 causes QRS learning"*, *"interval scale determines physiological structure."*

### B — `BROAD / GENERIC FIT IMPROVEMENT`
H50 improves waveform errors, but QRS-core relative improvement is not clearly stronger than background and
H50/H25 structural specificity is weak. Permitted: *"The C1 gain appears broad rather than QRS-localized."*
Consequence: do not build a QRS-specific novelty story from C1.

### C — `CALIBRATION-ONLY / STRUCTURAL SUPPORT NOT FOUND`
Aggregate QRS-energy/p2p metrics improve, but direct fixed-coordinate QRS waveform/derivative evidence does
not support local structural improvement. Permitted: *"The QRS-energy and p2p gains are better interpreted
as calibration effects under this audit, not as evidence of improved local QRS reconstruction."*
Consequence: kill the proposed interval → QRS-structure mechanism story.

## 15. Next-step rule

M1 returns a **recommendation only**. **No new training is started, in any branch of the outcome.**
Under A: recommend a cheap seed-42 compute-matched B/H25/H50 (or only the missing fixed-budget H25/H50 runs
if B@66 is reusable) **before** multi-seed. Under B: do not prioritise the QRS-specific mechanism. Under C:
stop the H50 structural-mechanism direction.
**Do not auto-start C2's 15 runs. Do not start distillation. Do not implement a new method.**

## 16. Deliverables

`docs/M1_C1_STRUCTURAL_MECHANISM_AUDIT_REPORT.md`; `artifacts/m1_c1_structural_audit/` with
`provenance.json`, `checkpoint_manifest.json`, `cohort_manifest.json`, `metrics_window.csv`,
`region_metrics.csv`, `event_error_profiles.csv`, `spectral_metrics.csv`, `site_metrics.csv`,
`paired_bootstrap.csv`, `nfe_interaction.csv`, `decision.json`, `visual_atlas/`; figures
`m1_event_error_profile_nfe2.png`, `m1_h50_minus_h25_event_profile.png`,
`m1_qrs_vs_background_improvement.png`, `m1_derivative_event_profile.png`,
`m1_frequency_error_decomposition.png`, `m1_qrs_structure_metrics.png`,
`m1_sitewise_structure_effect.png`, `m1_nfe2_vs_nfe4_effect.png`.

Checkpoints and prediction dumps never enter git. Submodules stay byte-identical.
