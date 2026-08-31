# A9 — ECG Target-Representation Mirror Control (WildPPG)

> **Metric semantics note (added 2026-08-31).** In this document `morph` / `morphology_corr` is
> **matched-beat morphology correlation** and `rr_mae_ms` is **matched-consecutive-beat RR MAE**. Both are
> conditional on successful ≤50 ms one-to-one R-peak matching and are averaged only over matched beats, so
> neither can fall when a beat is missed. Values and specifications here are unchanged; only the labels and
> their scope are made explicit. See [METRIC_SEMANTICS.md](METRIC_SEMANTICS.md).


Pre-registration `docs/A9_ECG_TARGET_REPRESENTATION_PREREGISTRATION.md` (commit `fc3519d`, pushed before any A9 training). Artefacts
`artifacts/a9_ecg_representation_control/` (`normalization.json`, `representation_geometry.json`, `controlled_results.csv`,
`representation_comparison.csv`, `prediction_similarity.csv`, `qrs_region_analysis.csv`, `timing_analysis.csv`,
`conditioning_analysis.csv`, `summary.json`, `figures/`). Runs `outputs/a9_{mse_fullbackbone,otcfm,imeanflow}_wildppg_globalz_seed42`;
window-normalised references reused unchanged from A4 (OT-CFM, iMeanFlow) and A6c (MSE proxy).

## Research question
> Does the ECG one-step conditional-mean-like attenuation persist when the ECG target is represented by a single train-only **global**
> affine transform instead of the frozen protocol's **per-window** normalisation?

**Yes.** Every element of the ECG story reproduces in the new representation: the MSE proxy and OT-CFM-1 both attenuate morphology and
high-frequency content relative to the 50-NFE reference, OT-CFM-1 remains by far the closest model to the deterministic proxy,
iMeanFlow-1 recovers morphology (recovery 0.88 vs 0.59 under window normalisation), and the pointwise-error inversion reappears.
**Frozen verdict: REPRESENTATION-ROBUST.**

## Remaining representation confound (what this tests)
A2–A6 used per-window z-score + min-max ECG targets; A8's ABP work used a global train-only affine transform. A reviewer could therefore
attribute the ECG attenuation to per-window target normalisation. A9 changes only that one thing on WildPPG, with everything else
(protocol, split, subsets, seeds, example windows, metric code) frozen.

## Frozen protocol
WildPPG, split `split_a4_wildppg_seed42.json` (sha256 `bc168144…`): train 12, val an0/k2s, test kjd/ssx; 8 s windows @128 Hz; the same
3,907-window test subset; paired noise seed 0; PPG-shuffle derangement seed 1; 220-step rounds, patience 20, min_delta 1e-4; AdamW 1e-3 /
wd 0.01 / batch 64 / seed 42. DaLiA was not run.

## ECG source stage
`scripts/build_processed_wildppg.py --ecg-normalization none` → `data/processed/wildppg_8s_prenorm/`: raw `.mat` → FFT resample 128 Hz →
Butterworth 0.5 Hz high-pass (zero-phase) → windowing, i.e. **exactly the signal the frozen pipeline would normalise per window**, with
the normalisation omitted. 389,355 windows, 861 dropped — identical to the frozen dataset; PPG tensors, window indices and site labels are
**bit-identical** for all 16 subjects. The existing window-normalised targets were never inverted.

## Global train-only normalisation
`y_global = (y − 1.575417)/10501.669122` (native WildPPG ECG units), one scalar pair over 300,309,504 samples of the 12 train subjects,
computed under a file-open guard (`leakage_check.ok = true`, no val/test file opened). No per-window/subject/site statistic, no clipping,
no robust/min-max/quantile scaling. PPG preprocessing untouched.

## Leakage / parity checks (all passed before training)
Subject-disjoint split; A9 test windows asserted equal to A4's; PPG bit-exact; source stage recorded in the manifest
(`zscore/normalize = false`); affine round-trip unit-tested; guard proves no val/test statistics; window-disjointness and hash checks in
preflight (`PREFLIGHT OK` for all three runs); no NaN/Inf; window indices (hence PPG↔ECG alignment) unchanged; identical filtering and
resampling calls.

## Representation geometry (measured before training)
| Quantity | window-norm | global-z | native |
|---|---:|---:|---:|
| target mean / std | −0.371 / 0.363 | 0.001 / 1.223 | 13.85 / 12838 |
| per-window mean (sd across windows) | 0.265 | 0.031 | 327.7 |
| per-window std (mean ± sd) | 0.244 ± 0.042 | 0.709 ± 0.995 | — |
| ‖y‖ / ‖e‖ vs standard-normal prior | 0.493 | 0.710 | 7460 |
| ‖y − e‖ | 35.96 | 42.90 | 238,465 |
| prior share of interpolant energy at t = 0.25 / 0.5 / 0.75 | 0.292 / 0.788 / 0.971 | 0.069 / 0.401 / 0.858 | ≈ 0 |
| HF-energy ratio (> 15 Hz) | 0.1854 | 0.1854 | 0.1854 |
Both representations are O(1) against the prior, so — unlike A8, where raw ABP was 81.6× — **A9 isolates local vs global normalisation,
not a scale mismatch**. The substantive difference is amplitude heterogeneity: window normalisation equalises every window, global-z
preserves the native spread (per-window std sd 0.04 → 1.00; train-subject ECG std spans 2,533–24,720, a 9.8× range; 10 % of test windows
have global-z GT std ≤ 0.009). As pre-registered, the **median** amplitude ratio is the amplitude statistic (means are inflated by
near-flat windows: e.g. OT-50 mean 16.28 vs median 1.72). An affine map leaves the spectrum unchanged (identical HF ratio) and the frozen
R-peak detector gives identical beats in 100.0 % of test windows (position agreement 1.000), so morphology/timing/HF metrics are directly
comparable across representations.

## Training
| Run | window-norm (reference) | global-z (A9) |
|---|---|---|
| MSE proxy | 54 rounds (best 34), 1.1 h | 69 (best 49), 1.4 h, peak 20.7 GiB |
| OT-CFM | 210 (best 190), 5.1 h | 134 (best 114), 3.2 h, peak 20.7 GiB |
| iMeanFlow | 66 (best 46), 3.6 h | 28 (best 8), 1.5 h, peak 19.2 GiB |
No NaN/Inf, no deviation, no tuning. The global-z iMF run stopped much earlier (best round 8) — recorded as a limitation, not corrected.

## Controlled results (3,907 test windows; RMSE is representation-internal only)
### Window-norm reference (A4 / A6c, unchanged)
| Model | NFE | HR err | Morph | Amp (median) | Cond gain | beats/ref | HF (GT .263) | R-peak F1 | RR MAE | RMSE* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSE proxy | 1 | 20.21 | 0.316 | 0.19 | 4.95 | 0.71 | 0.007 | 0.421 | 17.3 | 0.350 |
| OT-CFM 1 | 1 | 15.59 | 0.379 | 0.25 | 6.64 | 0.78 | 0.065 | 0.481 | 15.1 | 0.355 |
| OT-CFM 50 | 50 | 9.43 | 0.670 | 0.97 | 7.16 | 0.98 | 0.263 | 0.440 | 21.2 | 0.440 |
| iMeanFlow 1 | 1 | 11.85 | 0.551 | 1.00 | 4.29 | 0.93 | 0.220 | 0.385 | 25.7 | 0.485 |
### Global-z (A9)
| Model | NFE | HR err | Morph | Amp (median) | Cond gain | beats/ref | HF (GT .280) | R-peak F1 | RR MAE | RMSE* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSE proxy | 1 | 13.71 | 0.345 | 0.64 | 6.70 | 0.84 | 0.008 | **0.487** | 16.8 | 0.314 |
| OT-CFM 1 | 1 | 20.89 | 0.325 | 0.52 | 4.62 | 0.71 | 0.079 | 0.434 | 16.0 | **0.313** |
| OT-CFM 50 | 50 | 10.48 | 0.644 | 1.72 | 5.36 | 0.99 | 0.225 | 0.398 | 25.0 | 0.620 |
| iMeanFlow 1 | 1 | 10.52 | 0.605 | 0.89 | 4.96 | 0.98 | 0.244 | 0.382 | 27.7 | 0.406 |
(2/4/10/20 NFE and iMF 2/4 in `controlled_results.csv`. The intermediate-NFE anomaly persists: OT-CFM at 2 NFE is the worst arm in both
representations — global-z HR 28.4, morph 0.203, HF 0.651.)

## Structural attenuation (frozen §10 definition)
| Representation | Model | morph < morph(O50) − 0.10 | \|amp−1\| worse by 0.10 | HF < ½·HF(O50) | **attenuation** |
|---|---|---|---|---|---|
| window-norm | MSE | ✓ (0.316 vs 0.670) | ✓ | ✓ (0.007 vs 0.263) | **YES** |
| window-norm | OT-1 | ✓ (0.379) | ✓ | ✓ (0.065) | **YES** |
| global-z | MSE | ✓ (0.345 vs 0.644) | ✗ (0.36 vs 0.72 — OT-50 *overshoots* here) | ✓ (0.008 vs 0.225) | **YES** |
| global-z | OT-1 | ✓ (0.325) | ✗ | ✓ (0.079) | **YES** |
Both mean-like models keep only 14–26 % of the GT variance inside the ±100 ms QRS windows (`qrs_region_analysis.csv`: QRS energy
retention MSE 0.256, OT-1 0.144 vs OT-50 2.002, iMF-1 0.416) and their maximum-slope ratios are 0.12 / 0.19 vs 1.06 / 0.86 — the QRS is
flattened in exactly the same way as under window normalisation (0.066 / 0.121 vs 0.759 / 1.134). The amplitude clause fails in global-z
only because OT-50 itself overshoots amplitude on the heterogeneous scale (median 1.72), which the pre-registered rule handles via the
HF alternative.

## Is OT-CFM-1 still closest to the deterministic proxy?
| Distance from the MSE proxy | window-norm | global-z |
|---|---|---|
| R–O1 waveform RMSE / PCC | **0.078 / 0.512** | **0.134 / 0.537** |
| R–O50 | 0.259 / 0.180 | 0.525 / 0.164 |
| R–M1 | 0.349 / 0.153 | 0.288 / 0.176 |
| statistic votes (Δamp, Δmorph, ΔHF) | O1 3/3 | O1 3/3 |
**Yes, in both representations** (H9.2 ✓): OT-CFM-1 is 3–4× closer to the deterministic proxy than the 50-NFE solution or iMeanFlow-1.

## iMeanFlow structural recovery
| | window-norm | global-z |
|---|---|---|
| morphology (O1 → M1 → O50) | 0.379 → 0.551 → 0.670, **recovery 0.59** | 0.325 → 0.605 → 0.644, **recovery 0.88** |
| amplitude fidelity (median) | 0.25 → 1.00 → 0.97, recovery 1.03 | 0.52 → 0.89 → 1.72, **denominator negative** (O50 overshoots) — reported as ill-conditioned; M1 is nonetheless the best amplitude model (\|amp−1\| = 0.11) and improves over O1 |
| QRS energy retention | 0.121 → 1.134 (O50 0.759) | 0.144 → 0.416 (O50 2.002) |
H9.3 ✓ in both, and the morphology recovery is *larger* under global normalisation.

## Beat timing (H9.4)
| Representation | model | F1 | precision / recall | RR MAE | beats/ref |
|---|---|---|---|---|---|
| window-norm | MSE / OT-1 / OT-50 / iMF-1 | 0.421 / 0.481 / 0.440 / 0.385 | .47/.40, .52/.46, .44/.44, .40/.38 | 17.3 / 15.1 / 21.2 / 25.7 ms | 0.71 / 0.78 / 0.98 / 0.93 |
| global-z | MSE / OT-1 / OT-50 / iMF-1 | **0.487** / 0.434 / 0.398 / 0.382 | .51/.47, .49/.41, .40/.40, .39/.38 | 16.8 / 16.0 / 25.0 / 27.7 ms | 0.84 / 0.71 / 0.99 / 0.98 |
The dissociation persists and is if anything sharper: the mean-like models (MSE, OT-1) have the **best** beat timing (F1 and RR MAE) while
having the **worst** morphology, and iMeanFlow buys morphology at the cost of beat-placement variability (RR MAE 27.7 ms, F1 0.382).

## Conditioning (same PPG-shuffle derangement)
window-norm gains 4.95 / 6.64 / 7.16 / 4.29 bpm (MSE / OT-1 / OT-50 / iMF-1); global-z 6.70 / 4.62 / 5.36 / 4.96. Every model keeps a
clear PPG dependence in both representations; the ordering shifts (the MSE proxy has the largest gain under global-z), which we record
without interpreting further.

## Pointwise-error inversion
| Representation | RMSE ranking | morphology ranking | inversion |
|---|---|---|---|
| window-norm | R < O1 < O50 < M1 | O50 > M1 > O1 > R | **YES** |
| global-z | O1 < R < M1 < O50 | O50 > M1 > R > O1 | **YES** |
In both representations the best-RMSE model is the worst on morphology (H9.5 ✓). Absolute RMSE values are never compared across
representations, only these rankings.

## Verdict
| Frozen condition (global-z) | Result |
|---|---|
| MSE attenuation persists | ✓ |
| OT-1 attenuation persists | ✓ |
| OT-1 remains closest to the MSE proxy | ✓ (3/3 votes) |
| iMF-1 materially improves morphology over OT-1 | ✓ (recovery 0.88) |
**REPRESENTATION-ROBUST.** H9.1 ✓, H9.2 ✓, H9.3 ✓, H9.4 ✓, H9.5 ✓.

## Alternative explanations
1. **Different optimisation trajectories.** Each objective's selection metric lives in the trained space, so the three global-z runs stop
   at different points than their window-norm counterparts (iMF at round 8 vs 46). The qualitative pattern is unchanged, but the exact
   numbers are not a like-for-like optimisation comparison.
2. **Amplitude heterogeneity.** Global-z preserves a 9.8× inter-subject amplitude range, which is why OT-50's median amplitude overshoots
   (1.72) and why 10 % of windows are near-flat. This changes what "amplitude fidelity" measures, which is exactly why the pre-registered
   amplitude clause has the HF alternative and why medians are used.
3. **Single seed, single dataset.** WildPPG only, one seed; DaLiA was deliberately not run.
4. **Both representations are affine images of the same filtered signal**, so no experiment here can separate "attenuation of the
   conditional mean" from "attenuation of whatever is hard to predict from PPG" — that is the A7/A8 target-dependence result, not this one.

## Limitations
One seed per model; 3,907-window test subset; the global-z iMF run early-stopped after 28 rounds (best 8) and may be under-trained;
RMSE/MAE are not comparable across representations by construction; QRS statistics use the frozen ±100 ms definition; the MSE proxy is
the A6 full-capacity control, not an exhaustively optimised regressor.

## Updated ECG-vs-ABP interpretation
- **ECG (A2–A6, now A9):** the one-step conditional-mean-like attenuation, the closeness of OT-CFM-1 to a deterministic MSE proxy, the
  MeanFlow structural recovery and the pointwise-metric inversion are **not artefacts of per-window target normalisation** — they
  reproduce under a single global train-only affine target, in some respects more strongly (morphology recovery 0.59 → 0.88).
- **ABP (A7, A8):** none of these appear, in either target representation, and a deterministic regressor is best or joint-best.
- Together: the attenuation tracks **how much task-relevant structure a deterministic conditional predictor can retain** — near-complete
  for PPG→ABP, far from complete for PPG→ECG — rather than target sharpness alone or the choice of target normalisation.

## Recommended next experiment
The remaining single-factor threats are now (i) seed/fold variance — one seed underlies every effect size in the programme, and
(ii) dataset scope for the representation control (A9 covered WildPPG only). A pre-registered **multi-seed replication of the WildPPG
trio (seeds 42/43/44)** would put confidence intervals on the effects that A2–A9 established; a DaLiA representation control is the
cheaper alternative but tests a weaker version of the same objection. Neither is started: **A9 ends here as pre-registered.**
