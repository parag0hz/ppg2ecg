## Conditional fidelity
WildPPG changes the character of the OT-CFM 1-NFE failure. The four WildPPG devices are time-synchronised, so the one-step
conditional mean E[x₁ | PPG] is a *beat-aligned* average: OT-CFM-1 keeps the PPG dependence (shuffle gain 6.64 bpm vs 7.16 at 50 NFE),
the best beat timing of all arms (R-peak F1 0.48, PCC 0.14) and a moderate HR error (15.6 bpm), while losing amplitude (0.32) and QRS
sharpness (QRS-width error 75 ms, template corr 0.38). iMeanFlow-1 restores amplitude (1.04) and sharpness (41.7 ms; corr 0.55) and
improves HR (11.85, CI 11.5–12.3 vs 15.0–16.2), but its output depends *less* on the PPG than either OT-CFM arm (gain 4.29; right-target
HR error 11.9 vs wrong-target 16.2) and its beat timing is less precise (F1 0.385, PCC 0.05). With 2–4 MeanFlow steps the gain rises to
5.2–5.4 bpm and morphology to 0.60–0.64, HR stays ≈ 11.6–12.2 — still short of the 50-NFE reference on HR and gain. Per site
(sternum/head/wrist/ankle) the pattern is uniform: OT-1 HR 16.1/13.0/14.0/19.3 → iMF-1 12.0/10.7/12.0/12.8 (OT-50 9.7/8.3/9.1/10.6).

## Qualitative examples
Pre-registered windows (test kjd/ssx, OT-50 HR-error quantiles): on a clean window (ssx 2439) OT-CFM-1 shows small, correctly timed
QRS bumps riding on a flat baseline — the attenuated aligned mean — whereas iMeanFlow-1 produces full-amplitude spikes with more
baseline wander and a few extra spikes; on the noisy-ECG participant kjd (windows 297, 415) OT-CFM-1 is flat, OT-CFM-50 keeps sharp
beats, and iMeanFlow-1 produces plausible beats on 297 but a wandering, partly spurious trace on 415. Same PPG, same noise, same scale.

## Failure taxonomy
- F1 conditional-mean collapse (iMF-1): absent (amp 1.04, seed std 0.30).
- F2 QRS smoothing: partial (corr 0.551 vs 0.670; QRS-width error 41.7 vs 29.9 ms) — recovered to 0.637 / 33.7 ms with 4 steps.
- F3 amplitude collapse: absent (recovery 0.91) — the clearest replicated effect.
- **F4 conditioning neglect: PRESENT relative to this dataset's baseline** — iMF-1 gain 4.29 < OT-CFM-1 6.64 < OT-CFM-50 7.16
  (recovery −4.5 by the pre-registered formula, because the OT-CFM-1 reference already retains almost all of the gain).
- F5 unstable training: absent (66 rounds, best 46, smooth loss).
- **F6′/F8 beat-timing imprecision**: iMF-1 R-peak F1 0.385 vs 0.481 (OT-1) / 0.440 (OT-50); beats/reference 0.93 — the one-step
  generative sample places sharp beats less precisely than the aligned mean does (new sub-type on synchronised data).
- The OT-CFM 1-NFE failure on WildPPG is itself a different type than on DaLiA: F3+F2 (amplitude/sharpness collapse of an aligned mean)
  rather than F1 (beat-free mean).

## Limitations
- Single seed; two test participants (one flagged "noisy ECG" by the dataset authors); 4,096-window uniform test subset; validation on a
  3,785-window subset; rounds of 220 steps instead of epochs (pre-registered); 861 constant-gap windows dropped (0.22 %); four PPG sites
  pooled as in PENGUIN (per-site numbers reported). Same frozen recipe as A2/A3 (baseline optimiser, boundary v_θ, h-only conditioning).
- The conditioning-gain recovery score is ill-conditioned when OT-CFM-1 retains most of the gain (denominator 0.52 bpm); the
  pre-registered rule nevertheless flags it, and we report it as such.

## Verdict rationale
iMeanFlow-1 improves 3 of 4 physiological metrics over OT-CFM-1 (HR, morphology, amplitude) with recovery 0.61 / 0.59 / 0.91 and beats
0.93, but conditioning gain is worse than OT-CFM-1 (severe negative recovery) → **PARTIAL** under both the replication rule and the A2 rule.
Ordering test: A (OT-1 ≪ OT-50) holds for HR (+6.2 bpm), morphology (−0.29) and amplitude (0.32) but *not* for conditioning gain or beat
timing; B (iMF-1 ≫ OT-1) holds for HR/morphology/amplitude, not for gain/timing; C (iMF-1 → OT-50) leaves +2.4 bpm HR and −0.12 corr.
Pointwise-error inversion replicates (OT-1 RMSE 0.355 is the best while its physiology is the worst).

## Recommended next research question
The effect that replicates across subject *and* dataset is the recovery of **amplitude and QRS sharpness** in one step. Whether one-step
*generation* is also needed for rhythm/conditioning depends on whether the PPG–ECG pairing is beat-synchronised: on WildPPG the one-step
conditional mean already carries rhythm and conditioning. Next: (1) quantify this explicitly — compare iMF-1 against the one-step
conditional-mean regressor as a *second baseline* on both datasets; (2) seeds/folds for variance; (3) a DaLiA beat-level protocol with
documented re-synchronisation to test whether the DaLiA result changes character once the pairing is aligned.
