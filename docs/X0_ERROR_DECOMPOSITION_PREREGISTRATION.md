# X0 Pre-registration — Decomposing One-Step ECG Failure (timing vs event vs shape vs amplitude vs sharpness)

Written 2026-08-29 **before any X0 metric is computed** (prediction provenance was recorded first; no model is trained; no prediction
is regenerated). Frozen by the commit that introduces it. X0 is an ANALYSIS experiment on existing frozen artefacts.

## Question
> When a one-step PPG-conditioned ECG prediction looks severely degraded, which error component is responsible — absolute/global
> timing, beat-placement variability, missing/spurious events, genuine beat-shape smoothing, amplitude attenuation, or QRS
> sharpness / high-frequency loss? In particular: when OT-CFM-1 looks "completely wrong", is the correct ECG-like structure present
> nearby in time, or has that structure itself been destroyed?

## Competing hypotheses (none privileged)
H-X0-TIMING: a substantial fraction of the apparent one-step morphology deficit is recoverable by oracle temporal alignment.
H-X0-SHAPE: attenuation remains severe after alignment (genuine shape loss). H-X0-MIXED: both contribute materially.
H-X0-EVENT: missing/spurious events dominate rather than continuous lag.

## Frozen data, models, provenance
Primary: WildPPG, A4/A6 original representation, the frozen 3,907-window test subset (kjd, ssx; 8 (subject, site) clusters of
474–503 windows). Secondary: PPG-DaLiA S2 (1,025 windows) and S1 (1,151). Models (existing frozen prediction arrays only; hashes in
`artifacts/x0_error_decomposition/prediction_provenance.json`): MSE = A6 full-backbone proxy (a6c/a6a/a6b), OT1 = OT-CFM Euler-1,
iMF1 = iMeanFlow-1, OT50 = OT-CFM Heun-25 (50 NFE); iMF2/iMF4 context only (WildPPG). Paired noise seed 0. Nothing regenerated.
DaLiA wording rule: *beat-level temporal correspondence is less reliable under the current raw PPG-DaLiA evaluation protocol*, so
DaLiA timing analyses are secondary/exploratory.

## Frozen analysis definitions (`src/ppg2ecg/evaluation/alignment_diagnostics.py`, unit-tested)
- Detector / matching: frozen `rpeaks.detect_rpeaks` (neurokit, same cleaning on pred and GT) and `match_rpeaks` (one-to-one, 50 ms).
- **Level 1 raw**: frozen metrics (RMSE, MAE, PCC, template morphology on matched beats, amplitude ratio, HF > 15 Hz, R-peak P/R/F1,
  HR error, RR MAE, beats/reference) plus QRS (±100 ms) energy retention and max-slope ratio.
- **Level 2 global oracle lag**: integer lag L ∈ [−32, +32] samples (**±250 ms**) maximising the normalised cross-correlation of the
  overlapping samples (ties → smallest |L|); explicit crop of the vacated edge (no wrap); metrics recomputed on the overlap.
  `Δ_global = metric_after − metric_before`. A diagnostic that uses GT — never a deployable metric.
- **Level 3 event timing**: GT/pred R-peak counts, matched/missing/spurious counts, signed matched-peak error (pred − GT, ms), bias,
  SD, MAE; `missing_rate = FN / GT beats`, `spurious_rate = FP / GT beats` (both relative to GT beats).
- **Level 4 beat-centred shape (detector-independent, GT-anchored)**: for every GT R-peak whose frozen beat window (−0.25 s, +0.40 s)
  plus a ±19-sample margin lies inside the window (edge beats counted as skipped): (A) prediction segment at the SAME coordinates;
  (B) **oracle local translation** d ∈ [−19, +19] samples (**±150 ms**) maximising Pearson correlation (translation only; no
  amplitude scaling, warping or width change). Per beat: correlation, peak-to-peak amplitude ratio, R-sample amplitude ratio,
  QRS (±100 ms) energy ratio, max |slope| ratio, HF ratio, segment RMSE, QRS RMSE. Flattened predictions are included even when the
  predicted-peak detector finds nothing. Called *oracle local shape recoverability*, never "timing-corrected performance".
  A GT beat is **oracle-absent** if oracle correlation < 0.5 OR peak-to-peak ratio < 0.2 (no ECG-like structure within ±150 ms).
- **Recoverability**: Q = mean beat correlation; `raw_gap(M) = Q_A(OT50) − Q_A(M)`, `aligned_gap(M) = Q_B(OT50) − Q_B(M)`,
  `recoverable_gap_fraction = (raw_gap − aligned_gap)/raw_gap`, reported only when raw_gap ≥ 0.05 (else "unstable"); values > 1 or
  < 0 are reported as such. Also Δ_global and Δ_local per model on the frozen matched-beat morphology and on Q.
- **Pair similarity** (MSE vs OT1, and all pairs): waveform RMSE/PCC raw; after each model's global alignment to GT (both in the GT
  frame); and between oracle-aligned beat segments; plus amplitude/QRS-energy/slope/HF distances.
- **RMSE decomposition**: raw RMSE, globally aligned RMSE, oracle-local QRS RMSE, non-QRS RMSE, vs morphology/QRS energy/slope.
- **Statistics**: WildPPG cluster bootstrap over the 8 (subject, site) clusters (2,000 resamples) for means and paired deltas;
  DaLiA window bootstrap labelled exploratory (one test subject). Bootstrap uncertainty does not capture between-training-seed
  uncertainty. No p-values.
- **Qualitative windows**: the frozen IDs only — WildPPG [2439, 297, 415] (+ fixed 0/976/1953/2930/3906), S2 [880, 482, 824], S1
  [1067, 726, 221].

## Frozen decision rules (WildPPG primary; per model, then overall)
Per-model category: **EVENT-DOMINANT** if oracle-absent rate ≥ 0.5; otherwise **TIMING-MAJOR** if recoverable_gap_fraction ≥ 0.5 AND
after local alignment each of (peak-to-peak ratio, QRS energy ratio, slope ratio) ≥ 0.5 × OT50's aligned value; **SHAPE-DOMINANT** if
recoverable_gap_fraction < 0.2 AND each of the three retention ratios < 0.5 × OT50's; **MIXED** otherwise (0.2–0.5 recoverable, or timing
recoverable but shape materially deficient). Error-type labels (LOW/MODERATE/HIGH): timing MAE < 20 / 20–40 / > 40 ms; event =
oracle-absent rate < 0.1 / 0.1–0.3 / > 0.3; shape = oracle beat corr ≥ 0.8 / 0.6–0.8 / < 0.6; amplitude = p2p ratio ≥ 0.8 / 0.5–0.8 / < 0.5;
sharpness = slope ratio ≥ 0.8 / 0.5–0.8 / < 0.5. Dominant failure = the component with the largest deficit fraction (timing
min(1, MAE/50 ms); event = absent rate; shape 1 − corr; amplitude 1 − p2p; sharpness 1 − slope), computed on oracle-aligned values.
Overall WildPPG verdict = OT1's category (the central question), reported as **MODEL-DEPENDENT** if MSE and OT1 fall in different
categories, and **INCONCLUSIVE** if the recoverability quantity is unstable for both. DaLiA gets the same rules, labelled secondary.

## GO / STOP after X0
TIMING-MAJOR or strong MIXED (consistently on WildPPG) → recommend X1 (controlled temporal-ambiguity synthetic experiment); not run.
SHAPE-DOMINANT → do not pursue event-aligned transport; next question = what morphology information is missing from PPG conditioning
and why instantaneous one-step transport attenuates it. EVENT-DOMINANT → event-generation/count modelling first. INCONCLUSIVE →
report why the data cannot distinguish mechanisms; no new method.

## Forbidden
Changing the lag/shift ranges, detector, tolerance, thresholds or qualitative IDs after results; excluding windows by appearance;
regenerating predictions; training; any wording from the standing forbidden list (multimodality, phase modes, "transport is
necessary", "ECG is inherently ambiguous", "timing uncertainty causes collapse", conditional-mean theorem, "proves temporal ambiguity").
