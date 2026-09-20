# ED1 — Event-level consensus decoding: from K cheap samples to one waveform (preregistration; no training)

Frozen and pushed before any ED1 number. V1 VitalDB data; seed-42 checkpoints of arm I (iMF) and arm D (consistency
distillation), NFE 1, K = 16 samples per window (noise seeds 0–15).

## Algorithm (fixed)
1. Detect R peaks in each of the K samples (neurokit, the project's detector).
2. **Consensus events.** Pool all peaks of the window; count votes inside ±w of every position; keep local maxima with
   ≥ θ·K votes, ≥ 250 ms apart; the event position is the median of the votes inside its ±w box, plus a constant
   shift b (samples).
3. **Consensus waveform.** For each event, every sample that voted contributes its own 83-sample beat (R at index 32),
   cut around **its own** peak; the beats are averaged **after alignment** and written at the event position.
   Overlaps are averaged; samples outside any beat window take the pointwise median of the K samples.
4. **Per-beat reliability** = vote fraction and the SD of the voting positions (reported, not used for decoding).

## Parameters — chosen on validation patients only, then frozen
Grid w ∈ {25, 40, 50} ms × θ ∈ {0.25, 0.375, 0.5}, per generator, on the 289 V1 validation patients. b = median
signed timing error (consensus − target) of matched events at that configuration. Selection: highest standard-pipeline
R-peak F1 of the consensus waveform; ties within 0.002 → lower HR error. The chosen (w, θ, b) per generator are
committed as an amendment **before the test set is touched**, and the test set is evaluated once.

## Test evaluation (1,156 patients, 19,543 windows)
The consensus waveform goes through the **same standard pipeline as every other arm** (detector on the waveform → HR,
R-peak F1 @ 50 ms, RR-MAE, MAE, RMSE, FD) plus `morph_corr@GT` (PZ3). Paired, patient-clustered bootstrap (2,000).

## Decision, per generator, against its own single sample (arm I or D at NFE 1, noise seeds 0–3 average)
- **WORKS** iff F1 gain ≥ **+0.05** and `morph_corr@GT` gain ≥ **+0.15**, both with the 95 % CI excluding 0.
- **PARTIAL** iff exactly one of the two holds. **FAILS** otherwise.
Reported, not gated: the same contrasts against PENGUIN-50; HR error against the K = 16 HR-median consensus; FD;
reliability — Spearman(vote-position SD, |timing error|) over matched events, and F1 / coverage when only events with
vote fraction ≥ 0.75 are kept.
