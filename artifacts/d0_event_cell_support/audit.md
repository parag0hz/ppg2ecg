# D0 audit (2026-10-01) — before any event-cell validation result

BF0 is closed (prereg `504395d`, result `d380e3b`; Case A, G1 FAIL, BF1 NO-GO). Nothing here changes it. The audit reads
the frozen BF0 code and outputs and records what D0 must hold fixed.

## 1. Frozen BF0 facts

| # | item | fact (source) |
|---|---|---|
| 1 | beat support | 166 samples, R at index 64: R − 64 … R + 101 samples = −500 … +789 ms (`beatfirst/beats.py`) |
| 2 | original overlap-add | Hann weights `numpy.hanning(168)[1:-1]` over each segment; `y(t) = Σ w_i(t) b_i(t) / Σ w_i(t)` where `Σ w_i(t) > 1e-6` (`beatfirst/render.py::assemble`) |
| 3 | original normalization | division by the local weight sum. A sample covered by one segment only shows that segment's value at full amplitude, however small its Hann weight. This exposes a beat's far tail wherever no neighbour covers it |
| 4 | fill rule | uncovered samples take the nearest covered value (ties to the left); a window without positions is the template baseline −0.5418 (median of the template's first and last 8 samples) |
| 5 | placed events | `outputs/bf0_beat_first/render_val.npz` (`pos_idx`, `pos_off`): `t_i = e_i + round(μ_i · 128 / 1000)`, clipped to [0, 511], sorted. 18,557 events in 4,822 windows; 438 windows have none |
| 6 | deterministic model | `outputs/bf0_beat_first/beat_deterministic.pt`, sha256 `e06a328d…` (equal to BF0's `checkpoint_deterministic.json`) |
| 7 | stochastic model | `outputs/bf0_beat_first/beat_stochastic.pt`, sha256 `5fc7a803…` (equal to BF0's record); Euler 8 steps |
| 8 | stochastic seed 0 | beat j of window n: `torch.Generator().manual_seed(1_000_003·n + j)` (`bf0_run.beat_seed(n, 0, j)`) |
| 9 | validation population | VitalDB V1 validation, 4,822 windows, 289 patients, 21,890 reference R. The test split is never loaded. **Not fresh:** ED1, FBC1, M1 and BF0 analyzed it, and the D0 hypothesis came from it |
| 10 | bootstrap convention | patient-clustered, equal patient weight; 2,000 resamples per-window, 1,000 for FD; seed 20260930; percentile 95% CI; one shared sequence of patient resamples (`bf0_run.patient_resamples`) |

## 2. Per-beat outputs and exact reproduction

- BF0 saved rendered windows only, not per-beat outputs.
- D0 regenerates the per-beat outputs with the frozen checkpoints, BF0's conditioning (`bf0_run.conditions`) and the
  frozen seeds, then renders them with the **original** renderer.
- Result (`original_reproduction.json`): **max |difference| = 0.0 and 4,822 / 4,822 windows bit-identical** for the
  stochastic seed-0, deterministic and template arms.
- The evaluation stage repeats this check and stops on any difference. It also stops unless the S-ORIG recomputation of
  BF0's G1 equals the published −0.0284 [−0.0312, −0.0257] to 1e-12.

## 3. A gap in the specification, resolved before results

A cell boundary at the midpoint `r_i ± RR/2` can lie outside the beat's own output support:

- the left side whenever RR > 128 samples (1 s), because the support starts at R − 64;
- the right side when RR > 202 samples.

Filling such a region with the nearest-value rule would put a step in the middle of the window. Frozen rule:

- The crossfade sits at the midpoint unless it would leave either beat's output support. In that case it moves the least
  amount that keeps it inside both.
- If the two supports overlap by less than the crossfade width, the crossfade spans the overlap.
- If the supports do not overlap at all (RR > 165 samples), the samples between them are filled with BF0's rule, as the
  original renderer also does.
- The crossfade is also capped at RR − 2 samples, so it never reaches an R sample.
- Beat i still receives zero weight within ±6 samples of r_{i+1} whenever RR ≥ 14 + width (unit-tested). The neighbouring
  QRS ghost stays excluded.

## 4. Crossfade width (TRAIN only)

`crossfade_train_selection.json`:

- **Data:** 3,000 salted TRAIN windows (`d0-crossfade-v1`). Positions come from the BF0 event stage on TRAIN PPG. No R
  detector was run on any render, and no F1, FD or validation number was used.
- **Statistics at each crossfade (± 1 sample):**
  - A = max |first difference|;
  - B = max |second difference|.
- **Criterion (written in code before it ran):** the smallest candidate in {4, 8, 12, 16} samples for which, in both D0
  arms, P95(A) and P95(B) are no larger than the P95 of the same statistics of real TRAIN ECG at midpoints between
  consecutive reference R.
- **Selected: 4 samples (31.25 ms); criterion met.**

**Limitation, documented without changing the choice:**

- The real-ECG P95 is dominated by artifact or noise windows and by midpoints that land on a QRS the reference detector
  missed. The median is 0.023 per sample, but the P95 is 0.994.
- Two other readings of the reference — only normally spaced R pairs, or all samples — give P95 values of 0.557 and
  0.654. Every candidate passes under all three, so the choice of reading does not matter and the criterion never bound.
- At 4 samples, the stochastic arm's median boundary jump (0.096) is about 4 times that of real ECG (0.023) and about 5
  times the original renderer's at the same places (0.018). At 16 samples it is 0.049. The deterministic arm stays
  0.023–0.038 at every width.
- Boundary-induced false detections are therefore a real risk. The preregistered boundary-artifact audit and verdict
  safety rule cover it.
- A pre-specified sensitivity renderer at 16 samples is reported for description only and never enters the verdict.
- The design diagnostic behind these numbers ran on TRAIN only and is not committed (scratchpad).

## 5. Confirmation

No event-cell validation metric existed before the D0 preregistration commit. The only validation-data operation before
it was the bit-exact reproduction of the original BF0 renders (§2), which computes no D0 outcome.
