# X4-0 — iMeanFlow Event Reliability Diagnostic

Pre-registration: `docs/X4_0_EVENT_RELIABILITY_PREREGISTRATION.md` (commit `14a248e`, **pushed before any X4-0 real-data metric**).
Pre-preregistration disclosure: `docs/X4_0_PREPREREG_VISUAL_AUDIT.md`. Artefacts: `artifacts/x4_0_event_reliability/`.
**NO TRAINING.** Frozen-checkpoint inference and analysis only; no checkpoint created or modified, no historical artefact
overwritten. **WildPPG test subjects `kjd`/`ssx` were never loaded.**

## Executive verdict

**MIXED, dominated by FEW-STEP SATURATION + PERSISTENT SOURCE-SENSITIVE EVENT ORGANIZATION (CASE A + CASE C).**

Every quality axis of the frozen iMeanFlow — morphology, oracle beat correlation, event F1, spurious rate, RR MAE — reaches its
own high-NFE behaviour by **NFE 4–8**; the pre-registered FEW-STEP SATURATION flag fires at both NFE 8 and NFE 16. But it saturates
at an event-reliability level that is **materially below the OT-CFM-50 contextual reference** (F1 0.427 vs 0.480, oracle-absent 0.344
vs 0.250), and additional evaluations do not close that gap — several event metrics are in fact *non-monotone*, peaking at NFE 4 and
drifting slightly worse by NFE 50.

The reason is not sharpness and not interval length. With the PPG held fixed, **changing only the Gaussian source changes event
identity itself**: the median predicted-to-predicted event F1 between two sources is **0.30**, the per-window predicted beat count
varies with SD **1.2–1.8 beats**, and GT beats that are detected at all are detected at a timing that varies by **≈ 50 ms SD** across
sources. All four pre-registered materiality criteria fire at NFE = 1, and **none improves by 30 % at NFE 8 or 16** — this is
**PERSISTENT SOURCE SENSITIVITY**, which the revised §9.1 treats as the stronger result rather than a flag failure. The fixed-NFE
interval-stress flag does **not** fire (subject to the asymmetric-interpretability limitation below).

## Why the previous sharpness hypothesis was rejected

X0's committed measurements show iMF-1 at or above ground truth on the structural axes (whole-window QRS-energy retention 1.178,
max-slope ratio 1.083, amplitude ratio median 0.997), i.e. **sharper than the 50-NFE reference**. X4-0 reproduces the picture on
development validation: iMF global slope ratio 0.96–1.19 and QRS-energy ratio 0.55–0.61 across the whole NFE range, with HF ratio
0.222–0.245 against a GT HF of 0.193. The deficit is not sharpness. The earlier "57 % of QRS energy is in the wrong location"
statement is **retracted**: it divided a pooled-variance ratio by a median of per-beat ratios, which are not commensurable.

## Pre-prereg visual audit and data firewall

Four validation windows had already been viewed before this protocol was frozen — resolved to (`an0`, 9066), (`an0`, 18138),
(`k2s`, 5852), (`k2s`, 16436) — and are **excluded by construction** from every X4-0 subset (verified: no leak in any subset file).
`an0`/`k2s` are therefore **development validation**, never pristine confirmatory validation. Subsets were chosen by SHA256 ranking
over (subject, original window index) with the frozen salts: NFE 2×1024 = 2048 windows (19,834 GT beats), source 2×256 = 512,
schedule 2×512 = 1024. `provenance.json` records `test_subjects_loaded: []`.

## Frozen models

| Role | Checkpoint | Round |
|---|---|---:|
| Primary | `outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt` | 45 |
| Contextual yardstick (mandatory) | `outputs/a4_otcfm_wildppg_seed42/checkpoint_best.pt`, Heun-25 = **50 NFE** (asserted) | 189 |

OT-CFM-50 is a **contextual reference, not an oracle**; no verdict is gated on iMF-vs-OT50 F1 alone.

## Training interval exposure (X4-0A, no waveform data)

2,000,000 draws from the frozen `sample_tr` configuration, seed 20260830:

| quantity | value |
|---|---:|
| mass at exactly h = 0 (the forced r = t half) | **0.5000** |
| median of h > 0 | 0.2011 |
| q90 / q99 / q99.9 | 0.3798 / 0.6384 / 0.7721 |
| **maximum observed h** | **0.9297** |

| inference | h | P(train h ≥ h) |
|---|---:|---:|
| **NFE 1** | **1.0** | **0** |
| NFE 2 | 0.5 | 0.0422 |
| NFE 4 | 0.25 | 0.2009 |
| NFE 8 | 0.125 | 0.3372 |
| NFE 16 | 0.0625 | 0.4165 |

**Exact h = 1 has zero training probability and is an extreme boundary query; near-1 intervals carry vanishingly small training
mass** (P(h ≥ 0.9) = 8 × 10⁻⁶, P(h ≥ 0.95) = 0 in 2 M draws). This is *training-interval exposure only* — no performance
conclusion follows from it, and it is not an impossibility result. Notably the model nonetheless produces usable output at h = 1.

## NFE frontier — morphology and event reliability (2048 windows × 4 source seeds)

| Model | NFE | Morph | F1 | Prec | Recall | Spurious | Missing | RR MAE | Oracle absent | Oracle corr | QRS-E | Slope | HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iMF | 1 | 0.665 | 0.410 | 0.441 | 0.432 | 0.569 | 0.568 | 25.2 | 0.385 | 0.589 | 0.590 | 1.189 | 0.222 |
| iMF | 2 | 0.726 | 0.420 | — | — | 0.550 | — | 23.8 | 0.352 | 0.640 | 0.553 | 1.009 | 0.222 |
| iMF | **4** | 0.770 | **0.431** | — | — | 0.518 | — | 22.9 | **0.335** | **0.664** | 0.568 | 0.959 | 0.229 |
| iMF | **8** | **0.777** | 0.427 | — | — | 0.493 | — | 22.6 | 0.344 | 0.659 | 0.585 | 0.980 | 0.236 |
| iMF | 16 | 0.774 | 0.424 | — | — | 0.480 | — | 22.5 | 0.352 | 0.651 | 0.595 | 0.995 | 0.242 |
| iMF | 25 | 0.773 | 0.422 | — | — | 0.475 | — | 22.4 | 0.356 | 0.647 | 0.601 | 1.001 | 0.244 |
| iMF | 50 | 0.771 | 0.420 | — | — | 0.472 | — | 22.3 | 0.360 | 0.643 | 0.607 | 1.008 | 0.245 |
| **OT-CFM** | **50** | **0.818** | **0.480** | — | — | 0.522 | — | 21.3 | **0.250** | **0.716** | 0.632 | 1.021 | 0.267 |

Full per-metric bootstrap intervals, per-subject values and per-seed values are in `nfe_metrics.csv`,
`nfe_metrics_by_subject.csv` and `nfe_metrics_by_seed.csv`. GT HF ratio on these windows is 0.193.

**FEW-STEP SATURATION: FIRES at NFE 8 and at NFE 16.** Against the internal iMF-50 reference, NFE 8 satisfies all five frozen
conditions (|Δmorph| 0.006 ≤ 0.03; |Δoracle corr| 0.016 ≤ 0.03; |ΔF1| 0.007 ≤ 0.03; spurious 0.493 ≤ 0.472 + 0.05;
RR 22.6 ≤ 22.3 + 3), and NFE 16 likewise. The direction is consistent on both subjects (`an0` morph 0.627 → 0.732 → 0.726,
`k2s` 0.704 → 0.822 → 0.817 at NFE 1 → 8 → 50).

**Morphology and event reliability do not improve at the same rate.** From NFE 1 → 8 morphology gains **+0.112** (0.665 → 0.777,
a 73.3 % closure of the gap to OT-CFM-50) while F1 gains **+0.017** (0.410 → 0.427, an 24.5 % closure) and then *loses* ground by NFE 50.
Oracle-absent and oracle beat correlation are best at NFE 4 and drift worse afterwards. Spurious rate is the only event metric that
improves monotonically (0.569 → 0.472) and it remains high at every NFE.

## Same PPG × 32 Gaussian sources (X4-0C, 512 windows)

| NFE | seed-pair event F1 | beat-count SD | GT detection probability | conditional timing SD (≥16/32) | F1 SD |
|---:|---:|---:|---:|---:|---:|
| 1 | **0.300** | **1.226** | 0.750 | **57.1 ms** | **0.157** |
| 4 | 0.316 | 1.446 | 0.781 | 52.0 ms | 0.169 |
| 8 | 0.304 | 1.609 | 0.750 | 51.0 ms | 0.172 |
| 16 | 0.286 | 1.780 | 0.750 | 49.7 ms | 0.177 |

**SOURCE-SENSITIVE EVENT ORGANIZATION: MATERIAL — all four criteria fire at NFE 1** (seed-pair F1 0.300 < 0.80; beat-count SD
1.226 ≥ 0.75; conditional timing SD 57.1 ms ≥ 15 ms; F1 SD 0.157 ≥ 0.05).

**NFE response: PERSISTENT.** The largest improvement of any material indicator at NFE 8 or 16 is the conditional timing SD, which
falls only 57.1 → 49.7 ms (**13 %**, below the 30 % bar); seed-pair F1 and F1 SD do not improve at all, and beat-count SD gets
**worse** with more evaluations (1.23 → 1.78). Per the revised §9.1 this is recorded as **PERSISTENT SOURCE SENSITIVITY**, the
stronger of the two outcomes — under the original single-flag rule this pattern would have been scored as a flag *failure*.

Two sources given the identical PPG agree on only ~30 % of their detected events, disagree by ~1.2–1.8 beats in count, and place the
GT beats they do find with ~50 ms of scatter. This is measured on *event identity*, not on waveform realization, and is therefore
new information: the previously known A4 seed-to-seed waveform correlation of 0.025–0.040 established only that the *waveform* is
source-sensitive.

### Figure 5 — the raster, and a window-dependence caveat

`figures/fig5_source_peak_raster.png` (produced by `scripts/figure_x4_0_peak_raster.py`; windows chosen by the **same hash rank** as
the source subset, smallest hashes first, never by appearance: (`an0`, 6432), (`an0`, 13972), (`k2s`, 1067), (`k2s`, 7627)) plots one
tick per predicted R-peak for each of the 32 sources against the GT beats.

It makes the aggregate numbers concrete and adds a caveat the medians hide: **source sensitivity is strongly window-dependent.** On a
clean window (`k2s` 1067) the 32 sources form tight vertical columns on the GT beats (seed-pair F1 0.59 at NFE 1, 0.67 at NFE 8); on
noisy windows (`an0` 6432 / 13972) the column structure is absent (0.19 / 0.14) and the ticks scatter across the whole window. The
raster also shows that the predicted beat count sits systematically **below** the GT count and that its spread across sources
**grows** with NFE (e.g. 10.0 ± 1.95 → 9.0 ± 2.78 on `an0` 6432), which is the per-window view of the pooled beat-count SD rising
1.23 → 1.78 from NFE 1 to 16.

## Source perturbation vs a strong PPG-shuffle anchor (X4-0C2)

Reusing the project's `eval_a2.py` derangement; frozen source-pair mapping 0 → 1 and 2 → 3.

| NFE | perturbation | event disagreement (1 − F1) | beat-count difference | timing disagreement | waveform disagreement (1 − PCC) |
|---:|---|---:|---:|---:|---:|
| 1 | Gaussian source | 0.633 | 1.25 | 23.5 ms | 0.919 |
| 1 | PPG shuffle | 0.693 | 1.20 | 6.7 ms | 0.431 |
| 8 | Gaussian source | 0.615 | 1.48 | 22.5 ms | 0.907 |
| 8 | PPG shuffle | 0.726 | 1.27 | 6.7 ms | 0.555 |

**Descriptive context only.** The derangement is a **strong condition-perturbation anchor** which may substitute a PPG with a
different heart rate and rhythm; it is **not** on a calibrated common causal scale with a Gaussian-source change, and no
"source/condition importance ratio" is computed. What can be said descriptively is that changing the Gaussian source produces event
disagreement of the same order as this strong PPG perturbation (0.63 vs 0.69 at NFE 1), while the two differ sharply in the *other*
columns — the source change alters the waveform far more (0.92 vs 0.43 disagreement) and the matched-beat timing far more
(23.5 ms vs 6.7 ms). **This is not evidence that the model ignores PPG**, and it is not claimed.

## Fixed-NFE interval stress (X4-0D, 1024 windows × 4 seeds)

| Schedule | max h | F1 | Spurious | Oracle absent | Oracle corr | Morph | RR MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| U4 | 0.25 | 0.4327 | 0.5169 | 0.3353 | 0.6632 | 0.7661 | 23.1 |
| LN4 | 0.70 | 0.4132 | 0.4813 | 0.3824 | 0.6312 | 0.7469 | 23.5 |
| LD4 | 0.70 | 0.4138 | 0.5620 | 0.3582 | 0.6234 | 0.7040 | 24.4 |
| U8 | 0.125 | 0.4292 | 0.4913 | 0.3433 | 0.6581 | 0.7696 | 22.8 |
| LN8 | 0.50 | 0.4223 | 0.4718 | 0.3632 | 0.6433 | 0.7609 | 22.8 |
| LD8 | 0.50 | 0.4252 | 0.5485 | 0.3368 | 0.6483 | 0.7377 | 23.9 |

Paired deltas versus uniform: LN4 ΔF1 −0.020, Δoracle-absent +0.047, Δoracle corr −0.032; LD4 ΔF1 −0.019, Δspurious +0.045,
Δoracle corr −0.040, Δmorph −0.062; LN8 and LD8 are smaller still. **LARGE-INTERVAL STRESS: does NOT fire** — no comparison reaches
ΔF1 ≤ −0.05, Δoracle corr ≤ −0.05, Δspurious ≥ +0.10 or Δoracle-absent ≥ +0.10.

**Noise-end vs data-end.** The two large-step placements degrade *differently* rather than equally: the noise-end schedules (LN)
raise oracle-absent (+0.047 at NFE 4) while *lowering* spurious, whereas the data-end schedules (LD) raise spurious (+0.045 / +0.057)
and cost the most morphology (−0.062 / −0.032). This is a consistent, interpretable pattern — a large final jump leaves extra
un-refined deflections; a large first jump loses beats — but neither reaches the pre-registered material threshold.

### Required limitation (stated in the pre-registration before results)

These schedules **do not reproduce the NFE = 1, h = 1 query**. Maximum tested stress was h = 0.70 (NFE 4) and h = 0.50 (NFE 8),
whereas one-step inference uses h = 1 exactly. A positive stress result would have been informative evidence that large-h queries
degrade the frozen iMF; **this null result is weak evidence against H3 and does not establish that h = 1 is harmless.** The
direction of every measured delta is nonetheless unfavourable to large steps, which is consistent with — but does not demonstrate —
a mild large-interval cost.

## Event-matching tolerance calibration (X4-0E)

GT peak trains perturbed and re-scored against GT with the same frozen 50 ms one-to-one matcher (peak **count preserved**, so this
isolates timing only):

| jitter SD | 0 | 5 ms | 10 ms | 20 ms | 30 ms | 40 ms | 50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 | 1.000 | 1.000 | 1.000 | 0.989 | 0.909 | 0.790 | 0.686 |

Fixed shifts of ±10…50 ms all give F1 = 1.000 (a uniform shift of 50 ms sits exactly at the 6.4-sample tolerance).

**This is a timing-only degradation calibration, not an upper bound and not "the maximum achievable F1".** The permitted inference:
timing jitter of 20–30 ms still yields F1 ≈ 0.91–0.99, whereas the observed iMF F1 is 0.41–0.43 at every NFE. **The observed F1
deficit therefore cannot be explained by timing error alone; missing and spurious events must contribute materially** — consistent
with the measured spurious rate of 0.47–0.57 and beat-count SD of 1.2–1.8 across sources.

## Latency

Batch 64, 20 warm-ups, 100 timed repeats, CUDA-synchronised, data loading excluded.

| Model | NFE | median (ms) | p10 | p90 | samples/s |
|---|---:|---:|---:|---:|---:|
| iMF | 1 | 79.7 | 71.0 | 80.3 | 803 |
| iMF | 4 | 310.4 | 309.3 | 320.5 | 206 |
| iMF | 8 | 628.8 | 620.1 | 636.0 | 102 |
| iMF | 16 | 1252.5 | 1240.7 | 1263.7 | 51 |
| iMF | 25 | 1961.7 | 1951.2 | 1971.6 | 33 |
| iMF | 50 | 3923.4 | 3906.7 | 3933.8 | 16 |
| OT-CFM | 50 | 3922.5 | 3910.4 | 3933.4 | 16 |

Latency is essentially linear in NFE, and iMF and OT-CFM cost the same per evaluation (3922 vs 3923 ms at 50 NFE). **NFE 8 — the
point at which iMF reaches its own high-NFE behaviour — costs 629 ms, one sixth of the 50-NFE budget, and 102 samples/s.**

## Mechanism classification

**MIXED — CASE A (few-step saturation) + CASE C (persistent source-sensitive event organization).**

Additional integration buys most of the waveform-quality gap by NFE 4–8 and then stops; the FEW-STEP SATURATION flag fires at 8 and
16 against the model's own 50-NFE behaviour. Event reliability does not improve at the same rate: F1 moves 0.410 → 0.431 → 0.420
across the whole grid and oracle-absent is best at NFE 4, so integration is not the binding constraint on event identity. With the
PPG fixed, two Gaussian sources agree on only ~30 % of events, differ by 1.2–1.8 beats and scatter matched-beat timing by ~50 ms;
all four materiality criteria fire at NFE 1 and none is relieved by 30 % at NFE 8/16, so the source sensitivity is persistent rather
than NFE-responsive. Fixed-NFE interval stress does not reach the material threshold, though every delta points the same
(unfavourable) way and the test cannot speak to h = 1. The matching calibration rules timing error out as the sole explanation of
the F1 deficit. CASE D is excluded because source sensitivity is neither weak nor relieved by higher NFE. Both mechanisms are
recorded; no single dominant story is forced.

## What is supported

- iMF-1 already produces sharp ECG-like waveforms; structural sharpness is not the deficit.
- Waveform morphology approaches the frozen iMF's own high-NFE behaviour by NFE 4–8 (73.3 % of the gap to OT-CFM-50 closed by NFE 8).
- **Event reliability does not saturate together with morphology**: F1 gains 0.017 over the whole NFE grid and is non-monotone.
- **Event identity is materially and persistently source-sensitive at low NFE**, and additional evaluations do not relieve it.
- Under the frozen 50 ms matcher, timing error alone cannot explain the observed F1; missing/spurious events contribute materially.
- At fixed NFE, large-interval schedules degrade every measured event metric slightly, with a reproducible noise-end (more absent
  beats) versus data-end (more spurious beats, most morphology loss) asymmetry.

## What is NOT supported

- That h mismatch causes one-step failure — the stress test does not reach h = 1 and its result is null at h ≤ 0.70.
- That the Gaussian source is "wrong", or that PPG is ignored: the source and condition perturbations are not on a common causal
  scale, and the PPG-shuffle anchor changes the *waveform* far less than a source change does.
- That OT-CFM-50 is an oracle or a ceiling; it is a contextual yardstick on two development subjects.
- That any particular architecture (STFT, event conditioning) will fix this — X4-0 trains nothing and tests no intervention.
- Population-level inference: two development subjects, already visually inspected, with descriptive paired bootstrap intervals only.

## Recommended single next training experiment

**An event/rhythm-conditioned iMeanFlow.** The decision rules in §15 of the pre-registration are met on the event branch and not on
the others: sharpness is adequate (slope 0.96–1.19, QRS-energy 0.55–0.61, HF 0.222–0.245 vs GT 0.193); the event metrics are the
clear residual deficit and do not respond to integration; source variation changes event identity materially and persistently; and
the large-h lever is not supported by the data in hand. The natural shape is PPG → event/rhythm encoder → predicted soft event map →
iMF conditioning, with GT R-peaks supervising **training only** and inference using PPG alone. The complex-STFT branch is *not*
indicated: its favourable condition requires source-event sensitivity **not** to be the dominant lever, and here it is. A
horizon/large-h intervention is not indicated either, though the asymmetric limitation above means it is not excluded.

**Not started.** No training, no STFT model, no event-aware model, no horizon-aware model, no coupling, no reflow.
