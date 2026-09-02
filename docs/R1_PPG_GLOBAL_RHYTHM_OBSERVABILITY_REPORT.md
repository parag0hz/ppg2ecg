# R1 — PPG Global Rhythm Observability Probe — REPORT

**Preregistration:** `docs/R1_PPG_GLOBAL_RHYTHM_OBSERVABILITY_PREREGISTRATION.md` (frozen at `c7481f9`, untouched).
**Type:** observability / diagnostic probe. NO ECG generation, NO flow training, NO PENGUIN modification,
NO C2 training, NO test access, NO method-novelty claim. **No generator block is implemented.**

## 0. Verdicts (frozen gates, decided once)

| question | verdict |
|---|---|
| A. exact event timing (Q1) | **EXACT R-TIMING SIGNAL LIMITED** (§6) |
| B. coarse rhythm (Q2) | **GLOBAL RHYTHM SCAFFOLD SUPPORTED** — all four gates pass (§11) |

Permitted wording (preregistration §18): *whole-window PPG context contains extractable information about
ECG beat rhythm at a coarser temporal scale, even though exact R timing remains uncertain.* Nothing here
says that PPG determines the R peak, that R peaks are visible in PPG, or that PPG→ECG is identifiable.

## 1. Provenance

| item | value |
|---|---|
| start HEAD | `c1ada0637c38bee1c3844f4b6c188c8963e60e1c` |
| preregistration | `c7481f9` (pushed before any probe weight update) |
| implementation | `917e848` (code + 24 tests, no number computed) |
| memory fix before any result | `41a1a07` (§14.1) |
| all artifacts produced at | `41a1a07` |
| submodules | PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`, unchanged |
| A4 checkpoint md5 | `31c042d291052fbb6dc15263ad316be2`, unchanged |
| C2 | still deferred, zero weight updates |
| test subjects `kjd`/`ssx` | never loaded (`assert_no_test_subjects` in every script; `provenance.json`: `test_subjects_loaded: []`) |
| ECG at inference | none (`ecg_input_to_probe: false`, `r_available_at_inference: false`) |

## 2. Population

| role | subjects | windows per site | total |
|---|---|---|---|
| PROBE_TRAIN (10) | fex l38 n31 ngh p5d p9p qm9 trh tz8 w4p | 2,048 | 81,920 |
| PROBE_INTERNAL_DEV (2) | u7y, e61 | 2,048 | 16,384 |
| VALIDATION (2) | an0, k2s | 1,024 | 8,192 (79,111 GT beats) |

Cohorts are SHA256-ranked on metadata only (salt `r1-global-rhythm-observability-v1`), every stratum was
full (no subject × site had fewer windows than the cap). The internal-dev pair is the two smallest hashes of
`r1-internal-dev-v1|{subject}` over the 12 train subjects, exactly as frozen. Each PPG site is a separate
single-site input; sites are never fused.

## 3. Non-learned rhythm audit (preregistration §6) — computed first

Frozen V1 PPG systolic-peak detector and foot proxy, frozen V1 one-to-one forward matcher ([80, 800] ms),
RR compared with the PPG inter-pulse interval only where consecutive beats are both matched.
106,496 windows, 571,053 consecutive matched pairs. `ppi_rr_pairs.csv`, `ppi_rr_summary.csv`.

| population | signal | pairs | RR median AE | RR MAE | Pearson | ≤ 100 ms | ≤ 10 % |
|---|---|---|---|---|---|---|---|
| probe_train (10) | PPG peak intervals | 434,749 | 62.5 ms | 110.5 ms | 0.792 | 0.607 | 0.562 |
| probe_train (10) | PPG foot intervals | 434,749 | 39.1 ms | 90.6 ms | 0.824 | 0.694 | 0.650 |
| an0 + k2s | PPG peak intervals | 45,431 | 70.3 ms | 115.7 ms | 0.471 | 0.589 | 0.544 |
| an0 + k2s | PPG foot intervals | 45,431 | 46.9 ms | 92.0 ms | 0.517 | 0.682 | 0.632 |

Validation by site (foot intervals): head 23.4 ms median AE / 0.778 within 100 ms; ankle 54.7 / 0.652;
wrist 46.9 / 0.649; sternum 70.3 / 0.624. Worst stratum: k2s wrist (125.0 ms, 0.438).

Reading: a purely DSP inter-pulse interval already recovers RR to about 47 ms median error on the
validation subjects, with 63 % of pairs inside 10 %. This is the non-learned floor that the learned probe
has to clear. It measures rhythm-interval observability, not absolute R phase; the V1 absolute-delay result
(coverage 0.218 at 50 ms, median AE 172 ms) stays separate.

## 4. Probes, training, threshold freeze

| model | trainable params | receptive field | epochs (best) | steps | wall | peak VRAM | best internal-dev BCE |
|---|---|---|---|---|---|---|---|
| Global-TCN (dilations 1…128) | 328,897 | 2,041 samples (≥ 1,024) | 18 (12) | 11,520 | 145.7 s | 1,255 MiB | 0.4330 |
| Local-TCN (all dilations 1) | 328,897 | 65 samples = 507.8 ms (≤ 512 ms) | 19 (13) | 12,160 | 154.4 s | 1,255 MiB | 0.4928 |

Same seed 42 → bit-identical initial weights (asserted by test), same example order, AdamW 1e-3 / 1e-4,
batch 128, BCE-with-logits on the σ = 100 ms soft field, patience 5 on INTERNAL_DEV only. Step-100 runtime
projection: 0.07 h per probe, 0.14 h for both, far inside the 6 GPU-hour stop rule. Total probe training
≈ 5 GPU-minutes.

**Threshold freeze (internal-dev only, before any an0/k2s window was loaded):** grid 0.05…0.95, F1 at
±150 ms, NMS refractory 32 samples.

| model | threshold | internal-dev F1@150 |
|---|---|---|
| Global-TCN | 0.35 | 0.8411 |
| Local-TCN | 0.35 | 0.7559 |

`threshold_selection.json`. The evaluation script writes this file before the validation arrays exist; a
static test pins that ordering in the source.

## 5. Exact timing (Q1) — an0/k2s, ±50 ms, one-to-one

| model | F1@50 | precision | recall | matched coverage | missing | spurious | beats ratio | beats-ratio dev | median matched AE |
|---|---|---|---|---|---|---|---|---|---|
| Global-TCN | **0.620** | 0.609 | 0.635 | 0.621 | 29,977 / 79,111 (37.9 %) | 38,291 / 87,425 (43.8 %) | 1.105 | 0.117 | 23.4 ms |
| Local-TCN | 0.465 | 0.440 | 0.496 | 0.484 | 40,849 (51.6 %) | 53,933 / 92,195 (58.5 %) | 1.165 | 0.190 | 23.4 ms |
| Global-TCN on WINDOW-SHUFFLE | 0.134 | | | | | | | 0.218 | |
| Global-TCN on CIRCULAR-SHIFT | 0.134 | | | | | | | 0.251 | |
| V1 fixed-delay prior (site-specific) | coverage 0.218 | | | | | | | | 172 ms |

F1 is the equal-subject macro (an0 0.610, k2s 0.630). The median matched error is conditional on a match
inside ±50 ms and is therefore bounded by construction; at ±250 ms, where 96 % of beats are matched, the
median matched error is 31.2 ms and the mean 53.3 ms. Per window, Global F1@50 has median 0.70, is ≥ 0.8
in 43.9 % of windows and is exactly 0 in 5.4 %.

## 6. Verdict A — EXACT R-TIMING SIGNAL LIMITED

Argued as the preregistration requires, not from one threshold:

- The 50 ms signal is real. Global-TCN's 0.620 is 4.6× the same model's own input-independent floor
  (0.134 under both controls, which is the count-matched chance level for this beat count) and 2.8× the V1
  fixed-delay prior's coverage (0.218). It is also above every generator arm's event F1 in the S1.3 joint
  table (iMF-4 0.437, OT-CFM-50 0.483, MSE 0.487), with the caveat that those arms were scored on the
  X4-0 window selection of the same two subjects, not on this cohort, and were not trained on an event
  objective.
- It is not strong. Almost four beats in ten (37.9 %) have no predicted event within 50 ms, and 43.8 % of
  predicted events are not within 50 ms of any beat. Most of that residual is jitter rather than absence:
  F1 rises to 0.780 at 100 ms and 0.858 at 150 ms, i.e. the probe usually knows *that* a beat is there and
  is off by 50–150 ms. The site spread is wide (head 0.760, sternum 0.720, wrist 0.548, ankle 0.452).
- Over-detection is systematic: beats ratio 1.105 with 38,291 spurious events at 50 ms, concentrated on
  flat or artefacted PPG (§10).

A learned PPG-only probe therefore locates a majority of beats to within 50 ms but leaves a large,
site-dependent residual. This is a limited signal, better than any fixed-delay reading of PPG, and not a
basis for claiming R-peak accuracy.

## 7. Coarse localization (Q2, part 1)

| model | F1@100 | F1@150 | F1@200 | F1@250 | coverage@200 | beats-ratio dev |
|---|---|---|---|---|---|---|
| Global-TCN | **0.780** | **0.858** | **0.897** | **0.922** | 0.931 | 0.117 |
| Local-TCN | 0.644 | 0.750 | 0.810 | 0.853 | 0.869 | 0.190 |
| Global on WINDOW-SHUFFLE | 0.257 | 0.397 | 0.509 | 0.630 | | 0.218 |
| Global on CIRCULAR-SHIFT | 0.257 | 0.401 | 0.510 | 0.625 | | 0.251 |

Chance is high at coarse tolerances: with ~10 events per 8 s window, a ±200 ms window around each GT beat
covers about half the window, and the two input-independent controls land at exactly 0.51. Global-TCN's
0.897 at 200 ms is +0.39 above that floor; Local-TCN's 0.810 is +0.30. ±200/250 ms are coarse event
localization and are never read as R-peak accuracy. Per window, Global F1@200 has median 1.00 and is ≥ 0.8
in 79.1 % of windows.

## 8. RR rhythm (Q2, part 2)

RR from consecutive predicted events, compared with GT RR only where both consecutive GT beats have a
one-to-one match at 150 ms. The validation cohort has 70,919 consecutive GT pairs.

| model | pairs evaluated | RR median AE | RR MAE | RMSE | rel. median error | Pearson | ≤ 25 ms | ≤ 50 ms | ≤ 100 ms | ≤ 10 % | ≤ 20 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Global-TCN | 58,770 (82.9 %) | **15.6 ms** | **31.4 ms** | 50.7 ms | **2.1 %** | **0.891** | 0.651 | 0.824 | 0.932 | 0.897 | 0.971 |
| Local-TCN | 48,148 (67.9 %) | 31.2 ms | 51.9 ms | 73.1 ms | 4.3 % | 0.813 | 0.425 | 0.630 | 0.835 | 0.773 | 0.942 |
| non-learned PPG foot intervals | 45,431 | 46.9 ms | 92.0 ms | 144.3 ms | 5.9 % | 0.517 | | 0.516 | 0.682 | 0.632 | |
| non-learned PPG peak intervals | 45,431 | 70.3 ms | 115.7 ms | 168.0 ms | 8.3 % | 0.471 | | 0.436 | 0.589 | 0.544 | |

Per subject (Global): an0 median AE 15.6 ms, MAE 32.3 ms, corr 0.912, ≤ 100 ms 0.928; k2s 15.6 / 30.4 /
0.851 / 0.936. RR at 128 Hz is quantised to 7.8 ms, so the 15.6 ms median is two samples.

Denominators differ and are stated rather than harmonised: the probe rows count pairs that survive the
150 ms one-to-one match (survivorship on matched beats), the non-learned rows count pairs that survive the
V1 forward matcher. Pair counts under the two input controls were not recorded by the evaluation script
and are not reported.

## 9. Global vs Local — paired, subject-stratified bootstrap

8,192 paired windows, equal an0/k2s weight, 2,000 replicates, `default_rng(20260902)`, positive = Global
better. `paired_bootstrap.csv`.

| metric | Global − Local (oriented) | 95 % CI | verdict |
|---|---|---|---|
| **S1 F1@150** | +0.108 | [+0.105, +0.112] | improves |
| **S2 F1@200** | +0.087 | [+0.084, +0.090] | improves |
| **S3 RR MAE** | −22.0 ms | [−22.8, −21.1] | improves |
| **S4 beats-ratio deviation** | −0.073 | [−0.078, −0.069] | improves |
| F1@50 (reported, not decisive) | +0.155 | [+0.150, +0.161] | improves |
| F1@100 (reported, not decisive) | +0.137 | [+0.132, +0.141] | improves |
| F1@250 (reported, not decisive) | +0.069 | [+0.067, +0.071] | improves |

All four primary contrasts exclude zero with the same architecture, parameter count, init, data order,
budget and threshold rule; the only difference is the dilation schedule (whole-window vs ≤ 508 ms context).
The gap is largest at the tightest tolerance and shrinks monotonically with tolerance.

## 10. Input-dependence controls — no retraining

| input to the trained Global-TCN | F1@50 | F1@200 | RR MAE | RR median AE | beats-ratio dev |
|---|---|---|---|---|---|
| TRUE | 0.620 | 0.897 | 31.4 ms | 15.6 ms | 0.117 |
| WINDOW-SHUFFLE (derangement within subject × site) | 0.134 | 0.509 | 82.4 ms | 70.3 ms | 0.218 |
| CIRCULAR-SHIFT (uniform 1–4 s per window) | 0.134 | 0.510 | 62.4 ms | 39.1 ms | 0.251 |

TRUE ≫ SHUFFLE on both F1@200 and RR MAE: the probe uses window-specific information, not a subject or
site prior. Under CIRCULAR-SHIFT the absolute-event F1 collapses to the same floor as SHUFFLE (the phase is
destroyed), while RR error stays intermediate (62 ms against 82 ms; median 39 ms against 70 ms): the RR
sequence of a rotated window is preserved except at the seam, so part of the rhythm content survives a
phase destruction. This is the phase-versus-rhythm separation the preregistration named, with the caveat
that the RR rows are conditioned on the few events that still match by chance, and their pair counts were
not recorded.

## 11. Verdict B — GLOBAL RHYTHM SCAFFOLD SUPPORTED

| gate (frozen, §17) | requirement | observed | pass |
|---|---|---|---|
| 1 | Global beats Local on ≥ 2 of {F1@150, F1@200, RR MAE, beats dev} with CI excluding 0 | 4 of 4 | yes |
| 2 | TRUE clearly beats WINDOW-SHUFFLE on both F1@200 and RR MAE | 0.897 vs 0.509; 31.4 vs 82.4 ms | yes |
| 3 | validation RR median AE < 100 ms OR relative RR median error < 10 % | 15.6 ms; 2.1 % | yes |
| 4 | mean beats-ratio deviation < 0.20 | 0.117 | yes |

`decision.json`. Gates were not changed after results.

## 12. Site-wise (exploratory, Global-TCN)

| site | F1@50 | F1@100 | F1@150 | F1@200 | F1@250 | RR MAE | RR median AE | beats-ratio dev |
|---|---|---|---|---|---|---|---|---|
| sternum | 0.720 | 0.834 | 0.886 | 0.912 | 0.931 | 29.3 ms | 15.6 ms | 0.090 |
| head | 0.760 | 0.882 | 0.923 | 0.941 | 0.953 | 23.8 ms | 15.6 ms | 0.072 |
| wrist | 0.548 | 0.713 | 0.805 | 0.853 | 0.886 | 34.9 ms | 15.6 ms | 0.177 |
| ankle | 0.452 | 0.693 | 0.819 | 0.884 | 0.919 | 39.1 ms | 23.4 ms | 0.127 |

Head and sternum are best on every column; wrist and ankle lose most at the tight tolerances and recover
at 200–250 ms (ankle's F1 gap to head is 0.31 at 50 ms and 0.06 at 200 ms). The non-learned foot-interval
audit had the same ordering on head versus the rest but placed sternum last, which the learned probe does
not. No causal information-limitation inference is drawn from site differences.

## 13. Main visual observations — systematic only

Visual atlas: 8 windows per validation subject × site, salt `r1-visual-v1`, fixed before predictions, drawn
whether or not they fall in the metric cohort (membership annotated; 14 of 64 do). `visual_atlas/{sub}_{site}.png`
(figures A–C: PPG, GT R, GT soft field, Global and Local fields and events) and
`visual_atlas/controls_{sub}_{site}.png` (figure D: the same windows under TRUE / WINDOW-SHUFFLE /
CIRCULAR-SHIFT). Figures E–H are in `figures/`.

1. **On periodic PPG the Global field is a sharp train that sits on the GT soft field.** Across the clean
   panels of every site, predicted events fall within a sample or two of the GT R lines and the beat count
   equals the GT count. The field maximum consistently sits at the PPG trough / onset of the upstroke, i.e.
   the probe has learned that R precedes pulse arrival; it is not marking PPG peaks.
2. **On flat, low-amplitude or artefacted PPG the field collapses to a low ripple and NMS over-detects.**
   Panels such as an0 head w13 (GT 10, Global 17), an0 ankle w2911 (16 → 19), k2s wrist w53 / w2162 /
   w3556 / w4179 show the same failure: no clear maxima, many threshold crossings. This is where the 1.105
   beats ratio and the 38,291 spurious events at 50 ms come from, and it is the same failure mode in both
   probes.
3. **The Local field is broader and has sustained plateaus between beats**, with secondary bumps that
   produce its extra events; the Global field returns to zero between beats. On low-amplitude but still
   periodic PPG (e.g. an0 ankle w3972, w4887, w5151) the Global field remains periodic and aligned while the
   Local field degrades, which is the qualitative form of the Global − Local gap.
4. **Controls behave as the numbers say.** Under WINDOW-SHUFFLE the Global field is a clean train that
   follows the *input* window's pulses, so its events fall at the wrong beats of the original window; under
   CIRCULAR-SHIFT the field is periodic away from the seam and carries a visibly noisy segment where the
   rotated window is discontinuous, with events displaced by the shift relative to the original GT.

No panel was selected or removed.

## 14. Deviations, corrections and disclosures

1. **First launch OOM-killed before any weight update (kernel log 16:17:05, pid 1738402, 61 GB anon
   RSS).** Cause: `np.load(npz)[key]` re-reads the whole 91 MB array on every access and a slice of it is a
   view holding that base, so a per-window comprehension pinned one full copy per window. The non-learned
   audit was stopped at 2 min with only `subject_split.json` written and no number computed. All three
   scripts now bind `Xs, Ys, WIs` once per subject (`41a1a07`); labels, cohorts and every number are
   unchanged by construction (e61: 8,192 windows labelled in 2.3 s at 0.9 GB). Every artifact was produced
   after this fix, at `41a1a07`.
2. **Visual atlas script added after results.** `r1_evaluate.py` only drew atlas windows that happened
   to fall inside the metric cohort (14 of 64), leaving the other panels empty. `scripts/r1_visual_atlas.py`
   runs the frozen probes on exactly the frozen 8-per-stratum visual windows and draws figures A–D. It
   computes no metric; thresholds are read from `threshold_selection.json`.
3. **`model_manifest.json` parameter counts corrected from the checkpoints.** The evaluation script
   counted trainable parameters after freezing the loaded model and wrote 0; the file now carries the
   328,897 recorded at training, with a note. `r1_evaluate.py` is patched to read the recorded count. No
   other field changed.
4. **Optional §16 site-aware variant was not run.** The primary verdict does not depend on it, and adding
   it would require a second evaluation pass over the frozen validation cohort. It can be run later under
   the same frozen protocol without touching any primary number.
5. **RR denominators** differ between probe rows and non-learned rows (§8); pair counts under the two
   input controls were not recorded.
6. **The generator-arm comparison in §6** uses S1.3 numbers on the X4-0 window selection of the same two
   subjects; it is context, not a like-for-like comparison.
7. **One training seed only (42), by design of the frozen protocol.** No seed sweep.

## 15. Interpretation — three things kept apart

- **Exact R phase (±50 ms): limited.** A compact PPG-only probe locates 62 % of beats to within 50 ms,
  far above chance (0.13) and a fixed delay (0.22), but leaves 38 % missing and 44 % of its events
  spurious at that scale, with strong site dependence.
- **Coarse event location (±150–250 ms): supported.** 89–96 % of beats are covered at 150–250 ms with
  beat counts within 12 % on average; the residual at 50 ms is mostly 50–150 ms jitter, not absence.
- **RR rhythm: supported and informative.** Median RR error 15.6 ms (2 samples), 93 % of intervals
  within 100 ms, correlation 0.89 with GT RR; three times better than the non-learned inter-pulse interval
  on the same subjects and clearly better than the receptive-field-matched Local control.

Whole-window context is what buys the rhythm: with identical capacity and training, restricting the
receptive field to 508 ms costs 0.09–0.16 F1 and 22 ms of RR MAE.

## 16. What this does NOT prove

- No information-theoretic impossibility claim: a limited probe result does not show that exact R timing
  is absent from PPG.
- No test evidence: `kjd`/`ssx` were never loaded.
- No generator improvement: no generator was trained, modified or evaluated in R1.
- No novelty claim: the probe is a plain dilated TCN used as an instrument.
- One training seed only.
- The controls are inference-time input manipulations of one trained model, not retrained baselines.

## 17. Recommendation only (preregistration §22) — NOT IMPLEMENTED

The frozen condition *exact timing limited + coarse rhythm supported* is met. Recommendation: a
**Global Rhythm Conditioning / Rhythm Scaffold branch** — PPG → global rhythm encoder → soft event /
rhythm tokens → ECG latent conditioning — whose job is beat count, broad event placement and RR structure,
and which must not be claimed to determine QRS morphology. The R1 evidence that supports scoping it this
way: the Global probe's own event field is sharp where PPG is periodic and collapses where PPG is flat, so
any scaffold must carry its own confidence and the generator must not be forced to place a beat on a
collapsed field. Whether such a branch moves the joint event / waveform Pareto front is an open question
for a future preregistered stage; nothing in R1 measures it.

## 18. Artifacts

`artifacts/r1_global_rhythm/`: `provenance.json`, `provenance_audit.json`, `cohort_manifest.csv`,
`subject_split.json`, `ppi_rr_pairs.csv`, `ppi_rr_summary.csv`, `model_manifest.json`,
`training_log_global.csv`, `training_log_local.csv`, `threshold_selection.json`, `event_metrics.csv`,
`event_metrics_per_window.csv`, `rr_metrics.csv`, `rr_pairs_global.csv`, `paired_bootstrap.csv`,
`input_control_metrics.csv`, `site_metrics.csv`, `decision.json`, `figures/` (E `rr_scatter_global.png`,
F `f1_vs_tolerance.png`, G `site_coarse_f1.png`, H `subject_rr_error.png`), `visual_atlas/` (A–C
`{sub}_{site}.png`, D `controls_{sub}_{site}.png`). Checkpoints `outputs/r1_global_tcn_seed42/`,
`outputs/r1_local_tcn_seed42/` (1.3 MB each). Nothing pre-existing was overwritten; artifacts and
checkpoints stay out of git. Full test suite: 263 passed.

**R1 ends here. STOP.**
