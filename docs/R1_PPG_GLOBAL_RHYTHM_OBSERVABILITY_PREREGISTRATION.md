# R1 — PPG Global Rhythm Observability Probe — PREREGISTRATION

**Status:** frozen at this commit, pushed **before any learned-probe weight update**.
**Type:** observability / diagnostic probe. **NO ECG generation. NO flow training. NO new attention.
NO PENGUIN modification. NO C2 training. NO test access. NO method-novelty claim.**

---

## 0. Standing disclosures

- **This experiment was motivated after the V1 visual inspection** (`docs/V1_ALL_SUBJECT_STEPWISE_VISUALIZATION_REPORT.md`,
  Figma review board). It is **not independent confirmatory evidence**.
- **GT ECG R-peaks are TRAINING LABELS ONLY.** The GT ECG waveform is **never** an input to the probe.
  Validation R-peaks are **evaluation labels only**. **No R information is available at inference.**
- The probe asks whether the information is *extractable from PPG by a compact model*. **A failed probe
  does not prove information-theoretic impossibility** and will not be reported as such.

## 1. Question

> Can the full 8 s PPG waveform provide a **coarse cardiac-rhythm scaffold** even when it cannot provide
> **precise R-peak timing**?

**Q1** — can PPG alone locate R at the 50 ms scale? **Q2** — even if not, does whole-window PPG context
predict ECG beat timing / RR rhythm at a coarser scale? These are answered **separately**.

## 2. Provenance

Start HEAD `c1ada0637c38bee1c3844f4b6c188c8963e60e1c`; submodules PENGUIN `6cd70cd`, iMeanFlow
`bf60cd7`; C2 deferred with zero weight updates; no `outputs/r1_*` exists.

## 3. Population and split

Frozen A4 split. **TEST `kjd`/`ssx` never loaded.** Every PPG site is a **separate single-site input**
(sternum, head, wrist, ankle); **sites are never fused**, matching the information setting of the future
generator.

**Internal train-only model-selection split**, by SHA256 rank of `"r1-internal-dev-v1|{subject}"` over the
12 train subjects, smallest two:

| role | subjects |
|---|---|
| **PROBE_INTERNAL_DEV** (2) | **`u7y`, `e61`** |
| **PROBE_TRAIN** (10) | `fex l38 n31 ngh p5d p9p qm9 trh tz8 w4p` |
| **VALIDATION** (2) | `an0`, `k2s` — untouched until architecture, budget and threshold rule are frozen |

## 4. Cohort — metadata only, frozen before training

SHA256 rank of `"r1-global-rhythm-observability-v1|{subject}|{site}|{original_window_index}"`, smallest
hashes first within each subject × site. **PROBE_TRAIN and INTERNAL_DEV: ≤ 2,048 per subject × site;
VALIDATION: ≤ 1,024 per subject × site.** Strata with fewer windows use all and report the count.
Selection may **not** use R count, HR, PPG or ECG quality, model performance, detector success or
morphology. Written to `artifacts/r1_global_rhythm/cohort_manifest.csv`.

Visual cohort (§21): salt `"r1-visual-v1"`, 8 per validation subject × site, fixed before predictions.

## 5. GT R labels

The **exact frozen detector** `rpeaks.detect_rpeaks` at its existing configuration on the GT ECG window.
No new ECG peak detector. Probe input is **exclusively** the PPG `[1, 1024]` (plus a site ID only in the
optional §18 variant). **No ECG sample value may enter the network** — asserted by test.

## 6. Non-learned rhythm audit — FIRST

Reuse the frozen V1 PPG systolic-peak detector (`s1_audit.dsp_ppg_peaks`, library defaults) and the frozen
V1 foot proxy (`v1_timing.ppg_foot`, unchanged). Pair R-peaks to PPG pulses with the frozen V1 one-to-one
forward matcher (`v1_timing.match_r_to_ppg`, [80, 800] ms, no pulse reused). For consecutive **matched**
beats `j, j+1`: `RR_j = R_{j+1} − R_j`, `PPI_peak_j`, `PPI_foot_j` from the corresponding PPG events.

Per subject × site: MAE, median AE, RMSE, Pearson, Spearman, relative error `|PPI−RR|/RR`, and % within
25 / 50 / 100 / 150 ms and within 10 % / 20 %. **This measures rhythm-interval observability, not absolute
R phase**; the V1 absolute-delay result stays separate. Outputs `ppi_rr_pairs.csv`, `ppi_rr_summary.csv`.

## 7. Primary learned probe — Global-TCN

Input PPG `[B,1,1024]` → per-sample R-event logits `[B,1,1024]`. **≤ 1 M trainable parameters; no
transformer, no pretrained weights, no ECG input, no target-waveform input, no PENGUIN backbone, no
iMeanFlow component, no detected PPG peaks as input.**

Frozen architecture: input `Conv1d(1→64, k=1)`; **8 residual blocks**, each = two `Conv1d(64→64, k=5,
dilation d, "same")` with GELU, plus identity skip; **dilations 1, 2, 4, 8, 16, 32, 64, 128**; output
`Conv1d(64→1, k=1)`. Theoretical receptive field `1 + 2·(5−1)·Σd = 1 + 8·255 = 2041 samples ≥ 1024`,
**asserted by unit test.** Capacity is not increased after seeing results.

## 8. Local-receptive-field control — Local-TCN

**Identical** channels, block count, kernel sizes, parameter count, optimizer, budget, seed and example
order. **The only difference is the dilation schedule: all dilations = 1**, receptive field
`1 + 8·8 = 65 samples = 507.8 ms ≤ 512 ms`, asserted by test. Parameter-count equality is asserted by test.

## 9. Target and loss — frozen

Soft event field `y(t) = max_j exp(−(t − r_j)² / 2σ²)`, **σ = 100 ms = 12.8 samples** at 128 Hz
(conversion unit-tested). No one-sample impulse target. **Loss: binary cross-entropy with logits against
the soft field** (`BCEWithLogitsLoss`), chosen here and not changed. No auxiliary morphology, RR,
peak-count or phase loss.

## 10. Training budget — frozen

Seed **42**, shared by both probes; shape-compatible initialisation shared (same seed, same shapes → same
init); identical example order (one seeded loader). AdamW, lr 1e-3, weight decay 1e-4, batch 128.
**Maximum 30 epochs.** Early stopping on **INTERNAL_DEV soft-field BCE only**, patience 5. Runtime
profiled after 100 steps; **if the projection for both probes exceeds 6 GPU-hours, STOP and report.**
Wall time and peak VRAM recorded.

## 11. Event extraction — frozen, internal-dev only

Sigmoid probability → local maxima → **NMS with refractory 250 ms = 32 samples** → keep peaks above a
threshold. The threshold is chosen **on INTERNAL_DEV only**, from the grid `{0.05, 0.10, …, 0.95}`, by
**maximising F1 at ±150 ms**, separately for Global and Local under the same protocol. **Frozen before any
`an0`/`k2s` result is computed.** Recorded in `threshold_selection.json`.

## 12. Evaluation on `an0`/`k2s`

One-to-one greedy matching (`rpeaks.match_rpeaks`) at **±50 / 100 / 150 / 200 / 250 ms**: precision,
recall, F1, matched coverage, missing, spurious, predicted/GT beat ratio, median and mean matched
absolute timing error. **±200/250 ms are "coarse event localization", never "R-peak accuracy".**

**RR evaluation:** `RR_pred` from consecutive predicted events; compared with GT RR **only where consecutive
GT events both have one-to-one matches** (at 150 ms). RR MAE, median AE, RMSE, correlation, relative error,
% within 25 / 50 / 100 ms and 10 % / 20 %; per-window beat-count deviation.

## 13. Global-context specificity

Paired per-window differences on the frozen validation windows, subject-stratified bootstrap, equal
`an0`/`k2s` weight, **2,000 replicates, `default_rng(20260902)`**, positive = Global better.
**Primary:** S1 F1@150, S2 F1@200, S3 RR MAE, S4 beats-ratio deviation. @50 and @100 reported but not
decisive for coarse rhythm.

## 14. Input-dependence controls — no retraining

Trained Global-TCN on: **A TRUE** · **B WINDOW-SHUFFLE** (deterministic derangement within subject × site,
no fixed points, `default_rng(20260902)`) · **C CIRCULAR-SHIFT** (per-window offset uniform on
[1.0, 4.0] s = [128, 512] samples, same rng). TRUE ≫ SHUFFLE ⇒ the probe uses window-specific information.
Under CIRCULAR-SHIFT, absolute-event F1 dropping while RR stays similar ⇒ phase and rhythm separate.

## 15. Site analysis — exploratory

Global-TCN by site: F1@50/100/150/200/250, RR MAE, beat-count deviation. No causal information-limitation
inference from site differences.

## 16. Optional site-aware variant — SECONDARY

Only after the primary models are frozen: the same Global-TCN plus a 4-way learned site embedding injected
as FiLM (scale+shift on the 64-channel stem), **≤ 1 % added parameters**, same budget, no tuning. It may
not replace the primary no-site result.

## 17. Verdicts — frozen

**A. Exact event timing** — `EXACT R-TIMING SIGNAL STRONG` or `LIMITED`, argued from F1@50, median matched
timing error, missing/spurious, and comparison to the V1 fixed-delay prior (site-specific: F1-like coverage
0.218 at 50 ms, median AE 172 ms). Not defined by one arbitrary threshold; the report must explain.

**B. Coarse rhythm** — `GLOBAL RHYTHM SCAFFOLD SUPPORTED` requires **all four**: (1) Global beats Local on
≥ 2 of {F1@150, F1@200, RR MAE, beats-ratio deviation} with paired CI excluding zero; (2) TRUE clearly beats
WINDOW-SHUFFLE on **both** F1@200 and RR MAE; (3) validation **RR median AE < 100 ms OR relative RR median
error < 10 %**; (4) mean beats-ratio deviation **< 0.20**. If these fail but Global clearly beats Local →
`GLOBAL CONTEXT HELPS BUT RHYTHM SCAFFOLD REMAINS WEAK`. If Global does not clearly outperform Local →
`GLOBAL RHYTHM SCAFFOLD NOT SUPPORTED`. **Gates are not changed after results.**

## 18. Wording

If supported: *"Whole-window PPG context contains extractable information about ECG beat rhythm at a
coarser temporal scale, even though exact R timing remains uncertain."* Never: *"PPG determines the R
peak"*, *"R peaks are visible in PPG"*, *"PPG→ECG is identifiable."* If it fails: *"This compact probe did
not recover a sufficiently reliable coarse ECG rhythm scaffold from PPG under the frozen protocol."* Never:
*"PPG contains no R information."*

## 19. Deliverables and stop rules

`docs/R1_PPG_GLOBAL_RHYTHM_OBSERVABILITY_REPORT.md`; `artifacts/r1_global_rhythm/` with `provenance.json`,
`cohort_manifest.csv`, `subject_split.json`, `ppi_rr_pairs.csv`, `ppi_rr_summary.csv`, `model_manifest.json`,
`training_log_global.csv`, `training_log_local.csv`, `threshold_selection.json`, `event_metrics.csv`,
`rr_metrics.csv`, `paired_bootstrap.csv`, `input_control_metrics.csv`, `site_metrics.csv`, `decision.json`,
`figures/`, `visual_atlas/`. Checkpoints in `outputs/r1_global_tcn_seed42/`, `outputs/r1_local_tcn_seed42/`;
nothing existing is overwritten; checkpoints never enter git.

**R1 ends at its two verdicts and a recommendation. No generator block is implemented.**
