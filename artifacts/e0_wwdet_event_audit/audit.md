# E0 audit (2026-10-01) — before any event-level E0 outcome

## 1. What was read and reused

- **Reports:** C0 (`docs/C0_COHERENTBEAT_FEASIBILITY_REPORT.md`) and C0-A
  (`docs/C0A_COHERENTBEAT_ARCHITECTURE_ABLATION_REPORT.md`).
- **Code:**
  - `scripts/c0_coherentbeat.py`: `load_arch`, `reference_peaks`, `detect_events`, `render_c0`, `cluster_ci`;
  - `scripts/c0a_ablation.py`: `render_ww`, metric block;
  - `src/ppg2ecg/coherentbeat/{model,ablation}.py`;
  - `src/ppg2ecg/evaluation/rpeaks.py` (`match_rpeaks`, greedy one-to-one by |Δt|, ±50 ms);
  - `paper_metrics.rpeak_prf_at` / `beat_level_metrics`;
  - `s1_audit.dsp_ppg_peaks`, the existing project PPG peak detector.
- **Untouched:** no C0 or C0-A file is modified. E0 imports them.

## 2. Frozen checkpoints (`input_hashes.json`)

`detector.pt`, `c0.pt` (C0 `checkpoint_hashes.json`) and `ww.pt` (C0-A `checkpoint_hashes.json`) match their recorded
sha256. The stored C0 events and renders, split manifest and C0-A metric files are hashed too.

## 3. Exact reproduction (`reproduction.json`)

| | ARCH-VAL (433 patients, 28,649 windows) | ARCH-HOLDOUT (434 patients, 28,531 windows) |
|---|---|---|
| events re-detected = stored | yes | yes |
| C0 render = stored (float32) | yes | yes |
| max \|Δ\| vs C0-A stored F1 / precision / recall / FP / FN / RR-MAE / HR-MAE (point and CI) | 0.0 (placed, C0, WW) | 0.0 (placed, C0, WW) |
| raw false R (placed / C0 / WW) | 11,379 / 11,684 / 14,994 | 12,491 / 12,761 / 15,814 |
| raw missed R (placed / C0 / WW) | 29,627 / 29,714 / 29,125 | 28,845 / 28,918 / 28,374 |

## 4. Decisions the specification leaves open (frozen in the preregistration)

- **Matching:** the project's greedy one-to-one rule for all three matchings. The hard guard is deliberately not
  one-to-one (any placed event within 50 ms); "A + B only" is the one-to-one version.
- **ARCH-TRAIN outputs:** the frozen detector's events on ARCH-TRAIN, with C0 and WW-DET rendered by inference. These are
  in-sample for both models.
- **Features 13 / 14:** the RR intervals the candidate would form with the previous / next placed event.
- **Other feature details:**
  - ±80 ms becomes ±10 samples;
  - the detection is refined to the maximum within ±3 samples;
  - feature 2 is the absolute level x(t*);
  - feature 16 is a signed lag.
- **QRS template:** ±10 samples about every ARCH-TRAIN reference R (RD1 cache).
- **Probe:**
  - imputation by TRAIN median plus missingness indicators;
  - `GroupKFold(5)`;
  - C by mean fold AUROC (ties → smaller);
  - threshold = maximum TRAIN-OOF balanced accuracy.
- **Case rule:** operational thresholds of the specification's E0-A / B / C conditions (preregistration §11). The
  program's existing 0.02 margin is reused for materiality, and "most / meaningful" means at least half.
- **Final case:** the ARCH-VAL case, kept only if ARCH-HOLDOUT gives the same case.

## 5. Known before the preregistration (disclosed in it)

The C0-A aggregate counts in the table above. They imply on ARCH-VAL that type C ≥ 502 and type D ≥ 3,615. No type
count, rate, counterfactual, localization, feature or probe result was computed before the preregistration commit.

## 6. Synthetic dry run

Every stage after `reproduce` ran end to end on a synthetic world in a scratch directory. It also checked that
`evaluate` refuses to run without the preregistration manifest. The numbers mean nothing.

It found one bug, fixed before the commit: merging the feature table duplicated the `phase` / `dist_nearest_ms`
columns, which were then overwritten with NaN. A regression test now covers this. The figure panels' labels were
adjusted.
