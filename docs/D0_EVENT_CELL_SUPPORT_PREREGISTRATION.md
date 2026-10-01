# D0 — Event-Cell Support Diagnostic on Frozen BF0 Beat Outputs — PREREGISTRATION (frozen)

**Status: FROZEN (2026-10-01).** This file is committed and pushed with the D0 implementation before any event-cell
validation result is computed, and is never edited afterwards (dated amendments only). Hashes:
`artifacts/d0_event_cell_support/prereg_manifest.json`. Audit: `artifacts/d0_event_cell_support/audit.md`.

## 0. What D0 is, and is not

- **BF0 is closed and unchanged:** prereg `504395d`, result `d380e3b`, Case A (G1 FAIL), BF1 NO-GO. D0 cannot change
  this verdict, rescue BF0, or turn BF0 into a pass.
- **Post-hoc.** D0 is a post-hoc mechanistic intervention on frozen outputs from the same validation population. The
  hypothesis was generated from these validation data, after BF0. Holding everything else fixed makes D0 stronger than an
  observational correlation, but it is **not an independent confirmatory experiment, not causal proof, and not
  replication**. The intervention is frozen here, before its outcome is computed.
- **Nothing is trained.** No weights, morphology outputs, positions, seeds, detector or metric implementation change.
  Only temporal support / rendering changes.
- **Not tested:** QRS or PQRST topology, tangent projection, a new generator, hierarchical latents, phase normalization,
  a new training objective.
- **Limits on use:** D0 can only motivate or reject a future, separately preregistered architecture. The VitalDB
  validation split is not fresh: ED1, FBC1, M1 and BF0 analyzed it.

## 1. Hypothesis H_D0

A beat-local model whose target / support reaches into neighbouring-event territory can learn neighbouring cardiac
structure. With BF0's overlap-add renormalization, that structure is usually masked by the next placed beat. At window
edges, missed events and other incomplete coverage it can be exposed and detected as a false R.

If H_D0 holds, then restricting each frozen beat to an event-owned temporal cell, without retraining, should:

1. reduce net false R detections;
2. reduce false detections between normally spaced beats;
3. recover a substantial part of BF0's G1 F1 gap;
4. do the same in the deterministic arm.

## 2. Pre-D0 post-hoc observations (motivation only, never evidence)

These come from frozen BF0 validation renders, stochastic seed 0, analyzed after BF0:

- **Extra detections:** 1,450 detections unmatched to a placed event. None lies within ±50 ms of a placed R, and 1,201
  are far from any placed beat:
  - 543 before the first placed beat;
  - 389 after the last;
  - 253 between normally spaced beats;
  - 16 in gaps.
- **Lag:** the median far-ghost lag after the previous placed beat is about 750 ms.
- **Beat support and RR:** the beat target / support is R − 500 … R + 789 ms, and the median RR is about 781 ms. For
  about 38% of **placed validation beats** (not training targets), the next R lies inside the support.
- **Ghosts:** the learned beats draw a neighbouring QRS there, at a tail peak of about 0.9–1.0 × their own R amplitude.
- **Deterministic arm:** it shows the same pattern.

## 3. Frozen inputs (verified before this commit)

- **Fixed from BF0:**
  - checkpoints `beat_deterministic.pt` (sha256 `e06a328d…`) and `beat_stochastic.pt` (`5fc7a803…`);
  - placed events in `render_val.npz`;
  - BF0 conditioning;
  - stochastic seed 0 (`1_000_003·n + j`);
  - template and empty-window fill (−0.5418);
  - the 4,822 validation windows of 289 patients.
- **Reproduction:** regenerated per-beat outputs passed through the **original** renderer reproduce every saved BF0
  window bit for bit. The maximum difference is 0.0 for the stochastic seed-0, deterministic and template arms
  (`original_reproduction.json`).
- **Run-time checks:** the evaluation repeats this check and also requires S-ORIG to reproduce BF0's G1 (−0.0284
  [−0.0312, −0.0257]) to 1e-12. If either fails, it **stops**.

## 4. The intervention: event-cell renderer (`src/ppg2ecg/beatfirst/eventcell.py`)

Placed events in a window are r_1 < … < r_N (absolute samples).

- **Interior cells:** C_i = [(r_{i−1} + r_i)/2, (r_i + r_{i+1})/2].
- **First and last cells:** L_1 = r_1 − (r_2 − r_1)/2 and U_N = r_N + (r_N − r_{N−1})/2.
- **Single event:** r_1 ± RR_train/2, with RR_train = 0.78125 s = 100 samples. This is BF0's train-split median
  reference RR (`template.npz`, BF0 `train_beat` stage).
- **Absolute time:** cells are never mapped to [0, 1]. Nothing is warped, stretched or rescaled: a beat sample keeps its
  absolute offset from its R.
- **Neighbouring cells** meet in a complementary raised-cosine crossfade. The weights sum to 1, and there is no
  renormalization.
  - **Width:** 4 samples (§5), capped at r_{i+1} − r_i − 2 so it never reaches an R sample.
  - **Position:** centred on the midpoint, unless it would leave either beat's own output support (R − 64 … R + 101). It
    then moves the least amount that keeps it inside both.
  - **Overlap narrower than the crossfade:** it spans the overlap.
  - **No overlap at all:** the gap gets BF0's fill, as the original renderer does.
- **Outside every owned region:** samples before the first or after the last owned region, and true gaps, get BF0's
  nearest-covered-value rule (ties to the left). A window without events gets −0.5418. The fraction of each window filled
  this way is reported.
- **Unchanged between ORIG and CELL:** weights, beat outputs, PPG and RR conditioning, events, seed, scaling,
  preprocessing, detector, reference R, windows and metrics. Both renders are stored as float32.

## 5. Crossfade width (TRAIN only, frozen)

- **Data:** 3,000 salted TRAIN windows. Positions come from the BF0 event stage on TRAIN PPG. No R detection is run on
  any render, and no F1, FD or validation number is used.
- **Statistics at each crossfade (± 1 sample):**
  - A = max |first difference|;
  - B = max |second difference|.
- **Rule:** the smallest of {4, 8, 12, 16} samples for which P95(A) and P95(B) are no larger than the P95 of the same
  statistics on real TRAIN ECG at midpoints between consecutive reference R, in both arms. If no width passes, use 16
  and mark the criterion unmet.
- **Result: 4 samples (31.25 ms), criterion met.**
- **Limitation:** the real-ECG P95 is dominated by artifact windows and missed-beat midpoints. The median is 0.023 but
  the P95 is 0.994, so the criterion did not bind. At 4 samples, the stochastic arm's median boundary jump (0.096) is
  about 4 times real ECG's. Boundary-induced false detections are a known risk, audited in §6.6 and §8.
- **Sensitivity:** a pre-specified renderer at **16 samples** is reported for description only (S-CELL16, D-CELL16). It
  **never enters the verdict**, and the primary width is not revisited after results.

## 6. Outcomes

### 6.1 Arms
S-ORIG and S-CELL (stochastic seed 0), D-ORIG and D-CELL (deterministic), T-ORIG and T-CELL (template, a
negative/reference control), plus the §5 sensitivity arms.

### 6.2 False-positive definition (BF0 G1 convention)
- **FP:** a detected R (neurokit, as in BF0) not matched one-to-one to a reference R within ±50 ms
  (`rpeaks.match_rpeaks`).
- **Per-window quantity:** FP_count_w.
- **Patient quantity:** FP_rate = mean over the patient's windows (all 4,822 windows are eligible).
- **Effect:** the paired difference CELL − ORIG, with a patient-macro mean and a patient-bootstrap CI.
- Raw TP / FP / FN totals are reported descriptively.

### 6.3 D1 — net false-detection reduction (primary)
FP_rate(S-CELL) − FP_rate(S-ORIG): **PASS if the 95% CI lies entirely below 0.**

### 6.4 Attribution zones (post-hoc zones from §2, reused unchanged)
- **Zones,** from the offset of each FP from the nearest placed R:
  - A: |Δ| ≤ 50 ms;
  - B: +50 to +150 ms;
  - C: +150 to +450 ms;
  - D: −300 to −50 ms;
  - E: far (everything else).
- **Far FPs** are further split into:
  - before the first placed event;
  - after the last placed event;
  - inside a gap > 1.5 × the window's median placed RR;
  - **interior**: between normally spaced placed events.
- **Also reported:** distance to the nearest placed R; the far FP's lag from the previous placed R as a fraction of the
  local median RR; and the share of the total FP reduction coming from edge far, interior far, gap, P/PR, ST, T and
  ±50 ms.

### 6.5 D2 — interior far-ghost reduction
The interior far FP rate per window, as a patient mean, S-CELL − S-ORIG: **PASS if the 95% CI lies entirely below 0.**
The relative reduction (point estimate) is reported, with no threshold. Edge far FPs (before first / after last) are
descriptive only, because their removal is partly mechanical.

### 6.6 Boundary-artifact audit (mandatory, diagnostic)
- **New CELL-only FP:** a CELL FP with no ORIG FP within ±50 ms in the same window.
- **Boundary-near:** within one crossfade width (4 samples) of a crossfade centre ("crossfade"), or of an owned-region
  limit next to the fill ("edge limit").
- **Reported:** new FPs, boundary-near counts and fraction, and ORIG FPs removed, for S and D.

### 6.7 D3 — BF0 G1 recomputation (diagnostic target)
- **Statistic:** F1(S-CELL) − F1(placed events), with BF0's exact G1 definition (evaluable windows, ±50 ms, same
  detector and bootstrap).
- **Target:** 95% CI lower bound > −0.02.
- **Required sentence even if the target is met:** "The post-hoc event-cell rerender would have satisfied the former
  BF0 G1 non-inferiority margin, but cannot change the preregistered BF0 verdict."
- **Gap recovery:** (F1_S-CELL − F1_S-ORIG) / (F1_placed − F1_S-ORIG) on equal-patient-weight means. It is
  bootstrapped with the same resamples when the denominator stays positive, and reported conservatively ("on this same
  validation population, rerendering accounts for about X% of the observed degradation"), never as an independent causal
  effect size.

### 6.8 D4 — deterministic replication
FP_rate(D-CELL) − FP_rate(D-ORIG): **PASS if the 95% CI lies entirely below 0.**

### 6.9 Secondary (all descriptive)
- **Misses**, for S and D, ORIG and CELL:
  - FN totals and patient-macro FN rate;
  - placed-event redetection rate, overall, at the edge and in the interior;
  - missed placed events by distance to the window edge (< 0.25 s, 0.25–0.5 s, ≥ 0.5 s);
  - missed placed events whose T zone (R + 20 … R + 57 samples) is at least as tall as R.
- **Edge vs interior:**
  - EDGE event: first or last placed event, or a nominal cell reaching past the window;
  - INTERIOR event: all others;
  - FP by the nearest event's category, and misses by category.
- **Waveform side effects**, CELL vs ORIG (S, D; T for reference):
  - FD (`kanflow_fd`; CELL − ORIG CI for S and D with 1,000 resamples);
  - matched-pair beat-aligned correlation (pairs shared by every arm);
  - S4 / S5;
  - HR MAE;
  - join-artifact rate J (BF0 definition);
  - spectral ratio deviation (`m1_structural.spectral_metrics`, bands F1–F4 and their mean);
  - flat-fill fraction (mean, median, IQR, P95, distribution of patient means).

## 7. Statistics
- **Bootstrap:** BF0 convention. Patient-clustered, equal patient weight, 2,000 resamples (FD 1,000), seed 20260930.
  Paired arms use identical patient resamples, and windows are never resampled independently.
- **Sanity:** S-ORIG must reproduce BF0's G1 exactly, or the run stops.

## 8. Verdict on the support-leakage hypothesis (not on BF0)

**Safety conditions** (engineering thresholds, frozen here):

- **S1 flattening:** the patient-macro mean flat-fill fraction, S-CELL minus S-ORIG, is ≤ 0.25.
- **S2 boundary artifacts:** boundary-near new CELL-only FPs divided by ORIG FPs removed, stochastic arm.
- **S3 realism:** FD(S-CELL) − FD(S-ORIG), point estimate, is ≤ 6.46. This is the size of BF0's preregistered stochastic
  − deterministic FD advantage, so a larger loss would erase it.

**Verdict:**

| verdict | condition |
|---|---|
| NOT SUPPORTED | D1 fails, or S1 fails ("waveform damage makes the event improvement uninterpretable"), or S2 ≥ 0.5 ("CELL creates comparable new false events") |
| STRONGLY SUPPORTED | D1, D2, D3 and D4 all pass, S2 < 0.25, and S3 holds |
| PARTIALLY SUPPORTED | every other case (D1 passes but D2, D3 or D4 fails, or S2 is in [0.25, 0.5), or S3 fails) |

The specification lists "interior far ghosts do not decrease" under both PARTIAL (D1 pass, D2 fail) and NOT SUPPORTED.
This table resolves it: with D1 passing, a D2 failure gives PARTIAL. The sensitivity arms (§5) never enter the verdict.

## 9. Claim boundaries
- **Never said, whatever the result:**
  - BF0 passed or was fixed;
  - Beat-First succeeded;
  - support leakage is definitively causal;
  - EventCell is a validated architecture;
  - stochastic morphology is necessary;
  - a hierarchical latent is validated;
  - PPG identifies patient-specific morphology;
  - PPG contains no morphology information;
  - test performance or generalization is established.
- **Morphology wording:** "Prior experiments did not show useful PPG-conditioned gains in pointwise beat-morphology
  reconstruction over template-level prediction." BF0's shuffles showed that PPG conditioning changes the generated
  morphology distribution, so pointwise identifiability is kept separate from distributional or contextual conditioning
  effects.
- **Permitted wording:** "On the same validation population, a frozen-output post-hoc intervention supports the
  hypothesis that temporal-support leakage of neighbouring cardiac structure contributed substantially to BF0's false
  event detections."
- **If STRONGLY SUPPORTED, also permitted:** "Restricting each frozen beat to event-owned temporal support reduced both
  overall false detections and interior neighbour-ghost detections without retraining."

## 10. After the result
- **STRONGLY SUPPORTED:** write only `docs/EC0_EVENT_CELL_ARCHITECTURE_DRAFT.md`, a design note (not a preregistration,
  not an experiment). It covers:
  - event-cell support ownership;
  - an event-cell **training target** (the main difference from a renderer patch);
  - absolute-time local decoding;
  - a partition-of-unity renderer;
  - a hierarchical latent as a separate future hypothesis, labeled "not tested by D0";
  - future ablation order: long-target vs event-cell-target, deterministic and stochastic, before any latent change.
- **Otherwise:** a short report section on what support clipping fixed and did not fix: whether edge effects dominated,
  whether boundary artifacts appeared, and whether realism deteriorated.
- **Never trained:** EC0, any EventCell model, a new beat generator, a hierarchical latent model, a residual model or a
  larger model.

## 11. Implementation and tests
- **Code:** `scripts/d0_event_cell.py` (stages `reproduce`, `crossfade`, `manifest`, `evaluate`, `figure`) and
  `src/ppg2ecg/beatfirst/eventcell.py`.
- **Tests:** `tests/test_d0_event_cell.py`, 42 synthetic tests:
  - cells, midpoints, first / last / singleton rules, TRAIN RR;
  - no warping, R not shifted, inputs unmodified;
  - zero weight outside support, complementary crossfades, partition of unity;
  - BF0 fill rule, support-shift rule;
  - synthetic neighbour ghost exposed by BF0 and removed by cells;
  - step detection, zone and interior classification;
  - ±50 ms FP matching;
  - width selection rule; crossfade stage never touches validation or detection;
  - frozen seed and deterministic calls; hard stops;
  - paired gap-recovery bootstrap; verdict table; sensitivity never in the verdict.
- **Run order:** `reproduce` and `crossfade` (done), then `manifest`, then commit and push, then `evaluate` and `figure`,
  then the report `docs/D0_EVENT_CELL_SUPPORT_DIAGNOSTIC_REPORT.md`, then commit and push, then **HARD STOP**.

## 12. Artifacts
`artifacts/d0_event_cell_support/`: `audit.md`, `original_reproduction.json`, `prereg_manifest.json`,
`input_hashes.json`, `crossfade_train_selection.json`, `renderer_config.json`, `event_cell_manifest.json`,
`primary_metrics.json`, `bootstrap.json`, `fp_attribution.json`, `interior_far_fp.json`, `boundary_artifacts.json`,
`edge_interior.json`, `miss_analysis.json`, `waveform_metrics.json`, `flat_fill.json`, `gap_recovery.json`,
`sensitivity_w16.json`, `figure.png`.

`outputs/d0_event_cell_support/` (never committed): regenerated beats, CELL renders, logs. BF0 files are not modified.
