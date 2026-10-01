# D0 — Event-Cell Support Diagnostic on Frozen BF0 Beat Outputs — REPORT

**D0 verdict (frozen rule): NOT SUPPORTED.**

- With the preregistered event-cell renderer, false R detections in the stochastic arm went **up**, not down: D1
  +0.194 [+0.163, +0.230] per window.
- New false detections at cell boundaries were 1.78 times the false detections the renderer removed (S2 ≥ 0.5).
- The deterministic arm improved (D4 PASS).
- **BF0 is unchanged:** Case A, G1 FAIL, BF1 NO-GO. The former G1 margin was not met post hoc either.

| | |
|---|---|
| preregistration | `docs/D0_EVENT_CELL_SUPPORT_PREREGISTRATION.md`, commit `d85f3ef` (before any event-cell validation result) |
| BF0 | prereg `504395d`, result `d380e3b` (not modified) |
| artifacts | `artifacts/d0_event_cell_support/` |
| code | `scripts/d0_event_cell.py`, `src/ppg2ecg/beatfirst/eventcell.py`, `tests/test_d0_event_cell.py` |

Labels used below:

- **[BF0]** BF0 preregistered results;
- **[pre-D0 post-hoc]** observations that motivated D0;
- **[D0]** frozen D0 diagnostic outcomes;
- **[D0 descriptive]** preregistered descriptive outputs that never enter the verdict;
- **[post-hoc]** readings added after the D0 result.

## 한국어 요약

- **판정: NOT SUPPORTED.**
  - 사전등록한 셀 렌더러(크로스페이드 4샘플)는 확률 비트의 가짜 R 검출을 줄이지 못하고 늘렸다. 창당 +0.194 [+0.163, +0.230].
  - 경계 근처에서 새로 생긴 가짜 검출이 1,598개로, 원래 렌더에서 사라진 가짜 검출(898개)의 1.78배다.
- **창 가장자리의 이웃 비트 유령은 예측대로 사라졌다.**
  - 확률 비트: 첫 비트 앞 534 → 55, 마지막 비트 뒤 309 → 40.
  - 결정적 비트: 595 → 11, 232 → 12.
- **새 문제:** 따로 뽑은 확률 비트들은 기저선 높이가 서로 달라서, 짧은 셀 경계에서 계단이 생기고 그 계단이 R로 검출됐다.
  - 가짜 검출의 위치가 RR의 0.53 지점(셀 경계)에 모인다.
  - 결정적 비트는 계단이 적었고(146개) 순 가짜 검출이 줄었다(D4 통과). 템플릿은 전혀 변하지 않았다.
- **기술용 16샘플 민감도 (판정 무관):** 두 팔 모두 가짜 검출이 줄었지만, 확률 비트의 G1 격차는 −0.0215 [−0.0244, −0.0188]로
  여전히 예전 여백 밖이다.
- **BF0 판정(Case A)은 그대로다.** 지지 소유권만으로는 BF0 실패가 설명되지 않는다. 규칙대로 EC0 설계 초안은 만들지 않았다.

## 1. Status

- D0 is closed. Every stage ran once in the preregistered order on 2026-10-01:
  - `reproduce` and `crossfade` (TRAIN) before the preregistration commit;
  - `evaluate` 14:08–14:10 and `figure` after it.
- Hard checks passed:
  - the original renderer reproduced every saved BF0 window bit for bit (max |difference| 0.0, 4,822 / 4,822 windows,
    three arms);
  - S-ORIG reproduced BF0's G1 exactly (−0.0284 [−0.0312, −0.0257]);
  - the partition-of-unity error was 0.0.
- No deviation from the preregistration. No model was trained.

## 2. Why D0 was run

- **The BF0 failure:** BF0 failed G1 because its stochastic renders added false R detections (1,450 unmatched to a placed
  event) and missed 334 placed beats [BF0, post-hoc in the BF0 report].
- **Where those detections were [pre-D0 post-hoc]:**
  - none within ±50 ms of a placed R;
  - 83% far from any placed beat, mostly before the first or after the last beat of a window, at a lag of about one RR.
- **What the frozen beat models do:** they draw a neighbouring QRS at nearly full size inside their 166-sample support
  (−500 … +789 ms).
- **D0's question:** do those ghosts, exposed by BF0's renormalizing overlap-add where no neighbour covers them, explain
  the failure?

## 3. Same-data / post-hoc limitation

- **Post-hoc:** D0 is a post-hoc intervention on frozen outputs from the **same** validation population that suggested
  the hypothesis. It is not an independent confirmation, not causal proof, and not a replication.
- **Not fresh data:** the VitalDB validation split had been analyzed by ED1, FBC1, M1 and BF0.
- **Scope:** D0 can only motivate or reject a future, separately preregistered architecture, and cannot change BF0.

## 4. Frozen BF0 inputs

- **Arms** (`original_reproduction.json`, `input_hashes.json`):
  - S = stochastic, seed 0 (per-beat seed `1_000_003·n + j`);
  - D = deterministic;
  - T = template control.
- **Fixed inputs:**
  - BF0 checkpoints, with sha256 equal to BF0's records;
  - 18,557 placed events in 4,822 windows (438 windows without events);
  - the BF0 conditioning;
  - 289 patients.
- **Per-beat outputs:** BF0 never saved them, so they were regenerated from the frozen checkpoints and seeds and
  re-verified bit for bit.

## 5. Exact original-render reproduction

| arm | max \|difference\| | windows bit-identical |
|---|---|---|
| stochastic seed 0 | 0.0 | 4,822 / 4,822 |
| deterministic | 0.0 | 4,822 / 4,822 |
| template | 0.0 | 4,822 / 4,822 |

## 6. Event-cell intervention

**Cells:**

- Interior cells end at the midpoints of adjacent placed events.
- The first and last cells extend half the adjacent interval outward.
- A single event gets ±RR_train / 2 (100 samples).
- Everything is in absolute time, with no warping.

**Crossfades:**

- Complementary raised-cosine, width 4 samples, capped at RR − 2.
- Moved off the midpoint only when needed to stay inside both beats' output supports. This happened to 2,470 of 14,044
  crossfades. 129 boundaries had no overlap of the supports and got BF0's fill.

**Everything else:**

- No renormalization; the maximum partition-of-unity error was 0.0.
- Outside the owned regions, BF0's nearest-covered-value fill.
- Weights, outputs, events, seeds, detector and metrics were identical between ORIG and CELL.

## 7. Train-only crossfade freeze

`crossfade_train_selection.json`:

- **Data:** 3,000 TRAIN windows. No R detection on any render, and no validation data.
- **Selection:** the frozen rule chose the smallest candidate, **4 samples (31.25 ms)**, and the criterion was met.
- **Why the criterion did not constrain the width (documented before results):**
  - The real-ECG reference P95 (0.994 per sample at RR midpoints) is dominated by artifacts and missed-beat midpoints.
  - At 4 samples, the stochastic arm's median boundary jump on TRAIN was already 0.096, against 0.023 for real ECG and
    0.018 for the original renderer at the same locations.
- This is the risk that materialized (§11).

## 8. D1 — net false-detection reduction [D0] — FAIL

| arm | FP per window (patient mean) | raw TP / FP / FN |
|---|---|---|
| S-ORIG | 0.665 [0.583, 0.750] | 16,388 / 3,285 / 5,502 |
| **S-CELL** | **0.860 [0.779, 0.942]** | 16,296 / 4,189 / 5,594 |
| T-ORIG → T-CELL (control) | 0.406 → 0.406 | FP 2,051 → 2,050 |

**S-CELL − S-ORIG = +0.194 [+0.163, +0.230]**: the CI lies entirely above 0. The stochastic arm gained 904 false
detections net.

## 9. D2 — interior far-ghost reduction [D0] — FAIL

- Interior far FP rate:
  - S-ORIG 0.054 [0.041, 0.068] → **S-CELL 0.335 [0.307, 0.365]**;
  - difference **+0.281 [+0.258, +0.306]**;
  - relative change +524%.
- Raw interior far FPs: 250 → 1,574.
- The deterministic arm rose too: interior far +0.031 [+0.024, +0.039] (87 → 231).

## 10. Edge far-ghosts [D0 descriptive]

| | S-ORIG | S-CELL | D-ORIG | D-CELL |
|---|---|---|---|---|
| before the first placed event | 534 | **55** | 595 | **11** |
| after the last placed event | 309 | **40** | 232 | **12** |

- **What the cells removed:** the ghosts the hypothesis predicted, in both arms. 748 stochastic and 804 deterministic
  edge far FPs disappeared.
- **Where the stochastic net change came from:**
  - edge far FPs fell by 748;
  - interior far FPs rose by 1,324;
  - T-zone FPs rose by 333 (130 → 463).
- **Where the deterministic net reduction (660) came from:** all of it, and more, from edge ghosts (804), partly offset
  by interior far FPs (+144).

## 11. Boundary-artifact audit [D0]

| | new CELL-only FPs | near a crossfade (≤ 4 samples) | near an owned-region edge limit | ORIG FPs removed | boundary-near / removed |
|---|---|---|---|---|---|
| stochastic | 1,802 | 1,550 | 48 | 898 | **1.78** |
| deterministic | 149 | 144 | 2 | 809 | 0.18 |

- **Where the stochastic false detections sit:** their lag from the previous placed beat, as a fraction of the local RR,
  had a median of 0.53 (IQR 0.52–0.57). That is the cell boundary (0.5 RR). Under ORIG the median was 0.66 (0.59–0.84),
  spread toward one RR, which is where the ghosts were.
- **The steps are visible in the renders:** for example, window 1413 in `figure.png` panel C.
- **BF0's join-artifact rate J is not sensitive to them:** it stayed 0.0 in every arm, because its 99.9th-percentile
  real-ECG threshold is far above them.

## 12. D3 — BF0 G1 diagnostic recomputation [D0]

| F1 render − F1 placed events (placed events F1 0.7789) | value |
|---|---|
| S-ORIG (reproduces BF0 exactly) | −0.0284 [−0.0312, −0.0257] |
| **S-CELL** | **−0.0502 [−0.0543, −0.0463]** |
| D-ORIG | −0.0131 [−0.0147, −0.0115] |
| D-CELL | −0.0054 [−0.0067, −0.0041] |

- **Former −0.02 margin met: NO.** The post-hoc event-cell rerender makes the stochastic G1 gap worse.
- **BF0's verdict remains FAIL** in every case.

## 13. G1 gap recovery fraction [D0]

- (F1_S-CELL − F1_S-ORIG) / (F1_placed − F1_S-ORIG) = **−0.77 [−0.93, −0.64]**. The denominator was positive in every
  replicate.
- On this same validation population, event-cell rerendering **widened** the stochastic F1 degradation by about 77%
  instead of recovering it.

## 14. D4 — deterministic replication [D0] — PASS

- FP per window: D-ORIG 0.588 [0.506, 0.673] → D-CELL 0.452 [0.372, 0.534].
- Difference **−0.136 [−0.154, −0.118]**.
- Raw FP: 2,923 → 2,263.

## 15. Missed-beat analysis [D0 descriptive]

| | missed placed events | of which at an edge event | within 0.25 s of the window edge | T zone ≥ R | placed redetection rate |
|---|---|---|---|---|---|
| S-ORIG | 334 | 310 | 135 | 130 | 0.9820 |
| S-CELL | 382 | 342 | 152 | 156 | 0.9794 |
| D-ORIG | 2 | 2 | 1 | 0 | 0.9999 |
| D-CELL | 4 | 4 | 1 | 1 | 0.9998 |

- FN against the reference (patient-macro per window): S 1.128 → 1.144; D 1.074 → 1.091.
- Support ownership does not fix missed beats. Stochastic misses rose slightly, mostly at edge events.

## 16. Edge vs interior [D0 descriptive]

| | S-ORIG | S-CELL | D-ORIG | D-CELL |
|---|---|---|---|---|
| FP nearest an EDGE event | 2,236 | 2,328 | 1,988 | 1,321 |
| FP nearest an INTERIOR event | 1,049 | 1,861 | 935 | 942 |
| redetection, edge / interior | 0.964 / 0.998 | 0.961 / 0.996 | 1.000 / 1.000 | 1.000 / 1.000 |

## 17. Waveform side effects [D0 descriptive]

| | S-ORIG | S-CELL | Δ S [95% CI] | D-ORIG | D-CELL | Δ D [95% CI] |
|---|---|---|---|---|---|---|
| FD | 33.39 | 35.54 | +2.15 [+1.59, +2.67] | 39.86 | 49.33 | **+9.47 [+8.06, +10.86]** |
| matched-pair beat correlation | 0.715 | 0.707 | −0.008 [−0.009, −0.006] | 0.820 | 0.820 | −0.000 [−0.001, +0.000] |
| S4 / S5 | 0.349 / 0.286 | 0.350 / 0.287 | +0.000 / +0.000 | 0.344 / 0.275 | 0.345 / 0.275 | +0.000 / +0.000 |
| HR error (bpm) | 6.39 | **12.42** | **+6.20 [+5.54, +6.89]** | 5.49 | 4.96 | −0.25 [−0.47, −0.02] |
| join-artifact rate J | 0.0 | 0.0 | | 0.0 | 0.0 | |
| spectral ratio deviation (band mean) | 2.49 | 2.51 | +0.02 [−0.09, +0.15] | 2.74 | 2.44 | −0.29 [−0.41, −0.18] |

- Template control: FD 61.71 → 60.44, HR error 4.37 → 4.37.
- **Stochastic arm:** the HR error doubles because of the boundary detections.
- **Deterministic arm:** fewer false events cost realism (FD +9.5). Support clipping did not improve both event and
  waveform quality anywhere.

## 18. Flat-fill analysis [D0]

| | mean | median | IQR | P95 | patient means (P10 / P50 / P90) |
|---|---|---|---|---|---|
| ORIG (all arms) | 0.179 | 0.078 | 0.020–0.154 | 1.0 (empty windows) | 0.098 / 0.165 / 0.260 |
| CELL (all arms) | 0.223 | 0.131 | 0.070–0.211 | 1.0 | 0.148 / 0.213 / 0.306 |

- CELL − ORIG = **+0.044 [+0.042, +0.046]** of each window (safety S1 ≤ 0.25 holds).
- The edge-ghost removal partly works by turning edge content into flat fill.

## 19. Mechanistic interpretation [post-hoc]

1. **Neighbour ghosts at window edges are real and removable.** Event cells removed 89% (stochastic) and 97%
   (deterministic) of the before-first / after-last far FPs, and the deterministic arm's net FP fell. This part of H_D0 is borne out descriptively.
2. **Short cell boundaries expose a second problem in the stochastic arm.** Independently sampled adjacent stochastic beats
   sit at different baseline levels; BF0 measured within-window diversity 2.3 times real. BF0's long Hann overlap-add
   blended these mismatches over tens of samples. A 4-sample crossfade turns them into steps that neurokit reads as R:
   1,550 new FPs near crossfades, at a lag of about 0.5 RR. The template (identical beats) shows none, and the
   deterministic beat (consistent levels) few. So the same frozen stochastic outputs carry two failure sources:
   neighbour content in each beat's support, and incompatibility between neighbouring beats.
3. **The 16-sample sensitivity agrees [D0 descriptive].**
   - Stochastic: net FP fell, −0.126 [−0.143, −0.109], but interior far FPs still rose, +0.020 [+0.015, +0.025]. The
     G1-type gap shrank only to −0.0215 [−0.0244, −0.0188], still outside the former margin; FD was 35.48.
   - Deterministic: FP −0.167 [−0.182, −0.153], gap −0.0021 [−0.0030, −0.0014], but FD 49.30.
   - Even with the friendliest pre-specified crossfade, support ownership alone does not explain the stochastic arm's G1
     failure.

## 20. Alternative explanations

- **Crossfade width.** The frozen width came from a TRAIN rule that did not bind (§7). Part of the D1 failure is a
  property of that rule, as the 16-sample sensitivity suggests. That sensitivity is descriptive by design and cannot
  rescue D0.
- **Detector behaviour.** neurokit's cleaning (high-pass) turns level steps into spike-like transients. Another detector
  could react differently. D0 kept BF0's detector by design.
- **Flat fill.** Part of the edge-ghost removal is replacement by constant fill (+0.044 of each window). It is real in the
  sense of support ownership, but not evidence of better waveforms.
- **Same data.** The zones and the hypothesis came from these validation data, and 94 far "ghosts" in BF0 were in fact
  true beats that RD1 had missed (removing them costs TP).
- **No training-time change.** D0 clips outputs of models trained on long targets. It says nothing about models trained
  on event-cell targets.

## 21. D0 verdict

| criterion | result |
|---|---|
| D1 stochastic FP rate (CI < 0) | **FAIL**: +0.194 [+0.163, +0.230] |
| D2 interior far FP rate (CI < 0) | **FAIL**: +0.281 [+0.258, +0.306] |
| D3 former G1 margin (lower CI > −0.02) | **not met**: −0.0502 [−0.0543, −0.0463] |
| D4 deterministic FP rate (CI < 0) | **PASS**: −0.136 [−0.154, −0.118] |
| S1 flat-fill increase ≤ 0.25 | holds (+0.044) |
| S2 boundary-near new FP / removed | **1.78** (≥ 0.5) |
| S3 ΔFD(S) ≤ 6.46 | holds (+2.15) |
| **verdict** | **NOT SUPPORTED** (D1 fails; S2 ≥ 0.5) |

## 22. Implications for EventCell architecture

The frozen rule is: do not build a new architecture around support ownership, and do not draft EC0. No
`EC0_EVENT_CELL_ARCHITECTURE_DRAFT.md` was created.

- **What support clipping fixed:** edge neighbour-ghost detections in both arms (before-first and after-last far FPs fell
  by 89% and 97%), and the deterministic arm's net false detections (−0.136 per window).
- **What it did not fix:**
  - the stochastic arm's false events: they rose, with new detections at cell boundaries;
  - missed beats, which rose slightly;
  - the stochastic G1 gap, which widened (and was not recovered even in the 16-sample sensitivity).
- **Did edge effects dominate?** Yes. All of the deterministic improvement and all of the stochastic edge reduction come
  from before-first / after-last regions.
- **Did boundary artifacts appear?** Yes: 1,550 stochastic and 144 deterministic new FPs within 4 samples of a
  crossfade.
- **Did realism deteriorate?** Yes. FD +2.15 (stochastic) and +9.47 (deterministic); stochastic HR error +6.2 bpm; flat
  fill +0.044 of each window.

## 23. What D0 does NOT establish

- That BF0 passed, was fixed, or that Beat-First succeeded. BF0 remains Case A.
- That support leakage is the cause of BF0's failure, or that it is not a cause. D0 is a same-data post-hoc intervention.
- Anything about a model **trained** with event-cell targets, or about a hierarchical / shared morphology latent. Neither
  was tested.
- That stochastic morphology is necessary, that PPG identifies patient-specific morphology, or that PPG contains no
  morphology information.
- Any test-set, other-dataset or multi-seed result.

Figure: `artifacts/d0_event_cell_support/figure.png`.

- **Panel A:** a frozen beat output with neighbour QRS in its head and tail.
- **Panel B:** cell weights.
- **Panel C:** a removed FP and an added boundary FP; examples are the first windows in salted order, not curated.
- **Panels D–F:** FP rates, attribution and the G1 recomputation against the former −0.02 margin.
- **Table:** waveform side effects.
