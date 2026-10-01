# E0 — WW-DET Spontaneous Event Audit — REPORT

**E0 case: E0-C — no local guard motivation.** ARCH-VAL gives E0-C and the previously opened ARCH-HOLDOUT gives E0-C.
**Event Guard motivated: NO. E1 design draft: not created.**

> **Post-hoc mechanistic audit of frozen outputs.** ARCH-VAL was analysed by C0 and C0-A; ARCH-HOLDOUT was opened by C0
> and by C0-A. Nothing here is validation, fresh or otherwise. No model was trained or changed, and no waveform was
> modified.

| commit | content |
|---|---|
| `0e9021c` | preregistration + implementation (after exact reproduction, before any event-level outcome) |
| this commit | results, post-hoc readings, figure, atlas, report |

Labels:

- **[prereg]** preregistered analysis or rule;
- **[post-hoc]** reading added after the outcomes (`scripts/e0_posthoc.py`, `posthoc_readings.json`), never used by the
  case rule.

## 한국어 요약

- **판정: E0-C.** VAL과 HOLDOUT 모두 같은 판정이며, 이벤트 가드를 만들 근거가 없다. E1 설계 초안은 만들지 않았다.
- **WW-DET가 C0보다 가짜 R이 많은 이유:** 거의 전부가 D형(근거 없는 자발 이벤트)이다.
  - 창당 +0.115개로, WW−C0 가짜 R 차이(+0.117)의 98%를 차지한다.
  - 물려받은 오류(B형) 차이는 +0.002뿐이다.
- **위치:** D형의 75%는 검출기가 이벤트를 하나도 주지 않은 창에서 나오고, 나머지 대부분은 첫 이벤트 앞이나 마지막
  이벤트 뒤다. 이벤트 사이에서 나온 것은 2%뿐이다.
  - 이런 창에서 WW 출력은 거의 평탄한 선 위의 작은 굴곡이다. 돌출도가 0.037로, 진짜 박동(A형) 1.66의 약 45분의
    1이다. neurokit이 이 굴곡을 R로 잡는다.
- **C형(자발적 복원):** 507개, 기준 박동의 0.39%다. 위치와 모양이 D형과 같은 작은 굴곡이다.
  - [사후 해석] C형 비율 12.3%는 굴곡을 아무 시점에나 찍었을 때 ±50 ms 안에 실제 박동이 있을 확률(11.5%)과 거의
    같다. HOLDOUT도 12.4% vs 11.3%.
- **C와 D는 구분되지 않는다.**
  - ARCH-TRAIN에서 학습한 로지스틱 프로브의 AUROC: 이벤트 맥락 0.51, WW 파형 0.54, 전체 0.57 (HOLDOUT 0.51 / 0.51 /
    0.55).
  - 효과크기는 모두 |d| ≤ 0.24로 작다.
- **"WW F1 > 놓은 이벤트 F1"의 정체:** 창별 F1 평균에서만 나타난다.
  - 이득 +0.0031은 전부 이벤트 없는 창에서 온다(+0.0039). 그런 창에서는 놓은 이벤트의 F1이 0인데, WW는 우연히 맞는
    굴곡으로 F1 0.05를 얻는다. 이벤트가 있는 창의 기여는 −0.0007이다.
  - 모든 박동을 합쳐서 계산하는 F1(micro)은 오히려 WW가 낮다(0.820 vs 0.830).
- **반사실 진단(방법이 아님):**
  - 놓은 이벤트 50 ms 밖의 검출을 모두 버리는 하드 가드: WW의 이벤트 지표가 놓은 이벤트와 똑같아진다(F1 0.7901).
    가짜 R은 창당 −0.128, 창별 F1은 −0.0032다.
  - 사전등록 규칙상 "하드 가드는 F1을 낮춘다(K2b)"가 성립하므로, 가드를 만든다면 선택적이어야 한다는 문장을 적는다.
  - 다만 그 F1 손실은 위의 우연 일치를 버린 결과로 보인다(사후 해석).
  - D만 지우는 오라클-1: F1 +0.0024, 가짜 R −0.128. B+D를 지우는 오라클-2: F1 +0.019. 둘 다 불가능한 상한일 뿐이다.
- **결론:** WW−C0의 가짜 R 차이는 "이벤트가 없을 때 무엇을 그리느냐"의 문제로 좁혀졌다. 국소 파형으로 좋은 자발
  이벤트와 나쁜 자발 이벤트를 가를 근거는 없다. 이벤트 가드 분기는 멈추고, WW와 C0의 차이는 측정된 맞교환으로 둔다.

## 1. Status

- **Process:** E0 ran as preregistered: reproduction → preregistration `0e9021c` → ARCH-TRAIN inference → evaluation →
  figure → atlas, one run each.
- **Deviations:** none. The only additions are the post-hoc descriptive readings, which were added after the outcomes and
  are labelled.
- **Tests:** E0 18; full suite 940 passed at the preregistration commit.

## 2. Why E0 exists

On ARCH-HOLDOUT, C0-A found that WW-DET has better FD than CoherentBeat-C0 (20.01 vs 25.10), more false R detections
(0.554 vs 0.445 per window), and yet a slightly higher per-window F1 than both C0 and the placed events (0.7917 vs 0.7878
/ 0.7888). E0 asks what WW-DET's additional detected events are:

- inherited detector errors;
- recoveries of real beats;
- hallucinations;
- or a mixture;

and whether harmful and useful spontaneous events can be told apart by simple local features. That separability is the
precondition for a selective Event Guard.

## 3. Evidence-status limitation

- **No fresh population.** ARCH-VAL (primary) and ARCH-HOLDOUT (secondary) were both analysed before. The C0-A
  aggregate counts were known before the preregistration and imply type C ≥ 502 and D ≥ 3,615 on ARCH-VAL. The
  preregistration disclosed this, and the case thresholds were set with that knowledge.
- **The probe's training population is in-sample.** ARCH-TRAIN outputs of the frozen detector and WW-DET come from
  models trained on ARCH-TRAIN.
- **Status of the results:** everything below is a description of frozen outputs, not validation and not a causal
  explanation.

## 4. Frozen models and exact reproduction [prereg]

- **Frozen checkpoints:** detector, C0 and WW-DET match their recorded sha256.
- **Bit-identical reproduction:** the stored events and C0 renders were reproduced bit for bit, and WW-DET was
  re-rendered with C0-A's own `render_ww`.

| max \|Δ\| vs C0-A stored metrics (F1, precision, recall, FP, FN, RR-MAE, HR-MAE; point and CI) | ARCH-VAL | ARCH-HOLDOUT |
|---|---|---|
| placed events | 0.0 | 0.0 |
| CoherentBeat-C0 | 0.0 | 0.0 |
| WW-DET | 0.0 | 0.0 |
| raw FP / FN totals equal | yes | yes |

## 5. Four-way event taxonomy [prereg]

- **Matching:** one-to-one ±50 ms (`match_rpeaks`) for REFERENCE↔PLACED, REFERENCE↔RENDERED and PLACED↔RENDERED.
- **Types:**
  - A: placed-matched and reference-matched;
  - B: placed only;
  - C: reference only;
  - D: neither.
- **Run-time checks:** A + C equalled the metric's TP and B + D its FP in every window. No type-A detection had a
  placed partner different from its reference beat's placed partner (0 triangle cases).

| raw counts | ARCH-VAL WW-DET | ARCH-VAL C0 | ARCH-HOLDOUT WW-DET | ARCH-HOLDOUT C0 |
|---|---|---|---|---|
| A supported true | 100,193 (86.6%) | 100,101 (89.5%) | 99,501 (85.9%) | 99,412 (88.6%) |
| B inherited false | 11,374 (9.8%) | 11,327 (10.1%) | 12,485 (10.8%) | 12,419 (11.1%) |
| C spontaneous recovery | **507 (0.44%)** | 10 (0.01%) | **472 (0.41%)** | 17 (0.02%) |
| D spontaneous hallucination | **3,620 (3.13%)** | 357 (0.32%) | **3,329 (2.88%)** | 342 (0.30%) |
| C as % of reference beats | 0.39% | 0.01% | 0.37% | 0.01% |

Percentages are of the arm's rendered detections. ARCH-VAL has 129,825 reference beats and 111,577 placed events;
ARCH-HOLDOUT 128,347 and 111,993.

## 6. WW vs CoherentBeat event accounting [prereg]

| per window, patient mean [95% CI] | ARCH-VAL | ARCH-HOLDOUT |
|---|---|---|
| WW type C | 0.0179 [0.0158, 0.0203] | 0.0166 [0.0147, 0.0185] |
| WW type D | 0.128 [0.115, 0.141] | 0.118 [0.109, 0.130] |
| C0 type C / type D | 0.0003 / 0.0126 | 0.0006 / 0.0120 |
| **ΔC (WW − C0)** | **+0.0176 [+0.0154, +0.0199]** | **+0.0159 [+0.0141, +0.0179]** |
| **ΔD (WW − C0)** | **+0.115 [+0.103, +0.128]** | **+0.106 [+0.097, +0.117]** |
| ΔB | +0.0018 [+0.0011, +0.0025] | +0.0023 [+0.0014, +0.0033] |
| ΔFP (= ΔB + ΔD) | +0.117 [+0.105, +0.130] | +0.109 [+0.099, +0.120] |

**Reading [prereg].** WW-DET's false-detection excess over C0 is almost entirely type D: 98% on ARCH-VAL and 98% on
ARCH-HOLDOUT. Inherited detector errors (B) are essentially the same in both arms.

- **Carried placed events.** WW-DET carries nearly every placed event: only 5 placed true and 5 placed false events are
  not carried as A / B (L, M on ARCH-VAL). C0 does not carry 97 and 52.
- **Accounting identities (ARCH-VAL):**

  | arm | TP change: C − L | FP change: D − M |
  |---|---|---|
  | WW-DET | 507 − 5 = **+502** | 3,620 − 5 = **+3,615** |
  | C0 | 10 − 97 = −87 | 357 − 52 = +305 |

  Each term matches the raw totals exactly.

**Spec §9 reading.** WW has both more C and more D, which is the situation the specification describes as "a hard event
lock is inappropriate". In size, the excess is 6.5 D for every extra C.

## 7. Why WW F1 exceeds placed-event F1

**[prereg] Detection-level decomposition (ARCH-VAL; per-window means, paired):**

| | precision | recall | per-window F1 |
|---|---|---|---|
| placed events | 0.8206 | 0.7768 | 0.7901 |
| WW-DET | 0.8221 | 0.7806 | 0.7934 |
| contribution of type C: WW − (WW without C) | +0.0082 [+0.0071, +0.0095] | +0.0039 [+0.0034, +0.0045] | **+0.0049 [+0.0043, +0.0056]** |
| cost of type D: (WW without D) − WW | +0.0114 [+0.0102, +0.0127] | 0 | **+0.0024 [+0.0021, +0.0026]** |
| WW − placed | +0.0015 [+0.0001, +0.0030] | +0.0039 [+0.0034, +0.0044] | +0.0032 [+0.0026, +0.0039] |

- **[prereg] Pooled (micro) chain, ARCH-VAL:**
  - placed: precision 0.8980, recall 0.7718, F1 0.8301;
  - carried A + B: unchanged;
  - + C: precision 0.8985, recall 0.7757, F1 0.8326;
  - + D (= WW-DET): precision **0.8704**, recall 0.7757, F1 **0.8203**.
  - ARCH-HOLDOUT: 0.8280 → 0.8303 → **0.8190**.
  - **Pooled F1 is lower for WW-DET than for the placed events.** The "WW F1 > placed F1" result holds only for the
    per-window F1 average the program reports.
- **[post-hoc] Where the per-window gain comes from.**
  - The mean per-window ΔF1 (WW − placed) is +0.0031 on ARCH-VAL. Of that, +0.0039 comes from the 2,158 windows without
    any placed event and −0.0007 from all other windows.
  - In the windows without events, the placed events' F1 is 0 by construction. WW-DET's is 0.051, and WW-DET has at
    least one detection in 1,726 of them.
  - ARCH-HOLDOUT is the same: +0.0029 = +0.0037 − 0.0008 (1,993 windows).

## 8. Type C spontaneous recoveries [prereg; post-hoc where marked]

- **[prereg] Counts:** 507 on ARCH-VAL and 472 on ARCH-HOLDOUT, i.e. 0.39% / 0.37% of reference beats and 12.3% / 12.4% of
  WW-DET's spontaneous detections.
- **[prereg] Location:**

  | | no placed event in the window | before the first placed event | after the last | between placed events |
  |---|---|---|---|---|
  | ARCH-VAL | 376 (74%) | 54 | 68 | 9 |
  | ARCH-HOLDOUT | 349 | 39 | 77 | 7 |

- **[prereg] Shape:** low-amplitude structures. Median prominence 0.037 and amplitude above the local median 0.028 (in
  normalized ECG units); QRS-template correlation 0.62.
- **[post-hoc] Compared with WW-DET's own type-A detections:**
  - prominence 1.66 (C is about 45 times smaller);
  - amplitude 1.54;
  - template correlation 0.998.
- **[post-hoc] Chance coincidence:**
  - Suppose WW-DET's spontaneous detections were at random times in their own windows. The expected share within ±50 ms
    of a reference R would be 11.5% on ARCH-VAL and 11.3% on ARCH-HOLDOUT.
  - The observed type-C shares are 12.3% and 12.4%.
  - In windows without placed events: 12.2% observed vs 11.4% expected (HOLDOUT 12.5% vs 11.2%).
- **Reading.** Type C as preregistered (reference-matched, not supplied by the detector) exists, but it does not look
  like the decoder recovering missed beats. It looks like low-amplitude structures in otherwise flat output that happen
  to fall near a real beat at about the rate chance predicts. The atlas shows the same.

## 9. Type D spontaneous hallucinations [prereg]

- **Counts:** 3,620 on ARCH-VAL and 3,329 on ARCH-HOLDOUT (3.1% / 2.9% of WW-DET's detections). C0 has 357 / 342.
- **WW-DET type D location (ARCH-VAL):**
  - no placed event in the window: 2,708 (75%);
  - before the first placed event: 305;
  - after the last placed event: 536;
  - between placed events: 71 (2%).
- **Shape:** the same size as type C (prominence 0.036, amplitude 0.026, template correlation 0.63).
- **C0 type D:** these are a different phenomenon. Of 357 on ARCH-VAL, 315 lie between placed events, 200–500 ms from the
  nearest placed R, most often at phase 0.15–0.30. Neutral description only; no wave labels.

## 10. Localization in the cardiac interval [prereg]

- **Only a few WW-DET spontaneous detections lie between two placed events:** 9 C and 71 D on ARCH-VAL; 7 C and 64 D on
  ARCH-HOLDOUT.
  - Their phase distribution covers 0.15–0.85, with C and D overlapping (median phase 0.52 vs 0.59).
  - Distance to the nearest placed R exceeds 500 ms for 126 of 131 C and 861 of 912 D on ARCH-VAL, among detections
    with a placed event in the window.
- **Reading.** WW-DET's spontaneous detections are not intrusions into normal inter-beat intervals. They appear where
  the event raster is empty: whole empty windows, or the stretches before the first and after the last placed event.

## 11. Edge analysis [prereg]

| ARCH-VAL, distance to the nearest window edge | < 125 ms | 125–250 ms | 250–500 ms | ≥ 500 ms |
|---|---|---|---|---|
| WW type C | 14 | 20 | 51 | 422 |
| WW type D | 210 | 283 | 301 | 2,826 |

78% of type D and 83% of type C are ≥ 500 ms from a window edge. The mechanism is **not** a window-boundary effect. It is
an absence-of-events effect: the before-first / after-last regions are long event-free stretches, not boundary artifacts.

## 12. Same-location WW vs CoherentBeat morphology [prereg]

Patient-macro mean ±250 ms segments at WW-DET detections, no warping (`same_location.json`, figure panels E1 / E2):

- **Type C:** WW-DET shows a small symmetric bump (about 0.06 above its surroundings) on a flat level. The reference ECG
  shows a smeared R-like peak, because the reference R is within ±50 ms but not aligned. C0 is flat: it has no events
  there and draws only its smooth global field.
- **Type D:** WW-DET shows a similar small bump (about 0.04). The reference ECG is flat on average. C0 is flat.
- **The event raster is zero at both:** these are spontaneous by definition.

**What WW creates that CoherentBeat suppresses:** a small bump on a near-flat output where no event was placed. C0
cannot draw such structure without an event. WW-DET does, and the R detector picks it up.

## 13. Local feature analysis [prereg]

Cohen's d (C − D, patient-cluster bootstrap 95% CI); ARCH-VAL / ARCH-HOLDOUT:

| feature | d VAL | d HOLDOUT |
|---|---|---|
| amp_abs | +0.19 [+0.08, +0.30] | +0.22 [+0.11, +0.33] |
| max_pos_slope | +0.19 [+0.06, +0.33] | +0.14 [+0.01, +0.27] |
| amp_rel | +0.18 [+0.04, +0.32] | +0.19 [+0.05, +0.33] |
| prominence | +0.18 [+0.04, +0.32] | +0.17 [+0.02, +0.31] |
| dist_nearest_ms | −0.14 [−0.30, +0.03] | −0.20 [−0.35, −0.05] |
| qrs_corr | +0.02 [−0.07, +0.10] | +0.04 [−0.05, +0.14] |
| phase | +0.03 [−0.81, +0.85] | −0.34 [−1.31, +0.56] |

- **Every other feature:** |d| ≤ 0.18, except prev_rr_ms on ARCH-HOLDOUT (−0.24 [−0.44, −0.05]; VAL −0.07 [−0.29, +0.16]). Full table in `feature_summary.json`.
- **The largest effects:** type C bumps are marginally larger and steeper. Every effect is small (|d| ≤ 0.24), and the
  medians nearly coincide (prominence 0.037 vs 0.036).

## 14. Diagnostic separability probe [prereg]

**Fitting:**

- L2 logistic regression, `class_weight = balanced`.
- Fitted on ARCH-TRAIN WW-DET type C vs D only: 4,067 C, 29,455 D, 3,413 patients.
- Patient-grouped 5-fold CV chose C = 0.01 for every group. Mean CV AUROC: P1 0.501, P2 0.519, P3 0.569.

**Applied unchanged:**

| | prevalence C | AUROC VAL [95% CI] | AUPRC VAL | AUROC HOLDOUT | AUPRC HOLDOUT |
|---|---|---|---|---|---|
| P1 event context | 0.123 / 0.124 | 0.507 [0.489, 0.527] | 0.124 | 0.513 [0.492, 0.535] | 0.129 |
| P2 WW waveform | | 0.539 [0.511, 0.565] | 0.155 | 0.506 [0.478, 0.534] | 0.144 |
| P3 all features | | **0.573 [0.544, 0.601]** | 0.175 | **0.550 [0.522, 0.581]** | 0.164 |

- **At the TRAIN threshold (P3):**
  - balanced accuracy 0.549 / 0.531;
  - sensitivity for C 0.556 / 0.534;
  - specificity against D 0.542 / 0.529.
- **AUPRC** sits barely above prevalence. Calibration is flat: the observed C fraction is 0.07–0.20 across predicted
  0.38–0.61.
- **Verdict.** Type C and type D are **not** distinguishable from simple local waveform and event-context features. The
  bar was AUROC ≥ 0.80 (VAL) and ≥ 0.75 (HOLDOUT).

## 15. Hard-guard counterfactual [prereg; diagnostic, not a method]

The hard guard keeps a WW-DET detection only if it is within 50 ms of any placed event.

| | precision | recall | F1 | FP / win | FN / win | RR-MAE (ms) | HR-MAE (bpm) |
|---|---|---|---|---|---|---|---|
| WW-DET, VAL | 0.8221 | 0.7806 | 0.7934 | 0.530 | 1.025 | 7.490 | 6.43 |
| hard guard, VAL | 0.8206 | 0.7767 | 0.7901 | 0.402 | 1.042 | 7.452 | 4.68 |
| hard guard − WW, VAL | −0.0015 [−0.0029, −0.0001] | −0.0039 [−0.0045, −0.0034] | **−0.0032 [−0.0040, −0.0026]** | **−0.128 [−0.141, −0.115]** | +0.018 | −0.007 | −0.08 |
| hard guard − WW, HOLDOUT | −0.0009 [−0.0020, +0.0003] | −0.0037 [−0.0042, −0.0033] | −0.0029 [−0.0035, −0.0024] | −0.118 [−0.130, −0.109] | +0.017 | −0.001 | −0.05 |

- **What the guard removes.** It removes exactly the C and D detections; the result is identical to "A + B only". No
  WW-DET detection near a placed event lost its one-to-one assignment.
- **Result.** WW-DET's event metrics become those of the placed events: F1 0.7901 (VAL) and 0.7888 (HOLDOUT). Compared
  with C0, the guarded WW-DET has F1 +0.0012 [+0.0009, +0.0015] and FP −0.011 [−0.013, −0.009] (VAL).
- **[prereg statement rule]** The hard guard lowers per-window F1 (K2b) without material recall loss (K2c not met). By
  the preregistered rule, the report states: **a future guard would have to be selective rather than hard.**
- **[post-hoc] Caveats on that statement:**
  - The F1 loss is the loss of the type-C chance coincidences in event-free windows (§7, §8).
  - The per-window precision convention scores a window with no detections as precision 0.
  - E0 found no feature that could make such a guard selective (§14).

## 16. Oracle selective-guard ceiling [prereg; impossible-reference bounds]

| oracle − WW | precision | recall | F1 | FP / win |
|---|---|---|---|---|
| ORACLE-1 (remove D), VAL | +0.0114 [+0.0102, +0.0127] | 0 | **+0.0024 [+0.0021, +0.0026]** | −0.128 |
| ORACLE-1, HOLDOUT | +0.0111 | 0 | +0.0024 [+0.0021, +0.0026] | −0.118 |
| ORACLE-2 (remove B and D), VAL | +0.064 | 0 | +0.019 [+0.018, +0.020] | −0.530 |
| ORACLE-2, HOLDOUT | +0.067 | 0 | +0.020 [+0.019, +0.021] | −0.554 |

- **ORACLE-1** removes all of WW-DET's excess false detections. Its F1 headroom over WW-DET is +0.0024, against the
  program's 0.02 materiality margin.
- **ORACLE-2's** larger ceiling comes from type B, the timing detector's own false events. A waveform-side guard cannot
  claim it.
- These are not achievable results.

## 17. Precision-recall consequences

- **WW-DET relative to the placed events:**
  - gains a little recall (+0.0039), from coincidental type-C matches;
  - loses pooled precision (0.898 → 0.870);
  - gains per-window precision only through windows that had no placed event.
- **Removing the spontaneous events trades exactly that recall for precision:**
  - hard guard: recall −0.0039, FP −0.128 per window;
  - ORACLE-1: FP −0.128 at no recall cost.
- **The only way to keep the recall and drop the false detections is ORACLE-1.** It needs a C / D separation that E0
  found to be at chance level (§14). At the detection level, WW-DET's event behaviour reduces to the placed rhythm plus
  low-amplitude spontaneous structures in event-free stretches.

## 18. Architectural interpretation

- **Where the C0-A event trade-off lives.** WW-DET's extra false R detections (C0-A M3) are not decoder hallucinations
  inside normal beats, and not boundary artifacts. They are low-amplitude structures WW-DET draws where the event raster
  is empty. C0 draws a smooth field there instead (§12).
- **Upstream vs decoder errors are kept apart.**
  - Type B is inherited from the timing detector equally by both arms (ΔB +0.002 per window).
  - A waveform-side guard could not be credited with it.
- **Type C does not show useful decoder-side beat recovery.** Its size, shape and coincidence rate match type D and
  chance (§8, §13, §14).
- **No selective event-consistency module is motivated.** Its premise, a locally recognizable difference between useful
  and harmful spontaneous events, is absent.
- **What E0 narrows the WW / C0 trade-off to.** The event cost of WW-DET is a property of how it renders event-free
  stretches. Whether the FD advantage of WW-DET also lives there was not tested (§20).

## 19. E0 case [prereg rule]

| criterion | ARCH-VAL | ARCH-HOLDOUT |
|---|---|---|
| K1 ΔD > 0 and ≥ 0.5 × ΔFP | ✓ (0.115 vs 0.117) | ✓ (0.106 vs 0.109) |
| K2a ΔC > 0 and C / (C + D) ≥ 0.20 | ✗ (share 0.123) | ✗ (0.124) |
| K2b hard guard lowers F1 | ✓ | ✓ |
| K2c hard guard recall loss < −0.02 | ✗ (−0.0039) | ✗ (−0.0037) |
| K3 P3 AUROC ≥ 0.80 / 0.75 | ✗ (0.573) | ✗ (0.550) |
| K4 ORACLE-1 headroom | ✓ | ✓ |
| E0-B conditions | ✗ (hard guard lowers F1) | ✗ |
| **case** | **E0-C** | **E0-C** |

**Final case: E0-C** (identical on both populations). Event Guard motivated: **NO**. No E1 draft, and no
event-constrained draft, was created.

## 20. What E0 does NOT establish

- **That anything here is validated.** Both populations were previously analysed.
- **A causal explanation** of why WW-DET draws spontaneous structures.
- **That a learned guard would or would not work.** Only the local-feature separability tested here is absent.
- **Anything about FD.** No waveform changed and no FD was computed. Whether WW-DET's FD advantage comes from
  event-free windows remains open.
- **That type C represents clinically meaningful beat recovery.** The [post-hoc] chance-coincidence reading is
  descriptive, not a test.
- **Overall superiority** of CoherentBeat-C0 or WW-DET; justification of stochastic generation; test, external or
  clinical generalization.
- **Robustness of the probe result.** The probe was trained on in-sample ARCH-TRAIN outputs (the detector and WW-DET
  were trained there). Its failure to separate C from D on ARCH-TRAIN CV (0.50–0.57) is consistent with VAL / HOLDOUT.

## 21. Next-step recommendation

- **Per the frozen rule (E0-C), stop the Event-Guard branch.**
  - Do not draft or train E1.
  - Keep WW-DET vs CoherentBeat-C0 as a documented trade-off: lower FD vs fewer false R detections. E0 locates the event
    side of that trade-off in event-free stretches.
- **If the program wants to act on this, it is a new question for a new preregistration:**
  - what a waveform model should render where the timing module supplies no event;
  - whether FD and false-detection differences both come from those stretches.
  - E0 itself proposes no method and makes no FD claim.
- **HARD STOP.**

## Files

- **Code:**
  - `scripts/e0_event_audit.py`;
  - `scripts/e0_posthoc.py` [post-hoc];
  - `src/ppg2ecg/eventaudit/{taxonomy,features,probe}.py`;
  - `tests/test_e0_event_audit.py`.
- **Artifacts:** `artifacts/e0_wwdet_event_audit/`
  - `audit.md`, `input_hashes.json`, `reproduction.json`, `prereg_manifest.json`
  - `event_taxonomy_{val,holdout}.json`, `event_level_{val,holdout}.parquet`
  - `counterfactual_metrics.json`, `localization.json`, `feature_summary.json`, `same_location.json`
  - `probe_config.json`, `probe_cv.json`, `probe_{val,holdout}.json`
  - `bootstrap.json`, `e0_case.json`, `train_population.json`, `qrs_template.json`, `atlas_index.json`
  - `posthoc_readings.json`, `figure.png`, `atlas.png`
- **Not committed:** caches in `outputs/e0_wwdet_event_audit/`.
