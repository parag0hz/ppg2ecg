# R1 — RhythmField-WW — FINAL ARCHITECTURE REPORT

> **FINAL R1 VERDICT: FAILED (ARCH-VAL). TEST WAS NEVER OPENED.**
> Neither SOFT-RHYTHM-WW nor JOINT-RHYTHMFIELD-WW qualified on ARCH-VAL. Under the frozen rule:
> - the old V1 TEST was not audited, frozen, loaded or evaluated;
> - the architecture search is **closed**;
> - there is no R2.
>
> **The preregistered final RhythmField-WW architecture did not resolve the deterministic event-fidelity /
> waveform-realism trade-off.**

| commit | content |
|---|---|
| `2a598e4` | preregistration + implementation (before any R1 training or metric) |
| this commit | ARCH-VAL results, candidate selection (NONE), report |

Labels:

- **[gate]** preregistered rule;
- **[descriptive]** preregistered, not gated;
- **[post-hoc]** reading added after the results.

## 한국어 요약

- **판정: FAILED.** 두 후보 모두 ARCH-VAL 관문을 통과하지 못했다. SOFT는 다섯 중 V3 하나만 통과했고, JOINT는 하나도
  통과하지 못했다. 그래서 규칙대로 old TEST는 감사·동결·평가 모두 하지 않았고, 아키텍처 탐색은 여기서 끝난다.
- **가짜 R:** 거의 줄지 않았다.

  | | 가짜 R / 창 | WW 대비 |
  |---|---|---|
  | WW-DET | 0.530 | — |
  | SOFT | 0.524 | −0.006 [−0.017, +0.004] |
  | JOINT | 0.520 | −0.011 [−0.028, +0.005] |
  | C0 (목표 수준) | 0.413 | |

- **FD:** 오히려 크게 나빠졌다. SOFT 31.3, JOINT 33.0으로 WW 18.7은 물론 C0 23.7보다도 나쁘다.
- **박동 정렬 상관:** −0.10 / −0.15로 여백(−0.02)을 크게 벗어났다.
- **[사후 해석]** 반면 점별 지표는 좋아졌다: MAE 0.280 vs 0.293, PCC 0.42 vs 0.37, 스펙트럼 편차도 낮다.
  - 확신이 낮은 연속 리듬장을 받은 디코더가 박동을 "평균 쪽으로" 그린 것과 일치한다.
  - 분포 현실감(FD)과 박동 모양을 잃었다. 따로 시험하지는 않았다.
- **확인된 것:**
  - 놓친 박동 위치에도 검출기 리듬장에 임계값 아래 신호가 있다. 중앙값 0.057로, 이벤트 없는 곳(0.004)의 약 14배다.
  - JOINT의 4,225개 파라미터짜리 리듬 헤드는 동결 검출기와 거의 같은 수준의 리듬을 학습했다(환자 평균 F1 0.818 vs
    0.822).
  - 하지만 그 정보가 파형 쪽에서 가짜 R 감소로 이어지지 않았다.
  - 자발 이벤트는 오히려 늘었다. C형은 WW 507개 → SOFT 1,348개, D형은 3,620개 → 4,579개다.
- **절차 기록:**
  - 첫 ARCH-VAL 평가 실행은 도구의 백그라운드 시간 제한으로 아무 출력 없이 중단됐다(로그 0줄).
  - 같은 코드로 한 번 다시 실행했다.
  - 동결 참조(C0, WW-DET, placed)는 C0-A 저장값과 편차 0.0으로 재현됐다.

## 1. Status

- **Process:** R1 ran as preregistered (`2a598e4`): training → ARCH-VAL → gates → selection.
  - Both models were trained once on ARCH-TRAIN (seed 42, last checkpoint, no NaN steps).
  - Selection = NONE, so the TEST freshness audit, the final freeze and the TEST evaluation were not performed
    (`test_freshness_audit.md`: NOT PERFORMED).
  - ARCH-HOLDOUT was not used.
- **Execution note.** The first `evaluate_val` run was stopped by the tool's background time limit after 25 minutes.
  It printed and wrote nothing (empty log, kept as `outputs/r1_rhythmfield_ww/logs/evaluate_val.killed_attempt1.log`).
  The identical stage was then rerun once as a detached process.
- **Deviations from the preregistration:** none.

## 2. Why this is the final architecture experiment

C0 established a deterministic event-locked substrate. C0-A and E0 then showed:

- the event cost of the stronger whole-window model (WW-DET) lives in event-free stretches;
- harmful and useful spontaneous events are not locally separable;
- so neither a guard nor a stochastic extension is motivated.

R1 tested the one remaining architectural hypothesis: replace hard event locations by a dense rhythm field. The
preregistration made it final: failure closes the architecture search.

## 3. Prior evidence from C0-A and E0

| ARCH-VAL (C0-A / E0) | FP / window | FD | beat corr |
|---|---|---|---|
| CoherentBeat-C0 | 0.413 | 23.70 | 0.816 |
| WW-DET | 0.530 | 18.70 | 0.821 |

- **E0:** 98% of WW-DET's false-detection excess over C0 is type D (spontaneous, unmatched). 75% of it lies in windows
  without a placed event.
- **The trade-off to resolve:** WW-level FD with C0-level FP.

## 4. Architecture hypothesis

Thresholding and minimum-distance selection discard subthreshold rhythm information. A whole-window predictor that sees
the dense pre-threshold field could place beats where rhythm evidence exists and avoid drawing them where it does not.

## 5. Hard vs dense rhythm representation

- **WW-DET** sees a Gaussian raster at the thresholded events.
- **R1** sees the dense field p_R(t) ∈ [0, 1] at every ECG sample. No threshold enters the waveform decoder.
- **[descriptive] Is subthreshold information present?** Yes. The frozen detector's field has these medians:

  | position | median field |
  |---|---|
  | correctly placed events | 0.622 |
  | reference R overall | 0.469 |
  | missed reference R | 0.057 (mean 0.082) |
  | random event-free positions | 0.004 |

  Missed beats sit about 14 times above the event-free background, but far below the 0.35 threshold (0.1% of missed
  beats reach it).

## 6. SOFT-RHYTHM-WW

- **Architecture:** the WW-DET class unchanged (593,577 parameters, equal to WW-DET).
- **Conditioning:** the frozen detector's sigmoid field in place of the raster, on the native 512-sample grid. No
  interpolation, temperature or threshold.
- **Training input:** the frozen detector's field on ARCH-TRAIN PPG (inference only).
- **Loss:** full-window L1.
- **Full pipeline including the detector:** 922,474 parameters (the same as WW-DET's pipeline).

## 7. JOINT-RHYTHMFIELD-WW

- **Architecture:**
  - h = E(PPG), the C0 encoder;
  - z_R = RhythmHead(h), two 1 × 1 convolutions (4,225 parameters), [B, 512];
  - p_R = σ(z_R), not detached;
  - ECG = D(h, p_R), WW-DET's decoder.
- **Parameters:** 597,802 total (waveform 593,577 + head 4,225; +0.71% vs WW-DET). There is no external detector.
- **Loss:** L1 + 1.0 × BCE against the C0 detector target (Gaussian, σ = 20 ms, ARCH-TRAIN reference R).

## 8. Training protocol

| | SOFT | JOINT |
|---|---|---|
| data | ARCH-TRAIN (3,470 patients, 231,220 windows) | same |
| protocol | 20,000 × 64, AdamW 1e-3 / 0.01, clip 1.0, seed 42, last checkpoint | same |
| training time (RTX 5090) | 131.5 s | 135.2 s |
| peak GPU memory | 1,609 MiB | 1,635 MiB |
| last logged loss | L1 0.2888 | total 0.4142 (BCE 0.1266) |
| NaN steps | 0 | 0 |
| sha256 | `feb8ffbd…` | `011f85cd…` |

## 9. ARCH-VAL qualification [gate]

- **Population:** 433 patients, 28,649 windows (2,158 without placed events), 91,259 matched beat pairs.
- **Reproduction:** frozen references reproduced the C0-A stored values exactly (maximum deviation 0.0, FD included).
- **Effects:** candidate − reference, paired patient bootstrap (2,000, seed 20261001).

| gate | SOFT | JOINT | rule |
|---|---|---|---|
| V1 FP − WW | −0.006 [−0.017, +0.004] **FAIL** | −0.011 [−0.028, +0.005] **FAIL** | CI upper < 0 |
| V2 FP − C0 | +0.111 [+0.096, +0.127] **FAIL** | +0.106 [+0.086, +0.126] **FAIL** | CI upper < +0.03 |
| V3 recall − WW | +0.0023 [+0.0003, +0.0043] PASS | −0.0051 [−0.0082, −0.0018] **FAIL** | CI lower > −0.005 |
| V4 FD − WW | +12.58 [+11.95, +13.15] **FAIL** | +14.32 [+13.58, +15.02] **FAIL** | CI upper < +1.0 |
| V4 FD − C0 | +7.58 [+7.17, +7.88] **FAIL** | +9.32 [+8.93, +9.64] **FAIL** | CI upper < 0 |
| V5 corr − WW | −0.103 [−0.109, −0.097] **FAIL** | −0.154 [−0.162, −0.146] **FAIL** | CI lower > −0.02 |
| **QUALIFIED** | **NO** | **NO** | |

## 10. Candidate selection

**NONE** (`candidate_selection.json`). Case 1 of the frozen rule: R1 FAILED, TEST not opened, branch terminated.

**SOFT vs JOINT (specification §26).** Neither qualified, so the dense rhythm-field architecture is not supported under
R1. Descriptively, JOINT is worse than SOFT on FD (+1.74), morphology (−0.05) and recall (−0.007), and equal on FP.
Joint end-to-end learning added no demonstrated value.

## 11. Final TEST freshness audit

**Not performed.** No candidate qualified (`test_freshness_audit.md`). The preregistration disclosed that earlier
program phases report V1 TEST metrics; whether those meet the contamination rule was not decided.

## 12. Final freeze

**Not created.** `final_test_freeze_manifest.json` does not exist; the TEST loader stays sealed.

## 13. Final TEST results

**NOT EVALUATED.** TEST was never opened.

## 14. Event fidelity (ARCH-VAL) [descriptive except the gates]

| ARCH-VAL | FP / window (patient) | precision (patient) | recall (patient) | F1 (patient) | pooled P / R / F1 | window F1 (historical) |
|---|---|---|---|---|---|---|
| placed events | 0.4025 | 0.890 | 0.770 | 0.822 | 0.898 / 0.772 / 0.830 | 0.7901 |
| CoherentBeat-C0 | 0.4132 | 0.887 | 0.769 | 0.820 | 0.896 / 0.771 / 0.829 | 0.7890 |
| WW-DET | 0.5302 | 0.861 | 0.774 | 0.813 | 0.870 / 0.776 / 0.820 | 0.7934 |
| SOFT-RHYTHM-WW | 0.5240 | 0.863 | 0.776 | 0.815 | 0.872 / 0.778 / 0.822 | 0.7940 |
| JOINT-RHYTHMFIELD-WW | 0.5196 | 0.862 | 0.769 | 0.811 | 0.872 / 0.770 / 0.818 | 0.7879 |
| JOINT rhythm head as a detector (0.35 / 32) | 0.4013 | 0.887 | 0.765 | 0.818 | 0.898 / 0.767 / 0.827 | 0.7876 |

- **RR-MAE:** placed 7.45 ms, C0 7.45, WW 7.49, SOFT 6.54, JOINT 7.28.
- **HR-MAE:** 4.68 / 4.52 / 6.43 / 7.08 / 5.76 bpm. HR is evaluated on each arm's own window set.
- **Raw false R:** WW 14,994, SOFT 14,837, JOINT 14,726, C0 11,684, placed 11,379.
- **Reading.** Both candidates keep WW-DET's false-detection level, not C0's. JOINT's own rhythm head performs almost
  like the frozen detector, but its waveform does not inherit that event fidelity.

## 15. Waveform distributional quality [gate V4 + descriptive]

| ARCH-VAL | FD | MAE | PCC | spectral dev | S4 / S5 |
|---|---|---|---|---|---|
| CoherentBeat-C0 | 23.70 | 0.2911 | 0.349 | 2.00 | 0.346 / 0.266 |
| WW-DET | **18.70** | 0.2927 | 0.365 | 1.76 | 0.348 / 0.267 |
| SOFT-RHYTHM-WW | 31.27 | 0.2804 | **0.424** | 1.17 | 0.295 / 0.232 |
| JOINT-RHYTHMFIELD-WW | 33.02 | **0.2796** | 0.419 | **1.13** | 0.295 / 0.233 |

**[post-hoc]** The candidates are better on every pointwise measure (MAE −0.012 / −0.013 vs WW-DET; PCC and spectral
deviation also better) and much worse on FD.

- **A reading consistent with this pattern (not tested):** a decoder conditioned on a continuous, often uncertain field
  learns under L1 to hedge beat amplitude and shape toward a conditional average. That lowers pointwise error but moves
  the output distribution away from real ECG.
- **Contrast with WW-DET:** WW-DET was trained on the clean reference-R raster and learned to draw full beats wherever
  the raster says so.

## 16. Morphology [gate V5]

| matched-pair beat-aligned correlation | value | − WW |
|---|---|---|
| CoherentBeat-C0 | 0.817 | |
| WW-DET | 0.822 | |
| SOFT-RHYTHM-WW | 0.719 | −0.103 [−0.109, −0.097] |
| JOINT-RHYTHMFIELD-WW | 0.668 | −0.154 [−0.162, −0.146] |

The population is the pairs finite in all four waveform arms (91,259), so it differs slightly from C0-A's six-arm
population (C0 0.817 here vs 0.816).

## 17. A / B / C / D event accounting [descriptive; placed = frozen detector events]

| ARCH-VAL | A | B | C | D | C / D between placed events |
|---|---|---|---|---|---|
| CoherentBeat-C0 | 100,101 | 11,327 | 10 | 357 | 10 / 315 |
| WW-DET | 100,193 | 11,374 | 507 | 3,620 | 9 / 71 |
| SOFT | 99,649 | 10,258 | **1,348** | **4,579** | 145 / 156 |
| JOINT | 98,518 | 10,248 | **1,468** | **4,478** | 221 / 378 |

- **Dense rhythm conditioning did not reduce type D.** D rose by 26% (SOFT) and 24% (JOINT).
- **Type C rose 2.7–2.9×.** Most of it is after the last placed event or in event-free windows. More of it now lies
  between placed events.
- **Type B fell by 10%.** Fewer inherited false events.
- **Type A fell.** By 544 (SOFT) and 1,675 (JOINT).
- The net effect on false detections is near zero. The answer to the specification's question — "does dense rhythm
  conditioning reduce Type D without eliminating useful Type C?" — is no: it increased both.

## 18. Rhythm-field diagnostics [descriptive; a rhythm score, not calibrated]

| median field (fraction ≥ 0.35) | frozen detector field (SOFT input) | JOINT p_R |
|---|---|---|
| reference R | 0.469 (65%) | 0.460 (64%) |
| correctly placed events | 0.622 (100%) | 0.609 (97%) |
| missed reference R | 0.057 (0.1%) | 0.059 (2.4%) |
| random event-free positions | 0.004 (0%) | 0.004 (0%) |
| SOFT type C / type D detections | 0.254 / 0.052 | — |
| JOINT type C / type D detections | — | 0.302 / 0.066 |

- **JOINT** learned a field almost identical in distribution to the frozen detector's.
- **The spontaneous detections** of both candidates sit mostly where the field is low. Type C sits higher than type D.
- **The detector's field at JOINT's C / D detections** has median 0.070.

## 19. Pareto analysis (ARCH-VAL; `pareto.png`; no combined score)

| arm | FD | FP / window |
|---|---|---|
| WW-DET | 18.70 | 0.530 |
| CoherentBeat-C0 | 23.70 | 0.413 |
| SOFT | 31.27 | 0.524 |
| JOINT | 33.02 | 0.520 |
| PM-BF0 (C0-A, historical) | 30.31 | 0.601 |
| BF0-DET (C0-A, historical) | 32.72 | 0.622 |

- **Ideal region:** WW FD with C0 FP. Neither candidate is near it.
- **Where they landed:** WW-DET's FP level with BF0-like FD.
- **The frontier is unchanged:** C0 (fewest false R) and WW-DET (lowest FD).

## 20. Compute

| | params (full pipeline) | FLOPs, one window | GPU batch-1 | CPU 4-thread |
|---|---|---|---|---|
| CoherentBeat-C0 + detector | 921,667 | 671 M | 1.95 ms | 5.58 ms |
| WW-DET + detector | 922,474 | 942 M | 1.31 ms | 5.43 ms |
| SOFT + detector | 922,474 | 942 M | 1.15 ms | 5.33 ms |
| JOINT (integrated head) | 597,802 | 610 M | 0.76 ms | 3.55 ms |

- **JOINT is the cheapest pipeline,** with no external detector. This does not offset its gate failures.
- **Measurement protocol:**
  - FLOPs come from `torch.utils.flop_counter` on one salted ARCH-VAL window. They are a lower bound and depend on the
    window's event count for C0.
  - Latency is the median over 50 salted windows.
- The models are parameter-matched, not compute-matched.

## 21. What the architecture establishes

- **Subthreshold rhythm information exists.** On ARCH-VAL, the frozen detector's pre-threshold field carries it at
  missed reference beats: median 0.057 vs 0.004 at event-free positions.
- **An integrated rhythm head is cheap.** A 4,225-parameter head on the shared encoder, trained with the detector's
  target and loss, reproduces near-detector event quality (patient F1 0.818 vs 0.822) inside a 597,802-parameter model.
- **Dense-field conditioning failed the gates.** Conditioning a whole-window L1 decoder on that dense field, frozen or
  jointly learned:
  - does not reduce WW-DET's false R detections;
  - lowers pointwise error;
  - but substantially worsens FD and beat-aligned morphology.

## 22. What it does NOT establish

- **Anything about the V1 TEST population.** It was never opened.
- **That no other rhythm-conditioned design could work.** Only these two preregistered models were tested; the branch
  closes by rule, not by proof.
- **That the hedging reading (§15) is the mechanism.** It is post-hoc and untested.
- **That dense vs hard representation alone explains the difference.** SOFT also differs from WW-DET in its training
  input: the detector field on ARCH-TRAIN (in-sample, imperfect) vs the reference-R raster.
- **Anything beyond:** one seed per model, deterministic models only, VitalDB only. No claim of calibrated confidence,
  stochastic generation, clinical validity, external or multi-seed robustness, or state of the art.

## 23. Final branch verdict

**FAILED on ARCH-VAL; TEST not opened.**

> The preregistered final RhythmField-WW architecture did not resolve the deterministic event-fidelity /
> waveform-realism trade-off.

- **The architecture search in this PPG→ECG line is CLOSED.**
- **Not done:** R2, attention / transformers, extra event losses, peak penalties, a λ change, field-temperature tuning,
  an Event Guard, a return to CoherentBeat, stochasticity, opening TEST.
- **The measured frontier stays:** CoherentBeat-C0 (fewest false R) and WW-DET (lowest FD).
- **What comes next:** the next project decision must be a different research problem.

**HARD STOP.**
