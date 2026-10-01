# C0-A — CoherentBeat Architecture-Attribution Ablation — REPORT

**Outcome: Case C; C1 NO-GO.**

| claim | ARCH-VAL | ARCH-HOLDOUT (previously opened) | replication |
|---|---|---|---|
| M1 capacity control | SUPPORTED | SUPPORTED | REPLICATED |
| M2 time-varying global field | SUPPORTED | SUPPORTED | REPLICATED |
| M3 global-local decomposition | NOT SUPPORTED | NOT SUPPORTED | REPLICATED |
| shared absolute context carrier | SUPPORTED | SUPPORTED | REPLICATED |

The ARCH-HOLDOUT column is a frozen secondary replication on the population C0 had already opened once.

**Final architecture claim matrix:**

- capacity-only explanation: **DISFAVORED**;
- shared absolute context carrier: **SUPPORTED**;
- time-varying global field: **SUPPORTED**;
- explicit global-local decomposition: **NOT SUPPORTED**.

C0 itself is unchanged and stays STRONG from its own prospective evaluation (`f9e25b5`). C0-A only decides which
architectural explanations of C0's gain are defensible.

| commit | content |
|---|---|
| `1d3d024` | preregistration + implementation (before any C0-A training or metric) |
| `22c6ddb` | ARCH-VAL result (ARCH-HOLDOUT not yet used by C0-A) |
| `479f138` | holdout ablation freeze (sha256 of 27 files) |
| this commit | ARCH-HOLDOUT frozen secondary replication, compute, figure, table, report |

Labels used below:

- **[claim]** preregistered claim decision;
- **[descriptive]** preregistered and reported, but not part of a claim rule;
- **[post-hoc]** reading added after the result.

## 한국어 요약

- **판정: Case C.**
  - ARCH-VAL 결과: M1 지지, M2 지지, M3 지지 안 됨, 공유 맥락 운반자 지지.
  - C0가 이미 한 번 연 ARCH-HOLDOUT에서 동결 후 1회만 다시 평가했고(2차 재현), 네 판정 모두 같은 부호와 같은 결정으로
    재현됐다.
- **최종 행렬:** 용량만으로의 설명은 기각 쪽(DISFAVORED), 공유 절대 맥락 운반자는 지지, 시간가변 전역장도 지지, 명시적
  전역-국소 분해는 지지 안 됨.
- **용량:** BF0 비트 모델을 C0 크기로 넓혀도(폭 73, +1.1%) C0와의 차이가 거의 그대로다.
  - C0 − PM-BF0: 가짜 R 검출 −0.188/창, FD −6.6.
  - 홀드아웃: −0.186, −6.1.
- **공유 맥락:** 창마다 스칼라 수준값 하나만 줘도 두 가지가 해결된다.
  - LOCAL-ONLY의 가짜 검출 초과분이 사라진다.
  - LOCAL-ONLY → C0 FD 격차의 약 85%가 닫힌다(CONST − LOCAL: FD −13.0, FP −0.340).
- **시간가변 장:** 그 위에 스플라인을 쓰면 FD가 2.2 더 낮아지고, 이벤트·형태 손해는 없다. 효과는 실제지만 작다.
- **분해:** 같은 이벤트를 래스터로 받는 같은 크기의 창 전체 결정적 예측기(WW-DET)와 비교했다.
  - WW-DET가 나은 점: FD가 5.0 낮고(18.7 vs 23.7), F1과 박동 상관이 약간 높다.
  - C0가 나은 점: 가짜 R 검출이 창당 0.117개 적다. WW-DET의 0.530은 C0의 0.413보다 28% 많다.
  - 이렇게 서로 맞바꾸는 결과라서, 분해가 C0 이득의 원인이라고 말할 수 없다.
  - WW-DET는 이벤트에 묶여 있지 않다. 검출기가 놓친 박동을 일부 그리고, 가짜 박동도 더 그린다.
- **논문:**
  - CoherentBeat의 분해를 핵심 기여로 내세우지 않는다.
  - 서술 방향: 창 단위 이벤트 조건 결정적 모델링과 공유 맥락이 비트 단위 렌더링의 문제를 없앤다.
  - 그 안에서 이벤트에 고정된 분해(C0)는 리듬 충실도를, 창 전체 모델은 분포 현실감(FD)을 준다. 이 맞교환으로 쓴다.
- **C1:** 규칙상 NO-GO다. C1은 M3가 ARCH-VAL에서 PARTIAL 이상이어야 한다. C1 문서는 설계 노트로만 남는다.
- **한계:**
  - 모델마다 학습 1회(seed 42)라 학습 반복에 따른 분산은 CI에 들어 있지 않다.
  - 홀드아웃은 신선하지 않다(2차 재현).
  - 파라미터 수만 맞췄고 계산량은 맞추지 않았다(WW-DET FLOPs +18%).
  - M1은 학습 단위(창 vs 비트)와 구조 차이가 섞여 있다.

## 1. Question

C0 showed that a deterministic global-local renderer, placed at the same detector events, beats the retrained BF0
deterministic beat model on false R detections and FD while keeping the placed rhythm. C0-A asks **which part of C0
explains that gain**:

1. **capacity** — C0 has 1.23 × BF0-DET's parameters;
2. **a time-varying global field** vs any shared absolute context — LOCAL-ONLY removed both at once;
3. **the explicit global-local decomposition** vs window-level event-conditioned modelling in general.

Three new deterministic models, each matched to C0's parameter count, answer one question each (§5). C0-A is an
ablation study, not a new main model: no C0 file, checkpoint or metric was changed.

## 2. Frozen C0 result

From `docs/C0_COHERENTBEAT_FEASIBILITY_REPORT.md` (ARCH-HOLDOUT, 434 patients, 28,531 windows, evaluated once after the
freeze `dddaa4b`):

| gate | C0 − comparator | result |
|---|---|---|
| G1 rhythm (C0 − placed events) | ΔF1 −0.0010 [−0.0013, −0.0007]; ΔRR-MAE +0.0021 [−0.0005, +0.0055] ms | PASS |
| G2 false R / window (C0 − BF0-DET-RETRAIN) | −0.205 [−0.215, −0.195] (0.445 vs 0.650) | PASS |
| G3 FD (C0 − BF0-DET-RETRAIN) | −8.63 [−9.85, −7.50] (25.10 vs 33.74) | PASS |
| G4 beat correlation (C0 − BF0-DET-RETRAIN) | −0.0038 [−0.0048, −0.0028] (0.8222 vs 0.8260) | PASS |

Boundary-near false detections: C0 1 of 12,761. **Verdict: STRONG.** C0-A does not revisit this verdict.

## 3. Why attribution is still needed

C0's report (§22) listed what C0 did not establish. Three of those points are architectural:

- **Capacity.** C0 has 592,770 parameters and BF0-DET 482,049 (1.23 ×).
- **LOCAL-ONLY confound.** Setting g = 0 removed the time-varying field *and* the absolute baseline (the data sit near
  −0.5; LOCAL-ONLY is forced to 0 between supports). Its failure showed that something must carry the window context,
  not that the context must vary in time.
- **Decomposition vs window-level conditioning.** C0 sees the whole window and is trained on whole windows. A direct
  whole-window predictor with the same PPG and event input had never been compared at matched size.

A fourth confound from C0's report, window- vs beat-level training, is only partly addressed: PM-BF0-DET is still
trained on beats, so M1 compares training unit and architecture together (§20).

## 4. Evidence status / previously opened holdout

- **ARCH-VAL** (433 patients, 28,649 windows) is the development and qualification population for C0-A. All claim
  decisions are made here.
- **ARCH-HOLDOUT** (434 patients, 28,531 windows) **was already opened once, by C0** (`f9e25b5`). For C0-A it is a
  **frozen secondary replication population**: the three new models had never been evaluated on it, and it was opened
  for them exactly once, after the committed freeze `479f138`, with the same code. It is **not fresh, not untouched by
  the project, and not independent prospective validation** for C0-A.
- The primary prospective C0 result remains C0's own evaluation.
- The C0-A loader refused ARCH-HOLDOUT until the freeze manifest was committed and every hashed file unchanged; the
  synthetic dry run checked that refusal (`audit.md` §5).

## 5. Ablation designs

All arms render at **one shared event sequence per window**: C0's frozen detector events, re-verified bit for bit at
evaluation (28,649 / 28,649 ARCH-VAL windows; 2,158 windows have no event).

| arm | what it is | role |
|---|---|---|
| placed events | the detector's event sequence scored directly as R peaks | rhythm reference |
| BF0-DET-RETRAIN (frozen) | BF0's deterministic BeatFlowNet (width 64), 166-sample absolute-level beats, Hann overlap-add + renormalization, template baseline for windows without events | C0's original comparator |
| C0-LOCAL-ONLY (frozen) | C0 with g = 0 | context only (§12) |
| CoherentBeat-C0 (frozen) | g(t) = 17-coefficient cubic B-spline + Σ m_i·q_i (compact C^∞ bump supports) | reference model |
| **PM-BF0-DET** (new) | BF0-DET with channel width 73; everything else BF0 | M1: capacity |
| **CONST-GLOBAL-LOCAL** (new) | C0 with g(t) = a_global, one scalar per window (Linear 64→64 → GELU → Linear 64→1 on the encoder mean); encoder, local branch, supports, bump and forward inherited unchanged | M2: time-varying field; carrier |
| **WW-DET** (new) | C0's encoder (64 ch, 8 blocks) → 1 × 1 conv on [features, event raster] → 5 residual blocks (dilations 1, 2, 4, 8, 16) → 1 × 1 head → 512 samples. Raster: max of Gaussians, σ = 20 ms (detector target convention). No decomposition, supports or stitching | M3: decomposition |

Unit tests (`tests/test_c0a_ablation.py`, 15) check, among others, that CONST's global output is exactly constant in time
with no hidden time-varying path and C0's local branch, that WW-DET reads the raster, outputs 512 samples and has no
local renderer, and that no stochastic path is active.

## 6. Parameter matching

Parameter count was the only selection criterion, fixed before training (`parameter_match.json`).

| model | search grid | selected | params | vs C0 (592,770) | tolerance |
|---|---|---|---|---|---|
| PM-BF0-DET | BeatFlowNet width 32 … 128 | width 73 | 599,445 | +1.13% | ±2% ✓ (width 72: 585,761, −1.18%) |
| CONST-GLOBAL-LOCAL | none | — | 592,770 | 0.00% | the scalar head has exactly the layers of C0's coefficient head |
| WW-DET | decoder width {32 … 96 step 8} × depth 1 … 12 | width 72, depth 5 | 593,577 | +0.14% | ±5% ✓ |

The models are **parameter-matched, not compute-matched** (FLOPs and latency in §14).

## 7. Training protocol

Each new model was trained **once** on ARCH-TRAIN (3,470 patients), seed 42, last checkpoint, no early stopping, no
learning-rate or architecture change after any result.

| model | protocol | steps × batch | training time (RTX 5090) | peak GPU memory | last logged L1 | NaN steps |
|---|---|---|---|---|---|---|
| PM-BF0-DET | BF0-DET: beats at reference R, L1 | 20,000 × 256 beats | 148.2 s | 1,533 MiB | 0.2485 (beat-level) | 0 |
| CONST-GLOBAL-LOCAL | C0: full-window L1, events = reference R | 20,000 × 64 windows | 144.3 s | 1,377 MiB | 0.2563 | 0 |
| WW-DET | C0: full-window L1, raster at reference R | 20,000 × 64 windows | 130.8 s | 1,836 MiB | 0.2449 | 0 |
| *C0 (frozen, for reference)* | | 20,000 × 64 windows | 145.0 s | 1,390 MiB | 0.253 | 0 |
| *BF0-DET (frozen)* | | 20,000 × 256 beats | 101.2 s | 1,466 MiB | 0.249 (beat-level) | 0 |

AdamW lr 1e-3, weight decay 0.01, gradient clip 1.0 for every model. "Last logged L1" is the mean over the final 1,000
steps. The ARCH-VAL commit message (`22c6ddb`) quoted stage wall-clock times (158 / 150 / 143 s, including data
loading); the training-loop times are the ones above.

## 8. ARCH-VAL results

Patient-clustered means (433 patients, 28,649 windows; 95% CIs in `val_metrics.json`). Frozen arms reproduced C0's
stored renders and metrics exactly (maximum absolute deviation 0.0, including the beat-correlation population).

| ARCH-VAL | params | F1 | FP / win | FN / win | RR-MAE (ms) | HR err (bpm) | FD | beat corr | MAE | PCC | spectral dev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| placed events | — | 0.7901 | 0.4025 | 1.0422 | 7.453 | 4.68 | — | — | — | — | — |
| BF0-DET-RETRAIN | 482,049 | 0.7765 | 0.6224 | 1.0205 | 7.607 | 5.67 | 32.72 | 0.8206 | 0.3134 | 0.332 | 2.51 |
| PM-BF0-DET | 599,445 | 0.7780 | 0.6010 | 1.0223 | 7.607 | 5.61 | 30.31 | 0.8217 | 0.3122 | 0.338 | 2.64 |
| C0-LOCAL-ONLY | 588,545 | 0.7552 | 0.7532 | 1.0371 | 7.772 | 10.80 | 39.00 | 0.7475 | 0.3933 | 0.134 | 2.87 |
| CONST-GLOBAL-LOCAL | 592,770 | 0.7889 | 0.4127 | 1.0455 | 7.448 | 4.51 | 25.96 | 0.8156 | 0.2921 | 0.330 | 2.06 |
| WW-DET | 593,577 | **0.7934** | 0.5302 | 1.0245 | 7.490 | 6.43 | **18.70** | **0.8213** | 0.2927 | **0.365** | **1.76** |
| CoherentBeat-C0 | 592,770 | 0.7890 | 0.4132 | 1.0453 | 7.448 | 4.52 | 23.70 | 0.8163 | **0.2911** | 0.349 | 2.00 |

- **[descriptive] Raw totals:** false R detections — placed 11,379, C0 11,684, CONST 11,671, WW-DET 14,994, PM-BF0
  17,087, BF0-DET 17,709, LOCAL-ONLY 21,343. Missed R — placed 29,627, C0 29,714, CONST 29,719, WW-DET 29,125.
- **[descriptive] S4 / S5:** BF0-DET 0.345 / 0.266, PM-BF0 0.346 / 0.266, LOCAL-ONLY 0.351 / 0.269, CONST 0.346 /
  0.266, WW-DET 0.348 / 0.267, C0 0.346 / 0.266.
- **[descriptive] Boundary-near false detections** (±2 samples of a support boundary): C0 2 of 11,684, CONST 0 of
  11,671.
- **Populations:** beat correlation uses the same 91,321 matched pairs for every arm (3,621 windows without a pair). HR
  error needs two detected R peaks in both output and reference, so its window set differs by arm (not evaluable: C0
  2,808, CONST 2,807, LOCAL-ONLY 2,806, BF0-DET 2,168, PM-BF0 2,174, WW-DET 1,513). HR error is therefore not a paired
  comparison across arms.

## 9. M1 capacity control [claim] — C0 vs PM-BF0-DET

| C0 − PM-BF0-DET (ARCH-VAL) | estimate [95% CI] | rule | |
|---|---|---|---|
| false R / window | −0.188 [−0.198, −0.178] | CI < 0 | ✓ |
| FD | −6.61 [−7.70, −5.63] | CI < 0 | ✓ |
| beat correlation | −0.0054 [−0.0065, −0.0043] | lower > −0.02 | ✓ |
| rhythm: C0 − placed, F1 | −0.0012 [−0.0015, −0.0009] | lower > −0.02 | ✓ |
| rhythm: C0 − placed, RR-MAE | −0.0002 [−0.0019, +0.0013] ms | upper < +2 ms | ✓ |
| [descriptive] F1 | +0.0110 [+0.0101, +0.0119] | | |
| [descriptive] MAE | −0.0212 [−0.0233, −0.0191] | | |
| [descriptive] RR-MAE | −0.028 [−0.041, −0.015] ms | | |

**M1: SUPPORTED.** The capacity-only explanation is disfavored. This does not say capacity has no effect.

- **[descriptive] PM-BF0's own rhythm:** PM − placed ΔF1 −0.0122 [−0.0131, −0.0113], about ten times C0's loss. The
  wider beat model keeps BF0's failure pattern (FP 0.601 vs BF0-DET 0.622).
- **[post-hoc]** As point estimates, widening BF0-DET to C0's size closes about 27% of the C0 − BF0-DET FD gap
  (32.72 → 30.31 of 32.72 → 23.70) and about 10% of the FP gap (0.622 → 0.601 of 0.622 → 0.413). PM − BF0-DET has no
  CI and includes training-run variation (one run each).

## 10. M2 time-varying global field [claim] — C0 vs CONST-GLOBAL-LOCAL

| C0 − CONST (ARCH-VAL) | estimate [95% CI] | rule | |
|---|---|---|---|
| FD | −2.27 [−2.45, −2.09] | CI < 0 | ✓ |
| F1 | +0.0000 [−0.0002, +0.0002] | disadvantage if lower ≤ −0.02 | no disadvantage |
| beat correlation | +0.0007 [−0.0001, +0.0015] | disadvantage if lower ≤ −0.02 | no disadvantage |
| [descriptive] false R / window | +0.0005 [−0.0006, +0.0014] | | CI includes 0 |
| [descriptive] MAE | −0.0011 [−0.0016, −0.0005] | | CI < 0 |
| [descriptive] RR-MAE | −0.0022 [−0.0050, −0.0004] ms | | |

**M2: SUPPORTED.** On top of a shared absolute level, the time-varying spline lowers FD by 2.3 with no event or
morphology cost.

- **[descriptive]** CONST keeps the placed rhythm as well as C0 does (CONST − placed ΔF1 −0.0012 [−0.0015, −0.0009],
  ΔRR +0.0020 [−0.0001, +0.0048] ms), and its false detections equal C0's.
- **[post-hoc] Effect size.** The step from LOCAL-ONLY to C0 is FD 39.00 → 23.70. The constant level alone takes it
  to 25.96, about 85% of the step; the time-varying field adds the remaining 2.27, about 15%. For false detections the
  whole improvement over LOCAL-ONLY comes from the constant level (§12). The M2 effect is real on ARCH-VAL but modest,
  and it rests on one training run per model (§20).

## 11. M3 decomposition vs whole-window model [claim] — C0 vs WW-DET

| C0 − WW-DET (ARCH-VAL) | estimate [95% CI] | rule | |
|---|---|---|---|
| false R / window | **−0.117 [−0.130, −0.105]** | CI < 0 | ✓ C0 better |
| FD | **+5.00 [+4.47, +5.58]** | CI < 0 | ✗ CI entirely > 0: WW-DET better |
| beat correlation | −0.0050 [−0.0063, −0.0037] | lower > −0.02 | non-inferior |
| [descriptive] F1 | −0.0044 [−0.0052, −0.0037] | | WW-DET better |
| [descriptive] MAE | −0.0016 [−0.0031, −0.00001] | | C0 better |
| [descriptive] RR-MAE | −0.0069 [−0.0161, −0.0005] ms | | |

- **STRONG** needs both FP and FD CIs below 0: not met.
- **PARTIAL** needs one of them below 0 and the other's CI not entirely above 0: the FD CI is entirely above 0, so not
  met.

**M3: NOT SUPPORTED** (the preregistered "otherwise" branch). The result is a **trade-off**, not a tie:

- C0 makes 0.117 fewer false R detections per window. WW-DET's 0.530 is 28% more than C0's 0.413.
- WW-DET has lower FD (18.70 vs 23.70), slightly higher F1, beat correlation and PCC, and lower spectral deviation.

The specification's informal description of NOT SUPPORTED ("WW-DET matches or exceeds C0 on both axes") does not
describe this outcome literally. The frozen preregistration (§4, "otherwise NOT SUPPORTED") is the rule applied.

**[descriptive] WW-DET is not confined to the placed events.**

- Its F1 is above the placed events' (WW − placed ΔF1 +0.0032 [+0.0026, +0.0039]).
- It misses fewer reference beats than the event sequence does (29,125 vs 29,627).
- It also adds false ones: 14,994 vs 11,379 for the events and 11,684 for C0.
- HR is evaluable for WW-DET in all but 1,513 windows, but 2,158 windows received no event. So WW-DET's output has two
  or more detected R peaks in at least 645 windows that were given no event.

C0, by construction, adds compact structure only inside supports around the placed events. Its F1 and false detections
track the events' (ΔF1 −0.0012; +0.011 false R per window).

**[post-hoc reading, untested]**

- M3 therefore compares a renderer **locked to the events** with a model that uses the raster as a **soft hint** and
  also re-decides beats from the PPG.
- That would explain why the two axes move in opposite directions.
- Part of WW-DET's FD advantage may come from windows with few or no events, where C0 can only draw its smooth global
  field. This was not analysed and is not claimed.

**Case reading.** This is **Case C** (M3 not supported). It is not Case E: the preregistration does not operationalize
"clearly dominates", and WW-DET is clearly worse on false detections. Either case gives the same C1 decision (§19).

## 12. LOCAL-ONLY reinterpretation [claim: shared absolute context carrier]

| CONST − LOCAL-ONLY (ARCH-VAL) | estimate [95% CI] | rule | |
|---|---|---|---|
| FD | −13.04 [−16.44, −9.70] | CI < 0 | ✓ |
| false R / window | −0.340 [−0.367, −0.312] | CI < 0 | ✓ |
| [descriptive] F1 | +0.0337 [+0.0305, +0.0369] | | |
| [descriptive] beat correlation | +0.0682 [+0.0624, +0.0746] | | |

**Shared absolute context carrier: SUPPORTED.**

- One learned scalar level per window removes LOCAL-ONLY's whole false-detection excess. CONST's FP of 0.413 equals
  C0's.
- That scalar level also closes most of LOCAL-ONLY's FD gap.
- LOCAL-ONLY's failure in C0 was mainly the missing carrier, as C0's report suspected. LOCAL-ONLY is used here only as
  context, never as evidence for the time-varying field. That evidence is M2.

(The ARCH-VAL commit message `22c6ddb` rounded this ΔFP to −0.341. The stored value is −0.3405.)

## 13. ARCH-HOLDOUT frozen secondary replication

**Frozen secondary replication on the previously opened C0 holdout.**

- **Procedure.** `evaluate_holdout` ran once, starting after the freeze commit `479f138` had been pushed.
  - Its loader re-verified the 27 frozen hashes and that the manifest was committed.
  - The detector re-produced C0's stored holdout events.
  - Frozen arms reproduced C0's stored holdout renders and metrics exactly (maximum deviation 0.0).
  - After the run, all 27 C0-A and 19 C0 frozen files were re-hashed unchanged, and the three new checkpoints match
    `checkpoint_hashes.json`.
  - Nothing was modified after either evaluation. VAL and HOLDOUT were not pooled.
- **Population:** 434 patients, 28,531 windows. 1,993 windows have no event. 90,782 matched beat pairs (3,632 windows
  without a pair).

| ARCH-HOLDOUT | params | F1 | FP / win | FN / win | RR-MAE (ms) | HR err (bpm) | FD | beat corr | MAE | PCC | spectral dev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| placed events | — | 0.7888 | 0.4355 | 1.0140 | 7.815 | 4.29 | — | — | — | — | — |
| BF0-DET-RETRAIN | 482,049 | 0.7753 | 0.6498 | 0.9919 | 7.927 | 5.28 | 33.74 | 0.8260 | 0.3187 | 0.332 | 2.60 |
| PM-BF0-DET | 599,445 | 0.7766 | 0.6311 | 0.9938 | 7.945 | 5.21 | 31.21 | **0.8268** | 0.3180 | 0.338 | 2.71 |
| C0-LOCAL-ONLY | 588,545 | 0.7538 | 0.7893 | 1.0083 | 8.020 | 10.76 | 41.66 | 0.7540 | 0.3986 | 0.139 | 3.15 |
| CONST-GLOBAL-LOCAL | 592,770 | 0.7877 | 0.4456 | 1.0167 | 7.816 | 4.15 | 27.27 | 0.8214 | 0.2974 | 0.332 | 2.19 |
| WW-DET | 593,577 | **0.7917** | 0.5537 | 0.9974 | 7.850 | 5.93 | **20.01** | 0.8263 | 0.3000 | **0.366** | **1.93** |
| CoherentBeat-C0 | 592,770 | 0.7878 | 0.4450 | 1.0165 | 7.787 | 4.15 | 25.10 | 0.8222 | **0.2965** | 0.350 | 2.13 |

**Claim effects (C0 − ablation unless stated; 2,000 patient-clustered replicates, seed 20261001):**

| effect | ARCH-VAL | ARCH-HOLDOUT | holdout decision |
|---|---|---|---|
| M1 FP (C0 − PM-BF0) | −0.188 [−0.198, −0.178] | −0.186 [−0.196, −0.176] | |
| M1 FD | −6.61 [−7.70, −5.63] | −6.11 [−7.22, −5.11] | |
| M1 beat corr | −0.0054 [−0.0065, −0.0043] | −0.0046 [−0.0056, −0.0035] | |
| M1 rhythm (C0 − placed) F1 / RR-MAE | −0.0012 / −0.0002 ms | −0.0010 [−0.0013, −0.0007] / +0.0021 [−0.0005, +0.0055] ms | **SUPPORTED** |
| M2 FD (C0 − CONST) | −2.27 [−2.45, −2.09] | −2.17 [−2.34, −1.99] | |
| M2 F1 / beat corr | +0.0000 / +0.0007 | +0.0001 [−0.0001, +0.0002] / +0.0008 [−0.00004, +0.0017] | **SUPPORTED** |
| M2 [descriptive] FP / MAE | +0.0005 / −0.0011 | −0.0006 [−0.0015, +0.0004] / −0.0009 [−0.0015, −0.0004] | |
| M3 FP (C0 − WW-DET) | −0.117 [−0.130, −0.105] | −0.109 [−0.120, −0.099] | |
| M3 FD | +5.00 [+4.47, +5.58] | +5.10 [+4.63, +5.56] | |
| M3 beat corr | −0.0050 [−0.0063, −0.0037] | −0.0041 [−0.0052, −0.0030] | **NOT SUPPORTED** |
| M3 [descriptive] F1 / MAE | −0.0044 / −0.0016 | −0.0040 [−0.0045, −0.0033] / −0.0035 [−0.0046, −0.0025] | |
| carrier FD (CONST − LOCAL-ONLY) | −13.04 [−16.44, −9.70] | −14.39 [−18.27, −10.68] | |
| carrier FP | −0.340 [−0.367, −0.312] | −0.344 [−0.371, −0.315] | **SUPPORTED** |

**Replication:** every primary point estimate has the same sign on both populations, and every decision is the same.
So M1, M2, M3 and the carrier are each **REPLICATED** (`replication_summary.json`).

**[descriptive] Holdout details:**

- **CONST − placed rhythm:** ΔF1 −0.0011 [−0.0014, −0.0008]; ΔRR +0.0018 [−0.0008, +0.0052] ms.
- **PM-BF0 − placed:** ΔF1 −0.0122 [−0.0131, −0.0113].
- **WW-DET − placed:**
  - ΔF1 +0.0030 [+0.0024, +0.0035].
  - Missed R: 28,374 vs 28,845 for the event sequence.
  - False R: 15,814, against 12,491 for the events and 12,761 for C0.
  - HR is evaluable in all but 1,422 windows, while 1,993 windows had no event. So WW-DET draws two or more detected R
    peaks in at least 571 windows that were given no event (VAL: at least 645).
- **Boundary-near false detections:** C0 1 of 12,761; CONST 0 of 12,778.
- **[post-hoc] Effect shares:**
  - The constant level closes about 87% of the LOCAL-ONLY → C0 FD step (41.66 → 27.27 → 25.10; VAL 85%).
  - Width-matching BF0 closes about 29% of the BF0-DET → C0 FD gap and 9% of the FP gap (VAL 27% and 10%).
  - These shares have no CI.

## 14. Compute / latency / parameter counts

| | BF0-DET | PM-BF0-DET | LOCAL-ONLY | CONST | WW-DET | C0 |
|---|---|---|---|---|---|---|
| parameters | 482,049 | 599,445 | 588,545 | 592,770 | 593,577 | 592,770 |
| vs C0 | −18.7% | +1.13% | −0.71% | 0.00% | +0.14% | — |
| training time (RTX 5090) | 101.2 s | 148.2 s | 141.1 s | 144.3 s | 130.8 s | 145.0 s |
| peak GPU memory (training) | 1,466 MiB | 1,533 MiB | 1,377 MiB | 1,377 MiB | 1,836 MiB | 1,390 MiB |
| batch-1 GPU latency, median (P90) | 1.70 (1.79) ms | 1.79 (2.00) ms | 1.88 (2.36) ms | 1.90 (1.95) ms | **1.29 (1.32) ms** | 1.94 (2.00) ms |
| batch-1 CPU latency, median (P90) | 5.13 (6.64) ms | 5.55 (7.67) ms | 5.63 (9.73) ms | 5.54 (8.30) ms | 5.41 (5.86) ms | 5.60 (6.01) ms |
| FLOPs, one window with 4 events, incl. detector | 772.3 M | 903.5 M | 801.0 M | 801.0 M | 941.6 M | 801.2 M |

- **Latency:** detector plus waveform model, batch 1, on 50 salted ARCH-VAL windows after 3 warm-up runs; CPU uses 4
  threads.
- **FLOPs:** from `torch.utils.flop_counter` on one salted window with 4 events (CPU). This is a lower bound. Models that
  render per event (BF0, PM, LOCAL, CONST, C0) scale with the event count; WW-DET does not.
- **Training:** no NaN step in any model. Every model was trained once.
- **The models are parameter-matched, not compute-matched.**
  - On this window WW-DET uses 18% more FLOPs than C0, and PM-BF0 13% more.
  - WW-DET is still the fastest on GPU: one dense pass with no per-event gather.
- **Re-measured latencies:** C0 and BF0-DET were measured again here (1.94 / 1.70 ms GPU). They differ slightly from
  C0's report (1.90 / 1.66 ms), which was a separate measurement run.

## 15. Architecture claim matrix

| claim | ARCH-VAL | ARCH-HOLDOUT (secondary) | replication | final entry |
|---|---|---|---|---|
| M1 capacity control | SUPPORTED | SUPPORTED | REPLICATED | capacity-only explanation: **DISFAVORED** |
| shared absolute context carrier | SUPPORTED | SUPPORTED | REPLICATED | **SUPPORTED** |
| M2 time-varying global field | SUPPORTED | SUPPORTED | REPLICATED | **SUPPORTED** |
| M3 explicit global-local decomposition | NOT SUPPORTED | NOT SUPPORTED | REPLICATED | **NOT SUPPORTED** |

**Case C** (preregistration §6): M3 is not supported.

- No claim was downgraded: none was NOT REPLICATED.
- C0's own STRONG verdict against BF0-DET-RETRAIN is unaffected. It answered a different question.

Figure: `figure.png`.

- Panel A: schematics.
- Panel B: parameter counts.
- Panels C–E: FP, FD and beat correlation on both populations.
- Panel F: ARCH-VAL vs ARCH-HOLDOUT relative effects with CIs.

Compact table: `table.csv`.

## 16. What mechanism is supported

These findings hold on VitalDB, for deterministic models trained once each, rendered at the same frozen detector events.
They held on ARCH-VAL and replicated on the previously opened ARCH-HOLDOUT.

1. **C0's gain over the BF0 beat renderer is not explained by parameter count alone (M1).**
   - A BF0 beat model widened to C0's size (+1.1% parameters) keeps most of BF0's excess false detections and FD.
   - C0 − PM-BF0: false R −0.19 per window and FD −6.1 to −6.6, with beat correlation inside the margin.
2. **A shared absolute context carrier is the largest single ingredient (carrier).**
   - Add one learned scalar level per window to C0's compact local residuals.
   - That removes LOCAL-ONLY's excess false detections (−0.34 per window) and most of its FD gap (−13.0 / −14.4).
3. **A time-varying global field adds a smaller, consistent improvement (M2).**
   - The 17-coefficient spline lowers FD by a further 2.2–2.3 relative to the constant level.
   - It costs nothing in events or morphology.
4. **[descriptive, within M3] Rendering locked to the events has the best event fidelity.**
   - C0 and CONST reproduce the placed rhythm (ΔF1 about −0.001) and add few false R detections.
   - They have 0.11–0.12 fewer false R per window than the parameter-matched whole-window model. This is a property of
     the comparison, not a claim that the decomposition causes C0's gain.

## 17. What mechanism is NOT supported

- **That the explicit global-local decomposition is responsible for C0's gain (M3).**
  - A parameter-matched, event-conditioned whole-window deterministic predictor beats C0 on several metrics:
    - lower FD (18.70 vs 23.70 VAL; 20.01 vs 25.10 holdout);
    - slightly higher F1, beat correlation and PCC;
    - lower spectral deviation.
  - Its cost is more false R detections. Window-level event-conditioned deterministic modelling may explain much of the
    gain over beat-local rendering.
- **That CoherentBeat is the best deterministic design tested.** On FD, WW-DET is better.
- **That the time-varying spline is the main ingredient.** It is supported, but it carries about 13–15% of the
  LOCAL-ONLY → C0 FD step. The constant shared level carries the rest, and all of the false-detection improvement.
- **That capacity has no effect.** PM-BF0 improved slightly over BF0-DET as a point estimate (FD 32.72 → 30.31).
- **That architecture, rather than the training unit, explains M1.** C0 is window-trained and PM-BF0 beat-trained.
- **Anything else:**
  - that ARCH-HOLDOUT is fresh;
  - causality;
  - stochastic generation or uncertainty;
  - patient-specific morphology;
  - cross-dataset or multi-seed robustness;
  - clinical validity;
  - state of the art;
  - a first global-local architecture.

## 18. Implications for the paper

- **Do not headline CoherentBeat's global-local decomposition as the architectural contribution.** This is Case C: do not
  oversell CoherentBeat.
- **Defensible framing, following M1 / M2 / M3 / carrier exactly.**
  - Given the same detector events, a deterministic waveform model that is trained and conditioned at the window level,
    with a shared absolute waveform context, avoids the false R detections and FD penalty of beat-local rendering.
  - That gain is not explained by parameter count alone.
  - A time-varying shared field helps further, by a smaller amount.
  - Within window-level models there is a trade-off:
    - rendering locked to the events (CoherentBeat) gives the fewest false R detections and preserves the placed rhythm;
    - an event-conditioned whole-window predictor gives lower FD and slightly higher beat correlation.
- **Reporting.** If C0 is reported, report WW-DET and CONST beside it, on both axes.
- **The spline** may be described as a supported refinement on top of a shared context carrier, not as the core
  ingredient.
- **Unchanged:** C0's STRONG verdict against BF0-DET-RETRAIN, as stated in C0's report.

## 19. C1 go/no-go

**C1: NO-GO.**

- **Rule** (preregistration §7): GO requires both of these:
  - M1 SUPPORTED **and** M3 STRONG or PARTIAL on ARCH-VAL;
  - neither M1 nor M3 NOT REPLICATED on ARCH-HOLDOUT.
- **Outcome:** M3 is NOT SUPPORTED on ARCH-VAL, so the rule gives NO-GO. The holdout cannot change that. The
  specification adds "If M3 fails: C1 NO-GO as a CoherentBeat architecture paper".
- **The C1 note stays a note.** `docs/C1_SHARED_LATENT_DESIGN_DRAFT.md` remains a design note only; nothing from it is
  started.
- **Not done in this run:**
  - C1 training;
  - a stochastic latent;
  - enlarging CoherentBeat;
  - new losses;
  - rescue ablations;
  - old test or external data;
  - other seeds.

**HARD STOP.**

## 20. Limitations

1. **One training run per model (seed 42).** The bootstrap CIs cover patient sampling only, not training-run variance.
   Small effects could move with another seed: M2's −2.2 FD, and PM-BF0 vs BF0-DET. Multi-seed runs were excluded by
   design.
2. **The holdout is not fresh.** It is a secondary replication on a population C0 had opened once. Earlier project
   models also used these patients as training data (C0 report §3).
3. **Parameter-matched, not compute-matched.** FLOPs differ by up to +18% (WW-DET), and latency differs (§14).
4. **M1 does not separate training unit from architecture.** C0 is trained on windows and PM-BF0 on beats.
5. **M3 tests one whole-window design.**
   - The design: decoder width 72, depth 5, raster σ = 20 ms, all fixed by parameter count or convention.
   - Other whole-window designs could do better or worse.
   - WW-DET treats events as a soft raster; C0 treats them as hard support centres. The two therefore differ in how
     strictly they follow the events (§11).
6. **FD covers all windows.** That includes the windows without events (2,158 VAL; 1,993 holdout), where C0 draws only
   its global field and WW-DET may draw beats. How much of the FD gap comes from these windows was not analysed.
7. **HR error is not paired.** It uses a different window set per arm (§8).
8. **Scope:** deterministic models only, VitalDB only, 512-sample windows at 128 Hz.
9. **Case E is not operationalized** in the preregistration ("WW-DET clearly dominates"). It was read as not met,
   because WW-DET is worse on false detections. Cases C and E give the same C1 decision.
