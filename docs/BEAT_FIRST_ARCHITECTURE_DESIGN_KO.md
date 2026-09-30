# 비트 우선 생성기 (Beat-First, 가칭) — 아키텍처 설계 초안

작성일 2026-09-30 · 상태: **설계 초안 (사전등록 전)**. 이 구조의 결과는 아직 하나도 계산하지 않았다. 이름은 임시
이름이고 학계 용어가 아니다.

근거 문서: 1기 `docs/PROGRAM_SUMMARY_ALL_STAGES.md` (4부 O·R·E 시리즈, 8부 N 시리즈)와 각 보고서, 2기
`docs/RESULTS_MASTER.md`, 문헌 `docs/PAPER_RELATED_WORK_KO.md`, 앞선 설계 `docs/VTR_PIPELINE_DESIGN_KO.md`.
VitalDB 수치는 test 1,156명 기준 (`artifacts/paper_tables/numbers.json`)이다.

---

## 0. 한 문단 요약

지금의 PPG→ECG 생성기는 노이즈 하나로 타이밍과 형태를 한꺼번에 만든다. 그래서 샘플마다 비트 위치가 틀리고(샘플 하나의
R 피크 F1 0.674), 우리는 그 오류를 추론 단계의 합의로 고쳐 왔다(0.765). 그런데 우리 결과는 두 무작위성의 성격이 다르다고
말한다. 타이밍은 PPG에서 추정할 수 있고, 그 불확실성도 PPG에서 예측된다(1기에서 복제된 유일한 양성 결과). 반면 형태는
PPG에 없다. 비트 우선 생성기는 이 둘을 **구조에서 나눈다.** 타이밍은 판별 이벤트 단이 정하고 불확실성을 보정한다. 형태는
비트 하나 단위의 생성 모델이 모집단 분포에서 그린다. 그린 비트는 예측한 시각에 **놓는다.** 기대하는 결과는 네 가지다:
전용 검출기 수준의 비트 정확도, 생성기 수준의 파형 현실성, 포함률이 보정된 비트 시각 구간, 작은 계산량.

## 1. 설계 근거: 우리 결과 → 설계 결정

| 우리 결과 (근거) | 설계 결정 |
|---|---|
| 타이밍은 PPG에서 추정되고, 직접 검출기가 생성기보다 정확하다. RD1 F1 0.772 vs 샘플 하나 0.645–0.674 (VitalDB). 검출기 + 템플릿이 모든 학습 생성기를 이벤트 지표에서 이겼다 (N1, +0.487 vs +0.358) | 타이밍은 판별 이벤트 단이 정한다 |
| 비트별 타이밍 불확실성은 PPG에서 예측된다 (N7 sharpness +0.445 ± 0.021, 4 fold × 3 seed 복제). 하지만 과신한다 (포함률 0.424 / 0.688, 목표 0.50 / 0.80). 생성형 사후분포는 판별 헤드보다 나을 게 없었다 (N7 FM − HEAD = −0.212) | 이벤트 단에 판별 불확실성 헤드를 두고, 검증 환자로 conformal 보정을 해서 포함률을 맞춘다 |
| 형태는 PPG에 없다 (N2: 템플릿 +0.9062 vs 회귀기 +0.9057). 정답 위치에 템플릿을 찍으면 환자의 99.6 %에서 어떤 생성기보다 형태가 낫다 (VitalDB) | 형태는 모집단 분포에서 생성한다. 개인의 형태를 복원한다고 주장하지 않는다. 주장은 "실제 같은 박동 간 변이"까지 |
| 지금 생성기의 노이즈는 타이밍과 형태를 함께 흔든다. 샘플 하나 F1 0.674, 16개 합의 0.765 | 무작위성을 둘로 나눈다. 타이밍 무작위성은 보정된 이벤트 사후분포에서, 형태 무작위성은 비트 생성기에서 뽑는다 |
| 비트 하나의 누락·추가는 62.5 ms 지터보다 형태를 8.7배 더 망친다 (E1). 정답 좌표로 학습한 정준 생성기는 ±15.6 ms 지터만 견뎠다 (O3) | 파형의 시간축을 비트는 대신 비트를 "놓는다". 타이밍 오차가 형태를 망가뜨리지 않고 위치만 옮기게 한다 |
| 정답 좌표를 주면 정준 생성이 형태 상관을 0.10 → 0.84로 올린다 (O2c, 오라클 진단) | 비트 정준 좌표(R 피크 기준)에서 형태를 생성하는 것은 효과가 확인된 표현이다 |

## 2. 구조

```
PPG 창 (4 s, 128 Hz)
  │
  ▼
① 이벤트 단 E (판별): R 피크 확률장 p(t) → 후보 비트 i마다 존재 확률 π_i, 시각 중심 μ_i, 시각 폭 σ_i
  │
  ▼
② 보정: 검증 환자 단위 split conformal로 σ_i를 조정 → 비트 시각 구간이 명목 포함률(50 / 80 / 90 %)을 지키게
  │
  ▼  (렌더 r = 1 … R마다)
  ├─ 비트 존재 b_i ~ Bernoulli(π_i), 시각 t_i ~ N(μ_i, σ_i²)                     ← 타이밍 무작위성
  │
  ▼
③ 비트 형태 생성기 G (생성, flow matching): R 피크 기준 정준 창에서 비트 하나를 생성
     조건: 앞뒤 RR 간격, 그 비트 주변의 PPG 특징, 창 전체 PPG 임베딩         ← 형태 무작위성은 G의 노이즈만
  │
  ▼
④ 배치·연결: 생성한 비트를 t_i에 놓고 겹침-더하기(overlap-add)로 잇는다. QT 길이는 G가 RR 조건으로 조절
  │
  ▼
출력: 렌더 R개 (각각 실제 같은 ECG) · 비트 열과 비트별 시각 구간 · 이벤트 사후분포에서 계산한 HR·HRV · 기권 표시
```

| 구성요소 | 설계 | 재사용할 자산 |
|---|---|---|
| ① 이벤트 단 | RD1 구조(Global/Local-TCN, 328,897 파라미터) + 비트별 (π, μ, σ) 헤드. 학습 분할의 ECG R 피크를 정답으로 판별 학습 (추론 때는 PPG만) | `scripts/rd1_direct_rpeak.py`, `scripts/n5_timing_uncertainty.py`, `scripts/n7_marked_posterior.py` |
| ② 보정 | 검증 환자 단위 split conformal. 척도 하나(또는 σ 구간별 척도)만 맞추고 고정 | 새로 작성 |
| ③ 비트 생성기 | 정준 창(예: R 앞 250 ms ~ 뒤 500 ms) 위의 작은 flow matching 모델. FiLM으로 조건 주입. 1–2 NFE (iMF 방식) | 백본 `build_penguin_backbone`, iMF 학습 코드, 비트 창 추출 `src/ppg2ecg/evaluation/rpeaks.py` |
| ④ 배치 | 창 함수로 겹침-더하기. 창 경계의 반쪽 비트는 창 밖까지 생성한 뒤 자름 | 템플릿 찍기 `src/ppg2ecg/evaluation/stamping.py` (N1) |

계산량: 이벤트 단 1회 + 비트 수(4 s 창에 약 5개) × G. 렌더 R개를 뽑아도 G만 반복하므로 가볍다.

## 3. 주장하는 것과 주장하지 않는 것

- **주장:**
  1. 렌더링 ECG의 비트 정확도(F1, RR-MAE)가 전용 검출기 수준이고, 생성 기준선 전부보다 높다
  2. 파형 현실성(FD)이 생성 기준선 수준이고, 템플릿 찍기(N1)보다 낫다
  3. 비트 시각 구간이 명목 포함률을 지킨다. 어떤 기준선도 이 출력을 주지 못한다
  4. 계산량이 작다
- **주장하지 않음:** 개인의 비트 형태 복원(N2), 직접 모델보다 나은 심박수, 부정맥 판별.

## 4. 기준선 (모두 우리 VitalDB 분할에서)

- **생성 기준선:** PENGUIN (재현 완료), iMF·CD 샘플 하나와 16샘플 합의 (2기), PPGFlowECG와 RDDM (공개 코드로 우리 분할에서
  재학습), KANFlow (공개 코드를 찾지 못함. 2기에 KAN 모듈을 재구현한 적 있음), Cho et al. 2026 (R 피크 예측기를 둔 PPG 확산
  모델, 같은 4초 창. 코드 공개 문장이 없어 재구현 필요할 수 있음), CardioGAN (선택)
- **비생성 기준선:** RD1 검출기, DB1 회귀기, 검출기 + 템플릿 (N1 방식), PPG 피크 세기
- **ablation:**
  - G 대신 고정 템플릿 (= N1)
  - 불확실성 없이 μ만 사용 / conformal 보정 없이
  - 이벤트 단 대신 생성 샘플의 합의 투표 (2기 방식) 또는 이종 투표 (VtR)
  - 비트를 놓는 대신 시간축을 비트는 렌더링 (O2c 방식)

## 5. 지표

- **비트:** R 피크 F1 (±50 ms), RR-MAE, 누락·가짜 비율, HR MAE, HRV (SDNN·RMSSD, 2분 블록)
- **불확실성:** 비트 시각 구간의 포함률 @50 / 80 / 90 %, 구간 폭, CRPS
- **파형:** FD (T3과 같은 구현), 구조 지표 S4·S5 (QRS 핵심 미분 RMSE·곡률 오차), 창 전체 상관
- **계산량:** 파라미터, CPU·GPU 지연
- **규약:** 우리 규약(주) + KANFlow식 품질 필터 규약(부)

## 6. 사전등록할 가설 초안

| 가설 | 내용 | 역할 |
|---|---|---|
| H1 | 렌더링 ECG의 비트 F1이 가장 강한 생성 기준선보다 높고 (CI > 0), FD는 그 기준선의 1.25배 이하 | 주 가설 후보 |
| H2 | 보정 후 비트 시각 구간의 포함률이 50 / 80 / 90 %에서 명목값 ±0.05 안 | 주 가설 후보 |
| H3 | 같은 이벤트 단을 쓰는 검출기 + 템플릿(N1)보다 FD가 낮다 (CI > 0) | 이차 |
| H4 | HR·HRV가 이벤트 단(RD1) 단독과 비열등 | 이차 |

실패 시 결론: H1이 실패하면 "비트 우선 분해는 생성 기준선 대비 이득이 없다", H3이 실패하면 "생성 비트는 템플릿보다
현실적이지 않다"를 그대로 보고한다.

## 7. 1기 실패와 무엇이 다른가

| 1기 시도 | 무엇이 문제였나 | 이 설계에서의 차이 | 남는 위험 |
|---|---|---|---|
| R2·R3: 연속 파형 생성기에 비트 뼈대 주입 | 이벤트 이득 +0.019·+0.041로 작고 QRS 구조 악화 | 생성기에 뼈대를 주입하지 않는다. 타이밍은 이벤트 단이 전부 정하고 생성기는 비트 모양만 그린다 | 아래 N1 경고 |
| O2c·O3: 정답 좌표로 학습한 정준 생성기 | ±15.6 ms 넘는 지터에 무너짐 | 시간축을 비틀지 않고 비트를 놓으므로, 생성기는 타이밍 오차를 보지 않는다 | 비트 사이 연결 부분의 인공물 |
| N1: 검출기 + 템플릿 | 이벤트 지표는 최고지만 형태가 없고 창 상관 −0.003 | 템플릿 대신 박동마다 다른 생성 비트 | "N1에 변이만 더한 것"이라는 비판 |

**N1 보고서 §3.3의 경고는 이 설계에도 그대로 적용된다.** 구조 지표 S4·S5는 채점하는 신호에서 검출한 피크 위치에서
계산된다. 그래서 날카로운 비트를 조금 어긋난 자리에 놓으면 구조 점수가 나빠진다. 비트를 많이 놓는 방법은 모두 이 비용을
치른다. 이 설계도 이벤트 이득과 구조 비용을 맞바꿀 가능성이 크다. 따라서 구조 지표 개선은 주장하지 않고 그대로 보고한다.

## 8. 자산과 비용

| 항목 | 상태 | 예상 비용 |
|---|---|---|
| 이벤트 단 (RD1 구조, VitalDB 체크포인트) | 있음. 불확실성 헤드는 N5·N7 코드에서 이식 | 학습 1분 내외 × seed 3 |
| 비트 생성기 G | 새로 작성. 학습 분할에서 비트 약 140만 개 (288,400창 × 약 5비트) | 학습 수십 분–수 시간 × seed 3 |
| conformal 보정, 렌더링 | 새로 작성 | 작음 |
| 외부 생성 기준선 재학습 (PPGFlowECG 214 M, RDDM 145 M) | 공개 코드 있음 (우리 저장소 `external/`에 고정된 사본). 우리 분할로 다시 학습해야 공정 | 가장 큰 비용: 모델당 수일 가능. GPU는 다른 사용자 작업 확인 후 |
| 평가 (FD, S4·S5, 이벤트 지표) | 기존 구현 재사용 | 작음 |

## 9. 위험

- **이벤트 단이 천장이다.** VitalDB 검출기 F1 0.772, WildPPG에서 같은 구조(R1) 0.62. 비트 정확도는 이보다 좋아질 수 없다.
- **구조 지표 맞바꿈** (§7 N1 경고).
- **창 전체 상관은 0 근처일 것이다.** 형태를 복원하지 않기 때문이다 (N1 −0.003). 이 지표로는 기준선에 질 수 있다.
- **비트 연결 인공물:** 겹침-더하기 경계, 창 경계의 반쪽 비트, PR·QT 길이.
- **"N1 + 변이"라는 비판:** 새로움은 구조화된 무작위성(타이밍과 형태의 분리)과 보정된 비트 시각 구간에서 나와야 한다.
- **외부 기준선 재학습:** 비용이 크고, 원 논문 수치가 재현되지 않을 수 있다.
- **현실성의 가치:** 표시·판독 용도에 기댄다. 우리 데이터로는 판독 효용을 직접 보이지 못한다.

## 10. 선행 연구와 새로움

조사일 2026-09-30. ★ = 원문을 직접 열어 확인, ☆ = 조사 에이전트가 원문을 열어 확인했다고 보고 (재확인 안 함).

**판단:** 세 단계를 하나로 묶은 PPG→ECG 논문은 찾지 못했다. 세 단계란 PPG만으로 R 시각과 비트별 불확실성을 예측하고,
R 기준 좌표에서 생성형 형태 모델을 돌리고, 예측 시각에 비트를 놓는 것이다. 하지만 **구성 요소는 모두 선행 연구가 있다.**
새로움은 조합과 아래 차별점에서 나와야 한다.

| 주장 | 근거 (원문 인용) | 출처 |
|---|---|---|
| 구조가 가장 가까운 연구는 레이더→ECG다. R 앵커, 주기 길이, 단일 주기 형태를 따로 예측해 조각을 앵커에 맞춘다. 형태 모델은 결정적(ODE 사전지식)이고 불확실성은 없다 | "the three outputs can directly form the long-term ECG recovery by aligning the recovered ECG pieces (Task 1) with the predicted anchors (Task 2) after resampling the ECG pieces as the cycle lengths (Task 3)." / "because an ODE model is introduced in the ODE decoder to provide morphological feature as the prior knowledge to guide/constrain the ECG recovery." | ★ radarODE-MTL, arXiv 2410.08656v2 (IEEE TIM 2025는 ☆) |
| radarODE-MTL은 주기 길이 추정이 틀리면 형태가 좋아도 파형이 어긋나고, R을 맞추면 다른 파의 정확도가 떨어질 수 있다고 쓴다 | "misaligned with ground truth due to inaccurate PPI estimation [5], deteriorating the RMSE/PCC even if the morphological features are well-recovered" / "the alignment of the R peak may degrade the accuracy of other peaks" | ☆ 같은 논문 |
| PPG→ECG에서 내부 R 피크 예측기를 둔 확산 모델이 이미 있다. 우리와 같은 4초 창(512점)이고 피험자 분리다. 다만 창 단위 생성이고 코드 공개 문장은 없다 | "An R peak predictor, connected to the U-Net bottleneck, estimates R peaks incorporated into the decoder." / "4-second non-overlapping window, resulting in samples containing 512 data points each" / "randomly split subjects into training and test sets at an 8:2 ratio to make the model subject-independent" | ★ Cho et al., BioData Mining 19:48 (2026), Europe PMC 전문 |
| PPG 기반 2단계(RR 계열 추정 → RR 조건 ECG 생성) 발표가 있고, 비트별 미세 타이밍은 향후 과제로 남겼다 | "脈波からRR系列を推定するStage Aと、推定されたRR系列を条件としてECG波形を生成するStage Bからなる二段階構造を設計しました。" | ☆ FastNeura 보도자료 (JSAI 2026), 2차 출처·단일 출처, 논문 원문 미확인 |
| 비트 단위 PPG→ECG 사상은 오래됐지만 결정적이고, 분할·정렬에 ECG R 피크를 쓴다 | Zhu 2019: "R2R: we segment the signal according to the location of the R peak of the ECG signal…" / Zhu 2021: "assuming the availability of the ground truth cardiac cycle information obtained from the ECG signal" / Abdelgaber 2023: "Then, we aligned ECG R peaks and PPG peaks" | ☆ Zhu et al. (BHI 2019, IoT-J 2021), Tian et al. (IoT-J 2023), Li et al. (IEEE TAI 2024), Abdelgaber et al. (Information 2023) |
| Zhu 2021에서 PPG만으로 분할하면(O2O) R에 다시 맞추기 전 상관이 크게 떨어진다 | "ρ = 0.510 when using O2O segmentation without the peak alignment … and ρ increases to 0.823 once the peak is aligned." | ☆ Zhu et al. 2021 (bioRxiv 815258) |
| RR을 조건으로 하는 비트 단위 확산 모델이 있다 (PPG 조건과 배치 단계는 없음) | "heartbeats over the retained leads conditionally on the patient characteristics P := (A, S, RR)" | ☆ BeatDiff (NeurIPS 2024), 코드 공개 |
| R 피크 열을 먼저 만들고 그것을 조건으로 형태를 생성하는 ECG 합성 모델이 있다 | "Xpeak is the R-peak train i.e. a binary-valued signal with 1's at R-peak locations else 0's." | ☆ CardiacGen (ML4H 2022), 단일 출처 |
| 리듬(RR 타코그램)과 형태를 따로 정하는 합성은 고전이다 | "The operator can specify the mean and standard deviation of the heart rate, the morphology of the PQRST cycle, and the power spectrum of the RR tachogram." | ☆ McSharry et al., IEEE TBME 50(3), 2003 (ECGSYN) |
| PPG의 불확실성은 창 단위 HR 분포로 다룬 연구가 있다 (비트 단위는 아님) | "We derive a distribution over possible heart rate values for a given PPG signal window" | ☆ BeliefPPG (UAI 2023) |
| 사이클 정렬 방식에 대한 비판: PTT·HRV 같은 시간 정보를 잃는다 | "the cycle-wise alignment and segmentation in the preprocessing stage lose temporal information, such as pulse transit time and heart rate variation" | ☆ Chiu et al., IEEE Sensors J. 2020 |
| 비트 기반 방법의 정확도는 R·onset 검출 정확도에 달려 있다 | "the accuracy of these two methods depends on the accuracy of extraction algorithms for R waves in an ECG and systolic peak (or onset) in a PPG" | ☆ Tang et al., Front. Physiol. 2022 |
| 이어 붙일수록 정렬 오차가 쌓인다 | "the cycles in the concatenation become increasingly temporally misaligned" | ☆ Slapničar et al., Sensors 24(7), 2024 |

**이 설계에서 새롭다고 주장할 수 있는 것 (작성자 해석)**
1. **추론 때 PPG만 쓰는 비트 단위 설계.** 기존 PPG 비트 단위 연구는 테스트에서도 ECG R 피크로 분할·정렬하거나 피험자별로
   학습했다. 우리는 이벤트 단이 PPG에서 R 시각을 직접 예측한다.
2. **보정된 비트별 타이밍 불확실성을 생성에 쓰는 것.** 조사한 범위에서 비트별 타이밍 불확실성을 생성기 입력이나 렌더링에
   쓰는 설계는 찾지 못했다. 불확실성 연구는 창 단위 HR 분포(BeliefPPG)였다. conformal 보정으로 포함률을 맞추는 것도 해당.
3. **타이밍 무작위성과 형태 무작위성의 분리.** Cho 2026과 FastNeura는 타이밍을 먼저 추정하지만 생성은 창 단위라, 생성기
   노이즈가 여전히 타이밍을 흔들 수 있다.
4. **평가:** 직접 검출기, 검출기 + 템플릿(N1)과의 비교, 연속 파형 상관과 R 타이밍 오차 보고.

**새롭다고 쓰면 안 되는 것:** 비트 단위 사상과 연결(Zhu, Tian, Li, Abdelgaber), 앵커 예측 후 조각 배치(radarODE-MTL), RR 조건
비트 확산(BeatDiff), R·RR 조건 생성(CardiacGen, Cho 2026, FastNeura), 리듬 → 형태 합성(McSharry).

**설계에 반영할 점**
- Cho et al. 2026은 우리와 같은 4초·512점 창을 쓰는 가장 가까운 PPG 모델이다. 기준선 후보로 추가한다(코드 공개 문장이 없어
  재구현이 필요할 수 있음).
- radarODE-MTL이 지적한 "주기 길이 오차로 파형이 어긋남"과 Slapničar의 "이어 붙일수록 누적되는 정렬 오차"는 비트를 놓는
  방식에도 해당할 수 있다. 우리 설계는 비트마다 예측 시각에 독립적으로 놓으므로 누적되지 않을 것으로 예상하지만, 연속 파형
  상관과 R 타이밍 오차로 확인한다.
- 조사 에이전트 권고: 평가는 비트별 상관이 아니라 연속 파형 상관, R 타이밍 오차, 여러 비트를 이어 붙였을 때의 상관으로 한다.

## 11. 결정이 필요한 것

- 주 가설: H1(비트 정확도 + 현실성), H2(보정된 구간), 또는 둘 다
- 외부 기준선 재학습 범위: PPGFlowECG·RDDM만, 또는 KANFlow·CardioGAN까지
- 데이터 범위: VitalDB만, 또는 WildPPG·BIDMC·CapnoBase·DaLiA까지
