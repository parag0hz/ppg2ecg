# 연구 프로그램 전체 정리 — 2026-08-25 ~ 2026-09-13

모든 단계는 **사전등록 → 구현 → 실행 → 리포트** 순서로 진행됐고, 사전등록은 어떤 숫자가 나오기 전에
커밋·푸시됐습니다. WildPPG 테스트 피험자 `kjd`/`ssx`는 전 구간에서 **한 번도 로드되지 않았습니다**.

---

## 0. 준비 (2026-08-25) — 학습 없음

| | |
|---|---|
| 대상 논문 | **PENGUIN** (arXiv:2602.03858, ICASSP 2026), OT-CFM + Flow-SSM/S5, Heun 25 step = **50 NFE** |
| 코드 | `github.com/Neurogica/PENGUIN` @ `6cd70cd` — 전 구간 **무수정** |
| 감사 결과 | 출하 config가 PPG-DaLiA에서 **실행 불가**(`DaLiA` 키 불일치), "25 step"은 실제 50 NFE, EMA·스케줄러 없음, `cross_attn` 죽은 코드, split이 파일시스템 순서 의존 |
| 파이프라인 검증 | 우리 전처리가 upstream과 **비트 단위 일치** |

---

## 1부. 재현과 one-step 질문 (A 시리즈)

### 1-1. A0 / A0-b — PENGUIN 베이스라인 재현
- **데이터**: PPG-DaLiA (8 s, test S2)
- **한 것**: upstream 모델 클래스 + `train_flow`/`optimize`를 **우리 학습 루프**에 넣어 학습
- **결과**: HR error **10.99 bpm** (논문 15.64). NFE를 50→1로 줄이면 HR error 10.99 → 36~39 bpm, R-peak F1 0.141 → 0.069로 **붕괴**

### 1-2. A2 / A3 / A4 — iMeanFlow로 대체
- **데이터**: PPG-DaLiA (S2, S1), WildPPG
- **한 것**: 동일 백본(4,568,707 params), 목적함수만 OT-CFM → Improved MeanFlow
- **결과**: DaLiA **SUCCESS** (4/4 지표 회복), WildPPG **PARTIAL** (3/4). 회복되는 축은 **파형 구조**이지 이벤트 신뢰도가 아님

### 1-3. A5 / A6 — 조건부 평균 대조
- **데이터**: DaLiA S1·S2, WildPPG
- **한 것**: 순수 MSE 회귀기를 같은 백본에 학습, OT-CFM 1-NFE와 비교
- **결과**: **STRONG SUPPORT** — OT-CFM 1-NFE는 **MSE 회귀기의 거동에 수렴**. A6에서 용량 차이가 원인이 아님도 확인

### 1-4. A7 / A8 / A9 — 다른 타깃으로의 일반화
- **데이터**: MIMIC-BP (ABP), WildPPG (global-z ECG)
- **결과**:
  - A7 **NOT GENERALIZED** — ABP는 PPG로 거의 결정되므로 조건부 평균이 곧 정답, MSE 회귀기가 최선
  - A8 **SCALE SENSITIVITY CONFIRMED** — iMF의 ABP 붕괴는 **원시 mmHg 스케일** 탓. 전역 affine 하나로 pulse-template 상관 0.140 → **0.876**, RMSE 32.3 → **18.2 mmHg**
  - A9 **REPRESENTATION-ROBUST** — 창 단위 정규화 아티팩트 아님

---

## 2부. one-step에서 무엇이 깨지는가 (X 시리즈, S1)

### 2-1. X0 — one-step 실패 분해
- **데이터**: WildPPG(테스트 3,907창), DaLiA S1·S2
- **결과**: **EVENT-DOMINANT**. 파괴이지 이동이 아님 — GT 비트마다 ±150 ms 최적 정렬을 허용해도 OT-CFM-1은 진폭의 16%, QRS 에너지의 3%만 남음. **타이밍은 멀쩡**(matched-peak MAE 19~20 ms)

### 2-2. X2 — 엔드포인트 barycenter 항등식
- **결과**: **PARTIAL SUPPORT** — 소스 소거 99.9%, 야코비안 ≈ −I로 붕괴는 확인되나 `F̄ ≠ E[x₁|c]`(WildPPG PCC 0.545)라 "수렴 증명"은 금지

### 2-3. X3-G0 — 커플링 비용-기하
- **결과**: **INCONCLUSIVE** — raw L2가 QRS 대역 의존성을 이미 최대화하고, 스펙트럼 재가중은 예산을 QRS에서 **빼감**

### 2-4. X4-0 — 이벤트 신뢰도·소스 민감도
- **결과**: **FEW-STEP SATURATION + PERSISTENT SOURCE SENSITIVITY**. NFE 4~8에서 포화. PPG 고정·소스만 바꿔도 **seed-pair F1 0.30**, 비트 수 SD 1.2~1.8, 타이밍 SD ≈ 50 ms

### 2-5. S1 — 지표 타당성 감사
- **결과 (G1 PASS)**: 정확한 GT 위치에 QRS 템플릿을 찍으면 **macro F1 0.9993**. raw F1의 1/4은 우연, 검출기가 현실적 형태에서 ~0.14 F1 손실

---

## 3부. 압축과 구간 노출 (C 시리즈, M1)

| 단계 | 결과 |
|---|---|
| C0 | NFE **4**를 압축 타깃으로 확정 (압축 방법은 미선택) |
| C1 | **TARGET h=0.5 EXPOSURE SUPPORTED** |
| M1 | **CALIBRATION-ONLY** — C1의 이득은 진폭/에너지 보정이지 구조 개선 아님. 제안된 메커니즘 **기각** |
| C2 | 15-run·54 GPU-h 사전등록만 하고 **영구 보류** |

---

## 4부. PPG에서 무엇이 관측 가능한가 (O, Q, R, V1, E 시리즈)

### 4-1. V1 — PPG 피크 → ECG R 지연
- **결과**: 지연 **IQR 227 ms**, 피험자 내 분산 지배, 50 ms 커버리지 0.22 → **상수 지연으로 R 위치 불가**

### 4-2. R1 — PPG-only 리듬 추출기
- **한 것**: Global/Local-TCN(328,897 params)으로 PPG → R 확률 필드
- **결과**: **정확한 R 타이밍은 제한적**(F1@50 = **0.62**, 잔차 50~150 ms), **전역 리듬 scaffold는 지지**(RR 15.6 ms)

### 4-3. R2 / R3 — scaffold를 생성기에 주입
- **R2 (가산 어댑터)**: **SCAFFOLD INFORMATIVE, MINIMAL INTERFACE INSUFFICIENT** — 이벤트 **+0.0194**(임계 미달), QRS 구조 악화
- **R3 (target-side cross-attention + gate)**: **EVENT GAIN WITH STRUCTURE TRADE-OFF PERSISTS** — 이벤트 **+0.0406**, 그러나 구조 비용 **최악**

### 4-4. O 시리즈 — oracle 좌표에서의 정준화
- **O2 REJECTED** / **O2b ACCEPTED** / **O2c**: oracle 좌표를 주면 상관 0.104 → **0.841**, QRS-core 미분 RMSE 0.322 → **0.133**
- **O3 TOLERANCE REGION TOO NARROW**: oracle로 학습한 정준 생성기는 **±15.6 ms** 지터만 견딤 — R1이 주는 50~150 ms와 **양립 불가**

### 4-5. E 시리즈 / Q1
- **E1 MIXED TOPOLOGY AND PLACEMENT LIMITATION** — 비트 1개 누락/추가가 62.5 ms 지터보다 형태를 **8.7배** 더 망침
- **E2 CONTRACT ACCEPTED** (평가 계약 수립) / **E3 COUNT-ONLY CEILING NOT SUPPORTED**
- **Q1 NO CONSISTENT CONDITION-DEGRADATION PATTERN**

---

## 5부. 구조 가중 목적함수 (M2, M3) — 둘 다 실패

| 단계 | 한 것 | 결과 |
|---|---|---|
| **M2** | 손실에 구조 가중치 (GSW-iMF) | **VERDICT D** — G1 −3.87%, G2 −14.16%, 둘 다 **악화** |
| **M3** | 구조적 엔드포인트 일관성 (SEC-iMF) | **VERDICT D** — G1 +2.66%(5% 바 미달). 메커니즘: 개선이 QRS **바깥**(+26.9%)에 몰리고 **core 안은 +4.85%**뿐 |

---

## 6부. 다중 데이터셋 벤치마크 (D 시리즈)

### 6-1. D1 — 5개 corpus에서 동결 방법론
- **데이터**: PPG-DaLiA, WildPPG, BIDMC, CapnoBase, VitalDB
- **결과**: NFE는 출력을 0.094~0.133 움직이지만 **정답 쪽이 아님**. PCC는 VitalDB 외 0 근처. HR/비트배치 괴리가 모든 corpus에서 재현

### 6-2. D2 — 베이스라인 바닥 (가장 중요한 초기 발견)
- **한 것**: B0~B5 자명한 대조군 (상수, 평균 비트, PPG 피크 템플릿, 엉뚱한 창 등)
- **결과**:
  - **RMSE는 반(反)정보적** — 아무것도 안 쓰는 **B3가 모든 corpus에서 최저 RMSE**
  - **HR error는 박동 수만으로 만족**
  - 모델은 5개 중 **4개에서 실제로 PPG에 조건화**됨(B4가 F1 붕괴). **DaLiA만 진짜 실패**
  - BIDMC는 PPG-피크 템플릿과 **구별 불가**

### 6-3. D3 — PENGUIN 6개 데이터셋 프로토콜 복제
- **데이터**: PPG-DaLiA, WildPPG, BIDMC, WESAD, UCI-BP, MIMIC-BP
- **결과**: 논문 대비 **4승 4패**. `sample_num` = duration×3600/8로 8 s임이 확정되나 `train.py:43`의 assert가 8 s를 불가능하게 만드는 **내부 모순** 발견

### 6-4. D4 — KANFlow 프로토콜 지표표
- **데이터**: BIDMC, CapnoBase, VitalDB (MIMIC-AFib는 **의도적 공백**: 파형 1.08 TB 부재 + 코호트 정의 5가지 충돌 + 인용 체인 단절)
- **결과**: MAE·RMSE·FD·Micro/Macro-F1·RR-MAE·MAE_HR 전부 산출. **NFE를 올려도 단조 개선되는 지표가 하나도 없음**

---

## 7부. upstream 재현과 공정 비교 (U 시리즈)

### 7-1. U1 — upstream `train.py`를 출하 그대로 실행
- **데이터**: 6개 전부, `segment_len=4` (출하 기본값)
- **결과**: **6 REPRODUCED / 1 PARTIAL / 1 NOT REPRODUCED**

| 데이터셋 | U1 | 논문 | r |
|---|---|---|---|
| PPG-DaLiA HR | 16.350 | 15.64 | 1.045 |
| WESAD RR | 4.393 | 4.45 | 0.987 |
| BIDMC RR | 3.479 | 2.98 | 1.168 |
| MIMIC-BP SBP/DBP | 14.986 / 9.212 | 17.43 / 11.34 | 0.860 / 0.812 |
| UCI-BP DBP | 7.863 | 7.14 | 1.101 |
| UCI-BP SBP | 19.986 | 12.61 | 1.585 (PARTIAL) |
| WildPPG HR | 31.006 | 12.97 | 2.391 (NOT) |

- **부수 발견 1 — UCI-BP 누수**: `load_data.py:100-102`가 `part = sub_idx//4+1`과 `onset = sub_idx%2*1500`을 짝지어, **8명이 4명의 바이트 동일 복제**. 실제 split에서 test의 쌍둥이가 train에 있었음. `Part_3/Part_4`는 한 번도 안 읽힘
- **부수 발견 2 — WildPPG 실패 원인**: upstream 자체 patience가 **1-epoch 모델**을 최종 선택 (체크포인트 저장 1회)
- **정정**: A0의 "10.99 < 15.64"는 **우리가 프로토콜을 바꿔서** 생긴 것. 출하 그대로면 16.35

### 7-2. U2 — 계산량 맞춘 paired 비교 (iMF vs OT-CFM)
- **데이터**: U1의 6개 (4 s), 사전등록 subject-holdout split
- **한 것**: 동일 백본·데이터·split·시드, **정확히 14,000 optimizer step** (12 run, 22.4 GPU-h)
- **결과**: **PARTIAL (3/5)**

### 7-3. U3 — U2 자체 결함 2개 검증 (학습 0)
- **결과**: gate를 단방향으로 고치고 **동일 step 지점**에서 읽으면 **WildPPG만 생존 → NOT SUPPORTED**

| gate | 체크포인트 | tally | stage |
|---|---|---|---|
| 양방향(동결) | 각자 best | 3/5 | PARTIAL |
| 단방향 | 각자 best | 2/5 | PARTIAL |
| **단방향** | **둘 다 14k step** | **1/5** | **NOT SUPPORTED** |

- 발견: arm C는 예산의 **80%**, arm I는 **37%** 지점에서 최적 체크포인트를 고름. 비대칭이 **arm I에 유리하게** 작용하고 있었음

---

## 8부. 자명한 베이스라인과 방향 결정 (N 시리즈) — 총 GPU 13분

### 8-1. N1 — 검출기 + 템플릿 (학습 0, 39초)
- **데이터**: WildPPG 동결 2,048창 (an0/k2s), 19,834 GT 비트
- **결과**: **NULL DOMINATES**

| arm | `f1_excess` |
|---|---|
| N-GT (정답 위치, 진단용) | +0.8742 |
| **N-R1 (검출기+템플릿, 학습 0)** | **+0.4866** |
| GTF-TRUE (최고 학습 arm) | +0.3582 |
| B (동결 생성기) | +0.3176 |
| **N-CONST (개별 위치 버리고 균등 격자)** | **+0.2949** |
| N-SHUFFLE (엉뚱한 창) | +0.0054 |

- **생성기 이벤트 점수의 대부분은 "위치"가 아니라 "박동률"**
- 구조 trade-off는 R2/R3 인터페이스의 성질이 아니라 **비트를 더 놓는 무엇이든 치르는 대가**

### 8-2. N2 — oracle 타이밍에서 형태 관측 (48초)
- **결과**: **PPG DOES NOT CARRY BEAT SHAPE** — 템플릿 +0.9062 vs 회귀기 +0.9057 (**유의하게 짐**), PPG 기여분 +0.0039 (바 +0.05). IMF는 +0.6752로 훨씬 나쁨

### 8-3. N3 — 4부위 PPG 융합 (62초)
- **결과**: **MARGINAL** — REG-4 +0.9079, 템플릿 대비 **+0.0005**(바의 1/100). 부위 간 편차 0.0036. 입력을 늘리자 **일반화가 나빠짐**

### 8-4. N4 — 타이밍 분산의 캘리브레이션 (학습 0, 111초)
- **결과**: **CALIBRATED** — coverage 0.587/0.833/0.902 (nominal 0.50/0.80/0.90). 단 **중심이 −15.5 ms 조기**, per-beat 정보 **+0.03**(없음), 비트의 **30%는 소스 절반도 못 찾아 제외**

### 8-5. N5 — per-beat 불확실성 예측 가능성 (28초)
- **결과**: **PREDICTABLE** — sharpness Spearman **+0.475** (바 0.20), shuffle에서 **+0.002로 붕괴**. 생성기의 +0.03 대비 **16배**

### 8-6. N6 — 생성 샘플러 안에서 살아남는가 (77초)
- **동결 판정**: **DOES NOT PRESERVE** — 샘플 SD **7.1 ms**(실제 잔차 SD 53.7), coverage@90 **0.225**
- **사후 스윕(판정 변경 불가)**: 붕괴는 **NFE 1 고유**. NFE 2에서 sharpness **+0.491**로 완전 회복, NFE 16에서 CRPS가 판별 헤드도 이김. **모든 NFE에서 과신 지속**

### 8-7. N7 — 다섯 결함 전부 수정 (448초, 4 fold × 3 seed)
- **데이터**: `an0`/`k2s`를 제외한 WildPPG 12명을 4 fold로, **모든 subject가 자기를 못 본 모델에서 정확히 1회 평가**
- **결과**: **NOT SUPPORTED (3/12)**

| 조건 | 통과 |
|---|---|
| CRPS가 CONST를 이김 | **12/12** |
| sharpness ≥ 0.20 | **12/12** |
| shuffle을 이김 | **12/12** |
| **coverage ±0.10** | **3/12** |

- **sharpness가 복제됨**: **+0.445 ± 0.021** (범위 +0.395~+0.479) — 이 프로그램에서 **복제된 유일한 양성 결과**
- 캘리브레이션은 일관되게 실패: coverage 0.424/0.688 (목표 0.50/0.80). **사전등록 NFE 격자가 12/12에서 소진**(dev coverage가 32에서도 0.677로 상승 중)
- **생성 형식이 아무것도 벌지 못함**: FM − HEAD = −0.212, 12개 중 2개만 CI>0. **이봉성 12/12에서 0.0000**
- validity 헤드는 유효: Brier 0.1252 vs 기저율 0.1715 (**+0.046**)

---

## 9. 총괄

### 9-1. 확립된 양성 결과 (2개)
1. **PENGUIN 논문 수치는 그들 코드로 재현 가능** (8칸 중 6칸) — U1
2. **per-beat 타이밍 불확실성은 PPG에서 예측 가능** (+0.445 ± 0.021, 12 run 복제) — N5/N7

### 9-2. 확립된 음성 결과
1. 학습된 생성기가 **자명한 baseline을 이긴 적이 없음** (N1)
2. **비트 형태는 PPG에 없음** — 1시점·4시점, MSE·flow matching, oracle 타이밍 유무 전부 (N2/N3)
3. **iMeanFlow가 OT-CFM보다 낫지 않음** — 결함 보정 후 1/5 (U2/U3)
4. **손실 재가중으로 구조 개선 불가** (M2/M3)
5. **one-step flow matching은 조건부 평균으로 붕괴** — 1024샘플·83샘플·스칼라 1개 전부 (A5/A6/X2/N2/N6)

### 9-3. 지표에 대한 발견 (문헌 전체에 해당)
1. **RMSE는 반정보적** — 상수 예측기가 모든 corpus에서 최저 (D2)
2. **HR error는 박동 수만으로 만족** (D2), **F1도 대부분 박동률** (N1)
3. **RespRateError는 위상맹** — 양쪽 arm 파형 상관 ≈ 0인데 점수는 정상 (U1/U2)
4. **NFE를 올려도 단조 개선되는 지표 없음** (D4)
5. **공개 벤치마크에 바이트 단위 train/test 중복** (U1-P1, UCI-BP)
6. **발표 수치가 test subject 1~2명에 얹혀 있음** (U1)

### 9-4. 자원
- GPU: U1 ~36 h, U2/U3 ~28 h, N1–N7 합계 **~13분**
- 사전등록 문서 30+, 리포트 49개, 전부 커밋·푸시됨
- 테스트 778개 통과, `external/PENGUIN` 전 구간 무수정, `kjd`/`ssx` 전 구간 미로드
