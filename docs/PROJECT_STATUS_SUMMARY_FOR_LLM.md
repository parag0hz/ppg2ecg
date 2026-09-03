# ppg2ecg-one-step — 연구 프로그램 현황 요약 (외부 LLM 입력용)

작성 기준: 2026-09-03, `main` HEAD `55fe1e1` (R3 결과 커밋). 저장소 `https://github.com/parag0hz/ppg2ecg.git`.
이 문서는 프로젝트의 배경(베이스라인 논문·데이터·지표·규칙), 지금까지 수행한 모든 단계의 질문·설계·판정·수치,
**무엇이 실패했고 왜 실패했는지**, 그리고 열린 질문을 한 파일에 정리한 것이다. 각 단계의 원문은 `docs/`의
preregistration / report 문서에 있으며, 여기 적힌 숫자는 그 문서와 `artifacts/`의 결과 파일에서 옮긴 값이다.

읽을 때 전제할 것:
- 테스트 피험자 접근 이력: WildPPG 테스트 피험자 `kjd`/`ssx`는 학습 단계의 동결 테스트 평가(A4, A5c, A6c, A9)에 쓰였고, 그 동결 예측 배열이
  X0·X2(2026-08-29/30) 분석에 재사용되었으며, X3-G0의 사전-사전등록 설계 감사가 이미 공개된 동결 테스트 배열을 읽은 사실이 공개되어 있다.
  **X3-G0 본 분석과 X4-0(2026-08-30) 이후의 모든 단계(S1, C0, C1, M1, V1, R1–R3)는 `kjd`/`ssx`를 로드하지 않았다**(코드 방화벽
  `assert_no_test_subjects`, provenance에 `test_subjects_loaded: []`). 그 단계들의 결과는 개발 검증 피험자 an0/k2s 기준의 development-only다.
- 모든 학습은 **단일 seed 42**다. 신뢰구간은 검증 창(window) 표본의 부트스트랩이며 학습 seed 분산·피험자 분산은 포함하지 않는다.
- 판정(verdict)은 결과를 보기 전에 커밋한 사전등록 규칙의 출력이다. 판정 문자열은 원문 그대로 인용한다.
- 단계 R1–R3, V1, S1, C0–C2, M1은 `docs/EXPERIMENT_LOG.md`에 기록되지 않았다(로그는 X4-0, 2026-08-31에서 멈춤). 이들의 날짜는 git과 provenance에서 가져왔다.

---

## 0. 한 줄 요약 (TL;DR)

연구 질문: **"PPG 조건부 ECG 재구성을 임상적으로 의미 있는 ECG 형태(morphology)와 조건부 충실도(conditional fidelity)를 잃지 않고
one-step(NFE 1) 생성으로 줄일 수 있는가?"** (`docs/RESEARCH_QUESTION.md`, 2026-08-25 동결)

지금까지 확립된 것:
1. 베이스라인 PENGUIN(OT-CFM + S5, 50 NFE)은 NFE를 줄이면 무너진다. 1-NFE Euler 출력은 **MSE로 학습한 조건부 평균 proxy의 거동에 경험적으로 근접**한다(A5의 허용 문구; X0: 회귀기와 정성적으로 같은 물체)
   — 진폭·QRS 날카로움 소실, RMSE는 오히려 좋아지는 "pointwise-error inversion". A0/A0-b/A1, A5/A6(O1이 회귀기에 2–4배 더 가까움), X0(구조가 "이동"이 아니라 "파괴"), X2(source-noise 99.9 % 소거, J_x v ≈ −I; 알려진 barycenter 퇴화의 경험적 실현, PARTIAL SUPPORT)로 메커니즘까지 확인.
2. 동일 백본에서 목적함수만 Improved MeanFlow(iMF)로 바꾸면 1-NFE에서 진폭·형태·HR을 대부분 회복한다(A2 SUCCESS, A3 REPLICATED, A4 PARTIAL).
   단, 회복되는 것은 "파형 구조"이고, **비트(R-peak) 이벤트 신뢰도는 NFE를 늘려도 부족한 수준에서 포화**한다(X4-0: F1 0.41–0.43, 소스 노이즈에 따라 이벤트 정체가 바뀜).
3. 이 결과는 ECG에 특이적이다. PPG→ABP(MIMIC-BP)에서는 결정론적 MSE 회귀기가 최선이고 one-step 감쇠가 없다(A7 NOT GENERALIZED; A8: iMF의 ABP 실패는 타깃 스케일 문제). ECG 타깃 정규화 방식을 바꿔도 결론은 유지(A9 REPRESENTATION-ROBUST).
4. 시도한 개선 레버 중 성공한 것은 없다: 시간-간격 커리큘럼(B1, 중단), 커플링 기하(X3-G0, INCONCLUSIVE; 커플링 학습 권고 안 함), h=0.5 노출(C1 A판정 → M1에서 "calibration-only"로 재분류), PPG 피크-지연 prior(V1, 불안정), 리듬 scaffold 주입(R2 additive, R3 target-side cross-attention + gate: **이벤트 이득은 있으나 QRS 미분/곡률 구조 비용이 지속**).
5. 현재 열린 핵심 질문: PPG에서 뽑은 리듬 정보로 이벤트 대응을 높이면서 **QRS-core 구조(S4/S5)를 잃지 않는 인터페이스**가 있는가; R3의 gate 채널 vs attention 채널 분리(R4 권고, 미구현); 다중 seed 재현(C2 15-run은 사전등록만 되고 학습 보류).

---

## 1. 베이스라인 논문과 코드

### 1.1 PENGUIN (재현 대상 베이스라인)
- 논문: Suzuki, Koyama, Hirano, Nagashima (Neurogica Inc.), *PENGUIN: General Vital Sign Reconstruction from PPG with Flow Matching State Space Model*, arXiv:2602.03858 (2026-01-23), ICASSP 2026 (oral BISP-L5.5).
- 코드: `https://github.com/Neurogica/PENGUIN`, BSD-3-Clause-Clear, 서브모듈 `external/PENGUIN` @ `6cd70cd` (절대 수정 안 함; 매 실행 시 clean 트리 assert).
- 구조(감사 결과, `docs/PENGUIN_AUDIT.md`): Flow-SSM 이중 스트림 S5 백본. h_dim 128, SSM 블록 4, ssm_ratio 2, mlp_ratio 2. PPG 스트림과 타깃(노이즈 상태 x_t) 스트림, 각 스트림 = adaLN → 양방향 S5 → 게이트 → MLP. PPG 조건화는 블록마다 2-layer GELU MLP로 **가산** 융합(`target_cond = target_cond + ppg_cond`). 파라미터 4,568,707(그중 264,194 = 호출되지 않는 `cross_attn` 264,192 + `revin` 2 → 유효 4,304,513). 특이점: 타깃 스트림이 블록 간 체인되지 않고(같은 x_t 임베딩이 4 블록에 병렬 입력, 출력 합산), adaLN-Zero 초기화.
- 목적함수: "OT-CFM" = Lipman 조건부 OT 경로(σ_min=0, x_t=(1−t)x0+t·x1, 타깃 속도 x1−x0, MSE), **독립 커플링**(minibatch-OT 아님).
- 샘플러: Heun 25 스텝 = **50 NFE**(논문의 "25 steps"는 네트워크 평가 50회). 학습: AdamW 1e-3/wd 0.01, batch 64, ≤300 epoch, val MAE 조기 종료; EMA는 config(ema_decay 0.999)에 있으나 읽히지 않고, LR 스케줄러·클리핑·AMP는 코드·config 모두에 없음.
- 논문 수치(Table 1 HR error, bpm): PPG-DaLiA PENGUIN 15.64 (RDDM 16.43, CycleGAN 23.61, RespDiff 22.75, PaPaGei-S 40.89, w/o PPG cond 24.40); WildPPG PENGUIN 12.97; MIMIC-BP SBP/DBP 17.43/11.34. 파라미터·seed·분산·held-out 피험자 정보 없음.
- 감사에서 발견한 문제: 배포 config가 PPG-DaLiA에서 실행 불가(`DaLiA` vs `PPG-DaLiA` 키); 윈도 길이가 논문(4 s 그림 캡션)과 코드 부기(8 s, `sample_num=16181`)에서 불일치; 배포 HR 지표는 segment_len=4에서 8 s 창을 512샘플로 리샘플하여 **HR을 2배로 부풀리고** 고HR 창을 0으로 마스킹; split이 glob 순서 의존(재현 불가). → 본 프로젝트는 8 s 창, 결정론적 매니페스트 split, HR 지표 `corrected`/`as_shipped` 두 버전 보고로 고정.

### 1.2 MeanFlow / Improved MeanFlow (one-step 목적함수)
- MeanFlow: Geng, Deng, Bai, Kolter, He, arXiv:2505.13447, NeurIPS 2025 oral. Improved MeanFlow(iMF): Geng*, Lu*, Wu, Shechtman, Kolter, He, *Improved Mean Flows: On the Challenges of Fastforward Generative Models*, arXiv:2512.02012 v2, CVPR 2026 highlight. 공식 JAX 코드 `Lyy-iiis/imeanflow`, 서브모듈 `external/iMeanFlow` @ `bf60cd7`(수정 안 함).
- 핵심: 평균 속도 u(z_t, r, t), 항등식 u = v − (t−r)·du/dt. iMF의 v-loss 재매개변수화 V = u + (t−r)·sg(JVP(u; v_θ)), loss ‖V − (e−x)‖², 적응 가중 w = 1/(δ²+c)^p (p=1, c=0.01), (t,r) i.i.d. logit-normal(−0.4, 1), 50 % r=t(순수 flow matching), 1-NFE x̂ = e − u(e, r=0, t=1). 이 프로젝트의 이식(`src/ppg2ecg/flow/imeanflow.py`): PENGUIN 백본 **무수정**, 조건 h=t−r만 기존 timestep embedder로 입력(공식 iMF 설계), v_θ는 boundary u(z,t,t)(aux head 없음), forward-mode JVP(`torch.func.jvp`)가 S5 스캔을 통과함을 유한차분으로 검증, 공식 JAX 목적함수와 1e-9 gradient 일치. 최적화는 베이스라인 recipe 유지(AdamW 1e-3, no EMA). 메모리 때문에 micro-batch 32×2 누적.
- 주의(X4-0): 학습 시 h 분포에서 **정확히 h=1(1-NFE 질의)의 확률은 0**(2M 샘플 중 최대 h=0.9297). "support 밖"이 아니라 "극단 경계 질의"로 표현.

### 1.3 그 외 참조한 논문/방법
- Kim et al., *Understanding, Accelerating, and Improving MeanFlow Training*, arXiv:2511.19065 v2, CVPR 2026 — B1의 progressive temporal-gap weighting 출처(공식 코드 없음).
- 엔드포인트 barycenter 항등식(X2, 선행연구로 명시): Frans 2024 (2410.12557), Albergo–Boffi–Vanden-Eijnden 2023 (2303.08797), Albergo–VE 2022, Liu et al. 2022 (2209.03003), Lee et al. 2024 (2405.20320).
- minibatch OT-CFM (Tong 2302.00482), Multisample FM (2304.14772), C2OT (Cheng & Schwing 2503.10636) — X3-G0에서 검토, 학습은 안 함.
- 데이터셋 논문: PPG-DaLiA (Reiss et al. 2019, UCI #495), WildPPG (Meier, Demirel, Holz, NeurIPS 2024 D&B, arXiv:2412.17540), MIMIC-BP v2.2 (Sanches et al., Sci. Data 2024).

---

## 2. 데이터

공통 전처리(PENGUIN과 비트 단위 일치 검증): 비중첩 **8 s 창 @ 128 Hz = 1024 샘플**; PPG Butterworth 4차 0.5–4 Hz 대역통과, ECG 0.5 Hz 고역통과(filtfilt); **창 단위** z-score 후 min-max → [−1, 1](학습셋 통계 없음, 절대 mV 복원 불가).

### 2.1 PPG-DaLiA (A0–A3, A5/A6, B1, X0/X2)
- 15명(S1–S15), 손목 BVP 64 Hz(Empatica E4) → 입력, 가슴 ECG 700 Hz → 타깃, 총 ~36 h, 8 s 창 16,181개.
- Split(주제 단위, 매니페스트): P0 = train 13 / val S11 / test **S2**(A0, A0-b, A2); A3 = test **S1** / val S11. 누출 검사(피험자·창 해시·창-국소 정규화·타깃 비접근 추론) PASS.
- **치명적 데이터 한계**: 손목 PPG와 가슴 ECG가 초 단위로만 동기화됨(맥파 도달 지연 500–900 ms, ~20 ms/min 드리프트). 따라서 raw DaLiA에서 R-peak F1(50 ms), PCC, 매칭 RR MAE는 **모델이 아니라 데이터 아티팩트를 측정**(F1 ≈ 0.14, 50 NFE에서도). 결정에는 HR 오차·매칭 비트 형태 상관·진폭비·PPG-shuffle 조건화 이득만 사용. 재동기화 프로토콜은 권고만 되고 실행되지 않음.

### 2.2 WildPPG (A4 이후 모든 단계의 주 데이터)
- 16명(an0 e61 fex k2s kjd l38 n31 ngh p5d p9p qm9 ssx trh tz8 u7y w4p), 4개 **시간 동기화된** 기기(sternum, head, wrist, ankle), 녹색 PPG 128 Hz, ECG lead-I 128 Hz는 sternum에만. 총 ECG 216.8 h. 저자 표시 noisy-ECG: fex, kjd, p5d(제외 안 함, PENGUIN도 유지).
- PENGUIN 방식: 4 사이트의 PPG 창을 각각 독립 샘플로 쓰고 sternum ECG를 ×4 타일. 처리본 `data/processed/wildppg_8s/` 389,355창(상수-갭 창 861개 = 0.22 % 제거, 문서화된 편차).
- Split(`split_a4_wildppg_seed42.json`): **train 12** = e61 fex l38 n31 ngh p5d p9p qm9 trh tz8 u7y w4p(293,271창); **val(development) = an0, k2s**; **test = kjd, ssx**(A4/A5c/A6c/A9 동결 평가와 X0/X2 재분석에 사용; X3-G0/X4-0 이후 미로드; kjd는 noisy-ECG).
- 동결 개발 모집단(X4-0부터 S1, C0, C1, M1, R2, R3가 공유): `select_subset("x4-event-nfe-v2")` → an0 1,024 + k2s 1,024 = **2,048창, 19,834 GT beats**, 소스 노이즈 bank seed 0(sha `86808579…`). X4-0 전에 시각 검사한 4개 창은 구성상 제외. an0/k2s는 "이미 들여다본 개발 검증"이지 깨끗한 확증 검증이 아님.
- R1 site-wise 코호트: 1,024 per subject×site = 8,192창(79,111 GT beats). V1 VIZ 코호트: 14명×4부위×8 = 448창; 그중 an0/k2s 검증 슬라이스 64창(619 beats)을 R2/R3 시각 관찰에 재사용.
- 동기화된 데이터이므로 비트 단위 지표가 정보를 가진다(OT-CFM-50 F1 0.44–0.48). 부위별 난이도: head/sternum ≫ wrist/ankle.

### 2.3 MIMIC-BP v2.2 (A7/A8, PPG→ABP)
- 1,524명 ICU 동맥라인, 30 × 30 s @125 Hz, 공식 split 1,100/195/229. 8 s 창으로 137,160개(=PENGUIN sample_num). ABP는 raw mmHg(A7) 또는 학습셋 전역 z(A8: μ 77.57, σ 22.28 mmHg). 테스트 서브셋 3,435창. UCI-BP는 피험자 ID가 없어 기각.

---

## 3. 평가 지표와 통계 관례

- **이벤트/리듬**: R-peak 검출 = neurokit2 `ecg_clean`+`ecg_peaks`(예측과 GT에 동일 적용), 일대일 greedy 매칭 **50 ms**(프로젝트 동결 허용오차). `hr_abs_err_bpm`(시프트 불변, 위상 정보 없음), R-peak precision/recall/F1, missing(=1−recall), spurious, beats ratio(예측 비트 수/GT), beats-ratio deviation.
- **F1 excess** (S1 이후 표준): F1 − count-matched random-phase chance floor(≈0.11–0.12; 8 s 창에 ~10 beats면 우연 일치가 F1의 약 1/4). raw F1 단독 보고 금지.
- **형태**: `morph` = **매칭 비트 형태 상관**(−250…+400 ms, 50 ms 매칭된 비트만 평균) — 누락 비트는 분모에서 빠져 **커버리지에 단조가 아님**, recall이 다른 arm 간 직접 비교 불가(`docs/METRIC_SEMANTICS.md`, 2026-08-31). 진폭비 std(pred)/std(GT), QRS-width proxy, HF 비율(>15 Hz).
- **조건화 이득**(PPG-shuffle): 잘못된 PPG(derangement)로 생성했을 때의 HR 오차 − 올바른 PPG일 때 HR 오차(높을수록 PPG 의존).
- **구조(GT 고정 좌표, 검출기 무관; C0/M1/R2/R3)**: S1 raw RMSE, S2 raw corr(GT R 기준 83샘플 세그먼트), S3 QRS RMSE(R±13), **S4 QRS-core 미분 RMSE, S5 QRS-core 곡률 오차**(R±11; R2/R3의 핵심 구조 지표), S6 QRS 에너지비 편차, S7 p2p 편차, S8 HF.
- **철회된 지표**: `oracle_corr`, `oracle_absent`, `oracle_qrs_energy`(±150 ms 최적 시프트) — S1.4b에서 "무관한 비트와 짝지어도 같은 이득"이 재현되어 **증거로 사용 금지**(C0 이후 모든 결정에서 제외).
- **통계**: 유의성 검정 없음. 창 단위 paired, 피험자 층화 부트스트랩 2,000회(an0/k2s 동일 가중; seed 20260901 또는 20260902). "improves/worsens/unresolved" = 95 % CI가 0을 완전히 배제하는지. 다중 비교 보정 없음.
- **판정 규칙**: 각 단계의 preregistration에 결과 전 커밋된 게이트/결정 트리(`decision.json`이 총함수). 결과 후 임계값 변경 금지. 사전에 허용/금지 문구 목록 존재(예: "confidence-calibrated", "multimodality proven", "phase diversity", "SOTA" 금지).

---

## 4. 프로그램 운영 규칙

- 사전등록 → 커밋·푸시 → 그 다음에만 학습/지표. 동결 문서는 사후 편집 금지; 버그 수정은 공개하고 커밋.
- 격리 원칙: 비교마다 정확히 한 요인만 변경(백본·데이터·split·전처리·평가 코드 고정).
- 사전등록 없이 새 아키텍처/손실/조건화/EMA/CFG/옵티마이저/LR/seed/데이터셋 변경 금지. 결과 의존 튜닝 금지.
- 단일 seed 42(seed 43/44와 5-fold는 v0 사전등록에 있었으나 실행된 적 없음). 테스트셋으로 어떤 결정도 하지 않음.
- 체크포인트/예측/아티팩트는 git에 넣지 않음(`outputs/`, `artifacts/` gitignore). 서브모듈 SHA 고정.
- 환경: RTX 5090 32 GB 1장, torch 2.11.0+cu130, Python 3.13.9, fp32. 2026-08-25의 numpy-MKL `eigh` 세그폴트는 `ppg2ecg.utils.mkl_warmup`을 torch보다 먼저 import하여 해결.

---

## 5. 단계 연대기 (한눈에)

| 단계 | 날짜(KST) | 데이터 | 질문 | 판정(원문) | 한 줄 결과 |
|---|---|---|---|---|---|
| A0 | 08-25 | DaLiA S2 | PENGUIN 재현 | PASS (feasibility gate) | 50 NFE HR err 10.99 bpm(논문 15.64) |
| A1 | 08-25 | DaLiA S2 | NFE 곡선 | H1 confirmed, GO | 4 NFE에서 morph 붕괴 시작, ≤2 NFE 전면 붕괴 |
| A0-b | 08-25 | DaLiA S2 | 체크포인트 선택 안정화 | GO (gap persists) | A0 과소학습; 1-NFE 붕괴는 목적함수/샘플러 한계 |
| A2 | 08-25 | DaLiA S2 | iMF 1-NFE 회복? | SUCCESS | 회복률 HR .96 / morph .87 / amp .93 / cond .78 |
| A3 | 08-26 | DaLiA S1 | 피험자 재현 | SUCCESS / REPLICATED | 회복 .86/.80/.76/.53 |
| A4 | 08-26 | WildPPG | 데이터셋 재현 | PARTIAL | 진폭·형태 회복, 조건화 이득은 OT-1보다 낮음 |
| 통합 | 08-26 | — | — | SUBJECT-ROBUST, DATASET-UNCERTAIN | 동기화 여부가 1-NFE 실패의 성격을 바꿈 |
| A5 | 08-26 | 3셋 | 1-NFE OT-CFM ≈ MSE 회귀기? | STRONG SUPPORT | OT-1이 회귀기에 2–4배 더 가까움(S1 2.2–2.4×, S2 3.6–3.7×, WildPPG 3.3–4.4×) |
| A6 | 08-27 | 3셋 | 용량 반론 | CAPACITY OBJECTION RESOLVED | 풀 백본 MSE도 동일하게 평탄 |
| A7 | 08-27 | MIMIC-BP | ABP로 일반화? | NOT GENERALIZED | ABP는 회귀기가 최선, iMF-1 실패 |
| A8 | 08-27/28 | MIMIC-BP | iMF 실패=스케일? | SCALE SENSITIVITY CONFIRMED | 전역 z 후 iMF 정상화, 그래도 회귀기 최선 |
| A9 | 08-28 | WildPPG | 창 정규화 탓? | REPRESENTATION-ROBUST | 전역 z에서도 ECG 결론 유지 |
| B1 | 08-28/29 | DaLiA S2 | gap 커리큘럼 | ABORTED / INCOMPLETE — NO CONFIRMATORY VERDICT | S2 pair에서 커리큘럼이 더 나쁨(탐색적) |
| X0 | 08-29 | WildPPG+DaLiA | 실패 분해 | EVENT-DOMINANT ("destroyed, not displaced") | 구조가 평탄화됨, 타이밍은 정상 |
| X2 | 08-30 | 3셋 | 엔드포인트 항등식 | PARTIAL SUPPORT | 소스 소거 3/3, M2(PCC≥.60) 실패 |
| X3-G0 | 08-30 | WildPPG train | 커플링 기하 게이트 | INCONCLUSIVE | 커플링 학습 권고 안 함 |
| X4-0 | 08-30/31 | WildPPG dev | iMF 이벤트 신뢰도 | MIXED (CASE A + C) | NFE 8에서 포화, 소스 민감 이벤트 조직 지속 |
| S1 | 08-31 | WildPPG dev | 지표 타당성 | G1 PASS; oracle 철회 | F1의 1/4은 우연; oracle_* 무효 |
| C0 | 09-01 | WildPPG dev | 압축 타깃 | COMPRESSION TARGET = NFE 4 | NFE 2→4 실질 이득, 4→8은 trade |
| C1 | 09-01/02 | WildPPG | h=0.5 노출 | TARGET h=0.5 EXPOSURE SUPPORTED | H50@NFE2 > B@NFE4, 단 compute 미매칭 |
| C2 | 09-02 | — | 다중 seed 재현 | DEFERRED (학습 0) | 15-run 54 GPU-h 보류 |
| M1 | 09-02 | WildPPG dev | C1 효과의 본질 | CALIBRATION-ONLY | 진폭 보정일 뿐, 국소 QRS 개선 아님 |
| V1 | 09-02 | WildPPG 14명 | NFE별 시각화 + PPG 지연 | PPG PEAK TIMING TOO UNSTABLE | 지연 IQR ~227 ms vs 50 ms 허용 |
| R1 | 09-02 | WildPPG | PPG→리듬 관측 가능? | EXACT R-TIMING LIMITED; GLOBAL RHYTHM SCAFFOLD SUPPORTED | Global-TCN F1@50 0.62, RR median AE 15.6 ms |
| R2 | 09-02 | WildPPG dev | scaffold→생성기(1×1 add) | SCAFFOLD INFORMATIVE, MINIMAL INTERFACE INSUFFICIENT | +0.0194 F1 excess, S4/S5 악화 |
| R3 | 09-02/03 | WildPPG dev | target-side fusion + gate | EVENT GAIN WITH STRUCTURE TRADE-OFF PERSISTS | GTF +0.0406, S4 더 악화, gate ≠ confidence |

---

## 6. 단계별 상세

표기: O50 = OT-CFM Heun-25(50 NFE), O1 = OT-CFM Euler 1-NFE, M1 = iMF 1-NFE, R = MSE 회귀 proxy. 회복률 rec(m) = (M1−O1)/(O50−O1)(높을수록 좋은 지표 기준).

### 6.1 A0 / A1 / A0-b — 베이스라인 재현과 NFE 붕괴 (2026-08-25, DaLiA test S2)
- **A0**: 무수정 PENGUIN, 우리 loop(업스트림과 step-for-step 동일, 샘플러·전처리 비트 일치), 21 epoch(조기종료, best 11). 50 NFE: corrected HR err **10.99 bpm**(논문 15.64 → PASS ≤17.2), morph 0.662, R-peak F1 0.141(DaLiA 비동기 → 무의미). 예측-참조 HR r=0.40, 고HR에서 평균 회귀(−28 bpm bias >110 bpm). 판정은 "재현 성공"이 아니라 feasibility gate(피험자·창 길이 불명).
- **A1(NFE 곡선, A0 체크포인트)**: 50/20/10/4/2/1 NFE에서 HR 10.99/11.59/12.25/11.17/36.32/39.22 bpm, morph .662/.664/.639/.475/.230/.136. **4 NFE에서 morph가 먼저 무너지고(−0.19), ≤2 NFE에서 전부 붕괴**. 1-NFE 출력은 −0.3 근처 평탄선(창 std 0.046 vs GT 0.236, seed 다양성 소실, 조건화 이득 ≈0)이며 **RMSE는 0.472→0.295로 좋아짐** → RMSE/MAE/PCC를 품질 기준에서 제외. H1 확인, GO.
- **A0-b**: 유일한 변경 = 결정론적 고정 bank val CFM loss로 체크포인트 선택. 85 epoch(best 65), A0는 과소학습이었음(고정 bank loss 0.1904→0.1645). 50 NFE HR 8.08 bpm(CI 분리), cond gain 5.69, amp 0.95, **morph는 0.650으로 불변**(QRS 형태는 OT-CFM+S5의 포화점). 1-NFE 붕괴 그대로(HR 41.96, morph .217, amp .145, gain .24) → 네 기준 모두 실패 → **GO** (목적함수/샘플러 한계). 더 잘 학습된 모델이 NFE 축에서 **더 일찍** 무너짐(4 NFE HR 15.8 vs 11.2).

### 6.2 A2 — Improved MeanFlow on the identical backbone (2026-08-25, DaLiA S2)
- 두 번의 결과 전 수정: E(t)+E(h) 공유 임베더는 h를 거의 구분 못함(r 복호 R² 0.18) → h×1000은 JVP 항이 1000배 커져 2 epoch에 발산 → **h-only 조건화(공식 iMF 설계)** 채택. 중단 실행은 `outputs/aborted/`에 보존.
- 결과(NFE 1, 82 ms/batch64, O50 대비 51배 빠름): HR 9.58(O50 8.08, O1 41.96), morph 0.595(0.650/0.217), amp 0.90, gain 4.47(5.69/0.24), beats/ref 1.00. 회복률 **HR 0.96 / morph 0.87 / amp 0.93 / cond 0.78 → SUCCESS**(회복률 ≥0.5 규칙). RMSE는 iMF-1 0.443 > O1 0.304(inversion 확인). iMF 4 스텝은 HR·morph·gain에서 O50보다 좋음(HR 7.02, morph 0.719, gain 6.59; 진폭은 0.927 vs 0.949로 근소 열위; 비용 8 %; A2 보고서 원문은 "every physiological metric"); 2 스텝은 HR·morph에서 우위, 진폭(0.92 vs 0.95)·gain(5.60 vs 5.69)은 근소 열위.
- 한계: v0 비열등 마진(HR +1.0, morph −0.05)으로는 iMF-1이 통과 못함(+1.50 bpm, −0.055, CI 분리). 공식 iMF recipe(EMA, aux head, lr 1e-4)는 쓰지 않음.

### 6.3 A3 / A4 / 통합 — 재현 (2026-08-26)
- **A3(DaLiA test S1, 처음부터 재학습)**: O50 8.16/.683/.87/8.77; O1 35.23/.168/.21/.28; iMF-1 11.96/.581/.71/4.78, beats 1.03. 회복 .86/.80/.76/.53 → **SUCCESS / REPLICATED**. 조건화 회복이 가장 약함(0.53).
- **A4(WildPPG, test kjd+ssx 3,907창 서브셋, 220-step round 단위)**: O50 HR 9.43 / morph .670 / amp .98 / gain 7.16 / **F1 0.440**; O1 15.59 / .379 / .32 / **6.64** / F1 **0.481**(최고) / QRS-width err 75 ms; iMF-1 11.85 / .551 / 1.04 / 4.29 / F1 .385. 회복 HR .61 / morph .59 / amp .90 / **cond −4.47** → **PARTIAL**. 부위별 iMF-1은 4 부위 모두 HR 개선.
- **왜 PARTIAL인가**: WildPPG는 기기가 동기화되어 있어 1-NFE OT-CFM(조건부 평균)이 **비트 정렬된 평균**이 된다 — 리듬·PPG 의존(gain 6.64/7.16)·최고 비트 타이밍은 유지하고 진폭(0.32)과 QRS 날카로움만 잃는다. iMF-1은 진폭·날카로움을 복원하지만 PPG 의존이 낮고 비트 배치가 덜 정확(F1 .385, beats/ref .93; "beat-timing imprecision"). 조건화 회복률의 분모가 0.52 bpm이라 ill-conditioned지만 방향(iMF gain < O1 gain)은 실제.
- **통합 판정: SUBJECT-ROBUST, DATASET-UNCERTAIN**. 세 곳 모두 재현된 것: 진폭 회복 0.76–0.93, 형태 회복 0.59–0.87, pointwise inversion, 순서 O1≪O50, iMF-1≫O1. 재현 안 된 것: 조건부 리듬 충실도 — "one-step 조건부 평균이 비트에 대해 정보가 없을 때(비동기 pairing)만 iMF의 리듬 회복 주장이 성립".

### 6.4 A5 / A6 — 조건부 평균 대조 (2026-08-26/27, 3 데이터셋)
- **A5**: PENGUIN 백본에서 노이즈 스템·시간 임베더를 제거한 MSE 회귀기(2.9 M 유효). 사고: 사전등록된 zero-state 입력은 **학습이 되지 않음**(adaLN-Zero + final layer weight 0 초기화 → bias만 gradient) → 학습 가능한 상수 토큰(128 params)으로 결과 전 수정(Amendment 1). 결과: 회귀기 amp 0.06/0.05/0.25, morph .16/.15/.33, RMSE 최저, QRS 창 에너지 GT의 15–27 %(O1 27–35, O50 72–88, M1 61–109). O1↔R 파형 RMSE 0.079–0.134 vs O50/M1 0.26–0.35(2.2–4.4배) → **STRONG SUPPORT**: "OT-CFM 1-NFE는 MSE 조건부 평균 proxy의 거동에 경험적으로 근접". WildPPG에서 회귀기 F1 0.436(O50 0.440), gain 5.70 — 정렬이 무엇이 살아남는지 결정. 상수 GT-평균 예측기의 RMSE 0.250이 모든 모델보다 좋음(pointwise 지표는 오프셋/저주파 지배).
- **A6**: 파라미터 완전 일치(4,568,707) 풀 백본 MSE 회귀기(x_const 0.1, t 0.5, cond 0.05·E(t); spec 예시 x=1/cond 미스케일은 상수 해로 붕괴 → 사전 hard test로 결정). R_full ≈ R_small(파형 RMSE 0.042–0.045), 동일한 평탄화, O1이 여전히 가장 가까움 → **CAPACITY OBJECTION RESOLVED**. 최적화 매칭은 안 됨(파라미터만).

### 6.5 A7 / A8 / A9 — 타깃 의존성 대조 (2026-08-27/28)
- **A7(PPG→ABP, MIMIC-BP raw mmHg)**: MSE proxy가 모든 지표 최선(SBP/DBP MAE 14.31/8.72, 펄스 템플릿 상관 0.929, 피크 F1 0.945); O1(15.09/9.51, .904)이 O50(15.94/9.80, .884)보다 좋음(50 NFE는 HF·기울기 과잉); **iMF-1 붕괴**(corr 0.140, HF 0.550 vs GT 0.043, slope ratio 6.13, RMSE 32.3, 조건화 거의 0). 감쇠 없음, inversion 없음 → **NOT GENERALIZED**. 해석: ABP 펄스는 PPG로 거의 결정되므로 조건부 평균 자체가 날카로운 파형; ECG는 조건부 평균이 어떤 그럴듯한 ECG와도 멀어 QRS가 파괴됨.
- **A8**: ABP를 학습셋 전역 affine z로만 바꿈. 수송 기하 감사: ‖y‖/‖e‖ 81.6→0.95, t=0.5에서 prior 에너지 비중 0.0001→0.488. iMF-1: corr 0.140→0.876, HF 0.550→0.050, slope 6.13→1.49, F1 0.336→0.874, RMSE 32.3→18.2(SBP MAE는 16.28→18.75로 악화, PP 과대) → **SCALE SENSITIVITY CONFIRMED**. MSE/O1은 불변 → 교란 아님. 그래도 회귀기(14.05/13.03)와 O1(13.95/13.43)이 최선 → "ABP에는 결정론적 회귀가 충분·우월". 이상: 정규화 후 O-CFM 4 NFE(및 2 NFE)가 1·50 NFE보다 훨씬 나쁨(중간 NFE 불안정, ECG의 2 NFE와 같은 패턴, 미설명).
- **A9(WildPPG, ECG를 창 정규화 대신 전역 z)**: MSE·O1 감쇠 지속(QRS 에너지 보존 0.26/0.14 vs O50 2.00; 최대 기울기비 0.12/0.19 vs 1.06), O1이 proxy에 가장 가까움(3/3), iMF morph 회복 0.88(창 정규화 0.59), inversion 재현, 타이밍-형태 해리 지속(평탄 모델이 F1 최고·형태 최악) → **REPRESENTATION-ROBUST**. 전역 z iMF는 round 28(best 8)에서 조기 종료(과소학습 가능, 재학습 금지).
- **A7–A9 종합**: 감쇠는 "결정론적 조건 예측기가 과제 관련 구조를 얼마나 보존할 수 있는가"를 따라간다 — ABP는 거의 완전, ECG는 크게 부족.

### 6.6 B1 — 고정 컴퓨트 temporal-gap 커리큘럼 (2026-08-28/29, **중단**)
- Kim et al.의 β(h,s)=1−s+λ·s·(1−h)(λ=1.3046, 적응 가중 후 적용)를 iMF에 단독 이식, 6 run(DaLiA S2/S1, WildPPG × vanilla/curriculum), 조건별 고정 step(S2 66,000 / S1 65,400 / WildPPG 65,482). 완료 2/6.
- S2 pair(최종 체크포인트, 1-NFE): vanilla HR 8.15 / morph .576 / amp .77 / gain 6.42; curriculum 10.29 / .542 / .73 / 4.34(ΔMorph −0.034, HR +26 %). S1 vanilla 173/300 round에서 종료(X0로 자원 재배치, **결과를 본 뒤 결정**이라 판정 없음: "ABORTED / INCOMPLETE — NO CONFIRMATORY VERDICT").
- 부수 발견: vanilla를 역사적 조기종료(round 81) 대신 66k step까지 학습하면 HR 9.58→8.15, gain 4.47→6.42(morph .595→.576) — 고정 bank iMF-MSE 조기종료가 최고 HR 체크포인트를 고르지 않음.

### 6.7 X0 — one-step 실패 분해 (2026-08-29, 분석만, 동결 예측; WildPPG는 **테스트** 서브셋 kjd+ssx 3,907창)
- 질문: 1-NFE 출력이 "완전히 틀려 보일 때" ECG 구조가 근처에 **이동**해 있는가, 아니면 **파괴**되었는가. 4 레벨: raw, 전역 oracle lag ±250 ms, 이벤트 타이밍, GT 앵커 국소 oracle 이동 ±150 ms(당시 유효로 간주; 후에 S1이 oracle_* 통계를 철회).
- WildPPG: oracle-absent 비율 MSE 0.88, O1 0.76, iMF1 0.52, O50 0.38 → 세 one-step 모두 **EVENT-DOMINANT**. 하지만 정렬 후 보존비(p2p/QRS에너지/기울기) MSE .09/.01/.03, O1 .16/.03/.11 vs iMF1 .80/.50/.81, O50 .88/.81/.86 — **균일 감쇠(단봉, 0 근처)이지 선택적 비트 누락이 아님**. 타이밍은 모든 모델에서 가장 작은 결손(매칭 피크 MAE 19–22 ms; MSE/O1은 LOW, iMF1/O50은 MODERATE 라벨; 전역 lag 중앙값 O1/O50 0 ms, MSE −16, iMF1 −8 ms). 결론: **"destroyed, not displaced"**; 시간 모호성 실험(X1) 불필요. iMF-1의 잔차는 형태/배치(비트는 있으나 mis-shaped/extra spike).

### 6.8 X2 — 엔드포인트 barycenter 항등식 (2026-08-30, 동결 체크포인트 새 추론; WildPPG는 테스트 서브셋 3,907창, DaLiA S2/S1)
- 항등식(선행연구): 독립 커플링 + 선형 경로 + MSE 속도회귀 → F*(x0,c)=x0+v*(x0,0,c)=E[x1|c], x0 무관. 질문: **유한 학습된** OT-CFM이 실제로 그러한가.
- 32 소스 seed: O1 소스 분산 보존 R_source 0.0017/0.0017/0.0010(진폭 보존 3–4 %), β≈0, ρ_J 0.041/0.042/0.035, cos(J_x v·d, −d)=+0.999 → **J_x v ≈ −I**, C1–C3 3/3 통과. F̄는 A6 proxy에 3.7–4.3배 더 가깝고(M1 PASS), 같은 모델의 50-NFE 8샘플 평균(B50)에는 A6보다 5.5–14.3배 더 가까움. **M2(WildPPG PCC(F̄,A6) ≥0.60) 실패: 0.545** → **PARTIAL SUPPORT**(임계값 미조정). iMF-1은 소스 의존이 ~10배(진폭 28–39 %, ρ_J 0.42–0.58). t-profile: 소스 민감도가 t=0→0.10에서 매끄럽게 9배 증가(경계 아티팩트 아님).
- 허용된 독해: "동결 OT-CFM은 독립 커플링 flow matching의 알려진 엔드포인트 barycenter 퇴화를 경험적으로 실현 → 1-NFE 조건부 평균형 감쇠의 메커니즘적 설명". 커플링을 바꾸면 고쳐진다는 주장은 하지 않음.

### 6.9 X3-G0 — 커플링 비용-기하 게이트 (2026-08-30, 학습 0)
- 사전-사전등록 설계 감사에서 이미 공개된 테스트 배열과 검증 GT를 읽은 사실 공개 → 1차 추론을 12 train 피험자 4-fold cross-fit으로 이동. minibatch Hungarian 할당(B≤512), 비용 arms RAW/WHITE/HF/RESID, 잔차 PCA+ridge로 소스→잔차 선형 의존 dR²(순열 null 보정).
- RAW dR² FULL 0.339 / QRS 0.062 / HF 0.010(B≥256에서 포화); WHITE 0.010/0.006/0.010; HF 0.001/0.0005/0.050. 즉 **평범한 L2 비용이 이미 QRS 관련 의존을 최대화**하고, 스펙트럼 재가중은 예산을 QRS 모드(잔차 유효차원 d_PR≈4)에서 빼감. 선형 엔드포인트 proxy로 번역하면 QRS 에너지 gap의 21 %, 진폭 11 %, **최대 기울기 1.3 %, HF 0.5 %** 회복 — X0가 지목한 "날카로움"은 거의 못 사고 에너지만 일부 산다. 검증 피험자에서 효과 절반.
- 판정 **INCONCLUSIVE**(GO의 형태 leg가 matched-beat 분모 교란으로 음수). 권고: **커플링 학습 하지 말 것**; 대신 목적함수 레버(t≈0 질량/엔드포인트 손실) 시험.

### 6.10 X4-0 — iMF 이벤트 신뢰도·소스·간격 진단 (2026-08-30/31, 학습 0, an0/k2s)
- 학습 h 분포(2M 샘플): P(h=0)=0.50, 양의 h의 중앙값 0.201(전체 중앙값은 ≈0), **P(h=1)=0**, 최대 0.9297.
- NFE frontier(2,048창×4 seed): iMF morph .665→.777(NFE 8)→.771(50); F1 .410→.431(NFE 4)→.420(50); spurious .569→.472. O50 참조 morph .818, F1 .480. **FEW-STEP SATURATION**은 NFE 8·16에서 발동. 형태는 O50 gap의 73 %를 닫지만 F1은 24.5 %만 닫고 이후 후퇴 → **이벤트 신뢰도가 부족한 수준에서 포화**.
- 소스 민감 이벤트 조직(512창×32 소스): NFE 1에서 seed-pair 이벤트 F1 0.300, 비트 수 SD 1.23, 조건 타이밍 SD 57 ms, F1 SD 0.157 — 4 기준 모두 발동; NFE 8/16에서 30 % 개선 없음(비트 수 SD는 1.78로 악화) → **PERSISTENT**. 즉 PPG를 고정하고 가우시안 소스만 바꿔도 **이벤트의 정체(개수·존재·타이밍)가 바뀐다**.
- 매칭 캘리브레이션: 20–30 ms 지터만으로는 F1 0.91–0.99 → 관측 F1 0.41–0.43은 타이밍 오차만으로 설명 불가(누락/허위 이벤트가 실질 기여).
- 간격 스트레스(h≤0.70)는 발동 안 함(h=1 미검증 → H3의 약한 반증만).
- 판정 **MIXED = CASE A(few-step saturation) + CASE C(persistent source-sensitive event organization)**. 권고: event/rhythm-conditioned iMF(PPG→이벤트 인코더→soft event map→iMF 조건화; GT R은 학습 감독에만). 지연: NFE 8 = 629 ms/batch64(50 NFE의 1/6).

### 6.11 S1 — 지표 타당성 감사 (2026-08-31, 학습 0)
- G1 hard gate: GT 위치에 겹침 없는 QRS 템플릿(T-B)을 찍으면 F1 **0.9993** → PASS(검출기+매처는 정확 배치를 보상). 원래 사전등록(b749339)의 게이트 arm은 전체 비트 템플릿 T-A였고, 결과 전 Amendment 1(dc75079, `docs/S1_METRIC_VALIDITY_AMENDMENT_1.md`)이 겹침 교란을 이유로 게이트를 T-B로 옮겼다; T-A는 0.857(정확 위치인데도 후-QRS 골을 추가 비트로 검출)이라 원안이었다면 FAIL이었을 것 → 현실적 형태에서 검출기 아티팩트가 ~0.14 F1(`docs/S1_G1_METRIC_VALIDITY_REPORT.md`).
- S1.2 zero-parameter DSP floor(PPG 피크 + 상수 지연 296.9 ms): F1 0.2066(chance 0.1325, excess +0.074); 지연 0이면 chance 이하.
- S1.3 joint fidelity(seed 0): iMF-1/4/8/50 F1 excess +.294/+.320/+.320/+.315, O50 +.361, **MSE 회귀기 +.386(최고)** 이지만 oracle QRS 에너지 1 %, HF 0.007, QRS width err 98.6 ms → **"이벤트 F1 단독은 퇴화된 매끄러운 해를 허용"**, 어떤 arm도 이벤트·파형 충실도를 동시에 만족 못함.
- S1.4b: `oracle_corr`의 이득(+0.28~+0.59)이 **무관한 비트와 시프트-피팅해도 0.0007 이내로 재현** → "비트가 있으나 이동" 주장 철회, oracle_* 지표 사용 중단. S1.4c: raw F1의 ~26 %는 준주기 우연.
- S1.5: X4-0의 ppg-shuffle 타이밍 6.7 ms는 67–73 %가 정확히 0인 쌍(비트-동일 소스) → 소스 vs PPG 비교는 공통 척도가 아님.
- S1.6: 두 검출기 일치 F1 0.145는 pantompkins의 상수 85.9 ms 지연 때문(제거 후 0.786; 잔차 불일치는 실재).

### 6.12 C0 / C1 / C2 / M1 — NFE 압축 타깃과 h=0.5 노출 (2026-09-01/02)
- **C0**: A4 iMF 체크포인트, NFE {1,2,4,8}, GT 고정좌표 6개 1차 지표. 2→4: 6개 중 4개 improves, 0 worsens, F1 excess +0.011 → Gate A PASS; 4→8: 3 improves / 3 worsens + beats ratio 악화(0.949→0.916) → Gate B FAIL → **COMPRESSION TARGET = NFE 4**. raw_corr은 어떤 NFE에서도 0.104 이하(모두 약한 arm).
- **C1**: 학습 (t,r) 분포만 바꿈 — B(재현), H25(양의 h 중 절반을 h=0.25로 강제), H50(h=0.5). Stage 1: B 재현이 A4와 **비트 단위 동일**(state_dict 161/161 텐서 일치). Stage 2(NFE 2, vs B): H50 M1–M4 모두 improves(+0.033/+0.034/+0.021/+0.031), F1 excess +0.045; gap closure >1(H50@NFE2가 B@NFE4보다 좋음); H50−H25 특이성은 M1(QRS 에너지 편차), M2(p2p)에서만 → **TARGET h=0.5 EXPOSURE SUPPORTED**. 교란: H50은 101 round(B 66, H25 68)로 ~50 % 더 학습; NFE-2 특이적이지 않음(NFE 4도 비슷하게 개선).
- **C2**: compute-matched(14,409 step 고정) × seed 40–44 × 3 arm = 15 run, ~54 GPU-h 사전등록·구현. **학습 시작 전 보류**(효과의 본질을 M1로 먼저 판별). 지금까지 weight update 0.
- **M1**: 기존 C1 체크포인트에서 QRS-core(|τ|≤80 ms)/peri/background 영역별 고정좌표 오차. H50 vs H25: 집계 비율(QRS 에너지 편차 +0.063, p2p +0.044, 기울기 +0.027)은 개선이지만 **직접 QRS-core 오차는 7개 중 5개가 명확히 악화**(미분 오차 −0.0054, 곡률 −0.0058, raw corr −0.011), 배경 오차도 악화, 국소화 통계 L>0는 배경 악화 때문. 아틀라스: B는 진폭 과소, 두 개입 모두 진폭을 GT 쪽으로 올리며 H50은 비트 사이 기저선이 더 거칠어짐 → **CALIBRATION-ONLY / STRUCTURAL SUPPORT NOT FOUND**. "간격→QRS 구조" 스토리 폐기; C2 자동 시작 금지.

### 6.13 V1 — 14명 NFE별 시각화 + ECG-R→PPG 피크 지연 감사 (2026-09-02)
- Part 1(C1-B 체크포인트, 14 비테스트 피험자 × 4 부위, NFE 1/2/4/8/50): 검증(an0/k2s) 집계에서 raw RMSE·QRS RMSE·미분 RMSE·F1 excess가 **NFE 4 최적**이고 NFE 50은 NFE 2보다 나쁨; 피험자별 최적 NFE 4는 raw RMSE 13/14, QRS RMSE 12/14, 미분 RMSE 12/14, F1 excess는 7/14(8이 4명, 50이 2명, 2가 1명)로 잡음이 큼; 반면 QRS 에너지 편차는 NFE 1 최적, beats-ratio 편차는 NFE 2 이후 단조 악화 → NFE는 파형 오차와 진폭 보정/비트 생산을 맞바꿈. 부위: val F1 excess head .458 / sternum .433 / wrist .258 / ankle .216.
- Part 2(train-only 지연 통계, val 9,950 beats): R→PPG 피크 지연 중앙값 375 ms, **IQR 227 ms, p5–p95 ~500 ms**, 피험자 내 분산이 지배(피험자별 IQR 164–258 ms). 최고 예측기(부위별 상수) 50 ms 커버리지 **0.218**, median AE 172 ms; HR 조건 예측기는 더 나쁨. → **PPG PEAK TIMING TOO UNSTABLE FOR DIRECT CONDITIONING**. 학습된 PPG→R 타이밍 추출기는 미검증(→R1).

### 6.14 R1 — PPG 전역 리듬 관측가능성 프로브 (2026-09-02, 총 ~5 GPU-min)
- Global-TCN(dilated 1…128, RF 2,041 ≥ 1024, 328,897 params, GT R soft field σ=100 ms를 BCE로 학습, PPG만 입력) vs Local-TCN(동일 파라미터, dilation 1 → RF 508 ms). 임계값은 내부 dev(u7y/e61)에서만 선택.
- an0/k2s(8,192창): Global F1@50 **0.620**(입력 무관 floor 0.134, V1 prior 0.218; missing 37.9 %, spurious 43.8 %, 매칭 타이밍 중앙값 23.4 ms); F1@100 .780, @150 .858, @200 .897 → 잔차 대부분은 **50–150 ms 지터**. RR(150 ms 매칭 생존 쌍): median AE **15.6 ms**, Pearson 0.891(DSP foot-interval floor 46.9 ms / 0.517). Global−Local 4/4 지표 CI>0; SHUFFLE/SHIFT 대조에서 붕괴(창 특이 정보 사용). 부위 F1@50 head .760 / sternum .720 / wrist .548 / ankle .452.
- 판정 **EXACT R-TIMING SIGNAL LIMITED** + **GLOBAL RHYTHM SCAFFOLD SUPPORTED**. 과검출(beats ratio 1.105)은 평탄/아티팩트 PPG에서 필드가 낮은 물결로 무너져 NMS가 다수 교차. 권고: 리듬 scaffold 분기 — 단 scaffold는 자체 신뢰도를 가져야 하고, 무너진 필드에 비트를 강제하면 안 됨.
- 사고: 첫 실행이 npz view 누출로 OOM(61 GB) → 배열을 피험자당 1회 바인딩으로 수정(41a1a07), 결과 영향 없음.

### 6.15 R2 — 동결 scaffold → 생성기 전이 프로브 (additive 1×1 adapter, 2026-09-02)
- 동결 생성기 = C1-B 재현(=A4 iMF weights), 동결 R1 Global-TCN, scaffold s = sigmoid(TCN(PPG)) dense field. 유일한 새 파라미터: `ppg_e' = pre_conv_ppg(ppg) + Conv1d(1→128, k=1, no bias)(s)`(128 weights, zero-init). Arms: B, TRUE, SHUFFLE(부위·피험자 내 derangement 파트너의 scaffold), ORACLE(GT-R field, 누출 진단). 각 2,200 step(140,800 창 방문, 조기종료·선택 없음), 동결 iMF 목적함수만.
- NFE 4, 2,048창: F1 excess B 0.3176, TRUE 0.3369(**+0.0194 [+0.0160, +0.0227]**, 사전 최소효과 +0.02에 0.0006 미달 → item 1 FAIL), SHUFFLE 0.3267(TRUE−SHUFFLE +0.0102; **비정렬 scaffold가 이득의 절반**), ORACLE 0.3601(+0.0425). 구조: S1/S2/S3 개선이지만 **S4 −0.0013, S5 −0.0010 악화**(2/5 → item 4 FAIL); 악화는 주입 크기와 단조(SHUFFLE<TRUE<ORACLE). 위상 ablation(+2 s 롤)에서 이득 소멸(위상 의존 확인). 부위: sternum/head/wrist +0.03, **ankle −0.0063**(scaffold가 가장 약한 부위). 타이밍 정밀도(23.4 ms) 불변 → "어느 비트를 찾느냐"만 바뀜.
- 판정 **SCAFFOLD INFORMATIVE, MINIMAL INTERFACE INSUFFICIENT**(C). 해석: 이벤트 정보는 도움이 되나(ORACLE>TRUE>B) PPG scaffold와 가산 인터페이스가 함께 부족; 완벽한 필드도 이 경로로는 +0.043(B의 13 %)만 올리고 S4/S5를 더 해침. 권고: WHEN(배치)과 WHAT(형태) 분리 — 타깃 스트림 쪽 temporal fusion, scaffold 신뢰도로 gating, 생성기 재학습·대형 transformer 금지.

### 6.16 R3 — target-side cross-attention fusion + adaptive gate (2026-09-02/03)
- 훅 감사: 타깃 스트림 텐서 `z_e = pre_conv_target(z)`([B,128,1024], PPG 무관, 파형 정렬)에 삽입. 주의: 각 블록이 `z_e`를 raw residual로 디코더 입력에 그대로 전달(직접 경로 1차 점유율 0.21–0.70) → 순수 "WHEN" 포트가 아님; 채널-균일 가산은 LayerNorm이 소거.
- 모듈(12,768 params = 생성기의 0.279 %): scaffold 토큰 Conv1d(1→32,k7,s4) → 256 토큰, 쿼리 Conv1d(128→32), 고정 sinusoidal PE, 4-head cross-attention 1층(explicit q/k/v/o; SDPA는 forward JVP 불가라 정적 금지), 출력 Conv1d(32→128) zero-init; `z_e' = z_e + (g ⊙) out`. Gate(81 params; s, 33샘플 평균, |Δs|의 pointwise 함수, 초기 g≈0.90). Arms 9개(6개 학습, 각 2,200 step): B, ADD(R2 동결), TF-TRUE/SHUFFLE, GTF-TRUE/SHUFFLE/CONST(게이트 특징을 창 평균으로 대체)/ORACLE, ADD-ORACLE.
- NFE 4 F1 excess: B .3176 / ADD .3369 / **TF-TRUE .3341** / TF-SHUFFLE .3334 / **GTF-TRUE .3582** / GTF-SHUFFLE .3340 / GTF-CONST .3349 / GTF-ORACLE(누출) .8164 / ADD-ORACLE .3601.
  - TF-TRUE vs B +0.0165(U1 크기 미달), vs TF-SHUFFLE +0.0006 unresolved(U2 실패; shuffle share 0.96) → **게이트 없는 cross-attention은 scaffold 타이밍을 거의 쓰지 않음**(+2 s 롤에도 둔감). S4 −0.0028 악화, S5 +0.0024 개선(R2의 S5 악화는 반전).
  - GTF-TRUE vs B **+0.0406 [+0.0353, +0.0458]**, vs GTF-SHUFFLE +0.0242, vs GTF-CONST +0.0233, vs TF-TRUE +0.0241, vs ADD +0.0213; NFE 1에서 이미 +0.0575; 롤 시 이득 소멸(PHASE-GTF .3112 < B). **그러나 S4 −0.0069(9 arm 중 최악), S5 −0.0015, S3 −0.0044 악화**; vs ADD S4 −0.0056; gate는 구조를 보호하지 않음(vs TF-TRUE S4 −0.0041, S5 −0.0039; vs CONST 둘 다 악화). G3/G4/G5 실패.
  - Gate 진단: 창 평균 gate와 scaffold F1@50의 Spearman **−0.491**, 매칭 피크에서 0.128 vs 비매칭 0.471, 약한 scaffold 구간(s<0.35)에서 0.97 → **GATE NOT INTERPRETABLE AS CONFIDENCE**(신뢰도 가중의 역방향; scaffold 모양의 곱셈 마스크 채널로 해석 일관, "confidence-calibrated" 표현 금지).
  - 직접 경로 취소 후 이득 잔존 TF 82 % / GTF 93 % → "through the target stream". ORACLE: GTF-ORACLE vs ADD-ORACLE +0.4564, S4 개선 → **LIFTED**(GT 타이밍이 주어지면 이 인터페이스는 F1 excess 0.82까지 운반; 한계는 인터페이스 용량이 아님, 단 scaffold/신호형태/모듈 중 무엇인지 미식별).
  - 부위(8,192 코호트): GTF−B sternum +0.054, head +0.038, wrist +0.033, **ankle +0.036**(R2의 ankle 퇴행 반전); S4는 모든 부위에서 악화.
- 판정 **EVENT GAIN WITH STRUCTURE TRADE-OFF PERSISTS**(C). R2의 이벤트/구조 trade-off가 더 큰 용량의 다른 인터페이스를 거쳐도 지속하고 S4에서 더 커짐. 공개된 버그: 평가기의 변형 arm 소스 매핑(`endswith("TF")`)과 부위별 gate 부트스트랩 인덱싱 — 둘 다 수정·테스트 고정, 게이트 판정에 영향 없음.
- 권고(미구현; 아래 "R4"는 이 문서가 R3 권고에 붙인 임시 라벨이며 저장소 문서에는 없음): scaffold가 GTF에 들어오는 두 채널을 분리 — GATE-TRUE/TOKENS-SHUFFLE, GATE-SHUFFLE/TOKENS-TRUE, MASK-ONLY(≤209 params; g(s)만 타이밍 운반), GTF-TRUE 3-seed 복제, 동일 U/G 게이트. MASK-ONLY가 이득을 재현하면 cross-attention 불필요이고 구조 비용은 마스크 제약(평활/크기)으로 공략.

---

## 7. 실패·미해결 목록: 무엇이 실패했고 왜

| 시도/가설 | 결과 | 실패·한계의 이유(보고서가 논증한 메커니즘) |
|---|---|---|
| OT-CFM을 그대로 1–2 NFE로 | 붕괴(A1/A0-b) | 동결 OT-CFM의 1-NFE 엔드포인트가 독립 커플링 flow matching의 알려진 엔드포인트 barycenter 퇴화를 경험적으로 실현(X2: 소스 소거 99.9 %, J_x v≈−I; 단 F̄≠E[x1\|c] — WildPPG PCC 0.545로 M2 실패, PARTIAL SUPPORT; "수렴 증명"은 금지 문구). 그 조건부 평균형 출력은 ECG의 QRS를 평탄화(X0: 파괴, 이동 아님; A5/A6: MSE 회귀 proxy의 거동에 근접; X0: 회귀기와 정성적으로 같은 물체). |
| iMF 1-NFE로 완전 대체 | 형태·진폭·HR 회복(A2/A3), 리듬·조건화는 데이터 의존(A4) | 회복되는 축은 파형 구조. 비트 이벤트 신뢰도는 F1 0.41–0.43에서 포화하고 소스 노이즈가 이벤트 정체를 바꿈(X4-0). 어떤 arm도 이벤트·파형 충실도를 동시에 만족 못함(S1.3). |
| "ECG 결과가 일반적 one-step 성질" | 아님(A7) | A7의 선호(favoured) 해석: ABP는 PPG로 거의 결정 → 조건부 평균이 곧 정답 파형, 회귀기 최선(대안으로 전처리 비대칭·스펙트럼 내용도 열거). iMF의 ABP 붕괴는 raw mmHg 스케일(‖y‖/‖e‖ 81배) 탓(A8). |
| "감쇠는 창 단위 정규화 아티팩트" | 아님(A9) | 전역 affine z에서도 동일 패턴, iMF 회복은 오히려 큼. |
| temporal-gap 커리큘럼(B1) | 중단, S2에서 더 나쁨(탐색적) | 6 run 중 2개만 완료, 결과 본 뒤 중단이라 판정 불가. 메커니즘 미제시. |
| minibatch OT 커플링(X3-G0) | 학습 권고 안 함 | 소스→잔차 의존은 생기지만 QRS 에너지만 일부(21 %) 사고 날카로움(1.3 %)은 못 삼; 스펙트럼 비용은 QRS 모드에서 예산을 빼감; held-out에서 효과 절반. |
| 큰 h(=1) 노출 부족이 원인(H3) | 미지지(X4-0), h=0.5 노출은 C1 A판정 | 간격 스트레스 h≤0.7에서 무효과(약한 증거). C1의 H50 이득은 M1에서 **진폭/에너지 보정**으로 재분류(QRS-core 직접 오차는 악화), compute 미매칭(101 vs 66 round), 단일 seed. |
| NFE 2로 압축 | 타깃만 확정(NFE 4, C0/V1) | 압축 방법은 선택·구현 안 됨. NFE 4→8은 파형 오차↓와 진폭 보정·비트 생산↓의 trade. |
| PPG 피크 + 상수 지연으로 R 위치(V1) | 불가 | 지연 IQR 227 ms, 피험자 내 분산 지배; 50 ms 커버리지 0.22. |
| 학습된 PPG→리듬 추출기(R1) | 부분 성공 | F1@50 0.62(지터 50–150 ms, 부위 편차 큼), RR 15.6 ms — 정확 R은 제한, 전역 리듬 scaffold는 지지. 과검출·평탄 PPG에서 붕괴. |
| scaffold 가산 주입(R2) | 이벤트 +0.019(임계 미달), S4/S5 악화 | 주입 크기에 비례해 QRS-core 재형성; 절반은 비정렬 통계; ankle 악화; 완벽 필드도 +0.043 상한. |
| target-side cross-attention + gate(R3) | 이벤트 +0.041, S4 최악 | 게이트 없는 attention은 scaffold를 거의 안 씀; 81-param gate를 더하면 scaffold가 사용 가능해져 이득이 생기지만(+0.0241 vs TF-TRUE, 위상 고정) 구조 비용이 커짐; gate 거동은 scaffold 모양 곱셈 마스크 채널과 일관하나 마스크 채널 vs attention 토큰 중 어느 경로인지는 미식별(R4 권고); gate는 신뢰도가 아님. |
| 지표 자체 | 여러 교정 | DaLiA 비트 지표 무효(비동기); matched morph는 커버리지 조건부; oracle_* 무효(S1); raw F1의 1/4은 우연; 검출기가 현실적 형태에서 ~0.14 F1 손실. |

공통 구조적 한계(모든 단계): 단일 seed 42; 개발 피험자 2명(an0/k2s, 사전 검사됨) 또는 테스트 피험자 1–2명; X3-G0/X4-0 이후 단계는 테스트 kjd/ssx 미평가(그 이전의 A4/A5/A6/A9/X0/X2는 테스트 서브셋 3,907창 사용); R1–R3의 TRUE arm은 GT R로 학습된 추출기를 쓰므로 베이스라인에 없는 타깃 유래 감독을 가짐(비교 시 동등 감독 필요); ORACLE arm은 누출 진단 전용.

---

## 8. 현재 상태와 열린 질문

**현재 상태(2026-09-03)**: R3 결과 커밋 완료, 작업 트리 clean. C2(15-run, ~54 GPU-h) 사전등록·구현만 되고 보류(자동 시작 금지). X4-0 이후 S1/C0/V1/R2/R3가 쓰는 WildPPG iMF 생성기는 A4 체크포인트(=C1-B, state_dict 비트 동일)로 동결되어 있고 R1은 별도의 PPG→리듬 추출기(Global/Local-TCN, 각 328,897 params)를, R2/R3는 그 동결 생성기 위에 소형 모듈(128 / 12,849 params)만 학습했다. 단 A4 이후에도 A7/A8(MIMIC-BP), A9(global-z), B1(fixed-compute), C1(B/H25/H50)은 전체 생성기를 새로 학습했다. 마지막 권고(R3 §"Recommended next step")는 미구현.

**열린 질문**
1. PPG 유래 리듬 정보로 이벤트 대응을 올리면서 QRS-core(S4/S5)를 잃지 않을 수 있는가? 가산(R2)·target-side(R3) 모두에서 trade-off가 지속되고 실제로 사용한 scaffold 타이밍 양에 비례해 커짐.
2. R3 GTF 이득은 gate 마스크 채널과 attention 토큰 채널 중 어디를 통해 오는가(MASK-ONLY 등 R4 arms 필요).
3. 배포 가능 arm의 한계는 Global-TCN scaffold 품질인가, 신호 형태인가, 융합 모듈인가(ORACLE LIFTED가 인터페이스 용량은 배제).
4. iMF의 이벤트 정체가 소스 노이즈에 지속적으로 민감한 것은 목적함수/h-only 조건화의 성질인가, PPG의 R 타이밍 정보 한계인가.
5. 모든 효과 크기의 seed 분산(43/44, 5-fold, C2)이 미측정.
6. NFE 2 압축 방법(residual flow, condition-informed source, distillation, shortcut 등) 미선택.
7. 왜 gate가 신뢰도 가중의 역방향을 학습하는가(측정만, 설명 없음). 왜 OT-CFM 중간 NFE(2, ABP 4)가 불안정한가(미설명).
8. DaLiA 재동기화 프로토콜(비트 지표 복원), kjd/ssx GT 품질 감사(사전 QA 규칙 동결 필요), scaffold 신뢰도 정의.
9. 문서 부기: `EXPERIMENT_LOG.md`는 X4-0 이후 미갱신; README 상태는 A2에서 멈춤; R3 `gate_diagnostics.csv`의 (D) subject_boot 열은 무효로 표기됨.

---

## 9. 자산·재현 정보

- 주요 체크포인트(`outputs/`, git 외): A0-b `a0b_penguin_otcfm_ppgdalia_8s_seed42`, A2 `a2_imeanflow_s5_ppgdalia_8s_seed42`, A3 `a3_{otcfm,imeanflow}_…testS1`, A4 `a4_{otcfm,imeanflow}_wildppg_seed42`(iMF md5 `31c042d291052fbb6dc15263ad316be2`), A6c `a6c_fullbackbone_mse_wildppg_seed42`, C1 `c1_imf_{baseline_replay,h25,h50}_seed42`(B = A4와 state_dict 동일), R1 `r1_global_tcn_seed42`, R2 adapters `r2_{true,shuffle,oracle}_adapter_seed42`, R3 modules `r3_{tf_true,…,gtf_oracle}_seed42`.
- 아티팩트: `artifacts/<stage>/`(decision.json, provenance.json, CSV, 아틀라스). 데이터 매니페스트: `data/manifests/split_*.json`.
- 코드: `src/ppg2ecg/{data,flow,evaluation,models,training,utils}`; 핵심 파일 `flow/imeanflow.py`(iMF), `flow/rhythm_transfer.py`(R2), `flow/rhythm_fusion.py`(R3), `evaluation/rpeaks.py`(검출·매칭), `evaluation/s1_audit.py`(chance floor), `evaluation/event_reliability.py`(subset 선택·샘플링). 테스트 313개(2026-09-03).
- 문서 인덱스(`docs/`): RESEARCH_QUESTION, PREREGISTRATION_V0, DATA_PROTOCOL, METRIC_SEMANTICS, ENVIRONMENT, PENGUIN_AUDIT, IMEANFLOW_AUDIT, WILDPPG_AUDIT, EXPERIMENT_LOG, REPLICATION_SUMMARY, COMMIT_SHA_MAPPING; 각 단계의 `*_PREREGISTRATION.md` / `*_REPORT.md`(A0, A0B, A2, A3_A4, A5, A6, A7, A8, A9, B1, X0, X2, X3_G0, X4_0, S1, C0, C1, C2, M1, V1(`V1_ALL_SUBJECT_STEPWISE_VISUALIZATION_REPORT.md`), R1, R2, R3); 보조 문서 A7_ABP_DATASET_AUDIT, B1_GAP_CURRICULUM_SOURCE_AUDIT, B1_FIXED_COMPUTE_ABORT_NOTE, X3_G0_PREPREREG_DESIGN_AUDIT, X4_0_PREPREREG_VISUAL_AUDIT, S1_METRIC_VALIDITY_AMENDMENT_1, S1_G1_METRIC_VALIDITY_REPORT, C1_BASELINE_REPLAY_GATE_REPORT, C2_DEFERRED_BEFORE_TRAINING, R3_TARGET_STREAM_HOOK_AUDIT.
- 주요 커밋: A2 `80e2229` · A4 `bae0142` · A5 `e0a63ae` · A6 `76fb54a` · A7 `fed5b8c` · A8 `9ff77b5` · A9 `3f93a4d` · B1 abort `a56e0e9` · X0 `5bf255b` · X2 `7e71a10` · X3-G0 `8ad5f40` · X4-0 `bf725cd` · S1 `a29225a` · C0 `d618968` · C1 `94bc795` · C2 prereg `f5120f9` · M1 `5154c17` · V1 `3aa8c4b` · R1 `71aefb4` · R2 `5e064ef` · R3 `55fe1e1`.
