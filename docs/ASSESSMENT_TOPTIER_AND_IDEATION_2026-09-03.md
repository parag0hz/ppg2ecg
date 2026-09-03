# 탑티어 가능성 냉정 평가 + 방법론 아이데이션 (2026-09-03)

기준: `main` HEAD `55fe1e1`(R3 결과), `docs/PROJECT_STATUS_SUMMARY_FOR_LLM.md`. 이 문서는 **권고와 판단**만 담는다. 어떤 실험도 시작하지 않았고, 아래 제안은 모두 새 사전등록 없이는 실행 대상이 아니다.

작성 과정: 독립 리뷰어 3명(생성모델 이론 / 생리·임상 신호 / 방법론·통계), 문헌 조사(웹 검증), 아이디어 설계 3갈래, 판정 에이전트 1명의 결과를 저장소 증거로 대조해 종합했다. GPT의 평가(사용자 제공)는 가설로 취급해 항목별로 검증했다.

---

## 1. 가설이 틀렸는가

**결론: 틀린 것이 아니라, 한 축은 답이 났고 다른 축은 잘못 세팅됐다.**

동결된 질문 "PPG-conditioned ECG 재구성을 임상적으로 의미 있는 morphology와 conditional fidelity를 잃지 않고 one-step으로 줄일 수 있는가?"

- 전반부(one-step 가능?)는 답이 나왔다. 동일 백본에서 목적함수만 iMF로 바꾸면 1-NFE에서 진폭 0.76–0.93, 매칭 형태 0.59–0.87의 회복(A2/A3/A4), 81 ms vs 4,159 ms. C0/V1이 운영점을 NFE 4로 고정.
- 후반부는 세 가지 이유로 잘못 세팅됐다.
  1. **PPG가 결정하지 못하는 것을 요구한다.** PPG는 리듬(RR median AE 15.6 ms)과 대략적 R 타이밍(R1 F1@50 0.62, 잔차 50–150 ms; V1 지연 IQR 227 ms)만 담고, QRS 형태·ST-T·QT는 담지 않는다(생리학적으로도, 측정으로도). ABP 대조(A7/A8)가 이를 결정적으로 보여준다: 조건이 타깃을 거의 결정하는 곳에서는 조건부 평균이 곧 정답이고 one-step 감쇠가 없다.
  2. **"clinically meaningful"은 한 번도 측정되지 않았고 측정할 수도 없는 표현이다.** 128 Hz, lead I, 0.5 Hz 고역, 창 단위 min-max(mV 복원 불가), 건강 피험자, QT/PR/ST/부정맥/전문가 판독 없음. 지표는 R-peak F1@50, QS-골 폭 proxy, 매칭 비트 상관(커버리지 40 %), S4/S5(GT 고정좌표 ±80 ms 미분/곡률)뿐이다.
  3. **이벤트 지표와 형태 지표가 서로 다른 최적 객체를 보상한다.** 결정론적 MSE 회귀기가 F1 excess 최고(+0.386, S1.3)인데 QRS 에너지는 GT의 1 %다. 단일 샘플 이벤트 F1은 MAP 타이밍을, 형태 지표는 날카로운 샘플을 보상한다. 그래서 "어떤 arm도 두 축을 동시에 만족하지 못한다"(S1.3)는 지표 체계의 문제이기도 하다.

**GPT의 "exact timing + exact morphology from PPG는 너무 강하다"는 맞다.** 그러나 GPT의 중심 가설 "z는 WHAT만 모델링해야 하고 이벤트는 조건에 anchoring해야 한다"는 데이터에 대한 가설이 아니라 **출력 객체에 대한 결정론적 선택**이다. p(φ|c)가 넓다면(V1/R1/X4-0가 그렇다고 말한다) 올바른 조건부 샘플러는 소스에 따라 타이밍이 달라야 한다. 프로그램은 X4-0의 소스 민감 이벤트(seed-pair F1 0.30, 비트수 SD 1.2–1.8)를 "결함"으로 채점했지만, **그것이 넓은 타이밍 사후분포에서의 정상적 샘플링인지 캘리브레이션 검사를 한 적이 없다.** 이 한 가지가 논문의 thesis("샘플러를 고쳐라" vs "출력 객체를 골라라")를 가른다. R3의 gate 결과는 경고다: 조건에 anchoring하면 R1 품질(누락 37.9 %, 허위 43.8 %)에 anchoring하는 것이고, 학습된 gate는 신뢰도의 역방향(ρ −0.49)을 배웠다.

---

## 2. 탑티어 가능성 (있는 그대로)

| 항목 | 판단 |
|---|---|
| ML 탑티어(NeurIPS/ICLR/ICML) 현 상태 | **불가.** 방법 없음(R1/R2/R3 보고서 스스로 "instrument, not a method"), 메커니즘(X2)은 prior art의 조건부 재진술(X2 §2 명시; M2 실패로 PARTIAL SUPPORT), 단일 seed, 개발 피험자 2명, 테스트 소진(A4/A5/A6/A9/X0/X2), 외부 베이스라인 0개. |
| 응용 학회(CHIL/MLHC/ML4H findings, JBHI/TBME/Physiol. Meas.) | **조건부 가능.** "사전등록된 negative result + metric validity + 메커니즘" 논문. 단 아래 필수 증거 충족 후. |
| 임상 논문 | **불가**(위 §1-2). "clinically meaningful"은 제목에서 빼거나 측정해야 한다. |
| 진짜 강점 | 사전등록 체인·decision.json·비트 단위 재현·대조군(SHUFFLE/CONST/ORACLE/PHASE/Local-TCN/ABP)·지표 감사(S1). 내용을 대체하지 못하지만 registered-report 형식의 정체성이 된다. |

**가장 뼈아픈 사실(프로그램이 한 번도 돌리지 않은 베이스라인).** S1-G1은 정확 위치에 QRS 템플릿을 찍으면 F1 0.9993임을, R1은 PPG만으로 F1@50 0.62를 보였다. 따라서 "R1 이벤트 + 템플릿 stamping"(학습 0, CPU 분)은 저장소 숫자로 예측하면 **F1 excess ≈ +0.48–0.58**로 모든 생성 arm(B +0.318, GTF-TRUE +0.358, OT-CFM-50 +0.361, MSE +0.386)을 크게 이긴다. 리뷰어 에이전트가 R1·R3 개발 코호트의 공통 360창을 read-only로 조인해 확인했다: macro F1 scaffold 0.620 vs B 0.455 vs GTF-TRUE 0.502, 4 부위 모두 scaffold 우위. **즉 R3의 fusion arm은 자기 입력보다 이벤트 F1이 낮다.** 형태 축에서도 어떤 arm도(50-NFE 참조 포함) 정확 GT 좌표에서 비트 형태를 재현하지 못한다(same_coord_corr 0.09–0.12). 생성 파이프라인이 "검출기 + 템플릿"을 어떤 지표에서든 이긴다는 증거가 저장소에 없다.

**문헌(웹 검증됨).** CardioFlow(ICASSP 2025, NTT)가 이미 "one/few-step rectified-flow PPG→ECG on DaLiA/WESAD"를 선점 → "first one-step" 헤드라인 불가. MAGIC(JBHI 2026, 50-step DiT + gated cross-attention), PG-LRF(arXiv 2605.12541, 시뮬레이터 내부 cardiac phase, 가우시안 소스), PPGFlowECG(2509.19774, latent RF, T=5–10) 확인. 어느 것도 1-NFE·이벤트-구조 trade-off·chance-corrected F1·조건부 평균 대조를 보고하지 않는다. "WHEN vs WHAT / rhythm vs morphology" 어휘는 SE-Diff, ECG-RAMBA가 이미 쓴다 → **프레이밍이 아니라 측정을 주장해야 한다.** 새로 인용해야 할 것: InDI(TMLR 2023, regression-to-the-mean), PMRF(ICLR 2025), Flow Map Denoisers(2606.19802, MMSE–perceptual knob — C1→M1 "calibration-only"의 가장 깔끔한 언어), Velocity Deficit(2605.14819).

### GPT 주장 항목별 검증

| GPT 주장 | 판정 | 근거 |
|---|---|---|
| 질문은 좋으나 "정확 타이밍+정확 형태"는 과함 | 지지 | V1/R1/X4-0/S1.3 |
| 2단계 실패 분해가 최강 발견 | 부분 | 1단계는 prior art 재진술(X2 §2, M2 실패); 2단계의 "morphology recovers"는 과함 — 회복되는 것은 진폭/기울기/QRS 에너지이고 정확 좌표 형태(same_coord 0.1)가 아님; 소스 민감성이 결함인지 posterior sampling인지 미검증 |
| 현 상태로 탑티어 불가, R2/R3는 방법 아님 | 지지 | 보고서 원문 |
| MAGIC/PG-LRF/PPGFlowECG가 조건화 구조 공간을 점유 | 부분 | 존재 확인; 단 더 치명적인 선행은 CardioFlow(one/few-step PPG→ECG 선점) |
| "z는 WHAT만" 중심 가설 | 부분 | 현 iMF의 기술로는 맞음(X4-0); 데이터 가설로는 반증 불가하며, R3 gate가 경고 |
| Phase-canonicalized MeanFlow + E[W(y)\|c] 이론 | 미지지 | 참 warp에서만 성립; 추론 시 PPG 위상(R1 정확도)으로 inverse warp하면 blur가 misplacement로 바뀜; sharpness는 iMF 결함이 아님(X4-0); 비트-위상 좌표는 오랜 prior art(ECGSYN, Zhu 2021, PG-LRF, TimewarpVAE) |
| 결정론적 이벤트 + 확률적 형태 + renderer | 부분 | "R1 이벤트 + 템플릿"의 비자명 사촌; 형태에서 템플릿을 이겨야만 가치 — PPG가 per-beat 형태를 담는지 미측정 |
| condition-informed source | 미지지 | OT-CFM에선 X2 항등식 그대로(ε ⟂ x1\|c면 E[x1\|c]); iMF에선 소스 = z_e = R3가 주입한 텐서 → 같은 additive 주입 |
| R4(채널 분리) 먼저 | 부분 | R3 자체 권고이나 우선순위 오류 — +0.024 효과 해부보다 ~+0.13 gap·템플릿 baseline·타이밍 상한이 먼저 |
| 증거 gap 목록 | 부분 | "R1이 ECG R 라벨 사용"은 누출이 아니라 정당한 감독(supervision mismatch 문제, 고칠 수 있음); 누락: v0→A2 성공 기준 변경(iMF-1은 v0 마진 실패), S4/S5 MID 없음, 416 CI 미보정, 타일링된 창 부트스트랩(개발 2,048창 중 248창이 ECG 타깃 공유), X0 판정 트리거가 철회된 oracle 지표, 매칭 형태 커버리지 40 %, GT 품질(kjd noisy; 두 검출기 0.786 일치) |
| "interface capacity는 병목 아님"(ORACLE 0.82) | 미지지 | R3 보고서가 그 독해를 명시적으로 거부; 배포 arm은 자기 입력(scaffold median F1 0.70)보다 낮은 F1을 냄 → 노이즈 입력을 신뢰도 가중하지 못하는 것은 인터페이스/귀납편향 한계 |
| barycenter 스토리가 novel | 틀림 | X2 §2 "Prior art — what is NOT novel" |

---

## 3. 논문이 될 수 있는 가설 (재정의)

**Thesis(제안):** *조건부 관측가능성(conditional observability)이, 스텝 수가 아니라, one-step PPG-조건 ECG 생성기가 무엇을 anchoring할 수 있는지를 결정한다.*

정량 이론(검증 가능): 이벤트 구조 타깃 y = shift_φ(m) + noise (φ 타이밍, m 정준 형태)에서
E[y|c] = ∫ shift_φ E[m|c,φ] p(φ|c) dφ — 조건부 평균은 정준 형태 평균을 타이밍 사후분포로 컨볼루션한 것. X2 항등식은 OT-CFM 1-NFE가 여기에 착지한다고 말한다. 저장소가 이미 검증 가능한 두 예측:
- (a) X0의 감쇠(p2p 0.09–0.16, 기울기 0.03–0.11)가 측정된 사후분포 폭(V1 IQR 227 ms, R1 잔차 50–150 ms, X4-0 SD 57 ms)과 QRS 폭으로 설명되는가.
- (b) X4-0의 seed-pair F1 0.30·비트수 SD가 X4-0E 지터 캘리브레이션 하의 posterior sampling으로 설명되는가.

둘 다 맞으면 "one-step collapse = timing-posterior blur"가 결과가 되고, ABP는 좁은 사후분포의 깨끗한 반례(A7/A8), iMF의 소스 민감 이벤트는 결함이 아니라 calibrated sampling이 된다. 그 다음에야 "출력 객체 선택(MAP 타이밍 + 샘플 형태 vs posterior 샘플)"을 임상 근거로 정할 수 있다. 이 방향은 새 아키텍처 없이도 논문의 뼈대가 되며, 프로그램의 모든 단계가 왜 필요했는지 한 줄로 연결된다.

권장 논문 뼈대(판정 에이전트 종합): (1) 관측 한계 측정 — PPG는 RR 15.6 ms, R 타이밍 ~0.62 F1@50, 템플릿 이상의 형태는 **아래 1위 실험으로 측정**; (2) 지표 타당성 — 이벤트 F1은 퇴화 해를 허용하고 1/4은 우연; (3) negative result — 어떤 one-step/50-step arm도 그 한계 위에서 측정 가능한 형태를 더하지 못함; ECG 특이적 결과로 서술(A7 ABP 대조), "clinically meaningful" 삭제.

---

## 4. 방법론 아이데이션 (판정 순위, 모두 미실행·권고)

모든 후보가 넘어야 하는 베이스라인 두 개: **(i) R1 이벤트 + 템플릿 stamping**(supervision-matched null method; 예상 F1 excess +0.48–0.58), **(ii) R3 GTF-TRUE**(+0.358, S4 0.329). 형태 축에서는 (i)의 T-A 스탬프를 S3–S5·매칭 형태에서 이겨야 하고, 이벤트 축에서는 (i)를 이길 수 없다(타이밍을 상속하므로).

| 순위 | 아이디어 | 핵심 가설 | 첫 실험(가장 싼 반증) | 기각 수치 | 비용 |
|---|---|---|---|---|---|
| 0 (필수) | **R1 이벤트 + T-B/T-A 스탬프 채점 + S4/S5 지터 캘리브레이션** | 자명한 베이스라인이 모든 생성 arm을 이긴다; 고정좌표 S4/S5가 배치형 방법에 정보를 주는가 | 동결 2,048창·R2/R3 평가기로 R1 이벤트·GT 위치 스탬프 채점; GT 스탬프에 지터 SD 0–4샘플·탈락 0–40 % | 스탬프 F1 excess < 0.358이면 전제 붕괴; T-B@정확GT의 S4가 B 0.322보다 명확히 낮지 않으면 S4/S5 게이트 무정보 → 커버리지 명시 per-beat S4 사전등록 | CPU 분 |
| 1 (8) | **Oracle-timing WHAT-observability probe** | 타이밍이 정확할 때 PPG가 템플릿 이상의 per-beat 형태를 담는가 | GT R 앵커에서 (i) 템플릿 (ii) MSE beat regressor (iii) beat-level iMF(83샘플) 비교 + PPG-shuffle 대조; 고정좌표 상관·S4/S5·QRS폭·진폭비 | (ii)(iii)가 (i)를 못 이기거나 shuffle drop CI가 0 포함 → PPG는 템플릿 이상 형태를 안 담음, 모든 형태-측 방법 폐기 | <1 GPU-h |
| 2 (7.5) | **Best-of-K 소스 선택**(R1 scaffold = verifier; SEL-ORACLE/SHUFFLE/MIN/MEAN-K 대조) | K개 샘플 중 더 나은 이벤트 셋이 있고 PPG-only verifier가 고를 수 있다 | 32 소스 × 2,048창, NFE 4/1, 검출기-무관 8–40 Hz 포락선 상관 점수 | SEL-ORACLE−B < +0.02 → 샘플에 더 나은 이벤트 없음; SEL-SCAFFOLD ≈ SEL-SHUFFLE이면서 ORACLE만 큼 → scaffold-limited | <1 GPU-h, 학습 0 |
| 3 (7) | **Timing-head 상한** | F1@50 0.62는 PPG의 성질인가 5-GPU-min 프로브의 성질인가 | σ∈{50,75,100} ms × 3 seed × {BCE field, foot 오프셋 회귀}, 동일 dev 임계 | 어떤 구성도 0.65를 못 넘고 3-seed CI가 0.62 배제 못함 → PPG-only 이벤트 상한 ≈ +0.49 excess 확정 | ~1 GPU-h |
| 4 (6.5) | **BEATSEG / factorized**(GPT #2 실현형) | R1 이벤트 + beat-level one-step 형태 + overlap-add renderer | 1위 통과 후에만; GT 앵커+지터 증강 학습, R1/GT/SHUFFLE 앵커 평가 | 매칭 형태·S2·S3가 T-A 스탬프 위로 안 풀리거나 F1이 스탬프보다 0.01 이상 낮음 | 1.5–2.5 GPU-h |
| 5 (6) | **FIX-WHEN/FIX-WHAT 소스 부분공간 진단** | 이벤트 정체를 움직이는 소스 방향이 저차원인가 | X4-0C 512창×32소스에서 스펙트럴/Jacobian top-k/이벤트회귀 P로 공유·변동 분리 | k≤64·f_c≤8 Hz의 어떤 P도 seed-pair F1 ≥0.60·비트수 SD<0.75 못 달성 → 소스에서 WHEN/WHAT 분리 불가 | ~1 GPU-h |
| 6 (6) | **h=1 self-distillation**(+H100-JVP 노출 대조 + B+2200) | NFE 1→4 gap의 일부는 학습 안 된 경계 질의의 비일관성 | 2,200 step fine-tune 3 arm, NFE 1 평가 + M1 영역 감사 | CONSIST가 B+2200 대비 3/4 지표 개선 못함, 또는 H100-JVP와 unresolved(노출 효과), 또는 QRS-core 미분 악화(calibration-only) | <3 GPU-h |
| 7 (5.5) | **condition-informed source**(GPT #3) | 소스 평균에 scaffold를 넣으면 이벤트가 anchoring됨 | retrain-free e' = e + α·A(s), LEAK 대조(B 출력에 사후 스탬프) 필수 | ORACLE−LEAK CI가 0 포함(누출 스탬핑) 또는 S4/S5가 α에 단조 악화(R2 재현) | <1 GPU-h |
| 8 (5) | **event-envelope aux loss**(GT는 학습만) | 손실 수준 anchoring은 입력 주입의 trade-off를 피함 | AUX-ENV/AUX-RFIELD/AUX-MSE(양성대조)/B+2200 fine-tune | F1 excess < +0.02 또는 S4/S5 악화(trade-off 내재) 또는 HF/spurious 악화(게임) | ~2 GPU-h |
| 9 (5) | **joint(unfrozen) 학습 vs frozen adapter** | R2/R3 trade-off가 adapter 아티팩트인가 | train_r3_fusion 생성기 unfreeze, JOINT-GTF/SHUFFLE/MASK-ONLY/B+2200 | F1↑인데 S4/S5 CI<0 → "TRADE-OFF INTRINSIC", 입력 조건화 계열 종료 | ~2.5 GPU-h |
| 10 (4) | **CANON-WARP**(GPT #1 본체) | 전창 위상 canonicalization + inverse warp | 1·5위 통과 후에만; C2 예산 학습, R1/GT/롤/SHUFFLE 위상 평가 | BEATSEG 대비 S1/S2 무개선, QRS폭 오차 >35 ms, R1-위상 arm S2<0.15 | 4–8 GPU-h |
| 11 (3.5) | **R4 채널 분리**(MASK-ONLY 등) | GTF 이득의 운반 채널 | R3 U/G 게이트 재사용 | MASK-ONLY가 이득 대부분 재현하되 S4 CI<0 → 마스크가 trade-off 운반 | ~1.5 GPU-h |

**실행 순서(권고):** 0 → 1 → 2 → 3 → (1 통과 시) 4/5 → 6 → 7/8/9 → 10 → 11. 0–3은 학습이 거의 없고 정보량이 가장 크다. 어떤 것도 §5의 증거 패키지 없이는 논문 주장이 될 수 없다.

---

## 5. 제출 전 필수 증거 (무엇을 하든)

1. **템플릿 베이스라인**(0위) 채점과, 그에 대한 형태-매칭 F1 상한(T-A 0.857)을 기준으로 한 재해석.
2. **≥3 seed**(가능하면 5): 주장에 관여하는 모든 학습 arm. C2 템플릿(14,409 step, checkpoint_last) 재사용. 단일 seed의 improves/worsens 판정은 복제 전까지 보류.
3. **진짜 fresh test**: kjd/ssx 1회 평가(GT QA 규칙을 어떤 모델의 테스트 성능도 보기 전에 동결; kjd는 noisy) 또는 개발 중 열지 않은 두 번째 동기화 데이터셋(DaLiA는 재동기화 프로토콜 실행 후에만). 창이 아닌 피험자 단위 CI, 부위×시간블록 클러스터 부트스트랩(개발 창 12 %가 ECG 타깃 공유).
4. **supervision-matched 비교 + 외부 베이스라인**: 순수 생성기에 동일 GT-R aux loss, PENGUIN-50과 MSE 회귀기에도 scaffold, CardioFlow/PPGFlowECG/RDDM 중 1개를 같은 split·지표로, iMF 공식 recipe(EMA, lr 1e-4, aux head) 민감도 arm, PPG→ECG 조건화 구조 related-work.
5. **판정 재유도**: X0를 철회된 oracle 트리거 대신 GT 고정좌표·전창 보존비로; 매칭 형태 옆에 all-GT-beat 통계와 커버리지; v0 비열등 마진(HR +1.0 bpm, corr −0.05)을 A2 회복률 규칙과 나란히 제시하고 iMF-1이 전자를 실패함을 명시; DaLiA 비트 지표 주장 삭제.
6. **통계**: 이득과 해악 양쪽에 MID가 있는 단일 1차 endpoint(예: F1 excess ≥ +0.02 AND S4가 MID 이내), 1차 family 내 Holm 보정; "CI가 0 배제"를 구조 악화의 정의로 쓰지 말 것.
7. **임상 측정 또는 표현 삭제**: onset/offset QRS duration(7.8 ms 해상도 명시), QT/QTc, PR, ST/T(전역 affine mV 보존 표현, A9로 가능), 서브셋 수동 판독, wrist-only 별도 보고, 가속도계 기반 SQI/움직임 층화, 다운스트림 리듬 과제(AF/ectopy). 비트 수가 seed에 따라 SD 1.2–1.8 변하는 것은 임상 독자에게 헤드라인 안전 수치다.
8. **관측 상한 확정**(3위·1위)과 **결함 vs posterior sampling 캘리브레이션 검사**(§3-b).
9. X2/X3-G0가 권고했으나 안 돌린 **endpoint 노출 대조**(6위의 H100-JVP arm).
10. 부기: EXPERIMENT_LOG(X4-0에서 멈춤), README(A2에서 멈춤), 사전등록/decision.json/amendment/사전-사전등록 공개 문서를 보조자료로.

---

## 6. 한 문단 요약

가설은 틀리지 않았지만 "PPG로 정확한 ECG"라는 목표는 PPG가 담지 않는 정보를 요구했고, 프로그램의 실험이 그 한계를 스스로 측정했다. 현 상태는 ML 탑티어가 아니다: 방법이 없고, 메커니즘은 선행연구의 재진술이며, 단일 seed·개발 2명이고, 한 번도 돌리지 않은 "검출기 + 템플릿" 베이스라인이 모든 생성 arm을 이길 것으로 예측된다. 가능성이 있는 방향은 (1) 조건부 관측가능성 thesis를 정량 이론(타이밍 사후분포 컨볼루션)으로 세우고 X0/X4-0/A7로 검증하는 것, (2) 템플릿을 형태에서 이기는 방법을 찾되 그 전제(PPG가 템플릿 이상의 per-beat 형태를 담는가)를 1 GPU-h 이내의 oracle 실험으로 먼저 확인하는 것이다. GPT의 phase-canonicalization은 이론 스토리가 참 warp에서만 성립하고 추론 시 R1 정확도를 상속하므로 후순위이며, R4는 마지막이다.
