# 기술 명세 — 모델 구조 · 데이터셋 · 전처리 · 학습 하이퍼파라미터 · 평가 프로토콜

작성일 2026-09-03 · 기준 커밋 `55fe1e1` (R3 결과 커밋) · 저장소 `/home/kwy00/ppg2ecg-one-step`

이 문서는 현재 프로그램에서 **실제로 학습·평가에 사용된 모든 구성**을 코드와 실행 기록에서 읽어 한 곳에 모은 것이다. 다른 LLM 또는 공동 연구자가 이 문서만으로 각 실험을 재현·검증할 수 있도록 값·형상·라인 참조를 그대로 적었다. 실험의 서사(무엇을 시도했고 왜 실패했는지)는 `docs/PROJECT_STATUS_SUMMARY_FOR_LLM.md`, 연구 판단은 `docs/ASSESSMENT_TOPTIER_AND_IDEATION_2026-09-03.md`에 있다.

**목차** — [§0 범위·권위](#0-범위와-권위-규칙) · [§1 실행 환경](#1-공통-실행-환경-모든-런) · [§2 데이터셋](#2-데이터셋) · [§3 전처리·분할·누수검사](#3-전처리) · [§4 모델 구조](#4-모델-구조) · [§5 목적함수·샘플러](#5-목적함수와-샘플러) · [§6 학습 하이퍼파라미터](#6-학습-하이퍼파라미터) · [§7 평가 프로토콜](#7-평가-프로토콜) · [§8 기록 간 불일치](#8-기록-간-불일치-코드--train_metajson이-권위) · [§9 남은 모호성](#9-남은-모호성-문서-작성-시점에-저장소만으로-해소-불가) · [§10 자산 색인](#10-자산-색인)

**검증 이력**: 초안 작성 후 6개 영역(데이터셋 / 백본 / 보조모델 / 목적함수 / 학습 / 평가)을 각각 저장소 원본과 독립 대조했다 (총 1,132개 사실 주장 확인). 제기된 17건은 전부 적대적 재검증을 거쳐 확정된 뒤 본문에 반영되었다: seed.py 라인 참조, `preprocess_windows`의 저역통과 분기, A7/A8 `test_inputs.npz` 키, A5/A6의 `n_step=1`, `UNIFORM[n]` bit-exact 범위(n ≤ 16), `E[h]`의 출처 MC, gradcheck 진폭비 상한, 스모크 epoch 범위, R1 best 라운드의 1-based 환산, OT-CFM `--gen-diag-every` 예외, 존재하지 않는 B1-WildPPG 런 2곳, A9 평가기의 시드 다양성 예외, 지연 측정 워밍업 2 vs 3, R2/R3 회귀 검사 STOP 범위, `oracle_metrics_used` 기록 범위, GT 박동 40,523의 출처.

---

## 0. 범위와 권위 규칙

| 항목 | 규칙 |
|---|---|
| 권위 순서 | (1) 코드: `src/ppg2ecg/**`, `external/PENGUIN` @`6cd70cdefb91f10efeb8dce34019b5067cb25344` (읽기 전용) → (2) 트레이너가 직접 기록한 `outputs/<run>/train_meta.json`, `training_summary.json`, `training_log.csv`, 체크포인트 메타(`model_cfg`, `imf_cfg`, `selection`) → (3) `artifacts/*/provenance*.json`, `train_provenance_*.json` (R1–R3) → (4) `outputs/<run>/config.yaml`, `provenance.json` (`scripts/preflight_a0.py`가 **preflight의 argv**로 기록; 학습 명령과 다를 수 있음, §8) → (5) `docs/*.md` 산문 |
| 파라미터 수 규약 | torch `numel` (complex64 원소 1개 = 1). 실수 스칼라 기준이면 백본 5,095,043 (§4.1.8) |
| 라운드 인덱스 | 본문은 **1-based** ("best 46"). JSON의 `best_epoch`은 0-based (45) |
| 시간 규약 | OT-CFM: t=0 노이즈, t=1 데이터. iMF: **t=1 노이즈, t=0 데이터** (반대) |
| 텐서 규약 | 모든 신호 `[B, 1, 1024]` float32, T = 8 s × 128 Hz = 1024. numpy 지표 입력은 `[n, 1024]` float64 |
| 절대 규칙 | WildPPG 테스트 피험자 `kjd`, `ssx`는 X4-0 이후 어떤 분석에서도 로드하지 않음 (§7.9). A4 iMF 체크포인트 md5 `31c042d291052fbb6dc15263ad316be2` 불변 |

---

## 1. 공통 실행 환경 (모든 런)

| 항목 | 값 | 출처 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090 × 1, 32,109 MiB (`torch.cuda.get_device_properties`; `docs/ENVIRONMENT.md`는 32,607 MiB, prereg 산문은 31.4 GiB), compute capability 12.0, 드라이버 580.173.02 | `outputs/*/provenance.json: hardware` |
| 소프트웨어 | PyTorch 2.11.0+cu130, CUDA 13.0, cuDNN 91900, Python 3.13.9, numpy 2.3.5, scipy 1.16.3, neurokit2 0.2.12 | `provenance.json`, `pyproject.toml`, `docs/ENVIRONMENT.md` |
| 정밀도 | 전부 fp32. AMP/bf16/GradScaler/`torch.compile` 없음. `allow_tf32`를 건드리는 코드 없음 → PyTorch 기본값 (preflight 기록: `tf32_matmul false`, `cudnn_tf32 true`) | `scripts/preflight_a0.py:150`, 모든 `config.yaml: precision` |
| 결정성 | `seed_everything(seed, deterministic=True)`: `random`/`numpy`/`torch.manual_seed`/`cuda.manual_seed_all`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=True)` | `src/ppg2ecg/utils/seed.py:11-20` |
| 평가기 결정성 | R2/R3 평가 스크립트는 cuDNN 기본값(`cudnn_deterministic: false`)으로 실행 (학습은 true). 패리티 검사는 통과했으나 평가 수치의 비트 재현은 보장되지 않음 | R3 보고서 |
| 업스트림 핀 | `external/PENGUIN` 커밋 `6cd70cd…` (main, 2025-09-18). 매 런 시작 시 `assert_upstream_pinned()`가 커밋·클린 트리 확인. `import_upstream_penguin()`이 `external/PENGUIN/src`를 `sys.path[0]`에 삽입 | `src/ppg2ecg/utils/upstream.py:15, 29-53` |
| 옵티마이저 (전 런 공통) | `torch.optim.AdamW(params, lr, weight_decay)`; betas/eps 미지정 → (0.9, 0.999), 1e-8. LR 스케줄·grad clipping·EMA **없음** (upstream `train.yaml`의 `ema_decay 0.999`는 어떤 코드도 읽지 않음) | `train_a0.py:144`, `train_a2.py:131`, `train_a5.py:93`, `train_r2_adapter.py:161`, `train_r3_fusion.py:166`, `r1_train_probe.py:82` |
| 데이터 상주 | train/val 배열 전체를 float32로 GPU에 1회 적재; `DataLoader(TensorDataset(x_gpu, y_gpu), batch_size=64, shuffle=True, generator=gen, drop_last=False)` | `train_a0.py:136-148`, `train_a2.py:122-136` |

---

## 2. 데이터셋

### 2.1 세 데이터셋 요약

| | PPG-DaLiA | WildPPG | MIMIC-BP |
|---|---|---|---|
| 출처 | Reiss et al. 2019, UCI #495, CC BY 4.0 | Meier·Demirel·Holz, NeurIPS 2024 D&B (arXiv 2412.17540), 데이터 CC BY-NC-SA 4.0로 취급 | Sanches et al., Sci. Data 11:1233 (2024), Harvard Dataverse doi:10.7910/DVN/DBM1NF v2.2, ODbL 1.0 |
| 입력 | 손목 BVP (Empatica E4) 64 Hz | 녹색 PPG (530 nm) 128 Hz, **4부위(sternum/head/wrist/ankle)를 각각 별도 샘플로** | 손가락 PPG 125 Hz (raw ADC 0–4) |
| 타깃 | 가슴 ECG (RespiBAN) 700 Hz | 흉골 lead-I ECG 128 Hz, int32 ADC (±2^17, 단위 미보정), 4부위에 ×4 타일 | ABP 125 Hz, mmHg |
| 윈도우 | 8 s 비중첩, 꼬리 버림 | 8 s 비중첩, 부위별 | 30 s 세그먼트당 8 s × 3 (마지막 6 s 버림) |
| 원시 윈도우 길이 | PPG 512 / ECG 5,600 | 1,024 / 1,024 | 1,000 / 1,000 |
| 리샘플 후 | 1,024 @128 Hz | 1,024 | 1,024 |
| 피험자 | 15 | 16 | 1,524 |
| 총 윈도우 | 16,181 | 389,355 (+861 드롭) | 137,160 (드롭 0) |
| 분할 | P0: 13 / S11 / **S2**; A3: 13 / S11 / **S1** | 12 / **an0,k2s** / **kjd,ssx** (seed 42) | 공식 1,100 / 195 / 229 |
| train / val / test 윈도우 | 14,025 / 1,131 / 1,025 (P0); 13,899 / 1,131 / 1,151 (A3) | 293,271 / 49,200→3,785 / 46,884→3,907 | 99,000 / 17,550→3,510 / 20,610→3,435 |
| 처리본 경로 | `data/processed/v0_8s/S{n}.npz` | `data/processed/wildppg_8s/<id>.npz`, `wildppg_8s_prenorm/<id>.npz` (A9) | `data/processed/mimicbp_8s/<pid>.npz` |
| npz 키 | `x, y, window_start_s, subject` | `x, y, site, window_index, window_start_s, subject` | `x, y, segment_idx, window_start_s, label_sbp, label_dbp, pid` |
| 알려진 문제 | 손목–가슴 **비동기** (R→다음 맥파 지연 500–900 ms, 분당 ~20 ms 드리프트) → 박동 정렬 지표는 데이터 인공물 측정 | 잡음 ECG 피험자 fex/kjd/p5d 유지; 미문서화 flat run (2^-19); an0 notes의 wrist/ankle 오기 | ICU 동맥 라인 인구, 신생아 p028331 train 포함; 세그먼트 간 절대 시간 없음 |

### 2.2 PPG-DaLiA 상세

- 원시 파일 `data/raw/PPG-DaLiA/PPG_FieldStudy/S{1..15}/S{n}.pkl` (pickle latin1). 사용 채널: `signal.chest.ECG` 700 Hz (타깃), `signal.wrist.BVP` 64 Hz (입력). `rpeaks`, HR `label`, `activity`는 로드되지만 처리본에 쓰지 않음. zip sha256 `5772387956e34e2e…` (2,865,111,320 B).
- 피험자별 8 s 윈도우 수: S1 1,151 · S2 1,025 · S3 1,092 · S4 1,143 · S5 1,163 · S6 656 · S7 1,167 · S8 1,010 · S9 1,070 · S10 1,331 · S11 1,131 · S12 989 · S13 1,142 · S14 1,119 · S15 992 (합 16,181; 총 35.97 h). ECG/BVP 길이는 모든 피험자에서 초 단위로 일치.
- 윈도잉: `sliding_window_view(x, win)[::win]` (`dalia.py:74-76`), 두 스트림 공통 t=0, `align="strict"` (개수 불일치 시 예외; 실제 불일치 없음). 유한성 assert, 드롭 0.
- 처리본 `MANIFEST.json`: built 2026-08-25T15:10, `total_windows 16181`, manifest sha256 `57ce09d8…`. 파일별 sha256은 매니페스트에 기록 (예: S1 npz `60489ddf…`).
- 참고용 `data/processed/upstream{,_8s}/PPG-DaLiA/subject{0..14}.pkl`은 미수정 upstream `preprocess.py` 출력 (4 s / 8 s), 학습에 사용 안 함. 8 s 전 피험자 bit-exact 패리티 (`processed_parity_upstream_8s.json`).

### 2.3 WildPPG 상세

- 원시: ETH polybox `WildPPG_Part_<id>.mat` 16개 + `github_image.zip` = 19,608,554,456 B, MATLAB v5, `scipy.io.loadmat`. 참가자 `an0 e61 fex k2s kjd l38 n31 ngh p5d p9p qm9 ssx trh tz8 u7y w4p`. 4개 동기화 장치, 각각 `acc_{x,y,z}` 128 Hz, `ppg_g/ir/r` 128 Hz float64 [0,1], ECG는 sternum만 (int32, [−131072, 131071]). NaN 없음, 기록 갭은 상수 채움 (PPG=1.0, acc=0.0). ECG 총 216.79 h (×4 부위 = 867.15 h = PENGUIN `duration`).
- 로더 `load_wildppg_participant` (`wildppg.py:20-36`)는 PENGUIN `load_data.py:29-51`과 동일한 중첩 dict 파싱. 입력 `ppg_g` 4부위, 타깃 `sternum.ecg`. 부위별 `n = min(len(ppg_w), len(ecg_w))`, PPG는 **site-major** 연결(sternum 전부 → head → wrist → ankle), 같은 sternum ECG 윈도우를 ×4 타일. `site`, `window_index`(부위마다 0부터), `window_start_s = window_index×8` 저장 (PENGUIN은 site를 저장하지 않음).
- 빌드 드롭 규칙 (`build_processed_wildppg.py:46-51`): 전처리 전 `isfinite & std>0` (PPG·ECG 모두), 전처리 후 `isfinite`. **861 윈도우 (0.22 %) 드롭** = 상수 채움 갭 윈도우 (PENGUIN은 유지 — 문서화된 이탈).
- 참가자별 윈도우 (두 처리본 동일): an0 22,183 (드롭 1) · e61 22,228 · fex 24,072 · k2s 27,017 (399) · kjd 24,094 (6) · l38 23,714 (262) · n31 23,840 · ngh 26,440 · p5d 26,025 (7) · p9p 22,508 · qm9 23,541 (155) · ssx 22,790 (6) · trh 25,912 · tz8 26,115 (25) · u7y 23,856 · w4p 25,020. 부위별 개수는 갭 때문에 약간 다를 수 있음 (예: k2s 6,854/6,852/6,457/6,854).
- `wildppg_8s` MANIFEST built 2026-08-26T02:19:20 (sha256 `5cd54bbf…`); `wildppg_8s_prenorm` built 2026-08-28T01:21:54, `ecg_normalization: none` (sha256 `86d19e8c…`). 두 디렉터리의 `x`(PPG)·`site`·`window_index`는 bit-identical; `y`만 다름 (prenorm은 리샘플 + 0.5 Hz HP 후 원시 ADC 단위, e61 예: min −180,889, max 199,097, std 5,835.7).
- 분할 `data/manifests/split_a4_wildppg_seed42.json` (sha256 `bc168144…`): 정렬된 id → `random.Random(42).sample` → val = 처음 2, test = 다음 2, train = 나머지 12. WildPPG 학습 전 동결.

| 분할 | 피험자 | 윈도우 |
|---|---|---|
| train (12) | e61 fex l38 n31 ngh p5d p9p qm9 trh tz8 u7y w4p | 293,271 |
| val (2) | an0, k2s | 49,200 → stride 13 → 3,785 |
| test (2) | kjd, ssx | 46,884 → stride 12 → 3,907 (kjd 2,008 / ssx 1,899) |

### 2.4 MIMIC-BP 상세

- 원시 `data/raw/MIMIC-BP/{ppg,abp,ecg,resp,labels}/p<ID>_<kind>.npy` + `train/val/test_subjects.txt` (Python 리스트 리터럴, `ast.literal_eval`). 피험자당 `[30 세그먼트, 3750 샘플]` float64 @125 Hz; `labels [30,2]` = 세그먼트별 중앙값 (SBP, DBP). ABP 범위 28.4–188.7 mmHg. ECG/RESP 미사용.
- 윈도잉 (`mimicbp.py:40-52`): 세그먼트 내부에서 `L = 1000` 비중첩 → 세그먼트당 3개, 피험자당 **90 윈도우**; `segment_idx`, `window_start_s ∈ {0, 8, 16}` (세그먼트 내 시각), `label_sbp/dbp`. 드롭 0.
- 처리본 `mimicbp_8s` (`np.savez_compressed`): MANIFEST built 2026-08-27T00:30:35, `total_windows 137,160`, `n_subjects 1,524`, sha256 `4f02e933…`.
- 분할 `split_a7_mimicbp_official.json` (sha256 `c52de946…`): 공식 v2.2 리스트를 **정렬만** 하고 재샘플 없음 (`seed: null`), 피험자 disjoint 검증. train 1,100 (99,000) / val 195 (17,550 → stride 5 → 3,510) / test 229 (20,610 → stride 6 → 3,435). PENGUIN 자체 분할(glob 순 `random.sample`, fold 8)은 사용 안 함.

---

## 3. 전처리

### 3.1 공통 윈도우 파이프라인 (`src/ppg2ecg/data/preprocess.py:14-41`, PENGUIN `preprocess.py:14-62`의 라인 단위 재진술, atol=0 패리티 테스트)

| 단계 | 코드 | 정확한 설정 |
|---|---|---|
| 입력 | `x: [n_windows, native_len]` float64 | 피험자별 원시 윈도우 |
| 1. 리샘플 | `scipy.signal.resample(x, 128*8, axis=1)` | **FFT 리샘플, 윈도우 단위**, 1024 샘플로 (DaLiA PPG 512→1024 업, ECG 5,600→1024 다운; WildPPG 1024→1024 no-op; MIMIC 1000→1024) |
| 2. 필터 | `signal.butter(4, …)` + `signal.filtfilt` | Butterworth **4차, 영위상**, 윈도우 단위, `nyq = 64 Hz`. `bandpass=True`일 때 분기 순서 (`preprocess.py:25-34`): `freq_range[0] < 0` → **저역통과** `butter(4, high, 'low')`; `elif freq_range[1] < 0` → **고역통과**; 그 외 → **밴드패스**. `freq_range=(0.5, 4)` → 밴드패스 (정규화 [0.0078125, 0.0625]); `freq_range=(0.5, −1)` → **0.5 Hz 하이패스**. **필터가 없는 경우는 `bandpass=False`뿐이다** — MIMIC ABP가 무필터인 것은 `ABP_KW`의 `bandpass=False` 때문이며, `(−1, −1)`을 `bandpass=True`와 함께 쓰면 저역통과 분기로 들어가 `butter(4, −0.015625, 'low')`가 되어 `ValueError`가 난다. filtfilt `padlen` 기본 (BP 27, HP 15 샘플) |
| 3. z-score | `scipy.stats.zscore(x, axis=1)` | 윈도우 단위, ddof 0 |
| 4. min-max | `(x − mn)/(mx − mn + 1e-8)*2 − 1` | 윈도우 단위 → **[−1, 1]** |
| 출력 | `astype(np.float32)` | `[n, 1024]` |

모든 통계는 윈도우 국소적이다. 학습 집합 또는 피험자 간 통계는 **입력 경로에 절대 들어가지 않는다** (§3.6 검사 3). 윈도우 단위 FFT 리샘플 + 영위상 HP는 윈도우 양끝에 에지 과도 현상을 만든다 (중앙값 기준 ~4×, p90 ~10×; PENGUIN 충실성을 위해 유지). 타깃 min-max 때문에 절대 ECG 진폭(mV)은 복원 불가.

### 3.2 프리셋

| 프리셋 | bandpass | freq_range | zscore | normalize | 용도 |
|---|---|---|---|---|---|
| `PPG_KW` | True | (0.5, 4) Hz | True | True | 모든 데이터셋의 PPG 입력 |
| `ECG_KW` | True | (0.5, −1) = 0.5 Hz HP | True | True | DaLiA·WildPPG(`wildppg_8s`) ECG 타깃 |
| `ECG_KW` + `zscore=False, normalize=False` | True | 0.5 Hz HP | False | False | `wildppg_8s_prenorm` (A9) |
| `ABP_KW` | False | (−1, −1) | False | False | MIMIC-BP ABP (원시 mmHg, 리샘플만) |

### 3.3 전역 affine 타깃 정규화 (A8 · A9, `src/ppg2ecg/data/target_norm.py`)

`y_norm = (y − μ_train)/σ_train`, **train 피험자의 모든 샘플에 대한 스칼라 한 쌍**, 타깃에만 적용 (PPG 불변). 트레이너에서 `y_tr, y_va = tnorm.forward(...)` (`train_a0.py:127-130`, `train_a2.py:114-117`, `train_a5.py:74-77`); `mu/sigma/source`를 `train_meta.json`과 체크포인트에 저장. 계산 스크립트는 `builtins.open`을 패치해 val/test npz 열기를 예외로 만드는 누수 가드를 둠.

| | A8 (`artifacts/a8_abp_scale_control/normalization.json`) | A9 (`artifacts/a9_ecg_representation_control/normalization.json`) |
|---|---|---|
| 소스 | `mimicbp_8s` `y` (mmHg) | `wildppg_8s_prenorm` `y` (HP 후 원시 ADC) |
| μ_train | **77.57176666467686** mmHg | **1.5754169346247968** |
| σ_train | **22.27561138324233** mmHg | **10501.669121754265** |
| n 샘플 | 101,376,000 (1,100 × 90 × 1,024) | 300,309,504 (293,271 × 1,024) |
| 가드 결과 | `n_files_opened 1100`, `val_test_files_opened []` | `n_files_opened 12`, 동일 |
| 예측 역변환 | `predict_a7.py:104-115` `y = σ·y_norm + μ` 후 ABP 지표 | `eval_a9.py` global-z 공간에서 지표; `test_inputs.npz`에 `y_native, target_norm` 저장 |
| 동기 | 원시 mmHg 스케일이 N(0,1) prior의 ≈81.6× | window-norm 타깃 mean/std −0.371/0.363 vs global-z 0.001/1.223; 피험자 native std 2,533–24,720 (9.8×) |

### 3.4 분할 매니페스트 (`src/ppg2ecg/data/splits.py`, `scripts/make_split_manifest.py`)

| 매니페스트 | sha256 앞자리 | 규칙 | train | val | test | 사용 런 |
|---|---|---|---|---|---|---|
| `split_p0_holdout_seed42.json` | `11c154e4` | 정렬 → `random.Random(42).sample`, val 1 / test 1 | S1,S3–S10,S12–S15 (13) | S11 | S2 | A0, A0-b, A2, A5a, A6a, B1-S2 |
| `split_a3_testS1_valS11.json` | `6d2999bd` | 수동 동결 (S1은 이전에 val/test였던 적 없음), `seed: null` | S2–S10,S12–S15 (13) | S11 | S1 | A3 ×2, A5b, A6b, B1-S1 |
| `split_p1_kfold5_seed42.json` | `cf770d10` | 5-fold (10/2/3), 모든 피험자 1회 test | — | — | — | **어떤 런도 참조하지 않음** |
| `split_upstream_probe_thisfs.json` | `005154d6` | upstream glob-order `random.sample`이 이 파일시스템에서 고르는 것 (val S4, test S10) | 13 | S4 | S10 | 기록용 |
| `split_a4_wildppg_seed42.json` | `bc168144` | §2.3 | 12 | an0,k2s | kjd,ssx | A4 ×2, A5c, A6c, A9 ×3, C1 ×3, R1–R3 (train/val만) |
| `split_a7_mimicbp_official.json` | `c52de946` | §2.4 | 1,100 | 195 | 229 | A7 ×3, A8 ×3 |

### 3.5 평가 서브샘플과 SHA256 순위 코호트

**균일 stride 서브샘플** (모든 데이터셋, `eval_a0_nfe_curve.py:103-105`, `eval_a2.py:92-94`, `predict_a7.py:43-44`, `eval_a9.py:43-44`; 검증에는 `train_a0.py:131-133` 등): 분할의 피험자를 **매니페스트 순서**로 연결, `N > 4096`이면 `stride = ceil(N/4096)`, `[::stride]`; 아니면 전부. 모든 arm에 동일 적용. 페어드 노이즈 `torch.randn(n,1,1024, generator=Generator().manual_seed(0))` CPU; PPG-shuffle 교란 시드 `noise_seed+1 = 1`. 동결 입력은 `outputs/<run>/predictions/test_inputs.npz` (`x, y, sid, starts`; A7/A8은 `segment_idx, label_sbp, label_dbp` 추가, A9는 `y_native, target_norm` 추가).

| 데이터/분할 | N | stride | 부분집합 |
|---|---:|---:|---:|
| DaLiA P0 test S2 | 1,025 | 1 | 1,025 |
| DaLiA A3 test S1 | 1,151 | 1 | 1,151 |
| DaLiA val S11 | 1,131 | 1 | 1,131 |
| WildPPG val an0+k2s | 49,200 | 13 | 3,785 |
| WildPPG test kjd+ssx | 46,884 | 12 | 3,907 |
| MIMIC-BP val | 17,550 | 5 | 3,510 |
| MIMIC-BP test | 20,610 | 6 | 3,435 |

**SHA256 순위 코호트** (WildPPG, 메타데이터만, val/train 피험자만):

| 코호트 | salt | 해시 키 | 스트라텀당 | 총계 | 사용 |
|---|---|---|---|---|---|
| X4-0 NFE 프론티어 = **동결 개발 모집단** | `x4-event-nfe-v2` | `SHA256("{salt}|{subject}|{i}")`, **i = npz 배열 행 위치** (site-major; npz `window_index`가 아님), 미리 본 4행 제외 후 오름차순 stable argsort | 1,024 / 피험자 (an0, k2s) | **2,048** (GT 박동 19,834) | X4-0, S1, C0, C1, M1, R2, R3 (R2/R3는 `artifacts/x4_0_event_reliability/nfe_subset.json`과 원소 단위 일치 assert) |
| X4-0 source 진단 | `x4-event-source-v2` | 동일 | 256 | 512 | X4-0 §9–10 |
| X4-0 interval stress | `x4-event-schedule-v2` | 동일 | 512 | 1,024 | X4-0 §11 |
| 미리 본 윈도우 (제외) | — | (an0, 9066), (an0, 18138), (k2s, 5852), (k2s, 16436) | | | 모든 X4-0 부분집합 |
| R1 internal-dev 분할 | `r1-internal-dev-v1` | `SHA256("{salt}|{subject}")` 12 train 중 최소 2 | | internal_dev = **u7y, e61**; probe_train = fex l38 n31 ngh p5d p9p qm9 trh tz8 w4p | R1 |
| R1 코호트 | `r1-global-rhythm-observability-v1` | `SHA256("{salt}|{subject}|{site}|{window_index}")` | train/dev ≤ 2,048 / 부위; val ≤ 1,024 / 부위 | 106,496 행 = probe_train 81,920 + dev 16,384 + val **8,192** (GT 박동 79,111) | R1 학습/평가; R2/R3 부위별 2차 분석 |
| R1 visual | `r1-visual-v1` | 동일 | 8 / 부위 (val) | 64 | R1 atlas |
| V1 중첩 코호트 | `v1-all-subject-stepwise-visualization` | 동일 키, VIZ ⊂ METRICS ⊂ DELAY | 8 / 32 / 128 per (subject, site), 14 피험자 × 4 | 448 / 1,792 / 7,168 | V1 |
| C2 visual atlas | `c2-visual-atlas-v1` | 동일 키, 동결 eval 부분집합 안 | 8 / (subject, site) | 64 | M1 (C2는 미실행) |
| S1 템플릿 박동 | `s1-template-v1` | `SHA256("{salt}|{subject}|{i}")` | 256 / 피험자, train 10명 (fex, p5d 제외) | 2,560 | S1 G1 stamping |
| R2 shuffle 파트너 | `r2-rhythm-shuffle-v1` | 동일 키 → (subject, site) 내 순위 i → (i+1) mod n 교란 (고정점 없음) | train 293,271 (48 strata) + eval 2,048 + viz 64 | | R2/R3 SHUFFLE arm |

### 3.6 누수 검사 (`src/ppg2ecg/data/leakage.py`, `scripts/preflight_a0.py`, `scripts/run_leakage_checks.py`)

| # | 검사 | 구현 / 결과 |
|---|---|---|
| 1 | 피험자 disjoint (train∩val = train∩test = val∩test = ∅, 모든 피험자 정확히 1회) | `check_subject_disjoint`; 매니페스트 생성 시 assert, preflight L90 |
| 2 | 윈도우 disjoint: `sha1(round(float32, 6).tobytes())` 쌍별 교집합 0 | `check_window_disjoint`; P0 upstream 배열 28,055/2,262/2,051 unique, 겹침 0; 매 런 preflight |
| 3 | 윈도우 국소 정규화: 다른 행을 난수로 바꿔도 행 i 불변 (atol 1e-9) | `check_windowwise_normalization`; preflight는 PPG **및** 타깃 검사; `tests/test_splits_leakage.py` |
| 4 | 타깃 없는 추론: `sample(ppg)` 시그니처에 `target/target_signal/y/ecg` 없음 + 고정 시드 불변성 | `check_inference_signature_target_free`, `check_inference_target_invariance` |
| 5 | preflight 게이트 (실패 시 exit 1): 원시 체크섬 기록, 처리본 파일별 sha256 = `MANIFEST.json`, MIMIC mmHg 범위·NaN/Inf, `provenance.json` 기록 | 모든 `outputs/*/provenance.json` |
| 6 | A8/A9 train-only 통계 가드 (`builtins.open` 패치) | `ok true` (§3.3) |
| 7 | 전역 affine이 윈도우 단위가 아님, 지정 피험자만 읽음 | `tests/test_target_norm.py` |
| 8 | WildPPG 테스트 방화벽 `assert_no_test_subjects` | X4-0 이후 모든 분석 스크립트 첫 문장 (§7.9) |
| 9 | A9 패리티: `wildppg_8s` vs `_prenorm`의 PPG/`window_index`/`site` bit-identical, A9 테스트 윈도우 = A4 | 학습 전 게이트 |
| 10 | PENGUIN 패리티: 전처리 (atol 0), Heun 샘플러 (bit-exact), CFM 타깃 | `tests/test_upstream_parity.py:14-58` |
| 11 | 결정적 분할 (정렬 + `random.Random(seed)`) vs upstream glob 순서 | 설계 |
| 12 | 8 s 윈도우는 피험자 경계를 넘지 않음 | 설계 |

---

## 4. 모델 구조

### 4.1 주 백본 — PENGUIN Flow-SSM / S5 (upstream 클래스, 무수정)

빌드: `build_penguin_backbone(**overrides)` = `PENGUIN(**{**PENGUIN_DALIA_CFG, **overrides})`, `PENGUIN_DALIA_CFG = dict(n_step=25, sample_rate=128, h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0)` (`src/ppg2ecg/models/__init__.py:10-17`). iMF arm과 A5/A6 회귀는 `n_step=1`로 덮어씀 (`n_step`은 upstream `sample()`의 Heun 스텝 수일 뿐, `MeanFlowS5`도 회귀 클래스도 `sample()`을 호출하지 않으므로 무의미).

#### 4.1.1 `__init__`이 실제로 소비하는 설정 키

| 키 | 클래스 기본값 | upstream `model.yaml` | 본 프로그램 전 런 | 코드에서의 효과 |
|---|---|---|---|---|
| `n_step` | 25 | 25 | 25 (OT-CFM) / **1** (iMF·A5/A6 — 회귀 클래스가 `n_step=1`을 하드코딩, `regressor.py:30, 75`) | `sample()` Heun 스텝, NFE = 2·n_step |
| `sample_rate` | 128 | (preprocess에서) 128 | 128 | **conv 커널 = `sample_rate//4` = 32** (0.25 s) |
| `h_dim` | 16 | 128 | 128 | 두 스트림 채널 폭, adaLN cond 차원 |
| `ssm_block_num` | 4 | 4 | 4 | `Flow_SSM_Layer` 개수 |
| `ssm_ratio` | 2.0 | 2.0 | 2.0 | S5 상태 크기 `int(h_dim·ratio)` = **256** |
| `mlp_ratio` | 4.0 | **2.0** | 2.0 | MLP 은닉 = **256** |
| T (윈도우 길이) | — | 4 s → 512 | **8 s → 1024** (`train_a0.py:135`에서 assert) | 모델은 길이 불변 |
| smoke 설정 | — | — | `h_dim=16, blocks=2, n_step=2` → 50,355 params | 파이프라인 스모크 전용 |

34개 `train_meta.json` 전부에서 `h_dim 128, blocks 4, ssm_ratio 2.0, mlp_ratio 2.0, sample_rate 128, T 1024` 확인.

#### 4.1.2 순전파 `v_θ(x_t, ppg, t) = forward_step(x_t, ppg, timestep)` (`PENGUIN.py:197-209`)

인터페이스: `x_t [B,1,T]`, `ppg [B,1,T]`, `timestep` (B개 원소, 내부에서 `[B]`로 reshape; 학습·Heun 모두 `[B,1]` 전달) → `[B,1,T]` 속도.

```
ppg_e   = pre_conv_ppg(ppg)          # [B,128,T]
x_e     = pre_conv_target(x_t)       # [B,128,T]  ← 단 한 번 계산
cond    = timestep_embedder(t)       # [B,128]
all_dx  = 0
for blk in flow_ssm_list (4개):
    ppg_e, dx = blk(ppg_e, x_e, cond)   # PPG 스트림만 CHAIN, 타깃 스트림은 매 블록 같은 x_e
    all_dx += dx                        # 블록 출력 SUM
out = final_layer(all_dx, cond)      # [B,1,T]
```

#### 4.1.3 입력 스템

| 스템 | 구성 | 형상 | 비고 |
|---|---|---|---|
| `pre_conv_ppg` | `Conv1d(1→128, k=32, padding='same', bias)` → `SiLU` → `Conv1d(128→128, k=32, 'same', bias)` | `[B,1,T] → [B,128,T]` | stride/pooling/정규화 없음. 짝수 커널 + `'same'` ⇒ 좌 15 / 우 16 비대칭 패딩; 2-conv 수용영역 63 샘플 (`out[t]` ← `in[t−30 … t+32]`, 0.49 s) |
| `pre_conv_target` | 동일 구조, 별도 가중치 | 같음 | 노이즈 상태 `x_t`에 적용. 두 스템은 아무것도 공유하지 않음 |

각 528,640 파라미터 (4,224 + 524,416).

#### 4.1.4 `TimestepEmbedder` (`PENGUIN.py:13-36`)

| 항목 | 값 |
|---|---|
| 정현파 차원 | `frequency_embedding_size = 256`, `half = 128` |
| 주파수 | `exp(−ln(10000)·k/128)`, k=0..127 → 1.0, 0.9306, …, 1.0746e−4 (`max_period = 10000`) |
| 임베딩 | `cat[cos(t·freqs), sin(t·freqs)]` → `[B,256]` (짝수 차원, zero-pad 없음) |
| MLP | `Linear(256→128) → SiLU → Linear(128→128)` → `[B,128]`; **49,408 params** |
| 입력 스케일 | **없음** — `t ∈ [0,1]`을 그대로 투입. 모든 정현파 인자가 ≤ 1 rad이므로 낮은 k 채널만 실질 변화 (정수 diffusion step용 DiT/ADM 설계를 연속 t에 그대로 씀) |
| 용도 | `cond = timestep_emb`가 **유일한** adaLN 조건. **PPG는 cond에 들어가지 않는다** |

#### 4.1.5 `Flow_SSM_Layer` 한 블록 (`PENGUIN.py:39-123`)

모든 LayerNorm = `LayerNorm(128, elementwise_affine=False, eps=1e-6)` — 각 시점의 **채널 축** 정규화, 학습 affine 없음.

| 이름 | 정의 | params |
|---|---|---|
| `adaLN_modulation` | `SiLU → Linear(128 → 12·128 = 1536, bias)` | 198,144 |
| `cross_attn` | `nn.MultiheadAttention(128, heads=1, batch_first=True)` — **호출되지 않음** | 66,048 (dead) |
| `norm1_ppg/norm2_ppg/norm1_target/norm2_target` | non-affine LN | 0 |
| `ssm_ppg`, `ssm_target` | `S5(128, 256, bidir=True)` | 131,712 each |
| `pre_attn_ppg`, `mlp_ppg`, `pre_attn_target`, `post_attn_target`, `mlp_target` | 각 `Linear(128→256) → GELU → Linear(256→128)` | 65,920 each (합 329,600) |
| **블록 합** | | **857,216** |

adaLN-Zero 12청크 순서: `shift_ssm_ppg, scale_ssm_ppg, gate_ssm_ppg, shift_mlp_ppg, scale_mlp_ppg, gate_mlp_ppg, shift_ssm_target, scale_ssm_target, gate_ssm_target, shift_mlp_target, scale_mlp_target, gate_mlp_target`. `modulate(x, shift, scale) = x·(1+scale) + shift`, 게이트는 브랜치 출력에 곱함. 모두 샘플별 벡터를 T에 브로드캐스트.

```
# PPG 스트림 (블록 내부, [B,T,128]로 전치)
res_ppg  = ppg
ppg_cond = g_ssm_p ⊙ S5_ppg( modulate(LN1_p(ppg), …) )          # L105
ppg1     = res_ppg + ppg_cond
ppg_mlp  = g_mlp_p ⊙ MLP_ppg( modulate(LN2_p(ppg1), …) )
ppg_out  = res_ppg + ppg_mlp                                     # 스킵이 블록 '입력'에서 출발

# 타깃 스트림
res_t    = x_t                                                   # = pre_conv_target(x_t), 모든 블록 동일
t_cond   = g_ssm_t ⊙ S5_target( modulate(LN1_t(x_t), …) )
t_cond   = post_attn_target( pre_attn_target(t_cond) + pre_attn_ppg(ppg_cond) )   # ← PPG→타깃 융합점
x1       = res_t + t_cond
t_mlp    = g_mlp_t ⊙ MLP_target( modulate(LN2_t(x1), …) )
dx_t     = res_t + t_mlp                                         # 스킵이 블록 '입력'에서 출발
```

융합 사실: (i) 타깃 스트림에 더해지는 PPG 항은 `ppg_cond` = **게이트가 걸린 잔차 이전 S5 출력**이며 갱신된 `ppg_out`이 아니고, 재정규화되지 않는다. (ii) 융합은 2층 GELU MLP 뒤의 시점별 덧셈이고 그 뒤 또 한 번 MLP — 어텐션도, 연결도, CFG/조건 드롭아웃도 없다. (iii) 논문은 "linear projection"이라 쓰지만 코드는 MLP이며 **코드가 권위**다.

잔차 배선: 두 스트림 모두 `out = x + g·MLP(LN(x + branch))` 형태로, DiT의 `(x+branch) + g·MLP(...)`가 아니다. SSM(또는 융합된 SSM+PPG) 항은 출력 스킵 위에 있지 않고 MLP 입력을 통해서만 블록 출력에 도달한다 (수치 검증: `docs/PENGUIN_AUDIT.md` §24 finding 10).

#### 4.1.6 S5 레이어 (`S5(width=128, state_width=256, bidir=True)`)

| 항목 | 값 |
|---|---|
| 호출 인자 | `(h_dim, int(h_dim·ssm_ratio), bidir=True)` — 나머지는 전부 기본값 |
| 상태 크기 P | 256, `block_count = 1` → HiPPO 블록 1개 |
| Λ, V 초기화 | `make_DPLR_HiPPO(256)`: HiPPO-LegS `A`, `S = A + PPᵀ`; `Λ = mean(diag S) + i·eigvals(−iS)` |
| 초기 상수 (재계산) | **Re(Λ) = −0.5 정확히** (256개 전부); Im(Λ) ∈ [−20,860.2, +20,860.2] 켤레쌍 (|Im Λ| 분위 min/25/50/75/max = 0.2/33/105/291/20,860) |
| Λ 파라미터 | complex64 (256,), **학습 대상, 무제약** (고유값 클리핑은 주석 처리됨) |
| `bcInit` | `None` → `"dense"`: B, C는 LeCun-normal |
| B | float32 (256,128,2) = `Vinv @ B₀`의 [Re, Im], forward에서 `as_complex` |
| C | complex64 (128, **512**) = `cat([C₁V, C₂V], −1)`, 방향별 **독립** LeCun-normal |
| D | float32 (128,) ~ U[0,1) |
| `log_step` | float32 (256,), ~ U[ln 1e−3, ln 0.1] (`dt_min 0.001`, `dt_max 0.1`) |
| 이산화 | **ZOH**: `Δ = exp(log_step)`, `Λ̄ = exp(ΛΔ)`, `B̄ = ((Λ̄−1)/Λ)·B̃`; `step_scale ≡ 1.0` |
| 순환 | 샘플별 vmap: `Bu_k = B̄u_k`; `(Λ̄, Bu)`의 forward associative scan → `x_k = Λ̄x_{k−1} + Bu_k`; **같은** `(Λ̄, Bu)`의 reverse scan; 상태 concat → (T,512); `y_k = Re(C̃x_k) + D⊙u_k` |
| 양방향 | 전·후방이 **Λ와 B를 공유**, C만 2배 |
| 구현 | jax 포팅 `associative_scan` (재귀 pairwise reduce/interleave), `binary_operator`는 `@torch.jit.script` |
| S5당 params | Λ 256 + B 65,536 + C 65,536 + D 128 + log_step 256 = **131,712** |

#### 4.1.7 `FinalLayer`와 초기화

`LN(128, no affine, eps 1e−6)` → `modulate(shift, scale)` (`SiLU → Linear(128→256)`(cond)의 chunk 2) → `Linear(128→1, bias)` → 전치 → `[B,1,T]`. **33,153 params**.

초기화 (`init_adaLN_modulation`): 4개 블록과 `FinalLayer`의 `adaLN_modulation` 마지막 `Linear`(weight·bias)와 `final_layer.linear`(weight·bias)를 **0으로**. 나머지는 PyTorch 기본 (Conv1d/Linear kaiming-uniform, MHA xavier) + 위 S5 초기화. 결과 (스모크 측정): 초기 출력 ≡ 0, 모든 게이트 0 → step 0에서 PPG의 영향력 0; step 1에서 161개 파라미터 텐서 중 2개(`final_layer.linear`)에만 기울기, step 2에서 16개 (adaLN-Zero 캐스케이드); 첫 스텝 손실 = E‖x₁−x₀‖².

#### 4.1.8 파라미터 수 (h_dim 128, 4블록, state 256, MLP 256, k 32)

| 모듈 | numel |
|---|---|
| `revin` | 2 — **미사용** |
| `pre_conv_ppg` / `pre_conv_target` | 528,640 / 528,640 |
| `timestep_embedder.mlp` | 49,408 |
| 블록당: adaLN 198,144 + cross_attn 66,048(dead) + S5 ×2 263,424 + MLP ×5 329,600 | 857,216 |
| 4블록 | 3,428,864 |
| `final_layer` | 33,153 |
| **합 (161 텐서)** | **4,568,707** |
| dead/미사용 | cross_attn 4×66,048 = 264,192 + revin 2 = **264,194** |
| **effective** | **4,304,513** |
| adaLN `Linear.weight` 총합 | 819,200 (4×196,608 + 32,768) |

주의: (a) complex64를 1개로 세므로 실수 스칼라 기준은 4,568,707 + 8×(256+65,536) = **5,095,043**. (b) `cross_attn`은 grep 상 어디서도 호출되지 않고 `.grad`가 `None`으로 남아 AdamW가 건너뛰지만 `state_dict`에는 항상 들어 있다. (c) 제외 규약이 두 가지: 학습 스크립트는 `("cross_attn","revin")` → 264,194 / 4,304,513; 스모크 스크립트는 `cross_attn`만 → 264,192 / 4,304,515. (d) upstream `summarize()`의 thop `Params 3.25 M`, `GFLOPs 60.77`은 이 모델에 대해 **틀렸다** (thop 훅이 원시 S5 텐서와 MHA를 놓침) — 인용 금지.

#### 4.1.9 알려진 특이점 (전부 코드 검증)

1. **타깃 스트림 비연쇄**: `x_e`를 한 번 계산해 4블록에 동일하게 투입, 출력 합산 → 타깃 경로는 "깊이 1 × 병렬 4헤드" (헤드 k는 깊이 k의 PPG 특징을 봄). 논문/그림은 연쇄 스택을 암시하지만 코드가 권위이며 모든 체크포인트가 이 구조다. `dx = x_e + …`이므로 `all_dx = 4·x_e + Σ branch_k`이고 ×4 스케일은 non-affine LN이 제거한다.
2. **죽은 `cross_attn`** (264,192), **미사용 `revin`** (2), 미사용 `self.mean/std`. `pre_attn_*`/`post_attn_*`은 역사적 이름일 뿐 어텐션이 아니다.
3. **단일 스킵 잔차** (§4.1.5).
4. **adaLN-Zero + 0 초기화 final Linear** → 출력 ≡ 0에서 시작.
5. **t가 스케일 없이** 256차원 정현파(`max_period 10000`) 임베더로 들어감.
6. **PPG는 adaLN 조건이 아니다**; 블록마다 `pre_attn_ppg(ppg_cond)` 덧셈으로만 들어가며, 그 값은 게이트가 걸린 잔차 이전 S5 출력이고 재정규화되지 않는다.
7. **양방향 S5가 Λ·B를 공유**, C만 2배. 고유값 클리핑 없음 → 학습 중 Re(Λ)가 0을 넘는지는 저장소 어디서도 확인하지 않았다.
8. `mlp_ratio` 2.0이 클래스 기본 4.0을, `h_dim` 128이 16을 덮어씀 — 체크포인트는 이 값을 요구한다.
9. 짝수 커널(32) + `'same'` ⇒ 1샘플 비대칭 패딩; 스템 수용영역 63 샘플.
10. **채널 방향 non-affine LayerNorm**은 타깃 임베딩에 대한 채널 균일 덧셈을 정확히 소거한다 (측정 max|Δu| = 7.2e−7). 반면 원시 잔차 경로("직결 경로")는 1차 출력 응답의 0.21–0.70을 나른다 (`docs/R3_TARGET_STREAM_HOOK_AUDIT.md` §2.1).
11. `optimize()`는 인자 `pred_target`/`target_signal`을 무시하고 `train_flow`가 저장한 텐서를 쓴다.
12. upstream 샘플링은 x₀를 **CPU RNG**로 뽑고 `no_grad` 없이 돈다 (`heun_step` 내부는 `.detach()`).

### 4.2 iMF 래퍼 `MeanFlowS5` (`src/ppg2ecg/flow/imeanflow.py:26-63`)

파라미터를 **0개 추가**하지 않는다 (`cond_mode`, `h_scale`은 일반 속성). `u(z, ppg, t, h)`는 `forward_step`을 라인 단위로 재진술하되 `cond`만 바꾼다:

```
ppg_e = backbone.pre_conv_ppg(ppg)                                # [B,128,T]
z_e   = backbone.pre_conv_target(z)                               # [B,128,T]
cond  = backbone.timestep_embedder(h.reshape(-1) * h_scale)       # [B,128]   ← h_only 모드
for blk in backbone.flow_ssm_list: ppg_e, dx = blk(ppg_e, z_e, cond); all_dx += dx
u = backbone.final_layer(all_dx, cond)                            # [B,1,T]
```

| `cond_mode` | `cond` | 상태 |
|---|---|---|
| `h_only` | `E(h_scale·h)` | **A2, A3, A4, A7, A8, A9, B1, C1, R2, R3 전부** (`h_scale = 1.0`). 공식 iMF `imfDiT.py` L342-344 설계. `t`는 네트워크에 들어가지 않음 (동일 h·다른 t에서 출력이 bit-identical함을 테스트로 assert) |
| `t_plus_h` | `E(t) + E(h_scale·h)` | 결과 전 단계에서 폐기: `h_scale=1` 1 epoch 중단, `h_scale=1000` 발산 (train MSE 10.6 → 395) |
| `t_only` | `E(t)` | 패리티 테스트 전용 — upstream `forward_step`과 bit-exact |

보조 v-head 없음 (공식 코드의 8블록 aux head 미사용). `n_step`은 `sample()`에서만 쓰이므로 무의미.

### 4.3 보조 모델

#### 4.3.1 A5 `S5ConditionalMeanRegressor` (`src/ppg2ecg/models/regressor.py:27-57`)

목적: 같은 백본 위의 결정론적 MSE 회귀 대조군 ("MSE 조건부 평균 프록시"). 레지스트리 키 `"state_token"`.

수정 내용: (1) 백본을 `n_step=1`로 생성, (2) `pre_conv_target`·`timestep_embedder`를 `delattr` (−578,048), (3) 학습되는 상수 상태 토큰 `state_token = nn.Parameter(randn(1,128,1)·0.02)` (+128, Amendment 1), (4) 순전파: `ppg_e = pre_conv_ppg(ppg)`, `z_e = state_token.expand(B,128,1024)`, `cond = zeros(B,128)`, 4블록 합산 → `final_layer`. `cond = 0`이면 `SiLU(0)=0`이므로 모든 adaLN 변조가 bias만 남는다.

| 모듈 | params |
|---|---|
| `state_token` | 128 |
| `revin` (dead) | 2 |
| `pre_conv_ppg` | 528,640 |
| 4블록 | 3,428,864 (cross_attn 264,192 dead; adaLN weight 786,432은 cond=0으로 기울기 0) |
| `final_layer` | 33,153 (adaLN weight 32,768 비활성) |
| **합** | **3,990,787** = 4,568,707 − 578,048 + 128 |
| dead / 비활성 adaLN / effective | 264,194 / 819,200 / **2,907,393** |

산문 대비: prereg §3은 3,990,659 (타깃 스트림에 0 투입)이라 적었으나 그 모델은 영구 dead start (`final_layer.linear.bias`만 기울기)로 `outputs/aborted/*_zero_state_deadstart/`에 보관, **Amendment 1 코드가 권위**.

#### 4.3.2 A6/A7/A8/A9 `S5FullBackboneRegressor` (`regressor.py:60-112`)

목적: 용량 정합 결정론적 대조군 — **무수정** 백본을 상수·샘플 독립·비학습 상태와 시간 입력으로 구동. 레지스트리 키 `"full_backbone"`.

클래스 기본값은 `X_CONST 1.0, T_CONST 0.5, cond_scale 1.0`이지만 **모든 학습 런은 `x_const 0.1, t_const 0.5, cond_scale 0.05`** (`train_meta.json.model_cfg`). 순전파 (cond_scale ≠ 1이므로 명시 경로): `state_input = full_like(ppg, 0.1)`, `ppg_e = pre_conv_ppg(ppg)`, `x_e = pre_conv_target(state_input)`, `cond = timestep_embedder(0.5)·0.05`, 4블록 합산 → `final_layer`. 노이즈·`r`·타깃이 시그니처에 없음(테스트로 assert). 파라미터는 백본과 동일 (4,568,707 / dead 264,194 / effective 4,304,513, 비활성 adaLN 없음).

상수 선택 스크리닝 (A6 학습 전, `outputs/gradcheck_a6_*`, 각 12 epoch, 채택 규칙 = 어느 epoch에서든 beats/ref ≥ 0.3 또는 진폭비 ≥ 0.02):

| 런 | x_const | cond_scale | train MSE ep12 | max amp | max beats | 채택 |
|---|---|---|---|---|---|---|
| x1.0 | 1.0 | 1.0 | 0.1200 | 0.001 | 0.00 | 아니오 |
| x0.1 | 0.1 | 1.0 | 0.1200 | 0.001 | 0.11 | 아니오 |
| x0.1_cs0.0 | 0.1 | 0.0 | 0.1125 | 0.106 | 0.21 | 예 (단 adaLN 819,200 비활성) |
| **x0.1_cs0.05** | **0.1** | **0.05** | **0.1116** | **0.108** | **0.24** | **예 → 동결** |
| x1.0_cs0.05 | 1.0 | 0.05 | 0.1200 | 0.001 | 0.00 | 아니오 |

#### 4.3.3 R1 `RhythmTCN` — Global-TCN / Local-TCN (`src/ppg2ecg/probes/rhythm_tcn.py`)

목적: PPG만으로 조밀한 시점별 R-이벤트 확률장을 예측하는 관측가능성 프로브. Global-TCN은 이후 **동결되어 R2/R3의 리듬 스캐폴드 추출기**로 재사용. ECG 값은 순전파에 들어가지 않음(테스트 assert). I/O: `ppg [B,1,1024]` → logits `[B,1,1024]`.

| 레이어 | 정의 | params | 이론 수용영역 |
|---|---|---|---|
| `stem` | `Conv1d(1→64, k=1)` | 128 | 1 |
| `blocks[0..7]` | `h = GELU(c1(x)); h = c2(h); out = GELU(x+h)`, `c1,c2 = Conv1d(64→64, k=5, same, dilation d_i)` | 41,088 × 8 = 328,704 | Global d=(1,2,4,…,128) → 9,25,57,121,249,505,1017,**2041** 샘플 (15,945 ms); Local d=(1,…,1) → **65** 샘플 (507.8 ms) |
| `head` | `Conv1d(64→1, k=1)` | 65 | — |
| **합** | 두 변형 동일 | **328,897** | RF = `1 + 2(k−1)Σd` |

- 초기화: PyTorch 기본 + `seed_everything(42)` 후 생성 직전 `torch.manual_seed(42)` → Global/Local 초기 가중치 bit-identical.
- 소프트 타깃 (`soft_event_field`): `y(t) = max_j exp(−(t−r_j)²/(2σ²))`, **σ = 100 ms = 12.8 샘플** (max 결합, 합산 아님; 박동 없으면 0). `r_j`는 동결 검출기 `detect_rpeaks(ecg_float64, 128)`; 라벨링 후 ECG 폐기 (`_cache_*.npz`).
- 손실: `BCEWithLogitsLoss` (전 샘플 평균).
- 이벤트 추출 (`extract_events`): 확률장 국소 최대 → `p ≥ threshold` → 확률 내림차순 greedy NMS, refractory `|Δ| > 32` 샘플 (250 ms). **임계값은 internal-dev에서만** 그리드 0.05…0.95를 F1@150 ms로 최대화 → Global 0.35 (dev F1@150 0.8411), Local 0.35 (0.7559).
- 미학습 변형: `global_site` (부위 임베딩 FiLM, 0 초기화, +512 → 329,409) — 코드에만 존재, 실행 안 됨.

#### 4.3.4 R2 `RhythmAdapter` + `RhythmMeanFlowS5` (`src/ppg2ecg/flow/rhythm_transfer.py`)

목적: 최소 전이 프로브 — 동결 Global-TCN 스캐폴드를 동결 iMeanFlow 생성기의 **PPG 스템 출력**에 0 초기화 1×1 덧셈 어댑터로 주입. arm B / TRUE / SHUFFLE / ORACLE.

- 동결 부품: 생성기 `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` (라운드 46 = C1 arm B, A4와 가중치 동일; 파일 sha256 `557c7054…`, state sha256 `47d7ccb9…`, 4,568,707), Global-TCN (`0986a7af…`, 328,897). 둘 다 `requires_grad_(False)` + step 1 이후 기울기 없음 assert.
- 어댑터: `proj = Conv1d(1→128, k=1, bias=False)`, `zeros_` 초기화 → **학습 파라미터 128개**. 활성화·정규화·시간 혼합 없음. 전체 모델 4,568,835.
- 주입: 캐리어 `ppg2 = cat([ppg, s], dim=1) [B,2,1024]`를 `u` 안에서 분리해 동결 `imeanflow_loss`/샘플러를 그대로 호출; 그 뒤 **`ppg_e = pre_conv_ppg(ppg) + rhythm_adapter(s)`**. 어댑터가 0이면 baseline과 bit-exact (`torch.equal` 패리티).
- 스캐폴드: `s = sigmoid(GlobalTCN(ppg)).detach()` — 조밀한 NMS 이전 장, 임계·shift 없음 (R1의 0.35는 기록만).
- ORACLE 장: 해당 윈도우 자신의 GT ECG로 만든 R1 라벨 (`soft_event_field(detect_rpeaks(y), 1024)`) — **설계상 GT-R 누수, 진단 전용**. 학습 캐시 293,271×1,024 float32 (sha256 `2e6c548c…`, 장 평균 0.3014, 박동 미검출 24행).
- SHUFFLE 파트너: (subject, site) 층 안에서 `sha256("r2-rhythm-shuffle-v1|{subject}|{site}|{window_index}")` 순위 i → (i+1) mod n (전단사, 고정점 없음). eval 파트너는 자기 자신 대비 |Δ 박동| 중앙값 1.0, |Δ 평균 RR| 82 ms.

#### 4.3.5 R3 `RhythmCrossFusionAdapter` + 적응 리듬 게이트 (`src/ppg2ecg/flow/rhythm_fusion.py`)

목적: 스캐폴드를 **타깃 측** 잠재 상태에 교차 어텐션으로 융합 (ungated TF / gated GTF). arm: B, ADD(동결 R2 TRUE 어댑터), TF-TRUE, TF-SHUFFLE, GTF-TRUE, GTF-SHUFFLE, GTF-CONST, GTF-ORACLE, ADD-ORACLE.

삽입 텐서 (후크 감사): `z_e = backbone.pre_conv_target(z)`, `[B,128,1024]` — 파형과 샘플 단위 정렬 (같은 패딩 k=32 conv 2개, `z_e[t] ← z[t−30…t+32]`), 한 번의 `u` 평가 안에서 PPG-free, 타깃 스트림 비연쇄이므로 4블록에 동일하게 도달. 주의: 원시 블록 잔차(직결 경로)가 1차 출력 응답의 0.21–0.70을 나르고, 채널 균일 덧셈은 채널 LN이 소거한다 (max|Δu| 7.2e−7).

| 블록 | 정의 | 출력 | params |
|---|---|---|---|
| 토큰 `tok` | `Conv1d(1→32, k=7, stride=4, padding=3, bias=False)` on `s` | `[B,32,256]` → `[B,256,32]`; 토큰 j는 샘플 4j 중심 | 224 |
| 쿼리 `q` | `Conv1d(128→32, k=1, bias=False)` on `z_e` | `[B,1024,32]` | 4,096 |
| 위치 인코딩 | 고정 정현파 d=32, base 10,000; 쿼리 t=0..1023, 키 4j; Q와 토큰(K=V)에 더함. **non-persistent 버퍼** (체크포인트·해시 제외) | — | 0 |
| 교차 어텐션 `attn` | 명시적 `q,k,v,o = Linear(32,32)` + bias, **4헤드 × dim 8**, `softmax(QKᵀ/√8)V`, 마스크 없음, 어텐션 맵 `[B,4,1024,256]`. 명시 구현 이유: `torch.func.jvp`가 fused SDPA를 지원하지 않음 (`nn.MultiheadAttention` 사용 금지 테스트) | `[B,1024,32]` | 4,224 |
| 출력 투영 `out` | `Conv1d(32→128, k=1, bias=True)`, **weight·bias 0 초기화** | `[B,128,1024]` | 4,224 |
| **TF 합** | | | **12,768** (백본의 0.279 % / effective의 0.297 %; 예산 ≤ 50,000 & < 1.1 %) |
| 게이트 특징 | `s`만으로: `f1 = s`; `f2 = avg_pool1d(s, 33, stride 1, pad 16, count_include_pad=False)`; `f3 = |s[t] − s[t−1]|` (`f3[0]=0`) | `[B,3,1024]` | 0 |
| 게이트 MLP | `Conv1d(3→16,k=1) → SiLU → Conv1d(16→1,k=1) → sigmoid`; `gate.2.weight = 0`, `bias = logit(0.90) = ln 9 = 2.1972` → 초기 `g(t) = 0.90` | `[B,1,1024]` | **81** (예산 < 128) |
| **GTF 합** | fusion + gate | | **12,849** (전체 모델 4,581,556; TF 4,581,475) |
| CONST 변형 | 구조·초기화·파라미터 동일, 모든 게이트 특징을 **윈도우 시간 평균**으로 치환해 1,024에 브로드캐스트 → 윈도우 수준 강도만 | | 12,849 |

순전파: `ppg_e`는 손대지 않고, `delta = out(attn(q(z_e)+pe_q, tok(s)+pe_k))` (GTF는 `sigmoid(gate(features(s))) ⊙ delta`), **`z_e ← z_e + delta`**. 평가 전용 `cancel_direct_route`: 디코딩 직전 `all_dx −= n_blocks·delta` (= 4·delta)로 1차 원시 잔차 경로 제거.

초기화·난수: 생성 직전 `torch.manual_seed(42)`, 고정 순서(tok → q → attn q/k/v/o → out → gate). 초기화 해시: fusion 부분집합 `15bd2abb…` (6 arm 전부 동일), GTF 전체 `1dc18b2d…`, gate 부분집합 `6f5118b3…`. step-0 패리티: `out = 0`이므로 2,048 윈도우 NFE-4 전체 텐서에서 baseline과 `torch.equal` (11개 검사 전부 통과).

#### 4.3.6 모델 교차 요약

| 모델 | 단계 | 입출력 | 총 params | 학습 | 동결 | 새 부분 초기화 |
|---|---|---|---|---|---|---|
| PENGUIN 백본 | A0/A0-b/A3/A4/A7/A8/A9 OT-CFM | `(x_t, ppg, t) → v` | 4,568,707 (eff. 4,304,513) | 전부 | 없음 | adaLN-Zero |
| `MeanFlowS5` | A2/A3/A4/A7/A8/A9 iMF, B1, C1 | `(z, ppg, t, h) → u` | 4,568,707 | 전부 | 없음 | (백본과 동일, 추가 0) |
| `S5ConditionalMeanRegressor` | A5a/b/c | `ppg → ECG` | 3,990,787 (eff. 2,907,393) | 전부 | 없음 | `state_token ~ N(0,0.02²)` |
| `S5FullBackboneRegressor` | A6a/b/c, A7, A8, A9 | `ppg → ECG/ABP` | 4,568,707 (eff. 4,304,513) | 전부 | 없음 | 없음 (upstream) |
| `RhythmTCN` Global/Local | R1 (Global은 R2/R3에서 동결 재사용) | `ppg → 이벤트 로짓` | 328,897 | 전부 | 없음 | seed 42 공유 |
| `RhythmAdapter` in `RhythmMeanFlowS5` | R2 | 캐리어 `[B,2,1024]` → `u` | 4,568,835 | **128** | 생성기 4,568,707 + TCN 328,897 | zeros |
| `R3Fusion` (TF/GTF) in `FusionMeanFlowS5` | R3 | 동일 캐리어 → `u` | 4,581,475 / 4,581,556 | **12,768 / 12,849** | 생성기 + TCN (+ADD arm은 R2 어댑터) | `out` 0, 게이트 bias ln 9 |

---

## 5. 목적함수와 샘플러

### 5.1 OT-CFM (베이스라인 목적) — upstream `train_flow`/`optimize`, 프로그램 재진술 `src/ppg2ecg/flow/cfm.py:18-36` (bit-exact 패리티 테스트)

| 항목 | 구현 |
|---|---|
| 시간 샘플링 | `t ~ U(0,1)`, `torch.rand(B,1)` — 균등, 워핑·가중 없음 |
| 소스 | `x₀ ~ N(0,I)`, `randn_like(x₁)`, `[B,1,T]` |
| 경로 | `x_t = (1−t)x₀ + t·x₁`; **t=0 노이즈, t=1 ECG**; σ_min = 0 (선형 보간 = rectified flow) |
| 커플링 | 독립 `(x₀, x₁)` — **미니배치 OT (Tong et al.) 없음**. 즉 "OT-CFM"은 Lipman의 conditional-OT 경로를 뜻함 |
| 타깃 | `v* = x₁ − x₀` |
| 손실 | `F.mse_loss(v_θ(x_t, ppg, t), v*)` = B·1·T 평균, 보조 항 없음 |
| 스텝 | `zero_grad → backward → step`, grad clipping·EMA 없음 |
| 학습 모니터 | `pred_x₁ = x_t + (1−t)·pred_dx_t` (t=1로 오일러 1스텝) — 학습 MAE 출력용일 뿐 샘플링 MAE가 아님 |
| 헬퍼 | `cfm_targets(x1, t, x0) → (x_t, x1−x0, t, x0)`; `cfm_loss = F.mse_loss`; `euler_x1_estimate(x_t, v, t) = x_t + (1−t)v` |

### 5.2 Improved MeanFlow (iMF) 목적 — `src/ppg2ecg/flow/imeanflow.py`

시간 규약: **t=1 노이즈, t=0 데이터** (OT-CFM과 반대).

```
z_t = (1 − t)·x + t·e            # x = ECG 타깃, e ~ N(0,I), t는 [B,1,1] 로 reshape
v   = e − x                       # 순간 속도 타깃 (상수, 기울기 없음)
u(z_t, r, t) = 1/(t−r) ∫_r^t v ,  r ≤ t ,  경계 u(z_t,t,t) = v(z_t,t)
h   = t − r ∈ [0,1]
```

**경계 v_θ**: `v_θ(z_t,t) = u_θ(z_t, ppg, t, h=0)`을 `torch.no_grad()`로 평가 — 같은 네트워크의 h=0 (보조 v-head 없음).

**JVP (`compound_V`)**:
```
u_fn(z, t_, r_) := net.u(z, ppg, t_, t_ − r_)
tangents        = (v_θ, 1, 0)   on (z_t, t, r)
u, dudt = torch.func.jvp(u_fn, (z_t, t, r), tangents)   # 전방 모드, jvp_mode="forward"
dudt    = dudt.detach()                                  # JVP '결과'에만 stop-gradient
V       = u + (t − r)·dudt
```
`d/dt u = v·∂_z u + ∂_t u` (r 고정). 네트워크가 `h = t−r`을 보므로 `∂_t`는 `dh/dt = 1`을 통해 작용한다. stop-gradient 위치는 공식 `imf.py` L347-393의 JAX 포팅과 float64 1e-9 기울기 패리티로 고정. 대안 `jvp_mode="double_vjp"`(`torch.autograd.functional.jvp`)는 어떤 기록된 런에서도 쓰이지 않음.

**적응 가중 손실 (`imeanflow_loss`)**:
```
Δ²_b = Σ_{c,τ} (V − (e − x))²          # 샘플별, C와 T(=1024)에 대한 합
w_b  = 1 / (sg(Δ²_b) + c)^p            # p = norm_p = 1.0, c = norm_eps = 0.01
loss = mean_b( Δ²_b · w_b )
```
p=1이면 가중 손실이 `Δ²/(Δ²+0.01) ≈ 1`이라 `train_loss_weighted`는 1 근처에서 포화한다 (A2 epoch 1: 0.99996). 실질 모니터는 **비가중** `mse` (원소별 평균). A8 이후 `w_mean … w_p99, w_saturation_frac, w_near_lower_frac` 진단 기록.

**`(t, r)` 샘플링 (`sample_tr`)**:

| 단계 | 코드 |
|---|---|
| 추출 | `n1, n2 ~ N(0,1)` `[B,1]` (전용 CPU generator); `t = sigmoid(n1·p_std + p_mean)`, `r = sigmoid(n2·p_std + p_mean)` — i.i.d. **logit-normal(μ = −0.4, σ = 1.0)** |
| 정렬 | `t = max(t,r)`, `r = min(t,r)` |
| `r = t` 행 | `fm_mask = arange(B) < int(B·data_proportion)`, `data_proportion = 0.5` — **행 위치로 선택 (베르누이 추출이 아님)** |
| 마이크로배치 실현 | 학습은 `Bc = 32`마다 `sample_tr` 호출 → 행 0–15는 `r=t` (`h=0`, 순수 flow matching, `(t−r)=0`이라 `V=u`), 행 16–31은 `h>0` |

**노출 통계** (2,000,000 추출, seed 20260901): `P(h=0) = 0.5`, 양의 h 중앙값 0.2011, `P(h≥0.125) = 0.3370`, `P(h≥0.25) = 0.2010`, `P(h≥0.5) = 0.0422`, `P(h≥0.7) = 0.0042`, max h = 0.9276. 비경계 행만 (별도 MC — B1 λ 보정, seed 12345, 10⁶ 비경계 샘플): `E[h] = 0.233504`, `E[1−h] = 0.766496` (seed 20260901의 2×10⁶ 추출로는 E[h] = 0.233126). **정확히 h=1 (1-NFE 질의)의 학습 확률은 0**이다 — 이 프로그램의 핵심 구조적 사실 중 하나.

**RNG 스트림과 기울기 누적** (`train_a2.py`):

| 스트림 | 시드 |
|---|---|
| 전역 `seed_everything(42, deterministic=True)` | 42 |
| DataLoader shuffle generator | 42 |
| `(t,r)` CPU generator `tr_gen` | **seed + 1 = 43** |
| 노이즈 `e = randn(Bc,1,T, device=cuda)` | CUDA 전역 스트림 |
| 검증 뱅크 | 1000 + b |

옵티마이저 스텝마다: 배치 `B=64`를 마이크로배치 32 두 개로 나눠 각각 자체 `(t,r)`·`e`를 뽑고 `imeanflow_loss` 계산 후 `(loss·Bc/B).backward()`, 64 윈도우당 `AdamW` 1스텝. 손실·mse·`|du/dt|`가 non-finite면 학습 중단.

**검증 지표 `imeanflow_mse`**: 학습과 동일한 `z_t`, `v_θ`, `compound_V`를 `@torch.no_grad()`로, **비가중** 원소별 `(V − (e−x))²` 평균.

### 5.3 샘플러

| 샘플러 | 시간 격자 | 업데이트 | NFE |
|---|---|---|---|
| upstream `sample`/`heun_step`, 프로그램 `heun_sample` | `linspace(0, 1, n+1)`, Δt = 1/n | `k₁ = v(x,t_i)`; `x̃ = x + Δt·k₁`; `x ← x + (Δt/2)(k₁ + v(x̃, t_{i+1}))`; 각 v 출력 `.detach()` | **2n** (마지막 스텝도 오일러로 줄이지 않음) → n=25면 **50** |
| `euler_sample` | 동일 격자 | `x ← x + Δt·v(x,t_i)` | n |
| `sample_meanflow(net, ppg, e, n)` | `linspace(1, 0, n+1)` | `z_r = z_t − (t−r)·u(z_t, ppg, t, t−r)` | n; **1-NFE = `x̂ = e − u(e, ppg, t=1, h=1)`** (테스트 assert) |
| `sample_meanflow_schedule(net, ppg, e, h_list)` | `t₀=1`, `t_k = 1 − cumsum(h)`, `Σh = 1 ± 1e-9` | 동일 | `len(h_list)`; `UNIFORM[n] = [1/n]×n`은 **n ∈ 1,2,4,8,16에서만** 균등 샘플러와 bit-exact (테스트 `tests/test_x4_0_event_reliability.py:38-45`는 이 n들에 대해 `allclose(atol=1e-6)`만 검사). n = 25·50은 `sample_meanflow`의 `torch.linspace` float32 격자와 `1 − cumsum(h)` float64 격자가 float32 1 ulp(5.96e−08) 어긋나 출력이 10⁻⁷ 수준으로만 일치 |

- 패리티: `heun_sample`은 upstream `.sample()`과 CPU·CUDA 모두에서 **bit-exact** (`torch.equal`; tiny 6 NFE, full 50 NFE `max_abs_diff 0.0`).
- upstream 샘플링은 x₀를 CPU RNG로 뽑고 `no_grad` 없이 돈다 (수치 영향 없음; 프로그램 검증은 감쌈).
- 비균등 스트레스 스케줄 (X4-0D 전용, 시간 순 = 노이즈→데이터): `U4 [0.25]×4` · `LN4 [0.70,0.10,0.10,0.10]` (노이즈 끝 큰 구간) · `LD4 [0.10,0.10,0.10,0.70]` (데이터 끝) · `U8 [0.125]×8` · `LN8 [0.50]+[0.5/7]×7` · `LD8 [0.5/7]×7+[0.50]`. 결과(pooled): morph corr U4 0.7661 / LN4 0.7469 / LD4 0.7040, U8 0.7696 / LN8 0.7609 / LD8 0.7377; event F1 U4 0.4327 / LN4 0.4132 / LD4 0.4138, U8 0.4292 / LN8 0.4223 / LD8 0.4252.

### 5.4 고정 검증 뱅크와 체크포인트 선택

**iMF 뱅크 (`make_imf_banks` / `fixed_imf_mse`)**

| 항목 | 값 |
|---|---|
| 뱅크 수 / 시드 | 4개, 뱅크 b는 `torch.Generator().manual_seed(1000 + b)` |
| 뱅크 b 생성 순서 | (i) `(t_b, r_b) = sample_tr(n_val, g, **학습과 동일 tr_kw)`; (ii) `perm = randperm(n_val, g)`를 t·r에 적용 → `r=t` 절반이 시간순 앞쪽 절반이 아니라 무작위 부분집합이 됨; (iii) `e_b = randn(n_val,1,T,g)` |
| 고정 대상 | 윈도우별 `t_b, r_b, e_b` + 검증 윈도우 자체; 전체 float32 바이트의 SHA-256을 `bank_hash`로 기록 |
| 지표 | 뱅크별 `imeanflow_mse`(비가중)의 윈도우 가중 평균 → 4뱅크 평균 |
| 선택 규칙 | `is_best = val_fixed < best − 1e-4`, patience 20 라운드, 최대 300 라운드 |

**OT-CFM 뱅크 (`val_cfm_fixed`)**: 4뱅크, 시드 1000+b, `t_b = rand(n_val,1,g)` (t=0 노이즈 규약), `z_b = randn(n_val,1,T,g)`; `x_t = (1−t)z + t·x1`, `v* = x1 − z`, `v = forward_step(x_t, ppg, t)`의 윈도우 평균 MSE → 뱅크 평균.

**뱅크 해시** (`config.yaml: selection.bank_hash`): DaLiA OT-CFM `6b5c0139…` · DaLiA iMF `0f15f0d2…` · WildPPG OT-CFM `ed7cd22b…` · WildPPG iMF `2cd14468…` · MIMIC-BP OT-CFM `91206823…` · MIMIC-BP iMF `44c54e58…`.

| 런 | n_val | 뱅크 해시 | best 값 | best 라운드 (0/1-based) | 총 라운드 |
|---|---:|---|---:|---|---:|
| A2 (DaLiA val S11) | 1,131 | `0f15f0d2…` | 0.17381609575435927 | 60 / 61 | 81 (조기 종료) |
| C1-B (WildPPG) | 3,785 | `2cd14468…` | 0.11945885431656277 | 45 / 46 | 66 |
| C1-H25 | 3,785 | 동일 | 0.1278084015796057 | 47 / 48 | 68 |
| C1-H50 | 3,785 | 동일 | 0.11824402330613042 | 80 / 81 | 101 |
| A0-b (OT-CFM) | 1,131 | `6b5c0139…` | 0.16445091611826768 | 64 / 65 | 85 |
| B1-v2 vanilla (진단용) | 1,131 | `0f15f0d2…` | 0.1672501974268944 | 134 / 135 | 300 (고정 예산) |
| B1-v2 curriculum (진단용) | 1,131 | 동일 | 0.1726651715640074 | 24 / 25 | 300 |

C1은 세 arm의 뱅크 해시가 동일함을 assert (`identical_banks_hash = true`).

### 5.5 iMF 변형 두 가지

#### 5.5.1 C1 — 목표 구간 노출 `sample_tr_c1` (`src/ppg2ecg/flow/interval_exposure.py`)

| 항목 | 값 |
|---|---|
| arm | `("B", "H25", "H50")`, `FORCED_H = {B: None, H25: 0.25, H50: 0.50}` |
| arm B | `sample_tr`를 그대로 반환 — 바이트 동일, `tr_gen` 스트림 소비도 동일 |
| 강제 규칙 (H25/H50) | `t, r, fm = sample_tr(Bc, tr_gen, …)`; `idx` = 양의 h 행 위치(`~fm`, 위치상 접미부); `forced = idx[len(idx)//2:]` = **양의 h 행의 위치상 후반부**; 강제 행은 같은 `tr_gen`에서 `u ~ U[0,1)`을 뽑아 `t = h + (1−h)u` (즉 `t ~ U[h,1]`), `r = t − h` |
| 마이크로배치 실현 (Bc=32) | 행 0–15 `h=0` (50 %), 16–23 원래 양의 h (25 %), 24–31 정확히 h = 0.25 또는 0.50 (25 %) |
| 불변 | `tr_kw`(→검증 뱅크·선택 기준), `norm_p`, `norm_eps`, 옵티마이저, 구조 — `--c1-arm`만 다름 |
| 노출 (2×10⁶) | B: `P(h=0)=0.5`, `P(h≥0.5)=0.0422`, 양의 h 중앙값 0.2011 / H25: `P(h=0.25)=0.25`, `P(h≥0.5)=0.0210` / H50: `P(h=0.50)=0.25`, `P(h≥0.5)=0.2710`; `P(h≥0.25)`는 두 H arm 모두 0.3503; max h 0.9276 |
| RNG 통제 (`rng_control.json`, pass) | init/order/noise/banks 해시가 arm 간 동일, `tr_hash`만 다름; 평균 h B 0.1167 / H25 0.1192 / H50 0.1817 |
| 결과 | C1-B는 동결 A4 체크포인트를 **비트 단위로 재현** |

#### 5.5.2 B1-v2 — 점진적 시간 간격 커리큘럼 (`src/ppg2ecg/flow/imeanflow_curriculum.py`, 드라이버 `train_b1_fixed_compute.py`)

| 항목 | 값 |
|---|---|
| 커리큘럼 인자 | `β(h,s) = 1 − s + λ·s·(1 − h)`, `h = (t−r).detach()`; 경계 행(`r=t`)은 항상 `β = 1` |
| 스케줄 | `s = max(0, 1 − step/T_schedule)`, step = 옵티마이저 갱신 수, 선형 |
| λ | `1.304639 = 1/E[1−h]` (MC seed 12345, 10⁶ 비경계 샘플: `E[h] = 0.233504`) → s=1에서 평균 β = 1.0 |
| 손실 | `V`, `Δ²`, `w`는 동결 `imeanflow_loss`와 동일; `loss = mean_b(β_b · w_b · Δ²_b)` — **β는 적응 가중 뒤에만 적용되고 w에 들어가지 않음**; `β=None`이면 동결 손실을 그대로 재현 (패리티 테스트) |
| 미구현 | 원 논문의 α(t) 경계 가중 / 가속 v-학습 (고립된 개입) |
| 예산 `T_schedule = T_train` | DaLiA-S2 220×300 = **66,000**; DaLiA-S1 218×300 = **65,400**; WildPPG **65,482** |
| 드라이버 차이 (A2 대비) | 조기 종료 없음(`stop=False`, 역사적 트리거만 로깅); `checkpoint_final.pt`를 정확히 `T_train`에서; 0/10/25/50/75/100 % 분할 체크포인트; 첫 64 마이크로배치의 페어드 난수 프로브(두 arm 동일 `5c44decf…`); 라운드별 h-빈 진단(`[0,0.1),[0.1,0.3),[0.3,0.5),[0.5,0.7),[0.7,1]`)와 `schedule_state.csv`; `--gen-diag-every 5` |
| 로깅된 β | 라운드 1 (s=0.9967): β(0.05)=1.2386, β(0.25)=0.9786, β(0.50)=0.6535, β(0.75)=0.3284, β(0.95)=0.0683; 라운드 150 (s=0.5): 1.1197/0.9892/0.8262/0.6631/0.5326; 라운드 300 (s=0): 전부 1.0 |
| 완료 상태 | 6런 중 2런만 완료 (S2 vanilla·curriculum), S1 vanilla는 173/300 라운드에서 중단, S1 curriculum과 WildPPG 쌍은 미실행 → **확증 판정 없음, 탐색적 수치** |

---

## 6. 학습 하이퍼파라미터

### 6.1 A 계열 공통 레시피 (A0-b, A2, A3, A4, A5, A6, A7, A8, A9, B1-v2, C1)

| 요소 | 값 |
|---|---|
| 시드 | 42 (전부) |
| 배치 | 유효 64 (iMF·R2·R3는 마이크로배치 32 × 2 누적) |
| LR / weight decay | 1e-3 / 0.01, AdamW, 스케줄 없음 |
| 최대 라운드 | 300 |
| 라운드 정의 | `--val-every-steps None` → 1 epoch; `--val-every-steps 220` (WildPPG·MIMIC-BP) → **220 옵티마이저 스텝 또는 epoch 끝 중 먼저**; 셔플/epoch 순서는 그대로 |
| 조기 종료 | `is_best = sel < best − min_delta`, `no_improve ≥ patience` → 중단. **patience 20, min_delta 1e-4** (A0 원본만 patience 10, min_delta 0.0) |
| 체크포인트 | 개선 시 `checkpoint_best.pt`; 매 라운드 `checkpoint_last.pt` (옵티마이저 상태, 로더 generator, (t,r) generator, CPU·CUDA RNG 상태 포함). 실제로 재개된 런은 없음 |
| 선택 기준 | OT-CFM `fixed_cfm` (4뱅크) · iMF `fixed_imf_mse` (4뱅크, `train_a2.py`에는 `--select` 플래그 자체가 없음) · 회귀 `val_mse` (결정론적) |
| 검증 서브샘플 | `--val-subsample 4096` (WildPPG 49,200→3,785 stride 13; MIMIC 17,550→3,510 stride 5); DaLiA 1,131은 서브샘플 없음 |
| 진단 (선택에 미사용) | OT-CFM `--gen-diag-every 5` (**A7/A8/A9는 0**): 첫 128 검증 윈도우에 Heun 25스텝(50 NFE), 뱅크 0 노이즈. iMF `--gen-diag-every 1` (A7/A8/A9는 0, B1은 5): 128 윈도우 1-NFE 생성 |
| 벽시계 / 최대 VRAM | 라운드별 벽시계 합(학습+검증+진단) / `torch.cuda.max_memory_allocated` 최대값 |
| 타깃 정규화 | A8·A9만 `--target-norm` (§3.3), 나머지는 항등 |

**옵티마이저 스텝 수 주의**: A 계열과 C1은 스텝 카운터가 나중(커밋 `877841d`)에 추가되어 **기록이 없다**. 아래 표의 "스텝"은 `batch_rounds` 의미론에서 유도한 값이다 (epoch당 배치 = `ceil(n_train/64)`: DaLiA-S2 220, DaLiA-S1 218, WildPPG 4,583, MIMIC-BP 1,547; `--val-every-steps 220`이면 WildPPG는 21라운드마다 183스텝, MIMIC-BP는 8라운드마다 7스텝). 이 유도는 C2 prereg의 "66라운드 = 14,409스텝"을 재현한다. B1(`total_optimizer_steps`), R1(`steps`), R2/R3(`opt_steps`)는 기록값이다.

### 6.2 A 계열 전 런 표

표기: "best"는 1-based 라운드. "스텝"은 유도값(위 주의 참조).

| 런 (`outputs/…`) | 단계 · 트레이너 · git | 목적/모델 | 데이터 · 매니페스트 · n_train/n_val | 선택 · 라운드 정의 | 라운드 · best · 조기종료 · best 지표 | 스텝 (총 / best) | 벽시계 (s) | VRAM (MiB) | 특기 |
|---|---|---|---|---|---|---|---|---|---|
| `a0_penguin_otcfm_ppgdalia_8s_seed42` | A0 · `train_a0` · `6e5a4f1d` | OT-CFM, Heun 25 (50 NFE) | DaLiA · P0 `11c154e4…` · 14,025 / 1,131 | `val_mae` (매 epoch 전체 1,131 윈도우 Heun 25 샘플의 배치평균 MAE) · **patience 10 · min_delta 0.0** · epoch | 21 · 11 · yes · MAE 0.298890 | 4,620 / 2,420 | 2,997.3 | 18,409.6 | 선택에 고정 뱅크 미사용 |
| `a0b_penguin_otcfm_ppgdalia_8s_seed42` | A0-b · `train_a0` · `89983716` (dirty 1) | OT-CFM | 동일 | `fixed_cfm` `6b5c0139…` · 20 · 1e-4 · epoch | 85 · 65 · yes · 0.164451 | 18,700 / 14,300 | 6,365.4 | 18,409.6 | `--val-mae-every 0 --gen-diag-every 5` |
| `a2_imeanflow_s5_ppgdalia_8s_seed42` | A2 · `train_a2` · `62c2b151` | iMF (1-NFE) | 동일 | `fixed_imf_mse` `0f15f0d2…` · epoch | 81 · 61 · yes · 0.173816 | 17,820 / 13,420 | 11,660.1 | 16,903.9 | §5.2 기본값, `--gen-diag-every 1` |
| `a3_otcfm_ppgdalia_testS1_seed42` | A3 · `train_a0` · `41f565ec` | OT-CFM | DaLiA · A3 `6d2999bd…` · 13,899 / 1,131 | `fixed_cfm` · epoch | 114 · 94 · yes · 0.159977 | 24,852 / 20,492 | 8,465.5 | 18,408.7 | |
| `a3_imeanflow_ppgdalia_testS1_seed42` | A3 · `train_a2` · `2457fd5e` (dirty 1) | iMF | 동일 | `fixed_imf_mse` · epoch | 36 · 16 · yes · 0.171271 | 7,848 / 3,488 | 5,140.8 | 16,902.9 | |
| `a4_otcfm_wildppg_seed42` | A4 · `train_a0` · `5dd2b2db` | OT-CFM | WildPPG · A4 `bc168144…` · 293,271 / 3,785 | `fixed_cfm` `ed7cd22b…` · **220스텝 라운드** | 210 · 190 · yes · 0.104851 | 45,830 / 41,467 | 18,530.2 | 20,722.4 | `--val-every-steps 220 --val-subsample 4096` |
| `a4_imeanflow_wildppg_seed42` | A4 · `train_a2` · `5dd2b2db` | iMF | 동일 | `fixed_imf_mse` `2cd14468…` · 220스텝 | 66 · 46 · yes · 0.119459 | 14,409 / 10,046 | 12,982.8 | 19,216.2 | **동결 A4 iMF 체크포인트** (md5 `31c042d2…`) |
| `a5a_mse_regressor_dalia_testS2_seed42` | A5 · `train_a5` · `d9610006` | MSE, `state_token` | P0 · 14,025 / 1,131 | `val_mse` · epoch | 40 · 20 · yes · 0.088090 | 8,800 / 4,400 | 2,729.1 | 18,303.4 | |
| `a5b_mse_regressor_dalia_testS1_seed42` | A5 · `d9610006` | 동일 | A3 · 13,899 / 1,131 | 동일 | 31 · 11 · yes · 0.087837 | 6,758 / 2,398 | 2,097.8 | 18,302.4 | |
| `a5c_mse_regressor_wildppg_seed42` | A5 · `d9610006` | 동일 | A4 · 293,271 / 3,785 | `val_mse` · 220스텝 | 54 · 34 · yes · 0.085353 | 11,806 / 7,443 | 3,861.9 | 20,615.5 | |
| `a6a_fullbackbone_mse_dalia_testS2_seed42` | A6 · `train_a5` · `2fc7841f` (dirty 10) | MSE, `full_backbone` | P0 · 14,025 / 1,131 | `val_mse` · epoch | 44 · 24 · yes · 0.088305 | 9,680 / 5,280 | 3,040.9 | 18,408.5 | **`--x-const 0.1 --t-const 0.5 --cond-scale 0.05`** |
| `a6b_fullbackbone_mse_dalia_testS1_seed42` | A6 · `f4a115f9` | 동일 | A3 · 13,899 / 1,131 | 동일 | 26 · 6 · yes · 0.088824 | 5,668 / 1,308 | 1,780.6 | 18,407.4 | 동일 상수 |
| `a6c_fullbackbone_mse_wildppg_seed42` | A6 · `f4a115f9` | 동일 | A4 · 293,271 / 3,785 | `val_mse` · 220스텝 | 54 · 34 · yes · 0.083669 | 11,806 / 7,443 | 3,907.6 | 20,721.2 | 동일 상수 |
| `a7_otcfm_mimicbp_seed42` | A7 · `train_a0` · `f4a115f9` (dirty 5) | OT-CFM, 타깃 = 원시 mmHg ABP | MIMIC-BP · `c52de946…` · 99,000 / 3,510 | `fixed_cfm` `91206823…` · 220스텝 | 117 · 97 · yes · 6.491819 (mmHg² 스케일) | 22,758 / 18,784 | 9,099.2 | 19,202.3 | `--gen-diag-every 0` |
| `a7_imeanflow_mimicbp_seed42` | A7 · `train_a2` · `ac0016b0` | iMF, 원시 mmHg | 동일 | `fixed_imf_mse` `44c54e58…` · 220스텝 | 70 · 50 · yes · 112.488380 | 13,696 / 9,722 | 12,452.1 | 17,696.1 | |
| `a7_mse_fullbackbone_mimicbp_seed42` | A7 · `train_a5` · `ac0016b0` | MSE full backbone, 원시 mmHg | 동일 | `val_mse` · 220스텝 | 66 · 46 · yes · 218.957077 | 12,816 / 9,055 | 4,227.8 | 19,201.0 | x 0.1 / t 0.5 / cs 0.05 |
| `a8_otcfm_mimicbp_globalz_seed42` | A8 · `train_a0` · `27f14241` | OT-CFM, **타깃 z-정규화** | 동일 | `fixed_cfm` · 220스텝 | 207 · 187 · yes · 0.122447 | 40,215 / 36,241 | 16,091.1 | 19,202.3 | `--target-norm` μ 77.5718 / σ 22.2756 mmHg |
| `a8_imeanflow_mimicbp_globalz_seed42` | A8 · `train_a2` · `27f14241` (dirty 1) | iMF, z-정규화 | 동일 | `fixed_imf_mse` · 220스텝 | 88 · 68 · yes · 0.121259 | 17,017 / 13,256 | 15,420.9 | 17,696.1 | 동일 `--target-norm` |
| `a8_mse_fullbackbone_mimicbp_globalz_seed42` | A8 · `train_a5` · `0b11693d` (dirty 2) | MSE, z-정규화 | 동일 | `val_mse` · 220스텝 | 68 · 48 · yes · 0.436779 | 13,256 / 9,282 | 4,372.9 | 19,201.0 | 동일 |
| `a9_mse_fullbackbone_wildppg_globalz_seed42` | A9 · `train_a5` · `fc3519d9` (dirty 2) | MSE, **ECG 타깃 = 전역 z** | WildPPG `wildppg_8s_prenorm` · A4 · 293,271 / 3,785 | `val_mse` · 220스텝 | 69 · 49 · yes · 0.508057 | 15,069 / 10,706 | 4,955.7 | 20,721.2 | μ 1.575417 / σ 10,501.669122 |
| `a9_otcfm_wildppg_globalz_seed42` | A9 · `train_a0` · `fc3519d9` | OT-CFM, 전역 z ECG | 동일 | `fixed_cfm` · 220스텝 | 134 · 114 · yes · 0.288616 | 29,258 / 24,895 | 11,597.4 | 20,722.4 | `--gen-diag-every 0` |
| `a9_imeanflow_wildppg_globalz_seed42` | A9 · `train_a2` · `fc3519d9` | iMF, 전역 z ECG | 동일 | `fixed_imf_mse` · 220스텝 | 28 · 8 · yes · 0.362067 | 6,123 / 1,760 | 5,502.8 | 19,216.2 | |
| `b1v2_vanilla_fixed_dalia_s2_seed42` | B1-v2 · `train_b1_fixed_compute` · `bdd6419c` | iMF, **고정 연산량**, β ≡ 1 | P0 · 14,025 / 1,131 | 조기 종료 **진단 전용**; `fixed_imf_mse` 추적 · epoch | 300 · best-val 135 · 역사적 조기종료 81 · 0.167250 | **66,000 기록** / 29,700 | 43,138.7 | 16,904.0 | 분할 체크포인트 0/10/25/50/75/100 %, 프로브 `5c44decf…` |
| `b1v2_curriculum_fixed_dalia_s2_seed42` | B1-v2 · `bdd6419c` (dirty 1) | iMF 고정 연산량, 커리큘럼 β | 동일 | 동일 | 300 · best-val 25 · 역사적 45 · 0.172665 | 66,000 / 5,500 | 43,213.3 | 16,904.0 | `--curriculum-lambda 1.304639`; 오버헤드 비율 1.002 |
| `b1v2_vanilla_fixed_dalia_s1_seed42` (**중단**) | B1-v2 · `bdd6419c` (dirty 1) | iMF 고정 연산량, vanilla | A3 · 13,899 / 1,131 | T_schedule 65,400 | 173/300 라운드 · best 16 (0.171271) · 중단 시 s = 0.4233 | 37,714 / 65,400 (기록) | 24,841.2 | 16,903.0 | 2026-08-29 22:30 SIGTERM (X0로 우선순위 재배분) |
| `c1_imf_baseline_replay_seed42` | C1 arm **B** · `train_a2` · `38eaf45a` | iMF, 역사적 샘플러 재현 (`--c1-arm B`) | A4 · `wildppg_8s` · 293,271 / 3,785 | `fixed_imf_mse` · 220스텝 | 66 · 46 · yes · 0.119459 (= A4-iMF 전 자릿수 일치) | 14,409 / 10,046 | 12,911.1 | 19,216.2 | **R2/R3의 동결 생성기** (state sha `47d7ccb9…`) |
| `c1_imf_h25_seed42` | C1 arm **H25** · `a14f5eef` | 양의 h 후반부를 h = 0.25로 강제 | 동일 | 동일 | 68 · 48 · yes · 0.127808 | 14,849 / 10,486 | 13,469.0 | 19,216.2 | |
| `c1_imf_h50_seed42` | C1 arm **H50** · `a14f5eef` | h = 0.50 강제 | 동일 | 동일 | 101 · 81 · yes · 0.118244 | 22,072 / 17,709 | 19,814.3 | 19,216.2 | `P(h≥0.5)` 0.271 vs 0.042(B)/0.021(H25) |

### 6.3 R1 — 리듬 프로브 학습 (`scripts/r1_train_probe.py`, prereg `c7481f9`, git `41a1a071`)

| 항목 | Global-TCN | Local-TCN |
|---|---|---|
| 모델 | §4.3.3 (dilation 1..128) | 동일 구조, dilation 전부 1 |
| 학습 파라미터 | 328,897 | 328,897 |
| 손실 / 타깃 | `BCEWithLogitsLoss`, σ = 12.8 샘플 소프트 R-이벤트 장 | 동일 |
| 데이터 | probe_train 10 피험자 × 4부위 × 2,048 = **81,920** 윈도우; internal_dev (u7y, e61) × 4 × 2,048 = **16,384**; an0/k2s는 트레이너가 절대 읽지 않음 | 동일 |
| 옵티마이저 | AdamW lr 1e-3, **weight decay 1e-4**, **배치 128**, 마이크로배치 없음, `zero_grad(set_to_none=True)` | 동일 |
| 예산 / 조기 종료 | 최대 30 epoch, internal-dev BCE 기준 patience 5, 개선 임계 1e-6; 100스텝 후 런타임 투영이 두 프로브 합 6 GPU-h 초과면 STOP | 동일 |
| 시드 | `seed_everything(42)` + 생성 직전 `torch.manual_seed(42)`; 로더 `Generator().manual_seed(42)` → epoch마다 `randperm` | 동일 초기화·예제 순서 |
| 실현 | 18 epoch, best **13** (JSON `best_epoch` 12, 0-based), dev BCE 0.432998, **11,520 스텝**, 145.7 s, 1,255.5 MiB | 19 epoch, best **14** (JSON 13), 0.492849, **12,160 스텝**, 154.4 s, 1,255.5 MiB |
| 산출 | `checkpoint_best.pt`; Global state sha256 `0986a7af…` = R2/R3 동결 스캐폴드 | — |

### 6.4 R2 — 리듬 스캐폴드 전이 어댑터 (`train_r2_adapter.py`, prereg `f954e07`, git `5f3a3997`, dirty 0)

공통: 동결 생성기 + 동결 Global-TCN; 학습 대상은 `rhythm_adapter.proj.weight` **128개**뿐; 데이터는 A4 train 293,271 윈도우 전체 (`wildppg_8s`, 12 피험자), **검증 윈도우를 한 번도 읽지 않음**, 테스트 피험자 부재 assert; 손실은 동결 `imeanflow_loss` (`TR_KW` = p_mean −0.4 / p_std 1.0 / data_proportion 0.5, `IMF_KW` = norm_p 1.0 / norm_eps 0.01 / jvp forward, `sample_tr_c1(arm="B")`); AdamW lr 1e-3 wd 0.01, 배치 64 = 마이크로 32 × 2, **정확히 2,200 옵티마이저 스텝** (assert), 조기 종료·검증 기반 선택 없음; 체크포인트 step 0/550/1100/2200; 시드 42 (로더 42, tr_gen 43, e는 CUDA 전역), 페어드 난수 프로브 해시 `04aad6ae…` = preflight와 동일; 로더는 140,800 윈도우 방문(같은 윈도우 두 번 없음, 293,271의 1 epoch 미만), 첫 배치 sha256 `3bc4d526…`; 사전 100스텝 예산 게이트 38.6 s / 0.3812 s per step / 6,268 MiB → 3 arm 0.699 GPU-h (예산 6.0), `stop: false`.

| arm | 스캐폴드 | 스텝 | 벽시계 (s) | s/step | 최대 alloc/reserved (MiB) | 최종 ‖W‖₂ | mse 1–100 → 2101–2200 |
|---|---|---|---|---|---|---|---|
| `r2_true_adapter_seed42` | 자기 PPG의 Global-TCN 장 | 2,200 | 846.6 | 0.3843 | 6,268.2 / 6,508 | 7.6975 | 0.1200 → 0.1225 |
| `r2_shuffle_adapter_seed42` | 교란 파트너 윈도우의 장 | 2,200 | 848.0 | 0.3849 | 6,270.3 / 6,572 | 5.8367 | 0.1200 → 0.1224 |
| `r2_oracle_adapter_seed42` (GT-R 누수; 진단 전용) | GT-R 소프트 장 | 2,200 | 841.6 | 0.3820 | 7,414.0 / 7,728 | 12.0420 | 0.1200 → 0.1213 |

### 6.5 R3 — 분리형 타깃측 리듬 융합 (`train_r3_fusion.py`, prereg `3d779fc`, git `7f5dc7e8`, dirty 0)

공통: R2와 **동일한 난수 스트림** (로더 42, `tr_gen` 43, CUDA e; 프로브 해시가 preflight와도 R2의 `04aad6ae…`와도 같음을 assert), 정확히 2,200 스텝, AdamW 1e-3/0.01, 배치 64 = 32 × 2, 동결 `imeanflow_loss`, 검증 미열람, 체크포인트 `module_step{0,550,1100,2200}.pt`, 140,800 윈도우 방문, `cudnn.deterministic = true`; 모듈은 `torch.manual_seed(42)` 직후 고정 순서로 생성하고 초기화 해시를 assert. 융합 상수: d_model 32, 4헤드, 토크나이저 k7/stride4/pad3 → 256 토큰, 정현파 base 10,000; 게이트 hidden 16, pool 33, 초기 p 0.90. 학습 파라미터 TF 12,768 / GTF 12,849. 사전 게이트 (GTF-TRUE 100스텝): 39.7 s, 0.3915 s/step, 6,491.7 MiB → 6 arm 1.435 GPU-h (예산 6.0), `stop: false`.

| arm | 계열 / 게이트 / 스캐폴드 | 학습 params | 벽시계 (s) | s/step | 최대 alloc (MiB) | ‖fusion‖₂ / ‖out‖₂ | 최종 게이트 mean ± std | mse 1–100 → 2101–2200 |
|---|---|---|---|---|---|---|---|---|
| `r3_tf_true_seed42` | tf / – / 자기 | 12,768 | 887.9 | 0.4028 | 6,455.2 | 11.2875 / 4.7347 | – | 0.1205 → 0.1224 |
| `r3_tf_shuffle_seed42` | tf / – / 파트너 | 12,768 | 875.5 | 0.3970 | 6,457.3 | 11.2100 / 4.6673 | – | 0.1205 → 0.1229 |
| `r3_gtf_true_seed42` | gtf / 적응 / 자기 | 12,849 | 891.9 | 0.4047 | 6,491.7 | 12.2010 / 5.0034 | 0.7729 ± 0.3103 | 0.1205 → 0.1204 |
| `r3_gtf_shuffle_seed42` | gtf / 적응 / 파트너 | 12,849 | 874.9 | 0.3967 | 6,494.0 | 11.2886 / 4.7460 | 0.9166 ± 0.0039 | 0.1205 → 0.1230 |
| `r3_gtf_const_seed42` | gtf / const / 자기 | 12,849 | 872.4 | 0.3956 | 6,491.3 | 11.4334 / 4.8442 | 0.9220 ± 0.0005 | 0.1205 → 0.1224 |
| `r3_gtf_oracle_seed42` (GT-R 누수; 진단 전용) | gtf / 적응 / 오라클 | 12,849 | 865.8 | 0.3925 | 7,637.7 | 13.6809 / 5.9428 | 0.8877 ± 0.2747 | 0.1205 → 0.0946 |

### 6.6 스크리닝 · 중단 · 스모크 런 (결과 아님, 완전성 기록)

| 런 | 내용 | 주요 설정 | 실현 |
|---|---|---|---|
| `gradcheck_a6_x1.0`, `_x0.1` | A6 학습 전 하드 테스트 (git `b295a635`, dirty 12–21) | `--x-const {1.0,0.1} --t-const 0.5`, `--cond-scale` 플래그 이전, 12 epoch | best val MSE 0.094875 / 0.095562; 878.9 / 828.2 s; 진폭비 최대 x1.0 8.89e−4 / x0.1 7.70e−4 (학습 실패) |
| `gradcheck_a6_x0.1_cs0.0`, `_x0.1_cs0.05`, `_x1.0_cs0.05` | 동일 + `--cond-scale` | 12 epoch | 0.088654 / 0.089086 / 0.094398; 동결 선택 = x 0.1, t 0.5, cs 0.05 |
| `aborted/a2_hscale1_aborted_epoch1` | 첫 A2 시도 (git `5276bb90`): E(t)+E(h), h_scale 1 | A2 레시피 | 1 epoch(165.3 s) 후 중단, val fixed 0.244426 |
| `aborted/a2_hscale1000_aborted_epoch2` | 두 번째 시도 (`1a2f9daf`): `--h-scale 1000` | A2 레시피 | 발산 (epoch 2 train MSE 395.21, `|du/dt|` 20.73, val 38.109) |
| `aborted/a5{a,b}_…_zero_state_deadstart` | Amendment 1 이전 A5 (타깃 스트림에 0 투입, 3,990,659 params) | A5 레시피 | a5a 36 epoch (val 0.098300, 상수 출력), a5b 2 epoch 후 중단 |
| `smoke_a0/a0b/a0r/a2/a2r` | 파이프라인 스모크 (h_dim 16, 2블록, 50,355 params, `--limit-windows 128`, 2–4 epoch (a0/a0b/a0r 3, a2 2, a2r 4)) | — | 초 단위, 결과 아님 |
| `smoke_a2_full` | 풀 백본 iMF 스모크 (`--limit-windows 64`, 1 epoch) | — | 9.1 s, 16,792.8 MiB |

### 6.7 C2 — 계획되었으나 학습되지 않음

`docs/C2_COMPUTE_MATCHED_MULTISEED_INTERVAL_PREREGISTRATION.md` (동결 `f5120f9`, 구현 `877841d`): 시드 5개 (40–44) × arm 3개 (B, H25, H50) = **15런**, A4 iMF 레시피 그대로, **조기 종료 없이 고정 66라운드 = 63×220 + 3×183 = 14,409 스텝** (15런 전부 동일함을 assert), 평가 체크포인트는 `checkpoint_last.pt`, 추정 54 GPU-h. 상태: **가중치 갱신 이전에 유예** (`docs/C2_DEFERRED_BEFORE_TRAINING.md`). 존재하는 것은 읽기 전용 RNG 통제 preflight와 샘플러 노출 몬테카를로뿐이며 `outputs/*c2*` 디렉터리는 없다. 시드 규약 주의: 로더 시드 = `seed`, (t,r) 시드 = `seed+1`이므로 시드 k의 (t,r) 스트림은 시드 k+1의 로더 스트림과 정수를 공유한다.

### 6.8 메모리 · 연산 특성

| 항목 | 값 |
|---|---|
| 전방 모드 JVP 메모리 (T=1024) | 산문 기록: ≈ 0.51 GiB/샘플 (B = 8/16/32/48 → 4.1/8.2/16.3/24.4 GiB, B = 64 OOM); double-VJP ≈ 0.82; OT-CFM ≈ 0.29 → 그래서 마이크로배치 32 × 2 |
| 학습 최대 VRAM (기록) | A2 DaLiA 16,903.88 MiB (81 epoch 내내 일정); B1 16,904.02; C1 WildPPG 19,216.24; OT-CFM DaLiA 18,409.6; WildPPG OT-CFM 20,722.4 |
| 스텝 시간 | A2 epoch (220스텝 + 4뱅크 검증 + 128윈도우 1-NFE 진단) ≈ 144 s; C1 라운드 (220스텝 + 3,785 윈도우 검증) ≈ 196.6 s; B1 66,000스텝 / 43,139 s = 0.654 s/step (검증 포함) |
| 총 학습 시간 | A2 11,660 s (3.24 h) · A0-b 6,365 s · C1-B 12,911 s · C1-H50 19,814 s · B1 쌍 43,139 / 43,213 s · A4 OT-CFM 18,530 s |
| 추론 지연 (batch 64, 기록) | A2 DaLiA: 1 NFE 81.6 ms (784 samples/s, 1,765.8 MiB), 2 NFE 162.9, 4 NFE 326.9. X4-0 WildPPG (20 워밍업 / 100 반복): 1 NFE 79.7 ms, 4 310.4, 8 628.8, 16 1,252.5, 25 1,961.7, 50 3,923.4; OT-CFM Heun-25 (50 NFE) 3,922.5 ms — **iMF 1-NFE는 OT-CFM 50-NFE 대비 약 49배 빠름** |
| FLOPs | 신뢰할 수 있는 측정 없음 (thop은 S5 scan을 놓침). 감사 문서의 "~3.2 GFLOPs/속도 평가"는 thop 값을 50 NFE로 나눈 것으로 scan 제외 |

---

## 7. 평가 프로토콜

### 7.1 R-peak 검출과 매칭

**검출기 `detect_rpeaks(sig, fs, method="neurokit")`** (`src/ppg2ecg/evaluation/rpeaks.py:20-33`): `nk.ecg_clean(sig, 128, method="neurokit")` → `nk.ecg_peaks(clean, 128, method)["ECG_R_Peaks"]` → 정수 샘플 인덱스. **예측과 참조에 동일 파이프라인** 적용 (upstream PENGUIN은 예측만 clean). 실패 시 빈 피크열 `zeros(0)` 반환 조건: (a) `sig.size < 128`, (b) 비유한 샘플 존재, (c) `std < 1e-8`, (d) neurokit 예외. 경고는 억제. S1.6에서만 검출기 B `pantompkins1985`를 병기 (선택 불가).

**매칭 `match_rpeaks(ref, pred, fs, tol_ms=50.0)`**: `tol = 6.4 샘플` (반올림 없음; `|r−p| ≤ 6.4` ⇒ 실질 |Δ| ≤ 6 샘플 = 46.875 ms). 모든 후보쌍 `(i,j)`를 `(|dt|, i, j)`로 오름차순 정렬 후 greedy 일대일; 반환 `(matches, n_fp, n_fn)`. `prf`: P = m/(m+fp), R = m/(m+fn), F1 = 2PR/(P+R), 분모 0이면 0. **주의**: 참조도 예측도 박동이 없는 윈도우는 여기서 F1 = 0이지만, `event_reliability.peak_train_agreement`(예측-예측 비교)는 둘 다 비면 1.0을 준다.

### 7.2 박동 수준 원시 지표 (`rpeaks.py`)

| 함수 | 정의 | 128 Hz 산술 |
|---|---|---|
| `hr_bpm` | `60 / (mean(diff(rpeaks))/fs)`; 피크 < 2면 NaN. 상수 시간 이동에 정확히 불변 | — |
| `hr_abs_err` | `|hr_pred − hr_ref|`, 어느 쪽이든 NaN이면 NaN | — |
| `rr_mae_ms` | **양쪽 다 매칭된** 연속 참조 박동 쌍에 대한 `|(ref[i+1]−ref[i]) − (pred[m[i+1]]−pred[m[i]])|/fs·1000` 평균 ("매칭 연속 박동 RR MAE") | — |
| `beat_window(sig, r)` | `sig[r − round(0.25fs) : r + round(0.40fs)]`, 배열 밖이면 `None` | `sig[r−32 : r+51]`, **83 샘플**, R은 인덱스 32 |
| `morphology_corr` | 매칭 쌍마다 **각자 자신의 검출 R** 기준 83샘플 창의 Pearson 상관; `None`이거나 std < 1e-8이면 제외; **살아남은 매칭 박동에 대한 평균** ("매칭 박동 형태 상관"). 미매칭 GT 박동은 기여하지 않음 | 83샘플 |
| `qrs_width_ms` | QS 골 프록시: `Q = argmin(sig[r−10, r))`, `S = argmin(sig[r, r+15])`, 폭 `(S−Q)/fs·1000` | — |
| `qrs_width_error_ms` | 매칭 쌍의 `|width_ref − width_pred|` 평균 | — |

### 7.3 윈도우 수준 지표 (`metrics.py`)

- `signal_metrics`: 윈도우별 MAE, RMSE, PCC (중심화 내적 / `sqrt(Σpc²Σtc²)+1e-12`) — 저장된 윈도우 정규화 신호 위에서.
- `rhythm_morphology_metrics(pred, target, fs, tol_ms=50, detector="neurokit")`: 윈도우별 `rpeak_precision/recall/f1, hr_ref, hr_pred, hr_abs_err, rr_mae_ms, qrs_width_err_ms, morph_corr, n_ref_beats, n_pred_beats`.
- `summarize(per_window, n_boot=1000, seed=0)`: 비유한값 제거 후 `mean`, `std (ddof=1)`, `ci95` = 1,000회 **단순 윈도우 부트스트랩** 평균의 2.5/97.5 백분위, `n`. **피험자 층화가 아니다** (prereg 산문은 층화라고 적었으나 코드가 권위).
- `hf_energy_ratio(x, fs=128, cutoff=15 Hz)`: 평균 제거 → `|rfft|²` → **≥ 15 Hz** 전력 비율 (bin 0.125 Hz, 15 Hz는 정확한 bin이며 포함). 주의: `alignment_diagnostics.segment_stats`는 83샘플 박동 구간에서 `f > 15 Hz` (엄격 부등호) — bin 격자와 포함 여부가 다름.
- `penguin_hr_error`: 미수정 upstream `compute_metrics(..., "HeartRateError")`; `mode="corrected"`는 `segment_len=8`, `"as_shipped"`는 4 (upstream의 2× 시간 압축 병리, 진단용).
- `concat_consecutive(x, k)`는 모든 실행 스크립트가 `hr_window_segments=1`로 호출하므로 **실제로 쓰이지 않는다** (처리본이 이미 8 s). `metrics.py` 독스트링의 "4 s 윈도우 → 8 s 2연결"은 폐기된 v0 계획.

### 7.4 A 계열 평가 스크립트 파생량

| 컬럼 | 정의 |
|---|---|
| `amp_ratio`, `amp_ratio_median` | `pred.std(1) / (y.std(1) + 1e-8)`의 윈도우 평균·중앙값 |
| `hf_ratio_pred`, `hf_ratio_target` | `hf_energy_ratio` 윈도우 평균 |
| PPG-shuffle 조건화 이득 | `perm = derangement(n, seed 1)`; `pred_sh = sample(x_te[perm], 같은 x0)`; `hr_right` = 주어진 PPG에 속한 타깃 `y_te[perm]` 대비, `hr_wrong` = 원래 타깃 대비; `cond_gain_bpm = hr_wrong − hr_right`. (산문은 참조 HR을 `nk.ppg_findpeaks`로 얻는다고 적었으나 **코드는 페어드 ECG 타깃 사용**) |
| 시드 다양성 | 첫 `m = min(256, n)` 윈도우, 기본 예측 + 3개 추가 뱅크 (`manual_seed(0+100+k)`, k=1..3, 총 4시드; 산문은 8시드); `seed_std_mean` = `[4,256,1024]` 표준편차 평균, `seed_pairwise_corr` = 6개 시드쌍 윈도우별 Pearson 평균 |
| `frac_windows_no_pred_beats` | `mean(n_pred_beats == 0)` |
| `beats_ratio` (A2/A5/A9 CSV) | `mean(n_pred_beats) / max(mean(n_ref_beats), 1e-9)` — **윈도우 평균들의 비**, 윈도우별 비의 평균이 아님 |
| MSE 회귀 행 (A5/A6, `eval_a5.py`) | `seed_std_mean = 0.0`, `seed_pairwise_corr = 1.0` (구조상 결정론적), `params_total`/`params_effective` 컬럼 추가. **A9(`eval_a9.py`)는 예외**: 시드 다양성 뱅크를 아예 만들지 않고 `row_for`가 OT-CFM·iMF·MSE **모든 arm**에 0.0/1.0을 상수로 기록한다 (생성 arm의 값은 측정값이 아님); A9 CSV에는 `params` 컬럼이 없고 파라미터 수는 `metrics.json` 최상위 `params`에만 있다 |

### 7.5 이벤트 계열 지표 (X0/X4-0 이후: C0, C1, M1, R1, R2, R3)

`event_timing(gt, pred, fs, tol_ms=50)`: 동결 검출기 + 동결 매처 → `n_ref, n_pred, n_matched, n_missing(FN), n_spurious(FP)`, 매칭별 `signed_err_ms`, 두 피크열.

윈도우별 (n_ref = max(n_ref,1)): `f1/precision/recall` (50 ms), `beats_ratio = n_pred/n_ref`, `beats_ratio_dev = |beats_ratio − 1|`, `missing = n_missing/n_ref`, `spurious = n_spurious/n_ref`, `n_matched`, `morph` (매칭 박동 형태 상관, `zero_contrib_window_frac` 병기), `rr_mae_ms`, `hf_ratio/hf_gt/hf_err`, (R2/R3) 풀링된 `timing_median_abs_ms`, `timing_mean_ms`, `timing_frac_le25ms`, `n_matched_beats`. **모집단 수준 `matched_coverage` = `Σ n_matched / Σ n_valid_gt_beats`**.

집계: `macro(values, subjects)` = 피험자별 `nanmean`의 **동일 가중 평균**.

**커버리지 경고 (필수 병기)**: 2,048 윈도우 개발 모집단·소스 시드 0·iMF NFE 8 기준으로 `morph`는 19,834 GT 박동 중 7,910개 (39.9 %)만 평균하고, 2,048 윈도우 중 254개 (12.4 %)는 아무 기여도 하지 않는다. `rr_mae_ms`는 iMF-8에서 1,276 윈도우 (62 %), OT-CFM-50에서 1,383 (68 %)만 사용한다. 따라서 `morph`/`rr_mae_ms`는 **커버리지와 함께 보고해야 하고, recall이 다른 arm 간에 비교할 수 없다.**

### 7.6 개수 정합 무작위 위상 chance floor와 F1 excess

| 항목 | 규칙 |
|---|---|
| `chance_random_phase(n_beats, n_time, rng)` | `step = n_time/n`; `off = rng.random()·step`; 피크 = `unique(round(off + step·k) mod n_time)`, k=0..n−1 — 등간격, **그 윈도우의 검출된 예측 피크 수에 개수 정합**, 위상 무작위 |
| `chance_circular_shift` | `(peaks + rng.integers(n_time)) mod n_time` (S1.4c 대안) |
| 추출 수 / 시드 | `NULL_DRAWS = 20`, `NULL_SEED = 20260901`; rng는 **64 윈도우 채점 청크마다 한 번** 생성되어 그 청크 모든 윈도우의 20회 추출이 공유 (결정적이지만 윈도우별 독립은 아님) |
| `chance_f1` | 윈도우별 20회 추출의 F1 평균 (GT 대비 50 ms 매칭) |
| `f1_excess` | `f1 − chance_f1` (윈도우별) → `macro` |
| 보정 결과 | 무작위 위상과 순환 이동 floor가 ≤ 0.002 일치; **각 arm 50 ms raw F1의 약 ¼이 우연** (2,048 윈도우 모집단에서 floor 0.1013–0.1215) |

### 7.7 구조 지표 S1–S8 (GT 고정 좌표, 정렬 없음)

S1–S8이라는 이름은 R2/R3 preregistration에만 있고, 코드는 `STRUCT5 = ("raw_rmse","raw_corr","raw_qrs_rmse","qrs_deriv_rmse","qrs_curvature_err")`이며 S6–S8은 C0/M1 이름으로 보고된다.

**A 계열 — 83샘플 GT 박동 구간** (`beat_segments_gt` + `segment_stats`): 모든 GT R에 대해 `[r−32, r+51)`; 유효 박동 조건은 `r−32−19 ≥ 0` 그리고 `r+51+19 ≤ 1024` (여백 19 = max_shift, `raw_*`만 읽어도 적용); QRS 하위창 `[r_local−13, r_local+13]` = 27샘플.

| 라벨 | 컬럼 | 박동별 정의 | 윈도우 집계 | 방향 |
|---|---|---|---|---|
| S1 | `raw_rmse` | 83샘플 RMSE | `nanmean` | 낮을수록 좋음 |
| S2 | `raw_corr` | 83샘플 Pearson (예측 평탄 & GT 아님 → 0.0; GT 평탄 → NaN) | `nanmean` | 높을수록 |
| S3 | `raw_qrs_rmse` | 27샘플 QRS 하위창 RMSE | `nanmean` | 낮을수록 |
| S6 | `qrs_e_dev` | `var(p[QRS])/var(g[QRS])` 비의 `|중앙값 − 1|` | `nanmedian` → `|·−1|` | 낮을수록 |
| S7 | `p2p_dev` | `ptp(p)/ptp(g)` 비의 `|중앙값 − 1|` | 동일 | 낮을수록 |
| (C0 주지표) | `slope_dev` | `max|diff(p)·fs| / max|diff(g)·fs|` 비의 `|중앙값 − 1|` | 동일 | 낮을수록 |
| S8 | `hf_ratio` / `hf_err` | 윈도우 전체 ≥ 15 Hz 전력 비율, `hf_err = |hf_pred − hf_gt|` | `macro` | 방향 주장 없음 |

C0 동결 주지표: `raw_corr, qrs_e_dev, slope_dev, p2p_dev, raw_qrs_rmse, raw_rmse`. C1은 이를 `M1_qrs_e_dev, M2_p2p_dev, M3_qrs_rmse, M4_rmse` (간격 지표), `M5_raw_corr`, `M6_slope_dev`로 재라벨링.

**B 계열 — M1 QRS 코어** (`m1_structural.py`): `CORE_MS 80 → 10 샘플`, `PERI_MS 250 → 32`; `tau_map` = 각 샘플의 최근접 GT R까지 부호 있는 거리; 영역 `qrs_core |τ| ≤ 10` (박동당 21샘플), `peri_qrs 10 < |τ| ≤ 32`, `background |τ| > 32`.

| 라벨 | 컬럼 | 정의 (조건 `r−11 ≥ 0`, `r+12 ≤ 1024`) |
|---|---|---|
| — | `qrs_rmse_core` | `pred[r−10 : r+11]` (21샘플) RMSE, 박동 평균 |
| — | `qrs_ptp_dev` / `qrs_energy_dev` / `qrs_slope_dev` | `median|비 − 1|` (ptp / Σp² / max|d1|) |
| **S4** | `qrs_deriv_rmse` | 22개 1차 차분의 `sqrt(mean((d1p − d1g)²))`, 박동 평균 |
| **S5** | `qrs_curvature_err` | `mean|d2p − d2g|`, `d2[n] = x[n+1] − 2x[n] + x[n−1]` (23샘플 구간 → 21값) |
| 비고 | `d1`, `d2`는 `np.diff` (segment_stats와 달리 `·fs` 스케일 없음) |

영역 오차 A1–A5 (영역별): `a1_abs = mean|p−g|`, `a2_sq`, `a3_dabs = mean|d1p − d1g|`, `a4_amp = |ptp(p) − ptp(g)|`, `a5_energy = |Σp² − Σg²|/(Σg²+1e-12)`, `n`. `event_profile`: τ ∈ [−32, +32] (65점)의 평균 `|p−g|`와 `|d1p − d1g|`. 스펙트럼: `welch(fs=128, nperseg=256, noverlap=128, hann, detrend="constant")`, 사다리꼴 대역 에너지 F1 0.5–4, F2 4–8, F3 8–15, F4 15–64 Hz (GT·예측·잔차), `ratio_dev = |E_pred/E_gt − 1|`. 아틀라스 전용: `qrs_band_component` = 영위상 Butterworth 4차 **8–40 Hz**, `energy_envelope` = x²의 25샘플(≈195 ms) 이동 평균.

### 7.8 오라클(정렬) 진단 — 정의되었다가 철회됨

`alignment_diagnostics.py`: `GLOBAL_MAX_LAG_MS 250` (±32), `LOCAL_MAX_SHIFT_MS 150` (±19), `QRS_HALF_MS 100`, `HF_CUT_HZ 15`. `oracle_local_shift`는 `d ∈ [−19,19]` 중 Pearson 상관 최대를 고름 (평탄 예측은 −1.0, 동률은 최소 |d|). `oracle_absent = oracle_corr < 0.5 or oracle_p2p_ratio < 0.2`.

**철회 (S1.4b)**: 동일한 39-shift 최대화를 **불일치 쌍**(같은 윈도우의 다른 박동 j ≠ i, 20회 추출)에 적용하면 오라클 이득이 그대로 재현된다 — 참-대-널 이득 초과분이 +0.0001…+0.0004 (MSE −0.0007)인 반면 이득 자체는 +0.28…+0.59. 결론: `oracle_corr`, `oracle_qrs_energy_median`, `oracle_absent` 및 ±150 ms 이동 기반 통계는 **"사용된 방식으로는 신뢰할 수 없음"**이며 이에 근거한 변위 주장은 철회. 이후 C0·C1·M1·V1은 `oracle_metrics_used(_for_decision): false`를 기록한다 (R1/R2/R3 아티팩트에는 이 키가 없다). R2/R3의 "ORACLE" arm은 **다른 대상** (GT-R 파생 스캐폴드 입력, "GT-R 누수; 진단 전용" 라벨).

### 7.9 모집단과 테스트 피험자 방화벽

| 단계 | 스크립트 | 피험자 | 선택 규칙 | 윈도우 | GT 박동 | 노이즈 | 부트스트랩 |
|---|---|---|---|---|---|---|---|
| A0/A0-b/A2/A5a/A6a | `eval_a0_nfe_curve`/`eval_a2`/`eval_a5` | DaLiA test S2 | 전부 | 1,025 | — | seed 0 | 윈도우 1,000 |
| A3/A5b/A6b | 동일 | DaLiA test S1 | 전부 | 1,151 | — | seed 0 | 윈도우 |
| A4/A5c/A6c/A9 (**테스트**) | 동일 + `--subsample 4096` | **kjd, ssx** | 매니페스트 순 연결, stride 12 | 3,907 (kjd 2,008 / ssx 1,899) | 40,523 | seed 0 | 윈도우 1,000 |
| A7/A8 | `predict_a7` + `analyze_a7` (`abp_metrics`: 수축기 피크 허용 100 ms, 맥파 템플릿 −0.25…+0.55 s, HF > 5 Hz, 피크 영역 ±150 ms, 최소 맥파 0.3 s) | MIMIC 공식 test | stride 6 | 3,435 | | seed 0 | |
| X0 | `analyze_x0_error_decomposition` | 동결 테스트 예측 재사용 | | WildPPG 3,907 / S2 1,025 / S1 1,151 | | seed 0, 교란 1 | 클러스터 (8 subject×site) 2,000, seed 0 |
| X4-0 | `analyze_x4_0_event_reliability` | **개발 an0, k2s** | SHA256 코호트 (§3.5) | NFE 2,048 / source 512 / schedule 1,024 | 19,834 | 시드 0–3 (풀), 0–31 | 피험자 층화 2,000, 20260830 |
| S1 | `analyze_s1_*` | an0, k2s (+템플릿용 train 10) | `x4-event-nfe-v2` 2,048 | 2,048 | 19,834 | seed 0 | 피험자 2,000, 20260901 |
| C0 | `analyze_c0_compression_target` | an0, k2s | 동일 2,048 | 2,048 | 19,834 | seed 0, sha `86808579…` | 페어드 2,000, 20260901 |
| C1 / M1 | `eval_c1_arms`, `analyze_m1_structural` | an0, k2s | 동일 (+M1 아틀라스 8/스트라텀 = 64) | 2,048 | 19,834 | seed 0 | 페어드 2,000, 20260901 |
| V1 | `analyze_v1_stepwise` | 12 train + an0, k2s (14) × 4부위 | `v1-…` 중첩 prefix | VIZ 448 / METRICS 1,792 / DELAY 7,168 (51,957 R→PPG 쌍) | — | 스트라텀별 seed 0 | 없음 (평균) |
| R1 | `r1_evaluate` | probe_train 10 / dev u7y·e61 / **val an0·k2s** | `r1-…` 코호트 | 8,192 (검증) | 79,111 | — (결정론적) | 페어드 2,000, 20260902 |
| R2 / R3 주 분석 | `r2_evaluate`, `r3_evaluate` | an0, k2s | 2,048 부분집합을 `nfe_subset.json`과 **원소 단위 assert** (≠2,048 윈도우 또는 ≠19,834 박동이면 STOP) | 2,048 | 19,834 | seed 0, sha assert | 페어드 2,000, 20260902 |
| R2 / R3 부위별 2차 | `site_wise` | an0, k2s | R1 검증 코호트 | 8,192 | — | seed 0, sha `77a3e062…` | 부위 내 페어드 + 2군 대비 |

**방화벽**: `TEST_SUBJECTS = ("kjd","ssx")`; `assert_no_test_subjects(subjects)`가 요청 시 `WildPPGTestFirewallError`. 호출 위치는 X4-0 이후 모든 분석 스크립트의 첫 문장 (X4-0, C0, C1, M1, V1, R1, R2, R3); `stamping.collect_train_beats`는 an0/k2s를 템플릿 소스로도 금지. 모든 아티팩트가 `"test_subjects_loaded": []`를 기록.

**이력**: kjd/ssx는 A4, A5c, A6c, A9의 동결 테스트 평가와 그 배열을 재사용한 X0/X2 재분석에서 **실제로 사용되었다**. X3-G0/X4-0 이후 단계(S1, C0, C1, M1, V1, R1–R3)는 로드하지 않는다.

**개발 주의**: an0/k2s의 4개 윈도우(an0 9066, 18138; k2s 5852, 16436)는 X4-0 이전에 시각적으로 미리 보았고 모든 부분집합에서 제외된다. 따라서 an0/k2s는 "개발 검증"이지 순수 검증이 아니다.

### 7.10 노이즈 뱅크

| 뱅크 | 구성 | 형상 | 해시 |
|---|---|---|---|
| A 계열 테스트 뱅크 | `Generator().manual_seed(0)`으로 CPU에서 `randn(n,1,T)`; **같은 텐서**가 모든 OT-CFM arm의 `x0`이자 모든 iMF arm의 `e` | `[3907,1,1024]` / `[1025,…]` / `[1151,…]` | **기록 없음** (`metrics.json`에 `noise_seed: 0`만) |
| 개발 2,048 뱅크 | `randn(2048,1,1024, manual_seed(0))` | `[2048,1,1024]` | `868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f` (코드에서 assert) |
| R1 8,192 부위 코호트 뱅크 | seed 0 | `[8192,1,1024]` | `77a3e0624efcae751a61072ec5e31b56b6f16c7452b7d4d286212cbb9ad311d8` |
| X4-0 `source_bank(seed, n)` | 시드 0–3 (NFE 프론티어), 0–31 (소스 진단), `SOURCE_PAIR {0:1, 2:3}` (조건 섭동), 교란 `default_rng(1)` | | 없음 |
| 다양성 뱅크 | 시드 100+k, k=1..3 | `[256,1,1024]` | — |

### 7.11 통계

| 추정량 | 함수 | 재표집 단위 | 반복 | 시드 | 판정 |
|---|---|---|---|---|---|
| 윈도우 부트스트랩 | `metrics.summarize` | 윈도우 (풀링, **층화 아님**) | 1,000 | 0 | 2.5/97.5 백분위 |
| 클러스터 부트스트랩 | `analyze_x0…cluster_bootstrap` | WildPPG 8개 (subject, site) 클러스터 | 2,000 | 0 | 백분위 |
| 피험자 층화 (서술) | `event_reliability.subject_stratified_bootstrap` | 피험자 내 윈도우 → 피험자 평균 동일 가중 | 2,000 | 20260830 | 백분위 |
| 피험자 부트스트랩 | `s1_audit.subject_bootstrap` | 동일 | 2,000 | 20260901 | 백분위 |
| **페어드 피험자 층화** | `paired_stats.paired_subject_bootstrap` | 윈도우별 방향 있는 차이 `d`; 피험자마다 **같은** 재표집 인덱스를 `d`에 적용; 피험자 동일 가중 | 2,000 | 20260901 (C0/C1/M1) · **20260902** (R1/R2/R3) | **`improves` if lo > 0, `worsens` if hi < 0, else `unresolved`** (양수 = 나중 arm이 더 좋음) |
| 계층 (시드 × 윈도우) | `hierarchical.hierarchical_bootstrap` | 외부: 학습 시드, 내부: 피험자 층화 윈도우 | 5,000 | 20260902 | C2 — **미실행** |
| 2군 독립 부트스트랩 | `r2_evaluate.site_wise` | 원위(wrist+ankle) vs 근위(sternum+head)의 방향 있는 TRUE−B 효과 | 2,000 | 20260902 | `distal_gains_more` / `proximal_gains_more` / `unresolved` |
| 통계량 부트스트랩 | `r3_evaluate.subject_boot` | 피험자 내 윈도우, NaN 추출은 제외·계수 | 2,000 | 20260902 | R3 게이트 진단 (Spearman ρ, matched−unmatched) |
| NaN 규칙 | R2는 arm 간 NaN 패턴이 다르면 중단; R3는 쌍별 불완전 윈도우를 버리고 provenance에 표기 | | | | |

### 7.12 NFE 격자 (단계별)

| 단계 | arm × 스케줄 | 실현 NFE |
|---|---|---|
| A0/A0-b/A3/A4/A9 OT-CFM | `heun:25, heun:10, heun:5, heun:2, heun:1, euler:1` | 50, 20, 10, 4, 2, 1 |
| A2/A3/A4/A9/B1 iMF | meanflow 1(주), 2, 4 (진단) | 1, 2, 4 |
| A5/A6 회귀 | 단일 순전파 | 1 (생성 NFE 아님) |
| X4-0 | UNIFORM 1,2,4,8,16,25,50 × 소스 시드 0–3; 시드별 OT-CFM Heun-25 기준선; 32소스 (NFE 1,4,8,16); 조건 섭동 (NFE 1,8); 스트레스 스케줄 × 시드 0–3 | 위와 같음 |
| S1.3 | iMF 1,4,8,50; OT-CFM-50; MSE(a6c) | |
| C0 | iMF 1,2,4,8 (동결 격자) | |
| C1 / M1 | arm B/H25/H50 × NFE 2,4 | |
| V1 | C1 baseline replay 1,2,4,8,50 | |
| R2 | B/TRUE/SHUFFLE/ORACLE × 1,2,4 + PHASE(스캐폴드 +256샘플 roll) @4 | 주지표 **NFE 4** |
| R3 | 9 arm × 1,2,4 + PHASE-TF/GTF, NODIRECT-TF/GTF @4 | 주지표 NFE 4 |

### 7.13 지연 측정

`efficiency.benchmark(sample_fn, n_warmup, n_repeats, batch_size, device)`: 워밍업 → `synchronize(); reset_peak_memory_stats()` → 반복마다 `synchronize`, `perf_counter`, 호출, `synchronize` → `latency_ms_median`(주지표), mean, std, `samples_per_s`, `peak_mem_MiB`. A 계열 호출부는 **반복 10**이되 워밍업이 갈린다 — A0/A0-b/A2/A3/A4 (`eval_a0_nfe_curve.py:130`, `eval_a2.py:113`)와 A5/A6 (`eval_a5.py:91`)는 **워밍업 3**, A7/A8 (`predict_a7.py:123,135,144`)과 A9 (`eval_a9.py:145`)는 **워밍업 2**. 배치 = 테스트 첫 64윈도우, fp32, `torch.compile` 없음 (산문 prereg는 "워밍업 5 / 20회 중앙값 / CUDA 이벤트"라 적었으나 코드가 권위). X4-0은 **워밍업 20 / 100회 CUDA 이벤트** (median, p10, p90).

A4 결과 (batch 64): OT-CFM heun25/50 NFE 4,159.29 ms (15.39 samples/s, 1,766.54 MiB) · heun10/20 1,630.42 · heun5/10 810.88 · heun2/4 320.45 · heun1/2 159.69 · euler1/1 81.49 ms (785.41 samples/s); iMF 1/2/4 NFE 81.43 / 162.80 / 319.68 ms; MSE 회귀 a5c 79.73 ms, a6c 81.48 ms.

### 7.14 단계별 동결 결정 규칙 (코드에 고정)

| 단계 | 규칙 | 결과 |
|---|---|---|
| C0 | Gate A (2→4): 6개 주지표 중 ≥2 `improves`, 0 `worsens`, `f1_excess`가 `worsens` 아님. Gate B (4→8): Gate A ∧ ≥2 개선 ∧ 0 악화 ∧ `f1_excess`·`beats_ratio_dev` 악화 없음 | **압축 목표 = NFE 4** |
| C1 1단계 | B의 2→4: M1–M4 중 ≥3 개선, 악화 0, F1-excess 붕괴 없음 | PASS (4/4) |
| C1 2단계 | 간격 폐쇄 `G = dev(B,2) − dev(B,4)`, `I = dev(B,2) − dev(X,2)`, `C = I/G`; arm 게이트 (NFE 2): M1–M4 중 ≥3 개선, M1–M6 중 악화 0, `C ≥ 0.50`인 지표 ≥2, F1-excess·beats-ratio-dev 악화 없음; 특이성 = 페어드 H50−H25 개선 | `TARGET h=0.5 EXPOSURE SUPPORTED` |
| R1 | (1) Global이 Local을 {F1@150, F1@200, RR MAE, beats-ratio dev} 중 ≥2에서 이김; (2) TRUE > WINDOW-SHUFFLE (F1@200, RR MAE); (3) RR 중앙 AE < 100 ms 또는 상대 중앙값 < 0.10; (4) beats-ratio dev@150 < 0.20 | 4/4 → `GLOBAL RHYTHM SCAFFOLD SUPPORTED` |
| R2 | NFE 4에서 (1) TRUE vs B `f1_excess` 개선 **및 점추정 ≥ 0.02**; (2) TRUE > SHUFFLE; (3) {beats_ratio_dev, missing, spurious} 중 하나 개선; (4) S1–S5 중 악화 < 2개; (5) TRUE `beats_ratio_dev < 0.20` | item1 false (+0.01935 [+0.0160,+0.0227]), item4 false (S4·S5 악화) → **`SCAFFOLD INFORMATIVE, MINIMAL INTERFACE INSUFFICIENT`** |
| R3 | U1–U6 (TF) / G1–G6 (GTF): `GATE_MIN_EFFECT 0.02`, `GATE_BEATS_DEV_MAX 0.20`, `NONINFERIORITY_MARGIN −0.005`, `ORACLE_LIFT_MARGIN 0.010`; 직결 경로 판독 = B 대비 F1-excess 이득의 **≥ 50 %가 상쇄 후 생존**하면 "타깃 스트림 경유"; 게이트 문구는 Spearman ρ_A ≥ 0.20 & CI lo > 0 & 게이트 std ≥ 0.01일 때만 "RELIABILITY-LIKE BEHAVIOR OBSERVED" | U = (F,F,T,F,F,T), G = (T,T,F,F,F,T) → **`EVENT GAIN WITH STRUCTURE TRADE-OFF PERSISTS`** |
| X4-0 | few-step 포화 (NFE 8/16이 iMF-50의 morph 0.03 / oracle_corr 0.03 / F1 0.03 이내 등); 소스 민감도 (시드쌍 F1 < 0.80, 박동수 SD ≥ 0.75, 조건 타이밍 SD ≥ 15 ms, F1 SD ≥ 0.05 중 ≥2 → material) | |

**R2/R3 주요 수치 (NFE 4, F1 excess)**: B 0.3176 · ADD 0.3369 · TF-TRUE 0.3341 (+0.0165 [+0.0133,+0.0196] vs B) · TF-SHUFFLE 0.3334 · **GTF-TRUE 0.3582 (+0.0406 [+0.0353,+0.0458])** · GTF-SHUFFLE 0.3340 · GTF-CONST 0.3349 · **GTF-ORACLE 0.8164** (GT-R 누수; 진단 전용) · ADD-ORACLE 0.3601. 직결 경로 생존 비율 TF 0.82 / GTF 0.93; 게이트 ρ_A = −0.49, matched−unmatched = −0.345 → "게이트를 신뢰도로 해석할 수 없음".

### 7.15 R2/R3 보조 정의

- 위상 절제 `PHASE_SHIFT_SAMPLES = 256` (+2.0 s), `φ = frac(256 / 평균 GT RR)`, 층 `in_phase (φ<0.1 또는 ≥0.9)`, `anti_phase (0.4≤φ≤0.6)`, `rest`, `undefined`.
- 지속성 매처 `PERSIST_TOL_MS = 250` (32샘플) — NFE 1/2/4 간 greedy 일대일.
- `scaffold_event_f1`: `extract_events(s, threshold 0.35, refractory 32)`을 50 ms에서.
- 회귀 검사: R2는 `artifacts/c1_interval_exposure/stage1_metrics.csv`의 arm B / NFE 4 행과 대조해 `STAGE1_MAP` 컬럼의 |Δ| > 1e-6 플래그를 provenance(`regression_vs_c1_stage1_B4`)에 **기록만** 한다 (중단 없음). **R3만** R2 행과 비교해 arm B·ADD의 `f1_excess`, `qrs_deriv_rmse`, `qrs_curvature_err`가 1e-6을 넘으면 STOP (ADD-ORACLE은 플래그만).
- R1 이벤트 채점: 허용 오차 50/100/150/200/250 ms, RR 지표는 150 ms에서 연속 매칭 GT 박동; 통제군 WINDOW-SHUFFLE (subject×site 내 교란), CIRCULAR-SHIFT (`np.roll` 균등 [128,512] 샘플 = 1–4 s), `default_rng(20260902)`. Global 결과 (8,192 윈도우 / 79,111 GT 박동): F1@50/100/150/200/250 = 0.6199/0.7805/0.8582/0.8975/0.9223, 타이밍 중앙값 23.4 ms, beats ratio 1.105, RR MAE 31.4 ms (중앙 15.6 ms, corr 0.891); Local F1@50 0.4646, RR MAE 51.9 ms; SHUFFLE/SHIFT F1@50 = 0.134 / 0.134.
- V1 지연 감사: 이후 첫 PPG 수축기 피크를 [80, 800] ms (10–102 샘플)에서 일대일 전방 탐색; foot 프록시 = 400 ms(51샘플) 후방 창의 argmin, 20 % 초과 실패 시 중단.
- S1 G1 스탬핑 게이트: 템플릿 A = train 10 피험자(잡음 fex·p5d 제외)의 박동 중앙값, 피크-투-피크 중앙값으로 스케일; T-B = [−80, +120] ms 크롭 (26샘플, R은 인덱스 10); T-C Ricker σ 10 ms; T-D는 < 1 Hz FFT 기저선 추가. 게이트: T-B가 50 ms에서 macro F1 ≥ 0.95 → **관측 0.9993 PASS** (T-A 통제 0.857).

---

## 8. 기록 간 불일치 (코드 / `train_meta.json`이 권위)

| 위치 | 적힌 내용 | 실제 (권위) |
|---|---|---|
| `a7_imeanflow`, `a8_imeanflow`, `a9_imeanflow`, 모든 `b1v2_*`의 `config.yaml` | `early_stopping.metric: val_mae_batchmean`, `selection.criterion: val_mae` | preflight가 `--select` 없이 실행됨; `train_a2.py`/`train_b1`에는 `--select` 자체가 없고 항상 `fixed_imf_mse` |
| `a2`, `a3_imeanflow`, `a4_imeanflow`의 `config.yaml` | `selection.criterion: fixed_cfm`, `early_stopping.metric: val_cfm_fixed` | 동일 이유; 같은 파일의 `imeanflow.selection: fixed_imf_mse` 블록은 맞음. iMF 뱅크 해시는 정확히 기록됨 |
| 모든 A6형 런(`a6*`, `a7_mse`, `a8_mse`, `a9_mse`)의 `config.yaml` `model` 문자열 | "x_const=1, t_const=0.5" | `preflight_a0.py:141`의 하드코딩 문자열; 실제는 `--x-const 0.1 --t-const 0.5 --cond-scale 0.05` |
| A5/A6형 `config.yaml` `model_cfg.n_step` | 25 | 회귀 모델은 `n_step=1`로 생성 (무관: `n_step`은 upstream 샘플링에만 쓰이고 회귀는 호출하지 않음) |
| `a7_otcfm`, `a8_otcfm`, `a9_otcfm`, 모든 A5/A6의 `config.yaml` | `selection.val_mae_every: 1` | 실제 `--val-mae-every 0` (OT-CFM의 `val_mae_batchmean`은 로그에서 nan); A5/A6에는 해당 플래그 없음 |
| `a0_penguin…`의 `config.yaml` | `min_delta` 없음 | 코드 기본값 0.0 |
| `docs/A2_IMEANFLOW_REPORT.md` "Frozen protocol" | 조건화 "E(t)+E(h)" | 체크포인트 `cond_mode = "h_only"`, `h_scale = 1.0`; prereg §9가 결과 이전 전환을 문서화. `docs/IMEANFLOW_AUDIT.md` §3의 "h-only는 채택하지 않은 대안" 문구는 낡음 |
| `docs/A5_…PREREGISTRATION.md` §3 | 3,990,659 params, 타깃 스트림 0 입력 | Amendment 1 코드 = 3,990,787 (state token). 0 입력 모델은 영구 dead start로 `outputs/aborted/`에 보관 |
| `regressor.py` 독스트링 L13 | dead = 264,192 (`cross_attn`) | `count_regressor_params`는 264,194 (`revin` 포함) |
| `docs/PREREGISTRATION_V0.md:50` | 1,000회 부트스트랩이 "피험자 층화" | `metrics.summarize`는 **층화 없는 풀링 윈도우** 부트스트랩 |
| `docs/PREREGISTRATION_V0.md:47-49` | 지연 = 워밍업 5 / 20회 중앙값 / CUDA 이벤트; PPG-shuffle 참조 HR = `nk.ppg_findpeaks`; 다양성 8시드 | 코드 = 워밍업 3 / 10회 / `perf_counter`; 페어드 ECG 타깃 `y_te[perm]`; 4시드 |
| `metrics.py` 독스트링 L111-114 | 4 s 신호 윈도우, 8 s = 2연결 | 모든 실행은 8 s 네이티브, `hr_window_segments=1` |
| `samplers.py` 독스트링 | `tests/test_samplers_match_upstream.py` 인용 | 그런 파일 없음; 패리티 테스트는 `tests/test_upstream_parity.py` |
| upstream `configs/upstream/train.yaml` | `ema_decay 0.999`, `earlystop_patience 10` | EMA는 upstream 코드도 읽지 않음; 본 프로그램은 A0만 patience 10, A0-b 이후 20 |
| `model_manifest.json` (R1) | `params: 0` | `requires_grad_(False)` 이후 센 값 — 328,897로 정정 (보고서 §14 item 3) |
| `docs/C2_…PREREGISTRATION.md:32` | GPU 32,607 MiB | `provenance.json` 32,109 MiB (측정 방식 차이) |
| `outputs/*/…/best_epoch` | 0-based | 보고서는 1-based로 인용 (본 문서는 1-based로 통일) |
| X4-0 prereg §5 | 코호트 해시 키가 `window_index` | 코드는 **배열 행 위치 i**를 해시 (site-major npz에서 `window_index`와 다름). R1/V1/C2 코호트는 진짜 `window_index`+`site`를 해시 |
| PENGUIN 논문 / 그림 | PPG 융합 = "linear projection", L개 블록의 연쇄 스택 | 코드 = 2층 GELU MLP, 비연쇄(병렬·합산) 타깃 스트림. 커밋 `6cd70cd`가 모든 체크포인트에 대해 권위 |
| upstream `summarize()` | thop `Params 3.25 M`, `GFLOPs 60.77` | 이 모델에 대해 틀림 (S5 원시 텐서·MHA 누락) — 인용 금지 |

---

## 9. 남은 모호성 (문서 작성 시점에 저장소만으로 해소 불가)

**모델·파라미터**
1. 파라미터 수 규약: 4,568,707 (torch numel) vs 5,095,043 (실수 스칼라). 저장소 아티팩트는 전부 전자.
2. "effective" 두 값 공존: 4,304,513 (cross_attn+revin 제외, 모든 학습 provenance) vs 4,304,515 (cross_attn만, 스모크 스크립트).
3. dead `cross_attn`/`revin`이 학습 중 실제로 바뀌지 않는지는 PyTorch AdamW 의미론(`grad is None` → 건너뜀)과 스모크 관측에서 **추론**한 것; 학습 전후 값을 비교하는 테스트는 없다.
4. 속도 평가 1회의 FLOPs를 신뢰성 있게 측정한 값이 저장소에 없다.
5. 학습 중 어떤 S5 상태의 Re(Λ)가 0 이상이 되는지 (클리핑 없음) 확인한 적 없다.
6. S5 내부는 모듈 수준까지만 정리 (`ssm_ppg`/`ssm_target` 각 131,712); 그 아래 분해는 별도 작업.
7. upstream 기본 윈도우는 4 s (T=512)인데 본 프로그램은 전부 8 s (T=1024) — 모델은 길이 불변이나 지연·메모리 수치는 T에 따라 다르다 (스모크 수치는 T=512, A0 수치는 T=1024).

**학습**
8. A 계열·C1의 옵티마이저 스텝 수는 **유도값**이다 (§6.1). C2 prereg의 14,409를 재현하지만 MIMIC-BP 라운드 구조(1,547 배치/epoch → 8라운드마다 7스텝)는 런 로그로 대조하지 못했다.
9. 스텝당 JVP 벽시계 ("≈ 2 × 250 ms")는 산문에만 있다. `training_log.csv`는 검증·진단이 포함된 라운드 시간만 기록하므로 순수 forward+JVP+backward 시간을 분리할 수 없다.
10. 샘플당 JVP 메모리 수치(0.51 GiB 등)는 A2 prereg §8·보고서 산문이며 이를 뒷받침하는 측정 아티팩트(JSON/CSV)를 찾지 못했다. 마이크로배치 32의 기록된 최대치 16,903.9 MiB (≈16.5 GiB)는 산문의 16.3 GiB와 정합한다.
11. 여러 런이 dirty 워킹 트리에서 시작됐다 (a0b 1, a3_imeanflow 1, a6a 10, a7_otcfm 5, a8_imeanflow 1, a8_mse 2, a9_* 2, b1v2 1, gradcheck 12–22). dirty 파일 목록이 남아 있지 않아 그 런의 정확한 코드 상태는 저장소만으로 복원 불가.
12. B1은 `--resume`으로 시작됐다 (`args.resume = true`). 라운드 0부터 시작하는 300행 로그는 새 학습을 뜻하지만, 실행 시점에 `checkpoint_last.pt`가 없었다는 명시적 기록은 없다.
13. C1의 실현 옵티마이저 스텝 수가 자체 로그에 없다. C2 prereg는 66라운드 = 14,409스텝(66×220 = 14,520이 아님)이라 적었으나 C1 로그만으로는 검증 불가.
14. A/C1/R1의 최대 VRAM은 `max_memory_allocated`(allocated) 기준; R2/R3만 allocated·reserved 둘 다 기록.
15. TF32 상태는 preflight 프로세스만 기록 (기본값). R1/R2/R3는 아예 기록하지 않는다. 코드가 `allow_tf32`를 설정하지 않으므로 기본값으로 **추정**할 뿐 학습 프로세스 안에서 로깅되지 않았다.
16. 첫 A2 중단 런은 `--cond-mode` 플래그 이전이라 조건화가 `docs/EXPERIMENT_LOG.md` 산문 기준이다. `gradcheck_a6_x{0.1,1.0}`도 `--cond-scale` 이전 (미스케일 cond = 1.0으로 해석).
17. `tests/test_imeanflow.py`의 JAX 패리티 테스트는 jax 미설치 시 skip된다; 이 환경에서 실제로 실행됐는지 확인하지 않았다.
18. B1-v2는 미완 (6런 중 2런, S1 vanilla는 37,714/65,400 스텝에서 중단, WildPPG 쌍 미실행) → 탐색적 수치이며 확증 판정 없음.

**데이터**
19. X4-0 코호트 해시 키의 prereg vs 코드 불일치 (§8).
20. P1 5-fold 매니페스트는 `docs/DATA_PROTOCOL.md`가 "주 주장의 프로토콜"이라 적었지만 어떤 런도 참조하지 않는다 (실제 사용은 P0, A3, A4, A7뿐).
21. PENGUIN 논문 Table 1이 4 s를 썼는지 8 s를 썼는지는 저장소·논문에서 판정 불가.
22. WildPPG 라이선스가 출처마다 다르다 (LICENSE.md "review purposes only", 데이터시트 CC BY-SA, HF 카드 "mit") — 본 프로젝트는 CC BY-NC-SA 4.0으로 취급.
23. WildPPG ECG의 물리 단위는 미확정 (int32, "18-bit 부호 ADC"는 INFERRED, mV 보정 없음) → A9의 μ/σ는 미지정 native 단위.
24. WildPPG 검증 부분집합 3,785는 `n_val_windows` + stride 규칙에서 유도한 값이다 (문서는 "≈49 k → ≤4,096"만 적음). `wildppg_8s/MANIFEST.json`의 sha256은 어떤 provenance 파일에도 기록돼 있지 않다.
25. MIMIC-BP의 1,524개 ID 전체 목록은 매니페스트에만 있다. "ABP 스케일이 prior의 81.6×"는 산문(A9 prereg가 A8 인용) 재계산 아님.
26. `dalia.py`의 `windows_for_subject` 기본값은 `align='truncate'`인데 빌드는 `align='strict'` — 15명 모두 개수가 일치하므로 영향 없음.

**평가**
27. A 계열 테스트 노이즈 뱅크의 sha256이 어디에도 기록돼 있지 않다 (`noise_seed: 0`만). 해시가 있는 건 2,048·8,192 개발 뱅크뿐.
28. 3,907 WildPPG 테스트 부분집합의 GT 박동 수 40,523은 `artifacts/x0_error_decomposition/event_timing.csv`에서 확인되지만, kjd/ssx를 다시 로드하지 않고는 재계산할 수 없다.
29. S1–S8 라벨은 R2/R3 prereg·보고서에만 있고 코드는 `STRUCT5`와 C0/M1 이름을 쓴다.
30. `beats_ratio` 정의가 두 가지 공존 (§7.4 vs §7.5).
31. chance floor RNG가 64윈도우 청크마다 만들어지므로 한 윈도우의 20회 추출이 청크 내 위치에 의존한다 (결정적이지만 윈도우별 독립 아님).
32. MIMIC-BP (A7/A8) 평가 모집단 크기는 이번에 검증하지 못했다 (`outputs/a7_*`에 `metrics.json` 없음; `predict_a7.py` 기본 `--subsample 4096`).
33. C2 계층 부트스트랩(5,000회, 20260902)은 구현만 되고 실행된 적 없다.
34. R3 `gate_diagnostics.csv`의 (D) 부위별 `subject_boot_*` CI 컬럼은 인덱싱 버그로 무효 (모든 부위 0.7767 동일). 부위별 mean/p10/p90만 인용할 것. 코드는 고쳤으나 아티팩트는 재생성하지 않았다.
35. R2/R3 평가기는 cuDNN 기본값(비결정)으로 실행됐다 (학습은 결정적). 패리티는 통과했으나 평가 수치의 비트 재현은 보장되지 않는다.
36. R1 `global_site` FiLM 변형 (329,409 params)은 코드에만 있고 학습·평가된 적 없다.
37. R1 임계값 0.35가 Global과 Local에서 같은 것은 그리드 탐색의 우연이며 공유 상수가 아니다. R2/R3가 쓰는 스캐폴드는 **임계 이전 sigmoid 장**이고, 0.35는 스캐폴드 품질·게이트 진단에서만 쓰인다.
38. 수용영역 2,041 / 65 샘플은 이론값 (같은 패딩)이며 실측하지 않았다.
39. DaLiA 박동 수준 지표가 PPG/ECG 비동기 때문에 무효라는 판단은 `scripts/diagnose_dalia_sync.py` 측정에 근거하며 이번 정리에서 재검증하지 않았다.
40. R1은 최대 250 ms 허용오차와 150 ms RR 허용오차를 쓰는데 이는 생성기 평가의 50 ms 프로젝트 허용오차와 다르다 (각자 자기 프로토콜에 동결).

---

## 10. 자산 색인

**코드**
- 데이터: `src/ppg2ecg/data/{preprocess,dalia,wildppg,wildppg_sites,mimicbp,splits,target_norm,leakage}.py`
- 흐름/목적: `src/ppg2ecg/flow/{cfm,imeanflow,imeanflow_curriculum,interval_exposure,samplers,rhythm_transfer,rhythm_fusion}.py`
- 모델: `src/ppg2ecg/models/{__init__,regressor}.py`; 프로브 `src/ppg2ecg/probes/{rhythm_tcn,r1_cohort}.py`
- 학습: `src/ppg2ecg/training/{train_a0,train_a2,train_a5,train_b1_fixed_compute,train_r2_adapter,train_r3_fusion,valbank}.py`
- 평가: `src/ppg2ecg/evaluation/{rpeaks,metrics,s1_audit,m1_structural,event_reliability,paired_stats,alignment_diagnostics,efficiency,hierarchical,stamping,v1_timing,c2_cohort,abp_metrics,coupling_geometry}.py`
- 유틸: `src/ppg2ecg/utils/{seed,upstream}.py`
- 업스트림 (읽기 전용, 수정 금지): `external/PENGUIN` @`6cd70cd`, `external/iMeanFlow` @`bf60cd7`

**데이터·매니페스트**: `data/raw/{PPG-DaLiA,WildPPG,MIMIC-BP}` (+ `CHECKSUMS.sha256`, `INVENTORY.json`), `data/processed/{v0_8s,wildppg_8s,wildppg_8s_prenorm,mimicbp_8s,upstream,upstream_8s}` (+ 각 `MANIFEST.json`), `data/manifests/split_*.json`, `processed_parity_upstream{,_8s}.json`, `dalia_raw_inventory.json`

**실행 산출물**: `outputs/<run>/{config.yaml,provenance.json,train_meta.json,training_summary.json,training_log.csv,checkpoint_best.pt,checkpoint_last.pt,metrics.json,nfe_curve.csv,predictions/}`; 분석 아티팩트 `artifacts/{x0_error_decomposition,x4_0_event_reliability,s1_metric_validity,c0_imf_compression_target,c1_interval_exposure,m1_c1_structural_audit,v1_stepwise_visualization,r1_global_rhythm,r2_rhythm_transfer,r3_rhythm_fusion,a6_capacity_control,a8_abp_scale_control,a9_ecg_representation_control,b1_gap_curriculum,c2_compute_matched_multiseed}`

**문서**: 데이터 `docs/{DATA_PROTOCOL,WILDPPG_AUDIT,A7_ABP_DATASET_AUDIT}.md`; 감사 `docs/{PENGUIN_AUDIT,IMEANFLOW_AUDIT,R3_TARGET_STREAM_HOOK_AUDIT,B1_GAP_CURRICULUM_SOURCE_AUDIT}.md`; 지표 `docs/{METRIC_SEMANTICS,S1_METRIC_VALIDITY_REPORT,PREREGISTRATION_V0}.md`; 단계별 preregistration/report (`A2, A5, A6, A7, A8, A9, B1, C1, C2, X4_0, R1, R2, R3`); 로그 `docs/EXPERIMENT_LOG.md`; 환경 `docs/ENVIRONMENT.md`; 요약 `docs/PROJECT_STATUS_SUMMARY_FOR_LLM.md`; 연구 판단 `docs/ASSESSMENT_TOPTIER_AND_IDEATION_2026-09-03.md`

**동결 체크포인트**
- 생성기: `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` (라운드 46; 파일 sha256 `557c7054…`, state sha256 `47d7ccb9…`; A4 iMF와 동일, md5 `31c042d291052fbb6dc15263ad316be2`)
- 스캐폴드: `outputs/r1_global_tcn_seed42/checkpoint_best.pt` (파일 sha256 `bfe76ea6…`, state sha256 `0986a7af…`)
- R2 어댑터: `outputs/r2_{true,shuffle,oracle}_adapter_seed42/adapter_step2200.pt` (TRUE state sha `f98057ca…`, ORACLE `c8827b1b…`)
- R3 모듈: `outputs/r3_{tf_true,tf_shuffle,gtf_true,gtf_shuffle,gtf_const,gtf_oracle}_seed42/module_step2200.pt`

*체크포인트·예측·원시 데이터·아티팩트는 git에 들어가지 않는다.*
