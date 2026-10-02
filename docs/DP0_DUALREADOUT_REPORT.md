# DP0 — DualReadout-ECG — REPORT

> **DP0 CONFIRMED** on the locked AF-LOCK population (337 patients, 22,133 windows) for the development winner
> **S1 MIDDLE**: P1, P2, G1, PPG CONDITION-USE and E1 all pass with the preregistered margins.
> - S1 uses one conditioning encoder whose stem and first six blocks are shared between a deterministic point readout
>   and a ScaleFlow-style flow-matching readout. It has 943,372 waveform parameters against 1,191,910 for the two
>   separate specialists, i.e. **20.85 % fewer**.
> - **Not opened:** the old V1 TEST stayed closed. Only a DP1 draft was written.
> - **Architecture-paper development: GO.** This is conditional on the separate literature review, multiple training
>   seeds and the DP1 final test.
> - **Boundaries:**
>   - the point readout's event safety sits close to its margin on both populations (FP/window +0.037 DEV, +0.042 LOCK;
>     upper bounds 0.0498 and 0.0484 against +0.05);
>   - full sharing (S0) did not qualify on DP-DEV;
>   - every model has a single training seed.

| commit | content |
|---|---|
| `b4247f7` | preregistration (frozen) + split + implementation, before any DP0 training or DP-DEV outcome |
| `60240bb` | DP-DEV results + candidate selection (winner S1), before any AF-LOCK access |
| `fc39649` | AF-LOCK freeze manifest (S1, checkpoints and code hashed), before AF-LOCK was loaded |
| this commit | AF-LOCK results, report, method draft, DP1 draft, main figure |

## 한국어 요약

- **판정: DP0 CONFIRMED.**
  - DP-DEV에서 유일하게 자격을 얻은 S1 MIDDLE을 동결한 뒤, AF-LOCK을 한 번 열어 다섯 관문을 모두 통과했다.
- **S1의 구조:**
  - stem과 블록 1–6을 공유하는 조건 인코더 하나에 판독기가 두 개 달려 있다.
  - 블록 7–8과 디코더는 과제별로 따로 둔다.
  - 판독기는 결정론적 점 추정과 ScaleFlow식 흐름 매칭 생성이다.
  - 두 파형 출력은 더하지 않는다.
- **파라미터:** 두 전문가를 따로 쓰면 1,191,910개, S1은 943,372개로 20.85 % 적다.
- **AF-LOCK 결과 (S1 − 전문가):**

  | 관문 | 효과 [95 % CI] | 여백 |
  |---|---|---|
  | P1 박동 상관 | −0.0048 [−0.0063, −0.0034] | −0.02 |
  | P2 FP/창 | +0.042 [+0.036, +0.048] | +0.05 |
  | P2 recall | +0.0012 [+0.0007, +0.0017] | −0.01 |
  | G1 FD (11.43 vs 12.05) | −0.61 [−0.81, −0.42] | +1.0 |
  | PPG 셔플 FD | +11.5 [+4.0, +18.6] | > 0 |
  | PPG 셔플 상관 | −0.325 [−0.335, −0.315] | < 0 |

- **공유 깊이 (DP-DEV):**
  - S0 FULL(전부 공유)은 탈락했다. 점 판독의 FP가 +0.144로 여백을 넘었고, 생성 FD 상한도 +1.07로 여백 +1.0을 넘었다.
  - S2 STEM-ONLY는 성능 관문을 모두 통과했다. 하지만 경로 용량을 맞춘 설계라 절감이 0.17 %여서 구조상 E1을 통과할
    수 없다. 사전등록에 미리 적어 둔 사항이다.
  - 따라서 "전부 공유하면 점 판독의 이벤트 충실도에 측정 가능한 간섭이 생기고, 일부만 공유하면 여백 안에 머문다"고
    읽을 수 있다.
- **그래디언트 진단:** 공유 그래디언트 코사인은 S0와 S1이 비슷하다. 학습 끝에서 중앙값 +0.198 vs +0.210,
  음수 비율 4.9 % vs 2.9 %이다. 그래서 이 진단은 S0의 실패를 설명하지 못한다.
- **조심할 점:**
  - P2가 두 집단 모두에서 여백에 가깝다.
  - 점 판독의 HR-MAE가 전문가보다 높다(관문 아님): LOCK 6.22 vs 5.59 bpm.
  - 학습 seed가 하나뿐이다.
  - AF-LOCK 환자도 이전 프로젝트 모델(C0 / C0-A / R1 / SF0)의 학습 환자였다. 외부 집단이 아니다.
  - 생성 FD가 G보다 낮은 것은 부차 관찰이다. S0와 S2에서는 나타나지 않았다.
- **계산량:**
  - 두 출력을 모두 낼 때 GPU batch-1 지연은 13.3 → 11.1 ms, CPU는 29.3 → 23.9 ms, FLOPs는 1.85 → 1.16 G로 줄었다.
  - 주된 이유는 생성 경로가 조건을 NFE마다가 아니라 한 번만 인코딩하고 흐름 디코더가 가늘다는 점이다.
  - 두 판독기 사이의 trunk 캐시 효과 자체는 작다: GPU 0.37 ms, CPU 1.4 ms.
- **old TEST는 열지 않았다.** DP1 초안만 작성했다.

## 1. Status

- **Verdict: CONFIRMED** (`artifacts/dp0_dualreadout/lock_gates.json`). Every mandatory AF-LOCK gate passed for the
  frozen winner S1. PARTIAL is not used, and no rescue or re-run happened after the lock.
- **DP-DEV development:**
  - exactly the three preregistered candidates were trained once each (seed 42);
  - S1 was the only qualifying candidate, so it was the winner without needing the lexicographic rule;
  - no architecture, margin, NFE or loss change was made at any point;
  - the code is byte-identical to the preregistration (`post_prereg_code_changes: []`).
- **AF-LOCK** was loaded once, after the committed freeze `fc39649`. P, G and S1 were evaluated, and no unselected
  candidate was.
- **Old V1 TEST:** never opened (the DP0 script has no TEST loader). `docs/DP1_FINAL_TEST_PREREGISTRATION_DRAFT.md`
  is a draft only.
- **Method draft:** `docs/DUALREADOUT_ECG_METHOD_DRAFT.md`. Main figure: `artifacts/dp0_dualreadout/figure_main.png`.

## 2. Motivation from SF0 / AF0

- **SF0:** multiresolution flow matching (ScaleFlow) gave a large FD gain, an architecture-specific benefit from cross-scale
  coupling and real PPG dependence. Its single samples had poorer paired morphology and more false R detections than
  deterministic whole-window regression (WW-L1).
- **AF0:** x = deterministic anchor + stochastic residual modelled the residual distribution well. But the conditional
  sample centre drifted from the anchor in all 10 candidates (D2 0 / 10, D3 0 / 10).
- **DP0's reaction:** stochastic residuals are no longer added to the deterministic point estimate. The two needs are
  served by separate readouts of one shared condition representation.

## 3. Why two readouts

- **Different targets:**
  - paired estimation targets one representative ECG per condition, which L1 regression provides;
  - conditional generation targets the distribution p(x | c), which flow matching provides.
- **Different evaluation:** the point readout is judged against the point specialist (P1, P2), and the generative
  readout against the generative specialist (G1, condition use). A random flow sample is never compared with the point
  target as a pass / fail criterion.
- **No mixing:** the outputs are never added, and the flow samples are generated from noise directly (no μ + r).
- **What is shared:** only the PPG / event condition encoding. The noisy ECG x_t never enters the shared encoder.

## 4. Data and nested validation design

| role | patients | windows | status |
|---|---|---|---|
| DP-TRAIN | 2,100 | 139,962 | training (detector, P, G, S0–S2) |
| DP-DEV | 300 | 19,583 | development evaluation and selection |
| AF-LOCK | 337 | 22,133 | sealed for AF0 and DP0 development; opened once after the committed freeze |
| old V1 TEST | 1,156 | — | closed |

- **Split rule:** sorted former AF-TRAIN patients permuted with `default_rng(20261002)`; [0:300] → DP-DEV, the rest →
  DP-TRAIN.
- **Exclusions:** no AF-DEV, AF-LOCK, ARCH-VAL / HOLDOUT, SF-VAL, or old val / test patient (all checks pass).
- **Evidence status:**
  - DP-DEV is a new development population for DP0, but its patients trained earlier models.
  - AF-LOCK had never been opened by AF0 or by DP0 development, but its patients were also training patients of
    C0 / C0-A / R1 / SF0 models.
  - So AF-LOCK is an **internal locked confirmation set, not project-naive and not external**.

## 5. Timing condition

- **Detector:** RD1 / C0 RhythmTCN (328,897 parameters, not counted in any parameter ratio), trained on DP-TRAIN only,
  seed 42, 54.3 s, then frozen.
- **Events:** threshold 0.35, refractory 32; one raster per window shared by every model and run.

| population | FP/window | recall | precision | F1 (patient macro) |
|---|---|---|---|---|
| DP-DEV | 0.438 [0.367, 0.514] | 0.763 [0.742, 0.783] | 0.884 | 0.815 [0.795, 0.835] |
| AF-LOCK | 0.450 [0.382, 0.521] | 0.760 [0.740, 0.781] | 0.879 | 0.811 [0.792, 0.831] |

## 6. Point specialist

**SPECIALIST P:** WW-L1 = C0-A `WWDet(72, 5)`, 593,577 parameters; L1, 20,000 × 64, AdamW 1e-3 / 0.01, clip 1.0,
seed 42; reference-R raster in training, detector raster at inference; 130.8 s.

| population | corr | FP/window | recall | precision | F1 | pooled F1 | RR-MAE (ms) | HR-MAE (bpm) | MAE | FD (descr.) |
|---|---|---|---|---|---|---|---|---|---|---|
| DP-DEV | 0.821 [0.803, 0.838] | 0.523 [0.451, 0.602] | 0.766 | 0.864 | 0.809 | 0.811 | 7.91 | 5.51 | 0.308 | 16.62 |
| AF-LOCK | 0.804 [0.786, 0.820] | 0.534 [0.464, 0.607] | 0.763 | 0.860 | 0.806 | 0.811 | 7.57 | 5.59 | 0.315 | 19.12 |

## 7. Generative specialist

**SPECIALIST G:** SF0 SCALEFLOW-COUPLED = `ScaleFM(50, coupled)`, 598,333 parameters; linear-path flow matching, same
protocol; Euler NFE 8; one preregistered sample per window; 133.2 s.

| population | FD | corr (descr.) | FP/window (descr.) | recall | F1 | diversity ratio* |
|---|---|---|---|---|---|---|
| DP-DEV | 10.50 | 0.764 | 0.690 | 0.769 | 0.797 | 0.505 |
| AF-LOCK | 12.05 | 0.746 | 0.709 | 0.767 | 0.794 | 0.498 |

\* RMS(sample − μ_P) / RMS(ECG − μ_P).

**Reproduction sanity:** the expected pattern held on both populations.
- P has better paired morphology than G's single samples (corr +0.057 DEV, +0.058 LOCK).
- P has fewer false R (FP −0.167 DEV, −0.175 LOCK).
- G has the lower FD (10.50 vs 16.62; 12.05 vs 19.12).

## 8. Shared physiological encoder

- **Audit (`audit.md`):** the only learned condition path in either specialist is the WW-L1 encoder. ScaleFlow mixes
  raw Haar(PPG) and Haar(raster) with Haar(x_t) from its first layer.
- **Shared encoder:**
  - that family with the raster moved into its input: stem 1 × 1 (2 → 64), then eight C0 residual blocks;
  - multiscale taps H_256 and H_128 by the fixed Haar low-pass;
  - x_t never enters it (this is verified by a forward-hook test).
- **Natural stages:** E0 stem | E1 blocks 1–6 (the first dilation cycle; receptive field 505 samples) | E2 blocks 7–8.
- **Readout paths are matched to the specialists by parameter count:**
  - point path 590,503 (−0.52 %), with a point decoder of width 71;
  - flow path 599,589 (+0.21 %), with a flow decoder of width 30.

## 9. Full sharing (S0)

- **Design:** E0 + E1 + E2 are shared (328,896 parameters); the task adapters sit directly before the two decoders.
- **Size:** total 861,196 parameters (saving 27.75 %).
- **DP-DEV:** not qualified (§13–§17).
  - The point readout added false R detections: FP +0.144 [+0.128, +0.165]. Precision fell from 0.864 to 0.836, and
    HR-MAE rose from 5.51 to 7.06 bpm.
  - The generative FD upper bound slightly exceeded the margin: +0.65 [+0.24, +1.07].

## 10. Partial sharing (S1 MIDDLE, S2 STEM-ONLY)

- **S1:**
  - E0 + E1 shared (246,720 parameters); blocks 7–8 duplicated per task after the adapters;
  - total 943,372 (saving 20.85 %).
- **S2:**
  - only the 1 × 1 stem shared (192 parameters); all eight blocks duplicated per task;
  - total 1,189,900 (saving 0.17 %).
  - It fails E1 by construction (preregistered). It serves as the minimal-sharing control: the same paths and the same
    round-robin training, without meaningful sharing.

## 11. Round-robin multitask training

- **Optimizer:** one AdamW (1e-3, wd 0.01, constant LR).
- **Schedule:** 20,000 cycles. Odd cycles run POINT → FLOW and even cycles FLOW → POINT. This gives exactly 20,000 point
  and 20,000 flow updates per dual model, as recorded in every checkpoint.
- **Masking:** each substep touches only the shared parameters and that task's private parameters. The other task's
  private parameters had grad None, which was checked, and were skipped by AdamW.
- **Data order:** the POINT stream reproduced P's batch order; the FLOW stream reproduced G's batches and FM noise / t.
- **Not used:** loss weighting, GradNorm, PCGrad or early stopping.
- **Training:** no NaN step; about 304 s and 1,336–1,340 MiB peak GPU memory per dual model.
- **Final-2,000-cycle training losses:**

  | model | L1 | FM |
  |---|---|---|
  | S0 | 0.234 | 0.129 |
  | S1 | 0.233 | 0.129 |
  | S2 | 0.236 | 0.129 |
  | specialists P / G (final 2,000 steps) | 0.251 | 0.129 |

## 12. Parameter / compute accounting

**Parameters (detector excluded):**

| model | shared | point-private | flow-private | total | saving vs P + G |
|---|---|---|---|---|---|
| P + G separate | — | 593,577 | 598,333 | 1,191,910 | — |
| S0 FULL | 328,896 | 261,607 | 270,693 | 861,196 | 27.75 % |
| S1 MIDDLE | 246,720 | 343,783 | 352,869 | 943,372 | 20.85 % |
| S2 STEM-ONLY | 192 | 590,311 | 599,397 | 1,189,900 | 0.17 % |

- **Adapters:** 8,320 parameters (0.88 % of S1).
- **E1 threshold:** 1,013,123.5.

**Compute (one window; waveform models without the detector unless noted):**

| | P + G separate | S1 |
|---|---|---|
| FLOPs, point only | 606 M | 603 M |
| FLOPs, generation NFE 8 | 1,246 M (156 M per NFE) | 808 M (trunk once + 58.5 M per NFE) |
| FLOPs, both outputs | 1,852 M | 1,159 M cached / 1,411 M uncached |
| GPU batch-1, point / generation / both (ms) | 0.74 / 12.59 / 13.34 | 0.77 / 10.75 / 11.13 (uncached 11.50) |
| GPU batch-1 incl. detector, both (ms) | 13.87 | 11.73 |
| GPU one vector-field evaluation (ms) | 1.62 | 1.31 (+ 0.49 to encode the condition once) |
| CPU 4-thread, point / generation / both (ms) | 3.39 / 25.51 / 29.30 | 3.47 / 22.09 / 23.94 (uncached 25.31) |
| CPU incl. detector, both (ms) | 31.44 | 26.20 |
| training time | 130.8 + 133.2 s | 304.1 s |
| training peak GPU memory | 1,302 / 1,017 MiB | 1,337 MiB |

- **Observed compute saving for both outputs:** 17 % GPU, 18 % CPU and 37 % FLOPs. It comes mainly from encoding the
  condition once per generation instead of at every NFE, and from the narrower flow decoder.
- **Trunk caching between the two readouts adds only a little:** 0.37 ms GPU, 1.4 ms CPU and 0.25 GFLOPs.
- **Batch-1 inference memory above the weights is about 1 MiB for every arm.** The weights are 4.5 MiB for P + G and
  3.6 MiB for S1.
- **Training one dual model takes about 15 % longer than training both specialists sequentially.**

## 13. DP-DEV point results (equal-patient-weight means, patient-bootstrap 95 % CI)

| model | path params | corr | FP/window | recall | F1 | pooled F1 | RR-MAE | HR-MAE | MAE | GPU ms |
|---|---|---|---|---|---|---|---|---|---|---|
| SPECIALIST P | 593,577 | 0.8213 | 0.523 | 0.766 | 0.809 | 0.811 | 7.91 | 5.51 | 0.308 | 0.74 |
| S0 point | 590,503 | 0.8194 | 0.667 | 0.771 | 0.800 | 0.800 | 7.98 | 7.06 | 0.308 | 0.78 |
| S1 point | 590,503 | 0.8177 | 0.560 | 0.767 | 0.808 | 0.809 | 7.87 | 6.06 | 0.310 | 0.77 |
| S2 point | 590,503 | 0.8181 | 0.554 | 0.766 | 0.807 | 0.808 | 7.88 | 6.55 | 0.310 | 0.77 |

| dual − P (DEV) | corr (P1) | FP (P2) | recall (P2) | F1 (descr.) |
|---|---|---|---|---|
| S0 | −0.0019 [−0.0032, −0.0007] | **+0.144 [+0.128, +0.165]** | +0.0050 [+0.0042, +0.0059] | −0.0090 |
| S1 | −0.0036 [−0.0053, −0.0017] | +0.037 [+0.027, +0.0498] | +0.0015 [+0.0008, +0.0023] | −0.0018 |
| S2 | −0.0032 [−0.0043, −0.0021] | +0.031 [+0.025, +0.038] | +0.0003 [−0.0002, +0.0008] | −0.0027 |

Every dual point head adds false R detections relative to P, with intervals that exclude 0. Only S0 exceeds the margin.

## 14. DP-DEV generative results (one sample per window, Euler 8, identical noise)

| model | total params | FD | dual − G FD (G1) | corr (descr.) | FP (descr.) | diversity ratio | spectral | GPU ms |
|---|---|---|---|---|---|---|---|---|
| SPECIALIST G | 598,333 | 10.50 | — | 0.764 | 0.690 | 0.505 | 0.243 | 12.59 |
| S0 gen | 861,196 | 11.15 | **+0.65 [+0.24, +1.07]** | 0.765 | 0.714 | 0.506 | 0.244 | 10.71 |
| S1 gen | 943,372 | 9.84 | −0.66 [−0.90, −0.41] | 0.769 | 0.694 | 0.506 | 0.224 | 10.75 |
| S2 gen | 1,189,900 | 10.65 | +0.15 [−0.28, +0.56] | 0.771 | 0.712 | 0.489 | 0.306 | 10.73 |

## 15. Share-depth comparison (DP-DEV only)

| | S2 STEM-ONLY | S1 MIDDLE | S0 FULL |
|---|---|---|---|
| shared parameters | 192 | 246,720 | 328,896 |
| saving | 0.17 % | 20.85 % | 27.75 % |
| point FP − P | +0.031 | +0.037 | +0.144 |
| point corr − P | −0.0032 | −0.0036 | −0.0019 |
| generative FD − G | +0.15 | −0.66 | +0.65 |
| gates P1 P2 G1 COND E1 | P P P P **F** | P P P P P | P **F F** P P |

- **Event fidelity:** sharing the first two natural stages (S1) left the point readout's events close to the
  minimal-sharing control S2. Sharing the last two blocks as well (S0) made the extra false detections about four times
  larger (+0.144 vs +0.037 / +0.031).
- **Morphology:** the correlation was essentially unaffected by depth; all three are within 0.004 of P.
- **Generative FD:** non-monotone in depth (S1 lowest). The S1 improvement over G is therefore not a general property
  of sharing.
- **Interpretation (spec §47, allowed form):** complete representation sharing produced a measurable interference with
  the point readout's event fidelity. Partial sharing kept reusable condition features without exceeding the
  preregistered margins. The mechanism is not established (§16).

## 16. Gradient interaction

Cosine between ∇_shared L_point and ∇_shared L_FM over 512 fixed DP-TRAIN minibatches:

| pattern (shared params) | init | 25 % | final | fraction < 0 (init / 25 % / final) |
|---|---|---|---|---|
| S0 (328,896) | +0.043 [IQR 0.039, 0.047] | +0.313 [0.203, 0.409] | +0.198 [0.115, 0.291] | 0.000 / 0.027 / 0.049 |
| S1 (246,720) | +0.213 [0.209, 0.217] | +0.286 [0.183, 0.391] | +0.210 [0.125, 0.298] | 0.000 / 0.027 / 0.029 |
| S2 (192) | −0.064 [−0.070, −0.057] | +0.146 [−0.091, +0.324] | +0.167 [−0.097, +0.381] | 1.000 / 0.340 / 0.332 |

- **Overall:** the two objectives' shared gradients are mostly aligned for every pattern; a negative cosine is rare
  for S0 and S1.
- **S0 vs S1:** the cosines are very similar. **This diagnostic does not explain why S0 failed and S1 passed**; there is
  no evidence here that deeper sharing created stronger gradient conflict.
- **S2:** its 192-parameter stem gives noisy cosines.
- **Feature diagnostics** (`feature_diagnostics.json`, descriptive):
  - the flow adapter output is much smaller in RMS than the point adapter output: S0 0.105 vs 0.573, S1 0.245 vs 0.633;
  - no dead channels in the S0 / S1 shared outputs.

## 17. Candidate qualification

| | P1 | P2 | G1 | CONDITION | E1 | qualified |
|---|---|---|---|---|---|---|
| S0 FULL | PASS | FAIL | FAIL | PASS | PASS | NO |
| S1 MIDDLE | PASS | PASS (FP upper 0.0498) | PASS | PASS | PASS | **YES** |
| S2 STEM-ONLY | PASS | PASS | PASS | PASS | FAIL (structural) | NO |

Condition use on DP-DEV (PPG shuffled − conditioned):

| model | FD | corr |
|---|---|---|
| G | +11.85 [+4.92, +18.72] | −0.299 [−0.311, −0.288] |
| S0 | +22.53 [+14.44, +30.60] | −0.355 [−0.366, −0.344] |
| S1 | +16.23 [+9.06, +23.52] | −0.343 [−0.354, −0.333] |
| S2 | +21.04 [+13.19, +28.98] | −0.367 [−0.378, −0.356] |

## 18. Winner selection

- S1 was the only qualifying candidate, so it was selected directly (`candidate_selection.json`).
- The frozen lexicographic rule (saving, FD, corr, latency) was not needed. S0 would have won it on saving, but it did
  not qualify.

## 19. Locked AF-LOCK evaluation

- **Freeze:** `lock_freeze_manifest.json` (commit `fc39649`) hashed:
  - the preregistration and the DP0 code;
  - the detector, P, G and S1 checkpoints;
  - the ownership graph, training manifest, parameter accounting and splits;
  - the FM / Euler / noise / metric / bootstrap / shuffle code.
- **Access check:** the loader verified that the committed manifest was unchanged before reading AF-LOCK.
- **Evaluation:** one evaluation of P, G and S1 (22,133 windows, 337 patients), with the same detector and the same
  gates, margins and bootstrap (2,000 patient replicates, seed 20261002). No other candidate was evaluated, nothing was
  retrained, and there was no return to DP-DEV.

## 20. Point-readout confirmation (AF-LOCK)

| model | corr | FP/window | recall | precision | F1 | pooled F1 | RR-MAE | HR-MAE | MAE | FD (descr.) |
|---|---|---|---|---|---|---|---|---|---|---|
| SPECIALIST P | 0.8040 [0.7864, 0.8204] | 0.534 [0.464, 0.607] | 0.763 | 0.860 | 0.806 | 0.811 | 7.57 | 5.59 | 0.315 | 19.12 |
| S1 point | 0.7991 [0.7817, 0.8155] | 0.576 [0.506, 0.649] | 0.764 | 0.851 | 0.803 | 0.808 | 7.60 | 6.22 | 0.316 | 15.42 |

- **LOCK P1:** corr −0.0048 [−0.0063, −0.0034], above −0.02 → **PASS**.
- **LOCK P2:** FP +0.042 [+0.036, +0.048], below +0.05; recall +0.0012 [+0.0007, +0.0017], above −0.01 → **PASS**.
- **The point readout is not identical to the specialist.** It has slightly lower morphology correlation and about
  0.04 more false R per window (8 % relative), both statistically clear but inside the preregistered margins. HR-MAE is
  higher (descriptive, not a gate).

## 21. Generative-readout confirmation (AF-LOCK)

| model | params | FD | corr (descr.) | FP (descr.) | recall | diversity ratio | spectral |
|---|---|---|---|---|---|---|---|
| SPECIALIST G | 598,333 | 12.05 | 0.746 | 0.709 | 0.767 | 0.498 | 0.236 |
| S1 gen | 943,372 (total) | 11.43 | 0.747 | 0.708 | 0.766 | 0.498 | 0.224 |

**LOCK G1:** FD −0.61 [−0.81, −0.42], below +1.0 → **PASS**. The lower FD replicates the DP-DEV direction for S1. It
is a secondary observation, not a claim of better samples than ScaleFlow (§15, §26).

## 22. PPG condition use (AF-LOCK)

| model | FD conditioned → shuffled | FD shuffled − conditioned | corr shuffled − conditioned | condition-use test |
|---|---|---|---|---|
| S1 | 11.43 → 22.97 | +11.53 [+3.96, +18.58] | −0.325 [−0.335, −0.315] | **PASS** |
| G (reference) | 12.05 → 20.00 | +7.96 [+0.60, +14.88] | −0.285 [−0.295, −0.276] | — |

The shared representation is recomputed from the shuffled PPG for S1; the raster, noise and model are kept.

## 23. Parameter efficiency

**LOCK E1:**
- S1 has 943,372 waveform parameters vs 1,191,910 for P + G, i.e. 79.15 % of the separate total and a saving of
  **20.85 %**, which meets the ≤ 85 % requirement → **PASS**.
- Of S1's parameters, 246,720 (26 %) are shared, 343,783 point-private and 352,869 flow-private.
- Each readout path has the same parameter count as the corresponding specialist within 0.6 %. The saving is therefore
  the shared encoder, not a reduction of per-task capacity.

## 24. K16 characterization (descriptive)

2,000 windows per population (salted rank `dp0-k16-v1`), K = 16 samples each.

| | G (DEV) | S1 (DEV) | G (LOCK) | S1 (LOCK) |
|---|---|---|---|---|
| within-condition waveform diversity (pairwise RMS) | 0.245 | 0.238 | 0.247 | 0.241 |
| ratio to √2 · RMS(ECG − μ_P) | 0.401 | 0.390 | 0.398 | 0.388 |
| beat-aligned diversity | 0.224 | 0.221 | 0.225 | 0.222 |
| generated / real within-window beat diversity | 0.548 | 0.569 | 0.548 | 0.569 |
| R-time seed SD (median) | 0 ms | 0 ms | 0 ms | 0 ms |
| HR MAE: K16 consensus vs single sample (bpm) | 6.49 vs 6.88 | 6.36 vs 6.81 | 6.86 vs 7.15 | 6.86 vs 7.25 |
| mean16 corr / FP | 0.818 / 0.516 | 0.818 / 0.509 | 0.806 / 0.558 | 0.805 / 0.560 |

- **Within-condition spread:** both generators produce about 0.4 × √2 of the real deviation around μ_P. They are
  under-dispersed relative to a calibrated conditional distribution, as SF0 already found for ScaleFlow.
- **Timing is not stochastic:** R times are identical across seeds.
- **The sample mean was not required to equal the point head.** It differs from μ_P by MAE 0.081 (LOCK).

## 25. Supported claims

All of the following are conditional on this population, a single training seed and the preregistered margins.

1. On the locked AF-LOCK population, one shared PPG / event conditioning encoder (stem + first dilation cycle) with two
   task-specific readouts was non-inferior to a separately trained deterministic WW-L1 specialist for paired ECG
   estimation:
   - morphology within −0.02;
   - FP/window within +0.05;
   - recall within −0.01.
2. On the same population, the generative readout was non-inferior to a separately trained SCALEFLOW-COUPLED specialist
   for distributional fidelity (FD within +1.0; observed −0.61), and it depended on the PPG condition.
3. The above held with 20.85 % fewer waveform-model parameters than the two specialists. Each readout path had the same
   parameter budget as its specialist.
4. **Development evidence (DP-DEV only):**
   - full sharing of the encoder (S0) exceeded the event-safety margin of the point readout (+0.144 FP/window) and
     narrowly the FD margin;
   - partial sharing (S1) and minimal sharing (S2) did not exceed them.
5. **Descriptive:** requesting both outputs from S1 was observed to cost 1.16 vs 1.85 GFLOPs and 11.1 vs 13.3 ms
   (GPU batch-1) relative to running P and G separately.

## 26. Unsupported claims

- that DualReadout produces better point ECGs than WW-L1 — it is slightly worse within margins, with higher HR-MAE;
- that DualReadout produces better samples than ScaleFlow in general — the FD gain is S1-specific and secondary;
- that point and generative objectives are universally incompatible;
- that gradient conflict explains the S0 failure — the cosines do not differ between S0 and S1;
- deep shared representation learning beyond the stem and first dilation cycle;
- calibrated uncertainty, or patient-specific morphology uncertainty (the samples are under-dispersed and have no
  timing variability);
- clinical validity, external generalization, or final TEST performance;
- state of the art, a first dual-readout physiological model, or a first reconstruction / generation multitask
  architecture (a separate literature review is required);
- robustness across training seeds (only seed 42 was trained).

## 27. DP1 final-test plan

`docs/DP1_FINAL_TEST_PREREGISTRATION_DRAFT.md` (draft only; not frozen, not executed):
- freeze the S1 architecture, the training protocol and the gates;
- train several seeds;
- run a metadata-only freshness audit of the old V1 TEST;
- open TEST exactly once after a committed freeze;
- report P, G and S1 with the same P1 / P2 / G1 / condition / E1 definitions;
- report seed variability, especially for P2, which sits near its margin.

## Files

- **Preregistration:** `docs/DP0_DUALREADOUT_PREREGISTRATION.md`.
- **Code:** `scripts/dp0_dualreadout.py`, `src/ppg2ecg/dualreadout/{__init__,model}.py`, `tests/test_dp0_dualreadout.py`,
  `scripts/dp0_main_figure.py`.
- **Artifacts:** `artifacts/dp0_dualreadout/`
  - design: audit, split, configs, sharing graph, adapters, training manifest, parameter accounting, prereg manifest;
  - DP-DEV: point / gen metrics, bootstrap, gates, gradient conflict, condition shuffle, feature diagnostics,
    candidate selection, compute, checkpoint hashes, tables, `figure_dev.png`;
  - lock: freeze manifest, lock metrics / bootstrap / gates, condition shuffle, K16, `table_lock.csv`,
    `figure_lock.png`;
  - `figure_main.png`.
- **Checkpoints and outputs (not committed):** `outputs/dp0_dualreadout/`.
