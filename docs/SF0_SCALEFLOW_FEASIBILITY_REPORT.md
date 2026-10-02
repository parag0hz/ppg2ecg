# SF0 — ScaleFlow-ECG Feasibility — REPORT

> **FINAL SF0 VERDICT: FAILED (not qualified on SF-VAL). The old V1 TEST was NEVER OPENED.**
> SCALEFLOW-COUPLED passed G2, G3, G6 and G7 and failed G4 (morphology) and G5 (event safety). Under the frozen rule,
> SF0 is not qualified:
> - no freshness audit, final freeze or TEST evaluation;
> - no rescue;
> - architecture-paper **NO-GO**.
>
> Failure categories: **morphology degradation** and **event degradation**. All figures and numbers are SF-VAL.

| commit | content |
|---|---|
| `dbf60d7` | split + preregistration + implementation (before any SF0 training or SF-VAL outcome) |
| this commit | SF-VAL results, secondary analyses, report |

Labels:

- **[gate]** preregistered gate;
- **[descriptive]** preregistered, not gated;
- **[post-hoc]** reading added after the results.

## 한국어 요약

- **판정: FAILED.** SF-VAL 자격 미달이라 TEST는 열지 않았다.
  - 통과: G2, G3, G6, G7.
  - 실패: G4 형태, G5 사건 안전.
- **분포 품질(FD)은 크게 좋아졌다.**

  | 모델 | FD |
  |---|---|
  | WW-L1 | 15.05 |
  | WW-FM | 13.43 |
  | 독립 다중해상도 | 3.42 |
  | ScaleFlow | 2.71 |

  - ScaleFlow − WW-FM: −10.72 [−11.28, −10.10].
  - 결합(coupled) − 독립: −0.71 [−1.08, −0.36].
- **PPG를 실제로 쓴다.** PPG를 섞으면 FD가 +30.0, 박동 상관이 −0.30 변한다.
- **샘플 하나만 보면 결정적 WW-L1보다 나쁘다.** 그래서 자격을 얻지 못했다.
  - 박동 정렬 상관: 0.740 vs 0.820, 차이 −0.080 [−0.084, −0.076]. 여백은 −0.02.
  - 가짜 R: 창당 0.624 vs 0.531, 차이 +0.092 [+0.084, +0.101]. 여백은 +0.05.
- **[사후 해석]** 생성 모델의 샘플 하나는 조건부 평균을 내는 회귀 모델보다 정답과의 상관이 낮고, 여분의 R 모양 구조를
  더 만든다. NFE를 늘리면 FD는 좋아지지만(4.68 → 2.71 → 2.27) 상관과 FP는 나빠진다.
- **보조 분석:**
  - R 시점의 시드 간 표준편차는 0 ms다. 리듬은 이벤트 래스터가 정한다.
  - 16개 샘플 중앙값으로 계산한 심박수도 샘플 하나보다 낫지 않다.
- **결론:** 규칙상 SF0는 닫힌다. 다른 웨이블릿, 백본, NFE, 손실로 구제하지 않는다.

## 1. Status

- **Process:** SF0 ran as preregistered (`dbf60d7`): split → detector → four models → SF-VAL gates → secondary analyses.
  - Every model was trained once (seed 42, last checkpoint, no NaN step).
  - Not qualified, so `final_test_freshness_audit.md` is NOT PERFORMED, no freeze exists, and **TEST was never
    opened**.
- **Deviations:** none.

## 2. Why SF0 is a new architecture line

- **The closed line** (BF0 → R1) tested deterministic and beat-wise formulations. It ended with an unresolved
  event-fidelity / realism trade-off.
- **SF0 changes the formulation:**
  - whole-window conditional generation by flow matching;
  - a fixed two-level Haar representation;
  - hierarchical coarse → mid → fine coupling.
- **The question:** does that beat deterministic regression, vanilla whole-window FM, and an uncoupled multiresolution
  FM at matched capacity?
- **No novelty claim.**

## 3. Evidence-status limitations

- **SF-VAL is not project-naive.** Its 433 patients come from the former ARCH-TRAIN and were training data for earlier
  detectors and waveform models. They are new to this line only.
- **Not used:** ARCH-VAL and ARCH-HOLDOUT.
- **Old V1 TEST:** never loaded.
- **Single run:** one seed per model and one preregistered sample per window.

## 4. New SF-TRAIN / SF-VAL split

| role | patients | windows |
|---|---|---|
| SF-TRAIN | 3,037 | 201,997 |
| SF-VAL | 433 | 29,223 |

- **Rule:** the former ARCH-TRAIN, permuted with `default_rng(20261002)`.
- **Checks:**
  - disjoint;
  - the union is ARCH-TRAIN;
  - no ARCH-VAL or HOLDOUT patient;
  - no old V1 validation or test patient;
  - exact counts.
- **Records:** hashes in `split_hashes.json`.

## 5. Timing conditioning

- **Detector:** RD1 / C0 RhythmTCN retrained on SF-TRAIN only: 14,000 × 64, BCE to a Gaussian (σ = 20 ms) target,
  seed 42, 55 s, 328,897 parameters.
- **Events:** threshold 0.35, refractory 32.
- **SF-VAL event quality (placed events):**

  | measure | value |
  |---|---|
  | patient precision | 0.889 |
  | patient recall | 0.767 |
  | patient F1 | 0.820 |
  | FP / window | 0.404 |
  | pooled F1 | 0.826 |
  | RR-MAE | 7.64 ms |
  | HR-MAE | 4.93 bpm |

- **Shared raster:** all four models received the same Gaussian raster of these events (2,236 windows have none).
  Training used the reference-R raster.

## 6. Flow Matching formulation

- **Path and loss:** x_t = (1 − t) x₀ + t x₁; target x₁ − x₀; MSE in waveform space (after the inverse Haar for C / D).
  No auxiliary loss.
- **Sampling:** Euler, NFE = 8, with one fixed noise vector per window (sha256 of patient:window:20261002), identical
  across the three FM arms.
- **Primary metrics** use that single sample.

## 7. WW-L1

C0-A's WWDet (C0 encoder; decoder 72 × 5), 593,577 parameters, trained with L1 from scratch on SF-TRAIN (130.9 s).

## 8. WW-FM

The WW family plus x_t (decoder input) and a time embedding, added to the 5 decoder blocks. Decoder width 63,
597,664 parameters, 147.0 s.

## 9. Independent multiresolution FM

- **Structure:** Haar coefficients of x_t, PPG and raster feed three independent branches: coarse 128, mid 128,
  fine 256. Each branch has 6 time-conditioned residual blocks, width 50.
- **Output:** the inverse Haar of the three velocities.
- **Size:** 593,485 parameters, 127.9 s.
- **Unit-tested:** no hidden feature crosses scales.

## 10. ScaleFlow coupled architecture

- **Coupling:** as §9 plus 16-channel 1 × 1 projections of the final branch features: coarse → mid; mid and coarse →
  fine (×2 nearest).
- **No fine → coarse path** (unit-tested by coefficient perturbation).
- **Size:** 598,333 parameters, 131.7 s.

## 11. Parameter / compute matching

**Parameters.** Matched by count only; all within ±1% of SCALEFLOW (`parameter_match.json`).

**Compute is not matched:**

| | params (pipeline incl. detector) | FLOPs per VF eval | pipeline FLOPs | GPU batch-1 | CPU 4-thread |
|---|---|---|---|---|---|
| WW-L1 | 922,474 | 606 M | 0.94 G (1 pass) | 1.28 ms | 5.4 ms |
| WW-FM | 926,561 | 543 M | 4.68 G (NFE 8) | 7.53 ms | 30.2 ms |
| INDEPENDENT | 922,382 | 154 M | 1.57 G | 12.66 ms | 28.8 ms |
| SCALEFLOW | 927,230 | 156 M | 1.58 G | 13.18 ms | 29.9 ms |

The scale models need about 3.5× fewer FLOPs than WW-FM, because each branch runs on 128–256 samples. Their batch-1 GPU
latency is higher (three sequential branches).

## 12. SF-VAL results (433 patients, 29,223 windows; one sample per window; 91,471 matched beat pairs)

| SF-VAL | FD | beat corr | FP / win | recall | F1 | pooled F1 | RR-MAE | HR-MAE | MAE | PCC |
|---|---|---|---|---|---|---|---|---|---|---|
| placed events | — | — | 0.404 | 0.767 | 0.820 | 0.826 | 7.64 | 4.93 | — | — |
| WW-L1 | 15.05 | **0.820** | **0.531** | 0.771 | **0.811** | 0.816 | 7.74 | 6.57 | **0.298** | **0.371** |
| WW-FM | 13.43 | 0.746 | 0.654 | **0.773** | 0.802 | 0.806 | 8.00 | 7.11 | 0.349 | 0.297 |
| SCALE-FM-INDEPENDENT | 3.42 | 0.738 | 0.674 | 0.772 | 0.799 | 0.803 | 7.98 | 7.32 | 0.349 | 0.335 |
| SCALEFLOW-COUPLED | **2.71** | 0.740 | 0.624 | 0.770 | 0.802 | 0.806 | 7.92 | 7.05 | 0.349 | 0.333 |

- **Definitions:** FP / recall / F1 are patient-macro (pooled within patient). RR-MAE is in ms and HR-MAE in bpm.
- **Raw false R:** WW-L1 15,511, WW-FM 19,115, IND 19,650, SF 18,220, placed 11,725.

## 13. G1 Flow objective effect (context)

| WW-FM − WW-L1 | estimate [95% CI] |
|---|---|
| FD | −1.62 [−3.38, +0.21] |
| beat corr | −0.074 [−0.077, −0.071] |
| FP / window | +0.122 [+0.112, +0.134] |
| recall | +0.0025 [+0.0017, +0.0033] |
| MAE | +0.051 [+0.047, +0.055] |

**G1: MIXED.**

- The FM objective alone did not significantly lower FD.
- It lowered morphology correlation and added false detections.

## 14. G2 / G3 architecture distributional effect [gate]

- **G2:** FD(SF) − FD(WW-FM) = **−10.72 [−11.28, −10.10] — PASS.**
- **G3:** the best FD baseline is WW-FM (13.43 < 15.05), so G3 = G2 — **PASS.**
- **[descriptive] Where the gain comes from:** most of it comes from the multiresolution representation itself.
  - INDEPENDENT already reaches FD 3.42.
  - Coupling adds the rest (§17).

## 15. G4 morphology [gate]

- **Rule:** the best-correlation baseline is WW-L1 (0.820). The margin is −0.02.
- **Result:** corr(SF) − corr(WW-L1) = **−0.080 [−0.084, −0.076] — FAIL.**
- **[descriptive]** corr(SF) − corr(WW-FM) = −0.0055 [−0.0079, −0.0032]. All three FM arms are at 0.74–0.75.

## 16. G5 event safety [gate]

| SF − WW-L1 | estimate [95% CI] | rule | |
|---|---|---|---|
| FP / window | **+0.092 [+0.084, +0.101]** | CI upper < +0.05 | **FAIL** |
| recall | −0.0012 [−0.0020, −0.0004] | CI lower > −0.01 | PASS |

**G5: FAIL** (the FP condition).

**[descriptive]** SF has fewer false R than the other FM arms:

- SF − WW-FM: −0.030 [−0.040, −0.022];
- SF − IND: −0.050 [−0.057, −0.044].

## 17. G6 cross-scale coupling [gate]

| SF − IND | estimate [95% CI] | rule | |
|---|---|---|---|
| FD | **−0.71 [−1.08, −0.36]** | CI upper < 0 | PASS |
| beat corr | +0.0020 [+0.0004, +0.0038] | CI lower > −0.02 | PASS |

**G6: PASS.**

- Explicit coarse → fine coupling lowered FD beyond the uncoupled multiresolution model, at no morphology cost.
- It also lowered false R (descriptive, §16).
- The effect is small next to the representation effect (13.4 → 3.4).

## 18. G7 PPG conditional dependence [gate]

| PPG-shuffled − conditioned (SF) | estimate [95% CI] |
|---|---|
| FD | **+29.95 [+25.23, +34.60]** |
| beat corr | **−0.304 [−0.314, −0.294]** |
| FP / window | +0.374 [+0.354, +0.393] |
| F1 | −0.082 [−0.086, −0.078] |

- **G7: PASS.** SCALEFLOW strongly uses PPG.
- **[descriptive] Event shuffle** (PPG kept, rasters permuted) is catastrophic for events:
  - F1 −0.674 [−0.690, −0.658];
  - FP +2.94 per window;
  - FD +29.7;
  - correlation −0.724.

  The model follows the supplied raster closely.

## 19. Stochastic diversity [descriptive; 2,000 fixed SF-VAL windows, K = 16]

| | within-window waveform diversity (RMS) | beat-aligned diversity | generated / real beat-to-beat diversity | R-time seed SD (median) | K = 16 median-HR MAE vs single-sample HR MAE |
|---|---|---|---|---|---|
| WW-FM | 0.317 | 0.280 | 1.17 | 0.0 ms | 6.92 vs 6.57 |
| INDEPENDENT | 0.289 | 0.279 | 0.65 | 0.0 ms | 7.12 vs 7.45 |
| SCALEFLOW | 0.294 | 0.280 | 0.59 | 0.0 ms | 7.30 vs 6.78 |

**Readings:**

- **Timing:** samples differ in shape but not in R timing, which is fixed by the raster.
- **Beat-to-beat variation:** the scale models produce less variation between beats of a window than real ECG
  (ratio 0.59–0.65).
- **HR consensus:** K = 16 HR consensus does not improve HR.

## 20. Qualification

| gate | result |
|---|---|
| G2 | PASS |
| G3 | PASS |
| G4 | **FAIL** |
| G5 | **FAIL** |
| G6 | PASS |
| G7 | PASS |
| **QUALIFIED** | **NO** |

- **Failure categories:** morphology degradation; event degradation.
- **Not failure categories:** FM objective weak (G1 = MIXED, not NO BENEFIT), multiresolution weak, coupling
  unsupported, PPG unused.

## 21. Final TEST freshness audit

**NOT PERFORMED** (`final_test_freshness_audit.md`). The audit only follows qualification.

## 22. Final TEST results

**NOT EVALUATED — TEST was never opened.** No freeze commit exists, and no TEST file was created.

## 23. NFE / sampling efficiency [descriptive, after the frozen NFE = 8 verdict]

| | NFE 4 | NFE 8 | NFE 16 |
|---|---|---|---|
| SCALEFLOW FD / corr / FP | 4.68 / 0.761 / 0.593 | 2.71 / 0.740 / 0.624 | 2.27 / 0.727 / 0.643 |
| WW-FM FD / corr / FP | 12.81 / 0.770 / 0.590 | 13.43 / 0.746 / 0.654 | 13.97 / 0.730 / 0.698 |
| SF waveform-model GPU batch-1 latency | 6.3 ms | 12.4 ms | 24.6 ms |

- **[post-hoc]** More integration steps move SF samples closer to the data distribution (lower FD). Correlation with
  the true ECG falls and false R rise.
- At no NFE does SF reach WW-L1's morphology (0.820) or FP (0.531).
- The primary result is unchanged.

## 24. Scale diagnostics [descriptive]

| RMS (all / within ±10 samples of reference R) | coarse band | mid band | fine band |
|---|---|---|---|
| reference ECG band RMS | 0.582 / 0.572 | 0.107 / 0.222 | 0.129 / 0.174 |
| SF error vs reference | 0.425 / 0.508 | 0.135 / 0.276 | 0.156 / 0.214 |
| WW-L1 error vs reference | 0.375 / 0.490 | 0.131 / 0.278 | 0.137 / 0.203 |
| PPG-shuffle effect on SF | **0.353** / 0.331 | 0.066 / 0.111 | 0.169 / 0.174 |
| coupling effect (SF − IND) | 0.053 / 0.052 | 0.037 / 0.056 | 0.066 / 0.074 |

- **PPG shuffle:** its largest effect is on the coarse band, i.e. broad structure and baseline.
- **Coupling:** changes all bands a little.
- **Pointwise error:** SF's error is largest in the coarse band, where the deterministic WW-L1 is closer to the
  reference pointwise.
- **Cumulative SF error:** coarse only 0.425 → coarse + mid 0.446 → full 0.472.

## 25. Compute

| | WW-L1 | WW-FM | INDEPENDENT | SCALEFLOW |
|---|---|---|---|---|
| training time | 130.9 s | 147.0 s | 127.9 s | 131.7 s |
| peak memory | 1,664 MiB | 1,654 MiB | 1,373 MiB | 1,378 MiB |

- **Detector:** 55 s, 878 MiB.
- **Training:** no NaN step in any model.
- **Inference cost:** §11.

## 26. What SF0 establishes (SF-VAL only; one seed, one sample per window)

- **The multiresolution representation itself.** At matched parameters, a fixed two-level Haar representation for
  whole-window conditional flow matching gives FD far below a whole-window FM and a deterministic whole-window model:
  - SCALEFLOW 2.71 and INDEPENDENT 3.42;
  - WW-FM 13.43 and WW-L1 15.05.
- **Coupling.** Hierarchical coarse → fine coupling adds a small but significant FD improvement over independent
  scales (−0.71 [−1.08, −0.36]). It also lowers false R relative to independent scales, with no morphology cost.
- **PPG use.** The coupled model uses PPG strongly (shuffle FD +30, correlation −0.30), mostly through the coarse band.
- **Where the failure lies.** Single generated samples of all three FM models trade beat-aligned correlation (−0.07 to
  −0.08) and false-R safety (+0.09 to +0.14 per window) against the deterministic WW-L1. This is why the preregistered
  morphology and event-safety gates fail.

## 27. What SF0 does NOT establish

- **Anything on the old V1 TEST** (never opened), or any fresh validation.
- **That ScaleFlow is a better PPG→ECG model overall.** It fails morphology and event safety.
- **That the FD gain would replicate** on another population, seed or dataset.
- **That the post-hoc readings are mechanisms:** sample vs conditional-mean behaviour (§12), and NFE trends (§23).
- **Other claims never made:** novelty / first / state of the art; calibrated uncertainty; clinical validity;
  patient-specific morphology; usefulness of K-sample consensus (none was found).

## 28. Architecture-paper go/no-go

**NO-GO.** SF0 is closed under the no-rescue rule:

- no other wavelet, S5 / Mamba / Transformer / attention, NFE change, loss, width, scale, direction or fusion;
- TEST stays unopened.

A future line would need a new, separately preregistered hypothesis. **HARD STOP.**
