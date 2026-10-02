# AF0 — AnchorFlow-ECG — REPORT

> **AF0 DEVELOPMENT FAILED.**
> - **Budget:** all 10 predeclared candidate configurations were evaluated on AF-DEV (16 training jobs, budget 30).
> - **No candidate passed D1 – D7.** D2 and D3, the conditional-centre gates, failed for every candidate.
> - **Not opened:** AF-LOCK. **No lock freeze was created.**
> - **The old V1 TEST was never opened.** No AF1 draft was written.
> - **Architecture-paper NO-GO.**
>
> All results below are **AF-DEV adaptive-development evidence only**.

| commit | content |
|---|---|
| `0fcb5d1` | development protocol (frozen) + split + implementation (before any AF0 training or AF-DEV metric) |
| this commit | development record (10 candidates), summary artifacts, report |

## 한국어 요약

- **판정: AF0 DEVELOPMENT FAILED.**
  - 사전에 정한 탐색 공간(정규화 A, 앵커 주입 B, 결합 C, 용량 D, PPG 드롭아웃 G)에서 후보 10개를 AF-DEV로 개발했다.
  - D1 – D7을 모두 통과한 후보가 없어서 AF-LOCK과 old TEST는 열지 않았다.
- **가장 좋은 후보:** c05 (A0 정규화, FiLM 앵커 주입, 연결형 결합)
  - FD 9.05: 앵커 19.51, 전체 신호 ScaleFlow 9.66.
  - 잔차 FD 10.81: vanilla 잔차 22.98, 독립 다중해상도 14.65.
  - D1, D4, D5, D7 통과.
- **모든 후보가 막힌 곳: 조건부 중심(D2, D3).**
  - 16샘플 평균의 박동 정렬 상관이 앵커보다 0.12–0.16 낮다. 여백은 −0.02.
  - 16샘플 평균의 가짜 R이 창당 +0.11 ~ +0.26 많다. 여백은 +0.05.
  - 정규화, 앵커 주입, 결합, 용량, 드롭아웃 어느 것으로도 거의 움직이지 않았다.
  - 샘플마다 R 위치가 약 17 ms 흔들린다. 전체 신호 ScaleFlow는 0 ms다.
  - 16샘플 평균이 앵커에서 벗어나는 정도는 coarse 대역에 몰려 있다: 0.25, mid 0.056, fine 0.027.
- **조건 사용(D6):** 잔차 FD 기준으로 PPG나 앵커를 섞어도 나빠지지 않았다(통과 1/10).
- **[사후 해석, 미검증]**
  - 잔차를 검출기 이벤트에 놓인 앵커 기준으로 학습했다. 그래서 잔차 분포에 "검출기 위치 vs 실제 R"의 시점 오차와 큰
    저주파(기저선) 변동이 들어 있다.
  - 이 분포를 충실히 샘플링하면, 16개를 평균해도 앵커만큼 날카로운 중심이 나오지 않는다.
  - 잔차 학습 규약 변경은 사전 탐색 공간 밖이라 시도하지 않았다.
- **개발 단계의 기술적 관찰(확정 아님):**
  - 다중해상도 표현이 잔차 FD를 낮춘다: 22.98 → 14.65.
  - 결합이 잔차 FD를 더 낮춘다: −3.84 [−4.22, −3.51]. SF0의 결합 효과와 같은 방향이다.

## 1. Status

- **Process:** AF0 ran as protocolled (`0fcb5d1`):
  - split → detector and anchor (AF-TRAIN, frozen) → AF-TRAIN residual statistics;
  - baselines B1 – B3 → bounded adaptive search c01 – c10 on AF-DEV.
- **Budget:** 10 / 10 configurations and 16 / 30 training jobs. c08 was started twice (§11).
- **No development winner:** `development_winner.json` = NONE. Therefore:
  - no lock freeze, no AF-LOCK access;
  - no seed-robustness runs, no AF1 draft, no method draft.
- **Old TEST:** never loaded.
- **Implementation changes during development** (all recorded):
  - the stage-D width rule (round → floor, with a 1.5× cap check), §11;
  - a `summarize` stage, whose FLOP accounting was corrected once so that the anchor pass is not counted inside the
    residual-flow vector-field FLOPs.

## 2. Motivation from prior failures

- **The closed line** showed the trade-off: deterministic whole-window regression (WW-L1) is strong on paired
  morphology and events; full-signal ScaleFlow is strong on FD.
- **SF0's single-sample FM failed** morphology (−0.080) and event safety (+0.092) against WW-L1.
- **AF0's response** separates the two roles:
  - a frozen deterministic anchor μ(c) is the point estimate;
  - a conditional flow models only the residual x − μ(c).

## 3. Why point estimation and distribution generation are separated

- **Two outputs:** μ(c) is the paired estimate. μ(c) + r⁽ᵏ⁾ are conditional samples.
- **What is not asked:** a single sample is never asked to match the paired target as well as μ.
- **What is asked:**
  - the sample set must be distributionally realistic (D1, D4, D5, D7);
  - its conditional centre must stay a useful paired prediction (D2, D3: the mean of 16 samples vs μ);
  - it must use its conditions (D6).

## 4. Data and nested development design

| role | patients | windows | use |
|---|---|---|---|
| AF-TRAIN | 2,400 | 159,545 | detector, anchor, residual statistics, all flows |
| AF-DEV | 300 | 20,319 | adaptive development (repeated use) |
| AF-LOCK | 337 | 22,133 | sealed; **never opened** |

- **Construction:** nested inside SF-TRAIN (`default_rng(20261002)`).
- **Exclusions:** disjoint; no SF-VAL, ARCH-VAL / HOLDOUT, or old V1 validation / test patients.
- **Evidence status:** not project-naive.
- **Old V1 TEST:** closed.

## 5. Deterministic anchor

**Model:** AF-WW-ANCHOR — SF0's WW-L1 (WWDet 72 × 5, 593,577 parameters).

- **Training:** L1 on AF-TRAIN with the reference-R raster, 130 s.
- **Frozen:** sha256 in `anchor_checkpoint_hash.json`, re-verified before residual preparation.

**Point estimate on AF-DEV** (frozen AF detector events):

| metric | value [95% CI] |
|---|---|
| beat-aligned correlation | 0.805 [0.787, 0.821] |
| FP / window | 0.532 [0.463, 0.605] |
| recall | 0.780 [0.760, 0.799] |
| precision | 0.864 |
| F1 | 0.818 |
| RR-MAE | 8.04 ms |
| HR-MAE | 6.13 bpm |
| MAE | 0.308 |

- **Detector events** (54.7 s training): FP 0.424, recall 0.777, F1 0.826.

## 6. Residual formulation

- **Residual:** r = x − μ, where μ is the frozen anchor at the frozen detector's events.
- **Training residuals:** AF-TRAIN μ uses the in-sample AF-TRAIN events.
- **Residual statistics (AF-TRAIN):**

  | Haar band | std | IQR |
  |---|---|---|
  | coarse | 0.746 | 0.814 |
  | mid | 0.259 | 0.057 |
  | fine | 0.199 | 0.028 |

  The mid and fine bands are strongly heavy-tailed (IQR ≪ std), which is QRS-localized.
- **Normalization:** A0 / A1 / A2 / waveform statistics are frozen in `residual_stats.json`.

## 7. Multiresolution residual transport

- **Representation:** SF0's fixed two-level Haar (128 / 128 / 256).
- **Path:** the flow integrates the normalized coefficients from N(0, I) (Euler, NFE 8).
- **Output:** the decoded and inverse-Haar residual is added to μ.
- **Condition per scale:** [PPG_s, event_s, μ_s] (B0), FiLM from μ_s (B1), or both (B2), plus the time embedding.

## 8. Cross-scale coupling

| form | definition |
|---|---|
| C1 (base) | 16-channel projections of the coarse (→ mid, fine) and mid (→ fine) branch features, concatenated at the receiving stem |
| C0 | full-width additive |
| C2 | gated additive, learned channel-wise sigmoid gates |

There is no fine → coarse path (unit-tested).

## 9. Baselines (AF-DEV)

| arm | FD | residual FD | diversity ratio | mean16 corr | mean16 FP | R-time seed SD |
|---|---|---|---|---|---|---|
| B0 anchor (point) | 19.51 | — | — | 0.805 (point) | 0.532 (point) | — |
| B1 full-signal ScaleFlow | **9.66** | 24.60 | 0.60 | 0.791 | 0.557 | 0.0 ms |
| B2 anchor + vanilla residual FM | 21.69 | 22.98 | 0.69 | 0.684 | 0.788 | 16.6 ms |
| B3 anchor + independent multiscale residual FM | 13.11 | 14.65 | 0.71 | 0.672 | 0.753 | 15.9 ms |

- **K = 16 subset:** 2,000 windows. On it, the anchor's correlation is 0.801 and its FP 0.537.
- **B1's residual FD** measures B1's samples minus μ. It is reported for completeness; B1 does not use μ.

## 10. Adaptive development protocol

- **Rules:**
  - bounded at 10 configurations and 30 jobs;
  - one major factor changed at a time, in the predeclared order (normalization → anchor injection → coupling →
    capacity → amplitude → PPG dropout → NFE);
  - AF-DEV only; every result kept (`search_history.csv`, `candidate_configs/`, `candidate_metrics/`).
- **Stage F (learned residual amplitude α) was not run.** Under the FM-only objective a learned α has no training
  signal except through the normalized target. Dividing the target by α makes the loss decrease monotonically as α grows
  (degenerate). A loss constraining α would be an auxiliary loss, which is prohibited.
- **Stage E (NFE)** is defined only for a configuration that already passes at NFE 8, so it never applied.

## 11. Search history (AF-DEV; full table in `search_history.csv`)

| id | change (parent) | params | FD | res. FD | mean16 corr | mean16 FP | div | S1 effect | anchor-shuffle effect | D1–D7 |
|---|---|---|---|---|---|---|---|---|---|---|
| c01 | base: A0, B0 concat, C1, w 50 | 598,483 | 10.24 | 11.97 | 0.666 | 0.667 | 0.73 | −0.04 | −0.61 | F F F P P F P |
| c02 | A1 norm (c01) | 598,483 | 13.75 | 16.72 | 0.668 | 0.679 | 0.68 | +0.33 | +0.81 | F F F P F **P** P |
| c03 | A2 norm (c01) | 598,483 | 10.47 | 14.50 | 0.680 | 0.648 | 0.67 | −0.11 | −0.24 | F F F P F F P |
| c04 | B2 concat + FiLM (c01) | 597,583 | 9.26 | 11.58 | 0.678 | 0.697 | 0.72 | −0.15 | −0.79 | **P** F F P P F P |
| **c05** | B1 FiLM (c01) | 597,445 | **9.05** | **10.81** | 0.665 | 0.698 | 0.73 | −0.16 | −0.22 | **P** F F P P F P |
| c06 | C2 gated (c05) | 599,605 | 11.22 | 11.43 | 0.672 | 0.755 | 0.73 | −0.46 | −0.23 | F F F P P F P |
| c07 | C0 additive (c05) | 599,467 | 11.31 | 11.35 | 0.673 | 0.749 | 0.73 | −0.91 | −1.79 | F F F P P F P |
| c08 | width ×1.25 → 57 (c05) | 869,794 | 9.25 | 10.90 | 0.679 | 0.723 | 0.72 | −0.80 | −0.77 | **P** F F P P F P |
| c09 | PPG dropout 0.10 (c05) | 597,445 | 13.71 | 12.38 | 0.643 | 0.772 | 0.73 | −0.17 | −0.25 | F F F P P F P |
| c10 | A1 norm (c05) | 597,445 | 15.09 | 15.11 | 0.663 | 0.798 | 0.71 | −0.30 | −0.78 | F F F P F F P |

**Notes on the table:**

- mean16 values are on the 2,000-window K = 16 subset. S1 / anchor-shuffle effects are residual FD (shuffled −
  conditioned).
- **Stage decisions** (best on AF-DEV, mostly by residual FD and FD): A → A0, B → B1 (c05), C → C1, D → no gain, G → no
  gain.
- **c08 restart:** the first c08 run was stopped before completion. The rounded width 58 gave 897,145 parameters, 1.0016
  × the 1.5 × cap. The width rule became floor(46 × 1.25) = 57 (1.456 ×) and c08 was retrained. Both runs count in the
  job ledger.

## 12. Winning configuration

**None.** No configuration passed D1 – D7.

The best development configuration, by the protocol's lexicographic keys, is **c05**: A0, B1 FiLM, C1, width 46,
NFE 8, 597,445 parameters. It is recorded only as the best failed candidate.

## 13. Development results (c05, AF-DEV)

| gate | estimate [95% CI] | rule | |
|---|---|---|---|
| D1 FD − anchor | −10.46 [−12.56, −8.22] | CI < 0 | ✓ |
| D1 FD − full ScaleFlow | −0.61 [−2.15, +0.69] | upper < +1.0 | ✓ |
| D2 corr(mean16) − corr(anchor) | **−0.135 [−0.144, −0.127]** | lower > −0.02 | ✗ |
| D3 FP(mean16) − FP(anchor) | **+0.161 [+0.122, +0.199]** | upper < +0.05 | ✗ |
| D3 recall(mean16) − recall(anchor) | −0.004 [−0.009, +0.001] | lower > −0.01 | ✓ |
| D4 residual FD − B2 | −12.16 [−14.86, −9.51] | CI < 0 | ✓ |
| D5 residual FD − B3 | −3.84 [−4.22, −3.51] | CI < 0 | ✓ |
| D6 S1 (PPG in flow only) | −0.156 [−0.268, −0.041] | lower > 0 | ✗ |
| D6 anchor shuffle | −0.220 [−0.472, +0.144] | lower > 0 | ✗ |
| D7 diversity ratio | 0.73 | 0.5 – 1.5 | ✓ |

**Gate pass counts over the 10 candidates:**

| gate | passed |
|---|---|
| D4 | 10 |
| D7 | 10 |
| D5 | 7 |
| D1 | 3 |
| D6 | 1 |
| D2 | **0** |
| D3 | **0** |

## 14. Locked confirmation

**Not performed.** There was no development winner, so AF-LOCK was never loaded (`load_role("af_lock")` stays sealed;
no `lock_freeze_manifest.json`).

## 15. Point-estimate performance

The anchor (§5) is the pipeline's point output. It is a WW-L1-class estimator on AF-DEV (correlation 0.805, F1 0.818,
FP 0.532). AF0 did not change it.

## 16. Generative distribution fidelity

- **c05 vs the anchor:** residual sampling lowers population FD from 19.51 (anchor) to 9.05, and matches full-signal
  ScaleFlow (9.66; −0.61 [−2.15, +0.69]).
- **Single samples:**
  - correlation 0.408, FP 1.29 / window;
  - expected for stochastic samples, and not used as the point output.

## 17. Conditional center

This is the failure.

- **Size of the gap:** the mean of 16 samples has beat-aligned correlation 0.665 vs the anchor's 0.801 on the same
  windows, and FP 0.698 vs 0.537. The median of 16 is no better (0.663).
- **Where the centre drifts** (mean |x̄ − μ| per Haar band):

  | band | drift |
  |---|---|
  | coarse | 0.252 |
  | mid | 0.056 |
  | fine | 0.027 |

- **Within-condition spread** (coarse 0.46) means a 16-sample mean keeps substantial coarse-band variation.
- **R-time seed SD** is 17.3 ms (B1: 0 ms), so QRS corrections are sampled with timing jitter.
- **Every candidate sits in the same band:** correlation −0.121 to −0.157, FP +0.11 to +0.26.
- **[post-hoc reading, untested]** The residual target was built against an anchor placed at the frozen detector's
  events (in-sample on AF-TRAIN). It therefore contains timing corrections between detector events and true R, as well
  as large low-frequency residual variance. A faithful sampler of that distribution yields jittered QRS corrections and
  baseline variation that a 16-sample mean does not average out.
  - Alternatives that were not tried, because they lie outside the predeclared search space:
    - building residuals at reference-R events;
    - cross-fitting the anchor.
  - No claim is made that they would pass.

## 18. Residual distribution modeling

| | residual FD | spectral discrepancy |
|---|---|---|
| c05 | 10.81 | 0.51 |
| B3 | 14.65 | 0.88 |
| B2 | 22.98 | 0.68 |

- **Generated residual band variances are below the real ones** (diversity ratio 0.73). The generated coarse-band
  variance was 0.32 (c01) against a real 0.58.

## 19. Cross-scale attribution (development evidence only, not confirmed)

| comparison | arms | residual FD | difference [95% CI] |
|---|---|---|---|
| multiresolution | B3 vs B2 | 14.65 vs 22.98 | — |
| coupling | c05 − B3 | 10.81 vs 14.65 | −3.84 [−4.22, −3.51] |
| coupling, base form | c01 − B3 | 11.97 vs 14.65 | −2.68 [−2.99, −2.37] |

The direction is consistent with SF0's coupling effect. **[post-hoc]** C1 (concatenative) beat C0 and C2.

## 20. PPG condition use

- **Not demonstrated.**
  - The PPG-only shuffle inside the flow (S1) did not worsen residual FD for 9 of 10 candidates (c05: −0.16).
  - PPG dropout (c09) did not help.
- **The whole pipeline** depends on PPG through the anchor. S2 for c05 gives FD 43.6 and point correlation 0.09.

## 21. Anchor dependence

**Not demonstrated by the protocolled control.** The anchor shuffle did not worsen population residual FD; it was
negative or inconclusive for 9 of 10 candidates.

**[post-hoc]** Population residual FD compares marginal distributions. A residual generated under the wrong anchor
condition can keep a realistic marginal, so this control is weak at detecting conditional use. It was the frozen
criterion and is reported as such.

## 22. Diversity

| | value |
|---|---|
| diversity ratio std(r_gen) / std(r_real) | 0.67 – 0.73 (all within D7) |
| K16 within-condition residual diversity | 0.357, 0.81 × the real residual RMS (c05) |
| per-band within-condition SD | coarse 0.46 / mid 0.087 / fine 0.045 |
| beat-aligned diversity | 0.369 |

## 23. Compute

| | per-VF FLOPs | pipeline FLOPs (NFE 8) | GPU batch-1 latency | CPU 4-thread latency | training |
|---|---|---|---|---|---|
| detector | — | 336 M | — | — | 54.7 s |
| anchor | 606 M (one pass) | detector + anchor 942 M | 1.12 ms | 5.3 ms | 130 s |
| B1 full ScaleFlow | 156 M | 1.58 G | 13.7 ms\* | 31.8 ms\* | 132 s |
| B2 vanilla residual | 543 M | 5.29 G | 8.0 ms | 33.4 ms | 148 s |
| B3 independent residual | 154 M | 2.18 G | 12.9 ms | 31.9 ms | 128 s |
| c05 (best failed) | 158 M | 2.21 G | 18.3 ms | 36.7 ms | 178 s |

- **Latency measurement:** latencies include the detector and the anchor pass for every flow arm (\*B1 does not need the
  anchor).
- **FLOPs:** from `torch.utils.flop_counter` on one window.
- **Parameters:** residual flows about 600k (c08 869,794).
- **Matching:** parameters were matched, compute was not.

## 24. Supported claims

**Nothing is confirmed:** AF-LOCK was not opened. Development-only observations on AF-DEV:

- **Population FD.** Anchor-residual flow matching in a fixed Haar space reaches population FD comparable to
  full-signal ScaleFlow (c05 9.05 vs 9.66) and far below the anchor (19.51).
- **Representation and coupling.** Multiresolution residual representation and coarse → fine coupling each lower residual
  FD (B2 22.98 → B3 14.65 → c05 10.81).
- **Diagnostic record.** Within the predeclared space, no configuration kept the conditional centre (16-sample mean)
  within the anchor's correlation / false-detection margins.

## 25. Unsupported claims

- AnchorFlow as a confirmed method, and any architecture-paper claim.
- A conditional-centre property (D2, D3 failed everywhere).
- Residual-flow use of PPG or of the anchor, by the frozen shuffle controls.
- Any AF-LOCK or old-TEST result.
- The post-hoc explanations (§17, §21).
- Never claimed: first / novel / state of the art, clinical validity, calibrated uncertainty, patient-specific
  morphology, external generalization.

## 26. AF1 final-test plan

**Not created.** AF0 produced no development winner, so there is nothing to confirm; under the protocol the old V1 TEST
stays closed.

- **What a future line needs:**
  - a new, separately preregistered hypothesis;
  - a population that this program has not used for selection.
- **What it might address:** for example, the residual-target / timing convention identified in §17, which this
  protocol froze and did not search.
- **HARD STOP.**
