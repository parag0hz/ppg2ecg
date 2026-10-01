# C0 — CoherentBeat Deterministic Feasibility — REPORT

**Final verdict: STRONG (ARCH-HOLDOUT, prospective).** The holdout was opened exactly once, after a QUALIFIED ARCH-VAL
result and a committed freeze. All four preregistered gates pass on ARCH-HOLDOUT (434 patients, 28,531 windows).

The deterministic global-local model:

- preserves the placed-event rhythm (ΔF1 −0.0010 [−0.0013, −0.0007]);
- cuts false R detections relative to the retrained BF0 deterministic beat (−0.205 per window [−0.215, −0.195]);
- lowers FD (−8.63 [−9.85, −7.50]);
- keeps beat-aligned correlation within the margin (−0.0038 [−0.0048, −0.0028]).

| commit | content |
|---|---|
| `3b6ea98` | preregistration, split, implementation (before training) |
| `eaa6c39` | ARCH-VAL result: QUALIFIED |
| `dddaa4b` | holdout freeze (19 files hashed) |
| this commit | ARCH-HOLDOUT result, report, C1 design draft |

Labels used below:

- **[gate]** preregistered gate;
- **[descriptive]** preregistered and reported but not gated;
- **[post-hoc]** reading added after the result.

## 한국어 요약

- **판정: STRONG.** 홀드아웃 434명에서 네 게이트를 모두 통과했다. 사전등록 → ARCH-VAL 통과 → 동결 커밋 → 홀드아웃 1회 평가
  순서를 지켰다.
- 같은 검출기 이벤트에 놓았을 때 C0의 결과:
  - 이벤트 리듬을 그대로 지킨다. ΔF1 −0.001.
  - 재학습한 BF0 결정적 비트보다 가짜 R 검출이 창당 0.205개 적다.
  - FD가 8.6 낮다 (25.1 vs 33.7).
  - 박동 상관은 여백(−0.02) 안이다 (−0.004).
- 지지 경계 근처의 가짜 검출은 12,761개 중 1개로, BF0·D0식 경계 인공물이 사실상 없다.
- **한계 (사후 해석):**
  - LOCAL-ONLY(g = 0)는 기저선까지 없애서, "시간에 따라 변하는 전역장"의 역할은 분리되지 않았다.
  - C0는 창 단위로 학습했고 비교 대상은 비트 단위로 학습해서, 구조 효과와 학습 단위 효과가 섞여 있다.
  - 결정적 모델만 시험했다.
- 규칙에 따라 C1 설계 초안만 만들었고, C1은 학습하지 않았다.

## 1. Status

- **C0 is complete:** Stage A was QUALIFIED and Stage B is STRONG.
- **Process:**
  - every component was trained once on ARCH-TRAIN (seed 42, last checkpoint, no early stopping, no NaN step);
  - every stage ran once in the preregistered order;
  - no deviation from the preregistration.
- **Holdout seal:** ARCH-HOLDOUT was loaded only by `evaluate_holdout` and `figure`, after `dddaa4b`. The loader
  re-checked that the freeze manifest was committed and every frozen hash unchanged.

## 2. Motivation from BF0 and D0

- **BF0** (`d380e3b`, Case A): explicitly placed stochastic beats lost observable rhythm (ΔF1 −0.0284). Post hoc, the
  extra detections were neighbour-beat "ghosts" at window edges, drawn by beat outputs whose 166-sample support exceeds
  one RR.
- **D0** (`a54ec7e`, NOT SUPPORTED): clipping each beat to an event cell removed the edge ghosts, but adjacent,
  independently generated absolute-level beats then met as level steps that were detected as R.
- **C0's hypothesis:** a local generator should not set the absolute waveform context. A global continuous field
  should, and local residuals should vanish smoothly at their support boundary.

## 3. New prospective split

The V1 VitalDB TRAIN patient pool (4,337 patients) was split by `default_rng(20261001)` using the manifest metadata
only:

| role | patients | windows |
|---|---|---|
| ARCH-TRAIN | 3,470 | 231,220 |
| ARCH-VAL | 433 | 28,649 |
| ARCH-HOLDOUT | 434 | 28,531 |

- **Integrity:** patient-disjoint; zero overlap with the old validation (289) and test (1,156) patients; never
  reshuffled. Hashes are in `split_hashes.json`.
- **Holdout history (disclosed):** ARCH-HOLDOUT patients were training data of earlier project models and part of
  TRAIN-only diagnostics. They were untouched by every C0 component and decision, but not "never seen by the project".

## 4. Architecture

x̂(t) = g(t) + Σ_i m_i(t − r_i) · q_i(t − r_i)

- **Shared PPG encoder:** 8 dilated residual blocks, 64 channels.
- **Global field g:** a cubic B-spline with 17 coefficients at 250 ms spacing over the whole window.
- **Local residual q_i:** on τ = −38 … +57 samples, from encoder features around r_i, FiLM-conditioned on the window
  summary and (RR_prev, RR_next).
- **Envelope m_i:** a compact bump exp(1 − 1/(1 − u²)) on physical-time supports min(300 / 450 ms, 0.45 RR).
- **Rendering:** no renormalization, no fill, no warping.
- **Parameters:** 592,770, which is 1.23 × the BF0 deterministic beat's 482,049.

## 5. Why global + local

- **The global field** exists everywhere. Where every local residual vanishes, x̂ = g, so no beat can leave a level or
  slope discontinuity.
- **The local residuals** add only compact structure, mainly the QRS. Each returns to the shared field by construction.

## 6. Compact C¹ residual support

- **The bump is C^∞:** m(0) = 1, and m and every derivative are 0 at −L_i and +R_i.
- **Supports cannot overlap:** each side is at most 0.45 RR, so no residual reaches a neighbouring event's centre.
- **Unit tests:** the local output equals the envelope exactly (no renormalization or warping) and is exactly zero
  outside its support.
- **On real data [descriptive]:** the local residual at support boundaries had a median of 3 × 10⁻²¹ (ARCH-HOLDOUT P99
  1.3 × 10⁻⁷).

## 7. Training protocol

| component | data / target | steps × batch | time (RTX 5090) | params |
|---|---|---|---|---|
| timing detector (RD1 RhythmTCN) | ARCH-TRAIN; Gaussian σ = 20 ms at reference R; BCE | 14,000 × 64 windows | 55 s | 328,897 |
| BF0-DET-RETRAIN (BeatFlowNet, L1) | ARCH-TRAIN beats at reference R | 20,000 × 256 beats | 101 s | 482,049 |
| CoherentBeat-C0 | ARCH-TRAIN windows, events = reference R; full-window L1 | 20,000 × 64 windows | 145 s | 592,770 |
| C0-LOCAL-ONLY (g = 0) | same | 20,000 × 64 windows | 141 s | 588,545 |

- **Settings:** AdamW lr 1e-3, weight decay 0.01, clip 1.0 (beat and window models), seed 42, last checkpoint.
- **Final training losses:** C0 L1 0.253, LOCAL-ONLY 0.348, BF0-DET (beat-level) 0.249.
- **Evaluation events:** every arm was evaluated at the retrained detector's events (threshold 0.35, refractory 32), one
  shared sequence per window.

## 8. Retrained BF0 baseline

BF0-DET-RETRAIN is BF0's deterministic BeatFlowNet, retrained on ARCH-TRAIN and rendered exactly as in BF0. That means
the Hann overlap-add with renormalization, BF0's conditioning, and the template baseline for windows without events.
Historical BF0 checkpoints were not used, because they were trained on patients now inside the new split.

## 9. ARCH-VAL qualification [gate]

| gate | ARCH-VAL (433 patients, 28,649 windows) | rule | result |
|---|---|---|---|
| G1 ΔF1 C0 − placed | −0.0012 [−0.0015, −0.0009] | lower > −0.02 | PASS |
| G1 ΔRR-MAE C0 − placed | −0.0002 [−0.0019, +0.0013] ms | upper < +2 ms | PASS |
| G2 ΔFP / window C0 − BF0-DET | −0.209 [−0.219, −0.199] | CI < 0 | PASS |
| G3 ΔFD C0 − BF0-DET | −9.02 [−10.21, −7.94] | CI < 0 | PASS |
| G4 Δ beat corr C0 − BF0-DET | −0.0043 [−0.0055, −0.0032] | lower > −0.02 | PASS |
| **verdict** | **QUALIFIED** | | |

Development figure: `figure_val.png`.

## 10. G1 rhythm preservation [gate]

| | ARCH-VAL | ARCH-HOLDOUT |
|---|---|---|
| placed events (retrained detector) F1 | 0.7901 [0.7729, 0.8071] | 0.7888 [0.7723, 0.8045] |
| C0 render F1 | 0.7890 | 0.7878 |
| **ΔF1** | −0.0012 [−0.0015, −0.0009] | **−0.0010 [−0.0013, −0.0007]** |
| placed / C0 RR-MAE (ms) | 7.45 / 7.45 | 7.82 / 7.79 |
| **ΔRR-MAE** | −0.0002 [−0.0019, +0.0013] | **+0.0021 [−0.0005, +0.0055]** |

C0's render keeps the rhythm the detector supplied, without BF0's loss (BF0 ΔF1 was −0.0284). The rhythm itself (F1
about 0.79) is the detector's. C0 cannot fix missed or false events.

## 11. G2 false-event reduction [gate]

| false R detections per window (patient mean) | ARCH-VAL | ARCH-HOLDOUT |
|---|---|---|
| placed events (as a sequence) | 0.403 | 0.436 |
| BF0-DET-RETRAIN | 0.622 | 0.650 |
| C0-LOCAL-ONLY | 0.753 | 0.789 |
| CoherentBeat-C0 | 0.413 | 0.445 |
| **C0 − BF0-DET** | −0.209 [−0.219, −0.199] | **−0.205 [−0.215, −0.195]** |

- **Raw FP on ARCH-HOLDOUT:** C0 12,761, BF0-DET 18,617, placed events 12,491.
- **What remains:** almost all of C0's false detections are the detector's own false events.

## 12. G3 waveform distributional quality [gate]

| FD (lower = better) | ARCH-VAL | ARCH-HOLDOUT |
|---|---|---|
| BF0-DET-RETRAIN | 32.72 | 33.74 |
| C0-LOCAL-ONLY | 39.00 | 41.66 |
| CoherentBeat-C0 | 23.70 | 25.10 |
| C0 global field alone [descriptive] | 53.17 | 55.06 |
| **C0 − BF0-DET** | −9.02 [−10.21, −7.94] | **−8.63 [−9.85, −7.50]** |

- **Other waveform measures, ARCH-HOLDOUT [descriptive], C0 vs BF0-DET:**
  - MAE: 0.296 vs 0.319;
  - PCC: 0.350 vs 0.332;
  - spectral ratio deviation: 2.13 vs 2.60;
  - S4 / S5: 0.347 / 0.266 vs 0.346 / 0.265;
  - HR error: 4.15 vs 5.28 bpm.
- **What D0 got wrong, avoided:** unlike D0, events did not improve at the expense of FD.

## 13. G4 morphology non-inferiority [gate]

| matched-pair beat-aligned correlation | ARCH-VAL | ARCH-HOLDOUT |
|---|---|---|
| BF0-DET-RETRAIN | 0.8206 | 0.8260 |
| C0-LOCAL-ONLY | 0.7475 | 0.7540 |
| CoherentBeat-C0 | 0.8163 | 0.8222 |
| **C0 − BF0-DET** | −0.0043 [−0.0055, −0.0032] | **−0.0038 [−0.0048, −0.0028]** |

C0 is slightly but clearly lower than BF0-DET, and well inside the −0.02 non-inferiority margin. ARCH-HOLDOUT has
90,782 matched pairs.

## 14. Local-only ablation [descriptive]

| ARCH-HOLDOUT | LOCAL-ONLY − BF0-DET | C0 − LOCAL-ONLY |
|---|---|---|
| FP / window | +0.140 [+0.105, +0.173] | −0.344 [−0.372, −0.316] |
| FD | +7.93 [+3.55, +12.58] | (C0 25.10 vs 41.66) |
| beat correlation | −0.072 [−0.077, −0.067] | |
| F1 − placed | −0.0349 [−0.0381, −0.0318] | |

- **What the data show:** without g, the model is worse than both C0 and the BF0 baseline. 42% of its false
  detections lie within ±2 samples of a support boundary.
- **Why this does not isolate the global field [post-hoc, and stated in the prereg]:** with g = 0 the output is forced
  to 0 between supports, while the data's baseline is near −0.5. The flat plateaus are visible in the figures. So the
  ablation shows that *something* must carry the window context. It does not show that a *time-varying* field is better
  than a constant offset.

## 15. Boundary continuity [descriptive]

| ARCH-HOLDOUT | C0 | LOCAL-ONLY | reference ECG at the same places |
|---|---|---|---|
| FP within ±2 samples of a support boundary | **1 / 12,761** | 9,504 / 22,478 | — |
| value jump at boundary, median / P95 / P99 | 0.0069 / 0.0125 / 0.0142 | ~0 (forced to 0) | 0.0097 / 0.101 / 0.783 |
| first-difference jump, median / P95 | 0.0010 / 0.0022 | 0.0005 / 0.0119 | 0.0098 / 0.091 |

C0's waveform is smoother at support boundaries than real ECG is at the same locations. ARCH-VAL is the same: 2 of
11,684 false detections near a boundary.

## 16. Global / local decomposition [descriptive]

| | ARCH-VAL | ARCH-HOLDOUT |
|---|---|---|
| global RMS (incl. baseline) / global AC RMS | 0.457 / 0.146 | 0.458 / 0.146 |
| local RMS | 0.263 | 0.263 |
| share of QRS-region (±10 samples) energy from the local branch | 0.892 | 0.893 |
| share of non-QRS AC energy from the global branch | 0.394 | 0.396 |

**[post-hoc reading]** The intended division of labour emerged: the local branch carries the QRS, and the global field
the baseline and part of the slow structure. Outside the QRS, the local residuals still carry most of the AC energy
(P/T within each support).

## 17. Holdout freeze

- **Freeze:** `holdout_freeze_manifest.json` (commit `dddaa4b`) records sha256 for 19 files:
  - the preregistration;
  - all C0 code and tests;
  - the four checkpoints and the BF0-DET template;
  - the split manifest, audit and qualification;
  - the shared metric and renderer code.
- **Check at load time:** `check_freeze()` verified every hash and that the manifest was committed and unmodified before
  the holdout was loaded.

## 18. ARCH-HOLDOUT results [gate]

| gate | ARCH-HOLDOUT | result |
|---|---|---|
| G1 | ΔF1 −0.0010 [−0.0013, −0.0007]; ΔRR +0.0021 [−0.0005, +0.0055] ms | PASS |
| G2 | ΔFP −0.205 [−0.215, −0.195] | PASS |
| G3 | ΔFD −8.63 [−9.85, −7.50] | PASS |
| G4 | Δ corr −0.0038 [−0.0048, −0.0028] | PASS |
| **verdict** | **STRONG** | |

- **Population:** 28,531 windows, 434 patients. 1,993 windows (7.0%) had no detected event and were rendered as C0's
  global field (BF0-DET: the template baseline). There were no windows without reference beats.
- **Main figure:** `figure.png` (ARCH-HOLDOUT). Panel C shows the first salted windows, including a detector miss.

## 19. Compute / parameters / latency

| | detector | BF0-DET | C0 | LOCAL-ONLY |
|---|---|---|---|---|
| parameters | 328,897 | 482,049 | 592,770 | 588,545 |
| training time | 55 s | 101 s | 145 s | 141 s |
| peak GPU memory | 963 MiB | 1,466 MiB | 1,390 MiB | 1,377 MiB |
| NaN steps | 0 | 0 | 0 | 0 |

Batch-1 latency (detector + waveform model; 50 salted ARCH-VAL windows):

| | C0 | BF0-DET |
|---|---|---|
| GPU | 1.90 ms | 1.66 ms |
| CPU (4 threads) | 5.51 ms | 5.33 ms |

C0 is one forward pass: no sampling, no NFE loop.

## 20. Failed criteria

None. Every preregistered gate passed on ARCH-VAL and on ARCH-HOLDOUT.

## 21. What C0 establishes

On ARCH-HOLDOUT patients, untouched by every C0 component and evaluated once under a committed freeze, a
**deterministic** model has these properties when rendered at the same retrained-detector events:

- it combines a smooth global B-spline field with compact, C^∞-enveloped event-local residuals;
- it preserves the placed-event rhythm (ΔF1 −0.001);
- it produces 0.205 fewer false R detections per window than the retrained BF0 deterministic beat model with BF0's
  renderer;
- it has lower FD (−8.6) and beat correlation within −0.02.

Its support boundaries produced essentially no detector-visible artifacts (1 of 12,761). This validates the
deterministic architectural substrate only, on VitalDB, with one seed.

## 22. What C0 does NOT establish

- **That the time-varying global field is the mechanism.** The g = 0 ablation also removes the baseline offset, so a
  constant offset was never tested (§14).
- **That the gain is architectural rather than a training-level effect.** C0 was trained on full windows and BF0-DET on
  166-sample beats; this confound is not separated.
- **Stochastic generation, or any claim about sample spread or uncertainty.** No stochastic model was trained.
- **Realism comparable to generative models.** No generative comparator ran on the ARCH splits. iMF's FD (3.46) was
  measured on the old validation split, a different population, and is not comparable.
- **Better rhythm than the detector.** C0's F1 is the detector's F1.
- **Anything else:**
  - test-split, cross-dataset or multi-seed results;
  - patient-specific morphology;
  - clinical validity;
  - "best PPG→ECG method";
  - superiority to iMF or PENGUIN;
  - novelty as a first global-local or coherent generator.
- **That ARCH-HOLDOUT patients were never seen by the project.** They were training data for earlier models (§3).

## 23. C1 go/no-go

**GO for designing C1 only.** The preregistered STRONG verdict permits `docs/C1_SHARED_LATENT_DESIGN_DRAFT.md`. Its
first proposed comparison is independent beat noise vs a shared window latent, at matched noise dimension, NFE, data and
compute.

C1 is not trained, preregistered or started here. ARCH-HOLDOUT has now been used once, so a C1 claim needs a new primary
evidence split or an explicit statement of reuse. **HARD STOP.**
