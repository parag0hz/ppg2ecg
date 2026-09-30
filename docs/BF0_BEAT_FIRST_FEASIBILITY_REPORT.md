# BF0 — Beat-First Generator: Feasibility on VitalDB Validation — REPORT

**Verdict (preregistered, frozen gate order): Case A — G1 FAIL.** The stochastic-beat render loses R-peak F1 against
the positions it was placed at by −0.0284 [−0.0312, −0.0257], beyond the −0.02 non-inferiority margin. By the frozen
rule this is an architecture-premise failure and **BF1 does not start**. The other four gates pass and are reported
separately (G3, G4b, G4a, G2); they do not change the case.

Preregistration: `docs/BF0_BEAT_FIRST_FEASIBILITY_PREREGISTRATION.md`, committed with the implementation in `504395d`
before any BF0 number. Artifacts: `artifacts/bf0_beat_first/`. Every number below is from the VitalDB **validation**
split (4,822 windows, 289 patients), which earlier stages have already analyzed; the test split was never loaded.
95% CIs are patient-clustered bootstrap percentiles (equal patient weight; 2,000 resamples, FD 1,000; seed 20260930).

Labels: **[gate]** = preregistered gate result; **[secondary]** = preregistered secondary result;
**[post-hoc]** = analysis written after the result (`scripts/bf0_posthoc.py`, `posthoc.json`), never gate evidence.

## 한국어 요약

- **판정: Case A (G1 실패).** 확률 비트를 놓아 만든 ECG에서 다시 검출한 R 피크 F1이, 놓은 위치 자체보다 0.028 낮다
  (CI [−0.031, −0.026], 여백 −0.02 밖). 사전등록 규칙상 구조 전제 실패이며 BF1은 시작하지 않는다.
- 나머지 게이트는 모두 통과했다. 다만 Case를 바꾸지 않는다.
  - G3 분포 이득: FD가 템플릿보다 −28.3, 결정적 비트보다 −6.5 낮다.
  - G4b 붕괴 없음: 창 안 다양성이 결정적 비트보다 크다.
  - G4a 타이밍 불변: realization 간 R 시각 SD 0.0 ms (iMF 18.5 ms).
  - G2 중심 일치: 16개 평균의 상관이 템플릿과 같다.
- **G1 실패의 원인 (사후 분석):** 비트 시각이 흔들린 것이 아니다. 생성된 형태가 가짜 R 검출을 만들었다(놓지 않은 곳의
  검출 1,450개, 템플릿은 59개). 놓은 비트 334개는 검출되지 않았다.
- **G3의 한계:**
  - 위치가 있는 창 4,384개로 좁히면(사후) 결정적 비트 대비 차이가 사라진다: +0.98 [−2.85, +4.66].
  - 정답 R 위치에서는 결정적 비트가 확률 비트보다 FD가 낮다 (사전등록 진단).
  - 모든 BF0 arm의 FD(29–62)가 iMF 샘플 하나(3.5)보다 훨씬 나쁘다.
- 확률 비트의 창 안 박동 간 다양성은 실제 ECG의 2.3배로, 과분산이다.

## 1. Question

Can we fix the rhythm explicitly, randomize only beat morphology, and improve waveform distributional realism without
losing rhythm fidelity? The architectural hypothesis under test: *do not ask stochastic generation to reproduce
structure that is already identifiable from the condition.* R-peak timing is treated as condition-identifiable,
morphology as ambiguous, and randomness is spent on morphology only.

## 2. Why BF0 exists

The generators studied in this program draw one noise vector that moves both beat timing and beat shape. On VitalDB
test, one iMF sample reaches R-peak F1 0.674 and RR-MAE 20.8 ms. ED1's consensus over 16 samples repairs this to 0.765
and 10.8 ms (`docs/ED1_EVENT_CONSENSUS_DECODING_REPORT.md`). A direct detector does better still: RD1 F1 0.7725,
RR-MAE 7.67 ms (`docs/RD1_DIRECT_RPEAK_DETECTOR_REPORT.md`). So part of the randomness these generators spend goes
into re-deriving timing that a discriminative model already identifies. BF0 is the cheapest experiment that asks
whether a factorized design keeps the timing and still gains realism over a fixed or deterministic beat:

- timing from a discriminative event stage;
- morphology from a beat-level generator;
- beats placed at fixed times.

It is a feasibility test with falsification gates, not a benchmark.

## 3. Relation to Era-1 findings

| Era-1 finding (source) | how BF0 uses it |
|---|---|
| A detector plus a fixed template out-scores every trained generator on the event axis (N1: f1_excess +0.4866 vs +0.3582 for the best trained arm) | A1 (template at the predicted positions) is the baseline every stochastic claim must beat |
| PPG does not carry beat shape (N2: template 0.9062 vs PPG regressor 0.9057 per-beat correlation) | accuracy is not a success axis; S-PPG tests whether PPG conditioning matters |
| Per-beat timing uncertainty is predictable from PPG but was overconfident (N5 / N7) | the N5-style head is kept; conformal calibration is a secondary output, never a gate |
| One missing or extra beat damages morphology 8.7× more than ±8-sample (62.5 ms) jitter (E1); a canonical generator trained on oracle coordinates tolerates only ±15.6 ms jitter (O3) | beats are *placed* at fixed times; no phase warp, no RR-dependent stretching |
| Oracle canonical coordinates raise morphology correlation 0.104 → 0.841 (O2c, oracle diagnostic) | beats are generated in R-aligned coordinates; oracle arms O1–O3 separate model failure from placement failure |

[post-hoc reading] E1's lesson returns in BF0 in a new form. What costs rhythm fidelity here is **extra and missing
beats in the rendered waveform**, not timing jitter (§7).

## 4. Frozen architecture

```
PPG window (4 s, 128 Hz)
  -> RD1 detector (frozen, 328,897 params): events e_i (threshold 0.35, refractory 32)
  -> timing head (N5-style MLP, 132,610 params): mu_i, sigma_i of the residual to the reference R
  -> positions t_i = e_i + round(mu_i)                         (fixed; sigma never perturbs a waveform)
  -> beat model at every t_i, R-aligned 166-sample segment, conditioned on the PPG segment and (RR_prev, RR_next):
       A1 train-median template | A2 BeatFlowNet deterministic (L1) | A3 BeatFlowNet OT-CFM, Euler 8 steps, 16 realizations
  -> fixed-time Hann overlap-add -> ECG window
```

Other arms:

- **Oracle arms O1–O3:** the same three morphology sources placed at the reference R peaks.
- **Shuffles:** S-PPG and S-RR are A3 (realization 0) with each beat's PPG segment, or its (RR_prev, RR_next), taken
  from a beat of another patient.
- **Empty windows:** a window without any position is filled with the template's baseline value.

Full specification: prereg §2.

## 5. Preregistration and provenance

- **Before any BF0 number:** the preregistration and implementation were committed and pushed as `504395d`.
  - The pre-commit fidelity audit against the execution specification is `artifacts/bf0_beat_first/audit.md`. It
    lists 13 corrections, all made before results. A synthetic dry run of every stage found one path bug, fixed before
    the commit.
  - `prereg_manifest.json` holds the sha256 of the prereg and the code.
  - At run time the prereg and all seven code files hashed **identically** to the manifest (`input_hashes.json`), and
    the run's `git HEAD` was `504395d`.
  - RD1 checkpoint sha256 prefix: `084a316fa663f9dc`, as preregistered.
- **Run:** each stage ran once, in the preregistered order (`audit`, `train_timing`, `train_beat`, `render`,
  `evaluate`, `atlas`, `compute`, `figure`), on 2026-09-30 from 22:19 to 22:31. No stage crashed. **No deviation from
  the preregistration.**
- **Data:**
  - VitalDB V1 split.
  - Train: 288,400 windows, 4,337 patients.
  - Validation: 4,822 windows, 289 patients (14–64 windows per patient, median 16), 21,890 reference beats. Every
    window has at least one reference beat.
  - Train–validation patient overlap: 0.
  - Conformal halves: C 144 patients, E 145 patients.
  - The test split was never loaded.
- **Validation reuse:**
  - This split is not fresh. ED1 chose its decoding parameters on it, and FBC1 and M1 drew evidence from it.
  - No BF0 component was selected on it: every checkpoint is the last step, and no hyperparameter was chosen.
  - Nothing here is a held-out test result.
- **Software and hardware:** Python 3.13.9, torch 2.11.0+cu130, numpy 2.3.5, scipy 1.16.3, neurokit2 0.2.12;
  Intel i5-14600K (20 threads), NVIDIA RTX 5090 (CUDA 13.0). No other GPU process ran.

## 6. Training

| model | data | parameters | steps | loss (step 1,000 → last) | time | NaN steps | checkpoint |
|---|---|---|---|---|---|---|---|
| timing head | 1,076,903 train RD1 events matched to a reference R (of 1,120,172; residual SD 29.4 ms) | 132,610 | 6,000 | NLL 7.15 → 4.76 | 3.9 s | 0 | last step |
| deterministic beat (A2) | 972,455 R-aligned train beats | 482,049 | 20,000 | L1 0.276 → 0.248 | 101 s | 0 | last step |
| stochastic beat (A3) | same 972,455 beats | 482,049 | 20,000 | OT-CFM 0.205 → 0.138 | 105 s | 0 | last step |

- **A1 template:** the sample-wise median of the training beats. The train median RR is 0.781 s.
- **A2 and A3:** same architecture, parameter count, data, optimiser (AdamW, lr 1e-3, weight decay 0.01, clip 1.0,
  batch 256), steps and seed (42). They differ only in the loss.
- **Sweeps:** none.
- **Render:** 18,557 predicted-position beats and 21,890 oracle beats. 438 of 4,822 windows (9.1%) have no RD1 event
  and are flat-filled (value −0.542) in every predicted-position arm. No non-finite sample appeared in any arm.

## 7. G1 Rhythm preservation — **FAIL** [gate]

| scored against the reference R | R-peak F1 (evaluable windows) | RR-MAE (ms) |
|---|---|---|
| placed positions (as a peak sequence) | 0.7789 [0.7566, 0.8003] | 7.75 [7.34, 8.20] |
| A3 render, realization 0 (neurokit detections) | 0.7506 [0.7283, 0.7713] | 8.02 [7.60, 8.49] |
| **difference (render − positions)** | **−0.0284 [−0.0312, −0.0257]** (rule: lower bound > −0.02) | **+0.19 [+0.10, +0.28]** (rule: upper bound < +2.0) |

- **RR-MAE part:** passes.
- **F1 part:** fails. The whole CI lies beyond the margin. The paired populations are 4,822 windows for F1 and 3,789
  for RR-MAE.

Diagnostics that locate the loss:

- **R-location drift [secondary]:**
  - A1: 100% of placed positions are detected within ±50 ms.
  - A2: 99.99%.
  - A3: 98.2%. Of the detected beats, 99.6% sit within one sample of their placed position (mean offset +0.06 ms).
- **Post-hoc detection breakdown:**
  - The stochastic render has 334 placed beats without a detection.
  - It has **1,450 detections where no beat was placed**, against 59 for the template and 1,023 for the deterministic
    beat.
  - Pooled against the reference:
    - placed positions: TP 16,562, FP 1,995, FN 5,328;
    - A3 render: TP 16,388, FP 3,285, FN 5,502;
    - precision 0.893 → 0.833, recall 0.757 → 0.749.
- **Descriptive F1 of the other renders [secondary, not bootstrapped as differences]:**
  - template 0.7780 (−0.0009 vs positions);
  - deterministic 0.7659 (−0.0130);
  - stochastic 0.7506 (−0.0284).
  The loss grows with how much the learned morphology varies.

**Reading.** Rendering does not move the beats it places. The generated morphology adds peaks that a standard R-peak
detector reads as extra beats, and it hides some placed beats. Under the frozen rule this is Case A. A secondary
metric does not rescue it.

## 8. G3 Distributional gain — PASS [gate]

| arm (one waveform per window, 4,822 windows) | window FD ↓ | beat-level FD ↓ |
|---|---|---|
| A1 template | 61.71 | 11.85 |
| A2 deterministic | 39.86 | 8.71 |
| **A3 stochastic, realization 0** | **33.39** | **1.58** |
| **FD(A3) − FD(A1)** | **−28.32 [−31.62, −24.35]** | |
| **FD(A3) − FD(A2)** | **−6.46 [−9.77, −3.10]** | |

- Both CIs lie entirely below 0. Every bootstrap replicate kept ≥ 4,622 windows (raw-waveform FD regime).

Limits on how far G3 reaches:

- **Oracle placement [secondary]:** at reference R positions, the stochastic beat is worse than the deterministic beat
  (O3 − O2 = +8.53 [+5.87, +11.73]; §12).
- **Post-hoc, excluding the 438 flat-filled windows:** identical flat windows in every arm still change FD
  non-additively. On the 4,384 windows with positions:
  - FD(A3) − FD(A1) = −21.52 [−25.68, −16.95];
  - **FD(A3) − FD(A2) = +0.98 [−2.85, +4.66]**.
  So the advantage over the deterministic beat depends on the preregistered population.
- **Comparator [secondary]:** every BF0 arm is far less realistic by window FD than one iMF sample:
  - full population: iMF 3.46 vs A3 33.39;
  - post-hoc, windows with positions: iMF 3.11 vs A3 29.55;
  - ED1 decoded: 24.48.
  At the beat level the order reverses: A3 1.58 vs iMF single 3.70.

## 9. G4b Non-collapse — PASS [gate]

| within-window beat diversity D (mean pairwise RMS of 83-sample beat windows) | value |
|---|---|
| A1 template | 0.0062 [0.0053, 0.0072] |
| A2 deterministic | 0.0528 [0.0507, 0.0548] |
| A3 stochastic (s = 0) | 0.3240 [0.3198, 0.3283] |
| real ECG (reference R) | 0.1665 [0.1586, 0.1748] |
| **D(A3) − D(A2)** (same beat set, 4,281 windows) | **+0.2712 [+0.2670, +0.2756]** |
| D(A3) / D(real) (same 4,280 windows) | 2.33 (practical target ≥ 0.25: met; not a gate) |
| seed-to-seed diversity [secondary] | A3 0.318 (16,827 beats, placed R); iMF 0.488 (19,826 beats, reference R) |

- A1's D is not 0 because neighbouring beats overlap in the Hann overlap-add, and their spacing varies.
- The stochastic branch did not collapse. It **over-disperses**: beats within one window differ 2.3× more than real
  beats do. The seed-to-seed diversity of one beat (0.318) is about the same as the diversity between beats of one
  window (0.324). That is what independent per-beat noise implies.
- The atlas (`atlas.png`) shows the consequence [post-hoc reading]: consecutive stochastic beats in one window take
  different baseline levels. The Hann overlap-add blends them smoothly, so the join-artifact rate J stays 0 (§14).

## 10. G4a Timing invariance — PASS [gate]

| R-time SD across 16 draws (median over reference beats matched in ≥ 8 of 16) | value | beats |
|---|---|---|
| **BF0 A3, realizations 0–15** | **0.0 ms** | 16,629 |
| iMF, 16 samples | 18.54 ms | 15,986 |
| threshold | 7.8125 ms (one sample; "7.8 ms") | |

- Over all 16 realizations, 98.1% of placed beats are detected and 99.7% of those lie within one sample.
- Randomness in the beat generator does not move the detected R times. Section 7 shows it changes which peaks are
  detected, not when.

## 11. G2 Center consistency — PASS [gate]

| matched-pair beat-aligned correlation (15,082 pairs in 4,137 windows, the same pairs for every arm) | value |
|---|---|
| A3 16-sample mean | 0.8122 [0.7944, 0.8293] |
| A1 template (primary comparator) | 0.8112 [0.7926, 0.8295] |
| **difference A3-mean − A1** | **+0.0010 [−0.0027, +0.0048]** (rule: lower bound > −0.02) |
| A2 deterministic (secondary comparator) | 0.8199 [0.8013, 0.8373]; A3-mean − A2 = −0.0076 [−0.0095, −0.0058] |
| A3 single realization [secondary] | 0.7150 [0.6989, 0.7309] |

- The centre of the stochastic distribution is not shifted relative to the template.
- One realization is less correlated with the reference than either point estimate. This is the distortion–realism
  trade-off stated before the run (§5 of the prereg). No MSE-ratio statement is carried over to correlation.

## 12. Oracle R-location analysis [secondary]

| arm at reference R | F1 | window FD | beat FD | placed beats detected (±50 ms) | matched-pair corr |
|---|---|---|---|---|---|
| O1 template | 0.9975 | 61.90 | 11.33 | 99.98% | 0.7440 |
| O2 deterministic | 0.9668 | 23.02 | 3.96 | 99.5% | 0.7667 |
| O3 stochastic (s = 0) | 0.9090 | 31.54 | 0.82 | 93.3% | 0.6407 |
| O3 − O1 FD | −30.35 [−35.62, −24.77] | | | | |
| O3 − O2 FD | **+8.53 [+5.87, +11.73]** | | | | |

**Rhythm.** Even with perfect positions, the stochastic morphology is detected worse (F1 0.909, 6.7% of placed beats
undetected) than the template (0.998) or the deterministic beat (0.967). The detection loss behind G1 is therefore a
**property of the generated morphology**, not of predicted placement.

**Realism.** At reference R, stochastic beats template but not the deterministic beat. The A3-over-A2 FD advantage
exists only at predicted positions: the deterministic arm's FD drops from 39.86 to 23.02 with oracle placement, while
the stochastic arm moves from 33.39 to 31.54. So G3's advantage over the deterministic beat is not a pure
morphology-model advantage.

**Comparability.** Oracle correlations use identity pairs over all 19,826 reference beats, including beats in windows
RD1 missed. They are not directly comparable with the predicted-position pairs.

## 13. Condition-shuffle analysis [secondary; interpretation, not a gate]

| arm | window FD | FD − A3 | beat FD | D | matched-pair corr − A3 | F1 | HR error (bpm) |
|---|---|---|---|---|---|---|---|
| A3 conditioned | 33.39 | — | 1.58 | 0.324 | — | 0.7506 | 6.39 |
| S-PPG | 36.89 | +3.50 [+1.21, +5.83] | 1.29 | 0.389 | −0.091 [−0.099, −0.084] | 0.7156 | 8.64 |
| S-RR | 36.96 | +3.57 [+2.00, +5.08] | 1.32 | 0.372 | −0.065 [−0.069, −0.060] | 0.7028 | 10.21 |

- Both shuffles worsen window FD, with CIs above 0. The preregistered reading "realism is not PPG-conditioned" is
  therefore **not** triggered for either condition. PPG conditioning and RR conditioning each contribute to
  window-level realism, per-beat correspondence and detected rhythm.
- The beat-level FD is slightly *lower* under both shuffles. The pooled beat distribution does not need the right
  condition; the window and the per-beat correspondence do.

## 14. Secondary morphology / structure metrics [secondary]

| arm | matched-pair corr | detection-based corr | S4 QRS-core deriv. RMSE ↓ | S5 curvature err ↓ | whole-window r | HR error (bpm) |
|---|---|---|---|---|---|---|
| placed positions | — | — | — | — | — | 4.44 |
| A1 template | 0.811 | 0.811 | 0.338 | 0.270 | 0.339 | 4.37 |
| A2 deterministic | 0.820 | 0.817 | 0.344 | 0.275 | 0.350 | 5.49 |
| A3 stochastic (s = 0) | 0.715 | 0.718 | 0.349 | 0.286 | 0.241 | 6.39 |
| A3 16-sample mean | 0.812 | 0.809 | 0.335 | 0.272 | 0.323 | 5.57 |
| iMF single sample | — | 0.631 | 0.382 | 0.354 | 0.230 | 9.58 |
| iMF consensus-decoded (in-sample) | — | 0.788 | 0.331 | 0.279 | 0.358 | 7.07 |

- **S4 / S5:** the prereg expected BF0 to be worse than one iMF sample. It is **better**: A3 0.349 / 0.286 vs iMF
  0.382 / 0.354. The expectation was wrong, in the favorable direction.
- **Join / overlap artifact rate J:** 0.0 for A1–A3 and O1–O3, against a threshold of 1.96 (the 99.9th percentile of
  the reference).
- **Seed diversity:** see §9.
- **HR:** rendering adds HR error in the same order as the F1 loss (template < deterministic < stochastic), which
  matches the extra detections of §7.

## 15. Timing uncertainty diagnostics [secondary]

Matched events: 8,715 in half C (calibration) and 9,070 in half E (evaluation).

| nominal | conformal coverage (half E) | uncalibrated Gaussian | mean width | coverage low / mid / high detector confidence | width low / mid / high |
|---|---|---|---|---|---|
| 0.5 | 0.524 | 0.596 | 33.5 ms | 0.507 / 0.559 / 0.506 | 38.8 / 31.9 / 29.7 ms |
| 0.8 | 0.816 | 0.857 | 66.2 ms | 0.781 / 0.837 / 0.830 | 76.7 / 63.1 / 58.7 ms |
| 0.9 | 0.914 | 0.926 | 89.9 ms | 0.889 / 0.923 / 0.931 | 104.3 / 85.7 / 79.8 ms |

- Split conformal brings marginal coverage to within 0.03 of nominal on held-out patients. Coverage is not a gate.
- Unlike N7's overconfident head on WildPPG, the uncalibrated head here over-covers slightly.
- Intervals widen as detector confidence falls. Coverage is lowest in the low-confidence tertile.
- These intervals describe **timing only**. The morphology sample spread is not calibrated uncertainty and was not
  tested as such.

## 16. Compute / latency / parameters [secondary]

**Parameters:**

- RD1 detector: 328,897.
- Timing head: 132,610.
- Beat model: 482,049 (the deterministic and stochastic models are identical in size).

**Training (RTX 5090):** timing head 3.9 s; each beat model about 100 s.

**Rendering all 4,822 validation windows:**

- A3 with 16 realizations: 22.2 s;
- A2: 0.33 s;
- A1: 0.18 s.

**Batch-1 latency** (50 salted validation windows, 3.84 beats per window on average; detector + timing head + beat
model + overlap-add):

| pipeline | GPU median / p90 | CPU (4 threads) median / p90 | network forward passes per window |
|---|---|---|---|
| template | 0.76 / 0.81 ms | 2.24 / 2.75 ms | detector 1 + head 1 |
| deterministic beat | 1.73 / 2.09 ms | 5.12 / 9.35 ms | + beat model 1 (batched over beats) |
| stochastic beat (one realization) | 7.08 / 7.34 ms | 23.35 / 28.96 ms | + beat model 8 Euler steps (batched over beats) |

## 17. Failed criteria

- **G1, F1 part:**
  - ΔF1 = −0.0284 [−0.0312, −0.0257]; the rule needs a lower bound > −0.02.
  - This decides the verdict: **Case A**.
  - The RR-MAE part of G1 passed.
- **Preregistered expectations that did not hold (not criteria):**
  - "G1 expected to pass": it failed.
  - "S4 / S5 expected to be worse than one iMF sample": they are better.
- **No failed gate was rescued, no margin changed, no seed swapped, no arm rerun.**
- **Post-hoc work:** the only analysis added after the result is `scripts/bf0_posthoc.py`, which reads the frozen
  renders. It ran twice: the second run added the subset bootstrap CIs.

## 18. What BF0 establishes

Within the frozen setup, on VitalDB validation (one seed, one dataset, a previously analyzed split):

1. **[gate G4a]** Per-beat stochastic generation at fixed positions does not move detected R times. The median SD
   across 16 realizations is 0.0 ms, against 18.5 ms for 16 iMF samples.
2. **[gate G1]** The same stochastic morphology does not preserve the detected rhythm within the preregistered margin.
   It creates extra detected peaks and hides some beats: ΔF1 −0.028. It does so even at oracle positions (O3 F1 0.909).
   The rhythm–morphology separation therefore holds for **timing** but not for the **beat set** a detector recovers.
3. **[gate G3]** On the preregistered population, the stochastic beat has lower window FD than the template and the
   deterministic beat, with one waveform per window. The advantage over the deterministic beat is fragile: at oracle
   positions it reverses, and post-hoc, without the flat-filled windows, it vanishes.
4. **[gates G4b, G2]** The stochastic branch did not collapse, and its 16-sample mean has the template's beat
   correlation. Within a window it varies 2.3× more than real ECG.
5. **[secondary]** Split conformal calibration of the timing head gives near-nominal timing-interval coverage on
   held-out patients (0.524 / 0.816 / 0.914).

## 19. What BF0 does NOT establish

- **Rhythm preservation:** rhythm-preserving stochastic beat rendering was not shown. G1 failed.
- **The factorization premise:** not supported as a whole (Case A).
- **Realism relative to other generators:** BF0 is not competitive with a monolithic generator. iMF's window FD is
  3.46 against 33.39 for A3.
- **Robustness of the G3 gain:** the advantage over a deterministic learned beat is not robust to placement or
  population.
- **Realistic variability:** not shown. The stochastic arm over-disperses beats within a window.
- **Uncertainty:** the morphology sample spread was not tested as calibrated uncertainty.
- **Standing claim boundaries** (prereg §4.5), not addressed by BF0:
  - patient-specific ECG morphology, or true individual ST / T / QRS morphology;
  - best HR estimator or best R-peak detector;
  - superiority over other PPG→ECG architectures;
  - novelty as a rhythm-first, residual or beat-level generator;
  - any general conditional-generation result;
  - clinical validity.
- **Scope:** nothing here is a test-set result, a multi-seed result, or a result on another dataset.

## 20. BF1 go/no-go recommendation

**NO-GO.** Under the frozen verdict, Case A means no BF1. No BF1 is designed, preregistered or started here. The
work stops after this report (prereg §4.3).

---

Figures:

- `artifacts/bf0_beat_first/figure.png`, panels A–E. In D1 the BF0 bar is 0.0 ms, so it is invisible.
- `artifacts/bf0_beat_first/atlas.png`: six salted validation windows.

Artifact index: prereg §8, plus `posthoc.json`.
