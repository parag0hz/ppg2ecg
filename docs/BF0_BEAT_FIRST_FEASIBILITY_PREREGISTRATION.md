# BF0 — Beat-First Generator: Feasibility on VitalDB Validation — PREREGISTRATION (revision 2, frozen)

**Status: FROZEN (2026-09-30).** This file is committed and pushed together with the BF0 implementation **before** any
BF0 weight update, before any BF0 code reads a validation window, and before any BF0 metric is computed. After that
commit it is never edited; a change needs a separate dated amendment. The pre-commit fidelity audit against the
execution specification is `artifacts/bf0_beat_first/audit.md`; sha256 hashes of this file and of the BF0 code are in
`artifacts/bf0_beat_first/prereg_manifest.json`.

Design: `docs/BEAT_FIRST_ARCHITECTURE_DESIGN_KO.md` (working name "Beat-First"; not an established term).
Revision 2 incorporates an external review (2026-09-30): accuracy is no longer a success axis (N2 closed it); the
gates test rhythm preservation, centre consistency, distributional gain and whether randomness acts on morphology
rather than rhythm.

**Question BF0 answers:** can we fix the rhythm explicitly, randomize only beat morphology, and improve waveform
distributional realism without losing rhythm fidelity?

Central architectural hypothesis: *do not ask stochastic generation to reproduce structure that is already
identifiable from the condition.* In BF0, R-peak timing is treated as condition-identifiable structure, morphology as
substantially ambiguous; morphology randomness is generated per beat and the renderer places beats at fixed predicted
event locations.

## 한국어 요약

- **질문:** 리듬은 고정하고 형태만 무작위로 생성해서, 리듬 정확도를 잃지 않고 파형 현실성을 얻을 수 있는가?
- **구조:** 기존 PPG 검출기(RD1)가 비트 위치를 정하고, N5식 헤드가 위치 보정량과 불확실성을 낸다. 세 가지 형태 모델이
  모두 같은 위치에 비트를 놓는다.
  - A1 학습셋 중앙값 템플릿
  - A2 결정적 학습 비트: 같은 네트워크를 L1로 학습
  - A3 확률 비트: flow matching, 노이즈 realization 16개 (0번이 주 렌더)
- **오라클 arm:** 세 가지를 정답 R 위치에 놓은 버전 (진단용)
- **대조군:** 확률 비트의 PPG 조건 또는 RR 조건을 다른 환자 것으로 바꾼 버전 (진단용)
- **게이트:**
  - G1 리듬 보존: 렌더의 F1·RR-MAE가 놓은 위치 자체 대비 비열등
  - G3 분포 이득: realization 0 렌더의 FD가 템플릿·결정적 비트보다 낮음. 모든 arm이 창당 파형 하나로 표본 수가 같음
  - G4b 붕괴 없음: 같은 비트 위치에서 잰 창 안 박동 간 다양성이 결정적 비트보다 큼
  - G4a 타이밍 불변: realization 간 R 시각 SD ≤ 1샘플
  - G2 중심: 16개 평균의 박동 정렬 상관이 템플릿 대비 비열등 (같은 박동 쌍에서 비교)
- **판정 순서:** G1 → G3 → G4b → G4a → G2. 처음 실패한 게이트가 Case A–E를 정하고, 모두 통과하면 Case F다.
  어떤 경우에도 BF0 보고서 뒤에서 멈추며 BF1은 자동으로 시작하지 않는다.
- **정확도는 성공 축이 아니다.** 한 realization의 형태 상관은 확률 비트에서 낮아질 것으로 예상하고, 보고만 한다.

---

## 0. Scope and non-goals

- Development / feasibility stage. One training seed (42). VitalDB **train** split for every trained object,
  VitalDB **validation** split for conformal calibration and for every BF0 number. **The test split is never loaded**
  (`load_role` refuses it in every entry point).
- No external baseline, no retraining of any existing generator, no claim beyond the gate verdict. BF0 is not
  confirmatory evidence; BF1 (if any) is preregistered separately and never starts automatically (§4.3).
- Not in BF0: patient-specific morphology reconstruction, per-window adaptive inference, residual generation on a
  scaffold (residual flow), timing-tangent projection or any derivative projection, phase warping (R-to-R phase
  normalization, RR-dependent time stretching, O2c/O3-style warps), end-to-end rhythm/morphology fine-tuning.
- Standing rules: `external/*` untouched; no architecture, loss, optimizer or seed change after this document is
  committed; nothing is tuned on validation except the conformal quantile (§2.3), which is the method; no checkpoint
  selection (last step of every model).

## 1. Data

| item | value |
|---|---|
| split manifest | `data/manifests/split_v1_vitaldb_seed42.json` (V1 patient holdout): train 4,517 cases, val 302 cases, test 1,224 cases (never loaded) |
| windows | 4 s at 128 Hz (512 samples), PPG `x` and ECG `y` in [−1, 1], loaded with `scripts/vm1_evaluate.load(role)`; the patient is the file's `subjectid` |
| validation | 4,822 windows (the ED1 cache order) |
| reference R peaks | `ppg2ecg.evaluation.rpeaks.detect_rpeaks(y, 128)` (neurokit), the T3 / RD1 convention. Train-split peaks are read from RD1's cache `outputs/rd1_detector/train_rpeaks.npz` after recomputing 200 salted train windows (`"bf0-peakcheck-v1"`) and asserting equality; validation peaks are computed |
| conformal halves | validation patients split 50 / 50 by `sha256("bf0-conformal-v1" + subject id)` parity: half C (calibration), half E (evaluation) |

**Validation reuse (stated honestly).** The VitalDB validation split is **not fresh**: ED1 chose its decoding
parameters on it, FBC1 drew its primary evidence from it, and M1 built its validation bank from it. No BF0 component
is selected on it (no checkpoint selection, no hyperparameter choice), but BF0 numbers come from a population this
project has analyzed before, and nothing in BF0 is a held-out test result.

## 2. Components (frozen)

### 2.1 Event stage
The existing RD1 detector: `outputs/rd1_detector/checkpoint_last.pt` (RhythmTCN, 328,897 parameters, trained on the
VitalDB train split, file sha256 prefix `084a316fa663f9dc`, full hash recorded at run time). Events
`e_i = extract_events(sigmoid(logits), threshold 0.35, refractory 32)`, exactly RD1's rule. Frozen, `eval()`,
`requires_grad = False`.

### 2.2 Timing head
N5's `Head` (MLP, 2 hidden layers of 256, GELU) predicting `(μ_i, log σ_i)` of the residual
`r_i = (nearest reference R within ±150 ms) − e_i` in ms.
- Features: detector probability field `e_i ± 32` samples and PPG `e_i ± 96` samples (edge-padded with the edge value).
- Training: every RD1 event on the **train** split that matches a reference R within ±150 ms. Gaussian NLL with
  σ clamped to [1, 500] ms. 6,000 steps, batch 256 (indices drawn with replacement), AdamW lr 1e-3, weight decay
  0.01, seed 42. Last step kept.

### 2.3 Conformal calibration (secondary output)
Score `s = |r − μ| / σ` on half C's matched events. For α ∈ {0.5, 0.8, 0.9}, `q_α` = the ⌈(n+1)α⌉-th smallest score.
Interval `μ ± q_α σ`. Split conformal fixes marginal coverage by construction, so coverage is **not** a gate; interval
width and coverage by detector-confidence tertile (the signal-quality stratification of this draft) are the
informative quantities. The uncalibrated Gaussian interval `μ ± z_{(1+α)/2} σ` is reported for reference.

### 2.4 Beat models (one network architecture, three morphology sources)
- **Segment:** R-aligned, 64 samples before R (0.50 s) and 101 after (0.79 s): length 166, R at index 64.
- **Network (`BeatFlowNet`, ≈ 0.48 M parameters, identical for A2 and A3):**
  - input channels `[x_t, PPG segment]` (the PPG over the same 166 samples)
  - 1×1 stem to 64 channels
  - 8 residual blocks of two dilated convs (kernel 5, dilations 1, 2, 4, 8, 16, 32, 1, 2, GELU)
  - FiLM in every block (zero-initialised) from an MLP on `[sinusoidal(t) (32 dims), RR_prev, RR_next]` (seconds)
  - 1×1 head
- **Training beats:** every reference R on a train window whose segment lies fully inside the window. RR_prev /
  RR_next come from the neighbouring reference R. If a neighbour is outside the window, the window's median reference
  RR is used; beats in windows with fewer than 2 reference R are skipped.
- **Template (A1):** the sample-wise median of the training segments.
- **Deterministic learned beat (A2):** `BeatFlowNet` called with `x_t = 0` and `t = 0`, trained with L1 to the
  target segment. 20,000 steps, batch 256, AdamW lr 1e-3, weight decay 0.01, gradient clip 1.0, seed 42 (network
  initialisation and minibatch generator). Last step kept.
- **Stochastic beat (A3):** `BeatFlowNet` trained with OT-CFM (`x_t = (1 − t) x_0 + t x_1`, `x_0 ~ N(0, I)`, target
  `x_1 − x_0`, `t ~ U(0, 1)`), same data, optimiser settings, steps and seed. Sampler: Euler, 8 steps (fixed a
  priori). Last step kept.
- **Training fairness:** A2 and A3 have the same architecture and parameter count (asserted in the compute record),
  the same training beats, optimiser, step count and seed; they differ only in the loss (L1 on a deterministic call
  vs OT-CFM). Neither is tuned after seeing the other's results.

### 2.5 Seed roles (frozen)
- Realization s ∈ {0, …, 15} of beat j in window n uses `torch.Generator().manual_seed(1_000_003·n + 10_007·s + j)`.
- **s = 0 is the primary render:** G3 FD, every single-waveform metric, G1, G4b, and the noise of O3, S-PPG and S-RR.
- **s = 0 … 15:** the G2 16-sample mean, the G4a timing SD, and seed-to-seed morphology diversity.
- Never: choosing a best-looking realization; swapping the primary realization; averaging the 16 realizations for FD.

### 2.6 Renderer (all arms)
- **Positions (predicted arms):** `t_i = e_i + round(μ_i · 128 / 1000)`, clipped to [0, 511], sorted. No sampled timing
  jitter enters any waveform arm; σ is used only for the interval diagnostics (§2.3).
- **Positions (oracle arms):** the reference R peaks of the window.
- **Conditioning:** RR_prev / RR_next from consecutive positions. Edge beats use the window's median RR; a window with
  a single position uses the train-split median reference RR. PPG segment `x[t_i − 64 : t_i + 102]`, edge-padded.
- **Assembly:** fixed-time, R-centred weighted overlap-add with Hann weights `w` over the 166-sample segment
  (`numpy.hanning(168)` with its two zero end points removed, so every segment sample has positive weight):
  `y(t) = Σ_i w_i(t) b_i(t) / Σ_i w_i(t)` where `Σ_i w_i(t) > 1e-6`.
  - Uncovered samples take the value of the nearest covered sample
  - A window with no position is filled with the template's baseline value (median of its first and last 8 samples).
    The count is reported
- Because assembly is linear with fixed weights, the pointwise mean of the 16 A3 renders equals the render of the
  16-sample mean beats (used by G2).

### 2.7 Arms and controls

The execution specification's arms A (template), B (deterministic learned beat) and C (stochastic beat) are A1, A2
and A3 here. **A1, A2 and A3 use the same frozen predicted positions.**

| arm | positions | morphology |
|---|---|---|
| A1 template | predicted | fixed train-median segment (the N1-matched baseline) |
| A2 deterministic | predicted | deterministic learned beat |
| A3 stochastic | predicted | stochastic beat, realizations 0–15 (0 = primary) |
| A3-mean | predicted | pointwise mean of the 16 A3 renders |
| O1 / O2 / O3 (diagnostic) | reference R | template / deterministic / stochastic (realization 0), the same trained models |
| S-PPG control (diagnostic) | predicted | A3 realization 0 with each beat's PPG segment taken from a beat of another patient (RR unchanged) |
| S-RR control (diagnostic) | predicted | A3 realization 0 with each beat's (RR_prev, RR_next) taken from a beat of another patient (PPG unchanged) |

Oracle arms use reference R positions only at render and evaluation time; no model is trained, selected or tuned on
them. Shuffle rule (one donor list for both controls): all predicted-position beats of the validation split are
ordered by `sha256("bf0-shuffle-v1:" + global beat index)`; beat k receives the condition of the beat M/2 places later
in that order (cyclically), moving forward one place at a time until the donor belongs to a different patient. The
donor list and its sha256 are stored (`shuffle_manifest.json`).

Comparators on the same windows: iMF single sample (`outputs/ed1_cache/samples_I_val.npy`, draw 0), iMF 16 samples
(same cache, draws 0–15, for G4a and seed-to-seed diversity), iMF consensus-decoded (ED1 frozen parameters; chosen on
this validation split, so an in-sample comparator), RD1 events and corrected positions scored as peak sequences.

## 3. Metrics

| metric | definition |
|---|---|
| R-peak F1, RR-MAE, HR error | the `paper_metric_table` columns `rpeak_f1_50ms`, `rr_mae_ms`, `hr_abs_err` (`rpeak_prf_at` / `beat_level_metrics`, neurokit detection on each waveform, ±50 ms, greedy one-to-one matching). F1 is averaged over **evaluable windows** (reference has ≥ 1 beat; the project's KANFlow Eq. 26–27 convention), the same windows for every arm. RR-MAE is finite only where ≥ 2 consecutive reference beats are matched |
| beat-aligned morphology correlation (primary definition, used by G2) | **matched pairs:** reference R and placed positions matched one-to-one within ±50 ms (`rpeaks.match_rpeaks`); a pair is kept when both 83-sample windows (R − 32 … R + 50, i.e. R − 0.25 s … R + 0.40 s) lie inside the window. Pearson correlation between the reference ECG window at R and the arm's window at its placed position; mean over the window's pairs. The pairs depend only on (reference, positions), so every arm sharing the positions shares them. Oracle arms use identity pairs (R, R) |
| beat-aligned correlation on detections (secondary) | `rpeaks.morphology_corr` on each arm's own neurokit detections (the draft's earlier definition; its population differs between arms, so it is reported, never gated) |
| FD | `paper_metrics.kanflow_fd(waveforms, reference)` over all 4,822 validation windows (raw-waveform regime, n ≥ 3,000), **one waveform per window in every arm** (A3 uses realization 0 only) |
| within-window beat diversity D | for a waveform and its beat positions, the mean pairwise RMS between the 83-sample beat windows that lie fully inside the window (windows with ≥ 2 such beats). Real: reference ECG at reference R. Arms: the render at its placed positions, so A1, A2, A3 use identical beat sets |
| timing SD across realizations | for each reference R, the detections (neurokit) matched within ±50 ms (greedy one-to-one, `rpeaks.match_rpeaks`) in each of the 16 realizations; sample SD (ddof = 1) of the matched times (ms) for beats matched in ≥ 8 of 16; median over beats. Computed identically for A3 (renders) and for the 16 iMF samples |
| seed-to-seed morphology diversity | mean pairwise RMS across the 16 realizations of the same 83-sample window: A3 at its placed positions; iMF at reference R |
| beat-level FD | `paper_metrics.fid_frechet` between the 83-sample beat windows of an arm (at its placed positions) and those of the reference (at reference R), all beats fully inside their window |
| R-location drift | for A1–A3 (and A3 over all 16 realizations) and O1–O3: neurokit detections matched to placed positions within ±50 ms; fraction detected, median \|offset\|, mean offset, fraction within one sample |
| coverage (secondary) | half E matched events: fraction with the reference R inside `μ ± q_α σ`; mean width; coverage and width by detector-confidence tertile |
| other secondary | whole-window correlation (`pcc`), structure S4 / S5 (`m1_structural.qrs_core_morphology`: `qrs_deriv_rmse`, `qrs_curvature_err`), boundary (join / overlap) artifact rate J (fraction of windows whose maximum \|first difference\| outside ±12 samples of every placed R exceeds the 99.9th percentile of the same statistic on the reference windows outside ±12 samples of their reference R) |

### 3.1 Statistics
- **Unit: the patient.** Windows are never bootstrapped independently.
- Per-window statistics (and per-window paired differences) are averaged per patient first (equal patient weight, the
  project convention); non-finite windows are skipped. 2,000 patient resamples with replacement, seed 20260930;
  percentile 95% CI. Replicate r draws the same patients for every statistic.
- FD differences: 1,000 patient resamples (seed 20260930, the same draw sequence); each replicate takes every window
  of every drawn patient, applies that window set to both compared arms and the reference, and recomputes FD. Every
  replicate must keep ≥ 3,000 windows (asserted) so `kanflow_fd` never switches to its PCA regime.

### 3.2 Matched populations and missingness
- Every paired gate comparison uses the same patients, windows and beats in both arms:
  - G1: the same windows (evaluable windows for F1; windows where both RR-MAEs are finite for RR-MAE)
  - G3: all 4,822 windows, one waveform each, identical in every arm and every bootstrap replicate
  - G4b: A2 and A3 share positions, so D uses the same beat identities; windows where both are finite
  - G2: the matched pairs of §3, restricted to pairs whose correlation is finite in every arm of the group
    {A1, A2, A3, A3-mean}. Shuffle comparisons use the group {A3, S-PPG, S-RR}; oracle arms the group {O1, O2, O3}
  - G4a: reference beats matched in ≥ 8 of 16 draws, counted separately for BF0 and iMF (both reported)
- Missingness and exclusions are reported per arm and per metric (`missingness.json`): windows without positions,
  windows without reference beats, non-finite windows, paired-window counts, matched-pair counts, beat counts.

## 4. Gates and verdict

### 4.1 Gates

| gate | question | rule |
|---|---|---|
| G1 rhythm preservation | does rendering damage the detector's timing? | A3 primary render vs its own placed positions scored as a peak sequence, same reference: ΔF1 = F1(render) − F1(positions), lower CI bound > −0.02 **and** ΔRR = RR-MAE(render) − RR-MAE(positions), upper CI bound < +2.0 ms |
| G3 distributional gain | is the stochastic branch more realistic than a fixed or deterministic beat? | FD(A3 s = 0) − FD(A1) < 0 **and** FD(A3 s = 0) − FD(A2) < 0, both with bootstrap CI upper bound < 0 |
| G4b non-collapse | does randomness produce morphology variation? | D(A3) − D(A2) with CI lower bound > 0. Practical target, reported and **not** gated: D(A3) / D(real) ≥ 0.25 (same windows). Seed-to-seed diversity is reported separately |
| G4a timing invariance | does randomness leak into timing? | A3 timing SD across realizations ≤ 1 sample at 128 Hz (exactly 7.8125 ms in code; "7.8 ms" in tables); the iMF value is reported beside it |
| G2 centre consistency | is the centre of the stochastic distribution badly shifted? | A3-mean vs A1 (primary comparator, template): matched-pair beat-aligned correlation difference, lower CI bound > −0.02. A2 is a secondary comparator, reported |

Single-realization pointwise morphology is not required to match the template or the deterministic estimator; a
distortion–realism trade-off is expected. No "2× MSE" statement is transferred to correlation.

### 4.2 Verdict

Evaluated in the frozen order **G1 → G3 → G4b → G4a → G2**; the first failing gate decides:

| case | condition | verdict |
|---|---|---|
| A | G1 fails | architecture premise failure: beat morphology rendering destroys rhythm fidelity. NO BF1 |
| B | G1 passes, G3 fails | "BF0 did not demonstrate a distributional-realism advantage for stochastic morphology over deterministic/template rendering." Generative morphology branch unsupported. NO automatic BF1 |
| C | G1, G3 pass; G4b fails | the stochastic branch fails the meaningful-variation criterion; no meaningful stochastic-morphology claim |
| D | G1, G3, G4b pass; G4a fails | stochastic generation has value, but the rhythm–morphology disentanglement claim fails. An event-preserving BF1 may be motivated but is not started |
| E | G1, G3, G4b, G4a pass; G2 fails | distributionally useful and timing-stable but centre-biased; reported with that qualification. A centre-preserving BF1 may be motivated but is not started |
| F | all pass | BF0 supports the factorization premise. BF1 may be justified but is not started |

Every gate is reported separately regardless of the case. No failed gate is rescued with a secondary metric, and no
overall success definition is added after results.

### 4.3 Hard stop (no BF1)
After the BF0 report and result commit the work stops, whatever the case. Not done after BF0: starting BF1, tangent
projection, a phase renderer, residual flow, end-to-end fine-tuning, a new loss, margin changes, reruns with other
seeds, swapping the primary realization, added datasets, a larger network.

### 4.4 Mandatory diagnostics (not gates)
- **Condition shuffles.** FD (with patient-bootstrap CI of the difference), D, beat-level FD, matched-pair correlation
  and F1 of S-PPG and S-RR vs A3.
  - If FD(S-PPG) − FD(A3) has a CI that includes 0 or lies below 0, the report states that morphology realism is not
    demonstrably PPG-morphology-conditioned. The same reading is applied to S-RR for rhythm conditioning; if both are
    unchanged, realism may largely come from an unconditional population beat prior. These are interpretations, not
    gates
- **Oracle arms.** O1–O3 beside A1–A3 separate a morphology-model failure (no FD gain even at reference R) from a
  rhythm-placement bottleneck (gain at reference R that disappears at predicted positions).
- **R-location drift** caused by rendering (§3).

Also reported without gating: A3 single-realization matched-pair correlation (expected to be lower than A1 and A2),
detection-based correlation, HR error, whole-window correlation, S4, S5, coverage, interval width, artifact rate J,
seed-to-seed diversity (A3 and iMF), iMF comparators, number of windows without positions.

### 4.5 Claim boundaries
- The morphology sample spread is **not** calibrated uncertainty. Timing uncertainty comes from the timing head and
  the conformal step only; coverage is not a success gate.
- Even if every gate passes, BF0 does not show: patient-specific ECG morphology reconstruction; recovery of true
  individual ST / T / QRS morphology from PPG; best HR estimator; best R-peak detector; superiority over all PPG→ECG
  architectures; a first rhythm-first, residual or beat-level generator; a general conditional-generation theorem;
  clinical diagnostic validity.
- Permitted framing if supported: explicit rhythm/morphology factorization; rhythm-preserving stochastic beat
  rendering; improved distributional waveform realism vs template / deterministic beat baselines; randomness
  concentrated in morphology rather than rhythm; timing uncertainty separately calibrated by the timing head and
  conformal step; scope restricted to VitalDB validation and the models evaluated.
- The report separates preregistered gate results, preregistered secondary results and post-hoc observations.

## 5. Expectations stated before any number

- **G1:** expected to pass. The renderer places R-aligned beats at fixed positions.
- **G3:** the uncertain gate. A template and a deterministic beat carry little beat-to-beat variability, so their
  window covariance should be too small, but generated beats may add artifacts.
- **G4a:** expected to pass largely by construction (fixed positions, R-aligned training). Its value is the side-by-side
  contrast with the iMF samples, whose timing varies across samples.
- **Single-realization correlation:** expected to be lower for A3 than for A1 / A2. Distributional realism and
  pointwise accuracy are expected to trade off.
- **S-PPG:** N2 predicts that PPG conditioning adds little morphology information, so FD(S-PPG) ≈ FD(A3) is the
  expected outcome.
- **S4 / S5:** expected to be worse than the iMF single sample (N1 report §3.3).

## 6. Implementation and tests (written before the commit; no real-data number)

| file | content |
|---|---|
| `src/ppg2ecg/beatfirst/beats.py` | segment geometry, training-beat extraction, RR helpers |
| `src/ppg2ecg/beatfirst/model.py` | `BeatFlowNet`, OT-CFM loss, L1 deterministic loss, Euler sampler |
| `src/ppg2ecg/beatfirst/timing.py` | timing features, head, conformal quantiles, coverage |
| `src/ppg2ecg/beatfirst/render.py` | Hann overlap-add assembly, gap filling, template rendering, diversity, matched pairs, boundary statistic |
| `scripts/bf0_run.py` | stages `manifest`, `audit`, `train_timing`, `train_beat`, `render`, `evaluate`, `atlas`, `compute`, `figure`; gate rules (`compute_gates`), verdict (`verdict_case`); the test split can never be loaded |
| `tests/test_bf0_beatfirst.py` | synthetic-data tests only |

Test obligations (all synthetic):
- overlap-add reconstructs a signal exactly when each beat is its own true slice; the mean of renders equals the
  render of mean beats; segment geometry and edge padding
- conformal coverage on synthetic Gaussian residuals
- model and sampler shapes and determinism; the deterministic arm ignores `x_t` and `t`; realization s uses the frozen
  noise seeds (seed roles)
- the shuffle never pairs a beat with its own patient; the donors match a hand computation of the frozen rule; S-PPG
  and S-RR replace only their own condition
- the primary FD waveform is realization 0 and every arm contributes the same number of windows; the FD bootstrap
  applies one window set to both arms and refuses a regime switch
- G4a timing SD; G4b diversity (0 for identical beats, same beat set in both arms)
- patient bootstrap takes whole patients; equal patient weight
- matched pairs depend only on (reference, positions) and drop edge, missed and extra beats; pair means use only pairs
  finite in every arm
- gate rules at their margins; the verdict follows the frozen order
- training stages never read the validation split or oracle positions; the test role cannot be loaded

## 7. Run order

1. `manifest` (hashes only), then commit and push this document and the implementation (no number exists).
2. `audit`: window, patient and beat counts; input hashes.
3. `train_timing`, `train_beat` (A2 and A3). The GPU is checked idle with `nvidia-smi` first.
4. `render`, `evaluate`, `atlas` on validation.
5. `compute` (latency on an idle GPU), `figure`.
6. Report `docs/BF0_BEAT_FIRST_FEASIBILITY_REPORT.md`, then commit and push. **HARD STOP** (§4.3).

Each stage runs once. A crash is fixed only if the fix leaves every frozen definition unchanged, and is recorded as a
dated deviation in the report.

## 8. Artifacts

`artifacts/bf0_beat_first/` (committed, small): `audit.md` (pre-commit fidelity audit), `prereg_manifest.json`,
`audit.json`, `input_hashes.json`, `train_config.json`, `checkpoint_{timing_head,deterministic,stochastic}.json`,
`render_manifest.json`, `seed_manifest.json`, `shuffle_manifest.json`, `gate_metrics.json`, `bootstrap.json`,
`oracle_metrics.json`, `shuffle_metrics.json`, `diversity_metrics.json`, `timing_metrics.json`,
`waveform_metrics.json`, `missingness.json`, `compute_accounting.json`, `result.json`, `figure.png`, `atlas.png`.
`outputs/bf0_beat_first/` (never committed): checkpoints, renders, cached validation peaks.

## 9. What BF0 cannot show

It uses one seed, no test data and no external baseline, on a validation split analyzed before (§1). Realism is
measured by FD and beat statistics only, and clinical usefulness is not assessed.
