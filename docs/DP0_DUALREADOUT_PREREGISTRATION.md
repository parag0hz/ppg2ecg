# DP0 — DualReadout-ECG — PREREGISTRATION (frozen)

**Status: FROZEN (2026-10-02).** Committed and pushed together with the split, the design artifacts and the
implementation **before any DP0 model is trained and before any DP-DEV outcome exists**. It is never edited afterwards;
deviations are dated amendments. Hashes: `artifacts/dp0_dualreadout/prereg_manifest.json`.

## 0. What DP0 is

- **A new architecture line:** DualReadout-ECG (internal alias DualPath-ECG). One shared physiological conditioning
  representation H = E_shared(PPG, event raster) feeds **two task-specific readouts**:
  - a deterministic point readout, μ(c) = D_point(H);
  - a flow-matching generative readout, v_θ(x_t, t, H) = D_flow(x_t, t, H).
- **The two waveform outputs are never added.** The point readout is used when one paired representative ECG is
  needed; the generative readout is used when conditional ECG samples are needed.
- **This is not AnchorFlow:** there is no x = μ + residual formulation and no residual path anywhere in DP0.
- **The question:** can one shared cross-modal representation keep both specialist abilities (paired deterministic
  estimation and conditional generation) while using at least 15 % fewer waveform-model parameters than two independent
  specialists?
- **Closed lines are untouched:** BF0, D0, C0, C0-A, E0, R1, SF0, AF0. Their reports, checkpoints, metrics and verdicts
  are not modified. DP0 only reads AF0's split manifest (patient lists) and reuses library code by import
  (`scaleflow.model`, `anchorflow.fastfd`, evaluation helpers) without changing it.
- **No novelty claim** (first dual-head ECG model, first point / generative multitask model, novel, state of the art)
  without a separate current literature review.

## 1. Data (`split_manifest.json`, `split_hashes.json`; created before this commit)

- **Source:** the former AF-TRAIN (2,400 patients; `artifacts/af0_anchorflow/split_manifest.json`). AF-DEV, which
  AF0 analysed adaptively, is **never used** in DP0.
- **Rule:** sorted AF-TRAIN patients permuted with `default_rng(20261002)`; [0:300] → **DP-DEV**, the rest →
  **DP-TRAIN**.

| role | patients | windows | use |
|---|---|---|---|
| DP-TRAIN | 2,100 | 139,962 | training of the detector, both specialists and the dual models |
| DP-DEV | 300 | 19,583 | development evaluation and candidate selection |
| AF-LOCK | 337 | 22,133 | sealed confirmation set (AF0's, never opened by AF0); opened once after a committed freeze |
| old V1 TEST | 1,156 | — | **closed for all of DP0** |

- **Checks (all pass):**
  - DP-TRAIN and DP-DEV are disjoint and their union is AF-TRAIN;
  - no AF-DEV or AF-LOCK patient;
  - no ARCH-VAL / ARCH-HOLDOUT, SF-VAL, or old V1 validation / test patient.
- **Evidence status:** DP-DEV is a new development evaluation population for DP0, but its patients trained earlier
  project models (C0 / C0-A / R1 / SF0 / AF0). It is **not project-naive, not globally untouched and not external**.
  AF-LOCK was untouched by AF0 and stays untouched by DP0 development.
- **Reference R:** the RD1 neurokit cache (re-verified on salted windows at load time).

## 2. Timing condition

- **Detector:** RD1 / C0 RhythmTCN protocol (14,000 × 64, AdamW 1e-3 / 0.01, BCE to a Gaussian σ = 20 ms target),
  trained on **DP-TRAIN only**, seed 42, then frozen.
- **Events:** threshold 0.35, refractory 32; one event raster (Gaussian, σ = 20 ms) per window.
- **Same raster for every model:** at evaluation, every model (specialists and dual models, point and generative
  readouts, conditioned and PPG-shuffled runs) receives the identical raster computed once per window from the frozen
  detector.
- **Training raster:** the reference-R raster, as in the SF0 / AF0 WW-L1 and ScaleFlow protocols, for every model.
- The detector is not changed during DP0.

## 3. Specialist reference models (trained from scratch on DP-TRAIN; no old checkpoint is reused)

| model | architecture | params | objective | protocol |
|---|---|---|---|---|
| SPECIALIST P | SF0 / AF0 WW-L1 = C0-A `WWDet(72, 5)`: C0 encoder on PPG, raster concatenated at the decoder input | 593,577 | L1 | 20,000 × 64, AdamW 1e-3 / wd 0.01, clip 1.0, constant LR, seed 42, last step |
| SPECIALIST G | SF0 SCALEFLOW-COUPLED = `ScaleFM(50, coupled=True)`: fixed two-level Haar, coarse → mid → fine coupling | 598,333 | flow matching: x0 ~ N(0, I), x_t = (1 − t) x0 + t x1, u = x1 − x0, MSE | same; Euler NFE 8 at inference |

- **SEPARATE-PIPELINE:** SPECIALIST P + SPECIALIST G with the one common detector.
  - P_point = 593,577; P_gen = 598,333; **P_sep = 1,191,910** waveform parameters.
  - The detector (328,897) is common to every pipeline and is **not** counted in any ratio.
- **Specialist reproduction sanity (§39 of the spec):** on DP-DEV, check that P has better paired morphology and event
  behaviour than G's single samples, and that G has the lower FD. Historical numbers are not imported as thresholds.
  A reversed pattern is documented, and the study then continues by this preregistration.

## 4. Conditioning-path audit (`audit.md`)

- **WW-L1:** PPG passes through the C0 encoder (stem 1 × 1, then eight residual blocks of 64 channels, kernel 5,
  dilations 1, 2, 4, 8, 16, 32, 1, 2) at 512 samples. The event raster is concatenated after the encoder.
- **ScaleFlow:** there is no learned condition encoder. Haar(PPG) and Haar(raster) are raw channels at every branch
  input, entangled with Haar(x_t) from the first 1 × 1 stem.
- **Consequence:** the only learned PPG conditioning path in either specialist is the WW-L1 encoder family. It becomes
  the shared conditioning encoder, with the raster moved into its input.

## 5. DualReadout architecture (`src/ppg2ecg/dualreadout/model.py`; `sharing_graph.json`)

- **Shared conditioning encoder E:**
  - input [PPG, event raster] (2 channels) → stem 1 × 1 (2 → 64) → eight C0 residual blocks (64 channels, kernel 5,
    dilations 1, 2, 4, 8, 16, 32, 1, 2);
  - **the noisy ECG x_t never enters E.**
- **Multiscale features:** the closest exact resolutions the repository supports, using the fixed Haar low-pass
  (no parameters):
  - H_512 is the encoder or tower output;
  - H_256 = Haar low1(H_512);
  - H_128 = Haar low1(H_256).
- **Natural encoder stages:**

  | stage | contents | role |
  |---|---|---|
  | E0 | stem 1 × 1 | |
  | E1 | blocks 1–6 | first dilation cycle 1 … 32; the receptive field grows from 1 to 505 samples, i.e. the whole window |
  | E2 | blocks 7–8 | second cycle 1, 2; refinement at full context |

  These are the only natural boundaries; a half-depth split after block 4 would fall inside the first dilation cycle.
- **Adapters:**
  - one point adapter and one flow adapter, each 1 × 1 conv (64 → 64) + GELU, with no attention;
  - placed at the output of the last shared stage;
  - the same design in every pattern: 4,160 parameters each, 8,320 total (< 1 % of every dual model, < 5 % required).
- **Point readout:** point adapter → (point tower) → D_point(w_p) on H_512, where D_point = 1 × 1 (64 → w_p) → five C0
  residual blocks (dilations 1, 2, 4, 8, 16) → 1 × 1 → μ (512). It takes no noise and no time input.
- **Generative readout:** flow adapter → (flow tower) → (H_256, H_128) → D_flow(w_f), the SCALEFLOW-COUPLED vector
  field with the condition channels replaced by the shared features:
  - coarse branch input [Haar_c(x_t), H_128];
  - mid branch input [Haar_m(x_t), H_128, proj_cm(h_c)];
  - fine branch input [Haar_f(x_t), H_256, up2(proj_mf(h_m)), up2(proj_cf(h_c))];
  - six time-conditioned residual blocks per branch, with the SF0 time embedding;
  - the inverse Haar gives one whole-window velocity;
  - samples are generated directly from noise. They are **not** μ + residual.
- **Widths (by parameter count only, frozen now):** each readout path (encoder + adapter + decoder) is matched to its
  specialist (grid 16 … 160, closest count, ties → smaller).
  - w_p = 71 → point path 590,503 (−0.52 % vs P).
  - w_f = 30 → flow path 599,589 (+0.21 % vs G).
  - Because private towers are fresh copies of the blocks they replace, **every readout path has the same architecture
    and parameter count in S0, S1 and S2**. Only the sharing differs.

## 6. Sharing patterns (frozen ownership; `sharing_graph.json`, `parameter_accounting.json`)

| pattern | shared | private per task | shared params | point-private | flow-private | total | saving vs P_sep | E1 (≤ 0.85 · P_sep = 1,013,123.5) |
|---|---|---|---|---|---|---|---|---|
| S0 FULL | E0 + E1 + E2 | adapter + decoder | 328,896 | 261,607 | 270,693 | 861,196 | 27.75 % | PASS |
| S1 MIDDLE | E0 + E1 | adapter + blocks 7–8 + decoder | 246,720 | 343,783 | 352,869 | 943,372 | 20.85 % | PASS |
| S2 STEM-ONLY | E0 | adapter + blocks 1–8 + decoder | 192 | 590,311 | 599,397 | 1,189,900 | 0.17 % | **FAIL by construction** |

- **Ownership:** parameters under `stem` and `shared_blocks` are shared. `point_adapter`, `point_tower` and
  `point_dec` are point-private. `flow_adapter`, `flow_tower` and `flow_dec` are flow-private.
- **E1 is structural and known now.** With matched per-task path capacity, the saving equals the shared fraction.
- **S2 cannot qualify.** Stem-only sharing shares one 1 × 1 convolution, so S2 is effectively two path-matched
  networks trained jointly.
  - S2 is still trained and evaluated as preregistered, as the minimal-sharing control: same paths, same round-robin
    training, no meaningful sharing.
  - P1 / P2 / G1 differences between S0 / S1 and S2 isolate the effect of sharing from the effects of joint training,
    the raster placement and the adapters.
- **No further sharing patterns or depths are added** after outcomes, under any result.

## 7. Round-robin multitask training (`training_manifest.json`)

- **One AdamW** over all parameters (lr 1e-3, weight decay 0.01, constant LR), seed 42, reference-R raster.
- **20,000 cycles.** Cycle c (1-indexed) runs POINT → FLOW when c is odd and FLOW → POINT when c is even. This gives
  exactly **20,000 POINT and 20,000 FLOW updates**, the same per-task update count as each specialist.
- **One substep:**
  1. `zero_grad(set_to_none=True)`;
  2. the task loss (POINT: L1(μ, ECG); FLOW: the SF0 flow-matching MSE) is backpropagated;
  3. the other task's private parameters must have grad None. This is checked, and a violation stops training. AdamW
     skips them entirely (no moment update, no weight decay);
  4. `clip_grad_norm_` at 1.0 over the shared and this task's private parameters;
  5. `step`.
- **Resulting update counts:** shared parameters receive 40,000 updates; each private parameter receives 20,000.
- **Data order:**
  - the POINT stream draws exactly SPECIALIST P's batch sequence (`Generator(42)`);
  - the FLOW stream draws exactly SPECIALIST G's batch sequence and FM noise / t sequence (`Generator(42)`,
    `Generator(cuda, 42)`, draw order x0 then t);
  - consequently, in cycle c both substeps see the same window indices.
- **Excluded:** loss weighting (λ), adaptive task weighting, GradNorm, PCGrad, gradient surgery, early stopping.
  The last cycle is kept.
- **Checkpoints:** cycle 0 (initialization), cycle 5,000 (25 %) and cycle 20,000 (final), for the gradient diagnostic.
- **Seeds:** seed 42 for P, G and every dual candidate. There are no extra seeds for selection.

## 8. Evaluation on DP-DEV (after this commit only; `stage eval_dev` refuses an uncommitted preregistration)

### 8.1 Point readouts (SPECIALIST P, S0 / S1 / S2 point heads)

- **Beat-aligned morphology correlation:**
  - per window, the mean Pearson correlation over the (reference R, detector event) pairs (`matched_pairs`, ±50 ms,
    83-sample beat windows);
  - aggregated as an equal-patient-weight mean.
- **Events:**
  - neurokit R detection on the output, matched ±50 ms to the reference R;
  - patient-pooled FP / window, recall, precision and F1 (equal patient weight);
  - pooled precision / recall / F1.
- **Other:** RR-MAE (ms), HR-MAE (bpm) and MAE.
- **FD:** descriptive only.

### 8.2 Generative readouts (SPECIALIST G, S0 / S1 / S2 generative heads)

- **Noise:** one sample per window, Euler NFE 8, with the identical deterministic noise for every arm:
  `scaleflow.window_noise`, seed = sha256("patient:window:20261002").
- **FD:** KANFlow Gaussian Fréchet distance on raw 512-sample windows (≥ 3,000 windows; ddof 1 + 1e-4 · I).
- **Descriptive:**
  - diversity ratio = RMS(x_gen − μ_P) / RMS(ECG − μ_P), where μ_P is SPECIALIST P;
  - the population SD ratio;
  - morphology correlation, FP and event metrics of the single sample;
  - the AF0 spectral discrepancy (mean |log PSD ratio|).

### 8.3 PPG condition-use control

- PPG is shuffled across DP-DEV windows with `default_rng(20261002).permutation`.
- The event raster, noise, model and every other input are kept.
- For dual models, the shared representation is recomputed from the shuffled PPG.
- **Reported:** FD_shuffled − FD_conditioned and corr_shuffled − corr_conditioned. Computed for every generative head,
  including G (descriptive).

### 8.4 K16 characterization (descriptive, never a gate)

- **Subset:** a fixed 2,000 DP-DEV windows (salted rank `dp0-k16-v1`).
- **Arms:** G and every dual generative head; K = 16 samples per window (noise key ":k{k}" for k > 0).
- **Reported:**
  - within-condition waveform diversity (mean pairwise RMS) and its ratio to √2 · RMS(ECG − μ_P);
  - beat-aligned diversity;
  - generated / real within-window beat-diversity ratio (SF0 definition);
  - seed-to-seed R-time SD;
  - K16 HR consensus MAE vs single-sample HR MAE;
  - the mean generated waveform (descriptive);
  - feature spread.
- **The sample mean is not required to equal the point head.**

### 8.5 Diagnostics (descriptive, never gates)

- **Gradient interaction:**
  - 512 fixed DP-TRAIN minibatches of 64 (`default_rng(20261002)`); minibatch i uses CPU `Generator(20261002 + i)`
    for x0, then t;
  - cosine between ∇_shared L_point and ∇_shared L_FM at initialization, after 25 % (cycle 5,000) and after 100 %;
  - reported as median, IQR and the fraction below 0, per pattern.
- **Features on the K16 subset:**
  - shared-output RMS and channel variance;
  - point and flow adapter output norms;
  - tower outputs.
  - CKA is not used, because no implementation exists in the repository.

### 8.6 Compute (`compute_accounting.json`)

- **Latency:** batch-1 median over 50 salted DP-DEV windows after 3 warm-ups, on GPU and on CPU with 4 threads, for:
  - point-only, generation-only (NFE 8) and both outputs;
  - with and without the detector;
  - one vector-field evaluation.
- **Feature caching:** dual "both" reuses one shared trunk pass, and "both_uncached" recomputes it. The actual benefit
  is reported; compute savings are not claimed unless observed.
- **Other:** peak GPU inference memory, FLOPs (`torch.utils.flop_counter`), training time and peak training memory.

## 9. Bootstrap

- **2,000 patient-clustered replicates, seed 20261002.** The patient is the resampling unit. Paired differences are
  formed per window or per patient before resampling, so all paired comparisons use matched resamples.
- **FD:** recomputed inside every replicate on the matched generated windows, using the exact sufficient-statistics
  bootstrap (`anchorflow.fastfd`). It is verified against `kanflow_fd` at run time.

## 10. Gates (frozen margins; candidate − specialist unless stated)

| gate | definition | pass rule |
|---|---|---|
| P1 morphology | corr(dual point) − corr(P) | 95 % CI lower bound > −0.02 |
| P2 event safety | FP/window(dual point) − FP/window(P); recall(dual point) − recall(P) | FP CI upper < +0.05 **and** recall CI lower > −0.01 |
| G1 generative | FD(dual gen) − FD(G) | 95 % CI upper < +1.0 |
| CONDITION | dual gen with PPG shuffled − conditioned | FD CI entirely > 0 **and** corr CI entirely < 0 |
| E1 efficiency | P_dual | ≤ 0.85 · P_sep (structural, §6) |

- **A candidate QUALIFIES on DP-DEV only if P1, P2, G1, CONDITION and E1 all pass.**
- There is no requirement that flow samples have point-estimation quality, or that the point head reach ScaleFlow FD.
- Latency has no mandatory threshold.

## 11. Bounded development and winner selection

- **Exactly the frozen candidates S0, S1 and S2** are trained, each once. No further architectures are added after
  outcomes.
- **Selection:**
  1. discard every candidate that does not qualify;
  2. if none qualifies → **DP0 DEVELOPMENT FAILED**: STOP, and do not open AF-LOCK;
  3. if exactly one qualifies, select it;
  4. if several qualify, use the frozen lexicographic rule: (1) largest parameter saving, (2) lower generative FD,
     (3) higher point morphology correlation, (4) lower combined (both-output, GPU batch-1) latency.
- **Only one winner is frozen.**
- **If no winner:** none of the following is done — more sharing depths, loss weighting, PCGrad, attention, another
  head, FM tuning, NFE changes, margin changes, AF-DEV, AF-LOCK, TEST.

## 12. AF-LOCK (sealed)

- **Freeze:** a winner is frozen in `lock_freeze_manifest.json`, which records sha256 of:
  - this preregistration and the DP0 code;
  - the detector, P, G and selected dual checkpoints;
  - the ownership graph, training manifest, parameter accounting and split;
  - the FM / Euler / noise / metric / bootstrap / shuffle code.
- **Commit and push:** the manifest is committed and pushed first. The AF-LOCK loader refuses unless that manifest
  exists, names a winner, is committed and unchanged, and every hashed file is unchanged.
- **Evaluation:** exactly once, on AF-LOCK (337 patients), for SPECIALIST P, SPECIALIST G and the selected dual model
  only.
  - No retraining, and no return to DP-DEV.
  - The same P1, P2, G1, CONDITION and E1 definitions and margins apply.
  - K16 uses the same fixed 2,000-window rule (salted rank `dp0-k16-v1` over AF-LOCK windows, capped at 2,000) for G
    and the selected dual generative head, descriptive only.
- **Final verdict:**
  - **CONFIRMED** only if every AF-LOCK gate passes;
  - otherwise **FAILED**;
  - PARTIAL is not used, and there is no rescue after the lock.

## 13. Old V1 TEST

- **Closed for all of DP0, even if AF-LOCK confirms.** The DP0 script has no TEST loader (`load_test` always raises).
- If DP0 is confirmed, only `docs/DP1_FINAL_TEST_PREREGISTRATION_DRAFT.md` is written. DP1 is not executed.

## 14. Tests (`tests/test_dp0_dualreadout.py`, 28 test cases; written before any real training or metric)

The tests cover these 35 requirements:
- **Split:**
  - DP-TRAIN = 2,100 and DP-DEV = 300; disjoint; reproduced by the frozen rule;
  - no AF-DEV / AF-LOCK / ARCH-VAL / ARCH-HOLDOUT / SF-VAL / old val / old test patient.
- **Seals:**
  - the AF-LOCK loader is blocked without a freeze, without a winner, and with an uncommitted manifest;
  - lock evaluation calls `check_lock_freeze()` first;
  - the old TEST loader always raises, and there is no TEST, ARCH-VAL or HOLDOUT load in the source;
  - DP-DEV evaluation needs the committed preregistration.
- **Training sources:**
  - the detector and every model train on DP-TRAIN only;
  - P is point-only L1 (WWDet, 593,577);
  - G is the exact ScaleFlow FM (ScaleFM(50, coupled), 598,333; linear path; Euler 8).
- **Shared inputs:**
  - one identical raster for all models, including the shuffle;
  - the shared encoder receives only [PPG, raster]: a forward hook proves that x_t never reaches the stem;
  - the point head has no noise or time input;
  - the flow head responds to x_t and t.
- **No mixing:** outputs are never added, and there is no anchor or residual path.
- **Ownership and adapters:**
  - S0 / S1 / S2 ownership graphs, block dilations and parameter counts;
  - the frozen widths;
  - adapters are 1 × 1 + GELU, total < 5 %, with no attention.
- **Round-robin training:**
  - a point substep leaves the flow-private parameters bit-identical (including weight decay), and vice versa;
  - shared parameters change on both substeps;
  - 6 synthetic cycles give 6 + 6 updates with alternating order, and AdamW step counts are 12 (shared) and 6 (private);
  - the training stage asserts 20,000 + 20,000;
  - one optimizer, no GradNorm / PCGrad / weighting.
- **Sampling:**
  - the deterministic noise mapping;
  - Euler 8 makes exactly 8 vector-field calls and 1 condition encoding.
- **Gates and statistics:**
  - the parameter-saving and E1 arithmetic;
  - the patient-clustered bootstrap (patients, not windows, are resampled);
  - the gate rules at the margin boundaries;
  - the lexicographic selection.

**Synthetic dry run:** every stage from audit to lock_summarize ran on synthetic data in a scratch directory. The
numbers mean nothing.

## 15. Claim boundary

- **The intended claim, if confirmed:** one shared architecture preserves both specialist capabilities with fewer
  parameters and task-specific readouts.
- **Not claimed:**
  - "DualReadout produces better point ECGs than WW-L1";
  - "better samples than ScaleFlow". Any improvement beyond the specialists is secondary.
- **Never claimed, even if confirmed:**
  - universal incompatibility of point and generative objectives;
  - gradient conflict as a causal explanation;
  - calibrated uncertainty or patient-specific morphology uncertainty;
  - clinical validity or external generalization;
  - final TEST performance;
  - state of the art, first dual-readout physiological model, or first reconstruction / generation multitask
    architecture.
- **Interpretation rules (spec §47–49):**
  - full sharing failing while partial sharing works may be read as measurable task interference under complete
    sharing; gradient cosines are supportive, not causal proof;
  - if full sharing wins the frozen rule, it is preferred;
  - a stem-only success could not satisfy E1 here (§6).

## 16. Commits

1. this preregistration + split + implementation;
2. DP-DEV results + candidate selection;
3. if a candidate qualifies: the AF-LOCK freeze commit;
4. if the lock was opened: AF-LOCK results + report.

The report is `docs/DP0_DUALREADOUT_REPORT.md` (27 sections, spec §55).
