# R2 — Frozen Global Rhythm Scaffold → ECG Generator Transfer Probe — PREREGISTRATION

**Status:** frozen at this commit, pushed **before any R2 optimizer step** (the 100-step preflight is an
optimizer step and runs only after this commit).
**Type:** minimal transfer / falsification probe. **NO NEW ATTENTION. NO NEW TRANSFORMER. NO NEW FLOW
OBJECTIVE. NO C2. NO TEST. NO MULTI-SEED. NO FULL GENERATOR RETRAINING. NO HYPERPARAMETER SEARCH.
NO PENGUIN MODIFICATION.** Not a method-paper experiment.

---

## 0. Standing disclosures

- **R2 was motivated by the V1 waveform visual inspection and by the R1 global-rhythm probe**
  (`docs/V1_ALL_SUBJECT_STEPWISE_VISUALIZATION_REPORT.md`, `docs/R1_PPG_GLOBAL_RHYTHM_OBSERVABILITY_REPORT.md`).
  It is **not independent confirmatory evidence**.
- **Supervision disclosure.** The R1 Global-TCN was trained with GT ECG R-peaks (frozen detector on the GT
  ECG) as labels. TRUE rhythm conditioning therefore carries **target-derived training supervision** that the
  frozen generator never had. At generator inference **ECG is unavailable, GT R-peaks are unavailable, and
  only PPG enters the frozen Global-TCN**; the distinction is one of *training* supervision and is never
  hidden. Any future comparison must be against methods given equivalent event supervision.
- **Scaffold reliability shift.** The Global-TCN was trained on 10 of the 12 adapter-training subjects
  (R1 probe-train) and model-selected on the other 2 (u7y, e61). During adapter training `s_pred` is
  therefore an *in-sample* scaffold; on an0/k2s it is *out-of-sample* (R1: per-window F1@50 of the scaffold
  is exactly 0 in 5.4 % and < 0.3 in 25.7 % of the 8,192 validation windows). The adapter's single
  128-weight gain is calibrated on the more reliable in-sample field. This biases TRUE downward at validation
  and inflates ORACLE − TRUE; verdict C (§22) is read as *"predicted-scaffold quality and/or interface
  insufficient, with train/val scaffold reliability shift as an unseparated contributor"*. Free descriptive:
  `provenance.json` records mean and p10/p90 of per-window max(s_pred) and mean(s_pred) on the visited
  training windows and on the 2,048 validation windows.
- **ORACLE is target leakage by design** (§6) and partly detector-circular (§17). It is a diagnostic upper
  bound, never a method, never a benchmark row; every table row carries the label
  *"ORACLE (GT-R leakage; diagnostic only)"*.
- The adapter is trained on windows the frozen generator was itself trained on (in-sample residuals); the
  C1 log shows no train/val residual gap (unweighted MSE 0.125 vs 0.119), so this is disclosed, not corrected.
- A failed transfer does not prove that rhythm information is useless (R1 established observability) nor
  that a richer interface would fail.

## 1. Question

> Does transferring the frozen PPG-derived global rhythm scaffold into the existing ECG generator improve
> event correspondence **without sacrificing waveform structure**?

Three possibilities are separated: **P-A** rhythm information is useful and even a minimal transfer path
improves event correspondence; **P-B** the information exists but the current generator / interface cannot
exploit it adequately; **P-C** the scaffold does not improve generation beyond an unaligned / shuffled
control. Mapping to the verdicts of §22: P-A ↔ verdict A (or verdict B with the structure caveat);
P-B ↔ verdict C; P-C ↔ verdict D with TRUE ≈ SHUFFLE. Verdict B straddles P-A and P-B.

## 2. Provenance

| item | value |
|---|---|
| start HEAD | `71aefb44a180fe319c5343306bb35789a42ba78f` == origin/main, clean |
| submodules | PENGUIN `6cd70cdefb91f10efeb8dce34019b5067cb25344`, iMeanFlow `bf60cd7cb653f6628e59d48034b333c5eba445e2` |
| C2 | deferred, zero weight updates (`docs/C2_DEFERRED_BEFORE_TRAINING.md`) |
| `outputs/r2_*` | none exists |
| test subjects `kjd`, `ssx` | **never loaded** (`event_reliability.assert_no_test_subjects` in every R2 entry point) |
| GPU / stack | RTX 5090 (32.6 GB), torch 2.11.0+cu130; recorded in `provenance.json` with numpy / neurokit2 versions |

### 2.1 Frozen generator — resolved from provenance, not from the path

| field | value |
|---|---|
| path | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` |
| resolved from | `artifacts/v1_stepwise_visualization/checkpoint_manifest.json` + `provenance.json` (V1), `artifacts/m1_c1_structural_audit/checkpoint_manifest.json` (M1 arm B), `artifacts/c1_interval_exposure/stage1_result.json` (C1 arm B) |
| selected round | 45 (`ck["epoch"]`, C1 arm B, `fixed_imf_mse` 0.11945885431656277; run early-stopped at round 66) |
| file sha256 | `557c70541f5cdd07819a3da04bb53477ac988272855073806677ccdd0e9b14f2` |
| file md5 | `e9eea9993340b052ea5f62fe70a42757` (C1 stage1/2, M1) |
| state_dict sha256 (sorted keys, contiguous float bytes; V1 method) | `47d7ccb94e5dbf7190d777f852b18f107f3ce2628d160b5e01ff96ef2a1d0d0f` |
| architecture | `MeanFlowS5(build_penguin_backbone(n_step=1, sample_rate=128, h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0), cond_mode="h_only", h_scale=1.0)`, 4,568,707 parameters (161 `backbone.*` tensors) |
| training recipe (A4/C1; reused verbatim in §9) | AdamW lr 1e-3 wd 0.01, batch 64 = 2 × micro-batch 32, `imf_cfg` {p_mean −0.4, p_std 1.0, data_proportion 0.5, norm_p 1.0, norm_eps 0.01, jvp_mode forward}, loader generator seed 42, (t, r) generator seed 43, source noise from the CUDA global RNG seeded 42 |
| identity | B is the C1 arm-B replay checkpoint. Its **weights are identical** to the frozen A4 checkpoint (`outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt`): state_dict sha256 `47d7ccb9…` in both files; the two files differ only in saved metadata (file md5 `31c042d2…` A4 vs `e9eea999…` C1). B therefore *is* the A4 generator. |

### 2.2 Frozen R1 Global-TCN — resolved from R1 provenance

| field | value |
|---|---|
| path | `outputs/r1_global_tcn_seed42/checkpoint_best.pt` (`artifacts/r1_global_rhythm/model_manifest.json`, `provenance.json`) |
| file sha256 | `bfe76ea6fd2842dcf24870e243e79285a58930a58218672b33e768a0ecc70b55` |
| state_dict sha256 (V1 method) | `0986a7af1db291336046f3c7e9659aafc7ee77a381745a4e33344a7ac96a3287` |
| architecture | `RhythmTCN(dilations=(1,2,4,8,16,32,64,128), ch=64, k=5, n_sites=0)`, receptive field 2,041 samples |
| parameters | 328,897 |
| best epoch / internal-dev BCE | 12 / 0.4329977333545685 |
| R1 event threshold | 0.35 (recorded; **not used** in R2 — the dense pre-NMS field is the scaffold; it is used only for the exploratory scaffold-quality stratification of §17) |

Neither checkpoint is overwritten. R2 saves only adapter weights (§9.5).

## 3. Frozen components and the trainable set

- **Generator:** all 4,568,707 parameters `requires_grad = False` (no dropout / BN exist, so train/eval
  mode is numerically irrelevant), loaded once per process.
- **Global-TCN:** all 328,897 parameters `requires_grad = False`, `eval()`.
- **Trainable, in every trained arm:** the RhythmAdapter only (§5).
- Before every run the trainable parameter names are asserted to be exactly
  `{"rhythm_adapter.proj.weight"}` and written to `trainable_parameters.json`; after the first optimizer step
  of every process (preflight and each arm) every generator and Global-TCN parameter is asserted to have
  `.grad is None`. **Any gradient on a frozen parameter → STOP.**

## 4. Rhythm scaffold

For the PPG window designated by the arm (`x_ppg ∈ ℝ^{1×1024}`, the npz `x` row, float32, exactly as
stored — the representation both the generator and the probe were trained on):

    s_pred = sigmoid(GlobalTCN(x_ppg)).detach()      # [B, 1, 1024], dense PRE-NMS probability field

No thresholding, no NMS, no event extraction, no shift, no normalisation. No gradient reaches the
Global-TCN. The scaffold is computed on the fly per micro-batch under `torch.no_grad()` from the PPG of the
**current** window (TRUE) or of the **partner** window (SHUFFLE, §7). The Global-TCN is deterministic
(cudnn deterministic), so the same PPG yields the same field in every arm.

## 5. Minimal transfer adapter (frozen design)

Backbone stem (upstream, unmodified): `ppg_e = backbone.pre_conv_ppg(ppg)` ∈ ℝ^{B×128×1024}.
R2 adds, in **src/** only (`src/ppg2ecg/flow/rhythm_transfer.py`), a subclass `RhythmMeanFlowS5(MeanFlowS5)`
whose `u` re-states the frozen 13-line forward pass with one change:

    rhythm_e = RhythmAdapter(s)                       # Conv1d(1 → h_dim=128, kernel_size=1, bias=False)
    ppg_e'   = ppg_e + rhythm_e

- **No activation, no normalisation, no temporal convolution, no attention, no MLP.**
- **All adapter weights initialised to zero**, so at optimizer step 0 `rhythm_e == 0` and the model is
  **exactly** the frozen baseline. Verified before this freeze on the real checkpoint on this GPU: with a
  zero adapter the NFE-4 predictions are `torch.equal` to the frozen `MeanFlowS5` for a zero scaffold **and**
  for a real `s_pred` scaffold (exact ±0 products, exact `x + 0`). Asserted by tests (§23) and by the
  arm-B parity check (§12); the step-0 adapter checkpoint of every arm is a free identity check with that
  arm's actual scaffold.
- Trainable parameter count = `h_dim` = **128** (resolved from `model_cfg["h_dim"]`, asserted).
- The adapter output is channel-varying by construction (one learned weight per channel), which is required
  because the block-level `LayerNorm` acts over the channel axis and annihilates a channel-uniform addition;
  this is a property of the frozen design, not a tuning choice.
- **Interface:** the scaffold is passed to `u` concatenated to the PPG as a second channel
  (`ppg2 = cat([ppg, s], dim=1)`, split inside `u`), so that `imeanflow_loss` (training),
  `event_reliability.sample_meanflow_schedule` (evaluation) and `sample_meanflow` (tests) are called
  **unchanged**. `fixed_imf_mse`, `gen_diag` and the validation banks are **not used** in R2 (no validation
  quantity is computed during adapter training). Arm B uses `s = 0` and a zero adapter.
- The adapter is active on every forward call (all `t`, all `h`, including the `h = 0` flow-matching rows).
  The `h = 0` half is positional (first `int(Bc·0.5)` rows of each micro-batch, `sample_tr` `fm_mask`) and
  therefore falls on the same windows in every arm; `rhythm_e` is independent of `z`, `t`, `r`, so the
  adapter enters the JVP term only through `u`, never through its tangent.
- **No capacity is added after seeing results.**

## 6. Arms — exactly four

| arm | adapter | scaffold at training | scaffold at validation | trained? |
|---|---|---|---|---|
| **B** | zero (frozen baseline) | — | `0` | no |
| **TRUE** | trained | `s_pred(PPG of the current window)` | `s_pred(PPG of the current window)` | yes |
| **SHUFFLE** | trained (same init, same optimisation) | `s_pred(PPG of the partner window)` (§7) | `s_pred(PPG of the partner window)` (§7) | yes |
| **ORACLE (GT-R leakage; diagnostic only)** | trained (same init, same optimisation) | `s_oracle` = σ = 100 ms (12.8-sample) max-combined Gaussian field on the GT R-peaks of the current window — the exact R1 label `rhythm_tcn.soft_event_field(rpeaks.detect_rpeaks(y.astype(float64), 128), 1024)` | the same field from the **validation window's own GT R-peaks** — target leakage by design | yes |

- **B** predictions must reproduce the frozen baseline bit-exactly for the same source (§12).
- **SHUFFLE** preserves subject, site, adapter parameter count, training cost, and the stratum-level
  (marginal) distribution of scaffold values and beat rates. It does **not** preserve the window's own beat
  rate, beat count or phase (WildPPG windows have drifting HR; matching by HR / beat count is forbidden by
  §7). It therefore tests whether an adapter trained on unaligned scaffolds gains anything, not whether the
  TRUE adapter depends on alignment. A generic periodic prior at the correct rate but random phase is nulled
  at the metric level by the count-matched random-phase floor of E1 (§13); rate-preserving phase dependence of
  the trained TRUE adapter is probed only by the inference-time circular shift (§18).
- **ORACLE** answers only: *if perfect event information were supplied through this minimal interface,
  could the frozen generator use it?* On fex and p5d (dataset-flagged noisy ECG, 17 % of training windows)
  the ORACLE training field is itself noisy.
- **ORACLE training field cache.** Before any ORACLE optimizer step the fields for **all 293,271 training
  rows** are precomputed deterministically on CPU with the frozen detector and the numpy `soft_event_field`
  (float32), cached at `artifacts/r2_rhythm_transfer/_cache_oracle_train.npz` (sha256 in
  `provenance.json`), and loaded to the GPU as one float32 tensor; the ORACLE driver reads the cache, so its
  per-step cost equals TRUE's. The CPU build time is recorded in `runtime_preflight.json` and is outside the
  GPU-hour rule (§10).

## 7. SHUFFLE construction — fixed before any adapter result

Salt **`"r2-rhythm-shuffle-v1"`**. For every stratum (subject × site) of each of three populations:

| population | windows | strata |
|---|---|---|
| `train` | all 293,271 windows of the 12 A4 training subjects | 48 strata of ≈ 5.5–6.9 k |
| `eval` | the frozen 2,048-window development subset (§11) | 8 strata of 240–272 |
| `viz` | the 64 V1 validation VIZ windows (§24) | 8 strata of exactly 8 |

1. collect the stratum's windows, keyed by `(subject, site, npz window_index)`;
2. rank them by `sha256(f"{salt}|{subject}|{site}|{window_index}")` (stable argsort);
3. map the window at rank `i` to the window at rank `(i + 1) mod n`.

Deterministic, bijective within the stratum, **no fixed point for every n ≥ 2**. If any stratum has a
single element the run **STOPS**. No matching by HR, beat count, signal quality, R count or model score.
`shuffle_manifest.csv` columns: `population, subject, site, window_index, array_pos` (per-subject npz
positional row), `partner_window_index, partner_array_pos`, plus `train_row / partner_train_row`
(row in the 293,271-row concatenated training tensor, subject order e61 … w4p) for `train` and
`pop_row / partner_pop_row` (0 … 2047, an0 then k2s) for `eval`. The driver builds a length-293,271 int64
partner tensor asserted to be a fixed-point-free bijection and asserts `x_tr[train_row] == npz x[array_pos]`
on a sample. Free descriptive for `eval`: `|n_beats(partner) − n_beats(own)|` and `|RR̄(partner) − RR̄(own)|`
from the already-computed GT peaks, summarised in the manifest. Unit-tested: bijective, fixed-point-free,
salt-dependent, order-invariant, on all three populations.

## 8. Training data

The frozen A4 split (`data/manifests/split_a4_wildppg_seed42.json`): the 12 training subjects
`e61 fex l38 n31 ngh p5d p9p qm9 trh tz8 u7y w4p`, **all** their windows (293,271, loaded exactly as
`train_a0.load_arrays` does — asserted equal), from `data/processed/wildppg_8s`. No new split. No test.
**No validation window is read during adapter training.** No validation-based tuning.

## 9. Training protocol — frozen

| item | value |
|---|---|
| trainable | RhythmAdapter only (128 weights) |
| seed | 42: `seed_everything(42, deterministic=True)`; loader `torch.Generator().manual_seed(42)`; (t, r) `torch.Generator().manual_seed(43)`; source `e = torch.randn(Bc, 1, 1024, device=cuda)` from the CUDA global RNG — identical to A4/C1 |
| shared across TRUE / SHUFFLE / ORACLE | generator weights, zero adapter init, dataloader order (`DataLoader(TensorDataset(x, y, idx), batch_size=64, shuffle=True, generator=gen)`; adding the index tensor does not change the permutation — verified over 300 batches against the two-tensor A4 loader, asserted by test), Gaussian source stream, (t, r) stream (`sample_tr_c1(Bc, tr_gen, arm="B", **tr_kw)` = the bit-identical historical sampler), optimizer configuration, number of steps |
| only difference | the scaffold tensor |
| optimizer | `AdamW(adapter.parameters(), lr=1e-3, weight_decay=0.01)` — the A4 optimizer configuration, reused cleanly |
| loss | the **exact frozen** `imeanflow_loss(net, ecg, ppg2, e, t, r, norm_p=1.0, norm_eps=0.01, jvp_mode="forward")`; no event / R-peak / morphology / RR / auxiliary loss |
| batch | 64 windows = 2 micro-batches of 32, `(loss · Bc/B).backward()` accumulation, one `opt.step()` per batch, exactly as A4. **Micro-batch 32 is fixed because it defines the (t, r) / e draw shapes, not as a memory choice** |
| budget | **exactly 2,200 optimizer steps** per trained arm = the first 2,200 batches of the seed-42 epoch-1 order (140,800 window visits, 48 % of one epoch, no window visited twice); no early stopping, no best-checkpoint selection; `opt_steps == 2200` asserted |
| checkpoints | adapter weights at steps 0, 550, 1100, 2200 (diagnostic only); **the primary result is step 2200 exactly** |
| logging (per step) | weighted loss (≈ 1 by construction of the adaptive weight; reported only for parity with A4), unweighted `mse`, `delta2_mean`, `loss_before_weighting`, the full A8 w-statistics (`w_mean, w_median, w_p01/p10/p25/p75/p90/p99, w_min, w_max, w_saturation_frac, w_near_lower_frac`), `u_abs`, `dudt_abs`, adapter weight L2 norm, elapsed. **Arm training curves are compared on the unweighted `mse` / `delta2_mean` only.** |
| objective disclosure | the frozen weight `w = 1/(δ² + 0.01)` with δ² ≈ 10²–10³ makes the objective ≈ mean log δ²: each window contributes a relative-improvement gradient, so windows where B mis-times the QRS (largest δ²) receive the smallest absolute weight. This is a property of the mandated objective and biases against TRUE; it is not altered. |
| RNG hygiene | the adapter is created on CPU and zeroed (its init consumes only the CPU global RNG, which no training stream reads); the Global-TCN and generator are loaded (no RNG); scaffold computation consumes no RNG; the CUDA global RNG is consumed exclusively by the `e` draws in the same order as A4 |
| process isolation | preflight, TRUE, SHUFFLE and ORACLE are **four separate process invocations** (`--arm {preflight,true,shuffle,oracle}`), each calling `seed_everything(42)`, creating `gen` / `tr_gen` and a fresh loader iterator, and creating a zero adapter. A paired-randomness probe hash over the first 4 micro-batches of `(idx, t, r, e)` is written to `runtime_preflight.json` and to every training log and asserted identical across the four processes |
| per-arm provenance | `opt_steps` (asserted 2,200), wall time, s/step, `max_memory_allocated` and `max_memory_reserved`, git, seed, final adapter L2 norm; from the loader order alone: realised visits per subject × site in the first 2,200 batches and the share from fex/p5d |
| reachable magnitude | under AdamW lr 1e-3 each zero-initialised weight can move at most ≈ 2.2 in 2,200 steps; the realised `RMS(rhythm_e) / RMS(ppg_e)` at step 2200 on the primary population is reported per arm so a null result can be read against the reachable injection magnitude |

Nothing in this table is changed after seeing a number. If the A4 optimizer configuration cannot be reused
cleanly the run STOPS and reports before any change.

### 9.5 Saved outputs

`outputs/r2_{true,shuffle,oracle}_adapter_seed42/adapter_step{0,550,1100,2200}.pt` — adapter `state_dict`
(128 floats) plus provenance (generator sha256, Global-TCN sha256, step, git, seed, arm). The frozen
generator is **not** duplicated. Existing outputs are never overwritten (refuse-to-overwrite assert).

## 10. Runtime stop rule

Before full training: **exactly 100 optimizer steps of TRUE** from the zero init with the full protocol in
its own process. Record s/step = mean over steps 2–100 (step 1 reported separately: cuDNN warm-up), peak
`max_memory_allocated` / `max_memory_reserved` after the data tensors are on the device, in
`runtime_preflight.json`; **discard the adapter state** (never saved). Projected cost = 3 arms × 2,200 ×
s/step (evaluation and the ORACLE cache build excluded). **If the projection exceeds 6 GPU-hours: STOP and
report** before proceeding. No silent budget reduction. Pre-freeze measurement (forward + backward, no
optimizer step): ≈ 0.35 s per micro-batch, 3.9 GiB peak → expected ≈ 0.7 s/step, ≈ 1.3 GPU-hours.

## 11. Evaluation population

**PRIMARY:** the frozen C0/C1 development subset — `event_reliability.select_subset("x4-event-nfe-v2",
subject, len(npz["x"]), 1024)` for `an0` and `k2s` (positional npz row indices), **asserted element-for-
element** against `artifacts/x4_0_event_reliability/nfe_subset.json`; 2,048 windows in population order
an0-ascending then k2s-ascending; 19,834 GT beats. Source noise `e0 = torch.randn(2048, 1, 1024,
generator=torch.Generator().manual_seed(0))`, drawn once in population order, sha256 asserted equal to
`868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f` (C0/C1). All arms, all NFE, all
ablations use identical windows, identical PPG, identical source rows, identical preprocessing.

**SECONDARY (site-wise only, §20):** the existing R1 validation cohort — `r1_cohort.cohort_positions`,
salt `r1-global-rhythm-observability-v1`, 1,024 per subject × site = 8,192 windows, in R1 cohort order
(an0 sternum/head/wrist/ankle, then k2s; chunks of 64 never straddle a site). The spec's "V1 8192 cohort"
does not exist as such (V1's largest validation cohort is 1,024 windows); the 8,192-window site-balanced
validation cohort is R1's, and it is the cohort used here. Source noise `torch.randn(8192, 1, 1024,
generator=torch.Generator().manual_seed(0))` in cohort order; its sha256 is recorded in `provenance.json`
at first use. B and TRUE only, NFE 4 only. Run only if it fits a pre-declared budget of 1 GPU-hour +
1 CPU-hour after the primary evaluation; otherwise `site_metrics.csv` carries a `skipped` marker.
**This cohort cannot change the primary verdict.**

Never load test.

## 12. Sampler, NFE, arm-B parity

`event_reliability.sample_meanflow_schedule(net, ppg2, e, ER.UNIFORM[n])` (uniform t from 1 to 0,
`z ← z − (t−r)·u(z, ppg2, t, t−r)`), realised NFE asserted equal to n.

- **PRIMARY: NFE = 4.** NFE 4 is B's development optimum selected on this same population (C0) and on the
  V1 validation cohort; it is an in-sample choice for B and is **not re-searched for TRUE**.
- **SECONDARY: NFE = 1, 2.** All four arms are scored at NFE 1 and 2 with the full event family and S1–S8;
  TRUE − B and TRUE − SHUFFLE oriented paired effects are computed for the four event metrics only
  (non-gate). A shift of TRUE's optimum toward fewer steps would be visible; a shift toward more steps
  (NFE 8, not run) would be invisible and cannot change the verdict.

**Arm-B parity (STOP condition):** on the primary population at NFE 4, `RhythmMeanFlowS5` with a zero
adapter and zero scaffold must equal the frozen `MeanFlowS5` bit-exactly (`torch.equal` on the full
2,048-window prediction tensor, same e0 rows, same process). **Failure → STOP.**
**Historical regression (diagnostic, not a STOP):** B's macro metrics are compared with
`artifacts/c1_interval_exposure/stage1_metrics.csv` row (B, 4) (produced at git 38eaf45 with the then-
installed libraries; M1 later reproduced it to the last printed digit on this machine); every per-metric Δ
is written to `provenance.json` and any |Δ| > 1e-6 is flagged and reported, not stopped. Library versions
are recorded.

## 13. Event metrics — PRIMARY FAMILY

The frozen X0/C0 pipeline, copied verbatim from `scripts/analyze_c0_compression_target.py`
(`_score_chunk`, `_chance_chunk`, `_peaks`, `pmap`; chunks of 64 in population order): `rpeaks.detect_rpeaks`
(neurokit) on GT and prediction, `rpeaks.match_rpeaks` one-to-one at 50 ms. Per window and per arm:

raw F1, precision, recall, chance F1 (count-matched random-phase floor, 20 draws, `default_rng(20260901)`
per chunk, `s1_audit.chance_random_phase`), **F1 excess = F1 − chance F1**, missing fraction =
n_missing / n_ref, spurious fraction = n_spurious / n_ref, beats ratio = n_pred / n_ref, beats-ratio
deviation = |beats ratio − 1|, matched coverage (pooled; numerator counts matches among all detected GT
peaks, denominator the margin-valid GT beats — the C0 convention, descriptive only). Conventions: empty
prediction → F1 = 0, chance = 0, excess = 0, missing = 1, beats-ratio deviation = 1; `n_ref = max(n_ref, 1)`;
the counts of such windows per arm are written to `event_metrics.csv`. Macro = `s1_audit.macro` (equal
an0/k2s weight, nanmean).

- **E1 (primary event metric) = F1 excess.**
- **E2 = beats-ratio deviation, E3 = missing fraction, E4 = spurious fraction** (primary reliability).
- Raw F1 is never interpreted alone.
- **E1–E4 are detector-dependent** (neurokit on the generated ECG). Detector-independent protection is
  S3–S5 at GT-fixed coordinates. A TRUE gain on E1 accompanied by clear degradation of S3 or S4 is the
  signature of a detector-friendly local maximum rather than timing transfer and maps to verdict B, never A.
- Free secondary inside the same path: per arm, the matched signed timing error at 50 ms — median |Δt|,
  mean Δt, fraction |Δt| ≤ 25 ms — reported, not gated.

## 14. Waveform / structure metrics

All GT-fixed coordinates, oracle-free, no shift-max, no matched-morphology primary. Frozen definitions,
imported, not re-implemented:

| id | metric | frozen definition | aggregation |
|---|---|---|---|
| **S1** | raw RMSE | C0 `raw_rmse`: RMSE over the 83-sample GT-beat segment (r−32 … r+51), beat set r ∈ [51, 954], `alignment_diagnostics.segment_stats` | window nanmean → macro |
| **S2** | raw correlation | C0 `raw_corr`: Pearson over the same segment | window nanmean → macro |
| **S3** | fixed-coordinate QRS RMSE | C0 `raw_qrs_rmse`: RMSE over GT R ± 13 samples (local [19:46]) | window nanmean → macro |
| **S4** | fixed-coordinate derivative RMSE | M1 `qrs_core_morphology.qrs_deriv_rmse` (first difference over GT R ± 11 samples), beat set r ∈ [11, 1012] | window mean → macro |
| **S5** | fixed-coordinate curvature error | M1 `qrs_core_morphology.qrs_curvature_err` (second difference) | window mean → macro |
| S6 | QRS energy deviation | C0 `qrs_e_dev` = \|nanmedian(raw_qrs_energy_ratio) − 1\| | macro |
| S7 | p2p deviation | C0 `p2p_dev` = \|nanmedian(raw_p2p_ratio) − 1\| | macro |
| S8 | HF spectral fraction | `metrics.hf_energy_ratio` (whole window, ≥ 15 Hz) on the prediction, and \|hf_pred − hf_gt\| as the reconstruction error | macro |
| supp. | whole-window RMSE / correlation, M1 `qrs_rmse_core` (GT R ± 10) | V1's NFE-4 definitions | macro |

S1/S2/S3 are the C0/C1 frozen primaries on this population (chosen for comparability with
`stage1_metrics.csv`); S4/S5 are the M1 QRS-core definitions with a different valid-beat set; S1–S2 share one
83-sample segment and S3–S5 share one QRS core, so the five are two families and the report states which
family carried any clear degradation. V1's whole-window RMSE / corr and `qrs_rmse_core` are supplementary and
**never enter gate item 4**. **Primary structural protection: S1–S5.** S6/S7 are secondary only (M1: ratio
metrics can improve through calibration while direct structure worsens).

## 15. Paired comparisons — TRUE vs B (primary, NFE 4)

Paired per-window oriented effects on the 2,048 frozen windows, **positive = the later-named arm better**
(`paired_stats.paired_subject_bootstrap(earlier, later, subjects, orient)`): F1 excess, precision, recall,
raw correlation (higher better); beats-ratio deviation, missing, spurious, S1, S3, S4, S5 (lower better).
Subject-stratified, equal an0/k2s weight, **2,000 replicates, `seed=20260902` passed explicitly** (the
library default 20260901 is not used; recorded in every row); point effect + 95 % CI + verdict
(improves / worsens / unresolved). Every `paired_bootstrap.csv` / `oracle_gap.csv` row carries `orient`.

- **Scope of the CI.** The bootstrap resamples validation windows within an0 and k2s for the single
  realised training run. It does not include training-seed variance (all trained arms share one seed-42
  loader order, source stream and (t, r) stream and are paired at the stream level) nor between-subject
  variance (n = 2; subjects are never resampled). "CI entirely > 0" means "robust to which development
  windows were drawn", not "replicable across seeds or subjects". With two balanced subjects the equal-subject
  macro equals the pooled mean absent NaN. Because the rng is re-seeded per call, all comparisons share one
  resample-index set.
- **NaN handling.** S1–S5 are NaN only for windows with no valid GT beat (GT-determined, symmetric across
  arms; 0 such windows on this population); NaN pairs are dropped within subject by the frozen nanmean, the
  NaN count per (arm, metric) is written and asserted identical across arms (fail → STOP), and the effective
  n is reported alongside `n_pairs`.

## 16. TRUE vs SHUFFLE — specificity (NFE 4)

Same paired bootstrap; primary specificity metrics: F1 excess, beats-ratio deviation, missing, spurious.
TRUE must beat SHUFFLE to claim window-specific rhythm transfer. **SHUFFLE − B** is reported next to it
(same metrics, same bootstrap, no extra inference). Reading rule fixed now: gate item 2 is read as
window-specific transfer only if SHUFFLE − B on F1 excess is not "worsens"; if SHUFFLE clearly worsens vs B,
the report states that TRUE − SHUFFLE conflates benefit-of-alignment with harm-of-misalignment and that the
specificity claim rests on TRUE − B plus the phase ablation (§18). The final adapter weight L2 norms of TRUE,
SHUFFLE and ORACLE are reported side by side; a SHUFFLE norm ≪ TRUE norm is stated a priori to make item 2
nearly equivalent to item 1 (SHUFFLE ≈ B), not additional evidence.

## 17. ORACLE diagnostic (NFE 4)

Oriented paired effects ORACLE vs B, TRUE vs B, ORACLE vs TRUE (`earlier` = the first-named of B / TRUE,
`later` = ORACLE) on the event family and S1–S5, written to `oracle_gap.csv`. Let `v_OB`, `v_TB`, `v_OT` be
the F1-excess verdicts of (B→ORACLE), (B→TRUE), (TRUE→ORACLE). Exactly one row applies:

| case | (v_OB, v_TB, v_OT) | reading (diagnostic only) |
|---|---|---|
| 1 | (improves, improves, improves) | the generator / interface can exploit rhythm information; the PPG-derived scaffold is still a bottleneck |
| 2 | (improves, improves, unresolved) | the predicted scaffold already captures most of the useful rhythm information for this minimal interface |
| 3 | (improves, unresolved or worsens, improves) | exact / coarse scaffold quality is insufficient for transfer (with the §0 reliability-shift caveat) |
| 4 | (unresolved or worsens, any, any) | this minimal additive interface does not let the frozen generator exploit even near-perfect event information **within 2,200 steps** — this does **not** prove that attention would fail |
| other | anything else | the triple and the three point effects are reported without narrative |

**Circularity disclosure.** ORACLE's F1 excess is bounded only by the interface and the detector: the
injected field marks the very R locations the scorer detects, so ORACLE ≫ B on E1 is expected by
construction whenever the additive path can carry a localised field, and is evidence about the interface,
not about rhythm information. The informative ORACLE quantities are S3–S5 (does the injected location come
with QRS structure) and ORACLE vs TRUE. To separate interface from budget, the adapter weight-norm and
unweighted-mse trajectories and `RMS(rhythm_e)/RMS(ppg_e)` at step 2200 are reported per arm.

**Scaffold-quality stratification (exploratory, informs §27 only).** For each of the 2,048 windows the
scaffold's own event F1@50 is computed with the frozen R1 rule (`extract_events` at threshold 0.35,
refractory 32, one-to-one match at 50 ms to the already-computed GT peaks), and the TRUE − B paired effects
on E1–E4 are reported within terciles of that score.

## 18. Inference-time phase ablation — SECONDARY

Trained TRUE adapter, no retraining. On the primary population at NFE 4: (A) TRUE scaffold; (B) the same
scaffold circularly shifted by **+256 samples = +2.0 s** (`torch.roll(s, 256, dims=-1)`), which preserves
scaffold shape, approximate beat intervals and amplitude distribution while breaking phase. Because +2.0 s
is an integer number of beats at 60 / 90 / 120 bpm, the shift re-aligns with a fraction of beats (R1
validation RR: 11.4 % of intervals within 0.1 beat of re-alignment). Per window
`φ = frac(256 / mean GT RR in samples)` is written to `phase_ablation.csv`; the shifted-vs-TRUE oriented
paired effect (seed 20260902) on F1 excess, beats deviation, missing, spurious is reported overall **and**
stratified into φ ∈ [0, 0.1) ∪ [0.9, 1) (residual in-phase), φ ∈ [0.4, 0.6] (anti-phase) and the rest, with
n per stratum; the overall effect is a lower bound on phase dependence. The roll also relocates the TCN's
edge-padded field region; disclosed, not corrected. Not part of the gate.

## 19. Early-NFE event-persistence diagnostic — SECONDARY

B and TRUE at NFE 1, 2, 4 on the primary population. For each GT R event and each K, `delta_K =
(t_pred − t_GT)/128·1000 ms` from the frozen greedy one-to-one matcher `rpeaks.match_rpeaks(gt_pk,
pred_pk_K, 128, tol_ms=250.0)` (|dt| ≤ 32 samples; no waveform shifting). Per-beat rows
`(arm, subject, pop_row, beat, delta_1, delta_2, delta_4)` (NaN if unmatched) are written to
`nfe_event_persistence.csv` with a summary block. Statistics, each on its stated beat set: per-NFE match
fraction over all GT beats; |delta_K| over beats matched at K (and with unmatched beats censored at 250 ms);
sign consistency = fraction of beats matched at all three NFEs with sign(delta_1) == sign(delta_4)
(delta = 0 a third category, count reported) and with all three signs equal; change = delta_4 − delta_1
and |delta_4| − |delta_1| over beats matched at K = 1 and 4; "NFE 4 closer" = |delta_4| < |delta_1|
strictly, ties reported separately. The B-vs-TRUE comparison of these statistics is made on the
**intersection** of beat sets (matched in both arms), with B-only / TRUE-only / both set sizes reported.
Detector-dependent diagnostic only; no causal solver claim.

## 20. Site-wise — SECONDARY, exploratory

On the R1 8,192-window validation cohort (§11), B vs TRUE at NFE 4 per site (sternum, head, wrist, ankle):
F1 excess, beats-ratio deviation, QRS RMSE (S3), derivative RMSE (S4), chance floor with the same chunking.
Per-site paired bootstrap (subjects an0/k2s, seed 20260902) for each metric — 16 CIs, uncorrected,
labelled exploratory — plus one pre-specified contrast per metric, (wrist + ankle effect) − (sternum + head
effect), via the difference-of-improvement idiom. No single-site CI is cited as evidence; no site-based
primary verdict.

## 21. Primary success gate — frozen

All at NFE 4, TRUE = the step-2200 adapter, on the primary population. `decision.json` records every
quantity below to full precision.

1. TRUE vs B: F1 excess `verdict == "improves"` (CI entirely > 0) **and**
   `paired_subject_bootstrap(B, TRUE, "higher_better")["point"] ≥ +0.02`. (+0.02 is a pre-specified minimal
   effect of interest: +6 % of B's realised F1 excess 0.3176 on this population, exceeding B's own gain
   from doubling the sampler budget NFE 2 → 4 (+0.011). Applied to the point estimate; the CI condition is
   separate.)
2. TRUE vs SHUFFLE: F1 excess `verdict == "improves"`.
3. At least one of {beats-ratio deviation, missing fraction, spurious fraction} has TRUE-vs-B
   `verdict == "improves"` (oriented, lower better).
4. Fewer than two of {S1, S2, S3, S4, S5} have TRUE-vs-B `verdict == "worsens"` ("clear degradation" = CI
   entirely in the worse direction). The weaker reading (pass if at least two are not degraded) is explicitly
   rejected.
5. `s1_audit.macro(|n_pred / n_ref − 1|)` of TRUE < 0.20 (equal-subject; B's realised value on this
   population is 0.1067, so 0.20 is a catastrophe bound ≈ 1.9 × B, not a non-inferiority bound).

Reading rules fixed now (not gate changes): if the TRUE-vs-B verdict on beats-ratio deviation or on spurious
fraction is "worsens", every verdict statement carries the qualifier *"with beat-count distortion"*; item 3
offers three chances at a 95 % CI without correction and is the multiplicity-weakest item, so the report
states which metric carried it and whether the other two were unresolved or worsened.

**The gate is not loosened after results.**

## 22. Verdict — a total function of the decision record

`decision.json` holds `item1 … item5` (booleans), `v_OB`, `v_TB`, `v_OT` (F1-excess verdicts, §17),
`v_SB` (SHUFFLE-vs-B F1-excess verdict) and `residual_reason`. Exactly one verdict:

| order | verdict | condition |
|---|---|---|
| 1 | **A. RHYTHM SCAFFOLD TRANSFER SUPPORTED** | item1 ∧ item2 ∧ item3 ∧ item4 ∧ item5 |
| 2 | **B. EVENT GAIN WITH STRUCTURE TRADE-OFF** | ¬A ∧ item1 ∧ item2 ∧ ¬item4 (items 3 and 5 recorded as qualifiers, not required — this follows the spec wording "clearly improves event correspondence vs B and SHUFFLE, but fails structural protection"; "clearly" is item 1 with its +0.02 magnitude) |
| 3 | **C. SCAFFOLD INFORMATIVE, MINIMAL INTERFACE INSUFFICIENT** | ¬A ∧ ¬B ∧ `v_OB == "improves"` ∧ `v_OT == "improves"` |
| 4 | **D. RHYTHM CONDITIONING NOT SUPPORTED BY THIS INTERFACE** | otherwise (residual) |

Meanings: A — PPG-derived global rhythm information can improve generator event correspondence through even
a minimal conditioning path; still not a final method. B — rhythm information is useful; naive injection
damages waveform fidelity; motivates confidence-gated / structured fusion. C — rhythm / event information can
help, but the predicted scaffold and/or the minimal additive interface is insufficient (extractor vs
interface not separated causally; §0 reliability shift). D — the 1×1 additive path is insufficient; global
rhythm itself is not declared useless (R1). D's narrative *"TRUE does not beat B / SHUFFLE"* is asserted only
when item 1 or item 2 actually failed; otherwise `residual_reason` (one of: `sub-threshold event gain` —
CI > 0 but point < +0.02; `reliability item 3 failed`; `beat-count catastrophe item 5 failed`;
`ORACLE ≈ TRUE`; `ORACLE worsens`; combinations) is printed verbatim in the FINAL VERDICT section.

## 23. Tests — mandatory before the implementation commit

Generator fully frozen; Global-TCN fully frozen; only the adapter trainable (name set asserted); identical
adapter initialisation across TRUE / SHUFFLE / ORACLE; expected parameter count (128 = h_dim); step-0 output
exactly equals B with a zero scaffold **and with each arm's actual scaffold** (tiny random backbone
unconditionally; real checkpoint + real batch `torch.equal` when the files exist, `pytest.skip` otherwise,
with the set of real-file tests that ran recorded in `provenance.json`); the 2-channel interface passes the
frozen loss and sampler unchanged with finite adapter gradients; the same source tensor across arms; the
same dataloader order across trained arms (three-tensor loader equals the A4 two-tensor loader); the same
Gaussian noise and the same (t, r) (paired-randomness probe hash identical across arms); SHUFFLE has no fixed
points and is bijective on all three populations; ORACLE uses GT R only in its explicitly labelled arm; TRUE
never reads ECG at inference (the scaffold function takes PPG only — signature-level assertion); validation
not used for training / model selection (static: the driver never references the validation subjects or
`fixed_imf_mse`); test firewall (kjd/ssx) in every entry point; the frozen dev subset equals
`nfe_subset.json`; the seed-0 source bank sha256; the +256-sample roll; the ±250 ms persistence matcher
(greedy one-to-one, tolerance 32 samples); the paired bootstrap orientation, `n_boot == 2000` and
`seed == 20260902`; the verdict function is total and returns exactly one verdict on an exhaustive grid of
decision records. Full suite run.

## 24. Visualisation — deterministic, no cherry-picking

The **V1 validation VIZ cohort** (64 windows: 8 per subject × site, salt
`v1-all-subject-stepwise-visualization`, `artifacts/v1_stepwise_visualization/cohort_manifest.csv`
cohort == viz). Per (subject, site) the full 32-window METRICS stratum is sampled as **one batch** with the
seed-0 bank drawn over the 32 rows, exactly as V1, and the 8 VIZ rows are selected by position for all four
arms (the Global-TCN scaffold is computed for the 32 rows in the same batch), so the B row reproduces the V1
figures. Rows per window: 1 PPG, 2 rhythm scaffold TRUE (value at each GT R annotated), 3 GT ECG, 4 B NFE 4,
5 TRUE NFE 4, 6 SHUFFLE NFE 4 (viz-population partner, §7), 7 ORACLE NFE 4 (own GT R). GT R as vertical
reference only; predictions never shifted. Missing / spurious events (frozen detector + 50 ms one-to-one
matcher) are annotated per prediction row. A GT-R-centred −300 … +500 ms zoom per window on the first GT R
with r − 38 ≥ 0 and r + 64 ≤ 1024 (the V1 rule). The report's visual-observations section contains only
per-arm counts over all 64 windows (missing / spurious totals, windows with ≥ 1 spurious event, quantiles of
the scaffold value at GT R) plus the atlas index — no per-window prose selection. `visual_atlas/`.

## 25. Artifacts and report skeleton

`artifacts/r2_rhythm_transfer/`: `provenance.json`, `generator_checkpoint_manifest.json`,
`rhythm_checkpoint_manifest.json`, `trainable_parameters.json`, `shuffle_manifest.csv`,
`runtime_preflight.json`, `training_log_{true,shuffle,oracle}.csv`, `metrics_by_window.csv` (columns:
`arm, nfe, subject, array_pos` (positional npz row = `nfe_subset.json` value), `npz_window_index, site`, then
every per-window event and S1–S8 quantity under the exact names used by the bootstrap), `event_metrics.csv`,
`structure_metrics.csv`, `paired_bootstrap.csv`, `oracle_gap.csv`, `phase_ablation.csv`,
`nfe_event_persistence.csv`, `site_metrics.csv`, `decision.json`, `visual_atlas/`. Adapter checkpoints in
`outputs/r2_{true,shuffle,oracle}_adapter_seed42/`. Nothing in `outputs/`, `artifacts/`, `data/processed/`
enters git.

`docs/R2_RHYTHM_SCAFFOLD_TRANSFER_REPORT.md` and the final response follow the spec's §30 headings
verbatim: Repository · Frozen components · Runtime · NFE4 — Event correspondence (B / TRUE / SHUFFLE / ORACLE)
· TRUE vs B · TRUE vs SHUFFLE · Structural protection · Oracle diagnostic · Phase ablation · NFE event
persistence · Site-wise · Visual observations · FINAL R2 VERDICT · What this does NOT prove (stronger-
supervision caveat, single seed, no test, no SOTA, no novelty claim, no full generator retraining) ·
Recommended next architecture.

## 26. Claim boundaries

Even if R2 succeeds, the only permitted claim is: *"A PPG-derived global rhythm scaffold improves event
correspondence of the frozen generator under the single-seed development protocol."* Never: "we solved
PPG→ECG timing", "R peaks are directly recoverable from PPG", "the proposed method is novel", "this
establishes SOTA". The Global-TCN received target-derived R supervision; any paper must compare against
methods receiving equivalent event supervision. Gate items 3 and 4 are disjunctive / tolerant by
preregistration and no multiplicity correction is applied; no subject-level or training-seed uncertainty is
estimated — further reasons the verdict is development-only.

## 27. Next-architecture decision

No block is implemented in R2; a recommendation only, keyed to the verdict (§22): A → first ask whether the
simple adapter already suffices; consider *Confidence-Gated Global Rhythm Conditioning* only if a
missing / spurious residual, site-dependent failure or poor-confidence segments remain (the §17
stratification informs this); B → separate WHEN (rhythm scaffold) from WHAT (morphology pathway) with
confidence-gated fusion; C → cross-attention / temporal fusion becomes justified; D → do not train a large
transformer; inspect why ORACLE failed first.

## 28. Commit order

1 repository audit · 2 preregistration · 3 commit + push (this commit; its SHA is recorded in
`provenance.json`) · 4 implementation · 5 tests · 6 commit + push · 7 100-step runtime preflight ·
8 reinitialise adapters (new process) · 9 TRUE / SHUFFLE / ORACLE training · 10 primary evaluation ·
11 phase ablation · 12 NFE persistence · 13 visual atlas · 14 report · 15 result commit + push · 16 STOP.
