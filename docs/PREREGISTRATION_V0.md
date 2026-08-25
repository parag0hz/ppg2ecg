# Pre-registration v0 — OT-CFM NFE curve and the one-step question

Drafted 2026-08-25 **before any training run**. Freeze rule: this file is tagged `prereg-v0` before the first
NFE-curve number is viewed. After the tag, only bug fixes are allowed, each logged in `docs/EXPERIMENT_LOG.md` with
rationale; decision thresholds (§5) may not change.

## 1. Question and hypotheses
See `docs/RESEARCH_QUESTION.md`. H1: quality degrades below a margin at low NFE. H2: at 1 NFE the failure is morphology
blur rather than rhythm loss (H2′: rhythm/conditioning loss). H3 (conditional on H1): iMeanFlow on the identical backbone
at 1 NFE recovers the failed metrics.

## 2. Fixed factors (identical across all arms)
- Data: PPG-DaLiA, preprocessing v0 (docs/DATA_PROTOCOL.md §2), 4 s / 128 Hz / 512-sample windows.
- Splits: P0-holdout for reproduction; **P1-kfold5 for all claims**. Same manifests for every arm.
- Backbone: upstream PENGUIN Flow-SSM/S5, `h_dim=128, ssm_block_num=4, ssm_ratio=2, mlp_ratio=2`, **unmodified**
  (`external/PENGUIN` @ 6cd70cd, imported in place). The dead `cross_attn` module is left in place (it has no effect).
- Training budget (arms that train): AdamW lr 1e-3, wd 0.01, batch 64, ≤ 300 epochs, early stop on val MAE (patience 10),
  no EMA, no LR schedule, fp32 — exactly the audited upstream recipe.
- Evaluation code: `ppg2ecg.evaluation` (this repo), identical for all arms; upstream `HeartRateError` reported alongside.
- Seeds: {42, 43, 44} for training; noise seeds {0..7} at inference.

## 3. Arms
| Arm | What varies | Trains? | Purpose |
|---|---|---|---|
| **A0** reproduction | nothing (upstream `train.py`, upstream split logic, patched config only for the `DaLiA`→`PPG-DaLiA` key + paths) | yes | sanity: match paper/README numbers; produce the checkpoint for A1 |
| **A1** NFE curve | sampler only, on the **A0 checkpoint**: Euler {25,10,5,2,1} NFE; Heun {25,10,5,2,1} steps = {50,20,10,4,2} NFE | no | tests H1/H2 |
| **A1′** control loop | our training loop, OT-CFM objective, P1 splits | yes | isolates loop/split effects from the objective (needed before A2) |
| **A2** one-step | objective only: iMeanFlow, 1 NFE, same backbone/loop/splits as A1′ | yes | tests H3 — **only run if H1 confirmed** |

NFE definition: number of velocity-network forward passes per generated window (`ppg2ecg.flow.samplers.NFECounter`).
Heun with *n* steps costs 2n NFE (no Euler last step, as upstream). "One-step" means NFE = 1.

## 4. Outcomes
Evaluation unit: 4 s windows for signal metrics; **8 s windows (two consecutive segments of the same subject)** for
rhythm/morphology (matches upstream HR window). R-peaks are detected with the **same** neurokit2 pipeline on prediction
and reference (`ecg_clean` → `ecg_peaks`, method "neurokit"); matching tolerance 50 ms, one-to-one greedy.

| Class | Metric | Direction | Role |
|---|---|---|---|
| Rhythm | R-peak precision / recall / **F1** (50 ms) | ↑ | **primary** (F1) |
| Rhythm | **HR absolute error** (bpm; 60 / mean RR) | ↓ | **primary** |
| Rhythm | RR-interval MAE (ms, consecutive matched beats) | ↓ | secondary |
| Morphology | **beat-aligned template correlation** (−250…+400 ms around matched R-peaks) | ↑ | **primary** |
| Morphology | **QRS-width error** (ms; QS-trough proxy) | ↓ | **primary** |
| Signal | MAE, RMSE, PCC (normalised space) | ↓↓↑ | secondary |
| Upstream parity | `HeartRateError` (upstream code, unchanged) | ↓ | secondary / comparability |
| Efficiency | NFE, latency (ms per batch of 64, median of 20 after 5 warm-ups, fp32, no compile), samples/s, peak GPU memory | | reported for every arm |
| Conditional fidelity | **PPG-shuffle test**: derange PPG↔ECG pairs within the test set; HR error of the output vs. the *given* PPG's HR (from `nk.ppg_findpeaks`) and vs. the *true* target; conditioning gain = HR-err(mismatched target) − HR-err(matched target) | | diagnostic for H2 vs H2′ |
| Averaging / collapse | per-window std of outputs across 8 noise seeds; template-correlation between seeds | | diagnostic for H2 |

Aggregation: mean over windows within subject → mean ± SD across the 15 test subjects (P1) → mean ± SD across seeds.
Bootstrap 95 % CI over test windows (1000 resamples, stratified by subject). No hypothesis test decides success;
§5 margins do.

## 5. Decision rules (frozen with the tag)
Reference = A0 checkpoint sampled with Heun 25 steps (50 NFE). For arm/NFE *k*, a metric **fails** if the mean over
seeds exceeds its margin:

| Metric | Non-inferiority margin |
|---|---|
| R-peak F1 | drop > **0.02** absolute |
| HR abs error | increase > **1.0 bpm** |
| Template correlation | drop > **0.05** absolute |
| QRS-width error | increase > **10 ms** |

- **H1 confirmed** ⇔ at NFE ≤ 2 (Euler) at least one primary metric fails in ≥ 2 of 3 seeds.
- **H2 (blur)** ⇔ at 1 NFE the morphology metrics fail while R-peak F1 and HR error pass; **H2′ (conditioning)** ⇔
  R-peak F1 or HR error fails, or the PPG-shuffle conditioning gain collapses to < 50 % of its 50-NFE value.
- **H3 confirmed** ⇔ A2 at 1 NFE passes every primary metric against the 50-NFE reference (and A1′ at 50 NFE is
  itself within margin of A0, otherwise the control is invalid and A2 is compared to A1′@50).
- **Stop rule:** if H1 is *not* confirmed (OT-CFM is already fine at 1–2 NFE), A2 is **not** run; the result is
  reported as "one-step is free for this task" and the research question is closed negatively.

## 6. Reproduction acceptance (arm A0)
Paper (arXiv:2602.03858, Table 1): PPG-DaLiA **HR Error = 15.64 bpm** (RDDM 16.43, CycleGAN 23.61, RespDiff 22.75,
PaPaGei-S 40.89; ablation w/o PPG conditioning 24.40). No waveform MAE, no variance, no seeds are reported.
A0 is accepted if upstream-code `HeartRateError` (as_shipped) on the upstream-style 13/1/1 split is **≤ 17.2 bpm (15.64 + 10 %)**
for at least 2 of 3 seeds; because the paper's number rests on a single unknown test subject, the A0 report also lists
the per-subject range across the P1 folds. Otherwise the discrepancy is investigated before A1 (logged, not silently tuned).

**Window length decision (2026-08-25, fixed BEFORE any training, based on the upstream audit — not on results):**
**A0-primary = 8-second windows (1024 samples @ 128 Hz).** Rationale (`docs/PENGUIN_AUDIT.md` §5, §20):
1. upstream `HeartRateError` with `segment_len=4` resamples the 8 s evaluation window to 512 samples → 2× temporal compression,
   HR ≈ doubled (synthetic check: true ~60 bpm → 119.7 bpm on the broken path);
2. high-HR windows then fail beat detection, are masked, and a window with no valid pair scores 0.0;
3. upstream `sample_num = 16181` equals exactly the number of 8 s windows (Σ⌊n_4s/2⌋), i.e. the authors' bookkeeping was done at 8 s;
4. the paper (arXiv:2602.03858) does not state the window length explicitly.
The 4 s configuration remains documented as a paper/code ambiguity; it is **not trained in this stage**. All A0/A1 numbers
use 8 s windows; the HR metric is then computed on the native 8 s window without any resampling pathology.

**Split decision:** upstream's glob-order-dependent 13/1/1 split is *not* used. A0 uses the deterministic subject-disjoint
manifest `data/manifests/split_p0_holdout_seed42.json` (train 13 / val S11 / test S2). Limitation recorded: this is not
guaranteed to be the same held-out subject as the paper's (unknown).

**Seed decision:** A0 is a single-seed (42) feasibility/baseline gate; seeds 43/44 are deferred until a method comparison is
pre-registered separately.

**HR-error reporting rule:** upstream's `HeartRateError` is reported in two variants — `as_shipped` (reproduces upstream's
number, including its 2× compression when segment_len < 8) and `corrected` (no compression) — plus our own HR abs error.
Decision rules in §5 use *our* HR abs error only.

## 7. Things that are explicitly NOT allowed before H1 is settled
New architectures, new losses, new conditioning, changing the preprocessing, tuning the sampler schedule (grids other than
uniform), classifier-free guidance, any use of the dataset's reference `rpeaks` for training.

## 8. Pre-report checklist
`scripts/run_leakage_checks.py` PASS on the manifest used · `tests/` PASS · upstream checkout pinned and clean
(`assert_upstream_pinned`) · every number traceable to an `outputs/` JSON with config hash, seed, commit.

## 9. Deviations
- 2026-08-25 (before training): §6 window-length / split / seed decisions added from the audit. A0 is trained with our own loop
  (`ppg2ecg.training.train_a0`) that mirrors upstream `train.py` step-for-step but reads the manifest split; the upstream model
  class is imported unchanged. Val/test sampling runs under `torch.no_grad()` (numerically identical; upstream omits it).
Log entries go to `docs/EXPERIMENT_LOG.md`.
