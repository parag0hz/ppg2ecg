# EXP-D — Functional generalisation: PHASE 0 audit (no result numbers)

Audit 2026-09-23. Reproduced by `scripts/expd/expd_audit.py` → `artifacts/exp_d_functional_generalization/{respiration,
abp}/audit.json`. Every model call in the audit used **training-subject PPG**; no generated sample was compared with any
target, and no test-set functional was computed. No training was started.

## Summary table
| Task | Model/checkpoint exists? | Training data | Test data | Physical scale preserved? | Functional extractor exists? | Fixed-depth sampler available? | Usable? |
|---|---|---|---|---|---|---|---|
| Respiration→RR | **Yes** — U1 upstream PENGUIN as shipped: BIDMC (`PENGUIN_BIDMC_u1`, saved epoch 6), WESAD (saved epoch 9) | BIDMC 41 subjects / 4,920 windows; WESAD 13 / 18,719 | BIDMC **6 subjects / 720 windows / 48 blocks of 60 s**; WESAD **1 subject** / 1,605 / 107 blocks | not needed: target is per-window min-max normalised (every window spans exactly [−1, 1]); RR is a frequency and scale-free | **Yes** — upstream `RespRateError` (`help_func.py:190-208`), ported verbatim as `ppg2ecg.evaluation.penguin_metrics.resp_rate_error` (unit-tested) | **Yes** — Euler on upstream's grid (NFE = S, hook-verified), upstream Heun-25 (50 NFE, bitwise = `PENGUIN.sample`) | **BIDMC: yes, low power** (6 test subjects). WESAD: exploratory only (1 test subject) |
| ABP→SBP | **Yes** — U1 MIMIC-BP (saved epoch 17); U1 UCI-BP (saved epoch 9) | MIMIC-BP 1,144 subjects / 240,240 windows | MIMIC-BP **190 subjects / 39,900 windows / 19,950 blocks of 8 s** | **VALID** (mmHg end to end, below) | **Yes** — upstream `SBPError`: block maximum (`help_func.py:164-167`) | Yes | **MIMIC-BP: yes.** UCI-BP: **no** — test subject byte-identical to a training subject (U1-P1) |
| ABP→DBP | as SBP | as SBP | as SBP | **VALID** | **Yes** — upstream `DBPError`: block minimum (`help_func.py:169-172`) | Yes | MIMIC-BP: yes |
| ABP→MAP | as SBP | as SBP | as SBP | **VALID** | **Not upstream.** Defined here without parameters: time-average of the block (labelled *ours*) | Yes | MIMIC-BP: yes (MAP marked as not part of the original protocol) |

**ABP_PHYSICAL_SCALE = VALID.**

## Items 1–17
1. **PENGUIN implementation.** `external/PENGUIN` @ `6cd70cdefb91f10efeb8dce34019b5067cb25344` (unmodified; `git status`
   clean). Class `src/models/PENGUIN.py::PENGUIN` (4,568,707 parameters at the shipped `h_dim` 128, 4 Flow-SSM (S5) blocks).
   Project samplers `ppg2ecg.flow.samplers.{euler_sample, heun_sample}` call the upstream `forward_step`.
2. **PPG→respiration.** Supported upstream (`label: Resp`: BIDMC, WESAD).
3. **PPG→ABP.** Supported upstream (`label: ABP`: MIMIC-BP, UCI-BP).
4. **Checkpoints.** `outputs/u1_upstream/PENGUIN_<dataset>_u1/ckpt/pretrain_ckpt.pth`, written by upstream `train.py`
   run as shipped in U1 (`docs/U1_UPSTREAM_PENGUIN_ASSHIPPED_REPORT.md`, prereg `6c7a646`). sha256: BIDMC `fa8b3d96…`,
   WESAD `946dd2b1…`, MIMIC-BP `02dd37c1…`, UCI-BP `c204d6ae…` (full hashes in `audit.json`). Contents `{epoch, state_dict,
   cfg}` — the full training config is embedded. No EMA weights (upstream saves the online model). Upstream's own
   `load_checkpoint` uses `strict=False`; re-loaded here with **`strict=True`: no missing, no unexpected key** for all four.
   Other resp / ABP checkpoints exist (`u2_{bidmc,mimicbp}_armC_seed42` — PENGUIN in our training loop on U2's splits;
   `d3_*`, `a7_*`, `a8_*` — other models) and are **not used**: the earlier frozen EXP-D plan (`50948e2`, see below) fixed
   the U1 checkpoints, and U1 is the upstream pipeline as shipped (it reproduced the published BIDMC RR and MIMIC-BP
   SBP / DBP values).
5. **Training data (embedded cfg).** `train.dataset` = the corpus of the file, `seed 42`, `fold_num 8`; epoch caps: BIDMC /
   WESAD 300 (early stopping fired), **MIMIC-BP 20** (U1 deviation D2; best epoch 18 of 20, the cap bound), UCI-BP 12.
6. **Splits.** Upstream's own random subject split, recorded in `outputs/u1_upstream/split_<dataset>.json`; subject-disjoint
   for all four. BIDMC 41 / 6 / 6, WESAD 13 / 1 / 1, MIMIC-BP 1,144 / 190 / 190, UCI-BP 6 / 1 / 1 with 4 duplicate groups
   (test `subject2` = train `subject0`, byte-identical).
7. **Sample rates.** Everything resampled to 128 Hz (`scipy.signal.resample`, per window). Raw: BIDMC PPG / resp 125 Hz;
   WESAD PPG 64 Hz, resp 700 Hz; MIMIC-BP 125 Hz.
8. **Window lengths.** 4 s = 512 samples (`segment_len 4`). Metric blocks (upstream `train.py:41-47`): consecutive 4-s windows
   of the test stream concatenated — **RR 60 s = 15 windows**, **SBP / DBP 8 s = 2 windows**. Every test subject's window
   count is divisible by the block length (BIDMC 120, WESAD 1,605, MIMIC-BP 210), so no block crosses a subject. In
   MIMIC-BP each subject is 30 records of 30 s (7 windows each); 15 of a subject's 105 8-s blocks join the last window of
   one record with the first of the next (upstream behaviour, kept).
9. **Preprocessing (`src/preprocess.py`).** PPG: band-pass 0.5–4 Hz, z-score, min-max to [−1, 1], per window. Resp label:
   low-pass 1 Hz, z-score, min-max to [−1, 1], per window. **ABP label: no band-pass, no z-score, no normalisation**
   (paper: amplitude carries physiological meaning).
10. **Normalisation.** As above; parity with our port `ppg2ecg.data.preprocess` was tested in D3 / U1.
11. **Inverse normalisation.** ABP: none needed — the model is trained on, and outputs, mmHg. Resp: none needed for a rate.
12. **Official evaluation.** `help_func.py::compute_metrics`: `RespRateError` (prediction only low-passed at 1 Hz, order-8
    Butterworth, `filtfilt`; dominant positive FFT frequency of the 60-s block × 60; reference unfiltered), `SBPError` /
    `DBPError` (|max − max| / |min − min| of the 8-s block, mmHg), `FD` (raw-signal Fréchet distance). No MAP, no beat-level
    SBP / DBP, no respiratory-event metric exist upstream.
13. **Official sampler.** `PENGUIN.sample`: Heun, `n_step 25`, uniform grid `linspace(0, 1, 26)`, x₀ ~ N(0, I).
14. **NFE accounting (forward hook on `final_layer`, once per vector-field call).** Upstream sampler: 50 calls. Our
    `heun_sample(25)`: 50 calls, output **bitwise identical** to `PENGUIN.sample` for the same noise (max |diff| = 0 on all
    four checkpoints). Euler S ∈ {1, 2, 4, 8, 16, 32}: exactly S calls. No encoder / decoder exists outside the vector
    field, so **a fixed vector-field budget is a fixed network-compute budget** for PENGUIN (unlike PPGFlowECG).
    Latency (RTX 5090): batch 1 **20.5 ms per NFE** (652 ms at 32 NFE; Heun-25 1,020 ms); batched 512, 0.64 ms per NFE per window.
15. **Target waveforms.** Stored with the inputs (`data/processed/upstream_u1/<dataset>/subject*.pkl`: `x_data`, `y_data`
    [n, 512]).
16. **Patient / subject IDs.** One file per subject; the subject is the clustering unit.
17. **Cached generated samples.** None for EXP-D (`outputs/*expd*` absent). U1's `plot/` folders hold 20 qualitative test
    figures per dataset from the shipped sampler (PNG only). Known test numbers before this audit: U1's shipped-sampler test
    values (BIDMC RR 3.479 bpm; MIMIC-BP SBP 14.986, DBP 9.212 mmHg) — single upstream run, printed by `train.py`.

## ABP physical-scale check
| question | answer | source |
|---|---|---|
| original mmHg preserved? | **yes** — MIMIC-BP ABP stored as read by wfdb (mmHg); upstream applies only the per-window FFT resample 125 → 128 Hz | `config/preprocess.yaml` (`label_bandpass/zscore/normalize: False`); A7 dataset audit |
| normalisation parameters stored? | not applicable — there is no normalisation | same |
| inverse transform implemented / needed? | not needed; the loss is on mmHg velocities and the sampler returns mmHg | `PENGUIN.train_flow` (x₁ = raw target) |
| model output in mmHg? | yes: training-subject smoke outputs 57–129 mmHg; training targets (30 train subjects) 30.6–175.6 mmHg, per-window max median 112.9, min median 56.7 | `audit.json` |
| reference on the same scale? | yes — the same processed target array | — |

Caveats that do not invalidate the scale: the per-window FFT resample can ring at window edges and move a window's max / min
slightly — identically for training targets, references and the upstream metric; SBP / DBP here are upstream's block
max / min, not beat-level clinical values; A8 showed that in raw mmHg the flow's noise carries almost none of the
interpolant energy (‖y‖/‖e‖ ≈ 82), so samples may be nearly redundant — the smoke test shows draw-to-draw differences of
2–6 mmHg RMS on training inputs (S = 1 → 32), i.e. not deterministic.

## Relation to the earlier EXP-D plan
`docs/TOP_TIER_COMPLETION_PREREGISTRATION.md` (`50948e2`) contains an EXP-D design (B = 50 grid, validation-selected arm,
comparisons against the shipped sampler) that was **never executed** (no raw outputs, no numbers). The present request
replaces it with a different, fixed design (B = 16 / 32, preregistered width / intermediate / depth cells, K = 16
mechanism). The old section is not edited; the new preregistration states the replacement. What carries over unchanged:
the U1 checkpoints, upstream splits, upstream functionals, BIDMC primary / WESAD exploratory, MIMIC-BP primary.
