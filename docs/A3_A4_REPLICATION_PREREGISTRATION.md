# A3 / A4 Replication Pre-registration — does the frozen one-step iMeanFlow effect replicate?

Written 2026-08-26 **before any A3/A4 training**. Frozen by the commit that introduces this file; A4's dataset-specific values
(channel, split, preprocessing, schedule unit) are frozen by a second commit ("Part II freeze") **after the WildPPG audit and
before any A4 training**. No test result of A3 or A4 existed when the corresponding freeze was committed.

## 0. Frozen components (unchanged from A0-b / A2, commit `219a0b2`)
S5/Flow-SSM backbone (upstream PENGUIN `6cd70cd`, 4,568,707 parameters) · PPG conditioning path · OT-CFM implementation
(`train_a0`, upstream objective) · Improved MeanFlow implementation (`ppg2ecg.flow.imeanflow`, official `Lyy-iiis/imeanflow` `bf60cd7`
semantics, **h-only interval conditioning**, boundary v_θ, adaptive weight p = 1 / c = 0.01, (t, r) logit-normal(−0.4, 1), 50 % r = t,
forward-mode JVP, micro-batch 32 × 2) · AdamW lr 1e-3 / wd 0.01 / effective batch 64 / fp32 / no EMA · checkpoint selection by the
deterministic fixed-bank validation loss (4 banks, seed 1000, min_delta 1e-4, patience 20, max 300) · evaluation code
(`eval_a0_nfe_curve.py`, `eval_a2.py`), paired noise seed 0, PPG-shuffle derangement seed 1, recovery-score definition, NFE
counting (Heun = 2 evaluations/step) · **seed 42 for every new training**. Forbidden: new architecture/loss/conditioning,
hyper-parameter changes or sweeps, seeds 43/44, any change after a test result is seen.

Research questions: **RQ1** does the one-step recovery persist on another unseen PPG-DaLiA subject? **RQ2** does it replicate on a
different PPG–ECG dataset? This stage develops nothing new.

---
## Part I — A3: new PPG-DaLiA held-out subject

### 1. Split (frozen now, before any S1 result)
`data/manifests/split_a3_testS1_valS11.json` (sha256 `6d2999bd…`): **test = S1**, **val = S11**, train = S2–S10, S12–S15 (13 subjects;
14,047 train / 1,131 val / 1,151 test windows of 8 s). Rule: the new test subject is the lowest-numbered subject that was never a
val/test subject before (S1); S11 is kept as validation so that the checkpoint-selection environment (banks, n_val, hash) is identical
to A0-b/A2. Leakage gate (`scripts/run_leakage_checks.py`): subject-disjoint PASS, window-hash-disjoint PASS, window-local
normalisation PASS. Disclosure: *A3 is a subject-robustness replication, not a completely independent confirmatory study, because
the overall method was developed using the previous S2/S11 protocol.* S1 will not be swapped for another subject whatever the result.

### 2. Models (trained from scratch, seed 42)
- `a3_otcfm_ppgdalia_testS1_seed42` — A0-b recipe verbatim (`ppg2ecg.training.train_a0 --select fixed_cfm --min-delta 1e-4 --patience 20
  --n-val-banks 4 --bank-seed 1000 --val-mae-every 0 --gen-diag-every 5`); the only change is the manifest.
- `a3_imeanflow_ppgdalia_testS1_seed42` — A2 recipe verbatim (`ppg2ecg.training.train_a2 --cond-mode h_only --h-scale 1 --micro-batch 32
  --val-batch 32 --patience 20 --min-delta 1e-4 --n-val-banks 4 --bank-seed 1000 --gen-diag-every 1`); only the manifest changes.
  No fine-tuning from A2.

### 3. Evaluation (identical code and seeds)
OT-CFM checkpoint: Heun 25/10/5/2/1 steps (50/20/10/4/2 NFE) and Euler 1 (1 NFE) — **primary arms 50 / 4 / 1**. iMF checkpoint:
**1 NFE primary**, 2 and 4 steps diagnostic. Metrics: HR error, template correlation, amplitude ratio, conditioning gain (PPG
shuffle), beats/reference, RMSE, MAE, latency, NFE (+ secondary diagnostics). Recovery scores exactly as A2 (prereg A2 §5).
Qualitative windows: A3-OT-CFM 50-NFE HR-error 10/50/90 % quantiles + fixed positions {0, n/4, n/2} of **S1**, chosen by rule.

### 4. A3 verdict (frozen)
Let P = {HR error, template correlation, amplitude ratio, conditioning gain}; "improved" = iMF-1 better than OT-CFM-1 on the metric.
- **REPLICATED**: improved on ≥ 3 of 4 metrics in P **and** no severe negative recovery (recovery ≥ −0.25 on all four).
- **PARTIAL**: improved on ≥ 2 metrics but a core physiological metric does not replicate (fails the above).
- **NOT REPLICATED**: improved on ≤ 1 metric, or the iMF-1 output shows the collapse signature (amplitude ratio < 0.5 and
  beats/reference < 0.7).
The A2 SUCCESS/PARTIAL/FAIL recovery rule is reported alongside for comparability. The **pointwise-error inversion** (OT-CFM-1
better RMSE/MAE than OT-CFM-50 and iMF-1 while physiology collapses) is recorded as a secondary observation, yes or no.

---
## Part II — A4: WildPPG replication (rules frozen now; values frozen in the Part II amendment)

### 5. Dataset rules
WildPPG (PPG + ECG, used by PENGUIN for ECG reconstruction). `docs/WILDPPG_AUDIT.md` must establish: official source, paper, licence,
download method, subjects/recordings, PPG sites/channels, ECG channel, sampling rates, synchronisation, activities, missing data,
artefact handling, PENGUIN's actual channel/site choice, preprocessing, window length, split, normalisation, paper-vs-code gaps.
Download only from the official source; if login/consent/forms are required, **A4 = BLOCKED** and the exact user steps are reported;
no unofficial mirrors. Raw checksums + manifest are generated.
**PPG channel rule**: (1) the channel/site PENGUIN's official code explicitly selects; else (2) the paper's protocol; else (3) one
deterministic choice pre-registered before any result. Never chosen after looking at results.

### 6. Preprocessing, split, leakage
Start from PENGUIN's WildPPG preprocessing; execution bugs are patched only in our namespace (`configs/upstream`, documented).
Target input 8 s @ 128 Hz unless the PENGUIN/WildPPG protocol differs (dataset-specific preprocessing is allowed but must be
identical for OT-CFM and iMF). Split: PENGUIN's official split if reproducible; otherwise a deterministic subject-level split
(sorted subject ids, `random.Random(42)`, hold-out sizes mirroring PENGUIN's `subject_num // fold_num`), manifest + hash
pre-registered; **never window-level**. Hard leakage gate before training: subject-disjoint, recording/window-disjoint,
duplicate-window hash, window-local normalisation, no target statistics in the input path, resampling alignment (PPG vs ECG
window counts), waveform-length sanity, NaN/constant-window check. FAIL ⇒ no training.

### 7. Models and schedule unit
`a4_otcfm_wildppg_seed42` then `a4_imeanflow_wildppg_seed42`, same recipes as Part I. **Schedule-unit rule (frozen now):** the
A0-b/A2 epoch on PPG-DaLiA is ≈ 220 optimizer steps (14 k windows / 64). For A4, one *validation round* = min(one epoch, 220
optimizer steps); the fixed-bank validation metric, early stopping (patience 20 rounds, min_delta 1e-4) and the 300-round cap
are applied per round. Same rule for both objectives. Effective batch 64 is kept if VRAM allows (micro-batching as needed).
The 50-NFE baseline result stands whatever it is unless it is a NaN / leakage / implementation failure.

### 8. Evaluation, tables, verdicts
Same arms/metrics/recovery as Part I. Cross-dataset table (DaLiA S2 / DaLiA S1 / WildPPG) with OT-50, OT-1, iMF-1 for HR and
morphology plus per-dataset recovery. The scientific test is the **ordering** A: OT-1 ≪ OT-50, B: iMF-1 ≫ OT-1, C: iMF-1 → OT-50,
not absolute numbers. Integrated verdict: STRONG REPLICATION (A3 replicated and A4 replicated) / SUBJECT-ROBUST, DATASET-UNCERTAIN
(A3 replicated, A4 partial/fail) / DATASET-ROBUST, SUBJECT-UNCERTAIN (A4 replicated, A3 partial/fail) / MIXED (direction present,
inconsistent) / NOT ROBUST (only A2). Recorded as it comes out.

### 9. Not done in this stage
seeds 43/44, 5-fold, new architecture/loss, PPG alignment modules, temporal-shift correction in the model input, hyper-parameter search.
