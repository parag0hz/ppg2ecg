# D3 — Six-Dataset PENGUIN-Protocol Replication with iMeanFlow

**Status: PREREGISTRATION. Frozen on commit. Never edited post-hoc.**
Written 2026-09-08, BEFORE any D3 corpus is built, any D3 model is trained, and any D3 metric is computed.

## 1. Question

PENGUIN's Table 1 reports one metric per dataset across six corpora. We use PENGUIN's backbone
(`external/PENGUIN` @`6cd70cd`, unmodified) and its preprocessing. **Running their protocol ourselves,
what does iMeanFlow score on the same six datasets and the same four metrics?**

This is a **protocol replication with our objective substituted**, not a head-to-head against their
published numbers (§9).

## 2. Provenance

`D3_START_SHA` = `745b1f483118c28826f357de0a0f5f38dcca12ca` (the pushed M3 result).
PENGUIN `6cd70cdefb91f10efeb8dce34019b5067cb25344` · iMeanFlow `bf60cd7cb653f6628e59d48034b333c5eba445e2` ·
A4 md5 `31c042d291052fbb6dc15263ad316be2` · E2 contract sha256
`06e869412114e1efb9ab6624540aebc141495e2690150f4e512df7676c5a9115` · C2 deferred.
M2 verdict D and M3 verdict D stand unchanged; neither is reopened.

## 3. A decisive finding about PENGUIN's own configuration, recorded before use

`external/PENGUIN/config/preprocess.yaml` sets `segment_len: 4`, but every dataset's `sample_num` equals
`duration × 3600 / 8`, not `/4`:

| dataset | duration (h) | `sample_num` | h·3600/4 | **h·3600/8** |
|---|---|---|---|---|
| WildPPG | 867.15 | 390,216 | 780,435 | **390,218** |
| PPG-DaLiA | 35.96 | 16,181 | 32,364 | **16,182** |
| UCI-BP | 791.64 | 356,240 | 712,476 | **356,238** |
| MIMIC-BP | 304.80 | 137,160 | 274,320 | **137,160** (exact) |
| WESAD | 24.11 | 10,851 | 21,699 | **10,850** |
| BIDMC | 7.07 | 3,180 | 6,363 | **3,182** |

**All six agree with 8 s.** This resolves the "highest-impact ambiguity" recorded in
`docs/PENGUIN_AUDIT.md` §7: PENGUIN's own bookkeeping is 8-second windows throughout, and `segment_len: 4`
is the inconsistent field. It is also the only setting under which their `HeartRateError` measures true
bpm rather than doubled bpm (audit §34).

**But their config is internally contradictory.** `train.py:43` asserts `window_size % segment_len == 0`,
and `RespRateError` has `window_size: 60`. Since `60 % 8 = 4`, **the respiratory metric cannot run at 8 s
at all**; it requires `segment_len ∈ {4,5,6,10,12,15,20,30,60}`.

**Frozen resolution.** Per task, choose the segment length that is consistent with both the bookkeeping
and the metric:

| task | datasets | segment_len | why |
|---|---|---|---|
| ECG → HR error | PPG-DaLiA, WildPPG | **8 s** | matches `sample_num`; `60`-independent; the only setting where HeartRateError is unbiased |
| ABP → SBP/DBP error | UCI-BP, MIMIC-BP | **8 s** | matches `sample_num`; `window_size 8 % 8 == 0` ✓ |
| Resp → RR error | BIDMC, WESAD | **4 s** | `window_size 60 % 4 == 0` ✓; 8 s is impossible for this metric |

The resp corpora will therefore not match PENGUIN's 8-s `sample_num` bookkeeping. That is a consequence of
their contradiction, not a choice of ours, and it is disclosed in every resp row of the report.

## 4. Data and preprocessing — PENGUIN's, unmodified

Loaders exactly as `external/PENGUIN/src/utils/load_data.py`:
BIDMC `bidmc_csv/bidmc_NN_Signals.csv` columns `" PLETH"`/`" RESP"` @125 Hz ·
WESAD `S*/S*.pkl` `signal.wrist.BVP` @64 Hz / `signal.chest.Resp` @700 Hz ·
UCI-BP `Part_N.mat` col 0 PPG / col 1 ABP @125 Hz ·
MIMIC-BP `ppg/*.npy`, `abp/*.npy` @125 Hz (already built as `mimicbp_8s`, 137,160 windows = exact match).

Preprocessing is `ppg2ecg.data.preprocess.preprocess_windows`, the line-for-line restatement of their
`preprocess()`. PPG always `bandpass (0.5, 4)`, z-score, min-max. Labels per their `preprocess.yaml`:

| label | bandpass | freq_range | zscore | normalize |
|---|---|---|---|---|
| ECG | yes | `[0.5, -1]` high-pass | yes | yes |
| **Resp** | yes | `[-1, 1]` **low-pass 1 Hz** | yes | yes |
| **ABP** | **no** | — | **no** | **no** — stays in raw mmHg so SBP/DBP are in mmHg |

## 5. Metrics — PENGUIN's `compute_metrics`, reimplemented exactly

- **HR error (bpm)**: existing frozen implementation, corrected variant (audit §34–35); 8 s window.
- **SBP error (mmHg)** = `mean |max(pred) − max(target)|` per window; window 8 s.
- **DBP error (mmHg)** = `mean |min(pred) − min(target)|` per window; window 8 s.
- **RR error (breaths/min)**: Butterworth **order 8** low-pass at 1 Hz on the *prediction only*, then FFT of
  both; take the positive-frequency bin of maximum magnitude; `bpm = 60 · f`; error = `|pred_bpm − target_bpm|`.
  Window **60 s = 15 consecutive 4-s segments**, exactly `train.py:41-47`.

These are ports, not redesigns; each is unit-tested against a hand-computed reference before use.
**Our `rr_mae_ms` is a different quantity** (R-R interval error in ms) and is never reported as RR error.

## 6. Arms

| arm | what |
|---|---|
| **iMF-1 / iMF-2 / iMF-4** | iMeanFlow at NFE 1, 2 and 4 — the proposed few-step method |
| **OT-CFM-50** | the 50-NFE flow baseline, PENGUIN's own inference budget |

**All four are reported for every dataset. No NFE may be selected after seeing results** — the "best
iMeanFlow" is not chosen post hoc; the whole sweep is published. PENGUIN's number is quoted alongside at
its 50-NFE budget.

Training recipe is the frozen one, unchanged: seed 42, AdamW lr 1e-3, wd 0.01, batch 64, micro-batch 32,
h_dim 128, blocks 4, ssm_ratio 2.0, mlp_ratio 2.0, `cond_mode h_only`, p_mean −0.4, p_std 1.0,
data_proportion 0.5, norm_p 1.0, norm_eps 0.01, jvp forward, **66 validation rounds = 14,409 optimizer
steps**, early stopping disabled, checkpoint selection `fixed_imf_mse` (4 banks, bank_seed 1000,
min_delta 1e-4) **only** — never by HR, RR, SBP, DBP or any D3 table metric.

New trainings: **UCI-BP (ABP)**, **BIDMC (Resp)**, **WESAD (Resp)**. Reused without retraining:
MIMIC-BP (`outputs/a7_imeanflow_mimicbp_seed42`), WildPPG (A4), PPG-DaLiA (A3/A0-b).

## 7. Splits

Subject-level, seed 42, 70/15/15 by the frozen `d1_common` rule, one manifest per corpus committed before
training. **WildPPG `kjd`/`ssx` are never loaded.** For reused arms the original frozen split is kept and
named in the report.

## 8. Analysis plan

The headline artefact is one table: rows = (dataset × metric), columns = PENGUIN published · OT-CFM-50 ·
iMF-1 · iMF-2 · iMF-4. Uncertainty: subject-clustered bootstrap, 2,000 replicates, seed 20260904, the
frozen procedure. No hypothesis test and no gate — **D3 is descriptive**. There is no verdict tree; D3
cannot pass or fail, and no future work is licensed or blocked by its numbers.

## 9. Claim boundary — what this table may and may not say

D3 reports **our** numbers under **our** replication of PENGUIN's protocol. It may **not** be presented as
a like-for-like defeat or victory over PENGUIN's published values, because those values are not
verifiable: `docs/PENGUIN_AUDIT.md` §38 records that the repository ships no PENGUIN checkpoint or log, so
which split, segment length and metric variant produced 15.64 cannot be determined; the split is
filesystem-order dependent and **15.64 rests on one test subject**; and the shipped `HeartRateError` at
`segment_len=4` reports doubled bpm and zero-fills failed windows (§34–35).

Every published-number cell therefore carries an explicit non-comparability note. Additionally, D2
established that **HR error is satisfied by beat rate alone** — a PPG-peak template with F1 0.061 scores
1.58 bpm on CapnoBase — so no HR row may be read as evidence of reconstruction quality.

D3 claims no SOTA, no novelty, no causal or observability claim, and says nothing about generalisation
beyond the corpora and splits used. M2 remains verdict D and M3 remains verdict D.

## 10. Deviations

Recorded in a dated section of the D3 **report**, never by editing this document. Two are already known:
the per-task segment length forced by PENGUIN's own contradiction (§3), and the reuse of existing frozen
ECG/MIMIC-BP arms rather than retraining them.
