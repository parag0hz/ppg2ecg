# D3 — Six-Dataset PENGUIN-Protocol Replication with iMeanFlow: REPORT

Preregistration `docs/D3_PENGUIN_SIX_DATASET_PREREGISTRATION.md` (`f57e4fc`, pushed before any corpus was
built). `D3_START_SHA = 745b1f4`. Training 2026-09-08 14:05→20:22 (6.3 h); evaluation →23:52.
**Descriptive: no gate, no verdict tree.** M2 remains verdict D and M3 remains verdict D.

## 1. The protocol replication is exact where it can be checked

PENGUIN's `sample_num` bookkeeping is reproduced to the window:

| corpus | ours | PENGUIN `sample_num` | ratio |
|---|---|---|---|
| `bidmc_resp_4s` | 6,360 | 6,360 (@4 s) | **1.0000** |
| `uci_bp_8s` | 356,240 | 356,240 (@8 s) | **1.0000** |
| `wesad_resp_4s` | 21,711 | 21,702 (@4 s) | 1.0004 |
| `mimicbp_8s` (existing) | 137,160 | 137,160 | **1.0000** |
| `dalia_8s` (existing) | 16,181 | 16,181 | **1.0000** |

Loaders, label preprocessing and metric definitions are PENGUIN's, ported verbatim (§4–§5 of the
preregistration) and unit-tested, including the upstream asymmetry that low-passes the **prediction only**
in `RespRateError`.

## 2. Headline table — ours (iMeanFlow) across the whole NFE sweep, PENGUIN at its own 50-NFE budget

| dataset | metric | iMF-1 | iMF-2 | iMF-4 | iMF-50 | OT-CFM-50 | **PENGUIN @50** |
|---|---|---|---|---|---|---|---|
| PPG-DaLiA | HR (bpm) | 11.96 | 11.56 | 11.26 | — | **8.16** | 15.64 |
| WildPPG | HR (bpm) | 11.85 | 11.62 | 12.18 | — | **9.43** | 12.97 |
| BIDMC | RR (bpm) | 4.20 | **4.05** | 4.14 | 4.31 | — | **2.98** |
| WESAD | RR (bpm) | 4.99 | 5.10 | 5.12 | **4.90** | — | **4.45** |
| UCI-BP | SBP (mmHg) | 33.61 | 27.21 | 24.94 | **24.28** | — | **12.61** |
| UCI-BP | DBP (mmHg) | 23.73 | 16.09 | 10.23 | **7.87** | — | 7.14 |
| MIMIC-BP | SBP (mmHg) | 16.23 | **14.76** | 14.86 | 15.74 | 15.93 | 17.43 |
| MIMIC-BP | DBP (mmHg) | 21.98 | 15.21 | 12.06 | 10.73 | **9.87** | 11.34 |

DaLiA and WildPPG rows are the frozen A3/A4 arms (reused, not retrained; their own splits, named in §6).
All other cells are new. **Every NFE is published; none was selected after seeing results.**

MIMIC-BP with intervals — the only ABP corpus where the bootstrap is meaningful (229 test subjects):

| arm | SBP (mmHg) | DBP (mmHg) |
|---|---|---|
| iMF NFE 1 | 16.23 [15.43, 17.04] | 21.98 [20.78, 23.21] |
| iMF NFE 2 | **14.76 [14.00, 15.50]** | 15.21 [14.14, 16.29] |
| iMF NFE 4 | 14.86 [14.10, 15.63] | 12.06 [11.13, 13.00] |
| iMF NFE 50 | 15.74 [14.94, 16.56] | 10.73 [9.96, 11.53] |
| OT-CFM NFE 50 | 15.93 [15.20, 16.69] | **9.87 [9.21, 10.57]** |
| PENGUIN | 17.43 | 11.34 |

## 3. What the numbers say — three different stories, not one

**3.1 HR: we are better, and it does not mean much.** 8.16 vs 15.64 on DaLiA and 9.43 vs 12.97 on
WildPPG, at PENGUIN's own 50-NFE budget. But D2 established that **HR error is satisfied by beat rate
alone** — a PPG-peak template scoring R-peak F1 0.061 achieves 1.58 bpm on CapnoBase. These rows are not
evidence of reconstruction quality in either direction.

**3.2 MIMIC-BP: we beat PENGUIN on both metrics, on the one ABP corpus with usable statistics.**
SBP 14.76 vs 17.43 (iMF at NFE 2) and DBP 9.87 vs 11.34 (OT-CFM-50); the iMF NFE-50 DBP of 10.73 also
clears it. With 229 test subjects the intervals are informative, and PENGUIN's own published DBP was the
single cell where their method lost to a baseline (CycleGAN 10.49).

**3.3 The first ABP comparison I drew was unfair, and correcting it changed the picture.** I initially set
our NFE 1–4 against PENGUIN's NFE 50 — a 12× budget gap, exactly the error I had criticised in D1. At
matched budget on UCI-BP, **DBP converges to parity (7.87 vs 7.14, 1.10×)** from 23.73 at NFE 1. The
mechanism is measurable: PENGUIN's config leaves ABP in **raw mmHg** (`label_zscore: False`,
`label_normalize: False`; measured mean 88.8, sd 23.7), so flow matching from `N(0,1)` must integrate
across a ~89 mmHg offset and a ~24× scale gap. Few steps cannot; more steps can.

**But SBP does not converge** — 24.28 vs 12.61 on UCI-BP even at 50 NFE. DBP is the window **minimum**
(the smooth diastolic trough) and SBP the **maximum** (the sharp systolic peak). The budget explains the
diastolic gap; it does not explain the systolic one, which is a morphology limit.

**3.4 RR is a real deficit, not a budget artefact.** BIDMC 4.05–4.31 vs 2.98 and WESAD 4.90–5.12 vs 4.45,
and the metric is flat in NFE (BIDMC 4.20 → 4.05 → 4.14 → 4.31). Nothing here can be attributed to
inference budget. This is the clearest place where the method is simply worse.

## 4. Deviations

**D3-1 — the preregistration's step-budget arithmetic is wrong.** §6 states "66 validation rounds =
14,409 optimizer steps". That equality is **corpus-dependent**: `batch_rounds` defines a round as
`min(epoch, 220 steps)` (`train_a2.py:40-43`), so a corpus whose epoch is shorter than 220 batches yields
fewer steps. Realised: bidmc_resp **4,752** (72 batches/epoch), wesad_resp **8,250** (250 batches/epoch,
alternating 220 + 30), uci_bp **14,331**. 14,409 came from WildPPG, whose 293 k-window epochs always fill
a round. Unlike M2/M3 — where U-vs-S/E pairing required identical steps — D3 compares each arm against a
published number rather than against another arm, so step equality is not load-bearing. The realised
counts are published above.

**D3-2 — iMF at 50 NFE was added after the freeze.** §6 froze "iMF 1/2/4 plus a 50-NFE column supplied by
OT-CFM", but no OT-CFM arm exists for the three new corpora, so the budget match to PENGUIN could not be
made as written. iMF-50 completes it. This is the opposite of cherry-picking — **all of 1/2/4/50 are
published** — and it is what made §3.3 visible.

**D3-3 — UCI-BP's bootstrap is degenerate.** PENGUIN defines UCI-BP as 8 "subjects" (1500-record chunks of
`Part_N.mat`), so the frozen 70/15/15 leaves **one test subject**. The subject-clustered bootstrap
resamples a single cluster and returns an interval equal to the point estimate. **No uncertainty
statement is possible for any UCI-BP cell.**

**D3-4 — two wait loops self-matched and stalled the run for ~50 minutes.** `pgrep -f "d3_evaluate.py
--corpus"` and `until ! pgrep -f d3_eval_all` each matched their own shell's command line, so a tmux job
idled with the GPU at 193 MiB. No result was affected; the loops were removed and the steps chained
inside one script.

**D3-5 — the MIMIC-BP OT-CFM arm needed a second code path.** iMF checkpoints store the `MeanFlowS5`
wrapper (`backbone.`-prefixed keys) and sample on the MeanFlow schedule; OT-CFM arms store the bare
backbone and integrate with upstream Heun (NFE = 2 × steps). The evaluator now detects and handles both.

## 5. Claim boundary

D3 reports **our** numbers under **our** replication of PENGUIN's protocol. It is not a like-for-like
verdict on their published values: `docs/PENGUIN_AUDIT.md` §38 records that the repository ships no
PENGUIN checkpoint or log, so which split, segment length and metric variant produced 15.64 cannot be
determined; the split is filesystem-order dependent and **15.64 rests on one test subject**; and the
shipped `HeartRateError` at `segment_len=4` reports doubled bpm and zero-fills failed windows (§34–35).

Additionally, this replication itself surfaced an internal contradiction in their release: all six
`sample_num` values imply 8-second windows, while `train.py:43` asserts `window_size % segment_len == 0`
with `RespRateError` at 60 s, which 8 s cannot satisfy. Our per-task segment lengths (8 s for ECG and
ABP, 4 s for Resp) are forced by that contradiction, not chosen.

D3 claims no SOTA and no novelty, says nothing about generalisation beyond these corpora and splits, and
licenses no future work.
