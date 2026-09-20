# ED2 — Consensus decoding on the second corpus, and on heart-rate variability (preregistration; no training)

Frozen and pushed before any ED2 number. The decoding algorithm and its parameters are **ED1's, unchanged**
(`bc88553`: iMF w = 50 ms, θ = 0.375, b = 0; CD w = 50 ms, θ = 0.25, b = −1 sample; K = 16, NFE 1).

## Part A — WildPPG (transfer without re-tuning)
WP1 folds and checkpoints (`wp1_f{k}_armI`, `wp1_f{k}_armD`), the same 3,000 windows per test subject as WP1.
Per generator, against its own single sample (WP1 arrays): the ED1 rule — **WORKS** iff R-peak F1 gain ≥ +0.05 and
`morph_corr@GT` gain ≥ +0.15 (CIs excluding 0; cluster = subject, n = 14), **PARTIAL** iff one holds, else **FAILS**.
Reported: RR-MAE, HR, FD, and the contrasts against that fold's PENGUIN-50.

## Part B — Short-term HRV on VitalDB
- **Data.** For every V1 test case, one **contiguous 2-minute block** (30 consecutive 4 s windows) starting at the
  middle of the case; if any window of the block is non-finite or constant the block slides forward by 30 windows, up
  to 10 times, else the case is skipped (counted). Same preprocessing as V1.
- **Intervals.** RR = successive R-peak differences **within a window**, kept if 300–2000 ms (same rule for every arm and
  for the reference). A block needs ≥ 30 intervals on both sides to be scored.
- **Metrics per block.** mean RR, **SDNN** (SD of the pooled intervals), **RMSSD** (RMS of successive-interval
  differences within windows). Error = |estimate − reference|; agreement = Spearman across blocks.
- **Arms.** PENGUIN-50 (one sample, detector on the waveform), iMF-1 and CD-1 (one sample, seed 0), iMF and CD
  **consensus-decoded** (the consensus event positions are the beat sequence), and **PRV** — pulse-interval
  variability straight from the PPG (`find_peaks`, distance 42, prominence 0.3): the estimator that needs no model.
- **Claims (patient-clustered bootstrap, 2,000).** **H-HRV1:** decoding lowers SDNN error against its own single
  sample (CI of the difference below 0), per generator. **H-HRV2:** decoded iMF lowers SDNN error against PENGUIN-50.
  The comparison with **PRV is reported prominently and not gated**: if PRV is the best HRV estimator, the report
  says so in its first paragraph.
