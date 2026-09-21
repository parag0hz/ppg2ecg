# LW1 — Weighted auxiliary endpoint losses: **single-sample event metrics IMPROVE for all four arms; diversity, FD and the consensus gain collapse**

Prereg `28ebc40`, frozen before any LW1 weight update. Seed 42, V1 VitalDB recipe (14,000 steps),
`L = L_iMF + 0.5·L_aux` on the training-only one-step endpoint. Test = 1,156 patients / 19,543 windows, NFE 1,
paired patient-clustered bootstrap against arm I. Raw: `artifacts/lw1_aux_loss/verdict.json`.

| arm | L_aux | HR ↓ | F1 ↑ | RR-MAE ↓ | MAE ↓ | PCC ↑ | FD ↓ | sample diversity (HR SD) | K=16 consensus HR ↓ | consensus gain | verdict (frozen rule) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I (reference) | — | 10.14 | 0.674 | 20.8 | 0.403 | 0.228 | **4.13** | **9.90** | **6.68** | 3.47 | reference |
| W0 | 1 − PCC | 8.00 | 0.742 | **8.6** | 0.424 | **0.398** | 34.82 | 4.63 | 7.86 | 0.15 | IMPROVES |
| W5 | 0.5·MAE + 0.5·(1 − PCC) | 9.59 | 0.727 | 8.7 | **0.304** | 0.382 | 25.97 | 7.10 | 8.11 | 1.48 | IMPROVES |
| W10 | MAE | 10.64 | 0.719 | 9.5 | 0.311 | 0.371 | 22.96 | 8.94 | 8.48 | 2.16 | IMPROVES |
| SP | multi-resolution STFT | **7.64** | **0.753** | 13.1 | 0.321 | 0.311 | 23.31 | 4.29 | 6.94 | 0.70 | IMPROVES |

Paired differences vs I: F1 +0.068 / +0.053 / +0.045 / +0.079 (all CIs above +0.04); HR −1.61 / −0.39 / +0.71 / −2.43;
consensus HR **+1.19 / +1.44 / +1.81 / +0.26** (all CIs above 0 — worse).

## Against the expectation stated in the preregistration
- **W arms — confirmed.** MAE / PCC improve, diversity shrinks (9.9 → 4.6–8.9), the consensus gain shrinks (3.47 → 0.15–2.16)
  and FD rises (4.1 → 23–35).
- **SP — contradicted.** The spectral loss was expected to spare diversity; it collapsed it the most (4.29) and sent FD
  to 23.3. Any endpoint-matching term, phase-sensitive or not, pulls the one-step map towards a deterministic function
  of the PPG.

## What it means
1. **The auxiliary losses turn the generator into a point estimator.** The resulting profile — F1 0.72–0.75, RR-MAE ≈ 9 ms,
   low MAE, FD 23–35, little sample diversity — is the profile of PENGUIN at 1 NFE (F1 0.756, RR-MAE 8.8, FD 33.5).
   Two regimes exist: *distributional* generators (realistic samples, diversity, consensus works) and *point-estimate*
   models (best single-shot event metrics on this corpus, no diversity, unrealistic waveform distribution).
2. **On VitalDB a point estimator at 1 NFE is about as good on events as consensus decoding at 16 NFE**: SP single sample
   F1 0.753 / HR 7.64 / RR-MAE 13.1 vs iMF consensus-decoded F1 0.765 / HR 6.94 / RR-MAE 10.8 (and W0 has the lower
   RR-MAE, 8.6). The paper cannot present consensus decoding as the only route to good event metrics.
3. **What the point estimators lose:** the best HR of the programme (consensus 6.2–6.7 vs 7.6–10.6), and the waveform
   distribution (FD). They also cannot benefit from pooling (gain ≤ 2.2 bpm against 3.5).
4. **α reads cleanly:** moving from PCC (α = 0) to MAE (α = 1) makes HR and F1 worse (8.00 → 10.64, 0.742 → 0.719) and FD /
   diversity better. The correlation term is what buys the event metrics — and what costs the distribution.
5. **Open question, not tested here:** VitalDB is regular rhythm under anaesthesia, where a mean-like waveform keeps the
   beat. On the wearable corpora PENGUIN-1 (the existing point estimator) was the *worst* arm (WildPPG HR 15.4, DaLiA
   37.6) while consensus held. Whether these auxiliary-loss models inherit that failure needs a WildPPG run.
6. Single seed.
