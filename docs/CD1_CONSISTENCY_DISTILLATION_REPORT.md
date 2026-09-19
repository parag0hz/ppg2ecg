# CD1 — Consistency distillation: **IMPROVES over iMF on HR; better than PENGUIN-50 at 1 NFE; best consensus**

Prereg `1cc6582`, frozen before any CD1 weight update. Seed 42. Student initialised from arm C, 14,000 steps of
consistency distillation against arm C's own Heun steps (lr 1e-3 — the declared fallback was not needed), 57.8 min.
Evaluated with the SR1 protocol, paired with the seed-42 arrays of C, I, S, R. Raw: `artifacts/cd1_consistency/`
(the verdict keys keep the `RF_` prefix of the shared verdict script; here they describe arm D).

| arm (1 NFE unless stated) | HR error ↓ | R-peak F1 ↑ | RR-MAE (ms) ↓ | FD ↓ | K = 16 consensus HR ↓ | HR win rate vs PENGUIN-50 |
|---|---|---|---|---|---|---|
| PENGUIN 50 NFE (teacher) | 9.686 | 0.645 | 23.8 | 4.30 | — | — |
| **consistency distillation (D)** | **8.234** | **0.674** | 23.8 | 4.99 | **6.229** | **0.766** |
| iMF (I) | 10.139 | 0.674 | **20.8** | **4.13** | 6.675 | 0.407 |
| small vanilla DiT (S) | 8.712 | 0.629 | — | 7.84 | 6.548 | 0.691 |
| reflow 2-RF (R) | 9.942 | 0.644 | 24.3 | 4.25 | 6.860 | 0.446 |
| PENGUIN 1 NFE | 8.472 | 0.756 | 8.8 | 33.50 | 7.438 | 0.684 |

## Verdicts (frozen rules, NFE 1)
- **CD vs iMF: IMPROVES.** HR **−1.903 [−2.039, −1.769]** (beyond the 1.0 bpm margin); F1 −0.00004 [−0.0020, +0.0018] (tie).
- **CD vs PENGUIN-50: NON-INFERIOR — and better on both endpoints.** HR −1.451 [−1.576, −1.325]; F1 +0.029 [+0.028, +0.031].

## What it means
1. **Among the one-step generators, consistency distillation is the strongest on beat rate**, and it ties iMF on beat
   placement. It beats its own 50-NFE teacher at 1 NFE on HR and F1 — the first arm in the programme to do both.
2. **It also gives the best consensus** (6.23 bpm), confirming again that consensus is a property of cheap samples and
   helps every generator; the ordering of generators is preserved under consensus.
3. **iMF keeps two advantages only:** no teacher (CD needs a trained PENGUIN; its training here was 58 min on top of
   arm C's 52 min), and the better waveform distribution and interval accuracy (FD 4.13 vs 4.99, RR-MAE 20.8 vs 23.8 ms).
4. **For the paper, the generator is not the contribution.** Distillation from PENGUIN is a better one-step HR model
   than iMF; the defensible contribution is the consensus inference and the evaluation around it, shown across
   generators (iMF, reflow, consistency, small DiT).
5. The CD training loss rose from 0.25 (step 1,100) to 0.39 (step 14,000). With a moving EMA target this is not a
   convergence curve, but it is recorded; a longer run or a different target schedule may change CD's numbers.
6. Single seed for arm D, as for R. The SR1-replicated arms are C, I, S.
