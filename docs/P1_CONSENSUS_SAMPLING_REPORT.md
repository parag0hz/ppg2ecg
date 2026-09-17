# P1 — Consensus sampling on validation subjects: **NO-GO** (frozen rule)

Prereg `8e7cdad`, amendment 1 `b7f45a4`, both pushed before any number. U2 validation subjects only (DaLiA
`subject8`/`subject10`, WildPPG `e61`/`qm9`), 4,096 windows per corpus, U2 `checkpoint_last.pt`. No test subject
loaded, no weight update. Values: subject-macro HR abs error (bpm, ↓) and R-peak F1 @ 50 ms (↑); K = samples pooled;
cost = NFE per window. Raw: `artifacts/p1_consensus/*.json`.

| arm | cost | DaLiA HR | DaLiA F1 | WildPPG HR | WildPPG F1 |
|---|---|---|---|---|---|
| PENGUIN NFE 50, K=1 | 50 | 14.81 | 0.156 | 14.15 | 0.351 |
| PENGUIN NFE 50, K=4 | 200 | 12.35 | 0.152 | 11.05 | 0.404 |
| PENGUIN NFE 1, K=1 | 1 | 41.48 | 0.063 | 19.04 | 0.415 |
| PENGUIN NFE 1, K=16 | 16 | 38.08 | 0.004 | 17.10 | 0.404 |
| iMF NFE 1, K=1 | 1 | 18.92 | 0.149 | 12.89 | 0.355 |
| iMF NFE 1, K=4 | 4 | 16.20 | 0.143 | 9.61 | 0.397 |
| **iMF NFE 1, K=16** | 16 | 15.20 | 0.012 | **8.66** | 0.347 |
| iMF NFE 2, K=1 | 2 | 17.94 | 0.149 | 14.17 | 0.354 |
| iMF NFE 2, K=4 | 8 | 15.31 | 0.144 | 11.30 | 0.406 |
| **iMF NFE 2, K=16** | 32 | **14.53** | 0.014 | 10.39 | 0.350 |

## GO rule (both corpora; HR −10 % at K=16 AND consensus F1 not lower by > 0.01)

| | HR change K=16 vs K=1 | F1 change | pass |
|---|---|---|---|
| DaLiA iMF NFE 1 / 2 | −19.7 % / −19.0 % | −0.138 / −0.135 | **no** (F1) |
| WildPPG iMF NFE 1 / 2 | −32.8 % / −26.7 % | −0.008 / −0.004 | yes |

**Verdict: NO-GO** — DaLiA fails the F1 condition.

## What happened (diagnosis after the verdict; does not change it)
- DaLiA consensus-peak collapse is not a code defect. On 40 checked windows iMF samples place ~5.1 peaks per
  window, close to the reference 5.7, but the samples do not agree on where: few ±50 ms boxes reach 8 of 16 votes
  (0.3 consensus peaks per window at K=16 vs 5.3 at K=4, where 2 chance votes suffice). PENGUIN NFE 1 outputs are
  flat (1.6 peaks per window). WildPPG samples agree, so its consensus F1 holds.
- HR-median consensus lowered HR error on both corpora. At lower cost than one PENGUIN-50 sample, iMF K=16 is
  below it on WildPPG (8.66 at 16 NFE vs 14.15 at 50 NFE) and level on DaLiA (14.53 at 32 NFE vs 14.81).
  PENGUIN also improves with K (K=4, 200 NFE). These are observations on two validation subjects per corpus, not
  claims; an HR-only consensus hypothesis would be a new, post-hoc preregistration.
