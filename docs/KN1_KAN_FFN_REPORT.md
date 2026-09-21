# KN1 — KAN feed-forward in the PENGUIN backbone: **no meaningful change for iMF, worse for OT-CFM**

Prereg `84413a3`, frozen before any KN1 weight update. Seed 42, V1 VitalDB recipe (14,000 steps), test = 1,156
patients / 19,543 windows, SR1 protocol, paired patient-clustered bootstrap against the MLP arm of the same
objective. KANLinear as KANFlow describes it (RBF, G = 5, bounded input). Raw: `artifacts/kn1_kan/verdict.json`.

| arm | KAN placement | HR ↓ | R-peak F1 ↑ | RR-MAE ↓ | MAE ↓ | RMSE ↓ | FD ↓ | K=16 consensus HR | verdict |
|---|---|---|---|---|---|---|---|---|---|
| iMF, MLP (arm I) | — | 10.139 | 0.674 | 20.8 | 0.403 | 0.493 | 4.13 | 6.675 | reference |
| iMF, KAN outer | first + last block | 10.078 | 0.664 | 21.8 | 0.419 | 0.512 | **3.22** | 6.538 | **NO MEANINGFUL CHANGE** |
| iMF, KAN all | all four blocks | 10.181 | 0.673 | 20.4 | 0.409 | 0.496 | 4.10 | 6.381 | **NO MEANINGFUL CHANGE** |
| OT-CFM, MLP (arm C), NFE 50 | — | 9.686 | 0.645 | 23.8 | 0.396 | 0.497 | 4.30 | — | reference |
| OT-CFM, KAN outer, NFE 50 | first + last block | 11.047 | 0.618 | 21.4 | 0.395 | 0.496 | 4.04 | — | **WORSE** |

Paired differences (KAN − MLP): iMF outer HR −0.06 [−0.16, +0.03], F1 −0.010 [−0.012, −0.009]; iMF all HR +0.04
[−0.05, +0.14], F1 −0.001 [−0.003, +0.000]; OT-CFM outer HR **+1.36 [+1.27, +1.46]**, F1 **−0.027 [−0.029, −0.025]**,
MAE −0.0010 [−0.0018, −0.0002], RMSE −0.0012 [−0.0020, −0.0003].

## Reading it
1. **KAN does not move the event metrics for iMF in either placement.** Every HR / F1 difference is an order of
   magnitude below the preregistered margins. This is the sixth backbone-side change in the programme (4× width,
   attention mixer, official DiT, ImageNet pretraining, reflow initialisation aside) that leaves HR / F1 where they
   were: the ceiling is the information in the PPG, not the function class of the FFN.
2. **For OT-CFM the KAN arm is worse on HR and F1 and marginally better on MAE / RMSE** (by 0.001) — the direction the
   preregistration anticipated from KANFlow's own evidence, which is on MAE / RMSE: a gain on the mean-seeking
   metrics with no gain, here a loss, on the event metrics. The MAE / RMSE effect is tiny and should not be oversold.
3. **Caveat on "WORSE":** SR1 measured a seed-to-seed range of 1.50 bpm for arm C at NFE 50 (9.69 / 11.04 / 11.19), and
   seed 42 was arm C's best seed. The KAN arm's 11.05 is inside that range, so the honest statement is *no evidence
   that KAN helps OT-CFM*, not a demonstrated harm.
4. Two secondary observations, reported and not claimed: the outer-KAN iMF has the best FD so far among PENGUIN-backbone
   arms (3.22 vs 4.13), and both KAN iMF arms have slightly lower consensus HR (6.54, 6.38 vs 6.68). Neither was a
   preregistered endpoint and both are single-seed.
5. KANFlow's depth finding (deeper KAN insertion degrades) is not reproduced here: for iMF, "all" is if anything
   closer to the MLP reference than "outer".
