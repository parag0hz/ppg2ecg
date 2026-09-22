# DB1 — The direct discriminative baseline **beats every consensus arm on heart rate**

Prereg `cc902b3`, frozen before any DB1 weight update. A 1-D CNN (0.91 M params) regresses HR from the 4 s PPG window;
V1 VitalDB train split, 14,000 steps (17 s of training), seed 42; no ECG is generated. Test = 1,156 patients /
19,543 windows, patient-clustered bootstrap, paired. Raw: `artifacts/db1_discriminative_hr/result.json`.

| estimator | outputs an ECG-form signal | NFE / window | HR error (bpm) ↓ | regressor − it |
|---|---|---|---|---|
| **direct HR regressor** | no | 1 (0.002 ms/window, GPU) | **5.69 [5.31, 6.09]** | — |
| CD, K = 16 consensus | yes | 16 | 6.23 | −0.54 [−0.63, −0.45] |
| iMF, K = 16 consensus | yes | 16 | 6.68 | −0.98 [−1.11, −0.86] |
| PPG peak counting | no | 0 | 9.06 | −3.38 |
| PENGUIN, 50 NFE, one sample | yes | 50 | 9.69 | −3.99 |
| iMF, 1 NFE, one sample | yes | 1 | 10.14 | −4.45 |

**As the preregistration anticipated: the regressor wins, by 0.5–1.0 bpm over the best consensus arms and by 4 bpm over
the incumbent generative model.** It costs 17 seconds to train and 2 µs per window at inference.

## Consequences for the paper
1. **The HR claim must be scoped:** "among methods that output an ECG-form signal, consensus inference gives the best
   heart rate." Unscoped, "our method gives the best HR" is false, and a reviewer will build this baseline in an hour.
2. **It forces the honest framing of what the generative route is for:** not HR — a beat sequence (F1, RR intervals),
   a waveform with a realistic distribution (FD), and per-window sample dispersion. Those are the outputs a
   regressor cannot give, and the axes on which the paper's comparisons should be made.
3. **Together with LW1 and DW1, the programme now has the full spectrum on VitalDB:** point estimators (regressor 5.69;
   PENGUIN-1 7.33; auxiliary-loss iMF 7.6–8.0) are the best single-shot HR machines; distributional generators are worse
   single-shot but are the only ones that improve with pooling (iMF 10.1 → 6.2, CD 7.9 → 6.1); and neither family
   beats the regressor on the rate alone.
4. The regressor was trained only here (regular anaesthesia rhythm) and only on VitalDB; its behaviour on the wearable
   corpora, where every rate-only estimator (PPG peaks, PENGUIN-1) did badly, is untested. Single seed.
