# DW1 — Fixed inference budget, depth or width? **Width wins for every model; the best allocation is "mostly width, a little depth"**

Prereg `cc902b3`, frozen before any number. No training. V1 VitalDB test (1,156 patients, 19,543 windows), seed-42
checkpoints. Budget `B = K × S`; HR = median over the K samples; patient-clustered bootstrap. Raw:
`artifacts/dw1_depth_width/grid.csv`, `result.json`.

## Verdicts (frozen hypotheses, B = 16 and B = 32)
| model | trained for | width extreme − depth extreme, B = 16 | B = 32 | H | best cell B = 16 | best cell B = 32 |
|---|---|---|---|---|---|---|
| iMF | one step | −1.99 [−2.16, −1.84] | −2.20 [−2.37, −2.05] | **H-W holds** | (K 8, S 2) 6.50 | (K 16, S 2) 6.18 |
| consistency distillation | one step | −2.08 [−2.25, −1.92] | −2.20 [−2.37, −2.03] | **H-W holds** | (K 16, S 1) 6.23 | (K 32, S 1) 6.14 |
| PENGUIN (Euler) | many steps | −1.45 [−1.67, −1.22] | −1.92 [−2.17, −1.68] | **H-D fails** — width wins here too | (K 8, S 2) 6.51 | (K 8, S 4) 6.30 |

## The grid — HR error (bpm) by budget B (rows) and steps per sample S (columns); K = B / S

| model | B | S=1 | S=2 | S=4 | S=8 | S=16 | S=32 |
|---|---|---|---|---|---|---|---|
| iMF | 16 | 6.67 | **6.50** | 7.02 | 8.08 | 8.14 | — |
| iMF | 32 | 6.45 | **6.18** | 6.48 | 6.98 | 8.06 | 8.10 |
| CD | 16 | **6.23** | 6.54 | 6.90 | 7.78 | 8.12 | — |
| CD | 32 | **6.14** | 6.25 | 6.46 | 6.83 | 7.75 | 8.06 |
| PENGUIN (Euler) | 16 | 7.44 | **6.51** | 6.68 | 7.81 | 8.87 | — |
| PENGUIN (Euler) | 32 | 7.37 | 6.35 | **6.30** | 6.71 | 8.19 | 9.27 |
Reference: PENGUIN at its shipped Heun 25 (50 NFE), K = 1: 9.64; K = 4 (200 NFE): 7.55.

## Single-sample quality by depth (seed-0 sample): what the steps buy
| model | S=1 | S=2 | S=4 | S=8 | S=16 | S=32 |
|---|---|---|---|---|---|---|
| iMF — F1 / FD | 0.673 / 4.16 | 0.673 / 3.86 | 0.676 / 3.22 | 0.676 / 3.30 | 0.677 / 3.47 | 0.677 / 3.58 |
| CD — F1 / FD | 0.674 / 4.78 | 0.662 / 4.82 | 0.667 / 6.85 | 0.677 / 11.9 | 0.687 / 22.7 | 0.689 / 39.2 |
| PENGUIN (Euler) — F1 / FD | 0.756 / 33.3 | 0.700 / 18.7 | 0.680 / 9.20 | 0.666 / 5.81 | 0.656 / 4.80 | 0.650 / 4.45 |

## Reading it
1. **At a fixed budget, spend it on samples, not steps — for all three models.** Even PENGUIN, trained for many
   steps, is 1.4–1.9 bpm better at (K = 16, S = 1) than at (K = 1, S = 16). The preregistered expectation that a
   many-step model prefers depth was wrong.
2. **The optimum is interior for iMF and PENGUIN: S = 2 (or 4), the rest in width.** For iMF, S = 2 × K = 16 (6.18) beats
   both extremes at B = 32. N6 had shown that iMF's one-step samples are under-dispersed (SD 7 ms vs 54 ms real) and
   that two steps restore the spread; here that extra diversity is worth 0.2–0.3 bpm once pooled. CD is the exception
   (S = 1 optimal): its multistep sampler does not add useful diversity.
3. **Depth stops paying at S ≈ 4 for HR (and hurts beyond):** every model's single-sample HR at S = 16–32 is 8–9 bpm,
   worse than its S = 1 sample pooled 4 times. Depth mainly changes the *waveform distribution* — PENGUIN's FD falls
   33 → 4.5 and iMF's 4.2 → 3.2 with more steps — not the beat rate.
4. **CD's multistep sampler degrades FD sharply (4.8 → 39)**: re-noising a consistency model is not a good way to spend
   compute for this task.
5. **PENGUIN at S = 1 (7.33) is still the best single-NFE HR** on this corpus, and (K = 8, S = 2) makes PENGUIN itself
   competitive (6.51) — i.e. the consensus recipe transfers to the incumbent model with no retraining at all.
6. Single seed; VitalDB only (regular rhythm). The width-vs-depth ordering on the wearable corpora is untested.

## Erratum (2026-09-22, from DW2 part B)
The K = 1 cells above are the noise-seed-0 draw, not an expectation over draws. Averaged over 32 draws, single-sample HR
error at S = 1 is PENGUIN 8.68 (not 7.33), iMF 10.13, CD 8.27. Point 5 ("PENGUIN at S = 1 is still the best single-NFE
HR") therefore rests on one favourable draw and is withdrawn; cells with K > 1 are unaffected.
