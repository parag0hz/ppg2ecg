# C1 — Target-Interval Exposure Control for 2-NFE Compression — REPORT

## **FINAL VERDICT: `TARGET h=0.5 EXPOSURE SUPPORTED`**

Under the rule frozen at `b32c952` before any weight update. Stage 1 report: `docs/C1_BASELINE_REPLAY_GATE_REPORT.md`.

**No test access. No architecture, loss or optimiser change. Only the (t, r) sampler differs between arms.
No compression method was implemented, selected or trained.**

Prereg `b32c952` · implementation `38eaf45` · Stage-1 `a14f5ee`.

---

## 1. Training

All three arms ran the recovered A4 recipe, seed 42, patience 20, `min_delta` 1e-4, identical
checkpoint-selection banks (hash `2cd144684498f279…`).

| arm | rounds | best round | selection metric | wall (s) | peak VRAM | `--c1-arm` |
|---|---:|---:|---:|---:|---:|---|
| C1-B | 66 | 45 | 0.11945885431656277 | 12,911 | 19,216 MiB | B |
| C1-H25 | 68 | 47 | 0.12780840157960570 | 13,469 | 19,216 MiB | H25 |
| C1-H50 | **101** | **80** | 0.11824402330613042 | 19,814 | 19,216 MiB | H50 |

**C1-B reproduced the frozen A4 checkpoint bit-for-bit** (identical `state_dict` sha256, identical
selection metric to 17 digits), so the interventions are compared against a provably correct control.

## 2. Sampler exposure (2,000,000 draws, seed 20260901)

| arm | P(h=0) | P(h=.25) | P(h=.50) | pos median | P(h≥.25) | P(h≥.5) |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.5000 | 0.0000 | 0.0000 | 0.2011 | 0.2010 | 0.0422 |
| H25 | 0.5000 | **0.2500** | 0.0000 | 0.2500 | 0.3503 | 0.0210 |
| H50 | 0.5000 | 0.0000 | **0.2500** | 0.5000 | 0.3503 | **0.2710** |

Exactly as preregistered. H25 and H50 share `P(h≥0.25)` to 1e-9 because they replace the same half of the
positive-h rows, which is what makes H25 a genuine specificity control rather than a different dose.

**RNG control passed before training:** model-init, dataloader-order, Gaussian-noise and validation-bank
hashes identical across all three arms; the (t, r) hash differs pairwise between all three.

## 3. Development metrics — frozen C0 population, oracle-free, GT-fixed

2,048 windows, 19,834 GT beats, evaluation source seed 0, same source tensor at NFE 2 and NFE 4.

| arm | NFE | M1 \|QRS-E−1\| | M2 \|p2p−1\| | M3 QRS RMSE | M4 RMSE | M5 raw corr | M6 \|slope−1\| | F1 excess | beats dev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 2 | 0.6148 | 0.2616 | 0.5580 | 0.4395 | 0.1036 | 0.2722 | +0.3066 | 0.1058 |
| B | 4 | 0.6056 | 0.2425 | 0.5462 | 0.4233 | 0.1040 | 0.2733 | +0.3176 | 0.1067 |
| H25 | 2 | 0.6153 | 0.2771 | 0.5355 | 0.4079 | 0.1254 | 0.2676 | +0.3416 | 0.0927 |
| H25 | 4 | 0.5842 | 0.2400 | 0.5324 | 0.4035 | 0.1300 | 0.2450 | +0.3542 | 0.0995 |
| **H50** | **2** | **0.5821** | **0.2271** | **0.5370** | **0.4085** | 0.1165 | **0.2382** | **+0.3512** | **0.0861** |
| H50 | 4 | 0.5705 | 0.2175 | 0.5335 | 0.4044 | 0.1193 | 0.2332 | +0.3667 | 0.0829 |

## 4. Paired evidence at NFE 2, versus C1-B

Subject-stratified paired bootstrap, 2,000 resamples, `default_rng(20260901)`. Positive = intervention better.

| metric | H25 Δ | H25 CI | H25 | H50 Δ | H50 CI | H50 |
|---|---:|---|---|---:|---|---|
| M1 \|QRS-E−1\| | −0.00056 | [−0.01016, +0.00886] | unresolved | **+0.03263** | [+0.02062, +0.04391] | **improves** |
| M2 \|p2p−1\| | −0.01550 | [−0.02266, −0.00860] | **worsens** | **+0.03445** | [+0.02786, +0.04100] | **improves** |
| M3 QRS RMSE | +0.02251 | [+0.01977, +0.02519] | improves | +0.02097 | [+0.01850, +0.02339] | improves |
| M4 RMSE | +0.03159 | [+0.02859, +0.03463] | improves | +0.03102 | [+0.02849, +0.03360] | improves |
| M5 raw corr | +0.02176 | [+0.01805, +0.02548] | improves | +0.01288 | [+0.00903, +0.01698] | improves |
| M6 \|slope−1\| | +0.00460 | [−0.00273, +0.01145] | unresolved | +0.03402 | [+0.02572, +0.04193] | improves |
| *F1 excess* | +0.03500 | [+0.02822, +0.04162] | improves | +0.04463 | [+0.03797, +0.05176] | improves |
| *beats-ratio dev* | +0.01303 | [+0.00804, +0.01767] | improves | +0.01967 | [+0.01388, +0.02529] | improves |

## 5. Gap closure (§10) — denominator is the C1-B replay, unclipped

| metric | `G` (replay 2→4) | H25 `I` | H25 `C` | H50 `I` | H50 `C` |
|---|---:|---:|---:|---:|---:|
| M1 \|QRS-E−1\| | +0.00921 | −0.00056 | −0.06 | +0.03263 | **+3.54** |
| M2 \|p2p−1\| | +0.01914 | −0.01550 | −0.81 | +0.03445 | **+1.80** |
| M3 QRS RMSE | +0.01175 | +0.02251 | +1.92 | +0.02097 | **+1.78** |
| M4 RMSE | +0.01620 | +0.03159 | +1.95 | +0.03102 | **+1.92** |

All four H50 closures exceed 1: **H50 at NFE 2 is better than the baseline at NFE 4 on every gap metric**
(e.g. M1 0.5821 vs 0.6056; M2 0.2271 vs 0.2425).

## 6. H50 success gate (§11) — **PASS on all five**

1. clearly improves **4** of M1–M4 (≥ 3 required) ✓
2. clearly worsens **0** of the six primary metrics ✓
3. **4** of M1–M4 with closure ≥ 0.50 (≥ 2 required) ✓
4. event F1 excess does not worsen — it improves, +0.04463 ✓
5. beats-ratio deviation does not worsen — it improves, +0.01967 ✓

## 7. Specificity (§12)

**H25 fails the same gate**: only 2 of M1–M4 improve, and it **clearly worsens M2** (−0.01550).

Difference-of-improvement, H50 − H25 at NFE 2:

| metric | Δ | 95 % CI | verdict |
|---|---:|---|---|
| M1 \|QRS-E−1\| | +0.03319 | [+0.01989, +0.04509] | **H50 better** |
| M2 \|p2p−1\| | +0.04996 | [+0.04250, +0.05776] | **H50 better** |
| M3 QRS RMSE | −0.00154 | [−0.00361, +0.00056] | unresolved |
| M4 RMSE | −0.00056 | [−0.00266, +0.00148] | unresolved |

H50 exceeds H25 on **2** of M1–M4 with CI entirely > 0 (≥ 2 required), and H25 does not independently pass.
**Verdict A conditions met.**

**The specificity is metric-dependent, and that matters.** H50's advantage over H25 is confined to the two
amplitude/energy calibration metrics (M1, M2). On the two RMSE metrics the arms are indistinguishable, and
H25 is nominally ahead. So the honest reading is: **the RMSE-family improvement is generic to positive-h
reweighting, while the QRS-energy and peak-to-peak improvement is specific to h = 0.5.**

## 8. NFE 4 preservation check (§13)

| arm | primary metrics clearly worsened at NFE 4 | `NFE4 DEGRADATION` |
|---|---|---|
| H25 | none | **no** |
| H50 | none | **no** |

Neither intervention damages the reference operating point. H50 improves NFE 4 as well
(M1 +0.03510, M2 +0.02491, M3 +0.01267, M4 +0.01895, all CIs > 0).

## 9. Limitations, stated rather than left to be inferred

- **Single training seed.** The bootstrap quantifies development-window uncertainty only. Training-seed
  variance is not estimated anywhere in C1, so no causal certainty across training runs may be claimed.
- **The arms are not compute-matched.** H50 ran **101 rounds to H25's 68 and B's 66** under the same frozen
  early-stopping rule, so it received roughly 50 % more optimizer steps. The extra training is an emergent
  consequence of the frozen protocol rather than an imposed advantage, but it is **not controlled for**, and
  some of H50's advantage may be attributable to it rather than to the interval exposure.
- **The improvement is not NFE-2-specific.** H50 improves NFE 4 by a comparable margin, so this is a broad
  change to the model, not a narrow repair of the 2-step operating point. The preregistered verdict follows
  the frozen H50-vs-H25 rule, which is about interval specificity, not about NFE specificity.
- **The selection metric is not comparable across arms** in the direction one might assume: the fixed
  validation banks are built from the baseline `tr_kw` and therefore do not sample h = 0.25 or h = 0.50.
  H50 nonetheless scored best on it (0.11824 vs B's 0.11946), so no adjustment is warranted — but the
  column should not be read as an arm ranking.
- **Two development subjects, previously visually inspected.** No population-level inference follows.

## 10. What may and may not be said

Permitted: *"Targeted exposure to h = 0.5 specifically closes a material part of the NFE2→NFE4 gap under
this single-seed development protocol."*

**Not** permitted and **not** claimed here: that large h causes the one-step failure; that training mismatch
is proven; that h = 0.5 is the only problem; that NFE-4 quality is solved; that compression was achieved;
any SOTA claim; anything about `kjd`/`ssx`. **This is not a final method.** No oracle metric
(`oracle_corr`, `oracle_qrs_energy`, `oracle_absent`) was used anywhere, and matched morphology was not
used as evidence.

## 11. Next step (recorded, not started)

The natural follow-up is a properly compute-matched, multi-seed replication of the H50 effect, and only
then a compression protocol. **C1 selects no compression mechanism** — not residual iMF, not
condition-informed source, not distillation, not shortcut, not anchored flow. That requires a new
preregistration after review.

## Artifacts

`artifacts/c1_interval_exposure/`: `sampler_exposure.csv`, `rng_control.json`, `stage1_metrics.csv`,
`stage1_paired.csv`, `stage1_result.json`, `stage2_metrics.csv`, `stage2_paired.csv`,
`stage2_result.json`, `gap_closure.csv`, `specificity.csv`, and figures `c1_sampler_exposure.png`,
`c1_nfe2_gap_closure.png`, `c1_paired_effects.png`, `c1_h50_vs_h25_specificity.png`.
