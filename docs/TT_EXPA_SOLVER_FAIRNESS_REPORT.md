# EXP-A — Solver fairness: the depth baseline was **not** weakened by Euler; two of the three preregistered hypotheses fail, and why

Prereg `50948e2` (EXP-A), frozen before any number. No training. VitalDB V1 test (1,156 patients, 19,543 windows), PENGUIN
(OT-CFM) checkpoints seed 42 / 1 / 2, unchanged. Budget counted in **network function evaluations**: Euler step = 1 NFE,
upstream-exact Heun step = **2 NFE** (`ppg2ecg.flow.samplers`, bit-exact with upstream `PENGUIN.sample`), so the shipped
`n_step = 25` = 50 NFE. HR = median over K samples, noise seeds 0 … K−1. Patient-clustered bootstrap (2,000, seed 20260911),
paired within seed. Raw: `artifacts/tt_expa_solver/{grid.csv, result.json}`; figure `expa_solver.png`.
Command: `.venv/bin/python scripts/tt_expa_solver.py`.

## Verdicts on the frozen hypotheses
| hypothesis | requirement | seeds holding | verdict |
|---|---|---|---|
| **A-H1** within-Heun width vs depth | `(16, H1) − (1, H16)` **and** `(8, H2) − (1, H16)` CI upper < 0, 3/3 seeds | **0 / 3** | **REJECTED** |
| **A-H2** width-heavy Euler vs the strongest depth baseline | `(8, E4) − (1, H16)` **and** `(8, E4) − (1, E32)` CI upper < 0, 3/3 | **3 / 3** | **SUPPORTED** |
| **A-H3** shipped budget | `(5, H5) − (1, H25)` **and** `(25, H1) − (1, H25)` CI upper < 0, 3/3 | **0 / 3** | **REJECTED** |

**Both failures come from the same cell: a one-step Heun sample.** `(16, H1)` and `(25, H1)` put a single Heun step across the
whole interval [0, 1] (2 NFE). That sampler is degenerate — single-draw F1 **0.060**, FD **106.6**, HR error **29.98** (seed 42) —
and its consensus is 18.95 ± 2.78 (B = 32) and 18.37 ± 2.71 (B = 50). The preregistration should not have included it; that is a
defect of the preregistration, recorded here rather than repaired after the fact. The **other** half of each conjunction holds
in 3/3 seeds:
`(8, H2) − (1, H16)` = −1.95 [−2.15, −1.74] · −1.62 [−1.93, −1.29] · −3.53 [−3.74, −3.33] (win rate 74 / 67 / 86 %);
`(5, H5) − (1, H25)` = −2.55 [−2.74, −2.37] · −4.16 [−4.39, −3.94] · −3.56 [−3.76, −3.36] (83 / 88 / 86 %).

## The grid — HR error (bpm), 3-seed mean ± SD (per-seed values in `grid.csv`)
| B | sampler | (K, steps) | NFE / sample | HR error | GPU ms / window (batch 256) |
|---|---|---|---|---|---|
| 32 | Euler | (32, 1) | 1 | 7.396 ± 0.125 | — |
| 32 | Euler | (16, 2) | 2 | **6.418 ± 0.214** | 30.8 |
| 32 | Euler | (8, 4) | 4 | **6.438 ± 0.123** | — |
| 32 | Euler | (4, 8) | 8 | 7.400 ± 0.607 | 30.7 |
| 32 | Euler | (2, 16) | 16 | 9.299 ± 0.961 | 30.6 |
| 32 | Euler | (1, 32) | 32 | 10.423 ± 0.995 | — |
| 32 | **Heun** | (16, 1) | 2 | **18.951 ± 2.782** (degenerate) | 40.7 |
| 32 | Heun | (8, 2) | 4 | 8.187 ± 1.056 | 40.0 |
| 32 | Heun | (4, 4) | 8 | **7.503 ± 0.277** | 39.6 |
| 32 | Heun | (2, 8) | 16 | 9.293 ± 0.564 | 42.8 |
| 32 | Heun | (1, 16) | 32 | 10.492 ± 0.813 | 41.9 |
| 50 | Euler | (50, 1) | 1 | 7.375 ± 0.126 | 42.7 |
| 50 | Euler | (25, 2) | 2 | **6.354 ± 0.215** | 49.8 |
| 50 | Euler | (10, 5) | 5 | **6.376 ± 0.178** | 42.5 |
| 50 | Euler | (5, 10) | 10 | 7.115 ± 0.527 | 42.3 |
| 50 | Euler | (1, 50) | 50 | 10.565 ± 0.971 | 42.0 |
| 50 | Heun | (25, 1) | 2 | 18.371 ± 2.710 (degenerate) | 57.9 |
| 50 | Heun | (5, 5) | 10 | 7.200 ± 0.370 | 52.7 |
| 50 | **Heun (1, 25) = shipped configuration** | | 50 | 10.614 ± 0.841 | 55.6 |

Latency is measured only for cells generated in this run (cached cells blank). At **equal NFE Heun costs ≈ 30 % more
wall-clock** than Euler in this implementation (40–43 vs 30–31 ms/window at B = 32).

## The three questions the experiment was run to answer
### 1. Does width-heavy still beat pure depth when the depth arm uses Heun?
**Yes, for every non-degenerate Heun cell, in 3/3 seeds.** Within Heun, (4, H4) 7.50 ± 0.28 and (8, H2) 8.19 ± 1.06 both beat
(1, H16) 10.49 ± 0.81. Across solvers, the preregistered width-heavy Euler arm (8, E4) beats the **strongest** depth baseline
available: vs (1, H16) −3.30 [−3.50, −3.12] · −4.55 [−4.74, −4.36] · −4.46 [−4.68, −4.25] (win rate 88 / 93 / 92 %);
vs (1, E32) −2.98 · −4.57 · −4.44 (86 / 93 / 90 %). And at its own shipped budget, **(8, E4) at 32 NFE beats (1, H25) at 50 NFE**
by −3.37 [−3.57, −3.17] · −4.60 [−4.80, −4.40] · −4.62 [−4.85, −4.39], win rate 87 / 93 / 91 %.

### 2. How much does the better solver recover at pure depth?
**Essentially nothing.** `(1, H16) − (1, E32)` = +0.31 [+0.22, +0.41] · −0.01 [−0.15, +0.12] · +0.03 [−0.06, +0.13];
`(1, H25) − (1, E50)` = +0.20 [+0.13, +0.27] · −0.04 [−0.14, +0.06] · +0.01 [−0.07, +0.09]. Two of three seeds give an interval
containing 0 at both budgets, and where the difference is significant (seed 42) Heun is **worse**. The depth arm of DW1 / DW2-A /
WD1-B was therefore not handicapped by the solver: at matched NFE the upstream sampler lands on the same number.

### 3. Does the waveform-fidelity / functional-accuracy trade-off survive?
Yes, and it is where Heun's advantage actually is. Single sample (seed 42, noise seed 0):

| NFE / sample | 1 | 2 | 4 | 5 | 8 | 10 | 16 | 32 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| Euler F1 | 0.756 | 0.700 | 0.680 | 0.674 | 0.666 | 0.662 | 0.656 | 0.650 | 0.647 |
| Euler FD | 33.3 | 18.7 | 9.20 | 7.65 | 5.81 | 5.35 | 4.80 | 4.45 | 4.35 |
| Euler HR (1 draw) | 7.33 | 7.63 | 7.97 | 8.14 | 8.48 | 8.59 | 8.87 | 9.28 | 9.44 |
| Heun F1 | — | 0.060 | 0.623 | — | 0.653 | 0.651 | 0.648 | 0.645 | 0.643 |
| Heun FD | — | 106.6 | 20.7 | — | 5.43 | 4.64 | 4.24 | 4.21 | 4.22 |
| Heun HR (1 draw) | — | 29.98 | 11.89 | — | 8.76 | 8.73 | 9.16 | 9.55 | 9.64 |

(Heun columns are at 2 × steps, so its 4-NFE column is 2 steps, its 8-NFE column 4 steps, etc.) **Heun buys waveform
distribution, not the functional**: from 8 NFE upward its FD is 5–12 % lower than Euler's at the same NFE (5.43 vs 5.81;
4.24 vs 4.80; 4.22 vs 4.35), while its single-sample HR is equal or worse at every NFE and its F1 is lower everywhere. Both
solvers reproduce the DW1 pattern — deeper single samples have better FD and *worse* HR / F1.

## What this settles, and what it does not
- **Settles the reviewer objection.** "The depth baseline was weakened by an inferior solver" is not supported: at matched
  NFE the upstream-exact Heun gives the same pure-depth HR (≤ 0.31 bpm difference, 2/3 intervals spanning 0) and costs 30 %
  more wall-clock. The width-heavy allocation beats the strongest depth baseline by 3.3–4.6 bpm in 3/3 seeds and beats the
  shipped 50-NFE configuration at 32 NFE.
- **Does not settle the preregistered conjunctions.** A-H1 and A-H3 are **REJECTED** as written, because a one-step Heun is
  not a usable sampler. The paper must state the failure, not quietly substitute the surviving half.
- The exploratory best cells (test-set minima, **not** headline) are all Euler: (8, 4) / (16, 2) at B = 32 and (10, 5) / (25, 2)
  at B = 50, consistent with DW1 / DW2-A's (8, 4).
- VitalDB only; three seeds; the WildPPG allocation result (WD1-B) remains single-seed and Euler-only.
