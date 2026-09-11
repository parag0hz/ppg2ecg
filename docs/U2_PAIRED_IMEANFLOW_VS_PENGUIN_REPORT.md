# U2 — Is iMeanFlow better than PENGUIN? REPORT

Preregistration `docs/U2_PAIRED_IMEANFLOW_VS_PENGUIN_PREREGISTRATION.md` (`435c50f`, pushed before any
corpus was built or trained). Ran 2026-09-11 09:46 → 23:0x KST on one RTX 5090.

---

## 1. Stage verdict: **PARTIAL** — and the verdict flatters the result

The §9 rule returns **3 of 5 countable datasets non-inferior at some k ≤ 4**, which is PARTIAL.
That number should not be quoted on its own. **Two of the three passes are compromised**, one by a defect
in the frozen rule and one by margins that a significantly worse arm still fits inside.

| dataset | task | test subj | §9 verdict | what the numbers actually say |
|---|---|---|---|---|
| **WildPPG** | ECG | 2 | **NON-INFERIOR at k=1** | **genuine** — arm I wins both primary cells at 1 NFE vs 50 |
| PPG-DaLiA | ECG | 2 | NOT non-inferior | HR misses by 3.09 bpm at best k; F1 passes |
| BIDMC | Resp | 8 | NOT non-inferior | RR misses at every k |
| WESAD | Resp | 2 | NON-INFERIOR at k=1 | **narrow** — arm I significantly worse on *both* cells, inside margin |
| MIMIC-BP | ABP | 229 | NON-INFERIOR at k=1 | **misleading** — the gate suppressed a 13× DBP failure |
| UCI-BP | ABP | 1 | EXCLUDED (§5) | point estimates only; arm I far worse on MAE/RMSE/FD |

**Honest one-line summary: iMeanFlow reaches OT-CFM's 50-NFE quality at 1 NFE on one corpus of six
(WildPPG), is close but short on two (PPG-DaLiA, WESAD), and is substantially worse on the two ABP
corpora.** The headline claim of §1 — 12–50× less sampling compute at equal quality — is **not supported
in general**, and is supported on exactly one dataset.

## 2. Compute matching worked

All twelve runs took **exactly 14,000 optimizer steps**. Validation-round counts ranged 64–200 because
`batch_rounds` truncates a round at the epoch boundary, and training-set sizes span 133× (4,440 to
592,096 windows) — so `--epochs × --val-every-steps` would have produced wildly different step counts.
That is the M2/M3 arithmetic error the `--max-steps` gate (`f7cf04c`) exists to prevent, and it held.

| corpus | rounds | arm C hours | arm I hours |
|---|---|---|---|
| BIDMC | 200 | 0.63 | 2.94 |
| WESAD | 114 | 1.12 | 2.85 |
| PPG-DaLiA | 76 | 1.60 | 2.33 |
| WildPPG | 64 | 0.96 | 2.26 |
| MIMIC-BP | 65 | 1.14 | 2.71 |
| UCI-BP | 66 | 1.60 | 2.29 |

22.4 GPU-hours of training, 11.75 h wall clock on two streams.

## 3. Two defects found in this stage's own design

### 3.1 The §8 informativeness gate is two-sided, and it hid a failure

§8 drops a cell when arm C's own NFE 1→50 span is below the margin, and I wrote it as *"cannot contribute
to a dataset verdict, **whatever arm I scores on it**"*. It was meant to withhold **unearned passes** where
the sampling budget buys nothing. Being two-sided, it also withholds **earned failures**.

On **MIMIC-BP DBP** arm I scores 23.62 mmHg against arm C@50's 10.15 — it misses the +1.0 mmHg margin by
more than thirteenfold, on the one corpus with 229 test subjects. Arm C's own span there is 0.052, so the
cell was dropped and SBP alone carried the dataset to "NON-INFERIOR at k=1".

```
DBP k=1  C@50 10.146  I 23.617  CI(I-C) [+12.654,+14.230]  margin +1.00  span 0.052  GATED (would FAIL)
DBP k=2  C@50 10.146  I 22.657  CI(I-C) [+11.705,+13.253]  margin +1.00  span 0.052  GATED (would FAIL)
DBP k=4  C@50 10.146  I 20.538  CI(I-C) [ +9.614,+11.119]  margin +1.00  span 0.052  GATED (would FAIL)
SBP k=1  C@50 17.112  I 16.600  CI(I-C) [ -0.755, -0.253]  margin +1.00  span 2.782  PASS
```

The rule is frozen and was applied as written. `scripts/u2_verdict.py` now records for every gated cell
whether it *would PASS* or *would FAIL* and prints an explicit warning naming each suppressed failure, so
the suppression cannot be read past. UCI-BP DBP k=1 is suppressed the same way.

**A gate must be one-sided: able to withhold a PASS, never a FAIL.** Any successor preregistration must
say so.

### 3.2 The arms' own selection criteria peak at very different points

§6 matched **optimizer steps**, and §10.3 declared the selection-criterion asymmetry (arm C `fixed_cfm`,
arm I `fixed_imf_mse`) as a limitation. It turns out to do real work:

| corpus | rounds | arm C best | % of budget | arm I best | % of budget |
|---|---|---|---|---|---|
| BIDMC | 200 | 92 | 46 % | 52 | 26 % |
| WESAD | 114 | 101 | 89 % | 38 | 33 % |
| PPG-DaLiA | 76 | 47 | 62 % | 17 | 22 % |
| WildPPG | 64 | 56 | 88 % | 27 | 42 % |
| MIMIC-BP | 65 | 64 | 98 % | 47 | 72 % |
| UCI-BP | 66 | 66 | 100 % | 16 | 24 % |
| **mean** | | | **80 %** | | **37 %** |

Arm C's criterion improves almost to the end of the budget; arm I's plateaus at about a third of it. The
compute was matched, but **the selected checkpoints were not equally far into training** — arm I is
evaluated at a systematically earlier point in its own run. Whether that reflects genuine convergence or a
weaker selection signal cannot be settled here, and neither reading is assumed. It is a live alternative
explanation for part of arm I's deficit, and a successor design should either share one sampler-free
criterion or report both arms at a fixed step count as well as at their own best.

## 4. Results

Arm C: NFE 1 = Euler 1 step, 2 = Heun 1, 4 = Heun 2, 50 = Heun 25. Arm I: MeanFlow schedule at 1/2/4.
Every point is published; none was chosen after the fact. Values are subject-macro means over 4 evaluation
noise draws.


**PPG-DaLiA** (ECG) — 4,386 test windows, 2 subjects

| metric | C@1 | C@2 | C@4 | **C@50** | **I@1** | I@2 | I@4 |
|---|---|---|---|---|---|---|---|
| HR | 33.21 | 29.86 | 16.23 | **10.63** | **15.57** | 14.53 | 13.72 |
| Rpeak_F1 | 0.0705 | 0.09072 | 0.1268 | **0.14** | **0.1448** | 0.1448 | 0.1415 |
| Micro_F1 | 0.07635 | 0.09686 | 0.1306 | **0.1434** | **0.1481** | 0.1481 | 0.145 |
| Macro_F1 | 0.07051 | 0.09055 | 0.127 | **0.1402** | **0.145** | 0.1451 | 0.1417 |
| RR_MAE_ms | 35.59 | 33.47 | 31.26 | **32.41** | **33.22** | 33.72 | 33.36 |
| MAE | 0.2238 | 0.4263 | 0.3546 | **0.334** | **0.3646** | 0.3664 | 0.3729 |
| RMSE | 0.3012 | 0.5515 | 0.4606 | **0.4328** | **0.448** | 0.4537 | 0.4602 |
| FD_kanflow | 29.84 | 59.65 | 15.17 | **2.264** | **10.76** | 5.649 | 5.671 |
| FD_discrete | 1.259 | 1.167 | 0.7712 | **0.7036** | **0.7135** | 0.7218 | 0.7262 |

**WildPPG** (ECG) — 12,000 of 92,696 test windows, 2 subjects

| metric | C@1 | C@2 | C@4 | **C@50** | **I@1** | I@2 | I@4 |
|---|---|---|---|---|---|---|---|
| HR | 23.72 | 34.39 | 22.62 | **16.19** | **15.45** | 16.45 | 17.11 |
| Rpeak_F1 | 0.2887 | 0.077 | 0.2337 | **0.2652** | **0.2711** | 0.2613 | 0.261 |
| Micro_F1 | 0.3195 | 0.08232 | 0.238 | **0.2644** | **0.2715** | 0.2666 | 0.2667 |
| Macro_F1 | 0.2887 | 0.077 | 0.2337 | **0.2652** | **0.2711** | 0.2613 | 0.261 |
| RR_MAE_ms | 18.5 | 30.2 | 27.17 | **28.57** | **29.94** | 29.48 | 29.31 |
| MAE | 0.281 | 0.5683 | 0.4163 | **0.3455** | **0.3934** | 0.4137 | 0.4156 |
| RMSE | 0.3429 | 0.731 | 0.5347 | **0.4487** | **0.493** | 0.5092 | 0.5102 |
| FD_kanflow | 35.57 | 166.9 | 53.21 | **4.884** | **16.44** | 26.19 | 28.79 |
| FD_discrete | 1.185 | 1.867 | 0.9842 | **0.7218** | **0.7298** | 0.7352 | 0.7313 |

**BIDMC** (Resp) — 960 test windows, 8 subjects

| metric | C@1 | C@2 | C@4 | **C@50** | **I@1** | I@2 | I@4 |
|---|---|---|---|---|---|---|---|
| RR | 2.496 | 2.766 | 3.273 | **3.332** | **4.121** | 4.023 | 4.047 |
| Resp_corr | -0.06178 | -0.04791 | -0.03729 | **-0.02326** | **-0.007693** | -0.007387 | -0.007429 |
| MAE | 0.6951 | 1.17 | 0.7796 | **0.8041** | **0.8335** | 0.8261 | 0.8318 |
| RMSE | 0.7688 | 1.449 | 0.9255 | **0.9598** | **0.993** | 0.9851 | 0.9942 |
| FD_kanflow | 146.5 | 282.3 | 80.19 | **9.763** | **9.699** | 9.576 | 10.64 |
| FD_discrete | 1.014 | 3.4 | 1.459 | **1.204** | **1.324** | 1.267 | 1.282 |

**WESAD** (Resp) — 2,775 of 2,788 test windows, 2 subjects

| metric | C@1 | C@2 | C@4 | **C@50** | **I@1** | I@2 | I@4 |
|---|---|---|---|---|---|---|---|
| RR | 4.234 | 4.393 | 5.341 | **5.02** | **5.127** | 5.12 | 5.123 |
| Resp_corr | 0.1456 | 0.113 | 0.07939 | **0.0381** | **0.001757** | 0.002033 | 0.002177 |
| MAE | 0.5895 | 0.8743 | 0.698 | **0.7593** | **0.7788** | 0.7782 | 0.775 |
| RMSE | 0.6649 | 1.091 | 0.8373 | **0.8937** | **0.9197** | 0.918 | 0.9138 |
| FD_kanflow | 155.2 | 195.1 | 74.02 | **1.443** | **5.157** | 4.482 | 5.379 |
| FD_discrete | 0.959 | 2.702 | 1.401 | **1.138** | **1.174** | 1.151 | 1.149 |

**UCI-BP** (ABP) — 12,000 of 73,554 test windows, 1 subjects

| metric | C@1 | C@2 | C@4 | **C@50** | **I@1** | I@2 | I@4 |
|---|---|---|---|---|---|---|---|
| SBP | 17.86 | 18 | 18.46 | **26.47** | **19.76** | 23.33 | 26.37 |
| DBP | 8.912 | 8.839 | 8.826 | **9.593** | **15.86** | 9.255 | 9.135 |
| MAE | 13.87 | 13.88 | 13.96 | **15.55** | **25.83** | 22.41 | 23.48 |
| RMSE | 16.73 | 16.75 | 16.87 | **19.24** | **32.52** | 28.56 | 29.26 |
| FD_kanflow | 68,904 | 68,229 | 67,634 | **70,606** | **393,996** | 198,403 | 170,832 |
| FD_discrete | 31.68 | 31.43 | 31.58 | **37.23** | **49.27** | 45.46 | 45.96 |

**MIMIC-BP** (ABP) — 11,908 of 48,090 test windows, 229 subjects

| metric | C@1 | C@2 | C@4 | **C@50** | **I@1** | I@2 | I@4 |
|---|---|---|---|---|---|---|---|
| SBP | 14.33 | 14.46 | 14.67 | **17.11** | **16.6** | 14.19 | 14.18 |
| DBP | 10.09 | 9.704 | 9.527 | **10.15** | **23.62** | 22.66 | 20.54 |
| MAE | 13.17 | 13.22 | 13.28 | **13.77** | **25.46** | 23.44 | 22.22 |
| RMSE | 15.5 | 15.56 | 15.66 | **16.53** | **31.26** | 29.02 | 27.65 |
| FD_kanflow | 52,958 | 53,507 | 53,325 | **38,597** | **310,690** | 263,986 | 233,892 |
| FD_discrete | 26.03 | 25.89 | 25.97 | **28.81** | **46.58** | 42.54 | 39.78 |

## 5. What the six corpora say together

### 5.1 The one clean win, and why it is narrower than it looks

**WildPPG** is the only unambiguous pass: arm I at **1 NFE** beats arm C at **50 NFE** on both primary
cells — HR 15.45 vs 16.19 bpm, R-peak F1 0.2711 vs 0.2652 — with both cells informative (arm C's own spans
7.54 bpm and 0.0235).

But the second ECG corpus disagrees, and the reason is instructive:

| | arm C HR, NFE 1 → 50 | arm I best HR | verdict |
|---|---|---|---|
| WildPPG | 23.72 → **16.19** (7.5 gained) | **15.45** at k=1 | non-inferior |
| PPG-DaLiA | 33.21 → **10.63** (22.6 gained) | 13.72 at k=4 | fails by 3.09 bpm |

On PPG-DaLiA the 50-NFE budget genuinely buys a great deal — arm C improves threefold — and iMeanFlow does
not reach that level at any k ≤ 4. On WildPPG arm C plateaus at 16.19 and iMeanFlow catches it.
**WildPPG's win is as much a statement about how little OT-CFM gains there as about iMeanFlow's few-step
quality.** A single-corpus success should not be generalised.

### 5.2 Rate versus placement: the two ECG metrics point opposite ways

R-peak F1 favours arm I on **both** ECG corpora (DaLiA 0.1448 vs 0.1400; WildPPG 0.2711 vs 0.2652), while
HR error favours arm C on DaLiA by a wide margin. D2 established that HR error is satisfied by beat **rate**
alone, and D1 measured the same rate/placement dissociation. U2 adds that the two objectives sit on
opposite sides of it: **iMeanFlow places beats slightly better; OT-CFM counts them better.** Neither is
"better ECG reconstruction", and any paper claiming one from HR error alone would be reading D2's
anti-informative metric.

### 5.3 Respiration: neither arm reconstructs the waveform

| corpus | arm C@50 corr | arm I corr | RR error C@50 / I |
|---|---|---|---|
| BIDMC | −0.0233 | −0.0077 | 3.33 / 4.12 |
| WESAD | 0.0381 | 0.0018 | 5.02 / 5.13 |

The correlations are at or below noise for **both** arms, while RR error looks respectable. The metric is
unit-tested (+1 identical, −1 phase-inverted, ≈0 quarter-cycle shift, `tests/test_u2_evaluation.py`), so
this is the task and not the measurement: upstream's `RespRateError` is an FFT argmax and is **phase-blind**,
so a prediction can score well while being wrong about when the breath happened. U1's qualitative figures
showed exactly this; U2 quantifies it for both objectives.

WESAD's pass has to be read against that. Arm I is **significantly worse on both cells** — RR
CI(I−C) [+0.060, +0.153] against a +0.5 margin, correlation CI [−0.044, −0.028] against a −0.05 margin —
it simply fits inside. And arm C is **better at low NFE** on both respiration corpora (WESAD RR 4.23 at
NFE 1 vs 5.02 at NFE 50), so C@50 is arm C's worst setting there and the comparison was already tilted
toward arm I before the margin was applied.

### 5.4 ABP: iMeanFlow does not survive the raw-mmHg scale

This is the clearest negative finding, and it holds on both ABP corpora.

| | MIMIC-BP C@50 → I best | UCI-BP C@50 → I best |
|---|---|---|
| SBP (mmHg) | 17.11 → **14.18** (arm I better) | 26.47 → **19.76** (arm I better) |
| DBP (mmHg) | 10.15 → 20.54 (**2.0× worse**) | 9.59 → 9.13 (comparable) |
| MAE | 13.77 → 22.22 (**1.6×**) | 15.55 → 22.41 (**1.4×**) |
| RMSE | 16.53 → 27.65 (**1.7×**) | 19.24 → 28.56 (**1.5×**) |
| FD (KANFlow) | 38,597 → 233,892 (**6.1×**) | 70,606 → 170,832 (**2.4×**) |

Arm I wins SBP on both and loses everything that measures the waveform. SBP is the window **maximum** and
DBP the **minimum**: D3 already showed SBP is the noise-sensitive extreme. A model whose waveform MAE is
1.4–1.6× worse can still hit the maximum more often, and that is what the SBP column is recording — not
better ABP reconstruction.

Upstream leaves ABP **un-normalised in raw mmHg** (mean ≈ 88.8, sd ≈ 23.7), identically for both arms.
Arm C absorbs that scale and arm I does not. **The mechanism is not established here.** The obvious
candidate — iMeanFlow's adaptive weight `w = 1/(δ²+ε)^p` collapsing on a large-δ² scale — is *not*
supported by inspection: the loss is `(δ²·sg(w)).mean()`, so the weight largely cancels and the gradient is
approximately scale-invariant. §3.2's selection asymmetry is a second candidate (UCI-BP arm I selected
round 16 of 66, MIMIC-BP 47 of 65). Deciding between them needs its own preregistered probe and is listed
in §8.

### 5.5 Where the 50-NFE budget clearly earns its cost

FD favours arm C@50 on five of six corpora, often by a wide margin (WESAD 1.44 vs 4.48, PPG-DaLiA 2.26 vs
5.65, WildPPG 4.88 vs 16.44, MIMIC-BP 38.6k vs 234k, UCI-BP 70.6k vs 170.8k); BIDMC is the lone tie
(9.76 vs 9.58). **Distributional fidelity is where the sampling budget buys something**, including on
corpora where non-inferiority holds on the task metric. FD is secondary by preregistration and does not
enter the verdict — but the tension between "non-inferior on the task metric" and "6× worse in
distribution" is a real result, not a footnote.

The known **NFE-2 Heun collapse** preregistered in §8 appeared as predicted: arm C at NFE 2 is worse than
at NFE 1 on both ECG corpora (WildPPG HR 23.72 → 34.39, R-peak F1 0.2887 → 0.0770) and on FD everywhere.
No iMeanFlow advantage at k=2 is claimed on that basis.

## 6. Limitations

1. **One training seed (42).** §10.1. Nothing here speaks to robustness against training randomness, and
   no such claim is made. Sampling stochasticity *is* quantified (4 noise draws; per-draw spread is in
   `metrics_*.csv`).
2. **Three of five countable datasets have two test subjects.** PPG-DaLiA, WildPPG and WESAD have
   `n_test = 2`, so the §9 subject-clustered bootstrap resamples two clusters and its interval is coarse
   by construction. This is a consequence of the 15 % test fraction I fixed in §5 and it cuts both ways —
   it can leave a real difference unresolved and it can let a narrow pass through. The within-subject
   interval is reported beside it in `paired_*.csv` for exactly this reason.
3. **UCI-BP is excluded** (one test subject after the U1-P1 deduplication), so its bootstrap CI is
   degenerate — `ci_lo == ci_hi` in `paired_u2_ucibp.csv` — and only its point estimates are read.
4. **Selection asymmetry** — §3.2 above, promoted from a declared limitation to an observed effect.
5. **Two-sided gate** — §3.1 above.
6. U2 compares two objectives on one backbone at 4 s on its own splits. It is **not** a comparison against
   PENGUIN's published numbers — U1 is — and the two tables must not be merged.

## 7. Deviations

| # | deviation | effect |
|---|---|---|
| D1–D3 | as preregistered (§7 of the prereg): WildPPG 14-subject firewall, Tier-B caps n/a here, `DaLiA`→`PPG-DaLiA` | — |
| **D4** | `train_a0.py`'s hard-coded `T == sample_rate * 8` opened to `--segment-len` (default 8) | required to run PENGUIN's shipped 4 s on arm C at all; `train_a2.py` never had the guard, which is why D3 could train 4 s on arm I only |
| **D5** | evaluation-window budget: ≤ 12,000 windows per corpus, exact linspace within each subject, blocks of k for Resp/ABP | 26.1 → 5.2 GPU-h. Fixed by compute alone; BIDMC and PPG-DaLiA are under the cap and untouched. Commit `f26264f` |
| **D6** | §8's sign convention disambiguated for score metrics | the frozen wording ("upper bound below δ") is incoherent for δ < 0; resolved as *lower(I−C) > δ*, stated in `scripts/u2_verdict.py` |

## 8. What this justifies doing next

Recommendations only; nothing here is implemented.

1. **Re-run the verdict with a one-sided gate.** The frozen rule cannot be edited, but a successor
   preregistration should define the gate as able to withhold a PASS only. On these same numbers that
   alone moves MIMIC-BP from "non-inferior" to a failure and the stage tally from 3/5 to 2/5.
2. **Break the selection asymmetry.** Report both arms at a fixed step count as well as at each one's own
   best checkpoint. §3.2 shows the two criteria peak at 80 % vs 37 % of the budget.
3. **Settle the ABP mechanism.** A preregistered probe with ABP z-scored for both arms would separate
   "iMeanFlow cannot handle raw mmHg" from "iMeanFlow's selection stopped early on ABP".
4. **Raise test-subject counts.** Three of five datasets carry two test subjects; LOSO or k-fold would
   give intervals that can actually resolve a difference this size.
5. **Retire RespRateError as a sole respiration endpoint.** Both arms score 2.5–5.1 bpm with waveform
   correlation at noise. Any respiration claim needs a phase-aware co-primary.

## 9. Firewalls verified

- `external/PENGUIN` unmodified at `6cd70cd`; asserted before every run.
- WildPPG `kjd` / `ssx` absent from the corpus, from all three split lists, and re-asserted at evaluation.
- UCI-BP deduplicated to `{0,1,4,5}`; the build fails if two survivors share a target hash.
- Frozen A4 checkpoint md5 `31c042d291052fbb6dc15263ad316be2` and arm-U state sha256 `20ba7234…` unchanged.
- C2 remains deferred. No checkpoint, prediction, `.pkl` or raw data is in git.
