# N1 — The supervision-matched null method: REPORT

Preregistration `docs/N1_NULL_METHOD_STAMPING_BASELINE_PREREGISTRATION.md` (`f7a6b70`, pushed before any
N1 number existed). **No training, no weight update, no optimizer.** 39 s on one GPU + CPU.

---

## 1. Verdict: **NULL DOMINATES**

A training-free composition — the PPG-only R1 rhythm scaffold's detected events, with one fixed QRS
template stamped at each — **out-scores every trained generative arm in this program on the event axis of
the cohort those arms were judged on**, by six times the minimal effect they were held to.

| | `f1_excess` | vs GTF-TRUE | 95 % CI | rule |
|---|---|---|---|---|
| **N-R1** (null method) | **+0.4866** | **+0.1284** | [+0.1175, +0.1387] | ≥ +0.02, CI > 0 → **NULL DOMINATES** |
| GTF-TRUE (best trained arm, R3) | +0.3582 | — | — | |
| B (frozen generator, R2/R3 baseline) | +0.3176 | +0.1690 | [+0.1575, +0.1801] | |

**The window-specificity qualifier does not apply.** Replacing each window's events with a *partner*
window's (the R2/R3 derangement) collapses the null to **+0.0054** — paired effect +0.4812
[+0.4651, +0.4967]. The null's score is not an artefact of stamping *something*; it depends on which
window's PPG produced the events.

## 2. Full table

Subject-macro over 2,048 frozen windows (`an0`, `k2s`), 19,834 GT beats. Comparator rows are the frozen
R2/R3 values at NFE 4, read from `artifacts/r3_rhythm_fusion/`, not recomputed.

| arm | `f1_excess` ↑ | missing ↓ | spurious ↓ | S4 ↓ | S5 ↓ | matched morph | whole-window corr |
|---|---|---|---|---|---|---|---|
| **N-GT** (GT-R leakage; diagnostic only) | **+0.8742** | 0.001 | 0.000 | **0.1460** | **0.0878** | 0.5418 | **0.4680** |
| **N-R1** (null method) | **+0.4866** | **0.3786** | 0.4421 | 0.3562 | 0.2408 | 0.5410 | −0.0031 |
| N-R1-A (full-beat template) | +0.3948 | 0.3920 | 0.6840 | 0.3431 | 0.2194 | **0.8582** | 0.0367 |
| GTF-TRUE (R3, trained) | +0.3582 | 0.5233 | 0.4902 | 0.3289 | 0.2162 | — | — |
| GTF-CONST (R3, trained) | +0.3349 | — | — | — | — | — | — |
| ADD (R2, trained) | +0.3369 | — | — | — | — | — | — |
| TF-TRUE (R3, trained) | +0.3341 | — | — | — | — | — | — |
| B (frozen generator) | +0.3176 | 0.5662 | 0.5154 | 0.3220 | 0.2147 | — | — |
| **N-CONST** (uniform grid at own median RR) | **+0.2949** | 0.5680 | 0.6700 | 0.3491 | 0.2340 | 0.5432 | −0.0416 |
| N-SHUFFLE (partner window's events) | +0.0054 | 0.8620 | 0.9410 | 0.3091 | 0.1953 | 0.5383 | 0.0019 |

## 3. What this means — three readings, in order of consequence

### 3.1 No generative arm in this program has demonstrated event-axis value

The null method beats **B, ADD, TF-TRUE, GTF-CONST and GTF-TRUE**. R2's +0.019 and R3's +0.041 gains were
measured against a baseline that a detector and a template beat by +0.169. Those stage verdicts stay
frozen and are not reinterpreted here, but **the bar they were measured against was the wrong one**.

This is the bar any method proposal must clear first. It is also the bar the published literature using
this metric family has not reported clearing.

### 3.2 Most of a generator's event score is *rate*, not *placement*

**N-CONST** discards every individual event time — it stamps a uniform grid at the window's own median R1
RR interval — and still scores **+0.2949**, within 0.023 of the frozen generator B (+0.3176) and within
0.040 of the best trained arm. A method that knows *only the beat rate* reproduces most of what the
generators score on chance-corrected beat F1.

D2 established this for HR error ("satisfied by beat rate alone"). N1 extends it to `f1_excess`, the metric
R2, R3, X4-0 and the E-series all used as their event endpoint. Reading a difference of +0.02–0.04 on this
metric as evidence about beat *placement* is not supported.

### 3.3 The structure trade-off is not a property of R2/R3's interfaces — it is what event gain costs here

§6's gate is **STRUCTURE GATE INFORMATIVE**: stamping at exact GT positions beats the generator on S4 by
+0.1760 [+0.1732, +0.1787] and on S5 by +0.1269 [+0.1249, +0.1288]. So S4/S5 *can* tell a perfectly-placed
canonical beat from a generator's output; the metric is not degenerate.

And yet **the null method pays the same structure cost the trained arms paid**: N-R1 is worse than
GTF-TRUE on S4 (−0.0273 [−0.0298, −0.0246]) and S5 (−0.0246 [−0.0264, −0.0228]) — the identical
"event gain with structure trade-off" that R2 and R3 were assigned verdict C for. A training-free stamp
reproduces it.

The trade-off therefore is not evidence about additive adapters or cross-attention gates. On these
metrics, **anything that places more beats in roughly-right places pays it**, because S4/S5 are evaluated
at peaks detected in the signal being scored and a mis-placed sharp stamp is scored as bad structure.

## 4. What the null method cannot do — and why that is the point

N-R1 emits the *same canonical beat* at every event. It has no per-beat morphology by construction, and
its whole-window correlation with the target is **−0.0031** — indistinguishable from zero. Only with GT
positions does it reach 0.4680.

So the headline is not "stamping is a good method". It is:

> **The event metric family rewards, above every trained generator, a method that is by construction
> incapable of per-beat morphology and has zero whole-window correlation.**

A secondary observation with the same shape: N-R1-A (the full 83-sample beat template rather than the
26-sample QRS crop) scores **worse on events** (+0.3948) — 853 of 2,048 windows have overlapping stamps —
but **much better on matched-beat morphology** (0.8582 vs 0.5410). Matched morphology is conditional on
successful matching (`METRIC_SEMANTICS.md`), so a template that is simply longer buys 0.32 of it.

## 5. Limits — stated in the preregistration, restated against the numbers

1. **Not a test-set result.** `an0` / `k2s` are development validation and four of their windows were
   viewed before X4-0 was frozen. `kjd` / `ssx` were not loaded (`test_subjects_loaded: []`).
2. **Not a deployable method** (§4). N1 makes no claim that stamping should be used.
3. **N-GT is GT-R leakage**, labelled as such everywhere; it is the ceiling and the §6 gate, never a result.
4. **Single cohort, single detector threshold** (R1's frozen 0.35 / refractory 32, selected on internal dev
   only). The null's absolute score depends on that detector; its *dominance* over the trained arms is what
   N1 measured.
5. R2/R3/X4-0 verdicts are **not** reopened. What changes is the reading of what those numbers were
   measured against.

## 6. Provenance

Cohort, scoring and uncertainty are R2/R3's, imported not reimplemented: `scripts/r2_evaluate.py`
`score()` / `macro_rows()`; the 2,048 windows asserted element-wise against
`artifacts/x4_0_event_reliability/nfe_subset.json` (abort on mismatch — 19,834 GT beats confirmed); the
frozen S1 templates (`template_A.npy` file sha256 `1a67569f…`, T-B crop array sha256 `6f059015…`, both
asserted); R1's Global-TCN at state sha256 `0986a7af…`; paired subject-stratified bootstrap, 2,000
replicates, seed 20260901. Per-window comparator rows were joined to the cohort by explicit
`(subject, array_pos)` key — 2,048 / 2,048 matched for both B and GTF-TRUE — never by assumed row order.

Pins `6cd70cd` / `bf60cd7c` unchanged; A4 md5 `31c042d291052fbb6dc15263ad316be2` unchanged; C2 deferred.

## 7. What this justifies doing next

Recommendations only; nothing here is implemented.

1. **Every future arm reports N-R1 and N-CONST as required rows.** A method that does not beat a detector
   plus a template on the event axis has not shown event-axis value, and one that does not beat N-CONST has
   not shown it knows anything beyond rate.
2. **The open question is now sharply posed and is a morphology question.** N-GT reaches whole-window
   correlation 0.4680 and S4 0.1460 with *one canonical beat*; every generator sits near zero correlation
   at S4 ≈ 0.32. The gap a method must attack is **per-beat morphology at approximately-correct positions**,
   not event placement — which is exactly experiment 1 of the ideation list (oracle-timing WHAT probe,
   < 1 GPU-h): at GT anchors, can anything beat the template?
3. If experiment 1 says nothing beats the template, the honest paper is that PPG→ECG under these metrics is
   a detection problem with a canonical-beat renderer, and the generative framing is not supported.
