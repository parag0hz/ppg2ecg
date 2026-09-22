# Consensus inference as functional estimation — a minimal formalisation

Written 2026-09-22, **before EXP-B's numbers were read** (prereg `50948e2`, EXP-E). No theorem is claimed; the note fixes the
vocabulary and the decomposition that EXP-A/B/D report against. Nothing here is fitted to data, and no parameter of it is
adjusted afterwards.

## 1. Objects
- Condition `c` (a PPG window); target structured output `x*` (the paired ECG); downstream functional `T` (here heart rate;
  respiratory rate, SBP / DBP for the other PENGUIN targets), target value `y* = T(x*)`.
- A conditional generator run at per-sample depth `S` (NFE per sample) defines a conditional law
  `X_S ~ q_{θ,S}(x | c)`; the induced functional is the random variable `Y_S = T(X_S)`.
- Budget `B = K × S` network evaluations per window; `K` independent samples `X_{S,1}, …, X_{S,K}`.
- Estimator (the paper's): `m̂_{S,K}(c) = median(Y_{S,1}, …, Y_{S,K})`.
- Population conditional median `m_S(c) = median_{Y_S | c} Y_S`; population conditional mean `μ_S(c) = E[Y_S | c]`.

## 2. The decomposition
For every window,
```
m̂_{S,K} − y*  =  ( m_S − y* )  +  ( m̂_{S,K} − m_S )
                 ─────────────     ─────────────────
                 model / depth-      finite-K estimation
                 dependent term      term
```
The first term does not depend on K: it is the distance between the centre of the functional's conditional law at depth S
and the truth. Depth S can move it (by changing `q_{θ,S}`), and so can anything about the model. The second term is the
sampling error of a K-sample median around its population median; it is the only part width K acts on, and it shrinks with
K for any fixed S as long as the samples are exchangeable draws from `q_{θ,S}(· | c)`.

Consequences that are checkable, and are what EXP-B measures:
- At **fixed K**, changing S changes the first term (centre) and may change the spread of `Y_S | c` (which sets the size of the
  second term). B1 reports the centre error `|m̂_{S,16} − y*|` (as the best available proxy for the centre at K = 16), the
  dispersion of `Y_S | c` (MAD, SD), the individual-sample error, and their difference (the "median gain").
- At **fixed S**, changing K changes only the second term. B2 reports `R(K, S) = E|m̂_{S,K} − y*|` against K. Under the
  decomposition, `R(K, S)` decreases towards a floor `R_∞(S)` set by the first term; the marginal gain of doubling K should
  shrink. The plot of `R(K) − R(32)` against `1/K` is examined for that shape; no parametric fit is treated as a result.
- Pooling helps only if the functional errors of the K samples are **not identical** given c. B3 measures the pairwise
  correlation of `Y_{S,k} − y*` across windows. If the samples were identical (a degenerate `q_{θ,S}`), the second term would be
  zero for every K and width could not help; if they are merely correlated, width helps less than for independent draws.
  This is a statement about the functional's law, not about how different the waveforms look.

## 3. Two pieces of textbook intuition (stated, not asserted for our model)
**Sample median.** For K i.i.d. draws from a law with density `f_Y` positive and continuous at its median `m`,
```
Var(m̂_K)  ≈  1 / ( 4 K f_Y(m)² )        (K → ∞)
```
so the finite-K term scales like `K^{-1/2}` in standard deviation, and it is smaller when the conditional law is concentrated
near its median (large `f_Y(m)`), i.e. when the sample functionals are tight. It is robust to a minority of far-off draws (a
detector that occasionally halves or doubles the rate), which is why the median rather than the mean was preregistered in P1
and why AB1 found the mean worse for iMF. None of the regularity conditions is verified for our generators: the draws are
i.i.d. given c by construction (independent noise seeds), but `f_Y` is unknown and the K used (≤ 32) is not asymptotic.

**Mean estimator (intuition for the budget trade-off).** If the estimator were the mean, with bias `b_S = μ_S − y*` and
conditional variance `σ_S²`,
```
MSE(K, S)  ≈  b_S²  +  σ_S² / K .
```
With a fixed budget `K = B / S`,
```
MSE(S; B)  ≈  b_S²  +  S · σ_S² / B .
```
The second term grows linearly in S at fixed B: every unit of depth is paid for by a lost sample. The first term can fall
with S (if depth reduces the centre error) or not. An **interior optimum** in S exists when `b_S²` falls steeply at small S
and flattens afterwards, so that the loss `S σ_S² / B` overtakes the gain from further depth — "refine enough, then sample
wide". If `b_S` does not depend on S, the optimum is pure width (S at its minimum supported value). If `σ_S²` collapses at
small S (near-deterministic samples) the second term is small everywhere and the whole trade-off is governed by `b_S`, i.e.
depth is the only lever. The paper's estimator is the median, so this expression is intuition only; the median analogue
replaces `σ_S²` by `1 / (4 f_{Y_S}(m_S)²)`.

## 4. How the decomposition reads the existing results (qualitative; no numbers fitted)
- DW1 / DW2-A found the optimum at S = 1 for CD, S = 2 for iMF, S = 2–4 for PENGUIN. In the language above: for CD the centre
  term does not improve with depth, so the budget belongs to K; for iMF and PENGUIN a small amount of depth lowers the
  centre error (or tightens `Y_S | c`) by more than the K it costs, after which the `S σ_S² / B` penalty dominates.
- Whether the improvement from S = 1 → 2 (iMF) and 1 → 4 (PENGUIN) is a centre effect, a dispersion effect, or both is exactly
  what B1 separates. The note makes no prediction about which it is.
- A near-deterministic one-step generator (PENGUIN at S = 1 has almost identical waveforms, DW2-B) can still show a positive
  median gain in the HR functional if the functional extractor injects sample-to-sample variation. In the decomposition this is
  ordinary finite-K reduction of `Y_S`'s spread; its origin (generator or extractor) is invisible to the estimator and is what
  B3's error correlation probes. The note does not claim either origin.
- Depth's effect on the waveform (FD) is outside the decomposition: `T` discards it. This is why depth can improve FD while
  leaving `R(K, S)` unchanged or worse, and why the paper must report waveform metrics separately from the functional.

## 5. What the note does not do
- It does not prove that width beats depth; whether it does is an empirical property of `b_S` and `σ_S` for each model.
- It does not assume independence beyond the construction (independent noise seeds); conditional exchangeability is enough
  for the decomposition, and the correlation of *errors across windows* measured in B3 is a different quantity.
- It does not assign a mechanism to depth; it only names the two terms depth can move.
