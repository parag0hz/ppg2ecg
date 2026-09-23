# Consensus inference as conditional functional estimation — a minimal formalisation

**Revision 2026-09-23.** The first version of this note (commit `6b23d25`) was written before EXP-B's numbers were read
and is kept unchanged in history. This revision restructures it after EXP-B and B3-BOOT and adds §7.2's
shared / sample-specific decomposition. **Nothing here is fitted to data and no theorem beyond the derivations shown is
claimed.** Where a formula is exact only for a simplified estimator (the mean) it says so; the method uses the median.

Notation: condition `c` (a PPG window); reference structured output `x*` (the paired ECG); downstream functional `T`
(heart rate here; respiratory rate or SBP / DBP for PENGUIN's other targets); target value `T* = T(x*)`.
Per-sample depth `S` (network evaluations per sample), number of samples `K`, budget `B = K × S`.

## 7.1 Conditional functional estimation
A generator run at depth S defines a conditional law, and the functional inherits one:
```
X_{S,k} ~ q_{θ,S}(x | c),      Y_{S,k} = T(X_{S,k}),      k = 1 … K   (independent noise seeds)
```
The estimator is the sample median, and its population counterpart the conditional median:
```
m̂_{S,K}(c) = median(Y_{S,1}, …, Y_{S,K}),        m_S(c) = Median[ Y_S | c ].
```
For every window the error splits exactly into two terms:
```
m̂_{S,K} − T*  =  [ m_S − T* ]  +  [ m̂_{S,K} − m_S ]
                 ─────────────     ─────────────────
                 centre error:      finite-K error:
                 set by the model   shrinks with K, but only
                 and by depth S     through non-redundant samples
```
- **Depth S** can move the first term (it changes `q_{θ,S}` and hence `m_S`) and can also change the whole conditional
  law of `Y_S` — its spread and its shape — which sets the size of the second term.
- **Width K** acts only on the second term. It reduces it only to the extent that additional samples carry functional
  information the others do not: if every sample produced the same functional value, `m̂_{S,K} = m_S` for all K and width
  would buy nothing.

## 7.2 Why dependence between the samples' functional errors matters
**A model of the error (for intuition).** Write each sample's signed error as a part shared by all samples of the same
window plus a sample-specific part:
```
e_k(c) = Y_{S,k}(c) − T*(c) = b(c) + u_k(c),
```
where `b(c)` varies across windows with variance `σ_b²` and the `u_k(c)` are, given c, independent across k with variance
`σ_u²` and independent of `b`. Then, across windows,
```
Var(e_k) = σ² = σ_b² + σ_u²,      Cov(e_k, e_l) = σ_b²  (k ≠ l),      ρ = Corr(e_k, e_l) = σ_b² / σ².
```
So **the cross-sample error correlation ρ is the share of the error variance that is shared by all samples of a window**,
and the within-window spread of the sample functionals estimates `σ_u`, the sample-specific part. The two quantities
EXP-B / B3-BOOT found to order the consensus gain — low ρ̄, high within-window functional SD — are the relative and the
absolute measurement of the same component.

**For the mean of K samples (exact under this model):**
```
ē = b + ū,     Var(ē) = σ_b² + σ_u² / K = σ² [ ρ + (1 − ρ)/K ] = (σ²/K) [ 1 + (K − 1) ρ ],
K_eff = K / (1 + (K − 1) ρ).
```
- `ρ = 1`: every additional sample is redundant — `K_eff = 1`, width buys nothing.
- `ρ ≈ 0`: the samples' errors are independent — `K_eff → K`.
- In between, width removes only `σ_u²` and leaves the floor `σ_b² = ρ σ²` untouched. In the language of §7.1, `b(c)` plays
  the role of the centre error and `u_k` the sample's deviation from the centre.

**This expression is exact only for the simplified mean / equal-correlation setting and is used as intuition. The
empirical estimator is the sample median.** For the median of K i.i.d. draws with density `f_Y` positive and continuous
at the median, `Var(m̂_K) ≈ 1 / (4 K f_Y(m)²)` as K → ∞ — again a statement about the sample-specific part only; the
median is additionally robust to a minority of far-off draws, which is why it outperforms the mean on these heavy-tailed
errors (AB1; B3-BOOT §2.3). None of the regularity conditions is verified for our generators, and K ≤ 32 is not
asymptotic. Observed values for orientation (not fitted): ρ̄ = 0.56–0.86 across the 12 tested conditions, i.e. K_eff ≈
1.2–1.7 for K = 16 — most of the per-sample HR error variance is shared by the samples of a window.

**Finite-K note.** For K = 2 the sample median *is* the mean of the two draws, so the robustness of the median starts
only at K = 3. This is an operator property, visible as the K = 2 anomaly in EXP-B2 (B3-BOOT §2.3).

## 7.3 Fixed compute
With `B = K × S` fixed, choosing S also chooses `K = B / S`. Depth therefore changes, at once,
- the **per-sample functional quality** — the centre error `m_S − T*` and the spread of `Y_S | c`, and
- the **dependence structure** across samples — how much of the error is shared (ρ) versus sample-specific,

while width supplies the number of estimates the aggregation can use. For the mean-estimator intuition of §7.2, with
depth-dependent shared error `b_S` and sample-specific variance `σ_{u,S}²` (the mean squared error uses the uncentred
second moment `E[b_S²]`, which includes any systematic bias):
```
MSE(S; B) ≈ E[b_S²]  +  σ_{u,S}² · S / B .
```
The second term grows linearly with S at fixed B (each unit of depth costs samples). An **interior optimum** appears when
a little depth lowers the first term (or raises the removable share) steeply and further depth does not, so that the
`S / B` penalty takes over:
- **too little depth** → a large centre error, or functional errors that are highly redundant across samples (high ρ, so
  width has little to remove);
- **too much depth** → few samples (small K) and the finite-K reduction is lost.
If depth changes neither term, the optimum is the smallest supported S (pure width). The observed signs, stated without
fitting: for iMF, depth lowered the centre error up to S = 2; for PENGUIN, depth lowered both the centre error and the
shared share of the error (ρ̄ 0.86 → 0.65 from S = 1 to 8); for CD, depth did not change the centre error — S = 2 lowered
the shared share (ρ̄ 0.73 → 0.63) and raised the gain, but made each sample worse by about as much (individual error
+0.71 bpm), so the median did not move (EXP-B B1 / B3). These are consistent with the DW1 / DW2-A optima (CD S = 1, iMF S = 2, PENGUIN S = 4).
Because the method is the median, the expression above is intuition for why an interior optimum can exist, not a model of
where it is.

## What this note does not claim
- No theorem that width beats depth; whether it does depends on how `E[b_S²]` and `σ_{u,S}²` move with S for each model.
- No causal identification: EXP-B / B3-BOOT observe that the removable share orders the gain across 12 model-depth
  conditions; they do not manipulate it independently.
- The independence of the `u_k` given c is guaranteed only for the noise draws, not for how a detector responds to them;
  the shared / sample-specific split is a statistical description, not a statement about where the errors come from.
