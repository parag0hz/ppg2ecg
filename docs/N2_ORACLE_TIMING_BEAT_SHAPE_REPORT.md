# N2 — Given the beat's position for free, does PPG carry its shape? REPORT

Preregistration `docs/N2_ORACLE_TIMING_BEAT_SHAPE_PREREGISTRATION.md` (`d552eb4`, pushed before any N2
number existed). **(GT-R anchor; oracle coordinate — diagnostic only.)** 48.5 s on one GPU.

---

## 1. Verdict: **PPG DOES NOT CARRY BEAT SHAPE**

Hand every arm the correct R position for free. A **single fixed canonical beat**, the same one at every
anchor, reaches per-beat correlation **0.9062**. A regressor that reads a 1.5 s PPG context at that anchor
reaches **0.9057** — it is **worse**, with the paired CI entirely on the losing side.

| arm | beat corr ↑ | beat RMSE ↓ | S4 ↓ | S5 ↓ | p2p dev ↓ |
|---|---|---|---|---|---|
| **T-FIXED** (one template, 0 parameters, no training) | **+0.9062** | **0.1722** | **0.0852** | 0.0628 | **0.1357** |
| REG (611,539 params, MSE) | +0.9057 | 0.1997 | 0.0887 | **0.0613** | 0.1942 |
| REG-SHUFFLE (same weights, partner's PPG) | +0.9018 | 0.2105 | 0.0910 | 0.0625 | 0.2046 |
| IMF (687,059 params, one-step MeanFlow) | +0.6752 | 0.2752 | 0.2652 | 0.3565 | 0.1772 |
| IMF-SHUFFLE | +0.6694 | 0.2836 | 0.2662 | 0.3570 | 0.1871 |

| §6 condition | required | observed | |
|---|---|---|---|
| `BEST − T-FIXED` on beat corr | CI > 0 and ≥ **+0.05** | **−0.0005** [−0.0007, −0.0003] | **fails, on the wrong side** |
| `BEST − SHUFFLE` on beat corr | CI > 0 and ≥ **+0.025** | +0.0039 [+0.0037, +0.0042] | passes CI, **6× below the bar** |

`BEST` = REG (0.9057 > IMF's 0.6752). The first condition is not merely unmet — the trained model is
*significantly worse than the template*. Verdict is **PPG DOES NOT CARRY BEAT SHAPE**.

## 2. What the shuffle control shows

REG loses only **0.0039** of correlation when fed a *different beat's* PPG. That is the entire
PPG-attributable share of its output: **0.4 % of a correlation scale on which the template alone already
achieves 0.906.** The regressor is, to three decimal places, emitting a canonical beat regardless of its
input — which is precisely what a model learns when the conditioning signal carries no information about
the quantity being predicted.

There is headroom to be had — a fixed template leaves `1 − r² = 0.179` of per-beat variance unexplained —
and **PPG does not address it.** The negative result is not "there is nothing left"; it is "what is left
is not in the PPG."

## 3. Flow matching is not the issue, and at beat scale it is actively harmful

The secondary reading, recorded in advance and unable to change the verdict:

**IMF − REG on beat correlation = −0.2306 [−0.2316, −0.2297].** One-step MeanFlow at beat scale is
dramatically worse than plain MSE regression on the identical encoder, data, optimizer, seed and step
budget — and worse than a zero-parameter template by −0.2311. Its S4 is 3.1× the template's and its S5
5.7×.

A5/A6 showed OT-CFM-1 *collapses onto* an MSE regressor at window scale. N2 removes the timing blur that
was the suspected cause and the one-step generative objective still loses — now by a much wider margin.
The stochastic objective is paying a variance cost for modelling a distribution that, conditional on
timing, is nearly a point mass.

## 4. Consequence: the decomposition idea is dead, and the program's direction is decided

§1 of the preregistration named two futures. N2 selects the second, and does so without ambiguity.

**What is now ruled out, by measurement:**

- The proposed "detector for *when*, flow matching for *what* on a beat-canonical coordinate" method.
  Its entire premise was that a shape model on an aligned coordinate could beat a template. It cannot —
  the best trained arm is 0.0005 *behind* a template, and 0.0039 of its output is PPG-attributable.
- Every shape-side proposal in `ASSESSMENT_TOPTIER_AND_IDEATION_2026-09-03.md` ranked below the
  oracle-timing probe (BEATSEG, CANON-WARP, and the shape half of the fusion line). The assessment
  itself specified this: *"(ii)(iii)가 (i)를 못 이기면 … PPG는 템플릿 이상 형태를 안 담음, 모든 형태-측
  방법 폐기."*
- Any framing in which a *better generative model* is the contribution on this task.

**What N1 + N2 establish together:**

> PPG determines *when* a beat occurs, to a limited and measurable accuracy (R1: F1@50 = 0.62), and
> carries essentially **nothing** about what that beat looks like beyond the population-average QRS
> (N2: +0.0039 of correlation). Under the metrics this literature uses, PPG→ECG reconstruction is a
> **detection problem followed by a canonical-beat renderer** — and that composition, run in N1, already
> beats every trained generator in this program on the event axis (+0.4866 vs +0.3582).

## 5. Limits

1. **Oracle coordinate throughout.** Every arm got GT R positions; nothing here is a deployable number.
   The result is *conditional on perfect timing* — which makes the negative result stronger, not weaker,
   since real methods have worse timing.
2. **Correlation is a compressed scale near 0.9.** A fixed template already achieves 0.906, so the
   dynamic range for improvement is small. That is why §6 also required the shuffle margin, and why the
   RMSE, S4, S5 and p2p columns are reported: the template wins on beat RMSE (0.1722 vs 0.1997) and p2p
   deviation (0.1357 vs 0.1942) too, where the scale is not compressed.
3. **One architecture, one seed (42), one step budget** (6,000 steps, batch 256, 84,149 training beats).
   A larger or different shape model is not excluded in principle — but it would have to find
   information that the shuffle control says is not being used at all.
4. WildPPG only, wrist/sternum/head/ankle sites, development-validation subjects `an0` / `k2s`.
   `kjd` / `ssx` never loaded.
5. The 83-sample window is QRS-dominated by construction (`[r−0.25 s, r+0.40 s]`); P-wave and full T-wave
   morphology beyond it were not tested, and a P/T-specific probe is not excluded.

## 6. Provenance

Splits are R1's frozen `subject_split.json` — train `fex l38 n31 ngh p5d p9p qm9 trh tz8 w4p`, internal
dev `u7y e61`, evaluation `an0 k2s`, asserted disjoint. 84,149 training / 4,237 dev / 16,860 evaluation
beats, deterministic `select_subset`. Template is S1's `template_A.npy`, file sha256 `1a67569f…`,
asserted. REG and IMF share one encoder architecture, beat set, optimizer (AdamW 1e-3 / wd 0.01),
batch 256, seed 42 and 6,000 steps; only the objective and sampler differ. Both own-best and final-step
checkpoints were evaluated and agree (IMF's best *is* its final step). Shuffle is a verified derangement.
Paired subject-clustered bootstrap, 2,000 replicates, seed 20260911.

Pins `6cd70cd` / `bf60cd7c` unchanged; A4 md5 `31c042d291052fbb6dc15263ad316be2` unchanged; C2 deferred.

## 7. What this justifies doing next

Recommendations only; nothing here is implemented.

1. **Stop proposing shape-side generative methods for PPG→ECG on these data.** N2 is the cheapest possible
   test of their premise and it fails by a wide, well-controlled margin.
2. **The defensible paper is the measurement paper**, and N1 + N2 are its spine: the event metric is won by
   a method incapable of morphology (N1), the morphology that remains is not in the PPG (N2), and the
   literature's protocol does not test either (U1's leakage and phase-blind metric, D2's uninformative
   RMSE, D4's non-monotone NFE). That is a complete, reproducible argument that this task's benchmarks do
   not measure what their papers claim.
3. **If a method contribution is still wanted**, it must come from changing the *problem*, not the model:
   a different conditioning modality, a P/T-wave-specific target the 83-sample window does not cover, or
   an explicitly personalised setting — Slapničar et al. (PMC11014403) already report that personalised
   models succeed where general ones fail, which is consistent with everything measured here.
