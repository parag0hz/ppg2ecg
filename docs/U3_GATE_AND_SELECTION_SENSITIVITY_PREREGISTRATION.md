# U3 — Do U2's conclusions survive its two design defects?

**Preregistration. Frozen on commit and push; never edited afterwards.**

Start HEAD `c753d13`. Upstream pins `external/PENGUIN` @ `6cd70cd`, `external/iMeanFlow` @ `bf60cd7c`;
neither is modified. No model is trained in this stage.

---

## 1. Question

U2 returned **PARTIAL (3 of 5)** and its own report identified two defects in its design. U3 asks whether
either changes what U2 concluded:

- **U3-A** — U2's §8 informativeness gate is **two-sided**, so it suppressed a 13× DBP failure on
  MIMIC-BP. What does the stage conclude under a **one-sided** gate?
- **U3-B** — the two arms' selection criteria peak at **80 %** (arm C) and **37 %** (arm I) of the matched
  budget. Is arm I's deficit an artefact of being evaluated at an earlier checkpoint?

Neither question needs new training. U3-B is answered from `checkpoint_last.pt`, which every U2 run already
wrote at **exactly 14,000 optimizer steps**.

## 2. What U3 may and may not do

**U3-A is a sensitivity analysis and can never replace a U2 verdict.** The one-sided gate is a rule change
proposed *after seeing U2's numbers*, which is precisely what preregistration exists to prevent. U2's
frozen verdict stands as published. U3-A reports what the same frozen numbers would have said under the
corrected rule, labelled as such everywhere, and its purpose is to bound how much the defect mattered —
not to re-decide the stage.

**U3-B produces new real-data metrics** and is therefore preregistered here in full, before any of them
exists.

## 3. U3-A — one-sided gate, on U2's frozen numbers

The corrected rule, fixed here:

> A cell is gated **only when it would otherwise PASS**. If `Δ_C < |δ|` (arm C's own NFE 1→50 span is
> below the margin) **and** the cell would be non-inferior, it is UNINFORMATIVE and cannot contribute.
> A cell that **fails** its margin always counts, whatever arm C's span.

Rationale, stated before the recomputation: a gate exists to stop an arm claiming credit on a metric where
the sampling budget buys nothing. A failure needs no such protection — arm I missing a margin by 13× is
informative about arm I regardless of how flat arm C is.

Everything else — margins, the §9 dataset rule, the stage thresholds, the paired cluster bootstrap, the
UCI-BP exclusion — is U2's, unchanged. Inputs are the frozen `artifacts/u2_paired/paired_*.csv`; no metric
is recomputed and no arm is re-run.

**Reported:** the per-dataset and stage verdicts under both rules, side by side, with every cell whose
status changes named explicitly.

## 4. U3-B — both arms at a fixed step count

For all six corpora and both arms, evaluate **`checkpoint_last.pt`** — the weights after exactly 14,000
optimizer steps — through U2's evaluation code path, unchanged:

- same NFE grid (arm C 1/2/4/50, arm I 1/2/4), same 4 noise draws (seeds 0–3), same shared `e0`;
- same test splits, same U2-D5 evaluation-window budget and the same index sets, so U3-B's numbers are
  paired with each other **and** comparable to U2's best-checkpoint numbers window for window;
- same metrics, margins, gate (the U2 two-sided one **and** the U3-A one-sided one, both reported), and
  the same paired cluster bootstrap, seed 20260911.

Checkpoint provenance is asserted, not assumed: every loaded `checkpoint_last.pt` must report
`train_state.opt_steps == 14000`, and the run aborts for that corpus otherwise.

### What U3-B can conclude

| outcome | reading |
|---|---|
| arm I's deficit shrinks materially at fixed steps | U2's comparison was partly a **selection artefact**; U2's per-dataset verdicts are not safe as published |
| arm I's deficit is unchanged or grows | the selection asymmetry is **not** the explanation; U2's verdicts stand on their own terms |
| arm C degrades at 14,000 steps relative to its best | arm C's criterion was tracking something real and the fixed-step comparison is the less fair one; reported as such |

**No verdict is switched on U3-B.** It is a stated alternative explanation being tested, and whichever way
it falls it is reported against U2's numbers rather than replacing them. Both checkpoint choices are
defensible — "each arm at its own best" and "both arms at equal steps" — and U3 publishes both rather than
picking the flattering one.

## 5. Decision rule, fixed now

For each dataset × metric × k, the **shift** is `(I_k − C_50)` at last-checkpoint minus the same quantity
at best-checkpoint, with a paired cluster bootstrap over the same subjects.

| verdict | rule |
|---|---|
| **SELECTION ARTEFACT** | arm I's deficit shrinks by **≥ 50 %** of the margin on the primary cell of **≥ 3 of 5** countable datasets |
| **PARTIAL ARTEFACT** | the same on 1 or 2 datasets |
| **NOT AN ARTEFACT** | no dataset shifts by that much, or the shifts go the other way |

The 50 %-of-margin threshold is fixed here because a shift smaller than half a margin cannot flip a verdict
and so cannot be the explanation for one.

## 6. Explicitly out of scope

- **ABP normalisation.** U2 §8 recommended settling the ABP mechanism; that recommendation is **withdrawn**
  (U2 report §10): **A8** already returned a frozen SCALE SENSITIVITY CONFIRMED on MIMIC-BP, with one
  global train-only affine taking iMeanFlow-1 from RMSE 32.3 to 18.2 mmHg while OT-CFM was unchanged.
  Whether that fix transfers to the 4 s protocol and to UCI-BP (which A8 never covered) is a real question
  and needs its own preregistration and its own training. It is **not** part of U3.
- **More seeds, LOSO / k-fold splits, retraining of any kind.** U3 trains nothing.
- Any change to U2's frozen verdict.

## 7. Cost

U3-A: no compute. U3-B: 12 evaluation runs under the U2-D5 budget ≈ **5.2 GPU-h**, sequential.

## 8. Firewalls

- No weight update in this stage; `checkpoint_best.pt` and `checkpoint_last.pt` of every U2 run are
  read-only and their sha256 are recorded.
- WildPPG `kjd` / `ssx` re-asserted absent from every split list at load.
- Frozen A4 md5 `31c042d291052fbb6dc15263ad316be2` and arm-U sha256 `20ba7234…` re-checked at the end.
- `external/PENGUIN` and `external/iMeanFlow` untouched; pins asserted.
- C2 remains deferred. No checkpoint, prediction or raw data enters git.

## 9. Reporting

`docs/U3_GATE_AND_SELECTION_SENSITIVITY_REPORT.md`: the two-rule verdict table with every changed cell
named; the best-vs-last grid for all six corpora and both arms; the §5 shift verdict; and a restatement of
which U2 conclusions survive each analysis and which do not.
