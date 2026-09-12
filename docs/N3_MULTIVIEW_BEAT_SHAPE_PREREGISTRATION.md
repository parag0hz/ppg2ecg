# N3 — Four PPG views of one heartbeat: does the shape appear?

**Preregistration. Frozen on commit and push; never edited afterwards.**

Start HEAD `812c0d1`. Pins `external/PENGUIN` @ `6cd70cd`, `external/iMeanFlow` @ `bf60cd7c` — unmodified.

---

## 1. Question

N2 answered "given the correct R position, does PPG carry the beat's shape?" with **no**: a zero-parameter
template reaches per-beat correlation **0.9062**, a 611k-parameter regressor reading a 1.5 s PPG context
reaches **0.9057**, and only **+0.0039** of that is PPG-attributable (shuffle control).

N2 gave every model **one** PPG channel. WildPPG records **four simultaneous sites** — ankle, head,
sternum, wrist — against **the same ECG**, verified: for a given `(subject, window_index)` the four rows
carry a byte-identical ECG target and different PPG. Every stage of this program, N2 included, has treated
those four rows as four independent samples and has **never fused them**.

> **Q.** Is the N2 ceiling a property of **PPG as a modality**, or of **one PPG channel**?

The four sites differ in pulse-arrival time by tens to hundreds of milliseconds (V1: PAT IQR 227 ms), and
the *differences between them* are not available to any single-view model. If cardiac contraction shape
leaves a trace anywhere in peripheral optics, a four-view model is where it should appear.

- **If it appears** — the N2 negative was about channel count, not about PPG, and a multi-view conditional
  generator (flow matching included) has something real to model. The method direction is alive.
- **If it does not** — PPG as a modality does not carry beat morphology, N2's verdict generalises, and the
  shape-side direction is closed on the strongest available evidence rather than the cheapest.

## 2. N3 is an oracle probe end to end

Every arm receives ground-truth R positions at inference, exactly as in N2. Every table carries
**"(GT-R anchor; oracle coordinate — diagnostic only)"**. No N3 number is a deployable result.

## 3. Arms

All arms predict the same object as N2: the 83-sample beat window `[r − 32, r + 51]` at every GT anchor.
All arms are trained and evaluated on **the same beat set** — only beats whose window has all four sites
present and a complete 193-sample context in each.

| arm | input at the anchor | trained? |
|---|---|---|
| **T-FIXED** | nothing — the frozen S1 `template_A` at every anchor | **no**, 0 parameters |
| **REG-1** | **one** site's PPG context `[1, 193]`, sites pooled as N2 did | yes |
| **REG-4** | **four** sites' PPG contexts stacked `[4, 193]`, fixed order `ankle, head, sternum, wrist` | yes |
| **IMF-4** | same `[4, 193]`, one-step Improved MeanFlow | yes |
| **REG-4-SHUFFLE**, **IMF-4-SHUFFLE** | the same weights fed a **partner beat's** four-view context | no (controls) |

REG-1 is the matched single-view control: same beats, same architecture except the input channel count,
same optimizer, seed and step budget. It is what makes "the fourth view added something" a measurable
claim rather than a comparison across N2's different beat set.

## 4. Data, splits, training

Unchanged from N2 except the input: R1's frozen split (train `fex l38 n31 ngh p5d p9p qm9 trh tz8 w4p`,
internal dev `u7y e61`, evaluation `an0 k2s`); `select_subset(salt="n2-beat-v1", n_take=1024)`;
GT R peaks with complete windows; `kjd` / `ssx` never loaded. Beats are now **grouped by
`(subject, window_index)`** and kept only when all four sites are present — the realised beat count will
differ from N2's and is reported.

Every trained arm shares one architecture (channel count aside), one beat set, AdamW 1e-3 / wd 0.01,
batch 256, seed 42 and **6,000 optimizer steps**. Checkpoint selection is each arm's own deterministic
internal-dev metric; both own-best and final-step readings are reported.

## 5. Evaluation

Identical to N2: at the fixed oracle coordinate, no detection anywhere. Per-beat correlation (primary),
beat RMSE, S4 `qrs_deriv_rmse`, S5 `qrs_curvature_err`, p2p deviation. Paired bootstrap clustered by
subject, 2,000 replicates, seed 20260911.

## 6. Decision rule — fixed here, identical bars to N2

Let `BEST = max(REG-4, IMF-4)` on beat correlation and `SHUF` be that arm's own shuffle control.

| verdict | rule |
|---|---|
| **MULTI-VIEW CARRIES BEAT SHAPE** | `BEST − T-FIXED` CI entirely > 0 **and** point ≥ **+0.05**, **and** `BEST − SHUF` CI entirely > 0 **and** point ≥ **+0.025** |
| **MARGINAL** | `BEST − T-FIXED` CI entirely > 0 but either magnitude fails |
| **MULTI-VIEW DOES NOT CARRY BEAT SHAPE** | `BEST − T-FIXED` CI includes 0 or is entirely < 0 |

The bars are N2's (A0's +0.05 morphology margin; half of it for the PPG-attributable share) so N2 and N3
are directly comparable and the extra views are held to the same standard the single view failed.

**Secondary, recorded and unable to change the verdict:** `REG-4 − REG-1` on beat correlation — does the
fourth view add anything at all, independently of whether it clears the template? A positive result here
with a failed primary means the views carry information that is real but too small to matter, and must be
reported in those words.

## 7. Reported regardless of verdict

- Per-site single-view numbers, so "which site is best" is visible and the pooled REG-1 is not hiding a
  strong site.
- Both checkpoint readings for every trained arm.
- T-FIXED recomputed on the N3 beat set; it must agree with N2's 0.9062 up to the change of beat set, and
  a disagreement beyond that is a bug, reported as such.

## 8. Firewalls

- `kjd` / `ssx` never loaded; train / dev / eval subject sets asserted disjoint.
- Oracle label of §2 on every table.
- S1 template hash asserted; frozen A4 md5 `31c042d291052fbb6dc15263ad316be2` re-checked; C2 deferred.
- No checkpoint, prediction or raw data enters git.

## 9. Cost

N2 trained two beat-scale models in 48.5 s total. N3 trains four. Estimated well under 1 GPU-hour.

## 10. Reporting

`docs/N3_MULTIVIEW_BEAT_SHAPE_REPORT.md`: the arm × metric table at the oracle coordinate, the §6 verdict
with both magnitude conditions, the shuffle controls, the `REG-4 − REG-1` reading, per-site breakdown, and
an explicit statement of which of the two futures in §1 the result selects.
