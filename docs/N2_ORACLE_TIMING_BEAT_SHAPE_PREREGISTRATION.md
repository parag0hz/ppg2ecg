# N2 — Given the beat's position for free, does PPG carry its shape?

**Preregistration. Frozen on commit and push; never edited afterwards.**

Start HEAD `fed954c`. Pins `external/PENGUIN` @ `6cd70cd`, `external/iMeanFlow` @ `bf60cd7c` — unmodified.

---

## 1. Question

N1 showed that a detector plus **one fixed template** out-scores every trained generator in this program
on the event axis (`f1_excess` +0.4866 vs +0.3582), that most of a generator's event score is beat *rate*
rather than placement (N-CONST +0.2949), and that with **exact GT positions** the same single canonical
beat reaches whole-window correlation **0.4680** and S4 **0.1460**, where every generator sits near zero
correlation at S4 ≈ 0.32.

That leaves exactly one question standing between this program and a method:

> **Q.** Hand every arm the correct R position for free. Does PPG then let *anything* reconstruct a beat
> better than stamping one fixed template?

- **If yes** — PPG carries per-beat shape information, and a method that separates *when* (a detector) from
  *what* (a shape model on a beat-canonical coordinate) has something to learn. That is the decomposition
  currently under consideration.
- **If no** — PPG carries beat *timing* and nothing more about morphology, every shape-side method in this
  program and in the proposal is dead, and the honest paper is that PPG→ECG under these metrics is a
  detection problem with a canonical-beat renderer.

Either outcome ends the question. That is why N2 runs before any method is built.

## 2. N2 is an oracle probe end to end — nothing here is a performance number

**Every arm receives ground-truth R positions at inference.** This is the premise of the question, not a
leak to be hidden: N2 asks what is observable *conditional on correct timing*. No N2 number is a
deployable result, no N2 number may be compared to a published PPG→ECG figure, and every table and figure
carries the label **"(GT-R anchor; oracle coordinate — diagnostic only)"**.

## 3. Arms

All arms predict the same object: the 83-sample beat window `[r − 32, r + 51]` at 128 Hz (the frozen S1
geometry, `template_geometry()`), at every GT R anchor.

| arm | what it puts at the anchor | trained? |
|---|---|---|
| **T-FIXED** | the frozen S1 `template_A` — one canonical beat everywhere | **no** — 0 parameters |
| **REG** | MSE regression from the PPG context window | yes |
| **IMF** | one-step Improved MeanFlow from the PPG context window | yes |
| **REG-SHUFFLE** | REG's weights fed a **partner beat's** PPG context | no (control) |
| **IMF-SHUFFLE** | IMF's weights fed a **partner beat's** PPG context | no (control) |

**Why the shuffle controls decide the stage.** A trained model can beat a fixed template without reading
the PPG at all — by learning a better *average* beat, or one conditioned on rate. The shuffle arms feed
the identical weights a deranged partner's PPG. If a trained arm beats T-FIXED but does **not** lose that
advantage under shuffle, the gain is not PPG-derived and does not count (§6).

**PPG context.** `[r − 96, r + 96]`, 193 samples = 1.51 s, centred on the anchor — wide enough for the
pulse-arrival delay V1 measured (IQR 227 ms ≈ 29 samples) in either direction. Fixed here.

## 4. Data, splits, training — identical for both trained arms

- Subjects are R1's frozen split (`artifacts/r1_global_rhythm/subject_split.json`): **train** =
  `fex l38 n31 ngh p5d p9p qm9 trh tz8 w4p`; **internal dev** = `u7y e61` (checkpoint selection only);
  **evaluation** = `an0 k2s`. `kjd` / `ssx` are never loaded.
- Beats: `select_subset(salt="n2-beat-v1", n_take=1024)` windows per training subject, every GT R peak in
  them with a complete 83-sample window and a complete 193-sample PPG context.
- REG and IMF share **one encoder architecture, one beat count, one optimizer, one seed (42), and one
  exact optimizer-step budget**; only the objective and the sampler differ. Parameter counts are reported
  and will differ slightly (IMF needs a (t, r) embedding); the difference is declared, not hidden.
- Checkpoint selection: each arm's own deterministic internal-dev metric (REG: dev MSE; IMF: fixed-bank
  iMF MSE), never a metric on `an0`/`k2s`. **U3 showed this asymmetry can do real work**, so both arms are
  additionally evaluated at their final step and both readings are reported (§7).

## 5. Evaluation — at the fixed oracle coordinate, no detection anywhere

Scored on the same frozen 2,048-window cohort N1 used (`an0`, `k2s`, asserted element-wise against
`artifacts/x4_0_event_reliability/nfe_subset.json`, 19,834 GT beats).

Because every arm writes into the same GT coordinate, **no R-peak detection is involved in any metric**.
Per beat:

| metric | definition |
|---|---|
| **beat correlation** (primary) | Pearson r between the predicted and the true 83-sample beat |
| beat RMSE | per-beat RMSE |
| **S4** `qrs_deriv_rmse`, **S5** `qrs_curvature_err` | frozen `m1_structural.qrs_core_morphology`, GT-R-anchored ±80 ms core |
| p2p ratio deviation, QRS-width error | as in `m1_structural` / `metrics` |

Uncertainty: paired bootstrap clustered by **subject**, 2,000 replicates, seed 20260911, same rule as U2/N1.

Secondarily, each arm's beats are stamped back into the full window and scored with N1's imported
`r2_evaluate.score()` so the numbers sit in the same table as N1's — reported, never used for the verdict.

## 6. Decision rule — fixed here

Let `BEST = max(REG, IMF)` on beat correlation, and `SHUF` be that arm's own shuffle control.

| verdict | rule |
|---|---|
| **PPG CARRIES BEAT SHAPE** | `BEST − T-FIXED` paired CI entirely > 0 **and** point ≥ **+0.05**, **and** `BEST − SHUF` paired CI entirely > 0 with point ≥ **+0.025** |
| **MARGINAL** | `BEST − T-FIXED` CI entirely > 0 but either magnitude condition fails |
| **PPG DOES NOT CARRY BEAT SHAPE** | `BEST − T-FIXED` CI includes 0 or is entirely < 0 |

`+0.05` is A0's own preregistered morphology-correlation margin, reused so the shape question is held to
the bar this program already set. `+0.025` (half of it) is the minimum PPG-attributable share required
before any gain may be called PPG-derived.

**A second, independent reading is recorded and cannot change the verdict:** whether `IMF − REG` on beat
correlation has its CI entirely > 0. That asks whether flow matching buys anything over plain regression
*at beat scale*, which is the one place this program has never tested it. A5/A6 showed OT-CFM-1 collapses
onto an MSE regressor at window scale; N2 measures the same comparison where the timing blur is removed.

## 7. Reported regardless of verdict

- Both checkpoint readings (own-best and final-step) for both trained arms.
- T-FIXED's own numbers, which are N1's `N-GT` arm recomputed at beat scale — they must agree, and a
  disagreement is a bug, reported as such.
- Per-subject and per-site breakdowns (`an0` and `k2s`, four body sites).
- The full stamped-back table beside N1's.

## 8. Firewalls

- `kjd` / `ssx` never loaded; `ER.assert_no_test_subjects` at every entry; `test_subjects_loaded: []`.
- Training subjects and evaluation subjects are disjoint and asserted; the template is S1's, built from
  training subjects only, hashes asserted.
- Every table carries the oracle label of §2.
- Frozen A4 md5 `31c042d291052fbb6dc15263ad316be2` re-checked; pins asserted; C2 remains deferred.
- No checkpoint, prediction or raw data enters git.

## 9. Cost

Two small models on 83/193-sample windows, ~100k beats. Estimated well under 1 GPU-hour in total.

## 10. Reporting

`docs/N2_ORACLE_TIMING_BEAT_SHAPE_REPORT.md`: the arm × metric table at the oracle coordinate, the §6
verdict with both magnitude conditions shown, the shuffle controls, the IMF−REG reading, both checkpoint
readings, and an explicit statement of which of the two futures in §1 the result selects.
