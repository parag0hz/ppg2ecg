# M1 — C1 Structural Mechanism Audit — REPORT

## **DIAGNOSTIC VERDICT: `CALIBRATION-ONLY / STRUCTURAL SUPPORT NOT FOUND`**

Under the rule frozen at `959eb60` before any M1 prediction existed.

> *The QRS-energy and p2p gains are better interpreted as calibration effects under this audit, not as
> evidence of improved local QRS reconstruction.*

**Consequence: the proposed interval → QRS-structure mechanism story is killed.**

Prereg `959eb60` · implementation `72937f5`. Existing C1 checkpoints only. **No training. No test access.
No prediction was ever translated. No oracle statistic was computed.**

---

## 1. Provenance

| item | value |
|---|---|
| start HEAD | `877841d` · deferral+prereg `959eb60` · implementation `72937f5` |
| submodules | PENGUIN `6cd70cd`, iMeanFlow `bf60cd7` — unchanged |
| A4 checkpoint | md5 `31c042d291052fbb6dc15263ad316be2` — unchanged |
| **C2 weight updates** | **0 — C2 training never started** (`docs/C2_DEFERRED_BEFORE_TRAINING.md`) |
| population | `an0` 1,024 + `k2s` 1,024 = **2,048 windows**, 19,834 GT beats |
| evaluation source | Gaussian seed 0, identical tensor for every arm and both NFEs |
| atlas cohort | 64 windows, 8 strata × 8, salt `c2-visual-atlas-v1`, verified before loading predictions |

### Checkpoints, resolved from C1 provenance (md5-matched, none overwritten)

| arm | path | kind | rounds | sha256 (48) |
|---|---|---|---:|---|
| B | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` | best | 66 | `557c70541f5cdd07819a3da04bb53477ac98827285507380` |
| H25 | `outputs/c1_imf_h25_seed42/checkpoint_best.pt` | best | 68 | `c1c1b09bd84843dd61e4bb6cefd887edacd4760a8d96bb00` |
| H50 | `outputs/c1_imf_h50_seed42/checkpoint_best.pt` | best | 101 | `e9eb78f37726dd157dbe987432c8571d2328fa1d00ea0342` |

## 2. The result, in one line

**Every aggregate ratio deviation improves for H50 over H25. Not one direct fixed-coordinate QRS error
does — and five of seven get clearly worse.**

| H50 vs H25 @ NFE 2 | Δ | 95 % CI | verdict |
|---|---:|---|---|
| **DIRECT fixed-coordinate errors** | | | |
| QRS-core squared error | +0.00076 | [−0.00174, +0.00325] | unresolved |
| QRS-core absolute error | −0.00251 | [−0.00452, −0.00050] | **worsens** |
| QRS-core derivative error | −0.00538 | [−0.00615, −0.00459] | **worsens** |
| QRS-core RMSE | +0.00112 | [−0.00098, +0.00318] | unresolved |
| QRS derivative RMSE | −0.00799 | [−0.00896, −0.00699] | **worsens** |
| QRS curvature error | −0.00584 | [−0.00659, −0.00506] | **worsens** |
| raw correlation | −0.01098 | [−0.01412, −0.00786] | **worsens** |
| background squared error | −0.00535 | [−0.00755, −0.00319] | **worsens** |
| **AGGREGATE ratio deviations** | | | |
| QRS-energy deviation | **+0.06309** | [+0.05111, +0.07486] | improves |
| QRS p2p deviation | **+0.04402** | [+0.03683, +0.05120] | improves |
| QRS slope deviation | **+0.02685** | [+0.01924, +0.03504] | improves |
| QRS max-derivative deviation | **+0.02685** | [+0.01924, +0.03504] | improves |

The sharpest single contrast: **`qrs_maxderiv_dev` improves (+0.02685) while `qrs_deriv_rmse` worsens
(−0.00799)**. The *ratio* of maximum derivative moves toward 1 while the *actual derivative error* rises.
That is calibration without structure, measured on the same underlying quantity.

## 3. QRS localisation — the contrast passes, but for the wrong reason

| family | `R_core` | `R_background` | `L = R_core − R_bg` | 95 % CI | verdict |
|---|---:|---:|---:|---|---|
| waveform squared | **+0.0021** | **−0.0407** | **+0.0428** | [+0.0290, +0.0582] | improves |
| derivative absolute | −0.0239 | −0.0195 | −0.0044 | [−0.0126, +0.0041] | unresolved |

`L` for the waveform family is clearly positive — but read the components. `R_core = +0.0021` means H50 is
**0.2 % better than H25 in QRS-core**, i.e. essentially nothing. `R_background = −0.0407` means H50 is
**4.1 % worse than H25 in background**. **`L > 0` is produced entirely by background degradation, not by
core improvement.** A localisation statistic can pass this way, and here it did; reporting `L` without its
components would have been misleading.

The derivative family, which cannot be satisfied by amplitude rescaling, is unresolved.

## 4. Where the improvement actually sits — region errors @ NFE 2

| arm | QRS-core sq | peri-QRS sq | background sq | QRS-core deriv | QRS RMSE | curvature |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.38532 | — | 0.17943 | 0.22336 | 0.59554 | 0.21774 |
| H25 | 0.36010 | — | **0.13144** | **0.22516** | 0.57734 | **0.22017** |
| H50 | **0.35934** | — | 0.13680 | 0.23053 | **0.57623** | 0.22601 |

Both interventions improve squared error against B everywhere, and both **worsen** the derivative and
curvature errors against B. H50 is worse than H25 in background, derivative and curvature.

## 5. Frequency structure @ NFE 2

For H50 vs H25: `F1__err_energy` **worsens** (−0.00108), `F3__err_energy` **worsens** (−0.00098),
`F4__err_energy` **worsens** (−0.00165), while `F3__ratio_dev` **improves** (+0.03840) and
`F4__ratio_dev` **worsens** (−0.02744). The same split again: in the QRS-relevant upper bands the
**energy ratio** moves toward 1 in F3 while the **actual reconstruction error energy** rises in F3 and F4.
HF is not treated as synonymous with QRS.

## 6. NFE 2 vs NFE 4 localisation

| metric | E2 (H50−B @2) | E4 (H50−B @4) | D = E2−E4 | 95 % CI | verdict |
|---|---:|---:|---:|---|---|
| QRS-core squared | +0.02598 | +0.01621 | +0.00977 | [+0.00808, +0.01152] | improves |
| QRS-core derivative | −0.00718 | −0.00962 | +0.00244 | [+0.00201, +0.00287] | improves |
| QRS-energy deviation | +0.06010 | +0.06762 | −0.00752 | [−0.01288, −0.00219] | **worsens** |
| QRS p2p deviation | +0.04282 | +0.04236 | +0.00046 | [−0.00404, +0.00498] | unresolved |
| background squared | +0.04264 | +0.02178 | +0.02085 | [+0.01884, +0.02286] | improves |

Waveform-error gains are somewhat larger at NFE 2, but **the QRS-energy calibration gain is larger at
NFE 4** (`D < 0`) and the p2p gain is unresolved. So the calibration effect is not NFE-2-specific.
No equivalence language is used.

## 7. Site-wise (exploratory)

| site | metric | B | H25 | H50 |
|---|---|---:|---:|---:|
| sternum | QRS-energy dev | 0.59198 | 0.59143 | **0.53268** |
| head | QRS-energy dev | 0.58290 | 0.57993 | **0.52398** |
| wrist | QRS-energy dev | 0.61250 | 0.60783 | **0.56421** |
| ankle | QRS-energy dev | 0.66437 | 0.68434 | **0.59203** |
| sternum | QRS-core deriv | 0.22542 | 0.22909 | 0.23235 |
| head | QRS-core deriv | 0.23174 | 0.23340 | 0.23778 |
| wrist | QRS-core deriv | 0.21734 | 0.22097 | 0.22403 |
| ankle | QRS-core deriv | 0.21883 | **0.21701** | 0.22776 |

The pattern is **consistent across all four sites**, not concentrated in proximal ones: H50 reduces the
energy deviation everywhere and raises the QRS-core derivative error everywhere. No measurement-information
causality is inferred.

## 8. Visual atlas — systematic observations only

Across the frozen 64-window cohort (8 contact sheets, no example selected after seeing results):

1. **B is systematically under-amplitude.** Its QRS deflections are visibly small relative to GT in nearly
   every window.
2. **Both interventions raise amplitude toward GT.** H25 and H50 produce visibly larger QRS spikes than B
   across sites and subjects — the visual counterpart of the improved energy and p2p ratios.
3. **H50 has a visibly rougher inter-beat baseline than H25.** The between-beat segments are consistently
   noisier in H50, which is the visual counterpart of its worse background, derivative and curvature errors.

This is the **"globally rescaled" pathology** that preregistration §14 criterion 4 names. It supports the
calibration reading rather than a local-structure reading.

## 9. Verdict evaluation against the frozen criteria

**Verdict A requires all four; criterion 1 fails.**

1. H50 vs H25 clearly improves a direct fixed-coordinate QRS-core waveform metric — **NO**. QRS-core
   squared error is unresolved and QRS-core derivative error clearly **worsens**. ✗
2. localisation `L` CI entirely > 0 for one of those families — yes for waveform, **but driven by
   background degradation** (§3). ✓ in letter only
3. one additional structure-sensitive quantity improves — yes (`qrs_energy_dev`, `qrs_ptp_dev`,
   `F3__ratio_dev`), **all of them ratio deviations**. ✓
4. atlas shows no obvious metric pathology — **NO**; it shows exactly the globally-rescaled pattern. ✗

**Verdict B does not fit either**: the H50-vs-H25 specificity is not weak — it is strong, and it lives
*entirely* in ratio deviations. That is verdict **C**'s description.

## 10. What this does NOT establish

- **No causal h = 0.5 claim.** The C1 arms are **not compute-matched** (B 66, H25 68, **H50 101** rounds).
  Everything above is an **OBSERVATION about the H50 checkpoint**, never a statement that h = 0.5 exposure
  **caused** it. The cause remains unresolved because H50 trained longer.
- **No training-seed robustness.** One seed per arm; the bootstrap covers this frozen development-window
  population only and is not training-run uncertainty.
- **No compute-matched evidence. No test evidence. No SOTA claim. No novelty claim.**
- M1 was designed **after** seeing C1 and is **not** independent of it. It is a mechanism-generating
  diagnostic, not a confirmatory discovery.

## 11. Recommended next experiment — recommendation only, not launched

Under verdict C the preregistered consequence is to **stop the H50 structural-mechanism direction**. The
interval → QRS-structure story should not be built into a method or a paper claim.

What the audit does leave standing is narrower and cheaper to check: **both** interventions reliably fix an
amplitude/energy under-calibration that the baseline has, at every site, while **degrading** derivative and
background fidelity. If anything is worth a follow-up it is that trade-off itself — and it is a
*calibration* question, not an interval-exposure question, so it does not require the C2 15-run
replication.

**Not started, and not to be auto-started:** the C2 15-run experiment, any distillation, any new method.

## Artifacts

`artifacts/m1_c1_structural_audit/`: `provenance.json`, `checkpoint_manifest.json/.csv`,
`cohort_manifest.json`, `metrics_window.csv`, `region_metrics.csv`, `event_error_profiles.csv`,
`spectral_metrics.csv`, `site_metrics.csv`, `paired_bootstrap.csv`, `localization.csv`,
`nfe_interaction.csv`, `decision.json`, 8 figures, and `visual_atlas/` with 8 contact sheets.
