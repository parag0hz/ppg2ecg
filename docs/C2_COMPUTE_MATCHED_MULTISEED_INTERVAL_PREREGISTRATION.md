# C2 — Compute-Matched Multi-Seed Replication of Interval-Exposure Effects — PREREGISTRATION

**Status:** frozen at this commit, pushed **before any C2 weight update**.
**Population:** WildPPG **development only** — `an0`, `k2s`. Test subjects `kjd`/`ssx` are never loaded.

**NO NEW METHOD. NO TEST. NO DISTILLATION. NO NEW h VALUES. NO HYPERPARAMETER TUNING.**
C2 has no method-development verdict. Even on a full pass, no method is implemented.

---

## 1. Question

C1 returned `TARGET h=0.5 EXPOSURE SUPPORTED` (`94bc795`) with three limitations: a single training seed;
H50 received **101 rounds against B's 66 and H25's 68**, i.e. ~50 % more optimisation; and H50 improved
NFE 4 comparably, so the effect was not narrowly NFE-2-specific.

> **Under exactly matched optimisation compute and multiple training seeds, does h = 0.5 exposure
> reproducibly improve QRS-structure fidelity, especially relative to the h = 0.25 specificity control?**

Sub-questions, none of which C2 assumes an answer to: does the H50 advantage survive exact compute
matching (Q1); does it replicate across seeds (Q2); is the H50-specific benefit still concentrated in
QRS-energy / amplitude metrics (Q3); is H50 actually NFE-2-specific (Q4)?

## 2. Provenance, verified before writing this document

| item | value |
|---|---|
| start HEAD | `94bc795ca37ceaa2a9694ce9eb38ed7c7460b46b` (C1 final) |
| C0 result `d618968` · C1 prereg `b32c952` · C1 final `94bc795` | |
| frozen A4 checkpoint | md5 `31c042d291052fbb6dc15263ad316be2` |
| submodules | PENGUIN `6cd70cd`, iMeanFlow `bf60cd7` |
| GPU | one RTX 5090, 32,607 MiB |

## 3. Arms — the C1 implementation reused verbatim

`B` (historical sampler), `H25`, `H50`, via `flow/interval_exposure.sample_tr_c1`. Mass in both
intervention arms: `h = 0` 0.50 · forced h 0.25 · original positive-h 0.25. **No other sampler
modification.** Monte-Carlo parity is re-run before training and must reproduce
P(h≥0.5) ≈ 0.042 (B) / 0.021 (H25) / 0.271 (H50).

## 4. Seeds

Training seeds **40, 41, 42, 43, 44**. The A4 subject split stays frozen; the seed changes model
initialisation, dataloader order, training Gaussian noise and (t, r) sampling.

Within each seed, B/H25/H50 must share every RNG stream except the intended (t, r, h) intervention.
**Asserted per seed before training:** identical initial-state hash, identical window-order hash, identical
Gaussian-noise hash, identical validation-bank hash; differing (t, r) hash where the intervention requires.

Different seeds are **not** compared as paired trajectories.

**Disclosed recipe property, not a change:** the trainer derives `gen = seed` and `tr_gen = seed + 1`
(`train_a2.py:129-132`), so seed *k*'s (t, r) stream shares its seed integer with seed *k+1*'s dataloader
stream. The streams serve different purposes and no value is reused, but the five seeds are therefore not
fully independent integer draws. The A4 recipe is not modified to fix this.

## 5. Exact compute matching — and an arithmetic correction made before results

**No early stopping. All 15 runs stop at the same fixed point. Validation is monitoring only and MUST NOT
select the evaluation checkpoint.** The evaluation checkpoint is `checkpoint_last.pt`, i.e. the weights
after the final optimiser step — never `checkpoint_best.pt`.

Budget: **66 rounds**, matching C1-B's realised training length.

**Correction.** 66 rounds is **not** 66 × 220 = 14,520 optimiser steps. `batch_rounds`
(`train_a2.py:38-56`) ends a round after `steps_per_round` steps **or at the end of the epoch, whichever
comes first**. With 293,271 training windows at batch 64 there are 4,583 batches per epoch, so rounds
20, 41 and 62 are truncated to 183 steps:

```
66 rounds  =  63 x 220  +  3 x 183  =  14,409 optimiser steps
```

This structure depends only on the window count, the batch size and `steps_per_round` — **not** on the
sampler arm or the training seed — so **all 15 runs realise exactly 14,409 steps**, and compute matching
holds exactly. The budget is hereby frozen at **14,409 realised optimiser steps per run**, and the run
manifest asserts equality across all 15. This correction is recorded before any C2 result exists.

Early stopping is disabled by setting patience above any reachable value; no other argument changes.

## 6. Training protocol

The A4 recipe otherwise unchanged: architecture, conditioning (`h_only`, `h_scale` 1.0), dataset, the 12
frozen train subjects, normalisation, batch 64 / micro-batch 32, AdamW `lr` 1e-3 `wd` 0.01, MeanFlow loss
with `norm_p` 1.0 / `norm_eps` 0.01, `jvp_mode` forward, `p_mean` −0.4 / `p_std` 1.0 /
`data_proportion` 0.5, `val_every_steps` 220, `val_subsample` 4096, `gen_diag_every` 1,
`n_val_banks` 4 / `bank_seed` 1000. **The learning rate is NOT changed because early stopping was
removed.** No scheduler is added; the A4 recipe has none.

Only two things differ between runs: **the training seed** and **the C1 arm's (t, r) sampler**.

## 7. Data firewall

Train on the 12 frozen A4 train subjects. Development evaluation on the frozen C0/C1 subset:
`an0` 1,024 + `k2s` 1,024 = **2,048 windows**. `kjd`/`ssx` are **never loaded**, never QA'd, never
visualised.

## 8. Evaluation

Every seed × arm at **NFE 2 and NFE 4**, evaluation Gaussian source **seed 0**, the same source bank for
all arms, both NFEs and all training seeds.

**Primary — oracle-free, GT-fixed coordinates**, from the frozen `beat_level_analysis` `raw_*` outputs:
**M1** `|raw_qrs_energy_ratio − 1|` · **M2** `|raw_p2p_ratio − 1|` · **M3** `raw_qrs_rmse` ·
**M4** `raw_rmse` · **M5** `raw_corr` (higher-better) · **M6** `|raw_slope_ratio − 1|`.

**Secondary:** F1, chance floor, F1 excess, beats ratio, mean per-window beats-ratio deviation, precision,
recall, missing, spurious, matched coverage, matched morphology.

**Excluded everywhere:** `oracle_corr`, `oracle_absent`, `oracle_qrs_energy`, and every shift-max metric.

## 9. Within-seed statistics

Per seed, at **NFE 2** and at **NFE 4**: `H25 vs B`, `H50 vs B`, `H50 vs H25`. Subject-stratified paired
bootstrap, 2,000 resamples, `default_rng(20260901)`, equal subject weight, positive = intervention better.
**Every seed is reported separately; inconsistent seeds are not hidden.**

## 10. Across-seed summary

The **training seed is the replication unit**. For every comparison × metric: the 5 seed-level point
effects, mean, median, SD, min, max, and the number of seeds with a positive effect.

**Hierarchical bootstrap:** outer — resample the 5 training seeds with replacement; inner — within each
sampled seed, subject-stratified resample of development windows with equal `an0`/`k2s` weight;
**5,000 replicates, RNG seed 20260902**; report the 95 % interval.

**Stated in the report:** five seeds still provide limited training-run inference; the hierarchical CI is a
replication summary, **not** population-level or clinical inference.

## 11. Primary replication hypotheses

### H1 — generic positive-h reweighting
Replicated **iff all three**: (1) `H25 − B` at NFE 2 is positive on **both** M3 and M4 in **≥ 4 of 5**
seeds; (2) the same for `H50 − B`; (3) the hierarchical 95 % CI is entirely > 0 for at least **three** of
the four comparisons {H25−B M3, H25−B M4, H50−B M3, H50−B M4}.
Otherwise: `GENERIC REWEIGHTING EFFECT NOT ROBUSTLY REPLICATED`.

### H2 — h = 0.5-specific QRS calibration
Replicated **iff all four**: (1) `H50 − H25` at NFE 2 is positive on **both** M1 and M2 in **≥ 4 of 5**
seeds; (2) the hierarchical 95 % CI for `H50 − H25` is entirely > 0 for **both** M1 and M2; (3) `H50 − B`
at NFE 2 clearly improves M1 and M2 in **≥ 4 of 5** seeds; (4) H50 shows no reproducible degradation in
**≥ 2** of {M3, M4, M5, M6, F1 excess, beats deviation}.
Otherwise: `h=0.5-SPECIFIC EFFECT NOT ROBUSTLY REPLICATED`. **This rule may not be loosened after results.**

Operationalisation, fixed here: "positive seed-level effect" = the seed's point difference > 0; "clearly
improves" in H2(3) = that seed's within-seed paired 95 % CI entirely > 0; "reproducible degradation" in
H2(4) = a metric whose seed-level effect is negative in ≥ 4 of 5 seeds.

## 12. NFE-specificity interaction — SECONDARY

Per seed and metric, `E2 = improvement(H50 vs B @ NFE2)`, `E4 = improvement(H50 vs B @ NFE4)`,
`D = E2 − E4`. Seed-level D and the hierarchical bootstrap are reported. **No equivalence language.**
If M1/M2 specificity replicates but D is unresolved, the permitted reading is *"h = 0.5 exposure improves
QRS calibration broadly, not specifically at NFE 2."* If D is clearly > 0, *"the effect is stronger at the
intended NFE-2 deployment point."* **This does not override H1/H2.**

## 13. Compute accounting

Per run: seed, arm, exact optimiser steps, wall time, peak VRAM, samples processed, final train loss,
diagnostic validation values. **Assert `optimiser_step_count` identical across all 15 runs (= 14,409).**
A crash, OOM, differing step count or resume mismatch stops that run; **no silent substitution.**

**Estimated cost, disclosed before launch:** C1-B ran 66 rounds in 12,911 s, so 15 runs ≈ **54 GPU-hours**
(~2.25 days) on the single RTX 5090, plus evaluation.

## 14. Frozen systematic visual atlas — SECONDARY, does not affect any gate

Defined from **metadata only, before any C2 prediction is loaded**: 64 development windows, 8 strata
(`an0` × 4 PPG sites, `k2s` × 4 sites: sternum, head, wrist, ankle), 8 windows per stratum, chosen by
deterministic SHA256 rank with salt **`c2-visual-atlas-v1`**. **Selection may not use model error, F1,
morphology or visual quality.**

Rows per window: PPG input · GT ECG · B@NFE2 · H25@NFE2 · H50@NFE2 · B@NFE4 · H50@NFE4.
Panels: (A) full 8 s waveform, (B) GT-R-centred zooms, (C) high-pass/QRS-band component, (D) first
derivative, (E) local energy envelope. GT R locations are overlaid as reference markers only;
**predictions are never translated to align.** Contact sheets per subject × site. **No cherry-picked
examples.**

## 15. Site-wise secondary analysis

All primary metrics at NFE 2 for B/H25/H50, split by `sternum` / `head` / `wrist` / `ankle`.
**Exploratory; the primary verdict may not be changed by site results.**

## 16. Verdicts

Exactly one: **A** `COMPUTE-MATCHED h=0.5 QRS-SPECIFIC EFFECT REPLICATED` iff H2 passes ·
**B** `ONLY GENERIC POSITIVE-h EFFECT REPLICATED` iff H1 passes and H2 fails ·
**C** `C1 EFFECT NOT ROBUST UNDER COMPUTE MATCHING` iff neither passes.

## 17. Wording

Under A, permitted: *"Across five compute-matched training seeds, direct h = 0.5 exposure reproducibly
improved QRS-energy and peak-to-peak calibration relative to the h = 0.25 specificity control under the
frozen WildPPG development protocol."* **Prohibited: "h = 0.5 mismatch causes few-step failure."**
Under B: *"Redistributing positive-h training exposure improves waveform fidelity, but the h = 0.5-specific
QRS calibration effect from C1 did not replicate."*
Under C: *"The single-seed C1 effect does not survive compute-matched replication."*

Prohibited throughout: any SOTA claim, any statement about `kjd`/`ssx`, any population-level or clinical
inference, any resurrection of the oracle metrics, and any use of matched morphology as primary evidence.

## 18. Deliverables and stop rules

`docs/C2_COMPUTE_MATCHED_MULTISEED_INTERVAL_REPORT.md`; `artifacts/c2_compute_matched_multiseed/` with
`run_manifest.csv`, `rng_control.json`, `sampler_exposure.csv`, `metrics_per_seed.csv`,
`within_seed_bootstrap.csv`, `seed_effects.csv`, `hierarchical_bootstrap.csv`, `nfe_interaction.csv`,
`site_metrics.csv`, `decision.json`, `provenance.json`, `visual_atlas/`; figures
`c2_seed_effects_m1_m2.png`, `c2_seed_effects_rmse.png`, `c2_h50_vs_h25_qrs_specificity.png`,
`c2_nfe_interaction.png`, `c2_site_effects.png`, plus atlas contact sheets.

1. C2 ends at its verdict. **No method is implemented, selected or trained, even under verdict A.**
2. `outputs/a4_imeanflow_wildppg_seed42/` and the C1 output directories are never overwritten.
3. Submodules stay byte-identical; checkpoints and prediction dumps never enter git.
