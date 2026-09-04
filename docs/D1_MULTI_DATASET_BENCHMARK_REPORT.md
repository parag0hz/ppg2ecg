# D1 — Multi-Dataset Benchmark of the Frozen Methodology: REPORT

Preregistration: `docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md` (commit `9fcfd47`, pushed
before any weight update). Implementation: commit `f8cd747`. Run: 2026-09-04 18:31 → 2026-09-05
05:10 KST, one detached tmux session, 10 h 39 min, **five corpora trained and evaluated, zero
failed stages**. Machine-readable outputs: `outputs/d1_bench/{RESULTS.md,results_table.csv,STATUS.json}`
and `outputs/d1_bench/figures/` (six figures, PNG + PDF). None of those are in git by policy.

## 1. What ran

| corpus | train subj / windows | test subj / windows | epochs (best) | train time |
|---|---|---|---|---|
| PPG-DaLiA | 11 / 11,825 | 2 / 2,189 | 48 (best@27) | 1.95 h |
| BIDMC | 36 / 2,160 | 8 / 480 | 31 (best@10) | 0.24 h |
| CapnoBase | 30 / 1,800 | 6 / 360 | 31 (best@10) | 0.21 h |
| WildPPG | 10 / 250,165 | 2 / 44,736 | 37 (best@16) | 2.05 h |
| VitalDB | 199 / 270,998 | 43 / 62,474 | 80 (best@59) | 4.41 h |

Every run used the identical frozen recipe (4,568,707 params, iMeanFlow, seed 42, AdamW 1e-3,
patience 20). All five early-stopped. Small corpora evaluated **every** test window; WildPPG and
VitalDB were capped at 1,024 windows/subject (2,048 and 39,563 windows scored respectively) and the
cap is disclosed in TABLE 3 of `RESULTS.md`.

## 2. Headline result

Subject-macro mean [95 % subject-clustered bootstrap CI], NFE 1:

| corpus | R-peak F1@50 ms ↑ | HR error ↓ (bpm) | PCC ↑ | RMSE ↓ |
|---|---|---|---|---|
| VitalDB | **0.752** [0.711, 0.788] | 8.02 [6.47, 9.82] | 0.299 [0.266, 0.329] | 0.478 |
| CapnoBase | 0.499 [0.261, 0.738] | 17.51 [7.67, 28.89] | 0.058 [−0.053, 0.182] | 0.472 |
| BIDMC | 0.494 [0.270, 0.703] | **1.97** [1.29, 2.75] | 0.050 [−0.012, 0.110] | 0.560 |
| WildPPG | 0.363 [0.273, 0.454] | 16.15 [11.04, 21.26] | 0.049 [0.017, 0.081] | 0.445 |
| PPG-DaLiA | 0.150 [0.147, 0.153] | 12.23 [10.42, 14.05] | 0.002 [0.002, 0.002] | 0.489 |

## 3. The four findings that matter

**3.1 Inference budget buys nothing.** Across all five corpora, going from NFE 1 to NFE 50 moves
R-peak F1 by at most 0.045, and moves it in the *wrong* direction on WildPPG (0.363 → 0.318). RMSE
and PCC are equally flat; only CapnoBase improves RMSE materially (0.472 → 0.421). For a one-step
method this is the expected shape — but it also means **our earlier NFE-4-vs-PENGUIN-NFE-50 caveat
was worth less than assumed**: the budget gap was never the explanation for anything. The real gap
is elsewhere.

**3.2 Sample-level correlation is near zero everywhere except VitalDB.** PCC is 0.002 (DaLiA), ~0.05
(WildPPG, BIDMC), 0.058 (CapnoBase), 0.299 (VitalDB). This was verified independently of the metric
suite by recomputing PCC directly from the saved waveforms (VitalDB +0.269, DaLiA +0.009, BIDMC
+0.057 — the suite is right). FIG 1 shows why: the generator emits ECG-*shaped* content with sharp
QRS-like deflections, but the deflections are not where the reference R peaks are. It is producing
plausible ECG, not *this* ECG.

**3.3 The HR / beat-placement dissociation reproduces on a new corpus.** BIDMC reaches **1.97 bpm HR
error** — better than any published number this project has compared against — while its R-peak
F1@50 ms is only 0.494 and its PCC is 0.050. A rate metric averaged over 8 s is satisfied by getting
the *number* of beats roughly right; it is blind to putting them in the wrong places. This is the
same failure E1–E3 characterised on WildPPG, now shown to be a property of the method rather than of
one dataset. **Any HR-only comparison against the literature is uninformative about this method.**

**3.4 Tolerance dominates the F1 number.** Loosening the matching tolerance from 25 ms to 100 ms moves
F1 from 0.083 → 0.291 (DaLiA), 0.252 → 0.796 (BIDMC), 0.533 → 0.842 (VitalDB). Roughly half of what
looks like "detected beats" at 50 ms is only detected because 50 ms is 6.4 samples wide. Reporting a
single tolerance would have overstated the result; the three-tolerance sweep is the honest form.

## 4. Ordering across corpora, and why it must not be read as difficulty

VitalDB ≫ CapnoBase ≈ BIDMC > WildPPG ≫ DaLiA on beat placement. The ordering is coherent with what
the signals are — VitalDB is anaesthetised surgical patients with a clean fingertip PPG and a stable
rhythm; DaLiA is a wrist-worn Empatica during stairs, cycling and driving — but **D1 cannot separate
that from corpus size**, which spans 1,800 → 270,998 training windows, a 150× range, and correlates
with the same ordering (FIG 6). The preregistration forbids a difficulty claim for exactly this
reason and the report repeats the prohibition.

## 5. Limitations that must travel with these numbers

- **Two-subject test splits.** DaLiA and WildPPG have 2 test subjects each, so their subject-clustered
  bootstrap resamples 2 clusters. DaLiA's PCC CI is `[0.002, 0.002]` — degenerate, not precise. Any
  DaLiA or WildPPG interval in this report is a description of two people, not a population estimate.
- **BIDMC and CapnoBase overfit almost immediately** (best epoch 10 of 31, on 2,160 and 1,800 training
  windows). Their numbers describe a model that had ~10 epochs of signal. Hyperparameters were *not*
  adjusted in response — that is the preregistered behaviour, and the `small_corpus` flag records it.
- **FID against ECGFounder was not computed** (no checkpoint). The `fid_default_features` column is a
  surrogate feature space and is not comparable to PPGFlowECG's published FID. Left blank in TABLE 4
  rather than substituted.
- The evaluation cap on WildPPG (2,048 of 44,736) and VitalDB (39,563 of 62,474) is disclosed per
  corpus; the small corpora are uncapped.

## 6. Claim boundary

Reproduced verbatim from preregistration §11: D1 establishes only the performance of one fixed
methodology, trained separately under one fixed recipe, on five corpora, at stated inference budgets.
It does **not** establish superiority or inferiority against any published method — preprocessing
(PPG 0.5–4 Hz, 8 s windows), parameter count (4.57 M vs PENGUIN's 62.53 M), split and inference
budget all differ, and `docs/PREPROCESSING_CONVENTIONS_SURVEY.md` §6 documents that no cross-paper
number in this field is comparable without restating the pipeline. It does not rank dataset
difficulty. It says nothing about cross-corpus generalisation — no model was evaluated outside the
corpus it was trained on.

## 7. Deviations from the preregistration

1. **WildPPG corpus configuration, caught before launch.** The first implementation pointed WildPPG at
   the A4 split, whose test subjects are exactly the never-loaded `kjd`/`ssx`. Corrected to prereg §4
   before any training: `corpus_subjects()` removes both from the eligible pool, the firewall now
   rejects them in train, val **and** test, and WildPPG was given `split_d1_wildppg_seed42.json` over
   the remaining 14 subjects (10/2/2) and trained as a new D1 run. No D1 stage ever loaded either
   subject. The frozen C1 arm-B checkpoint is unchanged (sha256 `557c7054…`).
2. **A real-data metric was produced before the preregistration was pushed.** During adversarial
   verification of the build, a verifier agent ran `d1_evaluate.py --corpus wildppg` at 17:23 on
   2026-09-04, against the frozen C1 arm-B checkpoint. This violates the standing rule. The outputs
   (`outputs/d1_wildppg_seed42/`, `outputs/d1_bench/`) were deleted before the preregistration was
   committed, and every number in this report comes from the post-push run. No weight update occurred
   and no integrity anchor changed.
3. **Per-corpus evaluation caps** were not named in the preregistration. They were introduced after
   verification found the original uniform cap silently halved both DaLiA test subjects while being
   reported as complete. Small corpora are now uncapped; the two large ones are capped at
   1,024 windows/subject and the realised counts are printed in the report.

## 8. What this changes about the research direction

The NFE result (§3.1) removes inference budget as an explanation. The PCC result (§3.2) says the
model is not solving the alignment problem at all on four of five corpora. The dissociation (§3.3)
now has cross-corpus support. Together these point at the same conclusion E3 reached from a different
direction: **the binding constraint is where beats go, not how many there are or how long we sample.**

The two preprocessing outliers identified in `docs/PREPROCESSING_CONVENTIONS_SURVEY.md` §5 — the
4 Hz PPG ceiling (every other PPG→ECG work uses 8 Hz) and per-window `filtfilt` after segmentation —
remain untested and are now the cheapest remaining explanations for §3.2. Each needs its own
preregistration. Nothing in that direction has been started.
