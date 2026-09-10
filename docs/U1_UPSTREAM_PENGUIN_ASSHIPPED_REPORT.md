# U1 — Upstream PENGUIN, as shipped: six-dataset replication — REPORT

Preregistration `docs/U1_UPSTREAM_PENGUIN_ASSHIPPED_PREREGISTRATION.md` (`6c7a646`, pushed before any
dataset was preprocessed or trained). Scheduling amendment `521302b`. Upstream pin
`external/PENGUIN` @ `6cd70cd`, verified clean before and after every dataset.

Ran 2026-09-09 09:32 → 2026-09-10 20:51 KST on one RTX 5090.

---

## 1. Headline: does the released code reproduce the published table?

**6 REPRODUCED · 1 PARTIAL · 1 NOT REPRODUCED** (rule frozen in prereg §5: REPRODUCED
`0.80 ≤ r ≤ 1.25`, PARTIAL `0.60–0.80` or `1.25–1.67`, otherwise NOT REPRODUCED; `r = U1 / published`).

| dataset | metric | **U1 (this environment)** | published | unit | r | verdict |
|---|---|---|---|---|---|---|
| PPG-DaLiA | HeartRateError | **16.350** | 15.64 | bpm | 1.045 | **REPRODUCED** |
| WildPPG | HeartRateError | **31.006** | 12.97 | bpm | 2.391 | **NOT REPRODUCED** |
| BIDMC | RespRateError | **3.479** | 2.98 | bpm | 1.168 | **REPRODUCED** |
| WESAD | RespRateError | **4.393** | 4.45 | bpm | 0.987 | **REPRODUCED** |
| UCI-BP | SBPError | **19.986** | 12.61 | mmHg | 1.585 | **PARTIAL** |
| UCI-BP | DBPError | **7.863** | 7.14 | mmHg | 1.101 | **REPRODUCED** |
| MIMIC-BP | SBPError | **14.986** | 17.43 | mmHg | 0.860 | **REPRODUCED** |
| MIMIC-BP | DBPError | **9.212** | 11.34 | mmHg | 0.812 | **REPRODUCED** |

Every value is what upstream's own `train.py` printed for its own test split. Nothing was selected,
re-derived or dropped. Secondary values (upstream MAE / inference time on the test split): PPG-DaLiA
0.3602 / 1.88 s, WildPPG 0.3597 / 2.71 s, BIDMC 0.7718 / 1.84 s, WESAD 0.7528 / 1.85 s, UCI-BP 13.7118 /
1.89 s, MIMIC-BP 12.7393 / 1.89 s (ABP MAE is raw mmHg — upstream leaves ABP un-normalised).

**The two cells that did not reproduce are exactly the two where one of our own declared deviations
could bite.** Both are analysed in §3; neither is presented as a clean refutation of the published number.

## 2. What this settles, and what it does not

Before U1, `docs/PENGUIN_AUDIT.md` §38 recorded the published numbers as unverifiable because the release
ships no PENGUIN checkpoint, log or split. That remains true of the *artifacts*. What U1 adds is that the
released *pipeline*, run at its own shipped defaults, lands on the published values for six of eight cells.

It also corrects a claim of ours. Arm A0 (`docs/A0_PENGUIN_REPRODUCTION_REPORT.md`) reported
**10.99 bpm** on PPG-DaLiA against the paper's 15.64 — better than published. A0 was never a run of
upstream `train.py`: it put upstream's model class and `train_flow`/`optimize` inside this project's
training loop, on this project's split, at `segment_len = 8`. Run as shipped the same code gives
**16.350 bpm**. A0's advantage was a consequence of our protocol changes, not of the released method
underperforming its paper. A0's own report already said `train.py` was not run; the number was
nonetheless carried into later comparisons as if it characterised PENGUIN.

**The agreement is worth less than it looks.** PPG-DaLiA and WESAD are scored on **one test subject**,
drawn by `random.sample` over a `glob` whose order is filesystem-dependent. Our draw was `subject9`;
the paper's is unrecoverable. BIDMC — the only dataset here with a 6-subject test set, and so the one
that should be most stable — is the *worst*-agreeing of the three Tier A cells (+16.8%), while the
single-subject PPG-DaLiA agreed to +4.5%. That ordering is what one expects from a lottery, not from a
protocol being reproduced. No seed, spread or interval accompanies the published table, and none of these
runs establishes one either. A per-seed spread measurement is the obvious follow-up and is **not** part
of U1.

## 3. The two cells that did not reproduce

### 3.1 WildPPG — upstream's early stopping selected a one-epoch model

Upstream ran 11 epochs and stopped itself on `earlystop_patience: 10`. The **epoch cap never bound**
(cap 12, stopped at 11), so prereg deviation D2 is not the cause.

```
val MAE  0.3258  0.4122  0.3472  0.3865  0.3541  0.3936  0.3302  0.3275  0.3560  0.3319  0.3483
         ^ epoch 1 is the minimum; never beaten in the following 10 epochs -> patience fires
Saving checkpoint at epoch 0        <- the only checkpoint write of the entire run
Loaded checkpoint ... (epoch 0)     <- the model that produced 31.006 bpm
```

Training MAE fell monotonically (0.1227 → 0.1106) throughout. Validation MAE oscillated ±25 % around a
flat floor and never dipped below what epoch 1 happened to hit, so upstream's own selection rule kept the
one-epoch weights and discarded ten further epochs of training. No other dataset behaved this way:

| dataset | epochs | best epoch | checkpoint writes |
|---|---|---|---|
| BIDMC | 17 | 7 | 2 |
| WESAD | 20 | 10 | 4 |
| PPG-DaLiA | 36 | 26 | 6 |
| MIMIC-BP | 20 | 18 | 6 |
| UCI-BP | 12 | 10 | 4 |
| **WildPPG** | **11** | **1** | **1** |

**Our confound, stated plainly.** Deviation D1 (the `kjd`/`ssx` firewall) took WildPPG from 16 subjects to
14, which takes `val_size = subject_num // 8` from **2 subjects to 1**. A one-subject validation set is
exactly what makes the epoch-to-epoch val MAE noisy enough for a lucky first epoch to hold off ten
successors. The published configuration would have validated on two subjects. We cannot say whether that
alone would have changed the outcome.

Re-running WildPPG at `fold_num = 7` would restore `val_size = 2` on 14 subjects — but choosing that
*after* seeing this failure is result-dependent tuning and was not done. If it is to be run it needs its
own preregistration, fixed in advance, and must be reported as a separate result.

The verdict stands as **NOT REPRODUCED with a known confound of ours**. It is not evidence against the
published 12.97.

### 3.2 UCI-BP — SBP misses, DBP matches, and the corpus leaks

The epoch cap **did** bind here: 12 epochs, no early stop, last improvement at epoch 10. Deviation D2 is
therefore a live explanation, and specifically for SBP: D3 already established that SBP is the window
**maximum** and so is maximally sensitive to prediction noise overshooting the systolic peak, while DBP is
the **minimum** and is not. An undertrained, noisier model degrades SBP first. That is the observed
pattern — SBP r = 1.59, DBP r = 1.10 — so **UCI-BP SBP PARTIAL is reported as confounded by our cap**.

Independently of any metric, **U1-P1 is confirmed from file hashes**:

```
duplicate groups (byte-identical ABP targets and window counts)
  subject0 == subject2   (91,994 windows, y-sha a8d2ad6d…)
  subject1 == subject3   (73,554 windows, y-sha 8981db6d…)
  subject4 == subject6   (97,122 windows, y-sha d5da82fb…)
  subject5 == subject7   (93,570 windows, y-sha 5f310c4e…)

realised split: val = subject7, test = subject2, train = {0,1,3,4,5,6}
TEST SUBJECT DUPLICATED INTO TRAIN: True  (subject2's twin subject0 is in train)
```

`load_data.py:100-102` picks the file as `part = sub_idx // 4 + 1` but the record range as
`onset = sub_idx % 2 * 1500`, where `% 4` would be needed to tile 8 distinct chunks. With
`subject_num: 8` this yields parts `1,1,1,1,2,2,2,2` against onsets `0,1500,0,1500,…`, so the eight
"subjects" are four chunks used twice, and `Part_3.mat` / `Part_4.mat` are never opened. Under upstream's
random 6/1/1 split the test subject's twin lands in train with probability 6/7; in our draw it did.
**UCI-BP's test set is byte-identical to data the model trained on.**

This is a defect in the released loader, established from hashes and the code, and does not depend on
U1's metrics. What it does *not* establish is that the published 12.61 / 7.14 came from a leaked split:
that is likely (6/7 under their own sampler) but unproven, and our own leaked run produced SBP 19.99,
which is *worse* than published. Leakage alone does not account for the published SBP.

## 4. Pre-registered predictions

| | outcome |
|---|---|
| **U1-P1** UCI-BP subjects duplicate each other | **CONFIRMED** — 4 duplicate groups, test's twin in train (§3.2) |
| **U1-P2** `HeartRateError` returns `0.0` when no window has a valid HR pair | present in the executed code path (`help_func.py:184`); not repairable without modifying upstream, and its incidence is not observable from upstream's output. Noted, not quantified. |
| **U1-P3** shipped `sample_num` is an 8 s count while `segment_len` is 4 | **CONFIRMED on all six** |

U1-P3, realised counts at the shipped 4 s against the config's `sample_num`:

| dataset | realised @4 s | config `sample_num` | ratio |
|---|---|---|---|
| PPG-DaLiA | 32,368 | 16,181 | 2.0004 |
| BIDMC | 6,360 | 3,180 | 2.0000 |
| WESAD | 21,711 | 10,851 | 2.0008 |
| UCI-BP | 712,480 | 356,240 | 2.0000 |
| WildPPG | 686,640 (14 subj; 784,731 scaled to 16) | 390,216 | 2.0110 |
| MIMIC-BP | 320,040 | 137,160 | 2.3333 |

MIMIC-BP's 2.333 is not an exception but the sharpest confirmation. Its raw arrays are 30 records ×
3750 samples per subject, and upstream windows them with a non-overlapping stride, so a record yields
`floor` counts that do not simply halve:

```
8 s: 3 windows/record x 30 records x 1524 subjects = 137,160  == config sample_num  (exact)
4 s: 7 windows/record x 30 records x 1524 subjects = 320,040  == U1 realised        (exact)
```

The config's bookkeeping is the 8 s count to the window, while `segment_len` has been 4 since the initial
commit. `PENGUIN_AUDIT.md` §273 left this undecidable from the repo; the realised counts decide it for the
bookkeeping, though still not for which length produced the paper's table.

## 5. Realised protocol

Upstream's split (`fix_seed(42)` then `random.sample` over an unsorted `glob`) reproduced and recorded per
dataset; the stdlib RNG is untouched between those two points, so `scripts/u1_split_manifest.py` recovers
it exactly without modifying upstream.

| dataset | tier | val / test / train (subjects) | test subject(s) | windows train/val/test | epochs | best | stopped by |
|---|---|---|---|---|---|---|---|
| BIDMC | A | 6 / 6 / 41 | 6 subjects | 4,920 / 720 / 720 | 17 | 7 | patience |
| WESAD | A | 1 / 1 / 13 | `subject9` | 18,719 / 1,387 / 1,605 | 20 | 10 | patience |
| PPG-DaLiA | A | 1 / 1 / 13 | `subject9` (S10) | 27,419 / 2,287 / 2,662 | 36 | 26 | patience |
| MIMIC-BP | B | 190 / 190 / 1,144 | 190 subjects | 240,240 / 39,900 / 39,900 | 20 | 18 | **cap 20** |
| WildPPG | B | 1 / 1 / 12 | `qm9` | 591,568 / 47,680 / 47,392 | 11 | 1 | patience |
| UCI-BP | B | 1 / 1 / 6 | `subject2` | 526,916 / 93,570 / 91,994 | 12 | 10 | **cap 12** |

Epochs are upstream's own 1-based numbering. Tier A ran the shipped 300-epoch schedule uncapped and every
Tier A dataset stopped itself on patience.

## 6. Deviations, as declared and as realised

| # | deviation | did it bind? |
|---|---|---|
| D1 | WildPPG 14 of 16 subjects (`kjd`/`ssx` firewall) ⇒ `val = test = 1` instead of 2 | **yes** — the plausible cause of §3.1 |
| D2 | Tier-B `epoch_num` capped 20 / 12 / 12 | MIMIC-BP **bound** (best at 18 of 20); UCI-BP **bound** (best at 10 of 12); WildPPG **did not bind** (patience first) |
| D3 | `preprocess.DaLiA` → `preprocess.PPG-DaLiA` | mechanical; the release cannot run PPG-DaLiA otherwise |
| D4 | *(execution note, added `521302b`)* WildPPG and UCI-BP trained concurrently rather than back to back | scheduling only — separate processes, procdata, log and hydra directories, each seeding itself; no preregistered quantity depends on it |

D4's rationale was wrong and is recorded as such. It was justified on the model being latency-bound with
VRAM headroom (10.5 of 32.6 GiB). Measured: 96.3 min/epoch (WildPPG) and 128.4 min/epoch (UCI-BP) side by
side against solo estimates of 47 and 67 min — the GPU was throughput-bound and concurrency cost ~8 %
overhead on top of time-sharing. `nvidia-smi` utilisation (92–97 % throughout) does not measure saturation
and should not have been used as evidence of headroom. Net wall-clock was roughly a wash because WildPPG
finished first and left UCI-BP running solo.

## 7. Firewalls verified

- `external/PENGUIN` unmodified: `git status --porcelain` empty, HEAD `6cd70cd` before and after.
- WildPPG `kjd` / `ssx`: excluded at the filesystem level (14-subject symlink view, asserted twice) and
  absent from the realised train/val/test lists. Test was `qm9`, val `n31`.
- Frozen A4 checkpoint `outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt` md5
  `31c042d291052fbb6dc15263ad316be2` — unchanged. Arm U sha256 `8beaff6c…` — unchanged.
- No checkpoint, prediction, `.pkl` or raw data from this stage is in git.
- C2 remains deferred and untouched.

## 8. What U1 does not license

U1 is a replication of upstream's pipeline against upstream's published table. It is **not** a
like-for-like comparison against our arms, and `docs/D3_PENGUIN_SIX_DATASET_REPORT.md`'s "PENGUIN @50"
column — which holds published values — is **not** superseded by these numbers. U1 runs 4 s windows on
upstream's own filesystem-order split; our A3/A4/iMF arms run 8 s windows on this project's deterministic
splits. Putting the two side by side would repeat exactly the budget-mismatch error D3 already had to
correct. Establishing a fair joint comparison would require a separate, preregistered stage that fixes
window length and split for both.

## 9. Qualitative figures

`artifacts/u1_upstream/figures/`, produced by `scripts/u1_figures.py` from U1's own checkpoints —
upstream's model, upstream's sampler (Heun `n_step=25`, 50 NFE), upstream's test split, shipped 4 s
windows. Window selection is deterministic and was fixed before any window was viewed: the first six
windows of the test split in upstream's own concatenation order (first file of `file_path["test_path"]`,
stored order). Sampling is seeded with upstream's `fix_seed(42)`.

| file | what it shows |
|---|---|
| `u1_paper_style_qualitative.png/pdf` | the published figure's layout — rows PPG / original vital sign / PENGUIN, columns grouped into ECG, respiration and ABP — on our own results |
| `u1_paper_style_annotated.png/pdf` | the same, with each column captioned by its U1 metric against the published value and the checkpoint epoch that produced it |
| `u1_test_window_variability.png/pdf` | six consecutive test windows per dataset, target black over PENGUIN orange, so a single favourable window cannot stand in for the result |

The published figure also carries a **PaPaGei-S** row. PaPaGei-S was not trained in U1, so that row is
absent rather than filled from another source.

Three things are visible in the variability grid that the scalar table does not show:

1. **ECG beats are generated but not placed.** On PPG-DaLiA and WildPPG the model emits R-peak-shaped
   deflections at roughly the right rate, but they do not land on the target's beats, and the generated
   trace carries a baseline offset relative to the target. This is the rate-vs-placement dissociation D1
   and D2 already measured, now visible in the released model's own output.
2. **Respiration is frequently near-antiphase.** BIDMC windows 0, 2, 3 and 5 and WESAD windows 0 and 2
   show the prediction tracking the correct period while inverted or badly phase-shifted. Upstream's
   `RespRateError` reads the FFT argmax, which is phase-blind — so these windows can score well on the
   published metric while being wrong about when the breath happened.
3. **ABP tracks best of the three tasks, and overshoots systolic peaks.** UCI-BP and MIMIC-BP pulses
   align in shape and timing, with the prediction exceeding the target at the peaks. That is exactly the
   asymmetry behind SBP (window maximum, noise-sensitive) missing while DBP (window minimum) reproduces.

WildPPG's panel is the one-epoch checkpoint described in §3.1 — its visible degradation is the same
selection failure that produced 31.006 bpm, not a separate defect.
