# U3 — Do U2's conclusions survive its two design defects? REPORT

Preregistration `docs/U3_GATE_AND_SELECTION_SENSITIVITY_PREREGISTRATION.md` (`ec215e5`, pushed before any
U3 number existed). **No model was trained in this stage.** Ran 2026-09-12.

---

## 1. Headline: they do not

U2 published **PARTIAL (3 of 5)**. Correct the gate defect *and* read both arms at the same number of
optimizer steps, and only **WildPPG** survives.

| gate | checkpoint | tally | stage verdict |
|---|---|---|---|
| two-sided (U2 §8, as frozen) | each arm's own best | **3 / 5** | PARTIAL |
| one-sided (U3-A) | each arm's own best | **2 / 5** | PARTIAL |
| two-sided | both at 14,000 steps | 1 / 4 † | — |
| **one-sided** | **both at 14,000 steps** | **1 / 5** | **NOT SUPPORTED** |

† under the two-sided gate at fixed steps, BIDMC's primary cells all become uninformative and the dataset
is excluded, so the denominator drops to 4. The one-sided gate keeps it countable, which is why the last
row is the one to read.

**U2's PARTIAL was propped up by both defects.** Neither alone flips the stage label — the gate costs one
dataset (3→2, still PARTIAL) and the checkpoint choice costs another — but together they take the stage
from PARTIAL to **NOT SUPPORTED**.

This does not retract U2. U2's verdict was produced by a rule frozen before the data existed and stands as
published. What U3 establishes is how much of it was an artefact of that rule's defects.

## 2. U3-A — the one-sided gate

Corrected rule (§3, fixed before recomputation): a cell is gated **only when it would otherwise PASS**;
a cell that fails its margin always counts.

Exactly one dataset changes: **MIMIC-BP, NON-INFERIOR at k=1 → NOT non-inferior at any k**. Its DBP cells
miss the +1.0 mmHg margin by 9.6–14.2 mmHg at every k and were suppressed only because arm C's own
NFE 1→50 span there is 0.052. With the gate corrected they count, and SBP alone can no longer carry the
dataset.

No other dataset's verdict moves. Stage tally 3/5 → 2/5; the stage label stays PARTIAL because 2 is the
bottom of that band.

`scripts/u2_verdict.py --one-sided` prints which rule produced any table it emits.

## 3. U3-B — both arms at exactly 14,000 optimizer steps

Every U2 run's `checkpoint_last.pt` was verified to carry `opt_steps == 14000` before it was loaded.
Evaluation is U2's code path unchanged: same NFE grid, same four noise seeds, same shared `e0`, same
splits, same U2-D5 index sets, same metrics and paired cluster bootstrap.

### 3.1 §5 verdict: **PARTIAL ARTEFACT** (2 of 5)

Deficits below are `(I − C@50)` in units of that metric's margin; **positive means arm I is worse**.
The §5 threshold — a shift of half a margin — was fixed in advance.

| corpus | primary cell | deficit at best | deficit at 14k steps | shift | |
|---|---|---|---|---|---|
| PPG-DaLiA | HR k=2 | 3.897 | 1.446 | **+2.452** | shrinks |
| PPG-DaLiA | HR k=4 | 3.088 | **0.824** | **+2.264** | shrinks |
| MIMIC-BP | DBP k=1 | 13.442 | 11.192 | **+2.250** | shrinks |
| WildPPG | HR k=1 | −0.735 | −1.000 | +0.266 | — |
| BIDMC | RR k=1 | 1.586 | 1.555 | +0.031 | — |
| WESAD | RR k=1 | 0.213 | 0.409 | −0.196 | grows |
| MIMIC-BP | SBP k=1 | −0.504 | 6.888 | **−7.392** | grows |
| UCI-BP | SBP k=1 | −6.708 | 4.236 | **−10.944** | grows |
| UCI-BP | DBP k=1 | 6.271 | 23.288 | **−17.016** | grows |

Two of five countable datasets shift by at least half a margin, so §5 returns **PARTIAL ARTEFACT**.

### 3.2 The asymmetry cuts both ways — and on balance it was flattering arm I

The hypothesis U3-B was built to test is that arm I looked worse because its criterion selected an earlier
checkpoint (37 % of the budget against arm C's 80 %). That is **true on PPG-DaLiA and nowhere else**:

- **PPG-DaLiA** — arm I's HR improves from 13.72 to **10.92 bpm** at k=4 when read at 14,000 steps, against
  arm C's 10.10. Its deficit falls from 3.09 margins to 0.82. Here the early selection genuinely cost it.
- **ABP — the opposite, and violently.** UCI-BP SBP goes from arm I *better* (19.76 vs 26.47) to arm I far
  worse (30.71); UCI-BP DBP 15.86 → 32.88; MIMIC-BP SBP 16.60 → 22.00. **Arm I degrades with more training
  on ABP, and its own criterion was correctly stopping it early.** Reading it at fixed steps removes a
  protection it had earned.
- **WESAD** — both cells worsen (RR deficit 0.21 → 0.41 margins, correlation 0.73 → 1.44), and the
  correlation now exceeds its margin. WESAD's U2 pass does not survive.
- **WildPPG, BIDMC** — essentially unmoved (|shift| ≤ 0.27 margins).

So the selection asymmetry is **not** a systematic handicap on arm I. It helped arm I on three corpora and
hurt it on one. Correcting for it produces *fewer* passes, not more.

### 3.3 Best vs fixed-step, all six corpora

| corpus | metric | C@50 best | C@50 last | I@1 best | I@1 last | I@4 best | I@4 last |
|---|---|---|---|---|---|---|---|
| PPG-DaLiA | HR | 10.63 | 10.1 | 15.57 | 13.4 | 13.72 | 10.92 |
| PPG-DaLiA | Rpeak_F1 | 0.14 | 0.1471 | 0.1448 | 0.1381 | 0.1415 | 0.1396 |
| WildPPG | HR | 16.19 | 15.86 | 15.45 | 14.86 | 17.11 | 16.78 |
| WildPPG | Rpeak_F1 | 0.2652 | 0.2749 | 0.2711 | 0.2723 | 0.261 | 0.2721 |
| BIDMC | RR | 3.332 | 2.809 | 4.121 | 3.551 | 4.047 | 3.562 |
| BIDMC | Resp_corr | -0.02326 | 0.00309 | -0.007693 | -0.00109 | -0.007429 | 8.801e-05 |
| WESAD | RR | 5.02 | 4.863 | 5.127 | 5.068 | 5.123 | 5.069 |
| WESAD | Resp_corr | 0.0381 | 0.0759 | 0.001757 | 0.00376 | 0.002177 | 0.004684 |
| UCI-BP | SBP | 26.47 | 26.47 | 19.76 | 30.71 | 26.37 | 36.87 |
| UCI-BP | DBP | 9.593 | 9.593 | 15.86 | 32.88 | 9.135 | 15.99 |
| MIMIC-BP | SBP | 17.11 | 15.1 | 16.6 | 22 | 14.18 | 14.27 |
| MIMIC-BP | DBP | 10.15 | 12.55 | 23.62 | 23.76 | 20.54 | 22.35 |

Arm C also improves at fixed steps on most corpora (PPG-DaLiA HR 10.63 → 10.10, WildPPG 16.19 → 15.86,
BIDMC RR 3.33 → 2.81, WESAD 5.02 → 4.86, MIMIC-BP SBP 17.11 → 15.10), which is the third outcome §4
anticipated: arm C's criterion was mildly conservative too, so the fixed-step reading is not simply
"arm C at its best versus arm I handicapped". MIMIC-BP DBP is the exception — arm C degrades there
(10.15 → 12.55), and it is still half of arm I's 23.76.

## 4. What survives

**WildPPG is the only result that holds under every combination of gate and checkpoint.** Arm I at 1 NFE
matches or beats arm C at 50 NFE on both primary cells in all four cells of the §1 table
(best-checkpoint HR 15.45 vs 16.19, fixed-step 14.86 vs 15.86; R-peak F1 0.2711 vs 0.2652 and
0.2723 vs 0.2749). It is a single corpus, with two test subjects, on one seed.

**Everything else fails under the corrected reading.** PPG-DaLiA comes closest — at fixed steps its HR
deficit is 0.82 of a margin, within the margin on the point estimate but not on the interval the §8 rule
requires.

**U2's §1 claim — 12–50× less sampling compute at equal quality — is not supported.** It holds on one of
six corpora under U2's own rule, and on one of six under the corrected rule.

## 5. A sign error found and fixed during this analysis

`scripts/u3_analyze.py` initially inverted the deficit sign for **score** metrics (R-peak F1, respiration
correlation), reporting arm I as worse where it was better. Caught by cross-checking WildPPG R-peak F1
k=1 against U2's own CSV, where arm I is ahead (0.2711 vs 0.2652) but the first run printed a positive
(worse) deficit. The bootstrap reports *positive = arm I better* for loss and score metrics alike, so the
conversion is `−point/|δ|` in both cases, not sign-dependent.

Every §5 primary cell is a loss metric (HR, RR, SBP, DBP), so **the §5 verdict was unaffected**; only the
co-primary rows of §3.1 were wrong, and they are corrected here. The same class of error is why the
verdict rule was codified in `scripts/u2_verdict.py` rather than applied by hand.

## 6. Limitations

1. **U3-A is a sensitivity analysis, not a verdict.** The one-sided gate was proposed after seeing U2's
   numbers. U2's frozen verdict stands; U3-A bounds the defect's cost.
2. Everything inherited from U2: one training seed; two test subjects on three of five countable datasets;
   UCI-BP excluded with one test subject and a degenerate interval; the U2-D5 evaluation budget.
3. **Neither checkpoint choice is privileged.** "Each arm at its own best" and "both at equal steps" are
   both defensible, and §3.2 shows they disagree in *different directions* per corpus. U3 publishes both
   rather than the flattering one, and the §1 table is the honest summary precisely because it shows all
   four combinations.
4. U3 changes nothing about the ABP mechanism: A8 established scale sensitivity on MIMIC-BP
   (U2 report §10), and whether its fix transfers to 4 s and to UCI-BP remains unpreregistered and unrun.

## 7. What this justifies doing next

Recommendations only; nothing here is implemented.

1. **A successor preregistration should fix the gate as one-sided and name the checkpoint rule in advance.**
   Both defects were avoidable by stating the rule more carefully, not by collecting more data.
2. **The ABP degradation-with-training in §3.2 is a new, unexplained finding.** Arm I gets substantially
   worse between its own best checkpoint and 14,000 steps on both ABP corpora while arm C does not. A8's
   scale sensitivity is the obvious suspect; a preregistered probe that z-scores ABP and reads both
   checkpoints would test it and also settle the 4 s / UCI-BP transfer question in one run.
3. **WildPPG's single survival deserves replication before it is claimed.** Two test subjects, one seed.
   Additional seeds on that corpus alone are cheap relative to what has already been spent.
