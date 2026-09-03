# O3 — Event-Schedule Error Tolerance & Frozen R1 Bridge — REPORT

**How accurate must a supplied event schedule be before the O2c joint event–morphology advantage disappears,
and does the already-frozen PPG-only R1 Global-TCN schedule fall inside that region?**

Preregistration `docs/O3_SCHEDULE_ERROR_TOLERANCE_PREREGISTRATION.md`, committed and pushed as **`60d1810`**
before any perturbation was generated, any schedule metric computed, any generator run or any R1 output read.
Implementation `243764e` (+ preflight fix `c25c810`). **NO TRAINING, NO WEIGHT UPDATE, NO NEW PREDICTOR
anywhere in O3**: B, O2c, the O2b integer operator and the R1 Global-TCN were loaded frozen, in `eval()`, with
`requires_grad = False`, and no optimizer was constructed (asserted by test).

**All synthetic arms remain ORACLE DIAGNOSTICS** — every synthetic schedule is a perturbation of the GT ECG R
schedule. The R1 arm is PPG-only at inference, but **the R1 probe was supervised with ECG R labels during its
own training**, and **`O2C-CANON-ORACLE` was itself trained with GT-R-derived canonical coordinates**.

## FINAL O3 VERDICT

**C. TOLERANCE REGION TOO NARROW**

J1 and J2 survive all six gates in all three replicates; **J4 and every more severe jitter level fail**, and the
frozen R1 bridge fails. `J_MAX = 2 samples = 15.625 ms`. The oracle-trained canonical generator needs
substantially more accurate event geometry than the current PPG setting supplies.

## 1. Repository

| item | value |
|---|---|
| start SHA | `ef3be40` (O2c result commit) |
| prereg SHA | **`60d1810`** |
| implementation SHA | **`243764e`** (+ `c25c810`) |
| result SHA | this commit |
| clean? | yes — `git status` empty, `HEAD == origin/main` |
| test loaded? | **no** — `kjd` / `ssx` appear in no O3 source; `test_subjects_loaded: []` |
| C2? | **still deferred** — no `outputs/*c2*` |

Submodule pins `6cd70cde` / `bf60cd7c` unchanged; A4 md5 `31c042d291052fbb6dc15263ad316be2` unchanged.
Full suite at the result commit: **454 passed**, of which 34 are the O3 tests.

## 2. Frozen components

| component | file sha256 | state sha256 |
|---|---|---|
| B | `557c7054…` | `47d7ccb9…` |
| O2c (step 10,046, 4,568,707 params) | `5aab09be…` | `f1cc44b3…` |
| R1 Global-TCN | `bfe76ea6…` | `0986a7af…` |
| O2b operator `o2b_warp.py` / `o2_warp.py` | `cb4d1866…` / `046becfb…` | — |

`W = 10`, `MIN_INT_SPACING = 21`, `CORE_OFFSET_TOL = 1e-6`, `round_half_to_even`, bilinear
`grid_sample(align_corners=True, padding_mode="border")`. **No operator edit.** R1 ran at its frozen operating
point: threshold **0.35**, NMS refractory **32** samples, no phase, offset, site-delay or smoothing correction.

## 3. Frozen-model regression

Reproduced **bit-exactly**, `max |Δ| = 0.000e+00` against the preregistered tolerance 1e-6:

| arm | quantity | frozen | reproduced |
|---|---|---|---|
| B | F1 excess / raw F1 / chance / missing / spurious / beats dev | 0.3175618683 / 0.4367299151 / 0.1191680467 / 0.5661580180 / 0.5153896464 / 0.1066636102 | identical |
| O2c ORACLE | F1 excess | 0.8592510053 | identical |
| O2c ORACLE | T4 / T6 / T7 / T8 nAE | 0.4071540678 / 0.4019096984 / 0.4169922852 / 0.4138793945 | identical |

Cohort: 2,048 windows, 19,834 GT beats, 1,922 ECG-window clusters, NFE 4, source seed 0.

## 4. Runtime

| item | value |
|---|---|
| projected | **0.774 GPU-h** (sweep 0.441 + bootstrap 0.052 + multi-source 0.184 + R1 0.096) against a 2.0 h budget |
| actual (production runs) | stage A **1,016 s** + stage B **115 s** = **0.314 GPU-h** |
| VRAM | 1,763 MiB peak |
| training performed? | **NO** |

The commit order places the preflight (step 14) after the operator-floor curve (step 13), so steps 11–13 were
executed once before the preflight and then re-executed identically inside the production run; that ordering
pass and the preflight itself produced no O3 result. Nothing was subsampled: the full 2,048-window cohort, all
six jitter levels, both MISS and both EXTRA levels, three replicates each, and eight sources were used.

## 5. Synthetic schedule quality (the x-axis)

Equal-subject macro; identical across replicates for MISS/EXTRA because those families do not move a retained beat.

| condition | rep | F1@50 | F1@150 | timing median AE | timing MAE | missing | spurious | beats dev |
|---|---|---|---|---|---|---|---|---|
| ORACLE | 0/1/2 | 1.0000 | 1.0000 | 0.00 | 0.00 | 0.0000 | 0.0000 | 0.0000 |
| JITTER_1 | 0 / 1 / 2 | 1.0000 | 1.0000 | 6.73 / 6.78 / 6.64 | 5.21 / 5.24 / 5.17 | 0.0000 | 0.0000 | 0.0000 |
| JITTER_2 | 0 / 1 / 2 | 1.0000 | 1.0000 | 9.58 / 9.76 / 9.68 | 9.36 / 9.37 / 9.32 | 0.0000 | 0.0000 | 0.0000 |
| JITTER_4 | 0 / 1 / 2 | 1.0000 | 1.0000 | 17.57 / 17.45 / 17.37 | 17.41 / 17.29 / 17.15 | 0.0000 | 0.0000 | 0.0000 |
| JITTER_6 | 0 / 1 / 2 | 1.0000 | 1.0000 | 25.38 / 25.20 / 25.20 | 25.23 / 25.14 / 25.09 | 0.0000 | 0.0000 | 0.0000 |
| JITTER_8 | 0 / 1 / 2 | 0.7666 / 0.7634 / 0.7708 | 1.0000 | 25.59 / 25.26 / 25.36 | 25.28 / 25.16 / 25.30 | 0.2334 / 0.2366 / 0.2292 | same as missing | 0.0000 |
| MISS_1 | 0/1/2 | 0.9445 | 0.9445 | 0.00 | 0.00 | 0.1051 | 0.0000 | 0.1051 |
| MISS_2 | 0/1/2 | 0.8823 | 0.8823 | 0.00 | 0.00 | 0.2102 | 0.0000 | 0.2102 |
| EXTRA_1 | 0/1/2 | 0.9501 | 0.9501 | 0.00 | 0.00 | 0.0000 | 0.1051 | 0.1051 |
| EXTRA_2 | 0/1/2 | 0.9050 | 0.9050 | 0.00 | 0.00 | 0.0000 | 0.2102 | 0.2102 |

**Read the timing axis carefully.** Matched-timing MAE is conditional on a ±50 ms match, so it saturates: at J8
the true mean displacement is 8·9/17 = 4.24 samples = 33.1 ms, but 23 % of beats fall outside the ±50 ms window
and are excluded from the matched statistic, leaving a matched MAE (25.3 ms) barely above J6's. J8 is therefore
the only jitter level whose schedule F1@50 degrades (0.767); its F1@150 is still 1.0000. MISS and EXTRA move no
retained beat, so they are **pure count-error** conditions with timing MAE exactly 0.

## 6. Operator-floor curve — the coordinate itself is almost never the problem

Medians over the 2,048 windows, rep 0 (the other replicates agree to the printed precision):

| condition | raw RMSE | T4 | T6 | T7 | T8 | round-trip F1@50 | interpretation |
|---|---|---|---|---|---|---|---|
| ORACLE | 0.00170 | 0.00000 | 0.00000 | 0.00001 | 0.00000 | 1.0000 | below the 0.020 margin |
| JITTER_1 | 0.00173 | 0.00000 | 0.00000 | 0.00001 | 0.00000 | 1.0000 | below the margin |
| JITTER_2 | 0.00177 | 0.00000 | 0.00000 | 0.00003 | 0.00000 | 1.0000 | below the margin |
| JITTER_4 | 0.00184 | 0.00000 | 0.00000 | 0.00011 | 0.00000 | 1.0000 | below the margin |
| JITTER_6 | 0.00210 | 0.00000 | 0.00000 | 0.00036 | 0.00000 | 1.0000 | below the margin |
| JITTER_8 | 0.00296 | 0.00000 | 0.00000 | 0.00162 | 0.00000 | 1.0000 | below the margin |
| MISS_1 | 0.01382 | 0.00236 | 0.01099 | 0.00084 | 0.00000 | 1.0000 | below the margin |
| **MISS_2** | 0.01674 | 0.04351 | **0.05754** | **0.03840** | 0.00000 | 1.0000 | **OPERATOR-CONFOUNDED** |
| EXTRA_1 | 0.00207 | 0.00000 | 0.00000 | 0.00005 | 0.00000 | 1.0000 | below the margin |
| EXTRA_2 | 0.00223 | 0.00000 | 0.00000 | 0.00006 | 0.00000 | 1.0000 | below the margin |

**J1 early-falsification gate: PASS** (median T6 0.00000, T7 0.00001 in all three replicates), so the accepted
coordinate operator is *not* brittle to one-sample schedule error and the experiment proceeded.

Only **MISS_2** exceeds the 0.020 margin; its morphology interpretation is labelled OPERATOR-CONFOUNDED. Every
other condition — including all jitter levels — has an operator floor at least two orders of magnitude below the
generator effects reported below, so **the degradation measured in §7 is the generator's, not the operator's**.
The floor was never subtracted, never used to correct a metric and never used to adjust a CI.

## 7. Generator tolerance curve

Equal-subject macro. `adh` is the generated-event F1@50 against the **supplied** schedule. G-column lists the
gates that pass; a level survives only when all three replicates pass all six.

| condition | rep | F1 excess | T4 | T6 | T7 | T8 | adherence F1@50 | gates passing | survives (all 3 reps) |
|---|---|---|---|---|---|---|---|---|---|
| ORACLE | 0 | +0.8593 | 0.4072 | 0.4019 | 0.4170 | 0.4139 | 0.9840 | G1 G2 G3 G4 G5 G6 | **yes** |
|  | 1 | +0.8593 | 0.4072 | 0.4019 | 0.4170 | 0.4139 | 0.9840 | G1 G2 G3 G4 G5 G6 |  |
|  | 2 | +0.8593 | 0.4072 | 0.4019 | 0.4170 | 0.4139 | 0.9840 | G1 G2 G3 G4 G5 G6 |  |
| JITTER_1 | 0 | +0.8604 | 0.4089 | 0.4005 | 0.4124 | 0.4091 | 0.9852 | G1 G2 G3 G4 G5 G6 | **yes** |
|  | 1 | +0.8613 | 0.4073 | 0.4039 | 0.4101 | 0.4117 | 0.9861 | G1 G2 G3 G4 G5 G6 |  |
|  | 2 | +0.8600 | 0.4065 | 0.4025 | 0.4144 | 0.4089 | 0.9857 | G1 G2 G3 G4 G5 G6 |  |
| JITTER_2 | 0 | +0.8616 | 0.4114 | 0.4074 | 0.4116 | 0.4091 | 0.9873 | G1 G2 G3 G4 G5 G6 | **yes** |
|  | 1 | +0.8615 | 0.4094 | 0.4032 | 0.4017 | 0.4077 | 0.9865 | G1 G2 G3 G4 G5 G6 |  |
|  | 2 | +0.8607 | 0.4103 | 0.4047 | 0.4057 | 0.4088 | 0.9870 | G1 G2 G3 G4 G5 G6 |  |
| JITTER_4 | 0 | +0.8604 | 0.4257 | 0.4206 | 0.3996 | 0.3997 | 0.9863 | G1 G2 G3 G4 G6 | **no** |
|  | 1 | +0.8602 | 0.4232 | 0.4174 | 0.4039 | 0.3953 | 0.9867 | G1 G2 G3 G4 G6 |  |
|  | 2 | +0.8604 | 0.4205 | 0.4193 | 0.4000 | 0.3986 | 0.9862 | G1 G2 G3 G4 G6 |  |
| JITTER_6 | 0 | +0.8557 | 0.4442 | 0.4406 | 0.3963 | 0.3980 | 0.9853 | G1 G2 G3 G4 G6 | **no** |
|  | 1 | +0.8553 | 0.4439 | 0.4397 | 0.3971 | 0.3925 | 0.9855 | G1 G2 G3 G4 G6 |  |
|  | 2 | +0.8549 | 0.4432 | 0.4407 | 0.4027 | 0.3961 | 0.9857 | G1 G2 G3 G4 G6 |  |
| JITTER_8 | 0 | +0.6384 | 0.4771 | 0.4745 | 0.4101 | 0.3891 | 0.9816 | G1 G2 G3 G4 G6 | **no** |
|  | 1 | +0.6331 | 0.4733 | 0.4722 | 0.4043 | 0.4005 | 0.9822 | G1 G2 G3 G4 G6 |  |
|  | 2 | +0.6404 | 0.4729 | 0.4684 | 0.4048 | 0.3938 | 0.9826 | G1 G2 G3 G4 G6 |  |
| MISS_1 | 0 | +0.7316 | 1.0381 | 1.0777 | 0.6940 | 0.5304 | 0.8917 | G1 G5 | **no** |
|  | 1 | +0.7298 | 1.0519 | 1.0966 | 0.7020 | 0.5391 | 0.8904 | G1 G5 |  |
|  | 2 | +0.7306 | 1.0443 | 1.0887 | 0.6992 | 0.5361 | 0.8892 | G1 G5 |  |
| MISS_2 | 0 | +0.5482 | 1.7555 | 1.8368 | 1.0796 | 1.0151 | 0.7230 | G1 G5 | **no** |
|  | 1 | +0.5472 | 1.7529 | 1.8238 | 1.0752 | 1.0184 | 0.7213 | G1 G5 |  |
|  | 2 | +0.5440 | 1.7613 | 1.8308 | 1.0743 | 1.0204 | 0.7178 | G1 G5 |  |
| EXTRA_1 | 0 | +0.8125 | 0.8012 | 0.7984 | 0.5382 | 0.4592 | 0.9473 | G1 G3 G4 G5 | **no** |
|  | 1 | +0.8149 | 0.8027 | 0.7985 | 0.5370 | 0.4573 | 0.9479 | G1 G3 G4 G5 |  |
|  | 2 | +0.8157 | 0.7985 | 0.7963 | 0.5384 | 0.4558 | 0.9494 | G1 G3 G4 G5 |  |
| EXTRA_2 | 0 | +0.7335 | 1.3090 | 1.2922 | 0.7942 | 0.7120 | 0.8988 | G1 G5 | **no** |
|  | 1 | +0.7369 | 1.3138 | 1.2973 | 0.8025 | 0.7075 | 0.8996 | G1 G5 |  |
|  | 2 | +0.7343 | 1.3138 | 1.2973 | 0.8002 | 0.7206 | 0.8996 | G1 G5 |  |

(B reference: F1 excess 0.3176, T4 0.7467, T6 0.7513, T7 0.5856, T8 0.6504.)

### The binding constraint is G5, and G1 never binds

**G1 (event benefit) passes in every single condition tested**, including MISS_2 (+0.5482) and JITTER_8
(+0.6384). Event fidelity under this oracle-trained generator is remarkably robust to schedule error. What
fails is morphology:

- **Jitter fails only G5.** At J4 the GT-anchored QRS-core metrics cross from better than B to clearly worse —
  derivative RMSE 0.3563 vs B 0.3220 (effect −0.0343 [−0.0368, −0.0316], worsens) and curvature error 0.2395 vs
  0.2147 (−0.0248 [−0.0265, −0.0231]) — while T4/T6/T7/T8 all still improve and F1 excess stays at +0.86. At J2
  those same metrics are still better than B (0.2768 and 0.2005).
- **Count errors fail the morphology gates directly.** One missed beat drives T6 to 1.0777 against B's 0.7513
  (effect −0.3265 [−0.3556, −0.2964], worsens) and T4 to 1.0381 (−0.2914); one extra beat drives T6 to 0.7984
  (−0.0472 [−0.0688, −0.0248]) and T4 to 0.8012 (−0.0545). MISS_1 and EXTRA_1 have negligible operator floors,
  so these are **real generator effects, not operator confounds**.

## 8. Jitter tolerance

**`J_MAX = 2 samples = 15.625 ms`.** No threshold is interpolated between tested levels.

Schedule quality at `J_MAX` (mean over the three replicates): F1@50 **1.0000**, F1@150 **1.0000**, matched
timing median AE **9.67 ms**, matched timing MAE **9.35 ms**, missing **0**, spurious **0**, beat-count
deviation **0**.

That is the requirement: **sub-16 ms per-beat timing error with an exactly correct beat count.** A schedule
with perfect counts and 17 ms MAE (J4) already fails.

## 9. Beat-count errors

| condition | G1–G6 |
|---|---|
| MISS1 | **FAIL** (G2, G3, G4, G6) |
| MISS2 | **FAIL** (G2, G3, G4, G6) |
| EXTRA1 | **FAIL** (G2, G6) |
| EXTRA2 | **FAIL** (G2, G3, G4, G6) |

**One missed beat destroys the joint benefit. One extra beat destroys the joint benefit.** Both do so through
morphology, not through event fidelity — every one of these conditions still passes G1 with a large margin.
MISS is worse than EXTRA at the same count error (T6 1.0777 vs 0.7984), and MISS_2's morphology reading is
additionally **OPERATOR-CONFOUNDED**. Per the preregistration, MISS and EXTRA characterise the failure mode and
**do not select the global verdict**.

## 10. Retained oracle gain (descriptive, unclipped)

Mean over the three replicates; > 1 and < 0 values are reported as-is.

| condition | event | T4 | T6 | T7 | T8 |
|---|---|---|---|---|---|
| ORACLE | +1.0000 | +1.0000 | +1.0000 | +1.0000 | +1.0000 |
| JITTER_1 | +1.0025 | +0.9988 | +0.9988 | +1.0279 | +1.0168 |
| JITTER_2 | +1.0037 | +0.9905 | +0.9908 | +1.0631 | +1.0227 |
| JITTER_4 | +1.0020 | +0.9530 | +0.9508 | +1.0940 | +1.0678 |
| JITTER_6 | +0.9928 | +0.8921 | +0.8900 | +1.1084 | +1.0775 |
| JITTER_8 | +0.5903 | +0.8018 | +0.8002 | +1.0628 | +1.0821 |
| MISS_1 | +0.7626 | −0.8779 | −0.9630 | −0.6687 | +0.4871 |
| MISS_2 | +0.4226 | −2.9742 | −3.0893 | −2.9101 | −1.5542 |
| EXTRA_1 | +0.9171 | −0.1593 | −0.1331 | +0.2834 | +0.8158 |
| EXTRA_2 | +0.7704 | −1.6655 | −1.5581 | −1.2653 | −0.2662 |

These are **normalized performance-retention ratios**, not information-retention fractions. The self-referenced
O1 targets retain most of the oracle gain even at J8; the GT-anchored metrics (§7) do not. That divergence is
the shape-versus-placement distinction, quantified next.

## 11. Shape-only diagnostic (secondary; cannot enter G1–G6)

Per-beat O1 primitives compared at each waveform's own event centre, over retained original beats only.

| condition | matched coverage | primitive usable | T4 | T6 | T7 | T8 |
|---|---|---|---|---|---|---|
| ORACLE | 0.9833 | 0.9950 | 0.4984 | 0.5221 | 0.5249 | 0.6806 |
| JITTER_2 | 0.9863 | 0.9953 | 0.5030 | 0.5270 | 0.5151 | 0.6840 |
| JITTER_4 | 0.9855 | 0.9950 | 0.5160 | 0.5380 | 0.5099 | 0.6818 |
| JITTER_8 | 0.9798 | 0.9950 | 0.5698 | 0.5896 | 0.5207 | 0.6950 |
| MISS_1 | 0.9149 | 0.9967 | 1.0224 | 1.0357 | 0.7298 | 0.8770 |
| MISS_2 | 0.7323 | 0.9976 | 1.6008 | 1.5869 | 0.9742 | 1.2432 |
| EXTRA_1 | 0.9672 | 0.9954 | 0.8517 | 0.8465 | 0.6361 | 0.7879 |
| EXTRA_2 | 0.9071 | 0.9959 | 1.2872 | 1.2549 | 0.8246 | 1.0776 |

**This is the mechanistic result of O3.** Under jitter the beat *shape* is almost untouched — T6 moves 0.5221 →
0.5270 → 0.5380 → 0.5896 from ORACLE to J8, while the GT-anchored metric of §7 crosses B at J4. So the jitter
failure is a **placement** failure: the generator draws a good QRS in the wrong place. Under count errors the
shape itself degrades badly (T6 1.0357 at MISS_1, 1.5869 at MISS_2), so a wrong beat *count* corrupts what the
generator draws, not only where it draws it.

Coverage is the ±50 ms match rate of retained beats; primitive usability (the QRS core fitting inside the
window, the frozen O1 validity rule) is reported separately and stays ≥ 0.995 everywhere. This metric penalises
neither missing generated beats nor inserted schedule beats except through coverage, so it is not a waveform
fidelity metric and it entered no gate.

## 12. Schedule adherence — the generator follows a wrong geometry faithfully

| condition | adherence F1@50 | F1@100 | missing vs S | spurious vs S |
|---|---|---|---|---|
| ORACLE | 0.9840 | 0.9886 | 0.0156 | 0.0154 |
| JITTER_2 | 0.9869 | 0.9899 | 0.0131 | 0.0123 |
| JITTER_4 | 0.9864 | 0.9892 | 0.0137 | 0.0127 |
| JITTER_8 | 0.9821 | 0.9854 | 0.0188 | 0.0156 |
| MISS_1 | 0.8905 | 0.9031 | 0.0882 | 0.1315 |
| MISS_2 | 0.7207 | 0.7500 | 0.2768 | 0.2573 |
| EXTRA_1 | 0.9482 | 0.9517 | 0.0702 | 0.0271 |
| EXTRA_2 | 0.8993 | 0.9098 | 0.1359 | 0.0455 |

Adherence to a jittered schedule is as high as adherence to the true one (0.982–0.987 vs 0.984): O2c does not
"stop following" a displaced geometry, it follows it. Under MISS the generator partially *resists* the supplied
schedule — beats spurious relative to `S` amount to 13.2 % of |`S`|, i.e. it re-emits beats the schedule omitted, which
is consistent with the PPG conditioning still carrying beat evidence. **No causal attribution is claimed.**

## 13. Multi-source (frozen Q1 512-window subcohort, source seeds 0…7)

| arm | beat-count SD | pairwise F1@50 | pairwise F1@150 | waveform SD | pairwise waveform RMSE | adherence across sources |
|---|---|---|---|---|---|---|
| B | 1.2309 | 0.3859 | 0.6560 | 0.2981 | 0.4391 | — |
| O2c ORACLE | 0.2118 | 0.9748 | 0.9837 | 0.2256 | 0.3003 | 0.9818 |
| JITTER_4 | 0.2231 | 0.9802 | 0.9851 | 0.2303 | 0.3042 | 0.9862 |
| JITTER_8 | 0.2969 | 0.9720 | 0.9778 | 0.2403 | 0.3184 | 0.9807 |
| MISS1 | 1.0167 | 0.8167 | 0.8539 | 0.3005 | 0.4200 | 0.8920 |
| EXTRA1 | 0.7333 | 0.9216 | 0.9347 | 0.2723 | 0.3718 | 0.9442 |
| **R1-SCHEDULE** | 0.5185 | 0.9116 | 0.9295 | 0.2435 | 0.3300 | 0.9346 |

Source-driven event stochasticity tracks the *count* accuracy of the supplied schedule, not its timing accuracy:
JITTER_8 (23 % of beats displaced past 50 ms) stays at oracle-level stability, while MISS1 (one beat) nearly
returns to B's instability. Every arm's S1/S2-form contrast against B is favourable with the CI entirely above 0.

## 14. Synthetic curve frozen

**YES.** `artifacts/o3_schedule_tolerance/synthetic_curve_frozen.json`, sha256 `5fb72fc4a85a176e…`, written
before any R1 schedule was extracted. It carries `J_MAX`, the MISS/EXTRA tolerances, the full G1–G6 table, the
schedule-quality table, the per-level descriptive means and sha256 hashes of eleven synthetic artifacts. Stage B
re-verified every one of those hashes before running the R1 probe and refuses to start if the file is absent.

## 15. Frozen R1 schedule quality

Precheck: **OK** — M ∈ [7, 21], mean 10.73, **no window with M < 3**, no hard-invalid schedule, minimum spacing
≥ 21 by construction (the frozen R1 NMS refractory is 32 samples), maximum protected-core fractional coordinate
0. **No GT fallback and no correction of any kind.**

| quantity | value |
|---|---|
| F1@50 | **0.6157** |
| F1@100 | 0.7788 |
| F1@150 | 0.8598 |
| F1@200 | 0.8973 |
| timing median AE | **22.30 ms** |
| timing MAE | **22.80 ms** |
| missing | **0.3684** |
| spurious | **0.4780** |
| beat-count deviation | **0.1183** |

Consistent with the frozen R1 stage's own validation numbers (F1@50 0.620, median matched error 23.4 ms) on the
larger R1 cohort. **No threshold was selected here.**

R1 operator floor: raw RMSE 0.00470, T4 1.83e-05, T6 1.97e-05, **T7 0.00646**, T8 0.0, round-trip F1@50 1.0000
— all below the 0.020 margin, so the R1 arm is **not** labelled schedule/operator coupled and its morphology
behaviour is not attributable to coordinate distortion. Nothing was corrected or subtracted.

## 16. `O2C-R1-SCHEDULE`

| arm | F1 excess | T4 | T6 | T7 | T8 | QRS-deriv RMSE | QRS-curvature err |
|---|---|---|---|---|---|---|---|
| B | 0.3176 | 0.7467 | 0.7513 | 0.5856 | 0.6504 | 0.3220 | 0.2147 |
| O2c ORACLE | 0.8593 | 0.4072 | 0.4019 | 0.4170 | 0.4139 | **0.1329** | **0.1095** |
| **O2C-R1-SCHEDULE** | **0.4806** | 0.6970 | 0.6936 | 0.5220 | 0.5173 | **0.3298** | **0.2196** |

Paired, ECG-window clustered, subject-stratified, 2,000 replicates, `default_rng(20260904)`; positive means the
R1 arm beats B.

| metric | effect [95 % CI] | verdict |
|---|---|---|
| F1 excess | **+0.1630** [+0.1511, +0.1748] | improves |
| T4 nAE | +0.0497 [+0.0227, +0.0768] | improves |
| T6 nAE | +0.0576 [+0.0303, +0.0848] | improves |
| T7 nAE | +0.0636 [+0.0421, +0.0835] | improves |
| T8 nAE | +0.1331 [+0.1029, +0.1617] | improves |
| **QRS-core derivative RMSE** | **−0.0078** [−0.0100, −0.0053] | **worsens** |
| **QRS-core curvature error** | **−0.0049** [−0.0064, −0.0034] | **worsens** |

Secondary event detail: raw F1 0.6045, chance 0.1240, missing 0.3918, spurious 0.4088, beat-count deviation
0.0683, raw RMSE 0.4127, raw correlation 0.1833, HF error 0.0884.

**R1 schedule adherence**: generated-event F1@50 against `S_R1` **0.9357**, F1@100 0.9429, missing 0.0871,
spurious 0.0214. The generator follows the R1 geometry almost as faithfully as it follows the oracle geometry —
**the bottleneck is the schedule, not O2c's ability to follow one.**

## 17. R1 bridge gates

| gate | result |
|---|---|
| G1 event benefit (CI > 0 and ≥ +0.10) | **PASS** (+0.1630) |
| G2 T6 non-inferiority (margin 0.020) | **PASS** (lower bound +0.0303) |
| G3 T7 non-inferiority | **PASS** (+0.0421) |
| G4 T6 or T7 improves | **PASS** (both) |
| **G5 legacy structure not clearly worse** | **FAIL** (both QRS-core metrics worsen) |
| G6 T4 / T8 not clearly worse | **PASS** (both improve) |
| G7a beat-count SD lower than B | **PASS** (+0.7124 [+0.6416, +0.7813]) |
| G7b pairwise event F1@50 higher than B | **PASS** (+0.5258 [+0.5067, +0.5435]) |

**Seven of eight pass; the bridge is not supported because G5 fails.** The R1 arm's failure mode is *exactly*
the J ≥ 4 failure mode, and its matched timing MAE (22.80 ms) sits between J4 (17.3 ms) and J6 (25.2 ms) —
except that R1 also carries 37 % missing and 48 % spurious beats, a count error no synthetic jitter level has.
Its G5 margin (−0.0078) is milder than J4's (−0.0343), which is consistent with its adherence being lower: the
generator partially ignores the worst of the R1 schedule.

## 18. Site-wise R1 bridge (secondary; no site causality claim)

| site | R1 schedule F1@50 | F1@150 | beats dev | timing MAE | B F1ex | O2c F1ex | R1-arm F1ex | B T6 | O2c T6 | R1-arm T6 |
|---|---|---|---|---|---|---|---|---|---|---|
| sternum | 0.7031 | 0.8768 | 0.0928 | 21.00 | 0.3720 | 0.8633 | 0.5736 | 0.7179 | 0.3815 | 0.6186 |
| head | 0.7762 | 0.9300 | 0.0690 | 21.25 | 0.4453 | 0.8629 | 0.6434 | 0.6485 | 0.3932 | 0.5544 |
| wrist | 0.5227 | 0.8073 | 0.1846 | 24.37 | 0.2322 | 0.8582 | 0.3781 | 0.8049 | 0.3995 | **0.8895** |
| ankle | 0.4584 | 0.8222 | 0.1321 | 24.88 | 0.2177 | 0.8526 | 0.3241 | 0.8350 | 0.4332 | 0.7269 |

R1 schedule quality varies strongly by site (head 0.776 down to ankle 0.458) and the bridge tracks it: the
oracle arm is flat across sites (F1 excess 0.853–0.863) while the R1 arm spans 0.324–0.643. At the wrist the R1
arm's T6 (0.8895) is *worse* than B's (0.8049) — the only site where the bridge damages morphology outright.

## 19. Tolerance overlap

Placed on the synthetic axes without any composite signal-quality index and without claiming that the synthetic
corruption model matches the R1 error distribution:

| axis | J_MAX (J2) | J4 (first failing level) | frozen R1 |
|---|---|---|---|
| schedule F1@50 | 1.0000 | 1.0000 | **0.6157** |
| schedule F1@150 | 1.0000 | 1.0000 | **0.8598** |
| matched timing MAE | 9.35 ms | 17.3 ms | **22.80 ms** |
| beat-count deviation | 0.0000 | 0.0000 | **0.1183** |

The frozen R1 schedule is **outside the surviving region on every one of the four axes**, and outside it on
axes that no single synthetic family probes jointly: it is simultaneously worse than J6 in timing and worse than
MISS_1/EXTRA_1 in count error. This is why the answer is "outside", not "near".

## 20. Interpretation — eight things kept separate

1. **Supplied schedule quality.** Jitter families keep perfect counts; MISS/EXTRA keep perfect timing; R1 is bad
   on both (F1@50 0.616, MAE 22.8 ms, 37 % missing, 48 % spurious).
2. **Coordinate-operator distortion.** Negligible everywhere except MISS_2. The J1 gate passed, so the operator
   is not brittle to one-sample error, and the R1 floor is far below the margin. The floor was never subtracted.
3. **Generator adherence to the supplied schedule.** Very high throughout (0.98 under jitter, 0.94 for R1). O2c
   follows whatever geometry it is handed; it does not "give up" on a wrong one.
4. **Paired ECG event fidelity.** Robust: G1 passes in **every** condition tested, including MISS_2 and J8, and
   for R1 (+0.1630).
5. **Morphology fidelity.** This is what breaks. Jitter breaks it by *placement* (GT-anchored metrics cross B at
   J4 while the shape-only diagnostic barely moves). Count errors break the *shape* itself.
6. **Source stochasticity.** Tracks count accuracy, not timing accuracy; the R1 arm still passes both G7 legs.
7. **R1 schedule quality.** Reproduces the frozen R1 stage, varies strongly by site, and lies outside the
   tolerance region on all four axes.
8. **Deployability.** **None.** O2c was trained with GT-R-derived coordinates, R1 was trained with ECG R labels,
   and this is a development cohort of two subjects. Nothing here is deployable.

## 21. What this does NOT prove

- Development cohort only; **two validation subjects**; **no fresh test**; no multi-seed confirmation.
- **No training anywhere in O3** — and equally, no evidence about what a *retrained* generator could tolerate.
  O2c was trained on oracle coordinates; a generator trained on noisy coordinates was not tested and is out of
  scope.
- **O2c was trained using GT-R-derived coordinates** and **R1 used GT-R supervision during training**.
- The synthetic corruption model is not claimed to match the R1 error distribution; the four axes were never
  collapsed into one index.
- No uncertainty calibration, no clinical claim, no information-theoretic result, no general causal proof.
- Not: "exact R timing is observable from PPG" · "phase is solved" · "deployability established".
- The shape-only diagnostic was computed for the synthetic families only, where beat identity is defined; there
  is no equivalent decomposition for the R1 arm.
- G5 is a GT-anchored metric and therefore mixes shape with placement by construction — §11 is what separates
  them, and it is secondary by preregistration.

## 22. Recommended next experiment (recommendation only — nothing implemented)

The tolerance region is **narrow**: it requires sub-16 ms per-beat timing **and** an exactly correct beat count,
while the current frozen PPG-only schedule delivers 22.8 ms with 37 % missing and 48 % spurious beats. Per the
preregistered reading of verdict C:

**Do not invest in a large PPG phase/schedule network.** The gap is not a small engineering margin — a
perfect-count schedule with 17 ms error (J4) already fails, and the R1 error is worse on every axis at once.

What the evidence actually points at, as a question rather than a build:

1. **The failure is placement-versus-shape, and it is measurable.** §11 shows jitter leaves shape intact. A
   coordinate whose protected core is anchored on a *predicted* beat inherits the prediction's placement error
   directly, because O2c faithfully follows what it is given (§12). Any future factorization attempt should be
   evaluated with a placement-tolerant morphology criterion agreed **in advance**, not with a GT-anchored one
   that a coordinate method cannot pass unless its schedule is nearly exact.
2. **Count accuracy, not timing accuracy, is what destroys shape and source stability.** One missed beat costs
   more than 62 ms of jitter. If anything is to be improved first, it is beat-count correctness, and that is a
   detection problem, not a phase-regression problem.

Neither is implemented. C2 remains deferred. No architecture, loss, optimizer, seed, split or threshold was
changed anywhere in O3.

## Artifacts

`artifacts/o3_schedule_tolerance/`: `provenance.json`, `provenance_stage_b.json`,
`frozen_component_manifest.json`, `cohort_manifest.csv`, `runtime_preflight.json`,
`frozen_model_regression.json`, `perturbation_manifest.csv`, `schedule_precheck.csv`,
`schedule_quality_metrics.csv`, `operator_floor_metrics.csv`, `synthetic_generator_metrics.csv`,
`synthetic_paired_bootstrap.csv`, `schedule_adherence.csv`, `retained_gain.csv`, `shape_only_diagnostic.csv`,
`synthetic_curve_frozen.json`, `multisource_metrics.csv`, `multisource_bootstrap.csv`,
`r1_schedule_manifest.csv`, `r1_schedule_quality.csv`, `r1_operator_floor.csv`, `r1_generator_metrics.csv`,
`r1_paired_bootstrap.csv`, `r1_site_metrics.csv`, `joint_benefit_gates.csv`, `tolerance_summary.json`,
`decision.json`, `figures/` (FIG 1–7). **No new checkpoint, no training log and no model output directory was
created.** Artifacts, predictions and raw data are not committed.

Code: `src/ppg2ecg/evaluation/o3_schedule.py`, `scripts/o3_common.py`, `scripts/o3_synthetic.py`,
`scripts/o3_preflight.py`, `scripts/o3_r1_bridge.py`, `scripts/o3_figures.py`,
`tests/test_o3_schedule_tolerance.py` (34 tests).
