# RF1 — Rectified flow (reflow) as a competing few-step method (preregistration)

Frozen and pushed before any RF1 weight update. Single seed (42), VitalDB V1 data and split, 14,000-step budget —
identical to every other VitalDB arm.

**Why reflow is the right comparison.** PENGUIN's training *is* 1-rectified flow: upstream `train_flow` draws
`x_0 ~ N(0, I)` independently of the target, interpolates linearly and regresses `x_1 − x_0`. What the rectified-flow
line of work adds for few-step sampling is **reflow**: re-train on pairs the model itself produced, which straightens
the trajectories. Arm C is therefore the 1-RF model and arm **R** below is its 2-RF successor.

## Arm R — 2-rectified flow
1. **Pairs.** 64,000 training windows (exact `linspace` over the 288,400, fixed before generation), one noise draw
   each from seed 20260919, passed through **arm C (`outputs/v1_vitaldb_armC_seed42/checkpoint_last.pt`) at 50 NFE**
   (Heun 25, its own inference setting). Stored as float16 `(z0, x1_hat)` pairs.
2. **Training.** The same PENGUIN backbone, **initialised from arm C** (standard reflow practice, disclosed),
   trained on those pairs with the unchanged rectified-flow loss — `t ~ U(0,1)`, `z_t = (1−t) z0 + t x1_hat`,
   MSE against `x1_hat − z0`. Optimiser argv identical to arm C (AdamW 1e-3, wd 0.01, batch 64), exactly 14,000
   steps, `checkpoint_last.pt` evaluated.
3. **Cost, reported not hidden.** Reflow needs teacher sampling that iMF does not: the wall-clock of step 1 is
   recorded and reported beside the training time.

## Evaluation
V1 test (1,156 patients, 19,543 windows), NFE 1, 2, 4 with noise seeds 0–3, HR consensus K = 16 at NFE 1 — the SR1
protocol, so arm R pairs window-for-window with the existing arm C, I and S arrays (seed 42).

## Decisions (NFE 1; the BB1 margins: HR 1.0 bpm, R-peak F1 0.02)
- **RF vs iMF (arm I), paired:** IMPROVES if HR diff ≤ −1.0 with CI upper < 0 or F1 diff ≥ +0.02 with CI lower > 0;
  WORSE for the mirror; both → MIXED; neither → NO MEANINGFUL CHANGE.
- **RF vs PENGUIN-50, paired:** non-inferior if upper CI of the HR difference < +1.0 **and** lower CI of the F1
  difference > −0.02.
- Reported, not gated: NFE 2 and 4, consensus HR, FD, ms per window, teacher-sampling cost, and the per-patient win
  rate against PENGUIN-50 (the PZ3 statistic).
