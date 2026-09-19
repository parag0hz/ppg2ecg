# CD1 — Consistency distillation as the second competing few-step method (preregistration)

Frozen and pushed before any CD1 weight update. Seed 42; VitalDB V1 data and split; 14,000 steps; same backbone.

## Method (Song et al. 2023 consistency distillation, written for this project's flow convention)
Convention as upstream PENGUIN: t = 0 noise, t = 1 data. Teacher = arm C (`outputs/v1_vitaldb_armC_seed42`, frozen).
- **Consistency function** `f_θ(x, t) = x + (1 − t) · v_θ(x, t)` on the PENGUIN backbone — the one-Euler-jump-to-data
  parameterisation, so the boundary condition `f(x, 1) = x` holds by construction. Student initialised from arm C
  (standard CD practice; the same initialisation as reflow in RF1).
- **Discretisation** N = 25 uniform steps on [0, 1] (the teacher's own grid).
- **Per step:** real (PPG, ECG) batch; `z0 ~ N(0, I)`; `n ~ U{0 … N−2}`; `x_n = (1 − t_n) z0 + t_n x1`;
  one **teacher Heun step** `x̂_{n+1}` from `x_n` to `t_{n+1}`; loss = pseudo-Huber
  `sqrt(‖f_θ(x_n, t_n) − f_θ⁻(x̂_{n+1}, t_{n+1})‖² + c²) − c`, c = 0.00054·√d (d = 512; iCT's choice);
  target `θ⁻` = EMA of θ with decay 0.999, updated every step, no gradient.
- **Optimiser** as arm C and RF1 (AdamW 1e-3, wd 0.01, batch 64). **Declared fallback:** if the loss becomes non-finite,
  the run is restarted once at lr 1e-4 and that is reported.
- The **EMA target network is only a training device**; the evaluated model is the student θ at step 14,000.

## Sampling
- NFE 1: `x = f(z0, 0)`.
- NFE 2 / 4 (multistep consistency sampling): after each evaluation, re-noise to the next time on the fixed schedule
  {0.5} or {0.25, 0.5, 0.75} with fresh noise, `x_t = (1 − t) z + t x`, and evaluate `f(x_t, t)` again.

## Evaluation and decisions
SR1 protocol on the V1 test set (NFE 1, 2, 4; noise seeds 0–3; HR consensus K = 16 at NFE 1), paired with the seed-42
arrays of C, I, S and R. At NFE 1, with HR 1.0 bpm and R-peak F1 0.02 margins:
- **CD vs iMF (arm I):** IMPROVES / WORSE / MIXED / NO MEANINGFUL CHANGE (the BB1 rule).
- **CD vs PENGUIN-50:** non-inferior iff HR upper CI < +1.0 and F1 lower CI > −0.02.
- Reported, not gated: NFE 2 and 4, consensus HR, FD, per-patient HR win rate vs PENGUIN-50, training time.
