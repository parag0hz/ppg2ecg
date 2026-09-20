# AB1 — Consensus ablation and the cost-matched curve (preregistration; no training)

Frozen before any AB1 number. The headline claim stays as preregistered in SR1/WP1 (K = 16, median, NFE 1); this
stage describes how the gain depends on the sample budget and the pooling rule, and draws the compute-matched curve.

- **Data.** V1 VitalDB test (1,156 patients, 19,543 windows), the SR1 seed-42 checkpoints.
- **Arms and budgets.** iMF NFE 1 (K ≤ 32), iMF NFE 2 (K ≤ 16), consistency distillation NFE 1 (K ≤ 32),
  PENGUIN NFE 1 (K ≤ 32), PENGUIN NFE 50 (K ≤ 4 — its cost ceiling for this probe). Noise seeds 0…K−1.
- **Pooling rules**, applied to the per-sample HR estimates of the same window: **median** (the preregistered rule),
  mean, and 20 % trimmed mean. Rules are compared, not selected: the claim rules stay on median.
- **Reported.** HR error (patient-macro, subject-clustered CI) for every (arm, NFE, K, rule); the **cost-matched
  curve** with total NFE per window (= K × NFE) on the x-axis; and the saturation point, defined now as the smallest
  K whose doubling changes HR error by less than 0.1 bpm.
- **No verdict is issued here**; the SR1/WP1 verdicts are unaffected by anything measured in this stage.
