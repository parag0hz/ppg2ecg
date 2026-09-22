# DW2 — Depth-vs-width: three-seed replication of the DW1 endpoints, and the diversity mechanism (preregistration)

Frozen and pushed before any DW2 weight update. No new model, no new loss.

## Part A — three-seed replication at three fixed allocations (B = 32)
- **Hypothesis.** For each model, the width extreme and the DW1 best allocation beat the depth extreme on HR error in
  3/3 seeds; the ordering "mostly width + small depth" is seed-independent.
- **Models / checkpoints.** iMF (`sr1_I_seed{1,2}`, `v1_vitaldb_armI_seed42`), PENGUIN (`sr1_C_seed{1,2}`,
  `v1_vitaldb_armC_seed42`), consistency distillation (seed 42 existing; **seeds 1 and 2 distilled now from that seed's
  PENGUIN with the unchanged CD1 recipe**). Data: V1 VitalDB test, unchanged.
- **Conditions per model (fixed from the DW1 seed-42 grid; not re-chosen per seed):**
  width (K 32, S 1) · DW1 best — iMF (16, 2), CD (32, 1) [= width], PENGUIN (8, 4) · depth (K 1, S 32).
  For CD the best coincides with width, so its "balanced" point is (16, 2) for the ordering question.
- **Primary metric.** HR error, patient-macro; **paired subject-level difference against (1, 32)** with 2,000-replicate
  patient-clustered CI; seed values, 3-seed mean ± SD; per-patient win rate against (1, 32).
- **Success.** `width − depth` CI upper < 0 in 3/3 seeds for every model. PENGUIN's (32, 1) is reported but **not read as
  a consensus gain** (its S = 1 samples have ≈ no diversity).
- **Unchanged.** Pooling rule (median), noise seeds 0 … K−1, evaluation code (`dw1_depth_width.py` samplers).

## Part B — mechanism: diversity by depth (seed 42, no training)
- For S ∈ {1, 2, 4, 8, 16, 32} and each model: **HR diversity** = across-sample SD of per-sample HR (from the DW1 matrices,
  K = 32/S samples); **waveform diversity** = mean pairwise RMS distance between samples of the same window, and
  **feature diversity** = the same in the KANFlow FD feature space, both on a fixed `linspace` subset of 2,000 test windows
  regenerated with K = min(8, 32/S) samples; **consensus gain(S)** = single-sample HR error − K-sample median error at
  the largest K the budget allows (B = 32).
- **Hypothesis.** Diversity rises with S for PENGUIN from ≈ 0 at S = 1, and consensus gain appears where diversity does;
  for iMF and CD diversity is already present at S = 1. Reported as two figures: S → diversity, diversity → gain.
- **Reading rule stated now.** If PENGUIN's HR diversity at S = 1 is below one third of its S = 4 value, "PENGUIN at S = 1
  has no useful diversity" is confirmed; otherwise the DW1 caveat is withdrawn.
