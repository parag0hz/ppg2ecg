# BB1 — Is the PENGUIN backbone the bottleneck? (VitalDB, single seed; preregistration)

Frozen and pushed before any BB1 weight update. Single seed (42) by user decision.

**Question.** On VitalDB, does changing only the backbone of the iMF model move task metrics beyond the V1 iMF result?

| arm | backbone | effective params |
|---|---|---|
| I (reference, existing) | PENGUIN S5, h 128, 4 blocks — `outputs/v1_vitaldb_armI_seed42` | 4.30 M |
| **A** (larger) | PENGUIN S5, **h 256**, 4 blocks | 17.10 M (4.0×) |
| **B** (other family) | PENGUIN skeleton, every S5 mixer → 4-head bidirectional **self-attention** + sinusoidal positions (`src/ppg2ecg/models/attn_backbone.py`) | 3.78 M |

Size note: the chat plan said "h 256, 8 blocks, about 4×"; measured, that is 29.7 M (6.9×). Arm A uses 4 blocks to match
the stated 4× size. Chosen from parameter count and speed only, before any BB1 training.

**Everything else = V1 arm I.** Data `split_v1_vitaldb_seed42.json`, `scripts/v1_run.sh` COMMON argv, 14,000 steps,
micro-batch 32, seed 42, `checkpoint_last.pt`.

**Evaluation.** V1 test (1,156 patients, 19,543 windows). Arms I, A, B at NFE 1, 2, 4 with noise seeds 0–3 (shared across
arms, per-window metric averaged over seeds); HR consensus K = 16 at NFE 1 (seeds 0–15). Paired patient-clustered
bootstrap (2,000, seed 20260911) of (X − I) per metric. PENGUIN-50 is quoted from V1 (unpaired, descriptive).

**Decision, per candidate X ∈ {A, B}, at NFE 1:**
- IMPROVES: HR diff ≤ −1.0 bpm with CI upper < 0, **or** R-peak F1 diff ≥ +0.02 with CI lower > 0.
- WORSE: HR diff ≥ +1.0 bpm with CI lower > 0, **or** F1 diff ≤ −0.02 with CI upper < 0.
- Both → MIXED. Neither → NO MEANINGFUL CHANGE.
- Stage: **BACKBONE IS A BOTTLENECK** if any candidate IMPROVES (and is not MIXED); otherwise **NOT A BOTTLENECK AT THIS SCALE**.

Reported, not gated: NFE 2 and 4, consensus HR, RR-MAE, FD, training time, ms per window. Nothing is tuned after results.
