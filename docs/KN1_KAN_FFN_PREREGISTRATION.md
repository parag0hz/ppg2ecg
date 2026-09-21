# KN1 — Does a KAN feed-forward help? (VitalDB, single seed; preregistration)

Frozen and pushed before any KN1 weight update. Seed 42, V1 data / split / 14,000-step recipe, unchanged.

**Background.** KANFlow (IEEE IoT-J, DOI 10.1109/JIOT.2026.3717960) replaces the FFN MLPs of its **shallow** stages by
RBF-KAN layers (grid 5, bounded input) and reports MAE 0.64 → 0.57 and RMSE 0.94 → 0.84 against an MLP rectified-flow
baseline; it also reports that inserting KAN in deeper stages degrades performance. Its evidence is on MAE / RMSE.
This stage asks the same question under this project's protocol: patient-level split, event metrics, equal budget.

## Arms (only `mlp_ppg` and `mlp_target` of the selected Flow-SSM blocks change; `src/ppg2ecg/models/kan_backbone.py`)
| arm | objective | KAN placement | params |
|---|---|---|---|
| **I** (reference, existing) | iMF | none (MLP) | 4.30 M |
| **I-Ko** | iMF | first and last block ("outer", KANFlow's shallow placement) | 4.43 M |
| **I-Ka** | iMF | all four blocks | 4.56 M |
| **C** (reference, existing) | OT-CFM (PENGUIN) | none | 4.30 M |
| **C-Ko** | OT-CFM | outer | 4.43 M |
KANLinear: `y = W_b·SiLU(x) + W_s·RBF_G(tanh(x / 2))`, G = 5 — the design KANFlow describes. One KANLinear (h → h)
replaces the two-layer MLP (1.5× that layer's parameters, +3 % / +6 % overall).

## Evaluation
V1 test (1,156 patients, 19,543 windows), the SR1 protocol: iMF arms at NFE 1, 2, 4 plus K = 16 HR consensus; the
OT-CFM arm at NFE 50 and 1. Paired patient-clustered bootstrap against the MLP arm of the same objective.

## Decisions (the BB1 rule; iMF arms at NFE 1, the OT-CFM arm at NFE 50)
- **IMPROVES** iff HR diff ≤ −1.0 bpm with CI upper < 0, or R-peak F1 diff ≥ +0.02 with CI lower > 0; **WORSE** for the
  mirror; both → MIXED; neither → **NO MEANINGFUL CHANGE**.
- **Reported beside it, because it is where KANFlow's evidence lies:** the MAE and RMSE differences with CIs, and FD.
  If MAE / RMSE improve while HR / F1 do not, the report says that KAN's gain is on the mean-seeking metrics only.
