# DB1 — The direct discriminative baseline: PPG → heart rate, no ECG generated (preregistration)

Frozen and pushed before any DB1 weight update. The baseline every HR claim in this programme has to face.

- **Model.** 1-D convolutional regressor on the 4 s PPG window: 6 blocks (channels 32-64-128-128-256-256, kernel 7,
  stride 2, BatchNorm, GELU), global average pooling, two-layer head → one scalar. No ECG is generated.
- **Target.** The reference HR of the window — the same quantity every other arm is scored against (neurokit R peaks
  on the target ECG → `hr_bpm`). Loss: L1 on HR / 100.
- **Training.** V1 VitalDB train split, AdamW 1e-3, weight decay 0.01, batch 64, **14,000 steps**, seed 42 — the
  budget of every other arm. `checkpoint` at step 14,000; no selection.
- **Evaluation.** V1 test (1,156 patients, 19,543 windows): |HR_pred − HR_ref|, patient-macro, patient-clustered
  bootstrap; paired against (a) iMF K = 16 HR consensus, (b) CD K = 16 consensus, (c) one PENGUIN-50 sample,
  (d) PPG peak counting.
- **Stated now.** A direct regressor is expected to be a strong HR estimator and **may beat the consensus arms**. If it
  does, the report says so in its first line, and the paper's HR claim is restricted to *"among methods that output an
  ECG-form signal"*; the depth-versus-width result (DW1) does not depend on this outcome.
