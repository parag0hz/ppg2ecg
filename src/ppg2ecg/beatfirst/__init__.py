"""Beat-First generator components for BF0 (docs/BF0_BEAT_FIRST_FEASIBILITY_PREREGISTRATION.md).

Timing comes from a discriminative event stage (the RD1 detector plus an N5-style timing head); morphology comes from a
small beat-level flow-matching generator; beats are placed at the predicted times by weighted overlap-add.
Inference input is PPG only.
"""
