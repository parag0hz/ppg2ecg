"""DP0 — DualReadout-ECG (docs/DP0_DUALREADOUT_PREREGISTRATION.md): one shared PPG / event conditioning encoder with two
task-specific readouts, a deterministic point decoder mu = D_point(H) and a multiresolution flow-matching decoder
v = D_flow(x_t, t, H). The two waveform outputs are never added."""
