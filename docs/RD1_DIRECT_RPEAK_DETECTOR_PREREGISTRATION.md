# RD1 — Direct PPG → R-peak detector on VitalDB (preregistration)

Frozen and pushed before any RD1 weight update. Answers: must an ECG be generated to obtain the R-peak sequence?

- **Model.** The repository's existing R1 Global-TCN (`ppg2ecg.probes` / R1 stage; 328,897 params), unchanged: PPG window
  → per-sample R-peak probability; training target = Gaussian bumps (σ = 20 ms) at the target's R peaks; BCE loss;
  peaks extracted with R1's frozen rule (threshold 0.35, refractory 32 samples). No architecture search.
- **Training.** V1 VitalDB train split, AdamW 1e-3, wd 0.01, batch 64, 14,000 steps, seed 42.
- **Evaluation.** V1 test; the standard pipeline (`task_metrics`) applied to the detector's peak sequence: R-peak
  precision / recall / F1 @ 50 ms, RR-MAE, HR error; inference latency; parameter count.
- **Compared** (paired): iMF single; iMF consensus-decoded; CD consensus-decoded; PENGUIN 50 NFE.
- **Stated now.** If the detector matches or beats the generative arms on F1 / RR-MAE — N1 already showed this on
  WildPPG — the claim is scoped to "when a structured conditional generative output is required, how to allocate the
  inference budget and extract the functional", not "generation is necessary for event estimation".
