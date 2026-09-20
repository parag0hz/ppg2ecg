# MC1 — Three more ECG corpora, every metric used so far (preregistration)

Frozen and pushed before any MC1 weight update. Purpose: the consensus claims rest on two corpora (VitalDB, WildPPG);
this stage adds three, with the **unchanged recipe and the ED1 decoding parameters transferred without re-tuning**.

## Corpora (PPG → ECG, 4 s @ 128 Hz, the project's frozen preprocessing)
| corpus | source | subjects |
|---|---|---|
| BIDMC | `scripts/build_processed_bidmc.py --segment-len 4` | 51–53 |
| CapnoBase | `scripts/build_processed_capnobase.py --segment-len 4` | 42 |
| PPG-DaLiA | `data/processed/u2_dalia` (U2 corpus, upstream preprocessing) | 15 |

**Split (subject level, one holdout per corpus):** subjects sorted by name →
`numpy.random.default_rng(20260920).permutation`; test = round(0.30 n), val = max(2, round(0.10 n)), train = rest.
Evaluation: all test windows, capped at 3,000 per subject by exact `linspace`.

## Arms (seed 42, exactly 14,000 steps, the V1 / WP1 argv)
C = PENGUIN (NFE 50 and 1); I = iMF (NFE 1); D = consistency distillation from that corpus's C (NFE 1).
For I and D: K = 16 samples → HR-median consensus, and ED1 consensus decoding with the frozen parameters `bc88553`.

## Metrics — everything used in the programme
HR error, R-peak F1 @ 50 ms, RR-MAE, MAE, RMSE, KANFlow FD, Micro-F1, Macro-F1, `morph_corr@GT`; for the consensus
arms: consensus HR error, and the full set again on the decoded waveform.

## Claims, per corpus (cluster = subject; 2,000-replicate bootstrap)
- **H2** iMF K = 16 HR consensus beats one PENGUIN-50 sample (upper CI < 0).
- **H-ED** decoded R-peak F1 exceeds the generator's own single sample by ≥ +0.05 with the CI excluding 0 (iMF and CD).
- **Stated now:** these corpora are small (BIDMC 8 min and CapnoBase 8 min per subject; DaLiA 15 subjects). With 14,000
  steps the models will see each training window hundreds of times; the recipe is nevertheless left unchanged so the
  arms stay comparable. Wide intervals or a failure here are reported as such, and do not alter the VitalDB / WildPPG
  verdicts.
