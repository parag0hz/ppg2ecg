# V1 — iMF vs PENGUIN (OT-CFM) on VitalDB: **H1 NON-INFERIOR at k=1 · H2 SUPERIOR** (single seed)

Prereg `f075065`; code `a1aba7f` (before any weight update). Test = 1,156 patients / 1,224 cases / 19,543 windows,
patient-level split, no train–test patient overlap. Both arms at exactly 14,000 steps (`checkpoint_last.pt`), seed 42.
Values are patient-macro with patient-clustered 95 % bootstrap CI (2,000); F1 micro and FD are corpus-pooled.
Raw: `artifacts/v1_vitaldb/metrics.csv`, `verdict.json`. **Single seed — a fast check, not a final result.**

| arm | NFE / window | ms / window | HR err (bpm) | R-peak F1 | RR-MAE (ms) | MAE | FD (KANFlow) |
|---|---|---|---|---|---|---|---|
| PENGUIN | 50 | 28.7 | 9.69 [9.36, 10.0] | 0.645 | 23.8 | 0.396 | 4.30 |
| PENGUIN | 1 | 0.56 | 8.47 [8.06, 8.9] | **0.756** | **8.8** | **0.301** | 33.50 |
| PENGUIN | 2 | 1.12 | 30.04 | 0.061 | 31.2 | 0.531 | 106.94 |
| PENGUIN | 4 | 2.27 | 12.34 | 0.624 | 23.3 | 0.398 | 20.91 |
| iMF | 1 | 0.57 | 10.14 [9.80, 10.5] | 0.674 | 20.8 | 0.403 | 4.13 |
| iMF | 2 | 1.14 | 9.41 | 0.672 | 20.0 | 0.415 | 3.80 |
| iMF | 4 | 2.27 | 8.97 [8.62, 9.33] | 0.676 | 19.1 | 0.406 | **3.37** |
| PENGUIN 50, HR consensus K=4 | 200 | — | 7.55 [7.20, 7.90] | | | | |
| iMF 1, HR consensus K=16 | 16 | — | 6.67 [6.31, 7.05] | | | | |
| iMF 2, HR consensus K=16 | 32 | — | **6.18** [5.81, 6.56] | | | | |
| PPG peaks only (no model) | 0 | — | 9.06 [8.62, 9.50] | | | | |

## Verdicts (frozen rules)
- **H1 — NON-INFERIOR at k=1.** iMF − PENGUIN-50: HR +0.45 [0.35, 0.56] < +1.0; F1 +0.030 [0.028, 0.031] > −0.02.
  k=2 and k=4 also pass (HR −0.28 and −0.71). Gate: PENGUIN span |C₁ − C₅₀| = 1.21 bpm (HR), 0.111 (F1), both ≥ |δ|, no PASS withheld.
- **H2 — SUPERIOR.** iMF NFE 1 K=16 consensus − PENGUIN-50 single sample: **−3.01 bpm [−3.13, −2.90]**, at 16 vs 50 NFE.

## What must be reported with it
1. **PENGUIN at NFE 1 is the best single-sample arm on HR, F1, RR-MAE and MAE here.** Under anaesthesia the rhythm is
   regular, so the collapsed conditional-mean waveform (visible in `artifacts/qual_model_dataset/qual_v1_vitaldb.png`)
   still places peaks at the right rate. Its FD is the worst non-degenerate value (33.5 vs iMF 3.4–4.1). On VitalDB,
   "iMF beats few-step PENGUIN" holds on waveform distribution (FD), **not** on HR / F1.
2. **Single-sample HR of every model is at the level of counting PPG peaks (9.06).** Only consensus arms clearly beat it.
3. **PENGUIN NFE 2 (one Heun step) diverges** (HR 30.0, F1 0.06), as it did on U2 WildPPG.
4. H2 was formed on P1 validation data (two subjects per corpus) and confirmed here on unseen patients; the report must say so.
5. One seed. The H1/H2 margins are wide relative to the CIs, but seed variance is not measured.
